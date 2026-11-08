"""3D-ICE through threedice's 3D-ICE-Batch: factorise a stack once, then re-solve it for each new power map.

ICESimulator runs 3D-ICE-Emulator once per scenario, and every run parses the stack, builds the conductance matrix and
factorises it with SuperLU before a single solve. For a steady-state stack the matrix depends on the stack, layouts and
materials but not on the floorplans (they only place the power), so scenarios that differ only in their power maps can
share one factorisation. NativeICESimulator keeps one 3D-ICE-Batch process alive per distinct "everything except the
floorplans" and re-solves it, writing the same output_inst_N.txt Tmaps (3D-ICE's own "%7.3f  " format) so
parse_results and everything downstream are unchanged.

The generate -> run -> parse sequence of ICESimulator is kept exactly: generate_config_files rewrites config_dir
(including the floorplan files, at the same paths), run_simulation solves, parse_results reads output_dir. Because
3D-ICE-Batch reloads every die's floorplan file before each solve, the rewritten power maps are picked up without
re-sending them.

Falls back to the emulator (ICESimulator.run_simulation) for stacks 3D-ICE-Batch does not handle: a microchannel /
`channel` element or a pluggable heat sink. The binary comes from `batch_executable` or $THERMO_ICE_BATCH_EXECUTABLE
(e.g. "wsl /home/<you>/threedice/build/native/3d-ice/bin/3D-ICE-Batch"); it must be built from the same pinned 3D-ICE
as the emulator.
"""
from __future__ import annotations

import hashlib
import os
import queue
import re
import shlex
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .ice_simulator import _LAST_RUN, ICESimulator, verify_ice_executable

_TMAP = re.compile(r'Tmap\s*\(\s*(\w+)\s*,\s*"([^"]+)"\s*,\s*final\s*\)\s*;')
_UNSUPPORTED = re.compile(r"^\s*(channel\s+\w+\s*;|microchannel\b|.*\bpluggable\b)", re.M)


class _Session:
    """One 3D-ICE-Batch process (protocol: threedice native/ice_batch.cpp)."""

    def __init__(self, cmd: List[str], stk_arg: str, cwd: Optional[str], timeout: float):
        self.timeout = timeout
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env["WSLENV"] = ":".join(x for x in (env.get("WSLENV", ""), "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS") if x)
        self.proc = subprocess.Popen(cmd + [stk_arg], cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lines: "queue.Queue[Optional[str]]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._reply()
        if not ready.startswith("READY "):
            self.close()
            raise RuntimeError(f"3D-ICE-Batch did not start: {ready}")
        _, secs, rows, cols = ready.split()
        self.factor_seconds, self.shape = float(secs), (int(rows), int(cols))

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def _reply(self) -> str:
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"3D-ICE-Batch gave no reply within {self.timeout:.0f} s")
        if line is None:
            raise RuntimeError(f"3D-ICE-Batch exited (code {self.proc.poll()})")
        return line

    def command(self, line: str) -> str:
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"3D-ICE-Batch is not running: {e}")
        r = self._reply()
        if r.startswith("ERR"):
            raise RuntimeError(r[4:])
        return r

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        if self.proc.poll() is None:
            self.proc.kill()


def write_tmap(path: Path, field: np.ndarray) -> None:
    """A field in 3D-ICE's Tmap text format (one row per line, '%7.3f  ' per value), as 3D-ICE-Emulator prints it."""
    with open(path, "w") as f:
        f.write("% written by 3D-ICE-Batch (NativeICESimulator)\n")
        for row in field:
            f.write("".join("%7.3f  " % v for v in row))
            f.write("\n")


class NativeICESimulator(ICESimulator):
    """ICESimulator whose solves reuse one 3D-ICE-Batch factorisation per distinct stack (see the module docstring)."""

    def __init__(self, config_dir: Path, output_dir: Path, executable: str = None, batch_executable: str = None):
        kw = {} if executable is None else {"executable": executable}
        super().__init__(config_dir, output_dir, **kw)
        self.batch_executable = batch_executable or os.environ.get("THERMO_ICE_BATCH_EXECUTABLE", "")
        if not self.batch_executable:
            raise ValueError("NativeICESimulator needs batch_executable= or $THERMO_ICE_BATCH_EXECUTABLE")
        self._session: Optional[_Session] = None
        self._key: Optional[str] = None
        self.stats = {"native_solves": 0, "factorisations": 0, "emulator_fallbacks": 0}

    # -- the stack minus its floorplans: the identity of a factorisation
    def _matrix_key(self, stk_text: str) -> str:
        h = hashlib.sha256(stk_text.encode("utf-8"))
        for p in sorted(self.config_dir.glob("*.lyt")):
            h.update(p.name.encode() + b"\0" + p.read_bytes())
        return h.hexdigest()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session, self._key = None, None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def run_simulation(self, scenario_name: str) -> Path:
        stk_file = self.config_dir / "stack.stk"
        text = stk_file.read_text()
        if _UNSUPPORTED.search(text):
            self.stats["emulator_fallbacks"] += 1
            self.close()
            return super().run_simulation(scenario_name)
        for stale in self.output_dir.glob("output_inst_*.txt"):
            try:
                stale.unlink()
            except OSError:
                pass
        tmaps: List[Tuple[str, str]] = _TMAP.findall(text)
        if not tmaps:
            raise RuntimeError(f"{stk_file} requests no Tmap output")
        cmd = shlex.split(self.batch_executable)
        use_wsl = os.path.basename(cmd[0]).lower() in ("wsl", "wsl.exe")
        n_nodes = self._nx * self._ny * max(len(getattr(self, "_sublayers", [])), 1)
        timeout_s = max(300.0, 300.0 + n_nodes / 400.0)      # the emulator's per-solve allowance (ICESimulator)
        key = self._matrix_key(text)
        if self._session is None or key != self._key or not self._session.alive():
            self.close()
            # the same pinned-build check (and ice_commit provenance) as the emulator path; 3D-ICE-Batch is built by
            # threedice from that same pinned tree, and its own command is recorded next to it
            _LAST_RUN.update(verify_ice_executable(self.executable))
            stk_arg = self._to_wsl_path(stk_file.resolve()) if use_wsl else str(stk_file.resolve())
            self._session = _Session(cmd, stk_arg, str(self.config_dir), timeout_s)
            self._key = key
            self.stats["factorisations"] += 1
            if self._session.shape != (self._ny, self._nx):
                self.close()
                raise RuntimeError(f"3D-ICE-Batch grid {self._session.shape} != scenario grid {(self._ny, self._nx)}")
        out_prefix = (self.output_dir / "native_").resolve()
        prefix = self._to_wsl_path(out_prefix) if use_wsl else str(out_prefix)
        try:
            reply = self._session.command("SOLVE " + prefix + " " + " ".join(inst for inst, _ in tmaps))
        except RuntimeError as e:
            self.close()
            raise RuntimeError(f"3D-ICE-Batch failed for {scenario_name}: {e}")
        for inst, _ in tmaps:
            raw = self.output_dir / f"native_{inst}.f64"
            field = np.fromfile(raw, dtype="<f8").reshape(self._session.shape)
            raw.unlink()
            if not np.all(np.isfinite(field)):
                raise RuntimeError(f"3D-ICE-Batch: non-finite temperatures in {inst} for {scenario_name}")
            write_tmap(self.output_dir / f"output_{inst}.txt", field)
        self.stats["native_solves"] += 1
        _LAST_RUN.update({"backend": "native", "batch_executable": self.batch_executable,
                          "solve_seconds": reply.split()[1], "factor_seconds": f"{self._session.factor_seconds:.3f}"})
        return self.output_dir
