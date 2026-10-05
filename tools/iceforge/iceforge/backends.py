"""Locating and invoking a 3D-ICE emulator: native, WSL, or docker."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

EMU = "3D-ICE-Emulator"
WSL_CANDIDATES = ("3d-ice-4.0/bin/" + EMU, "3d-ice/bin/" + EMU)   # relative to $HOME
DOCKER_IMAGE = "iceforge:latest"
DOCKER_EXE = "/opt/3d-ice/bin/" + EMU
IS_WINDOWS = os.name == "nt"


@dataclass
class Backend:
    kind: str            # native | wsl | docker
    exe: str             # path (inside WSL / container for those kinds)
    detail: str = ""

    def __str__(self):
        return f"{self.kind}:{self.exe}" + (f" ({self.detail})" if self.detail else "")


class BackendError(RuntimeError):
    pass


def _run(cmd: List[str], timeout=30, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace", **kw)


def to_wsl_path(p: str) -> str:
    """Translate a Windows path to its WSL form (/mnt/<drive>/...). Paths with spaces are fine
    because we never go through a shell."""
    if not (p.startswith("/") or p.startswith("\\\\")):
        p = os.path.abspath(p)
    m = re.match(r"^\\\\wsl(?:\$|\.localhost)\\[^\\]+(\\.*)$", p, re.I)
    if m:
        return m.group(1).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        return "/mnt/" + m.group(1).lower() + "/" + m.group(2).replace("\\", "/")
    return p.replace("\\", "/")


def _wsl_available() -> bool:
    return shutil.which("wsl") is not None


def _wsl_home() -> Optional[str]:
    try:
        r = _run(["wsl", "-e", "printenv", "HOME"], timeout=30)
        h = r.stdout.strip()
        return h if r.returncode == 0 and h.startswith("/") else None
    except Exception:
        return None


def _wsl_is_exe(path: str) -> bool:
    try:
        return _run(["wsl", "-e", "test", "-x", path], timeout=30).returncode == 0
    except Exception:
        return False


def probe_wsl(extra: Optional[str] = None) -> Optional[Backend]:
    if not _wsl_available():
        return None
    cands: List[str] = [extra] if extra else []
    home = _wsl_home()
    if home:
        cands += [f"{home}/{c}" for c in WSL_CANDIDATES]
    for c in cands:
        if _wsl_is_exe(c):
            return Backend("wsl", c)
    return None


def probe_native() -> Optional[Backend]:
    if IS_WINDOWS:
        return None
    p = shutil.which(EMU)
    return Backend("native", p) if p else None


def probe_docker() -> Optional[Backend]:
    if shutil.which("docker") is None:
        return None
    try:
        r = _run(["docker", "image", "inspect", DOCKER_IMAGE], timeout=30)
    except Exception:
        return None
    return Backend("docker", DOCKER_EXE, DOCKER_IMAGE) if r.returncode == 0 else None


def from_exe(exe: str) -> Backend:
    """Interpret an explicit executable string (--exe or ICE_EXECUTABLE)."""
    exe = exe.strip()
    if exe.lower().startswith("wsl "):               # repo convention: "wsl /home/u/3d-ice/bin/3D-ICE-Emulator"
        return Backend("wsl", exe[4:].strip())
    if exe.startswith("/") and IS_WINDOWS:
        return Backend("wsl", exe)
    return Backend("native", exe)


def detect(kind: str = "auto", exe: Optional[str] = None) -> Backend:
    if exe:
        b = from_exe(exe)
        if kind not in ("auto", b.kind):
            b = Backend(kind, exe)
        return b
    if kind == "auto":
        env = os.environ.get("ICE_EXECUTABLE")
        if env:
            return from_exe(env)
        for probe in (probe_native, probe_wsl, probe_docker):
            b = probe()
            if b:
                return b
        raise BackendError("no 3D-ICE backend found (tried ICE_EXECUTABLE, native, WSL, docker); "
                           "see `iceforge doctor`")
    probe = {"native": probe_native, "wsl": probe_wsl, "docker": probe_docker}[kind]
    b = probe()
    if not b:
        raise BackendError(f"backend '{kind}' is not available; see `iceforge doctor`")
    return b


def command_for(b: Backend, stk_path: str) -> (List[str], str):
    """Return (argv, cwd) that runs the emulator on stk_path with cwd = the stk's directory."""
    d, name = os.path.split(os.path.abspath(stk_path))
    if b.kind == "native":
        return [b.exe, name], d
    if b.kind == "wsl":
        return ["wsl", "--cd", to_wsl_path(d), "-e", b.exe, name], d if os.path.isdir(d) else None
    if b.kind == "docker":
        return ["docker", "run", "--rm", "-v", f"{d}:/work", "-w", "/work",
                "--entrypoint", b.exe, b.detail or DOCKER_IMAGE, name], d
    raise BackendError(f"unknown backend {b.kind}")


def run_emulator(b: Backend, stk_path: str, timeout: Optional[float] = None):
    argv, cwd = command_for(b, stk_path)
    return subprocess.run(argv, cwd=cwd if b.kind != "wsl" else None, capture_output=True, text=True,
                          errors="replace", timeout=timeout)


_MINI_STK = """material SI :
   thermal conductivity 1.30e-4 ;
   volumetric heat capacity 1.628e-12 ;
bottom heat sink :
   heat transfer coefficient 2.0e-8 ;
   temperature 300 ;
dimensions :
   chip length 1000, width 1000 ;
   cell length 250, width 250 ;
die D :
   source 50 SI ;
stack :
   die TOP D floorplan "d.flp" ;
solver :
   steady ;
   initial temperature 300 ;
output :
   Tmap ( TOP, "t.txt", final ) ;
"""
_MINI_FLP = "blk :\n position 0, 0 ;\n dimension 1000, 1000 ;\n power values 1.0 ;\n"


def _root_text(b: Backend, rel: str) -> str:
    root = os.path.dirname(os.path.dirname(b.exe))
    try:
        if b.kind == "native":
            with open(os.path.join(root, rel), errors="replace") as f:
                return f.read(20000)
        if b.kind == "wsl":
            return _run(["wsl", "-e", "head", "-c", "20000", f"{root}/{rel}"]).stdout
    except Exception:
        pass
    return ""


def emulator_version(b: Backend) -> str:
    """The emulator prints no version, so report: smoke-test result, the version named in the
    install tree's README, and `git describe` when the tree is a git checkout."""
    parts = []
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.stk"), "w") as f:
                f.write(_MINI_STK)
            with open(os.path.join(td, "d.flp"), "w") as f:
                f.write(_MINI_FLP)
            r = run_emulator(b, os.path.join(td, "m.stk"), timeout=120)
            ran = r.returncode == 0 and os.path.exists(os.path.join(td, "t.txt"))
    except Exception as e:
        return f"unknown ({e})"
    parts.append("smoke test ok" if ran else "smoke test FAILED")
    if b.kind in ("native", "wsl"):
        m = re.search(r"3D-ICE\s+(\d+\.\d+)", _root_text(b, "README.md"))
        if m:
            parts.insert(0, m.group(1))
        root = os.path.dirname(os.path.dirname(b.exe))
        try:
            cmd = ["git", "-C", root, "describe", "--tags", "--always"]
            r = _run((["wsl", "-e"] + cmd) if b.kind == "wsl" else cmd)
            if r.returncode == 0 and r.stdout.strip():
                parts.append("git " + r.stdout.strip())
        except Exception:
            pass
    return "; ".join(parts)
