"""Run a model through 3D-ICE, parse Tmap outputs, and apply post-solve checks.

Tmap orientation (verified against 3D-ICE 4.0 sources and by test_orientation):
3D-ICE's ``stack_element_print_thermal_map`` loops ``for row ... for column ...`` and writes
one text line per ROW.  A floorplan element's ``position x, y`` / ``dimension length, width``
map x -> column (along chip LENGTH) and y -> row (along chip WIDTH).  So a Tmap file has
n_width_cells lines of n_length_cells values (rows run along the chip width).  iceforge
stores ``T`` as (length_index, width_index), i.e. the transpose of the file layout.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import backends
from .check import Finding, check_stack, format_findings, power_of
from .model import Stack, parse_stk


@dataclass
class TmapResult:
    instance: str
    source: str
    npz: Optional[str]
    shape: Optional[tuple]
    t_max: Optional[float]
    t_ref: Optional[float]
    max_rise: Optional[float]
    hottest: Optional[dict]
    findings: List[Finding] = field(default_factory=list)

    def to_dict(self):
        d = dict(self.__dict__)
        d['findings'] = [f.to_dict() for f in self.findings]
        return d


def read_tmap_blocks(path: str) -> List[List[List[float]]]:
    """Return numeric rows from a Tmap text file (comment lines starting with % are skipped)."""
    rows = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            rows.append([float(v) for v in s.split()])
    return rows


def parse_tmap(path: str, n_rows: int, n_cols: int):
    """Return (T as (length_index, width_index), findings). Takes the last map in the file."""
    F: List[Finding] = []
    rows = read_tmap_blocks(path)
    if not rows:
        return None, [Finding('P001', 'error', "Tmap file holds no numeric data", file=path)]
    widths = {len(r) for r in rows}
    if widths != {n_cols} or len(rows) % n_rows != 0:
        F.append(Finding('P001', 'error',
                         f"Tmap shape mismatch: file has {len(rows)} lines of {sorted(widths)} values; "
                         f"grid expects blocks of {n_rows} lines x {n_cols} values "
                         f"(rows run along chip width, columns along length)", file=path))
        return None, F
    block = np.asarray(rows[-n_rows:], dtype=float)         # (width_index, length_index)
    return block.T.copy(), F                                  # (length_index, width_index)


def _powered_rects(st: Stack):
    rects = []
    for se in st.stack:
        fp = st.floorplans.get(se.floorplan_path) if se.floorplan_path else None
        if not fp:
            continue
        for e in fp.elements:
            if any(p > 0 for p in e.power):
                rects += [(e.name, r) for r in e.rects]
    return rects


def post_checks(st: Stack, T: np.ndarray, source: str) -> (List[Finding], dict):
    """P002: hottest cell centre must lie under a powered floorplan element."""
    d = st.dims
    F: List[Finding] = []
    i, j = np.unravel_index(int(np.argmax(T)), T.shape)
    xc, yc = (i + 0.5) * d.cell_l, (j + 0.5) * d.cell_w
    hot = dict(i_length=int(i), j_width=int(j), x_um=float(xc), y_um=float(yc), T=float(T[i, j]))
    if st.solver.mode == 'transient':
        return F, hot
    eps = 1e-6
    under = [n for n, r in _powered_rects(st)
             if r.x - eps <= xc <= r.x1 + eps and r.y - eps <= yc <= r.y1 + eps]
    if not under:
        F.append(Finding(
            'P002', 'warning',
            f"hottest cell (length idx {i}, width idx {j}; centre x={xc:g}, y={yc:g} um; {T[i, j]:.3f} K) is not "
            f"under any powered floorplan element; max principle says the peak belongs under a source. "
            f"Signature of the die-edge heated-insulator artefact (see S001) or of an orientation error",
            file=source, detail=hot))
    else:
        hot['under'] = under[0]
    return F, hot


def write_npz(outdir: str, name: str, st: Stack, T: np.ndarray, src: str, instance: str, t_ref):
    d = st.dims
    x = (np.arange(T.shape[0]) + 0.5) * d.cell_l
    y = (np.arange(T.shape[1]) + 0.5) * d.cell_w
    path = os.path.join(outdir, name + '.npz')
    np.savez(path, T=T, x_um=x, y_um=y, units=np.array('K'),
             axes=np.array('T[length_index, width_index]; x_um along chip length, y_um along chip width'),
             t_ref=np.array(np.nan if t_ref is None else t_ref))
    side = dict(name=name, instance=instance, units='K', source_tmap=src, stk=st.path,
                shape=list(T.shape), axes=['length', 'width'],
                orientation="T[length_index, width_index]; file rows ran along chip width",
                cell_um=[d.cell_l, d.cell_w], chip_um=[d.chip_l, d.chip_w],
                t_max=float(T.max()), t_min=float(T.min()),
                heat_sink_temperature=t_ref,
                max_rise=None if t_ref is None else float(T.max() - t_ref))
    with open(os.path.join(outdir, name + '.json'), 'w') as f:
        json.dump(side, f, indent=2)
    return path


def collect_outputs(st: Stack, outdir: str) -> List[TmapResult]:
    d = st.dims
    results: List[TmapResult] = []
    used = set()
    for o in st.outputs:
        if o.kind != 'Tmap' or not o.path:
            continue
        if d.non_uniform:
            results.append(TmapResult(o.instance or '?', o.path, None, None, None, None, None, None,
                                      [Finding('I001', 'info', "non-uniform grid: Tmap parsing skipped",
                                               file=o.path)]))
            continue
        if not os.path.isfile(o.path):
            results.append(TmapResult(o.instance or '?', o.path, None, None, None, None, None, None,
                                      [Finding('P001', 'error', "Tmap output file was not produced", file=o.path)]))
            continue
        T, F = parse_tmap(o.path, d.n_rows, d.n_cols)
        if T is None:
            results.append(TmapResult(o.instance or '?', o.path, None, None, None, None, None, None, F))
            continue
        pf, hot = post_checks(st, T, o.path)
        F += pf
        stem = os.path.splitext(os.path.basename(o.path))[0]
        name = stem if stem not in used else f"{stem}_{o.instance}"
        used.add(name)
        t_ref = st.sink_temperature
        npz = write_npz(outdir, name, st, T, o.path, o.instance or '', t_ref)
        results.append(TmapResult(o.instance or '?', o.path, npz, tuple(T.shape), float(T.max()), t_ref,
                                  None if t_ref is None else float(T.max() - t_ref), hot, F))
    return results


def run_model(stk_path: str, backend: str = 'auto', exe: Optional[str] = None, outdir: Optional[str] = None,
              do_check: bool = True, timeout: Optional[float] = None, log=print):
    """Returns (exit_code, report dict)."""
    st = parse_stk(stk_path)
    outdir = outdir or os.path.join(st.base_dir, os.path.splitext(os.path.basename(st.path))[0] + '_iceforge')
    os.makedirs(outdir, exist_ok=True)
    report = dict(stk=st.path, outdir=outdir)
    if do_check:
        pre = check_stack(st)
        report['check'] = [f.to_dict() for f in pre]
        log(format_findings(pre))
        if any(f.severity == 'error' for f in pre):
            log("run: aborted by check errors (use --no-check to override)")
            report['status'] = 'aborted'
            return 1, report
    try:
        b = backends.detect(backend, exe)
    except backends.BackendError as e:
        log(f"run: {e}")
        report['status'] = 'no-backend'
        return 2, report
    report['backend'] = str(b)
    log(f"run: backend {b}")
    t0 = time.time()
    try:
        r = backends.run_emulator(b, st.path, timeout=timeout)
    except Exception as e:
        log(f"run: could not start solver: {e}")
        report['status'] = 'launch-failed'
        return 2, report
    report['seconds'] = round(time.time() - t0, 2)
    with open(os.path.join(outdir, '3dice.log'), 'w', errors='replace') as f:
        f.write((r.stdout or '') + '\n--- stderr ---\n' + (r.stderr or ''))
    if r.returncode != 0:
        log(f"run: 3D-ICE exited with {r.returncode}; see {os.path.join(outdir, '3dice.log')}")
        tail = (r.stderr or r.stdout or '').strip().splitlines()[-5:]
        for line in tail:
            log("  " + line)
        report['status'] = 'solver-failed'
        return 2, report
    results = collect_outputs(st, outdir)
    report['tmaps'] = [t.to_dict() for t in results]
    code = 0
    for t in results:
        for f in t.findings:
            log(str(f))
            if f.severity == 'error':
                code = 1
        if t.npz:
            hot = t.hottest or {}
            log(f"run: {t.instance}: Tmax {t.t_max:.3f} K, max rise {t.max_rise:.3f} K above "
                f"{t.t_ref:g} K  -> {t.npz}  (T shape {t.shape} = (length, width))"
                if t.t_ref is not None else f"run: {t.instance}: Tmax {t.t_max:.3f} K -> {t.npz}")
    if not results:
        log("run: no Tmap outputs declared; nothing to parse")
    report['status'] = 'ok' if code == 0 else 'post-check-error'
    return code, report
