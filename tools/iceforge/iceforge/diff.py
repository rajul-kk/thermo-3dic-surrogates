"""Differential test: 3D-ICE against the independently discretised reference solver.

Per Tmap output layer the reference field (block-averaged to the 3D-ICE grid) is compared with the
3D-ICE Tmap. A cell is flagged when |dT| exceeds both ``tol`` x (reference max rise) and three times
the reference's own r -> 2r change in that cell, so that reference discretisation error cannot be
mistaken for a 3D-ICE error. Flagged cells are classified as ``die-edge`` (a floorplan or layout edge
crosses the cell interior) or ``interior``.

A flagged cell is "explained" when it is reproduced (within EXPLAIN_FRACTION x tol x max rise) by the
reference solver run in an emulation of 3D-ICE's own vertical rule (r = 1, one cell per layer, the
end-layer coupling of ``thermal_grid.c``; exact area-fraction materials): the difference to the
independent reference is then 3D-ICE's coarse vertical discretisation, a known approximation, and not a
model error or a partial-cell artefact. Only unexplained cells decide the verdict.

Cell classes: ``die-edge`` (a floorplan/layout edge crosses the cell interior), ``edge-adjacent`` (a
neighbour of such a cell: the heated-insulator artefact conducts into it), ``interior`` (everything else).

Verdicts
  AGREE                         no unexplained flagged cell
  DISAGREE (die-edge)           >= EDGE_SHARE of the unexplained cells are die-edge or edge-adjacent:
                                partial-cell artefact suspected
  DISAGREE (diffuse)            otherwise: model or boundary-condition mismatch
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np

from .model import Rect, Stack, parse_stk
from . import refsolve as R

EDGE_SHARE = 0.7        # share of flagged cells that must be die-edge cells for the localised verdict
DIFFUSE_FRACTION = 0.25  # more flagged cells than this fraction of the grid is always diffuse
EXPLAIN_FRACTION = 0.25  # emulation residual allowed, as a fraction of the tolerance
EPS = 1e-6


class DiffError(RuntimeError):
    pass


def _layer_rects(model: R.RefModel, layer_idx: int) -> List[Rect]:
    L = model.layers[layer_idx]
    rects = [r for _, rs in L.patches for r in rs]
    rects += [r for r, _ in L.power_rects]
    # a die's floorplan is drawn on its source layer; include it for every layer of that die
    for other in model.layers:
        if other.instance == L.instance:
            rects += [r for r, _ in other.power_rects]
    return rects


def edge_crossing_mask(xe: np.ndarray, ye: np.ndarray, rects: List[Rect]) -> np.ndarray:
    """True for every cell (nx, ny) whose interior is crossed by an edge of any rectangle."""
    nx, ny = len(xe) - 1, len(ye) - 1
    mask = np.zeros((nx, ny), dtype=bool)
    xlo, xhi, ylo, yhi = xe[:-1], xe[1:], ye[:-1], ye[1:]
    for r in rects:
        for ex in (r.x, r.x1):                      # vertical edges at x = ex spanning [r.y, r.y1]
            cx = (xlo + EPS < ex) & (ex < xhi - EPS)
            cy = (np.minimum(yhi, r.y1) - np.maximum(ylo, r.y)) > EPS
            mask |= np.outer(cx, cy)
        for ey in (r.y, r.y1):
            cy = (ylo + EPS < ey) & (ey < yhi - EPS)
            cx = (np.minimum(xhi, r.x1) - np.maximum(xlo, r.x)) > EPS
            mask |= np.outer(cx, cy)
    return mask


def load_ice(st: Stack, ice_npz: Optional[str], backend: str = 'auto', exe: Optional[str] = None,
             outdir: Optional[str] = None, log=lambda *a: None) -> Dict[str, dict]:
    """Return {instance: dict(T=(length,width) K, source=str)} for each Tmap output of the model."""
    tm = [o for o in st.outputs if o.kind == 'Tmap' and o.instance]
    if not tm:
        raise DiffError("the model declares no Tmap output to compare")
    out: Dict[str, dict] = {}
    if ice_npz:
        if os.path.isdir(ice_npz):
            for o in tm:
                stem = os.path.splitext(os.path.basename(o.path or ''))[0]
                for cand in (stem, f"{stem}_{o.instance}"):
                    p = os.path.join(ice_npz, cand + '.npz')
                    if os.path.isfile(p) and o.instance not in out:
                        out[o.instance] = dict(T=np.load(p)['T'], source=p)
        elif os.path.isfile(ice_npz):
            out[tm[0].instance] = dict(T=np.load(ice_npz)['T'], source=ice_npz)
        else:
            raise DiffError(f"--ice-npz not found: {ice_npz}")
        if not out:
            raise DiffError(f"no matching .npz for the model's Tmap outputs in {ice_npz}")
        return out
    from .run import run_model
    outdir = outdir or tempfile.mkdtemp(prefix='iceforge_diff_')
    code, rep = run_model(st.path, backend, exe, outdir, do_check=False, log=log)
    if rep.get('status') not in ('ok', 'post-check-error'):
        raise DiffError(f"3D-ICE run failed: {rep.get('status')}")
    for t in rep.get('tmaps', []):
        if t.get('npz') and t['instance'] not in out:
            out[t['instance']] = dict(T=np.load(t['npz'])['T'], source=t['npz'])
    if not out:
        raise DiffError("3D-ICE produced no parsable Tmap")
    rep_out = os.path.abspath(outdir)
    for v in out.values():
        v['outdir'] = rep_out
    return out


def compare_layer(model: R.RefModel, instance: str, T_ice: np.ndarray, sol_r: R.RefSolution,
                  sol_2r: R.RefSolution, tol: float, cell_l: float, cell_w: float,
                  sol_emu: Optional[R.RefSolution] = None) -> dict:
    li = model.tmap_layer(instance)
    t_ref = sol_r.t_ref
    ref_r = R.block_average(sol_r.layer_field(li), sol_r.r)
    ref_2r = R.block_average(sol_2r.layer_field(li), sol_2r.r)
    if T_ice.shape != ref_2r.shape:
        raise DiffError(f"{instance}: 3D-ICE Tmap shape {T_ice.shape} != reference coarse grid {ref_2r.shape}")
    dT = T_ice - ref_2r
    own = np.abs(ref_2r - ref_r)
    max_rise_ref = float(ref_2r.max() - t_ref)
    thr = np.maximum(tol * max_rise_ref, 3.0 * own)
    flag = np.abs(dT) > thr

    # edge classification on the 3D-ICE grid
    xe = np.arange(T_ice.shape[0] + 1) * cell_l
    ye = np.arange(T_ice.shape[1] + 1) * cell_w
    edge = edge_crossing_mask(xe, ye, _layer_rects(model, li))
    near = edge.copy()
    near[1:, :] |= edge[:-1, :]; near[:-1, :] |= edge[1:, :]
    near[:, 1:] |= edge[:, :-1]; near[:, :-1] |= edge[:, 1:]

    emu = sol_emu.layer_field(li) if sol_emu is not None else None
    ii, jj = np.nonzero(flag)
    cells = []
    for i, j in sorted(zip(ii.tolist(), jj.tolist()), key=lambda ij: -abs(dT[ij])):
        cells.append(dict(i_length=i, j_width=j, x_um=(i + 0.5) * cell_l, y_um=(j + 0.5) * cell_w,
                          T_ice=float(T_ice[i, j]), T_ref=float(ref_2r[i, j]), dT=float(dT[i, j]),
                          threshold=float(thr[i, j]),
                          cls='die-edge' if edge[i, j] else 'edge-adjacent' if near[i, j] else 'interior',
                          T_emulated=None if emu is None else float(emu[i, j]),
                          explained=bool(emu is not None and
                                         abs(T_ice[i, j] - emu[i, j]) <= EXPLAIN_FRACTION * tol * max_rise_ref)))
    n_raw = len(cells)
    n_expl = sum(1 for c in cells if c['explained'])
    cells = [c for c in cells if not c['explained']] + [c for c in cells if c['explained']]
    unexpl = [c for c in cells if not c['explained']]
    n_flag = len(unexpl)
    n_edge = sum(1 for c in unexpl if c['cls'] == 'die-edge')
    n_adj = sum(1 for c in unexpl if c['cls'] == 'edge-adjacent')
    if n_flag == 0:
        verdict = 'AGREE'
    elif n_flag > DIFFUSE_FRACTION * T_ice.size or n_edge + n_adj < EDGE_SHARE * n_flag:
        verdict = 'DISAGREE (diffuse)'
    else:
        verdict = 'DISAGREE (die-edge)'

    ia = np.unravel_index(int(np.argmax(T_ice)), T_ice.shape)
    ib = np.unravel_index(int(np.argmax(ref_2r)), ref_2r.shape)
    disp = float(np.hypot((ia[0] - ib[0]) * cell_l, (ia[1] - ib[1]) * cell_w))
    return dict(
        instance=instance, layer=model.layers[li].name, verdict=verdict,
        max_abs_dT=float(np.abs(dT).max()),
        max_abs_dT_cell=[int(v) for v in np.unravel_index(int(np.argmax(np.abs(dT))), dT.shape)],
        max_rise_ice=float(T_ice.max() - t_ref), max_rise_ref=max_rise_ref,
        peak_diff=float(T_ice.max() - ref_2r.max()),
        argmax_ice_um=[(ia[0] + 0.5) * cell_l, (ia[1] + 0.5) * cell_w],
        argmax_ref_um=[(ib[0] + 0.5) * cell_l, (ib[1] + 0.5) * cell_w],
        argmax_displacement_um=disp,
        ref_r_vs_2r_max_change=float(own.max()),
        ref_r_vs_2r_peak_change=float(ref_2r.max() - ref_r.max()),
        tolerance_K=tol * max_rise_ref, n_cells=int(T_ice.size),
        max_abs_dT_vs_emulation=None if emu is None else float(np.abs(T_ice - emu).max()),
        n_flagged_raw=n_raw, n_explained_by_3dice_vertical_rule=n_expl,
        n_flagged=n_flag, n_flagged_die_edge=n_edge, n_flagged_edge_adjacent=n_adj,
        n_flagged_interior=n_flag - n_edge - n_adj,
        flagged=cells,
        _fields=dict(ice=T_ice, ref=ref_2r, dT=dT, flag=flag),
    )


def run_diff(stk_path: str, r: int = 4, nz: Optional[int] = None, tol: float = 0.02,
             ice_npz: Optional[str] = None, backend: str = 'auto', exe: Optional[str] = None,
             outdir: Optional[str] = None, method: str = 'auto', log=lambda *a: None) -> dict:
    st = parse_stk(stk_path)
    model = R.build_model(st)            # raises RefusedModel
    ice = load_ice(st, ice_npz, backend, exe, outdir, log)
    log(f"diff: reference solve at r={r} ...")
    s1 = R.solve(model, r=r, nz=nz, method=method)
    log(f"diff: reference solve at r={2 * r} ...")
    s2 = R.solve(model, r=2 * r, nz=nz, method=method)
    s_emu = R.solve(model, r=1, nz=1, emulate_3dice_ends=True)
    layers = []
    for inst, d in ice.items():
        try:
            model.tmap_layer(inst)
        except KeyError:
            raise DiffError(f"Tmap instance {inst} is not a die or layer of the stack")
        c = compare_layer(model, inst, d['T'], s1, s2, tol, st.dims.cell_l, st.dims.cell_w, s_emu)
        c['ice_source'] = d['source']
        layers.append(c)
    verdicts = [c['verdict'] for c in layers]
    if all(v == 'AGREE' for v in verdicts):
        overall = 'AGREE'
    elif any(v == 'DISAGREE (diffuse)' for v in verdicts):
        overall = 'DISAGREE (diffuse)'
    else:
        overall = 'DISAGREE (die-edge)'
    return dict(stk=st.path, verdict=overall, r=r, r_fine=2 * r, nz=nz, tol=tol,
                reference=dict(r=s1.info, r2=s2.info), layers=layers, _model=model)


def strip_private(rep: dict, max_cells: int = 200) -> dict:
    out = {k: v for k, v in rep.items() if not k.startswith('_')}
    out['layers'] = []
    for c in rep['layers']:
        c2 = {k: v for k, v in c.items() if not k.startswith('_')}
        c2['flagged_truncated'] = len(c['flagged']) > max_cells
        c2['flagged'] = c['flagged'][:max_cells]
        out['layers'].append(c2)
    return out


def format_report(rep: dict, max_cells: int = 12) -> str:
    L = [f"diff {rep['stk']}", f"reference: r={rep['r']} and r={rep['r_fine']} lateral refinement, "
         f"tolerance {rep['tol'] * 100:g}% of max rise"]
    for k, nm in (('r', 'r'), ('r2', '2r')):
        i = rep['reference'][k]
        L.append(f"  ref {nm}: {i['nx']}x{i['ny']}x{i['nz_total']} = {i['unknowns']} unknowns, {i['method']}, "
                 f"residual {i['relative_residual']:.1e}, heat out/in {i['heat_out']:.6g}/{i['power_in']:.6g} W")
    for c in rep['layers']:
        L.append("")
        L.append(f"layer {c['instance']} ({c['layer']})   3D-ICE max rise {c['max_rise_ice']:.3f} K   "
                 f"reference {c['max_rise_ref']:.3f} K   peak diff {c['peak_diff']:+.3f} K")
        L.append(f"  max |dT| {c['max_abs_dT']:.3f} K (tolerance {c['tolerance_K']:.3f} K)   "
                 f"reference r->2r change: max {c['ref_r_vs_2r_max_change']:.3f} K, "
                 f"peak {c['ref_r_vs_2r_peak_change']:+.4f} K")
        L.append(f"  argmax: 3D-ICE ({c['argmax_ice_um'][0]:g}, {c['argmax_ice_um'][1]:g}) um, reference "
                 f"({c['argmax_ref_um'][0]:g}, {c['argmax_ref_um'][1]:g}) um, displacement "
                 f"{c['argmax_displacement_um']:g} um (unreliable on a flat plateau)")
        L.append(f"  cells beyond tolerance: {c['n_flagged_raw']} of {c['n_cells']}; "
                 f"{c['n_explained_by_3dice_vertical_rule']} explained by 3D-ICE's vertical rule "
                 f"(match the r=1, one-cell-per-layer emulation; max |3D-ICE - emulation| "
                 f"{c['max_abs_dT_vs_emulation']:.3f} K)")
        L.append(f"  unexplained: {c['n_flagged']}  ({c['n_flagged_die_edge']} die-edge, "
                 f"{c['n_flagged_edge_adjacent']} edge-adjacent, {c['n_flagged_interior']} interior)")
        for cell in [x for x in c['flagged'] if not x['explained']][:max_cells]:
            L.append(f"    ({cell['i_length']:3d},{cell['j_width']:3d}) x={cell['x_um']:g} y={cell['y_um']:g} um  "
                     f"3D-ICE {cell['T_ice']:.3f}  ref {cell['T_ref']:.3f}  dT {cell['dT']:+.3f}  {cell['cls']}")
        if c['n_flagged'] > max_cells:
            L.append(f"    ... {c['n_flagged'] - max_cells} more (see --json)")
        L.append(f"  verdict: {c['verdict']}")
    L.append("")
    L.append(f"VERDICT: {rep['verdict']}")
    if rep['verdict'] == 'DISAGREE (die-edge)':
        L.append("  disagreement is localised at die/layout edges: partial-cell artefact suspected "
                 "(run `iceforge check`, then `iceforge snap`)")
    elif rep['verdict'] == 'DISAGREE (diffuse)':
        L.append("  disagreement is not localised at edges: look for a model or boundary-condition mismatch "
                 "(heat-sink side, layer order, units, material k)")
    return '\n'.join(L)


def plot_report(rep: dict, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("diff: matplotlib not installed, --plot skipped", file=sys.stderr)
        return False
    layers = rep['layers']
    fig, ax = plt.subplots(len(layers), 3, figsize=(13, 4 * len(layers)), squeeze=False)
    for row, c in enumerate(layers):
        f = c['_fields']
        vmin, vmax = min(f['ice'].min(), f['ref'].min()), max(f['ice'].max(), f['ref'].max())
        for col, (arr, title, cmap) in enumerate(((f['ice'], '3D-ICE', 'inferno'),
                                                  (f['ref'], 'reference', 'inferno'),
                                                  (f['dT'], '3D-ICE - reference', 'RdBu_r'))):
            kw = dict(vmin=vmin, vmax=vmax) if col < 2 else dict(vmin=-abs(f['dT']).max() or -1,
                                                                 vmax=abs(f['dT']).max() or 1)
            im = ax[row][col].imshow(arr.T, origin='lower', cmap=cmap, **kw)
            ax[row][col].set_title(f"{c['instance']}: {title}")
            fig.colorbar(im, ax=ax[row][col], label='K')
            if col == 2:
                jj, ii = np.nonzero(f['flag'].T)
                ax[row][col].plot(ii, jj, 'k.', ms=2)
        ax[row][2].set_xlabel(f"{c['verdict']}; flagged cells dotted")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True
