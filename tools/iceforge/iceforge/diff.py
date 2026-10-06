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

Non-uniform 3D-ICE meshes (``non-uniform true``): there is no regular grid, so the comparison is made PER
3D-ICE CELL: the reference (its own uniform fine grid at r and 2r times the nominal ``cell`` size, mid-plane of
the Tmap layer) is area-averaged exactly over each cell's rectangle (``refsolve.cell_average``) and compared
cell by cell with the same flagging rule.  Edge classes and the vertical-rule explanation need a regular grid
are not computed.  Flagged cells are classified by MESH MISMATCH instead: a cell is ``mesh-edge`` when a cell edge of
another stack element's mesh cuts through it, or one of its own edges cuts through a cell of another mesh (the
non-conforming interface that a regular grid never has), ``mesh-adjacent`` when it touches such a cell of its own
mesh, else ``interior``.  The verdict rule is the uniform one with ``die-edge`` read as ``mesh-edge``.

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


def _all_rects(model: R.RefModel) -> List[Rect]:
    """Every floorplan-element and layout rectangle of every layer (the in-plane union)."""
    rects: List[Rect] = []
    for L in model.layers:
        rects += [r for _, rs in L.patches for r in rs]
        rects += [r for r, _ in L.power_rects]
    return rects


NEAR_EDGE_CELLS = 3      # "near-edge" bucket: within this many cells (chessboard) of an edge cell


def _neighbour_mask(edge: np.ndarray) -> np.ndarray:
    near = edge.copy()
    near[1:, :] |= edge[:-1, :]; near[:-1, :] |= edge[1:, :]
    near[:, 1:] |= edge[:, :-1]; near[:, :-1] |= edge[:, 1:]
    return near


def _edge_distance(edge: np.ndarray) -> np.ndarray:
    """Chessboard distance in cells to the nearest edge-crossing cell (-1 everywhere if there is none)."""
    if not edge.any():
        return np.full(edge.shape, -1, dtype=int)
    from scipy.ndimage import distance_transform_cdt
    return distance_transform_cdt(~edge, metric='chessboard').astype(int)


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


def _load_npz(path: str) -> dict:
    z = np.load(path)
    d = dict(T=z['T'], source=path)
    if 'cells' in z.files:
        d['cells'] = z['cells']
    return d


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
                        out[o.instance] = _load_npz(p)
        elif os.path.isfile(ice_npz):
            out[tm[0].instance] = _load_npz(ice_npz)
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
            out[t['instance']] = _load_npz(t['npz'])
    if not out:
        raise DiffError("3D-ICE produced no parsable Tmap")
    rep_out = os.path.abspath(outdir)
    for v in out.values():
        v['outdir'] = rep_out
    return out


def compare_layer(model: R.RefModel, instance: str, T_ice: np.ndarray, sol_r: R.RefSolution,
                  sol_2r: R.RefSolution, tol: float, cell_l: float, cell_w: float,
                  sol_emu: Optional[R.RefSolution] = None, edge_source: str = 'union',
                  union_edge: Optional[np.ndarray] = None) -> dict:
    if edge_source not in ('union', 'own'):
        raise ValueError("edge_source must be 'union' or 'own'")
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
    edges = {'own': edge_crossing_mask(xe, ye, _layer_rects(model, li)),
             'union': union_edge if union_edge is not None else edge_crossing_mask(xe, ye, _all_rects(model))}
    nears = {k: _neighbour_mask(e) for k, e in edges.items()}
    dists = {k: _edge_distance(e) for k, e in edges.items()}

    def _cls(src, i, j):
        if edges[src][i, j]:
            return 'die-edge'
        if nears[src][i, j]:
            return 'edge-adjacent'
        d = dists[src][i, j]
        return 'near-edge' if 0 <= d <= NEAR_EDGE_CELLS else 'interior'

    emu = sol_emu.layer_field(li) if sol_emu is not None else None
    ii, jj = np.nonzero(flag)
    cells = []
    for i, j in sorted(zip(ii.tolist(), jj.tolist()), key=lambda ij: -abs(dT[ij])):
        cells.append(dict(i_length=i, j_width=j, x_um=(i + 0.5) * cell_l, y_um=(j + 0.5) * cell_w,
                          T_ice=float(T_ice[i, j]), T_ref=float(ref_2r[i, j]), dT=float(dT[i, j]),
                          threshold=float(thr[i, j]),
                          cls=_cls(edge_source, i, j), cls_own=_cls('own', i, j), cls_union=_cls('union', i, j),
                          dist_cells_union=int(dists['union'][i, j]), dist_cells_own=int(dists['own'][i, j]),
                          T_emulated=None if emu is None else float(emu[i, j]),
                          explained=bool(emu is not None and
                                         abs(T_ice[i, j] - emu[i, j]) <= EXPLAIN_FRACTION * tol * max_rise_ref)))
    n_raw = len(cells)
    n_expl = sum(1 for c in cells if c['explained'])
    cells = [c for c in cells if not c['explained']] + [c for c in cells if c['explained']]
    unexpl = [c for c in cells if not c['explained']]

    def _verdict(key):
        n_flag = len(unexpl)
        n_edge = sum(1 for c in unexpl if c[key] == 'die-edge')
        n_adj = sum(1 for c in unexpl if c[key] == 'edge-adjacent')
        if n_flag == 0:
            v = 'AGREE'
        elif n_flag > DIFFUSE_FRACTION * T_ice.size or n_edge + n_adj < EDGE_SHARE * n_flag:
            v = 'DISAGREE (diffuse)'
        else:
            v = 'DISAGREE (die-edge)'
        dk = 'dist_cells_' + key.split('_')[1]
        hist: Dict[str, int] = {}
        for c in unexpl:
            d = c[dk]
            lab = 'no-edge' if d < 0 else str(d) if d < 8 else '>=8'
            hist[lab] = hist.get(lab, 0) + 1
        return dict(verdict=v, n_flagged=n_flag, n_die_edge=n_edge, n_edge_adjacent=n_adj,
                    n_near_edge=sum(1 for c in unexpl if c[key] == 'near-edge'),
                    n_interior=sum(1 for c in unexpl if c[key] == 'interior'), distance_hist=hist)

    by_source = {'own': _verdict('cls_own'), 'union': _verdict('cls_union')}
    sel = by_source[edge_source]
    verdict, n_flag, n_edge, n_adj = sel['verdict'], sel['n_flagged'], sel['n_die_edge'], sel['n_edge_adjacent']
    n_near = sel['n_near_edge']

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
        n_flagged_near_edge=n_near, n_flagged_interior=n_flag - n_edge - n_adj - n_near,
        edge_source=edge_source, by_edge_source=by_source, distance_hist=sel['distance_hist'],
        flagged=cells,
        _fields=dict(ice=T_ice, ref=ref_2r, dT=dT, flag=flag),
    )


def _pair(a: np.ndarray, b: np.ndarray, tol_k: float) -> dict:
    """Agreement of field a with field b: raw, and after removing the best affine map a ~ s*b + o.

    The affine fit is what makes a third solver usable: a different boundary model (HotSpot's package, its
    vertical convention) mostly rescales and offsets the rise field, and that is not a partial-cell effect."""
    d = a - b
    A = np.vstack([b.ravel(), np.ones(b.size)]).T
    (s, o), *_ = np.linalg.lstsq(A, a.ravel(), rcond=None)
    fit = a - (s * b + o)
    return dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())), mean=float(d.mean()),
                fit_max_abs=float(np.abs(fit).max()), fit_rms=float(np.sqrt((fit ** 2).mean())),
                fit_scale=float(s), fit_offset_K=float(o), within_tol=bool(np.abs(d).max() <= tol_k),
                fit_within_tol=bool(np.abs(fit).max() <= tol_k))


def hotspot_side(hr: dict, hi: dict, ir: dict, ice_agrees: bool, tol_k: float) -> str:
    """Which solver HotSpot sides with, from the affine-fit residuals (see ``_pair``).

    When 3D-ICE and the reference already agree (``ice_agrees``, including differences the diff explains by
    3D-ICE's vertical rule), HotSpot is held to the tolerance against the reference and to the tolerance plus
    that accepted 3D-ICE offset against 3D-ICE.  When they disagree, HotSpot's residual to each decides."""
    if ice_agrees:
        ok_hr = hr['fit_max_abs'] <= tol_k
        ok_hi = hi['fit_max_abs'] <= tol_k + ir['fit_max_abs']
        if ok_hr and ok_hi:
            return 'all three agree'
        if ok_hr:
            return 'HotSpot agrees with the reference, not with 3D-ICE'
        return 'HotSpot differs from both (its own modelling difference)'
    ok_hr, ok_hi = hr['fit_max_abs'] <= tol_k, hi['fit_max_abs'] <= tol_k
    if ok_hr and not ok_hi:
        return 'HotSpot sides with the reference'
    if ok_hi and not ok_hr:
        return 'HotSpot sides with 3D-ICE'
    if not ok_hr and not ok_hi:
        if hr['fit_max_abs'] * 2 <= hi['fit_max_abs']:
            return 'HotSpot is closer to the reference (neither within tolerance)'
        if hi['fit_max_abs'] * 2 <= hr['fit_max_abs']:
            return 'HotSpot is closer to 3D-ICE (neither within tolerance)'
    return 'inconclusive'


def _model_t_ref(model: R.RefModel) -> float:
    for h in (model.top, model.bottom):
        if h:
            return h[1]
    return 0.0


def hotspot_opinion(st: Stack, model: R.RefModel, layers: List[dict], tol: float, exe: Optional[str] = None,
                    r: int = 1, grid: Optional[int] = None, log=lambda *a: None) -> dict:
    """Run HotSpot on the same stack and compare it with 3D-ICE and the reference, layer by layer.

    Never raises for a model HotSpot cannot represent: returns ``status`` 'refused' with the reason."""
    from . import hotspot as H
    try:
        case = H.translate(st, r=r, grid=grid)
    except (H.HotspotRefused, R.RefusedModel) as e:
        return dict(status='refused', reason=str(e))
    try:
        log(f"diff: HotSpot grid solve ({case.rows}x{case.cols}, {len(case.layers)} layers) ...")
        fields = H.hotspot_fields(case, exe)
    except H.HotspotError as e:
        return dict(status='error', reason=str(e))
    out = []
    t_ref = _model_t_ref(model)
    for c in layers:
        li = model.tmap_layer(c['instance'])
        lcf = case.lcf_index(li)
        if lcf not in fields:
            return dict(status='error', reason=f"HotSpot output has no layer {lcf} for {c['instance']}")
        T_ice, T_ref = c['_fields']['ice'], c['_fields']['ref']
        T_hs = H.remap(fields[lcf], T_ice.shape[0], T_ice.shape[1])
        tol_k = c['tolerance_K']
        hr, hi, ir = _pair(T_hs, T_ref, tol_k), _pair(T_hs, T_ice, tol_k), _pair(T_ice, T_ref, tol_k)
        out.append(dict(instance=c['instance'], layer=c['layer'], tolerance_K=tol_k,
                        max_rise=dict(ice=float(T_ice.max() - t_ref), reference=float(T_ref.max() - t_ref),
                                      hotspot=float(T_hs.max() - t_ref)),
                        hotspot_vs_reference=hr, hotspot_vs_3dice=hi, ice_vs_reference=ir,
                        side=hotspot_side(hr, hi, ir, c['verdict'] == 'AGREE', tol_k), _T_hs=T_hs))
    return dict(status='ok', rows=case.rows, cols=case.cols, n_hotspot_layers=len(case.layers),
                detailed_3d=case.heterogeneous, grid_remapped=(case.cols, case.rows) != (case.n_cols, case.n_rows),
                n_cols=case.n_cols, n_rows=case.n_rows, layers=out)


def _cut_by(cells: np.ndarray, others: np.ndarray, chunk: int = 512) -> np.ndarray:
    """True for every cell of ``cells`` (N, 4) whose interior is crossed by an edge of a cell in ``others``."""
    N = len(cells)
    out = np.zeros(N, dtype=bool)
    ox0, oy0, ox1, oy1 = others[:, 0], others[:, 1], others[:, 0] + others[:, 2], others[:, 1] + others[:, 3]
    for s0 in range(0, N, chunk):
        c = cells[s0:s0 + chunk]
        cx0, cy0 = c[:, 0:1], c[:, 1:2]
        cx1, cy1 = cx0 + c[:, 2:3], cy0 + c[:, 3:4]
        y_ov = np.minimum(cy1, oy1[None]) - np.maximum(cy0, oy0[None]) > EPS
        x_ov = np.minimum(cx1, ox1[None]) - np.maximum(cx0, ox0[None]) > EPS
        hit = np.zeros(y_ov.shape, dtype=bool)
        for ex in (ox0, ox1):
            hit |= (cx0 + EPS < ex[None]) & (ex[None] < cx1 - EPS) & y_ov
        for ey in (oy0, oy1):
            hit |= (cy0 + EPS < ey[None]) & (ey[None] < cy1 - EPS) & x_ov
        out[s0:s0 + chunk] = hit.any(axis=1)
    return out


def mesh_mismatch_mask(cells: np.ndarray, others: List[np.ndarray]) -> np.ndarray:
    """Cells that are not a clean union/sub-division of the cells of the other meshes (see module docstring)."""
    mask = np.zeros(len(cells), dtype=bool)
    for o in others:
        if len(o) == 0:
            continue
        mask |= _cut_by(cells, o)                 # an edge of the other mesh cuts this cell
        mask |= _edges_cut_cells(cells, o, 512)   # an edge of this cell cuts a cell of the other mesh
    return mask


def _edges_cut_cells(cells: np.ndarray, o: np.ndarray, chunk: int) -> np.ndarray:
    N = len(cells)
    out = np.zeros(N, dtype=bool)
    ox0, oy0, ox1, oy1 = o[:, 0], o[:, 1], o[:, 0] + o[:, 2], o[:, 1] + o[:, 3]
    for s0 in range(0, N, chunk):
        c = cells[s0:s0 + chunk]
        cx0, cy0 = c[:, 0:1], c[:, 1:2]
        cx1, cy1 = cx0 + c[:, 2:3], cy0 + c[:, 3:4]
        y_ov = np.minimum(cy1, oy1[None]) - np.maximum(cy0, oy0[None]) > EPS
        x_ov = np.minimum(cx1, ox1[None]) - np.maximum(cx0, ox0[None]) > EPS
        hit = np.zeros(y_ov.shape, dtype=bool)
        for ex in (cx0, cx1):
            hit |= (ox0[None] + EPS < ex) & (ex < ox1[None] - EPS) & y_ov
        for ey in (cy0, cy1):
            hit |= (oy0[None] + EPS < ey) & (ey < oy1[None] - EPS) & x_ov
        out[s0:s0 + chunk] = hit.any(axis=1)
    return out


def compare_layer_nu(model: R.RefModel, instance: str, T_ice: np.ndarray, cells: np.ndarray, sol_r: R.RefSolution,
                     sol_2r: R.RefSolution, tol: float, meshes: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """Per-cell comparison on a non-uniform 3D-ICE mesh (see the module docstring)."""
    li = model.tmap_layer(instance)
    t_ref = sol_r.t_ref
    if T_ice.shape != (len(cells),):
        raise DiffError(f"{instance}: {T_ice.shape} temperatures for {len(cells)} cells")
    ref_r = R.cell_average(sol_r.layer_field(li), sol_r.xe, sol_r.ye, cells)
    ref_2r = R.cell_average(sol_2r.layer_field(li), sol_2r.xe, sol_2r.ye, cells)
    dT = T_ice - ref_2r
    own = np.abs(ref_2r - ref_r)
    max_rise_ref = float(ref_2r.max() - t_ref)
    thr = np.maximum(tol * max_rise_ref, 3.0 * own)
    flag = np.abs(dT) > thr
    others = [m for k, m in (meshes or {}).items() if k != instance]
    mism = mesh_mismatch_mask(cells, others) if others else np.zeros(len(cells), dtype=bool)
    # adjacency to a mismatched cell of the same mesh
    x0, y0, x1, y1 = cells[:, 0], cells[:, 1], cells[:, 0] + cells[:, 2], cells[:, 1] + cells[:, 3]
    adj = np.zeros(len(cells), dtype=bool)
    if mism.any():
        mi = np.nonzero(mism)[0]
        touch = ((np.minimum(x1[:, None], x1[mi][None]) - np.maximum(x0[:, None], x0[mi][None]) >= -EPS) &
                 (np.minimum(y1[:, None], y1[mi][None]) - np.maximum(y0[:, None], y0[mi][None]) >= -EPS))
        adj = touch.any(axis=1) & ~mism

    def cls(i):
        return 'mesh-edge' if mism[i] else 'mesh-adjacent' if adj[i] else 'interior'

    order = np.argsort(-np.abs(dT))
    flagged = [dict(cell=int(i), x_um=float(cells[i, 0] + 0.5 * cells[i, 2]), y_um=float(cells[i, 1] + 0.5 * cells[i, 3]),
                    length_um=float(cells[i, 2]), width_um=float(cells[i, 3]), T_ice=float(T_ice[i]),
                    T_ref=float(ref_2r[i]), dT=float(dT[i]), threshold=float(thr[i]), cls=cls(i), explained=False)
               for i in order if flag[i]]
    n_flag = len(flagged)
    n_edge = sum(1 for f in flagged if f['cls'] == 'mesh-edge')
    n_adj = sum(1 for f in flagged if f['cls'] == 'mesh-adjacent')
    if n_flag == 0:
        verdict = 'AGREE'
    elif n_edge + n_adj < EDGE_SHARE * n_flag:      # no DIFFUSE_FRACTION test: most cells of a mesh may be mismatched
        verdict = 'DISAGREE (diffuse)'
    else:
        verdict = 'DISAGREE (die-edge)'
    ia, ib = int(np.argmax(T_ice)), int(np.argmax(ref_2r))
    return dict(
        grid='non-uniform', instance=instance, layer=model.layers[li].name, verdict=verdict,
        max_abs_dT=float(np.abs(dT).max()), max_abs_dT_cell=[int(np.argmax(np.abs(dT)))],
        max_rise_ice=float(T_ice.max() - t_ref), max_rise_ref=max_rise_ref,
        peak_diff=float(T_ice.max() - ref_2r.max()),
        argmax_ice_um=[float(cells[ia, 0] + 0.5 * cells[ia, 2]), float(cells[ia, 1] + 0.5 * cells[ia, 3])],
        argmax_ref_um=[float(cells[ib, 0] + 0.5 * cells[ib, 2]), float(cells[ib, 1] + 0.5 * cells[ib, 3])],
        argmax_displacement_um=float(np.hypot(cells[ia, 0] + 0.5 * cells[ia, 2] - cells[ib, 0] - 0.5 * cells[ib, 2],
                                              cells[ia, 1] + 0.5 * cells[ia, 3] - cells[ib, 1] - 0.5 * cells[ib, 3])),
        ref_r_vs_2r_max_change=float(own.max()), ref_r_vs_2r_peak_change=float(ref_2r.max() - ref_r.max()),
        tolerance_K=tol * max_rise_ref, n_cells=int(T_ice.size), n_mesh_mismatch_cells=int(mism.sum()),
        max_abs_dT_vs_emulation=None, n_flagged_raw=n_flag, n_explained_by_3dice_vertical_rule=0,
        n_flagged=n_flag, n_flagged_die_edge=n_edge, n_flagged_edge_adjacent=n_adj, n_flagged_near_edge=0,
        n_flagged_interior=n_flag - n_edge - n_adj, edge_source='mesh', by_edge_source={}, distance_hist={},
        flagged=flagged, _fields=dict(ice=T_ice, ref=ref_2r, dT=dT, flag=flag, cells=cells),
    )


def run_diff(stk_path: str, r: int = 4, nz: Optional[int] = None, tol: float = 0.02,
             ice_npz: Optional[str] = None, backend: str = 'auto', exe: Optional[str] = None,
             outdir: Optional[str] = None, method: str = 'auto', edge_source: str = 'union',
             log=lambda *a: None, with_hotspot: bool = False, hotspot_exe: Optional[str] = None,
             hotspot_r: int = 1, hotspot_grid: Optional[int] = None) -> dict:
    st = parse_stk(stk_path)
    model = R.build_model(st)            # raises RefusedModel
    ice = load_ice(st, ice_npz, backend, exe, outdir, log)
    log(f"diff: reference solve at r={r} ...")
    s1 = R.solve(model, r=r, nz=nz, method=method)
    log(f"diff: reference solve at r={2 * r} ...")
    s2 = R.solve(model, r=2 * r, nz=nz, method=method)
    nu = bool(st.dims.non_uniform)
    s_emu = None if nu else R.solve(model, r=1, nz=1, emulate_3dice_ends=True)
    layers = []
    union_edge = None
    meshes = None
    for inst, d in ice.items():
        try:
            model.tmap_layer(inst)
        except KeyError:
            raise DiffError(f"Tmap instance {inst} is not a die or layer of the stack")
        if nu:
            if 'cells' not in d:
                raise DiffError(f"{inst}: the non-uniform Tmap result carries no cell geometry (rerun `iceforge run`)")
            if meshes is None:
                from .model import non_uniform_cells
                meshes = {}
                for se in st.stack:
                    cl = non_uniform_cells(st, se)
                    if cl:
                        meshes[se.instance] = np.array([[q.x, q.y, q.l, q.w] for q in cl])
            c = compare_layer_nu(model, inst, d['T'], d['cells'], s1, s2, tol, meshes)
            c['ice_source'] = d['source']
            layers.append(c)
            continue
        if union_edge is None:      # once per stack: in-plane union of every layer's floorplan and layout edges
            T0 = d['T']
            union_edge = edge_crossing_mask(np.arange(T0.shape[0] + 1) * st.dims.cell_l,
                                            np.arange(T0.shape[1] + 1) * st.dims.cell_w, _all_rects(model))
        c = compare_layer(model, inst, d['T'], s1, s2, tol, st.dims.cell_l, st.dims.cell_w, s_emu,
                          edge_source=edge_source, union_edge=union_edge)
        c['ice_source'] = d['source']
        layers.append(c)
    def _overall(verdicts):
        if all(v == 'AGREE' for v in verdicts):
            return 'AGREE'
        if any(v == 'DISAGREE (diffuse)' for v in verdicts):
            return 'DISAGREE (diffuse)'
        return 'DISAGREE (die-edge)'
    overall = _overall([c['verdict'] for c in layers])
    verdict_by_source = {k: _overall([c['by_edge_source'].get(k, {'verdict': c['verdict']})['verdict'] for c in layers]) for k in ('own', 'union')}
    hs = hotspot_opinion(st, model, layers, tol, hotspot_exe, hotspot_r, hotspot_grid, log) if with_hotspot else None
    return dict(stk=st.path, verdict=overall, hotspot=hs, edge_source=edge_source, verdict_by_edge_source=verdict_by_source, r=r, r_fine=2 * r, nz=nz, tol=tol,
                reference=dict(r=s1.info, r2=s2.info), layers=layers, _model=model)


def strip_private(rep: dict, max_cells: int = 200) -> dict:
    out = {k: v for k, v in rep.items() if not k.startswith('_')}
    if rep.get('hotspot'):
        hs = dict(rep['hotspot'])
        if 'layers' in hs:
            hs['layers'] = [{k: v for k, v in c.items() if not k.startswith('_')} for c in hs['layers']]
        out['hotspot'] = hs
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
        if c.get('grid') == 'non-uniform':
            L.extend(_format_layer_nu(c))
            continue
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
                 f"{c['n_flagged_edge_adjacent']} edge-adjacent, {c['n_flagged_near_edge']} near-edge, "
                 f"{c['n_flagged_interior']} interior)")
        for cell in [x for x in c['flagged'] if not x['explained']][:max_cells]:
            L.append(f"    ({cell['i_length']:3d},{cell['j_width']:3d}) x={cell['x_um']:g} y={cell['y_um']:g} um  "
                     f"3D-ICE {cell['T_ice']:.3f}  ref {cell['T_ref']:.3f}  dT {cell['dT']:+.3f}  {cell['cls']}")
        if c['n_flagged'] > max_cells:
            L.append(f"    ... {c['n_flagged'] - max_cells} more (see --json)")
        L.append(f"  verdict: {c['verdict']}")
    hs = rep.get('hotspot')
    if hs:
        L.append("")
        L.extend(format_hotspot(hs))
    L.append("")
    L.append(f"VERDICT: {rep['verdict']}")
    if rep['verdict'] == 'DISAGREE (die-edge)':
        if any(c.get('grid') == 'non-uniform' for c in rep['layers']):
            L.append("  disagreement is localised at cells that do not conform to the mesh of another stack element: "
                     "make the discretizations of adjacent elements/layers line up (equal cell edges)")
        else:
            L.append("  disagreement is localised at die/layout edges: partial-cell artefact suspected "
                     "(run `iceforge check`, then `iceforge snap`)")
    elif rep['verdict'] == 'DISAGREE (diffuse)':
        L.append("  disagreement is not localised at edges: look for a model or boundary-condition mismatch "
                 "(heat-sink side, layer order, units, material k)")
    return '\n'.join(L)


def _format_layer_nu(c: dict, max_cells: int = 12) -> List[str]:
    L = ["", f"layer {c['instance']} ({c['layer']})   non-uniform mesh, {c['n_cells']} cells, compared per 3D-ICE cell   "
             f"3D-ICE max rise {c['max_rise_ice']:.3f} K   reference {c['max_rise_ref']:.3f} K   "
             f"peak diff {c['peak_diff']:+.3f} K",
         f"  max |dT| {c['max_abs_dT']:.3f} K (tolerance {c['tolerance_K']:.3f} K)   reference r->2r change: "
         f"max {c['ref_r_vs_2r_max_change']:.3f} K, peak {c['ref_r_vs_2r_peak_change']:+.4f} K",
         f"  cells beyond tolerance: {c['n_flagged']} of {c['n_cells']} ({c['n_mesh_mismatch_cells']} cells of this "
         f"mesh do not conform to another stack element's mesh; no vertical-rule explanation on a non-uniform mesh)",
         f"  flagged: {c['n_flagged_die_edge']} mesh-edge, {c['n_flagged_edge_adjacent']} mesh-adjacent, "
         f"{c['n_flagged_interior']} interior"]
    for cell in c['flagged'][:max_cells]:
        L.append(f"    cell {cell['cell']:4d} x={cell['x_um']:g} y={cell['y_um']:g} um ({cell['length_um']:g} x "
                 f"{cell['width_um']:g})  3D-ICE {cell['T_ice']:.3f}  ref {cell['T_ref']:.3f}  dT {cell['dT']:+.3f}  "
                 f"{cell['cls']}")
    if c['n_flagged'] > max_cells:
        L.append(f"    ... {c['n_flagged'] - max_cells} more (see --json)")
    L.append(f"  verdict: {c['verdict']}")
    return L


def format_hotspot(hs: dict) -> List[str]:
    if hs['status'] != 'ok':
        return [f"HotSpot third opinion: {hs['status']}: {hs['reason']}"]
    g = f"{hs['rows']}x{hs['cols']}"
    if hs['grid_remapped']:
        g += f" (area-averaged onto the {hs['n_cols']}x{hs['n_rows']} 3D-ICE cells)"
    L = [f"HotSpot third opinion: grid {g}, {hs['n_hotspot_layers']} HotSpot layers (two per 3D-ICE layer), "
         f"per-tile conductivity (detailed-3D): {'yes' if hs['detailed_3d'] else 'no'}"]
    for c in hs['layers']:
        hr, hi, ir = c['hotspot_vs_reference'], c['hotspot_vs_3dice'], c['ice_vs_reference']
        mr = c['max_rise']
        L.append(f"layer {c['instance']} ({c['layer']}), tolerance {c['tolerance_K']:.3f} K")
        L.append("  solver     max rise K | max|dT| vs reference raw/fit | max|dT| vs 3D-ICE raw/fit")
        L.append(f"  3D-ICE     {mr['ice']:10.3f} | {ir['max_abs']:9.3f} / {ir['fit_max_abs']:7.3f}        |"
                 f" {'-':>9s}")
        L.append(f"  reference  {mr['reference']:10.3f} | {'-':>9s}                    |"
                 f" {ir['max_abs']:9.3f} / {ir['fit_max_abs']:7.3f}")
        L.append(f"  HotSpot    {mr['hotspot']:10.3f} | {hr['max_abs']:9.3f} / {hr['fit_max_abs']:7.3f}        |"
                 f" {hi['max_abs']:9.3f} / {hi['fit_max_abs']:7.3f}")
        L.append(f"  (fit = after the best affine map; HotSpot ~ {hr['fit_scale']:.3f} x reference rise "
                 f"{hr['fit_offset_K']:+.3f} K, its own boundary/vertical model)")
        L.append(f"  => {c['side']}")
    sides = {c['side'] for c in hs['layers']}
    L.append(f"HotSpot overall: {sides.pop() if len(sides) == 1 else 'mixed: ' + '; '.join(sorted(sides))}")
    return L


def plot_report(rep: dict, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("diff: matplotlib not installed, --plot skipped", file=sys.stderr)
        return False
    layers = [c for c in rep['layers'] if c.get('grid') != 'non-uniform']
    if not layers:
        print("diff: --plot is not available for non-uniform meshes", file=sys.stderr)
        return False
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
