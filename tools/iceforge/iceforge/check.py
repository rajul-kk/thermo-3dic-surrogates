"""Pre-solve lint rules (S001-S009) for a parsed 3D-ICE stack.

Background.  3D-ICE gives a floorplan element's power to grid cells by OVERLAP AREA
(floorplan_matrix.c), but a ``layout`` (.lyt) assigns material per layer cell by CELL
CENTRE.  When a die edge falls mid-cell, the edge cell receives power but its centre
may fall outside the layout footprint, so it becomes the layer's gap material: a
heated insulator that runs hot.  The rules below detect that situation before a solve.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import Stack, Rect, Layout, DieLayer, resolved_die_layers, parse_stk, non_uniform_cells

GRID_TOL = 1e-6     # in cell fractions
GEOM_EPS = 1e-6     # um
K_RANGE = (1e-9, 1e-2)
HTC_RANGE = (1e-12, 1e-3)
H_RANGE = (1.0, 1e5)
GAP_RATIO = 0.5     # base material counts as "gap" if k < GAP_RATIO * max k over layout materials


@dataclass
class Finding:
    code: str
    severity: str          # error | warning | info
    message: str
    file: Optional[str] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        loc = f"{os.path.basename(self.file)}: " if self.file else ""
        return f"{self.code} {self.severity:7s} {loc}{self.message}"


def edge_offset(v: float, cell: float) -> float:
    """Signed distance from v to the nearest cell boundary, in cell fractions."""
    r = v / cell
    return r - round(r)


def _off(v, cell):
    return abs(edge_offset(v, cell)) > GRID_TOL


def _rect_edges(r: Rect, d):
    """Yield (edge name, value, cell size, chip extent) for the four edges."""
    yield 'x_min', r.x, d.cell_l, d.chip_l
    yield 'x_max', r.x1, d.cell_l, d.chip_l
    yield 'y_min', r.y, d.cell_w, d.chip_w
    yield 'y_max', r.y1, d.cell_w, d.chip_w


def _off_grid_edges(r: Rect, d):
    out = []
    for name, v, c, ext in _rect_edges(r, d):
        if abs(v - ext) < GEOM_EPS or abs(v) < GEOM_EPS:   # chip boundary: always fine
            continue
        if _off(v, c):
            out.append((name, v, edge_offset(v, c)))
    return out


def _material_grid(layout: Layout, base: str, d) -> np.ndarray:
    """Material name per cell (n_cols, n_rows) as 3D-ICE assigns it: by cell centre."""
    nc, nr = d.n_cols, d.n_rows
    xc = (np.arange(nc) + 0.5) * d.cell_l
    yc = (np.arange(nr) + 0.5) * d.cell_w
    grid = np.full((nc, nr), base, dtype=object)
    for mat, rects in layout.shapes.items():
        for r in rects:
            mx = (xc >= r.x) & (xc < r.x1)
            my = (yc >= r.y) & (yc < r.y1)
            grid[np.ix_(mx, my)] = mat
    return grid


def _overlap_cells(r: Rect, d):
    """(i0, i1, j0, j1) inclusive index ranges of cells with positive overlap."""
    i0 = int(np.floor(r.x / d.cell_l + GRID_TOL))
    i1 = int(np.ceil(r.x1 / d.cell_l - GRID_TOL)) - 1
    j0 = int(np.floor(r.y / d.cell_w + GRID_TOL))
    j1 = int(np.ceil(r.y1 / d.cell_w - GRID_TOL)) - 1
    return max(i0, 0), min(i1, d.n_cols - 1), max(j0, 0), min(j1, d.n_rows - 1)


def _centre_cells(r: Rect, d):
    xc = (np.arange(d.n_cols) + 0.5) * d.cell_l
    yc = (np.arange(d.n_rows) + 0.5) * d.cell_w
    mx = (xc >= r.x) & (xc < r.x1)
    my = (yc >= r.y) & (yc < r.y1)
    m = np.zeros((d.n_cols, d.n_rows), dtype=bool)
    m[np.ix_(mx, my)] = True
    return m


def _overlap_mask(r: Rect, d):
    m = np.zeros((d.n_cols, d.n_rows), dtype=bool)
    i0, i1, j0, j1 = _overlap_cells(r, d)
    if i1 >= i0 and j1 >= j0:
        m[i0:i1 + 1, j0:j1 + 1] = True
    return m


def _k_of(st: Stack, layout: Layout, name: str) -> Optional[float]:
    m = layout.local_materials.get(name) or st.materials.get(name)
    return m.k_max if m and m.k else None


def _is_gap(st: Stack, layout: Layout, base: str) -> bool:
    kb = _k_of(st, layout, base)
    ks = [_k_of(st, layout, m) for m in layout.shapes]
    ks = [k for k in ks if k is not None]
    if kb is None or not ks:
        return False
    return kb < GAP_RATIO * max(ks)


def power_of(e) -> float:
    return e.power[0] if e.power else 0.0


def check_stack(st: Stack) -> List[Finding]:
    F: List[Finding] = []
    d = st.dims
    dims_ok = d.cell_l > 0 and d.cell_w > 0 and d.chip_l > 0 and d.chip_w > 0
    grid_ok = (not d.non_uniform) and dims_ok

    if d.non_uniform:
        _check_non_uniform(st, F)

    # S008 missing files
    for kind, p in st.missing_files:
        F.append(Finding('S008', 'error', f"missing {kind} file: {p}", file=p))

    # S006 chip vs cell (also sets the default discretisation of a non-uniform model)
    if dims_ok:
        for what, ext, c in (('length', d.chip_l, d.cell_l), ('width', d.chip_w, d.cell_w)):
            if _off(ext, c):
                F.append(Finding('S006', 'warning',
                                 f"chip {what} {ext:g} um is not an integer multiple of cell {what} {c:g} um "
                                 f"(3D-ICE truncates to {int(ext / c)} cells)", file=st.path))

    # S007 unit sanity
    for m in st.materials.values():
        for k in m.k:
            if not (K_RANGE[0] <= k <= K_RANGE[1]):
                F.append(Finding('S007', 'warning',
                                 f"material {m.name}: k = {k:g} outside [{K_RANGE[0]:g}, {K_RANGE[1]:g}] W/um/K "
                                 f"(W/m/K value entered unconverted?)", file=st.path))
                break
    for p, ly in st.layouts.items():
        for m in ly.local_materials.values():
            for k in m.k:
                if not (K_RANGE[0] <= k <= K_RANGE[1]):
                    F.append(Finding('S007', 'warning',
                                     f"layout material {m.name}: k = {k:g} outside "
                                     f"[{K_RANGE[0]:g}, {K_RANGE[1]:g}] W/um/K", file=p))
                    break
    for h in st.heatsinks:
        if h.htc is not None and not (HTC_RANGE[0] <= h.htc <= HTC_RANGE[1]):
            F.append(Finding('S007', 'warning',
                             f"{h.side} heat sink HTC {h.htc:g} outside [{HTC_RANGE[0]:g}, {HTC_RANGE[1]:g}] "
                             f"W/um^2/K", file=st.path))
    heights = [(ld.name, ld.height) for ld in st.layers.values()]
    for dd in st.dies.values():
        heights += [(f"{dd.name} layer", dl.height) for dl in dd.layers if dl.ref is None]
    for name, hgt in heights:
        if hgt is not None and not (H_RANGE[0] <= hgt <= H_RANGE[1]):
            F.append(Finding('S007', 'warning',
                             f"layer {name}: height {hgt:g} um outside [{H_RANGE[0]:g}, {H_RANGE[1]:g}] um",
                             file=st.path))

    # per-floorplan geometry rules (S005 independent of grid)
    total_power = 0.0
    seen_flp = set()
    for se in st.stack:
        if se.kind != 'die' or not se.floorplan_path or se.floorplan_path not in st.floorplans:
            continue
        fp = st.floorplans[se.floorplan_path]
        total_power += sum(power_of(e) for e in fp.elements)
        dd = st.dies.get(se.ref)
        layers = resolved_die_layers(st, dd) if dd else []
        key = (se.floorplan_path, se.ref)
        if key in seen_flp:
            continue
        seen_flp.add(key)
        _check_floorplan(st, F, fp, layers, grid_ok, se)

    # S003 layout rectangles off grid
    if grid_ok:
        for p, ly in st.layouts.items():
            for mat, rects in ly.shapes.items():
                for r in rects:
                    offs = _off_grid_edges(r, d)
                    if offs:
                        txt = ', '.join(f"{n}={v:g} ({o:+.3f} cell)" for n, v, o in offs)
                        F.append(Finding(
                            'S003', 'error',
                            f"layout rectangle ({r.x:g},{r.y:g},{r.l:g},{r.w:g}) [{mat}] edge(s) {txt} off the "
                            f"grid; 3D-ICE assigns material by cell centre, not by the drawn shape", file=p,
                            detail=dict(material=mat, edges=[dict(edge=n, value=v, offset_cells=o)
                                                             for n, v, o in offs])))

    # S009
    if total_power <= 0:
        F.append(Finding('S009', 'warning', "total floorplan power is <= 0", file=st.path))

    order = {'error': 0, 'warning': 1, 'info': 2}
    F.sort(key=lambda f: (order[f.severity], f.code))
    return F


def _layout_users(st: Stack):
    """(stack element, layout path) for every stack element whose layers use a layout."""
    for se in st.stack:
        paths = []
        if se.kind == 'layer':
            ld = st.layers.get(se.ref)
            if ld and ld.layout_path:
                paths.append(ld.layout_path)
        elif se.kind == 'die':
            dd = st.dies.get(se.ref)
            for dl in (resolved_die_layers(st, dd) if dd else []):
                if dl.layout_path and dl.layout_path not in paths:
                    paths.append(dl.layout_path)
        for p in paths:
            yield se, p


def _rect_edges_in_cells(r: Rect, cells: List[Rect]) -> int:
    """Number of the four edges of ``r`` that cut through the interior of some cell of ``cells``."""
    n = 0
    for name, v, lo, hi in (('x0', r.x, r.y, r.y1), ('x1', r.x1, r.y, r.y1),
                            ('y0', r.y, r.x, r.x1), ('y1', r.y1, r.x, r.x1)):
        vert = name[0] == 'x'
        for c in cells:
            a0, a1, b0, b1 = (c.x, c.x1, c.y, c.y1) if vert else (c.y, c.y1, c.x, c.x1)
            if a0 + GEOM_EPS < v < a1 - GEOM_EPS and min(b1, hi) - max(b0, lo) > GEOM_EPS:
                n += 1
                break
    return n


def _check_non_uniform(st: Stack, F: List[Finding]) -> None:
    """Rules for ``non-uniform true`` models (3D-ICE 4.0; behaviour verified by experiment, see README).

    The mesh is not a grid: each die is cut along its floorplan elements and every cell is one element
    sub-cell, so a floorplan edge is a cell edge by construction (S001/S002 cannot happen) and a layout is not
    read at all (S003/S004 are replaced by N001).  What can go wrong instead is N001-N004."""
    d = st.dims
    cells_of = {se.instance: non_uniform_cells(st, se) for se in st.stack}
    n_cells = {k: len(v) for k, v in cells_of.items() if v is not None}
    F.append(Finding('I001', 'info',
                     "non-uniform grid: the mesh follows the floorplan elements, not a regular grid; "
                     "uniform-grid rules S001-S004 do not apply (see N001-N004). Cells per stack element: "
                     + ', '.join(f"{k} {v}" for k, v in n_cells.items()), file=st.path,
                     detail=dict(cells=n_cells)))
    # N001: layouts are ignored
    reported = set()
    for se, p in _layout_users(st):
        ly = st.layouts.get(p)
        if ly is None or (se.instance, p) in reported:
            continue
        reported.add((se.instance, p))
        cells = cells_of.get(se.instance) or []
        rects = [r for rs in ly.shapes.values() for r in rs]
        cutting = sum(1 for r in rects if _rect_edges_in_cells(r, cells) > 0) if cells else 0
        F.append(Finding(
            'N001', 'error',
            f"layout {os.path.basename(p)} on {se.instance} has no effect in a non-uniform model: 3D-ICE 4.0 builds "
            f"the cells from the floorplan elements and takes every cell's material from the layer "
            f"(identical Tmap with and without the .lyt); {cutting} of its {len(rects)} rectangle(s) also cut "
            f"through cells. Use a uniform grid, or put the material in the floorplan element (material line)",
            file=p, detail=dict(stack_element=se.instance, n_rectangles=len(rects), n_cutting_cells=cutting)))
    # N002: die area not covered by floorplan elements has no cells
    chip = d.chip_l * d.chip_w
    for se in st.stack:
        if se.kind != 'die':
            continue
        cells = cells_of.get(se.instance)
        if cells is None:
            continue
        area = sum(c.l * c.w for c in cells)
        if chip > 0 and area < chip * (1 - 1e-6):
            F.append(Finding(
                'N002', 'error',
                f"die {se.instance}: floorplan elements cover {100 * area / chip:.1f} % of the {d.chip_l:g} x "
                f"{d.chip_w:g} um chip; in a non-uniform model the uncovered {100 * (1 - area / chip):.1f} % "
                f"has no thermal cells at all (a void, no heat flows through it)", file=se.floorplan_path,
                detail=dict(stack_element=se.instance, covered_fraction=area / chip)))
    # N003: bottom heat sink segfaults 3D-ICE 4.0
    if any(h.side == 'bottom' for h in st.heatsinks):
        F.append(Finding(
            'N003', 'error',
            "bottom heat sink with a non-uniform grid crashes 3D-ICE 4.0 (segmentation fault while preparing "
            "thermal data in the tested build; the shipped non-uniform example uses a top heat sink)", file=st.path))
    # N004: only the first rectangle of an element is meshed; fractional discretisation is truncated
    for p, fp in st.floorplans.items():
        for e in fp.elements:
            if len(e.rects) > 1:
                F.append(Finding('N004', 'warning',
                                 f"element {e.name} has {len(e.rects)} rectangles; a non-uniform mesh uses only the "
                                 f"first one", file=p, detail=dict(element=e.name)))
            if e.discretization and any(abs(v - int(v)) > 1e-9 for v in e.discretization):
                F.append(Finding('N004', 'warning',
                                 f"element {e.name}: discretization {e.discretization} is truncated to integers",
                                 file=p, detail=dict(element=e.name)))
    for se in st.stack:
        if se.discretization and any(abs(v - int(v)) > 1e-9 for v in se.discretization):
            F.append(Finding('N004', 'warning',
                             f"{se.instance}: discretization {se.discretization} is truncated to integers",
                             file=st.path))


def _check_floorplan(st, F, fp, layers: List[DieLayer], grid_ok, se):
    d = st.dims
    # S005: outside chip / overlapping
    for e in fp.elements:
        for r in e.rects:
            if (r.x < -GEOM_EPS or r.y < -GEOM_EPS or r.x1 > d.chip_l + GEOM_EPS
                    or r.y1 > d.chip_w + GEOM_EPS):
                F.append(Finding('S005', 'error',
                                 f"element {e.name} ({r.x:g},{r.y:g},{r.l:g},{r.w:g}) lies outside the "
                                 f"{d.chip_l:g} x {d.chip_w:g} um chip", file=fp.path,
                                 detail=dict(element=e.name)))
    flat = [(e.name, r) for e in fp.elements for r in e.rects]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            (na, a), (nb, b) = flat[i], flat[j]
            if na == nb:
                continue
            ox = min(a.x1, b.x1) - max(a.x, b.x)
            oy = min(a.y1, b.y1) - max(a.y, b.y)
            if ox > GEOM_EPS and oy > GEOM_EPS:
                F.append(Finding('S005', 'error',
                                 f"elements {na} and {nb} overlap ({ox:g} x {oy:g} um)", file=fp.path,
                                 detail=dict(elements=[na, nb])))
    if not grid_ok:
        return

    layout_layers, _seen = [], set()
    for dl in layers:    # one entry per distinct layout file (split die layers often share one)
        if dl.layout_path and dl.layout_path in st.layouts and (dl.layout_path, dl.material) not in _seen:
            _seen.add((dl.layout_path, dl.material))
            layout_layers.append(dl)
    reported_s002 = set()
    for e in fp.elements:
        if power_of(e) <= 0 and not any(p > 0 for p in e.power):
            continue
        # off-grid edges of this element
        offs = []
        for r in e.rects:
            for name, v, off in _off_grid_edges(r, d):
                offs.append((r, name, v, off))
        if not offs:
            continue
        artefact = False
        for dl in layout_layers:
            ly = st.layouts[dl.layout_path]
            if not ly.shapes:
                continue
            base = dl.material or ''
            gap = _is_gap(st, ly, base)
            if not gap:
                continue
            mat = _material_grid(ly, base, d)
            inside = np.zeros_like(mat, dtype=bool)
            touched = np.zeros_like(mat, dtype=bool)
            for r in e.rects:
                inside |= _centre_cells(r, d)
                touched |= _overlap_mask(r, d)
            bad = touched & ~inside & (mat == base)
            if bad.any():
                artefact = True
                ij = np.argwhere(bad)
                edges_txt = ', '.join(f"{n}={v:g} ({o:+.3f} cell)" for _, n, v, o in offs)
                F.append(Finding(
                    'S001', 'error',
                    f"element {e.name}: off-grid edge(s) {edges_txt} with layout {os.path.basename(dl.layout_path)}; "
                    f"{len(ij)} powered edge cell(s) have their centre outside the layout footprint and become "
                    f"gap material '{base}' (heated insulator); first cell (col,row) = {tuple(int(v) for v in ij[0])}",
                    file=fp.path,
                    detail=dict(element=e.name, layout=dl.layout_path, n_cells=len(ij),
                                edges=[dict(edge=n, value=v, offset_cells=o) for _, n, v, o in offs])))
        if not artefact and e.name not in reported_s002:
            reported_s002.add(e.name)
            edges_txt = ', '.join(f"{n}={v:g} ({o:+.3f} cell)" for _, n, v, o in offs)
            why = ("power is split by overlap area; no layout cell is affected"
                   if layout_layers else "power is smeared over the partly covered edge cells (mild)")
            F.append(Finding('S002', 'warning',
                             f"element {e.name}: edge(s) {edges_txt} off the cell grid; {why}",
                             file=fp.path,
                             detail=dict(element=e.name,
                                         edges=[dict(edge=n, value=v, offset_cells=o) for _, n, v, o in offs])))

    # S004: powered area (cells whose centre lies in the element) over gap material
    for dl in layout_layers:
        ly = st.layouts[dl.layout_path]
        if not ly.shapes:
            continue
        base = dl.material or ''
        if not _is_gap(st, ly, base):
            continue
        mat = _material_grid(ly, base, d)
        for e in fp.elements:
            if not any(p > 0 for p in e.power):
                continue
            inside = np.zeros_like(mat, dtype=bool)
            for r in e.rects:
                inside |= _centre_cells(r, d)
            bad = inside & (mat == base)
            if bad.any():
                F.append(Finding(
                    'S004', 'error',
                    f"element {e.name}: {int(bad.sum())} of {int(inside.sum())} cells under the element lie "
                    f"outside the layout footprint ({os.path.basename(dl.layout_path)}) and take gap material "
                    f"'{base}'", file=fp.path,
                    detail=dict(element=e.name, layout=dl.layout_path, n_cells=int(bad.sum()))))


def check_file(path: str) -> Tuple[Stack, List[Finding]]:
    st = parse_stk(path)
    return st, check_stack(st)


def exit_code(findings: List[Finding], strict: bool = False) -> int:
    if any(f.severity == 'error' for f in findings):
        return 1
    if strict and any(f.severity == 'warning' for f in findings):
        return 1
    return 0


def format_findings(findings: List[Finding]) -> str:
    if not findings:
        return "check: no findings"
    n_e = sum(f.severity == 'error' for f in findings)
    n_w = sum(f.severity == 'warning' for f in findings)
    return '\n'.join(str(f) for f in findings) + f"\ncheck: {n_e} error(s), {n_w} warning(s)"
