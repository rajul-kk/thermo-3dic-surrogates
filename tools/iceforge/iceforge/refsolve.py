"""Independent steady-state reference solver for the 3D-ICE stack subset (needs scipy).

Purpose: a second opinion that shares NONE of 3D-ICE's discretisation choices, so that a
disagreement points at a discretisation artefact rather than at a shared assumption.

Differences from 3D-ICE, on purpose
  * Lateral grid refined by ``r`` relative to the .stk cell size (3D-ICE: r = 1).
  * Each stack layer is split into ``nz`` sub-layers, at most 25 um thick by default
    (3D-ICE: one cell per layer).
  * Material comes from the EXACT area fraction of every fine cell covered by each ``.lyt``
    rectangle (3D-ICE: material of the cell centre). Power is spread uniformly over the element
    area and over the source layer thickness; the fine-cell share is the exact rectangle
    overlap (3D-ICE does overlap area too, but on the coarse grid and with centre-sampled material).

Modelling choices (documented because they are choices)
  * Mixed cells use the arithmetic area-weighted conductivity for all three directions. For the
    vertical direction this is exact (side-by-side columns conduct in parallel); in-plane it is the
    parallel-to-interface bound. The harmonic mean would be the other bound for in-plane flow across
    an interface. Refinement shrinks the number of mixed cells, so the r vs 2r check covers the
    sensitivity.
  * Heat sink: ``top heat sink`` is a convective condition on the top face of the first stack
    element, ``bottom heat sink`` on the bottom face of the last one (verified against 3D-ICE 4.0
    ``thermal_grid.c`` / ``system_matrix.c``: conductance 2 k h A / (dz h + 2 k), i.e. half a cell of
    solid in series with the film). All other faces are adiabatic.
  * The temperature compared with a Tmap is the mid-plane value of the die's source layer (3D-ICE's
    node sits at the layer centre; with an even number of sub-layers the two middle ones are averaged).

Units follow 3D-ICE: um, W, K, k in W/um/K, HTC in W/um^2/K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import Material, Rect, Stack, resolved_die_layers


class RefusedModel(ValueError):
    """The model uses a feature the reference solver does not implement."""


# --------------------------------------------------------------------------- geometry

def overlap_1d(edges: np.ndarray, a: float, b: float) -> np.ndarray:
    """Length of [a, b] inside each interval [edges[i], edges[i+1]]."""
    lo = np.maximum(edges[:-1], a)
    hi = np.minimum(edges[1:], b)
    return np.clip(hi - lo, 0.0, None)


def rect_overlap_area(xe: np.ndarray, ye: np.ndarray, rect: Rect) -> np.ndarray:
    """Exact overlap area (um^2) of ``rect`` with every cell of the grid (nx, ny)."""
    return np.outer(overlap_1d(xe, rect.x, rect.x1), overlap_1d(ye, rect.y, rect.y1))


def area_fractions(xe: np.ndarray, ye: np.ndarray, rects: List[Rect]) -> np.ndarray:
    """Fraction (0..1) of every cell covered by the union of non-overlapping rectangles.

    Overlapping rectangles are not unioned exactly; the sum is clipped to 1.
    """
    cell = np.outer(np.diff(xe), np.diff(ye))
    tot = np.zeros_like(cell)
    for r in rects:
        tot += rect_overlap_area(xe, ye, r)
    return np.clip(tot / cell, 0.0, 1.0)


# --------------------------------------------------------------------------- model

@dataclass
class RefLayer:
    name: str                       # layer or die-layer description
    instance: str                   # stack instance name
    height: float
    base: Material
    # (material, rectangles) in file order; the remainder of the layer takes ``base``
    patches: List[Tuple[Material, List[Rect]]] = field(default_factory=list)
    power_rects: List[Tuple[Rect, float]] = field(default_factory=list)   # (rect, W/um^2)
    is_source: bool = False
    nz: int = 2


@dataclass
class RefModel:
    stk: Stack
    layers: List[RefLayer]                  # top to bottom
    chip_l: float
    chip_w: float
    cell_l: float
    cell_w: float
    top: Optional[Tuple[float, float]]      # (HTC, T_amb)
    bottom: Optional[Tuple[float, float]]
    total_power: float = 0.0

    def tmap_layer(self, instance: str) -> int:
        """Index (top to bottom) of the layer a Tmap of ``instance`` reports."""
        idx = [i for i, L in enumerate(self.layers) if L.instance == instance]
        if not idx:
            raise KeyError(instance)
        src = [i for i in idx if self.layers[i].is_source]
        return (src or idx)[0]


def _k3(m: Material) -> Tuple[float, float, float]:
    if not m.k:
        raise RefusedModel(f"material {m.name} has no thermal conductivity")
    if len(m.k) == 1:
        return (m.k[0],) * 3
    if len(m.k) >= 3:
        return tuple(m.k[:3])
    raise RefusedModel(f"material {m.name}: unsupported conductivity list {m.k}")


def build_model(st: Stack) -> RefModel:
    d = st.dims
    if d.non_uniform:
        raise RefusedModel("non-uniform grids are not supported by the reference solver")
    if st.solver.mode == 'transient':
        raise RefusedModel("transient solves are not supported (the reference solver is steady state only)")
    if any('pluggable' in w.lower() for w in st.warnings):
        raise RefusedModel("pluggable heat sinks are not supported")
    if any(se.kind == 'channel' for se in st.stack):
        raise RefusedModel("microchannel layers are not supported by the reference solver")
    if st.missing_files:
        raise RefusedModel("missing files: " + ", ".join(p for _, p in st.missing_files))
    if not d.cell_l or not d.cell_w or d.chip_l <= 0 or d.chip_w <= 0:
        raise RefusedModel("dimensions are incomplete")
    if abs(d.chip_l / d.cell_l - round(d.chip_l / d.cell_l)) > 1e-9 or \
            abs(d.chip_w / d.cell_w - round(d.chip_w / d.cell_w)) > 1e-9:
        raise RefusedModel("chip size is not an integer multiple of the cell size")
    if not st.stack:
        raise RefusedModel("empty stack")

    def mat(name: Optional[str], lyt=None) -> Material:
        if name is None:
            raise RefusedModel("layer without a material")
        if lyt is not None and name in lyt.local_materials:
            return lyt.local_materials[name]
        if name in st.materials:
            return st.materials[name]
        for ly in st.layouts.values():
            if name in ly.local_materials:
                return ly.local_materials[name]
        raise RefusedModel(f"unknown material {name}")

    def patches(layout_path: Optional[str]):
        if not layout_path:
            return []
        ly = st.layouts.get(layout_path)
        if ly is None:
            raise RefusedModel(f"layout not loaded: {layout_path}")
        return [(mat(n, ly), list(rs)) for n, rs in ly.shapes.items()]

    layers: List[RefLayer] = []
    total_power = 0.0
    for se in st.stack:
        if se.kind == 'layer':
            ld = st.layers.get(se.ref)
            if ld is None or ld.height is None:
                raise RefusedModel(f"unknown layer {se.ref}")
            layers.append(RefLayer(name=ld.name, instance=se.instance, height=ld.height,
                                   base=mat(ld.material), patches=patches(ld.layout_path)))
        elif se.kind == 'die':
            dd = st.dies.get(se.ref)
            if dd is None:
                raise RefusedModel(f"unknown die {se.ref}")
            fp = st.floorplans.get(se.floorplan_path) if se.floorplan_path else None
            prects: List[Tuple[Rect, float]] = []
            if fp:
                for e in fp.elements:
                    area = sum(r.l * r.w for r in e.rects)
                    p = e.power[0] if e.power else 0.0
                    if area > 0 and p:
                        prects += [(r, p / area) for r in e.rects]
                        total_power += p
            for k, dl in enumerate(resolved_die_layers(st, dd)):
                if dl.height is None:
                    raise RefusedModel(f"unknown layer in die {dd.name}")
                L = RefLayer(name=dl.ref or f"{dd.name}[{k}]", instance=se.instance, height=dl.height,
                             base=mat(dl.material), patches=patches(dl.layout_path), is_source=dl.is_source)
                if dl.is_source:
                    L.power_rects = prects
                layers.append(L)
    if not layers:
        raise RefusedModel("no layers")
    n_src = sum(1 for L in layers if L.is_source)
    if n_src > 1:
        total_power *= n_src

    if total_power <= 0:
        raise RefusedModel("the model dissipates no power (floorplan power is zero or missing): nothing to compare")
    top = bot = None
    for h in st.heatsinks:
        if h.htc is None or h.temperature is None:
            raise RefusedModel(f"{h.side} heat sink without HTC/temperature")
        if h.side == 'top':
            top = (h.htc, h.temperature)
        else:
            bot = (h.htc, h.temperature)
    if top is None and bot is None:
        raise RefusedModel("no heat sink declared: the steady problem is singular")
    return RefModel(st, layers, d.chip_l, d.chip_w, d.cell_l, d.cell_w, top, bot, total_power)


# --------------------------------------------------------------------------- solve

@dataclass
class RefSolution:
    r: int
    T: np.ndarray                      # (nz_total, nx, ny) K, index 0 = top sub-layer
    layer_slices: List[slice]          # per model layer
    xe: np.ndarray
    ye: np.ndarray
    nx: int
    ny: int
    t_ref: float
    info: dict

    def layer_field(self, i: int) -> np.ndarray:
        """Mid-plane temperature of layer ``i`` on the fine grid (nx, ny)."""
        s = self.layer_slices[i]
        n = s.stop - s.start
        if n % 2:
            return self.T[s.start + n // 2]
        return 0.5 * (self.T[s.start + n // 2 - 1] + self.T[s.start + n // 2])


def block_average(F: np.ndarray, r: int) -> np.ndarray:
    nx, ny = F.shape
    return F.reshape(nx // r, r, ny // r, r).mean(axis=(1, 3))


def default_nz(height: float, max_dz: float = 25.0) -> int:
    return max(2, int(math.ceil(height / max_dz - 1e-9)))


def solve(model: RefModel, r: int = 4, nz: Optional[int] = None, method: str = 'auto',
          tol: float = 1e-10, emulate_3dice_ends: bool = False) -> RefSolution:
    """Solve the steady problem. ``emulate_3dice_ends`` is a diagnostic (use with r=1, nz=1): it
    reproduces 3D-ICE 4.0's end-layer rule (a stack-end layer without a heat sink couples to its
    neighbour through its FULL height, ``thermal_grid.c`` get_conductance_top/bottom, instead of half),
    to show that this rule explains the bulk difference on aligned models."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("the reference solver needs scipy: pip install iceforge[diff]") from e

    nx = int(round(model.chip_l / model.cell_l)) * r
    ny = int(round(model.chip_w / model.cell_w)) * r
    xe = np.linspace(0.0, model.chip_l, nx + 1)
    ye = np.linspace(0.0, model.chip_w, ny + 1)
    dx, dy = model.chip_l / nx, model.chip_w / ny
    cell_area = np.outer(np.diff(xe), np.diff(ye))

    kx_l, ky_l, kz_l, dz_l, slices, P_l = [], [], [], [], [], []
    z0 = 0
    for L in model.layers:
        n = nz if nz else default_nz(L.height)
        base = _k3(L.base)
        # arithmetic area-weighted conductivity per direction
        rem = np.ones((nx, ny))
        k = [np.zeros((nx, ny)) for _ in range(3)]
        for m, rects in L.patches:
            f = np.minimum(area_fractions(xe, ye, rects), rem)
            rem -= f
            for c, kc in enumerate(_k3(m)):
                k[c] += f * kc
        for c in range(3):
            k[c] += rem * base[c]
        p = np.zeros((nx, ny))
        for rc, dens in L.power_rects:
            p += dens * rect_overlap_area(xe, ye, rc)
        for _ in range(n):
            kx_l.append(k[0]); ky_l.append(k[1]); kz_l.append(k[2])
            dz_l.append(L.height / n)
            P_l.append(p / n if L.is_source else np.zeros((nx, ny)))
        slices.append(slice(z0, z0 + n))
        z0 += n
    NZ = z0
    kx, ky, kz = (np.array(a) for a in (kx_l, ky_l, kz_l))
    dz = np.array(dz_l)[:, None, None]
    P = np.array(P_l)
    idx = np.arange(NZ * nx * ny).reshape(NZ, nx, ny)

    rows, cols, vals = [], [], []
    diag = np.zeros((NZ, nx, ny))

    def couple(a_idx, b_idx, G):
        rows.append(a_idx.ravel()); cols.append(b_idx.ravel()); vals.append(-G.ravel())
        rows.append(b_idx.ravel()); cols.append(a_idx.ravel()); vals.append(-G.ravel())

    # x neighbours: A = dy dz, two half cells in series
    G = (dy * dz) / (0.5 * dx / kx[:, :-1, :] + 0.5 * dx / kx[:, 1:, :])
    couple(idx[:, :-1, :], idx[:, 1:, :], G)
    diag[:, :-1, :] += G; diag[:, 1:, :] += G
    G = (dx * dz) / (0.5 * dy / ky[:, :, :-1] + 0.5 * dy / ky[:, :, 1:])
    couple(idx[:, :, :-1], idx[:, :, 1:], G)
    diag[:, :, :-1] += G; diag[:, :, 1:] += G
    hz_up = np.full(NZ, 0.5)       # fraction of the layer height in the node-to-face resistance
    if emulate_3dice_ends:
        if model.top is None:
            hz_up[0] = 1.0
        if model.bottom is None:
            hz_up[-1] = 1.0
    hz_dn = hz_up[:, None, None]
    G = cell_area[None] / (hz_dn[:-1] * dz[:-1] / kz[:-1] + hz_dn[1:] * dz[1:] / kz[1:])
    couple(idx[:-1], idx[1:], G)
    diag[:-1] += G; diag[1:] += G

    t_ref = (model.top or model.bottom)[1]
    b = P.copy()
    for face, sink in ((0, model.top), (NZ - 1, model.bottom)):
        if sink is None:
            continue
        htc, tamb = sink
        Ga = cell_area / (0.5 * dz[face] / kz[face] + 1.0 / htc)
        diag[face] += Ga
        b[face] += Ga * (tamb - t_ref)

    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(idx.size, idx.size))
    n_unk = idx.size
    meth = method
    if method == 'auto':
        try:
            import pyamg  # noqa: F401
            meth = 'amg'
        except ImportError:
            meth = 'direct' if n_unk <= 150_000 else 'cg'
    info = dict(method=meth, unknowns=int(n_unk), nz_total=int(NZ), r=r, nx=nx, ny=ny)
    bv = b.ravel()
    if meth == 'direct':
        x = spla.splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                      options=dict(SymmetricMode=True)).solve(bv)
    elif meth == 'amg':
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=500)
        res_hist: List[float] = []
        x = ml.solve(bv, tol=tol, maxiter=2000, accel='cg', residuals=res_hist)
        info['amg_iterations'] = len(res_hist) - 1
    else:
        # Jacobi-scaled CG (symmetric positive definite)
        dinv = 1.0 / A.diagonal()
        M = spla.LinearOperator(A.shape, matvec=lambda v: dinv * v)
        it = [0]
        x, flag = spla.cg(A, bv, rtol=tol, atol=0.0, maxiter=20000, M=M,
                          callback=lambda xk: it.__setitem__(0, it[0] + 1))
        info['cg_iterations'] = it[0]
        if flag != 0:
            raise RuntimeError(f"CG did not converge (flag {flag}, {it[0]} iterations)")
    res = np.linalg.norm(A @ x - bv) / max(np.linalg.norm(bv), 1e-300)
    info['relative_residual'] = float(res)
    if res > 1e-7:
        raise RuntimeError(f"linear solve did not converge (relative residual {res:.1e}, method {meth}); "
                           "try --method direct, or reduce --r / --nz")
    # global energy balance: power in vs heat out through the sinks
    out = 0.0
    for face, sink in ((0, model.top), (NZ - 1, model.bottom)):
        if sink is None:
            continue
        htc, tamb = sink
        Ga = cell_area / (0.5 * dz[face] / kz[face] + 1.0 / htc)
        out += float((Ga * (x.reshape(NZ, nx, ny)[face] - (tamb - t_ref))).sum())
    info['power_in'] = float(P.sum())
    info['heat_out'] = out
    T = x.reshape(NZ, nx, ny) + t_ref
    return RefSolution(r, T, slices, xe, ye, nx, ny, t_ref, info)
