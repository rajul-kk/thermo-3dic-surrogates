"""Independent steady-state finite-volume conduction solver, used to check 3D-ICE's ground truth.

Reads the same Geometry + scenario dict the ICE wrapper turns into a .stk file, but shares none of
3D-ICE's code: its own grid, material mapping, power deposition and sparse solve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

UM = 1e-6
ACTIVE_SPLIT_UM = 100.0     # same rule as ICESimulator.ACTIVE_SPLIT_UM: power in the middle third


def source_cells(geometry, g) -> np.ndarray:
    """z-cells that carry an active layer's power (its middle third when the layer is split)."""
    zc = 0.5 * (g.ze[:-1] + g.ze[1:])
    out = np.zeros(len(zc), bool)
    for i, layer in enumerate(geometry.layers):
        if not layer.is_active:
            continue
        lo, hi = layer.z_bottom, layer.z_top
        if layer.thickness > ACTIVE_SPLIT_UM:
            lo, hi = lo + layer.thickness / 3, hi - layer.thickness / 3
        out |= (g.z_layer == i) & (zc > lo) & (zc < hi)
    return out


@dataclass
class FVGrid:
    xe: np.ndarray            # cell edges along die_width (µm)
    ye: np.ndarray            # cell edges along die_length (µm)
    ze: np.ndarray            # cell edges in z (µm), bottom = heat-sink face
    z_layer: np.ndarray       # geometry layer index of each z-cell

    @property
    def shape(self):
        return len(self.xe) - 1, len(self.ye) - 1, len(self.ze) - 1


def z_edges(geometry, max_sub_um: float = 1200.0, max_per_layer: int = 12,
            active_sub: int = 1, min_per_layer: int = 1, mult: int = 1):
    """Split each layer into sub-cells; the defaults reproduce the ICE wrapper's stack elements."""
    edges, owner = [geometry.layers[0].z_bottom], []
    for i, layer in enumerate(geometry.layers):
        if layer.is_active:
            n = active_sub * (3 if layer.thickness > ACTIVE_SPLIT_UM else 1)
        else:
            n = int(np.ceil(layer.thickness / max_sub_um))
            n = max(min_per_layer, min(n, max_per_layer))
        n *= mult                       # mult > 1 nests inside the mult = 1 grid
        for j in range(1, n + 1):
            edges.append(layer.z_bottom + layer.thickness * j / n)
            owner.append(i)
    return np.array(edges), np.array(owner)


def make_grid(geometry, lateral_refine: int = 1, **zkw) -> FVGrid:
    """Lateral cells match 3D-ICE's (mesh_resolution[1] along width, [0] along length) times refine."""
    n_len, n_wid, _ = geometry.mesh_resolution
    xe = np.linspace(0.0, geometry.die_width, n_wid * lateral_refine + 1)
    ye = np.linspace(0.0, geometry.die_length, n_len * lateral_refine + 1)
    ze, owner = z_edges(geometry, **zkw)
    return FVGrid(xe, ye, ze, owner)


def _centre_mask(g: FVGrid, x0, y0, w, h):
    xc = 0.5 * (g.xe[:-1] + g.xe[1:])
    yc = 0.5 * (g.ye[:-1] + g.ye[1:])
    mx = (xc >= x0) & (xc < x0 + w)
    my = (yc >= y0) & (yc < y0 + h)
    return np.outer(mx, my)


def _overlap(g: FVGrid, x0, y0, w, h):
    """Fraction of a rectangle's area falling in each lateral cell."""
    ox = np.clip(np.minimum(g.xe[1:], x0 + w) - np.maximum(g.xe[:-1], x0), 0, None)
    oy = np.clip(np.minimum(g.ye[1:], y0 + h) - np.maximum(g.ye[:-1], y0), 0, None)
    return np.outer(ox, oy) / (w * h)


def conductivity(geometry, scenario: Dict[str, Any], g: FVGrid):
    """(k_lateral, k_vertical) per cell in W/m·K, following the scenario's overrides, footprints and TSV maps."""
    from src.core.material import MaterialLibrary
    from src.scenario.tsv_maps import quantise_materials, tsv_effective_k

    nx, ny, nz = g.shape
    over = scenario.get('layer_k_overrides') or {}
    tsv_maps = scenario.get('tsv_map_by_layer') or {}
    fps: Dict[str, list] = {}
    for fp in geometry.die_footprints:
        fps.setdefault(fp.die_layer_name, []).append(fp)

    k_lat = np.empty((nx, ny, nz))
    k_ver = np.empty((nx, ny, nz))
    for li, layer in enumerate(geometry.layers):
        k_own = float(over.get(layer.name, layer.k_thermal))
        kl = np.full((nx, ny), k_own)
        kv = kl.copy()
        if layer.name in tsv_maps:
            idx, centres = quantise_materials(np.asarray(tsv_maps[layer.name], float), 12)
            n_l, n_w = idx.shape
            xc = 0.5 * (g.xe[:-1] + g.xe[1:])
            yc = 0.5 * (g.ye[:-1] + g.ye[1:])
            b = np.minimum((xc / (geometry.die_width / n_w)).astype(int), n_w - 1)
            a = np.minimum((yc / (geometry.die_length / n_l)).astype(int), n_l - 1)
            lvl = idx[a[None, :], b[:, None]]
            k_l, k_v = tsv_effective_k(centres)
            kl, kv = k_l[lvl], k_v[lvl]
            if layer.name in fps:              # TSV map clipped to the chiplet footprints
                inside = np.zeros((nx, ny), bool)
                for fp in fps[layer.name]:
                    inside |= _centre_mask(g, fp.x, fp.y, fp.width, fp.height)
                k_gap = (over.get(f'{layer.name}__gap', MaterialLibrary.get(layer.gap_material).k_thermal)
                         if layer.gap_material else geometry.underfill_k)
                kl = np.where(inside, kl, k_gap); kv = np.where(inside, kv, k_gap)
        elif layer.name in fps:
            if layer.gap_material:
                k_gap = over.get(f'{layer.name}__gap', MaterialLibrary.get(layer.gap_material).k_thermal)
            else:
                k_gap = geometry.underfill_k
            kl = np.full((nx, ny), float(k_gap))
            for fp in fps[layer.name]:
                kl[_centre_mask(g, fp.x, fp.y, fp.width, fp.height)] = k_own
            kv = kl.copy()
        sel = g.z_layer == li
        k_lat[:, :, sel] = kl[:, :, None]
        k_ver[:, :, sel] = kv[:, :, None]
    return k_lat, k_ver


def power(geometry, scenario: Dict[str, Any], g: FVGrid) -> np.ndarray:
    """Watts per cell, spread over each active layer's z-cells by thickness and laterally by overlap area."""
    nx, ny, nz = g.shape
    q = np.zeros((nx, ny, nz))
    blocks = scenario.get('power_blocks', {})
    pmaps = scenario.get('power_map_by_layer') or {}
    dz = np.diff(g.ze)
    src = source_cells(geometry, g)
    for li, layer in enumerate(geometry.layers):
        if not layer.is_active:
            continue
        lat = np.zeros((nx, ny))
        if layer.name in pmaps:
            pm = np.asarray(pmaps[layer.name], float)
            n_l, n_w = pm.shape
            tl, tw = geometry.die_length / n_l, geometry.die_width / n_w
            for a, b in zip(*np.nonzero(pm > 0)):
                lat += pm[a, b] * _overlap(g, b * tw, a * tl, tw, tl)
        else:
            for blk in geometry.power_blocks:
                if blk.layer_name == layer.name and not blk.is_tsv_region:
                    lat += blk.power_watts(blocks.get(blk.name, 0.0)) * _overlap(
                        g, blk.x, blk.y, blk.width, blk.height)
        sel = np.nonzero((g.z_layer == li) & src)[0]
        frac = dz[sel] / dz[sel].sum()
        q[:, :, sel] += lat[:, :, None] * frac[None, None, :]
    return q


def assemble(g: FVGrid, k_lat, k_ver, htc: float):
    """Conductance matrix (W/K) for adiabatic sides/top and convection to ambient at z = 0."""
    nx, ny, nz = g.shape
    dx, dy, dz = np.diff(g.xe) * UM, np.diff(g.ye) * UM, np.diff(g.ze) * UM
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    rows, cols, vals = [], [], []

    def link(a, b, G):
        rows.extend([a.ravel(), b.ravel()])
        cols.extend([b.ravel(), a.ravel()])
        vals.extend([-G.ravel(), -G.ravel()])
        return G

    diag = np.zeros((nx, ny, nz))
    # x faces
    r = dx[:-1, None, None] / (2 * k_lat[:-1]) + dx[1:, None, None] / (2 * k_lat[1:])
    G = link(idx[:-1], idx[1:], (dy[None, :, None] * dz[None, None, :]) / r)
    diag[:-1] += G; diag[1:] += G
    # y faces
    r = dy[None, :-1, None] / (2 * k_lat[:, :-1]) + dy[None, 1:, None] / (2 * k_lat[:, 1:])
    G = link(idx[:, :-1], idx[:, 1:], (dx[:, None, None] * dz[None, None, :]) / r)
    diag[:, :-1] += G; diag[:, 1:] += G
    # z faces
    r = dz[None, None, :-1] / (2 * k_ver[:, :, :-1]) + dz[None, None, 1:] / (2 * k_ver[:, :, 1:])
    G = link(idx[:, :, :-1], idx[:, :, 1:], (dx[:, None, None] * dy[None, :, None]) / r)
    diag[:, :, :-1] += G; diag[:, :, 1:] += G
    # bottom convection
    g_bot = (dx[:, None] * dy[None, :]) / (dz[0] / (2 * k_ver[:, :, 0]) + 1.0 / htc)
    diag[:, :, 0] += g_bot

    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(idx.size, idx.size))
    return A, g_bot


def solve(geometry, scenario: Dict[str, Any], g: Optional[FVGrid] = None,
          tol: float = 1e-11) -> Dict[str, Any]:
    """Temperature (K) on grid `g`, plus an energy-balance residual."""
    g = g or make_grid(geometry)
    k_lat, k_ver = conductivity(geometry, scenario, g)
    q = power(geometry, scenario, g)
    htc = float(scenario.get('htc', 10000.0))
    t_amb = float(scenario.get('t_ambient', 25.0)) + 273.15
    A, g_bot = assemble(g, k_lat, k_ver, htc)
    b = q.ravel()
    if A.shape[0] <= 60_000:
        from scipy.sparse.linalg import spsolve
        theta = spsolve(A.tocsc(), b)
    else:
        import pyamg
        from scipy.sparse.linalg import cg
        ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric', max_coarse=2000)
        theta, info = cg(A, b, rtol=tol, maxiter=2000, M=ml.aspreconditioner(cycle='V'))
        if info != 0:
            raise RuntimeError(f'CG did not converge (info={info})')
    theta = theta.reshape(g.shape)
    out_w = float((g_bot * theta[:, :, 0]).sum())
    return {'T': theta + t_amb, 'grid': g, 'P_in': float(q.sum()), 'P_out': out_w}


def coarsen_to(T: np.ndarray, fine: FVGrid, coarse: FVGrid) -> np.ndarray:
    """Volume-average a fine solution onto a coarser grid whose edges it nests."""
    def weights(fe, ce):
        W = np.zeros((len(ce) - 1, len(fe) - 1))
        for i in range(len(ce) - 1):
            lo, hi = ce[i], ce[i + 1]
            ov = np.clip(np.minimum(fe[1:], hi) - np.maximum(fe[:-1], lo), 0, None)
            W[i] = ov / ov.sum()
        return W
    Wx, Wy, Wz = weights(fine.xe, coarse.xe), weights(fine.ye, coarse.ye), weights(fine.ze, coarse.ze)
    return np.einsum('ai,bj,ck,ijk->abc', Wx, Wy, Wz, T, optimize=True)


def ice_to_grid(coords: np.ndarray, temp: np.ndarray, g: FVGrid) -> np.ndarray:
    """Place 3D-ICE's (x, y, z) node output onto an FVGrid with the same cells."""
    nx, ny, nz = g.shape
    xc = 0.5 * (g.xe[:-1] + g.xe[1:])
    yc = 0.5 * (g.ye[:-1] + g.ye[1:])
    zc = 0.5 * (g.ze[:-1] + g.ze[1:])
    i = np.abs(coords[:, 0:1] - xc[None]).argmin(1)
    j = np.abs(coords[:, 1:2] - yc[None]).argmin(1)
    k = np.abs(coords[:, 2:3] - zc[None]).argmin(1)
    out = np.full((nx, ny, nz), np.nan)
    out[i, j, k] = temp
    return out
