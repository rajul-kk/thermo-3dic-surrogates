"""Three-tier thermal solve for interactive floorplanning (docs/report.md §9.25, §9.30, §9.32).

  preview   layered DCT x tridiagonal solve with laterally averaged conductivity: ~10-25 ms, no training
  exact     conjugate gradients on the full finite-volume system, preconditioned by the preview solver
  sign-off  a real 3D-ICE run (app/job_queue.py), not in this module

Placements are always snapped to the 3D-ICE cell grid first: a die edge that falls mid-cell makes a heated
insulator cell in 3D-ICE and in src/validation/fv_solver.py alike (the die-edge artefact, §9.32).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import dctn, idctn

from src.core.placement import _overlaps, associate_blocks_to_dies, grid_steps, lateral_chiplets, place_chiplets
from src.hybrid.layered_backbone import _thomas
from src.validation import fv_solver as fv

Offset = Tuple[float, float]
K0 = 273.15


class PlacementError(ValueError):
    """The requested placement is out of bounds, overlapping, or cannot be snapped to the grid."""


@dataclass
class Movable:
    """One thing the user can drag: a chiplet (all its layers and blocks) or, on block-only stacks, a power block."""
    id: str
    x: float
    y: float
    width: float
    height: float
    names: List[str]                       # placement keys that move together
    layers: List[str]
    blocks: List[str]


@dataclass
class Floorplan:
    geometry: Any
    movables: List[Movable] = field(default_factory=list)

    def by_id(self) -> Dict[str, Movable]:
        return {m.id: m for m in self.movables}


def describe(geometry) -> Floorplan:
    """The draggable items of a geometry at its nominal placement."""
    prints = {d.name: d for d in (getattr(geometry, 'die_footprints', None) or [])}
    groups = associate_blocks_to_dies(geometry)
    out = []
    if prints:
        for (x, y, w, h), names in lateral_chiplets(geometry):
            out.append(Movable(names[0], x, y, w, h, list(names), sorted({prints[n].die_layer_name for n in names}),
                               sorted(b for n in names for b in groups.get(n, []))))
    else:
        for b in geometry.power_blocks:
            out.append(Movable(b.name, b.x, b.y, b.width, b.height, [b.name], [b.layer_name], [b.name]))
    return Floorplan(geometry, out)


def _snap(v: float, step: float) -> float:
    return float(np.round(v / step) * step)


def place(geometry, offsets: Dict[str, Offset]):
    """(placed geometry, snapped offsets per movable id). Raises PlacementError with a readable reason."""
    fp = describe(geometry)
    ids = fp.by_id()
    unknown = set(offsets) - set(ids)
    if unknown:
        raise PlacementError(f'unknown item(s) {sorted(unknown)}; movable items are {sorted(ids)}')
    gx, gy = grid_steps(geometry)
    W, H = geometry.die_width, geometry.die_length
    out_ids: Dict[str, Offset] = {}
    boxes = []
    for m in fp.movables:
        dx, dy = offsets.get(m.id, (0.0, 0.0))
        x, y = m.x + float(dx), m.y + float(dy)
        # more than one cell outside is a real error; within a cell, snapping pulls the item back inside
        if x < -gx or y < -gy or x + m.width > W + gx or y + m.height > H + gy:
            raise PlacementError(f'{m.id} leaves the package ({W / 1000:.1f} x {H / 1000:.1f} mm)')
        x = min(max(_snap(x, gx), 0.0), np.floor((W - m.width) / gx + 1e-9) * gx)
        y = min(max(_snap(y, gy), 0.0), np.floor((H - m.height) / gy + 1e-9) * gy)
        out_ids[m.id] = (float(x - m.x), float(y - m.y))
        boxes.append(Movable(m.id, x, y, m.width, m.height, m.names, m.layers, m.blocks))
    snapped = {n: out_ids[m.id] for m in fp.movables for n in m.names}
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if set(a.layers) & set(b.layers) and _overlaps(a, b, 0.0):
                raise PlacementError(f'{a.id} overlaps {b.id}')
    try:
        placed = place_chiplets(geometry, {k: v for k, v in snapped.items()})
    except ValueError as exc:
        raise PlacementError(str(exc)) from exc
    return placed, out_ids


def backbone_preconditioner(g: fv.FVGrid, k_lat, k_ver, htc: float):
    """r -> M r, the exact solve of the FV system with each z-cell's conductivity replaced by its lateral mean."""
    nx, ny, nz = g.shape
    dx = float(np.diff(g.xe)[0]) * fv.UM
    dy = float(np.diff(g.ye)[0]) * fv.UM
    dz = np.diff(g.ze) * fv.UM
    area = dx * dy
    kl, kv = k_lat.mean((0, 1)), k_ver.mean((0, 1))
    ex = (2 * np.sin(np.pi * np.arange(nx) / (2 * nx))) ** 2
    ey = (2 * np.sin(np.pi * np.arange(ny) / (2 * ny))) ** 2
    lat = (kl * dz)[:, None, None] * ((dy / dx) * ex[None, :, None] + (dx / dy) * ey[None, None, :])
    gz = area / (dz[:-1] / (2 * kv[:-1]) + dz[1:] / (2 * kv[1:]))
    gb = area / (dz[0] / (2 * kv[0]) + 1.0 / htc)
    diag = lat.copy()
    diag[:-1] += gz[:, None, None]
    diag[1:] += gz[:, None, None]
    diag[0] += gb
    lo = np.zeros_like(diag); lo[1:] = -gz[:, None, None]
    up = np.zeros_like(diag); up[:-1] = -gz[:, None, None]

    def apply(r):
        rh = np.moveaxis(dctn(r.reshape(g.shape), type=2, axes=(0, 1), norm='ortho'), 2, 0)
        th = _thomas(lo, diag, up, rh)
        return idctn(np.moveaxis(th, 0, 2), type=2, axes=(0, 1), norm='ortho').ravel()
    return apply


def _preview_risk(geometry, g, k_lat) -> Optional[str]:
    """The preview averages conductivity laterally. That fails when a strongly heterogeneous layer lies in the heat
    path between the sources and the sink (geometry7's bridge layer, §9.30); name that layer if there is one."""
    active = [i for i, l in enumerate(geometry.layers) if l.is_active]
    if not active:
        return None
    for li in range(min(active)):
        k = k_lat[:, :, g.z_layer == li]
        if k.size and k.max() / k.min() > 10.0:
            return geometry.layers[li].name
    return None


def solve(geometry, scenario: Dict[str, Any], mode: str = 'preview', tol: float = 1e-8, max_iter: int = 300):
    """Temperature rise above ambient on the FV grid. `geometry` must already be placed (see `place`)."""
    if mode not in ('preview', 'exact'):
        raise ValueError(f"mode must be 'preview' or 'exact', got {mode!r}")
    t0 = time.perf_counter()
    g = fv.make_grid(geometry)
    k_lat, k_ver = fv.conductivity(geometry, scenario, g)
    htc = float(scenario.get('htc', 10000.0))
    b = fv.power(geometry, scenario, g).ravel()
    M = backbone_preconditioner(g, k_lat, k_ver, htc)
    x = M(b)
    iters, resid = 0, None
    if mode == 'exact' and b.any():
        A, _ = fv.assemble(g, k_lat, k_ver, htc)
        r = b - A @ x
        z = M(r); p = z.copy(); rz = r @ z
        bn = np.linalg.norm(b)
        resid = float(np.linalg.norm(r) / bn)
        while resid > tol and iters < max_iter:
            Ap = A @ p
            a = rz / (p @ Ap)
            x += a * p
            r -= a * Ap
            z = M(r)
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
            iters += 1
            resid = float(np.linalg.norm(r) / bn)
        if resid > tol:
            raise RuntimeError(f'PCG did not reach {tol:g} in {max_iter} iterations (residual {resid:.2e})')
    theta = x.reshape(g.shape)
    return {'theta': theta, 'grid': g, 'mode': mode, 'iterations': iters, 'residual': resid,
            'power_w': float(b.sum()), 'seconds': time.perf_counter() - t0,
            'preview_risk_layer': _preview_risk(geometry, g, k_lat)}


def summarise(geometry, sol, t_ambient_c: float, layer: Optional[str] = None) -> Dict[str, Any]:
    """Peak, hotspot and one layer's 2D map (deg C; rows = x cells along die_width, columns = y cells)."""
    theta, g = sol['theta'], sol['grid']
    xc = 0.5 * (g.xe[:-1] + g.xe[1:])
    yc = 0.5 * (g.ye[:-1] + g.ye[1:])
    i, j, k = np.unravel_index(int(theta.argmax()), theta.shape)
    hot_layer = geometry.layers[int(g.z_layer[k])].name
    layers = []
    for li, l in enumerate(geometry.layers):
        sel = g.z_layer == li
        layers.append({'name': l.name, 'active': bool(l.is_active), 'thickness_um': float(l.thickness),
                       'k': float(l.k_thermal), 'peak_c': float(theta[:, :, sel].max() + t_ambient_c)})
    name = layer if layer and layer != 'hottest' else hot_layer
    idx = [l.name for l in geometry.layers].index(name)
    plane = theta[:, :, g.z_layer == idx].max(axis=2) + t_ambient_c
    return {'peak_c': float(theta.max() + t_ambient_c), 'rise_k': float(theta.max()),
            'hotspot': {'x_um': float(xc[i]), 'y_um': float(yc[j]), 'layer': hot_layer},
            'layer': name, 'layers': layers,
            'field': np.round(plane, 2).tolist(), 'field_min_c': float(plane.min()), 'field_max_c': float(plane.max()),
            'nx': int(plane.shape[0]), 'ny': int(plane.shape[1])}


def optimise(geometry, scenario: Dict[str, Any], offsets: Optional[Dict[str, Offset]] = None, n_evals: int = 200,
             seed: int = 0, max_step_cells: int = 12) -> Dict[str, Any]:
    """Lower the peak temperature by moving chiplets: stochastic hill-climb scored by the preview solver.
    Moves: shift one item by a few cells, or swap the positions of two items."""
    rng = np.random.default_rng(seed)
    fp = describe(geometry)
    gx, gy = grid_steps(geometry)
    t0 = time.perf_counter()

    def score(offs):
        placed, snapped = place(geometry, offs)
        return float(solve(placed, scenario, 'preview')['theta'].max()), snapped

    best_peak, best = score(dict(offsets or {}))
    start_peak, evals, accepted = best_peak, 1, 0
    ids = [m.id for m in fp.movables]
    nominal = fp.by_id()
    while evals < n_evals and len(ids) > 0:
        cand = dict(best)
        if len(ids) > 1 and rng.random() < 0.3:
            a, b = rng.choice(ids, 2, replace=False)
            ma, mb = nominal[a], nominal[b]
            pa = (ma.x + best[a][0], ma.y + best[a][1])
            pb = (mb.x + best[b][0], mb.y + best[b][1])
            cand[a] = (pb[0] - ma.x, pb[1] - ma.y)
            cand[b] = (pa[0] - mb.x, pa[1] - mb.y)
        else:
            a = rng.choice(ids)
            sx, sy = rng.integers(-max_step_cells, max_step_cells + 1, 2)
            cand[a] = (best[a][0] + sx * gx, best[a][1] + sy * gy)
        try:
            peak, snapped = score(cand)
        except PlacementError:
            evals += 1
            continue
        evals += 1
        if peak < best_peak - 1e-6:
            best_peak, best, accepted = peak, snapped, accepted + 1
    return {'offsets': best, 'rise_before_k': start_peak, 'rise_after_k': best_peak, 'evaluations': evals,
            'accepted': accepted, 'seconds': time.perf_counter() - t0}
