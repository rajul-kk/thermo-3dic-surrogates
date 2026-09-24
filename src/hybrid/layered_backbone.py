"""Exact layered-stack thermal backbone: lateral cosine modes x a tridiagonal solve in z.
Solves the FV conduction system with each z-cell's conductivity replaced by its lateral mean.
"""
# With layer-uniform conductivity, the FV system (src/validation/fv_solver.py: adiabatic sides
# and top, convection at z = 0) is diagonalised exactly by a 2D DCT-II on the uniform lateral
# grid, because DCT-II is the eigenbasis of the Neumann Laplacian. Each lateral mode then
# decouples into a tridiagonal system in z, the discrete form of the layered transfer-matrix
# solution. Exact for laterally uniform stacks; lateral heterogeneity (chiplet vs underfill,
# TSV fields) is what a learned correction has to supply (docs/report.md 9.25).
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.fft import dctn, idctn

from src.validation import fv_solver as fv


def _thomas(lower, diag, upper, rhs):
    """Batched tridiagonal solve along axis 0; lower[0] and upper[-1] are ignored."""
    n = diag.shape[0]
    c, d = np.empty_like(diag), np.empty_like(rhs)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for k in range(1, n):
        m = diag[k] - lower[k] * c[k - 1]
        c[k] = upper[k] / m if k < n - 1 else 0.0
        d[k] = (rhs[k] - lower[k] * d[k - 1]) / m
    x = np.empty_like(rhs)
    x[-1] = d[-1]
    for k in range(n - 2, -1, -1):
        x[k] = d[k] - c[k] * x[k + 1]
    return x


def solve(geometry, scenario: Dict[str, Any], g: Optional[fv.FVGrid] = None,
          k_lat=None, k_ver=None) -> Dict[str, Any]:
    """Temperature (K) on grid `g` for the layer-averaged stack; conductivities may be passed in."""
    g = g or fv.make_grid(geometry)
    if k_lat is None or k_ver is None:
        k_lat, k_ver = fv.conductivity(geometry, scenario, g)
    q = fv.power(geometry, scenario, g)                     # W per cell
    htc = float(scenario.get('htc', 10000.0))
    t_amb = float(scenario.get('t_ambient', 25.0)) + 273.15
    nx, ny, nz = g.shape
    dx = float(np.diff(g.xe)[0]) * fv.UM
    dy = float(np.diff(g.ye)[0]) * fv.UM
    dz = np.diff(g.ze) * fv.UM
    area = dx * dy

    kl = k_lat.mean(axis=(0, 1))                            # (nz,) lateral conductivity per z-cell
    kv = k_ver.mean(axis=(0, 1))                            # vertical: columns act in parallel

    # Neumann-Laplacian eigenvalues of the unit-spacing 5-point stencil, per lateral mode.
    ex = (2.0 * np.sin(np.pi * np.arange(nx) / (2 * nx))) ** 2
    ey = (2.0 * np.sin(np.pi * np.arange(ny) / (2 * ny))) ** 2
    lat = (kl * dz)[:, None, None] * ((dy / dx) * ex[None, :, None] + (dx / dy) * ey[None, None, :])

    gz = area / (dz[:-1] / (2 * kv[:-1]) + dz[1:] / (2 * kv[1:]))      # (nz-1,) W/K between cells
    gb = area / (dz[0] / (2 * kv[0]) + 1.0 / htc)                      # to ambient at z = 0

    diag = lat.copy()
    diag[:-1] += gz[:, None, None]
    diag[1:] += gz[:, None, None]
    diag[0] += gb
    lower = np.zeros_like(diag); lower[1:] = -gz[:, None, None]
    upper = np.zeros_like(diag); upper[:-1] = -gz[:, None, None]

    qhat = np.moveaxis(dctn(q, type=2, axes=(0, 1), norm='ortho'), 2, 0)   # (nz, nx, ny)
    theta_hat = _thomas(lower, diag, upper, qhat)
    theta = idctn(np.moveaxis(theta_hat, 0, 2), type=2, axes=(0, 1), norm='ortho')
    return {'T': theta + t_amb, 'grid': g}


def sample(T: np.ndarray, g: fv.FVGrid, coords: np.ndarray) -> np.ndarray:
    """Read a grid field at a file's (x, y, z) nodes (nearest cell centre)."""
    xc = 0.5 * (g.xe[:-1] + g.xe[1:]); yc = 0.5 * (g.ye[:-1] + g.ye[1:]); zc = 0.5 * (g.ze[:-1] + g.ze[1:])
    i = np.abs(coords[:, 0:1] - xc[None]).argmin(1)
    j = np.abs(coords[:, 1:2] - yc[None]).argmin(1)
    k = np.abs(coords[:, 2:3] - zc[None]).argmin(1)
    return T[i, j, k]
