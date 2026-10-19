"""Generators for the two self-generated v5 sets (plan_v5 T6 and A5). Own code, fixed seeds, frozen BEFORE generation.

  python -I tools/fieldlint/bench/gen_v5.py chip_fem_a|chip_fem_b|advdiff [--out D:/fieldlint_data_v5/generated]

chip_fem_a / chip_fem_b (T6, thermal, independent solver): Q1 (bilinear) finite elements on a 65 x 65 node grid of the unit square (64 x 64 elements),
  -div(k grad r) + g r = q  in (0,1)^2,  r = 0 on the boundary nodes (an isothermal heat sink at the die edge); stored temperature
  T = 300 + 50 r (kelvin, ambient 300). The stiffness matrix is assembled with the exact Q1 element matrix for a constant element conductivity
  (the mean of the four corner values of the nodal conductivity); the load is the lumped (nodal) power density q h^2. A 9-point operator for
  uniform k, NOT the 5-point stencil of fieldlint's F008: this is the "independent discretisation" part of the design. Nothing is tuned.
    chip_fem_a: 300 samples; k nodal field: background 1, 2-4 rectangles (random position, size 8-24 nodes) with log-uniform k in [0.2, 10];
                q: 1-5 non-negative rectangles (size 4-14 nodes) with amplitude U(0.5, 2); g = 0.
    chip_fem_b: 300 samples; k = 1 everywhere; q: 1-4 Gaussian hot spots (sigma U(0.03, 0.08) of the side) with amplitude U(0.5, 2) plus one
                rectangle; g = 5 (lumped vertical loss).
advdiff (A5, advective control): -eps Laplace(u) + v . grad(u) = s on (0,1)^2, u = 0 on the boundary, cell-centred finite volume 64 x 64 cells,
  first-order upwind convection, central diffusion, eps = 0.02, v = |v| (cos a, sin a) with |v| ~ U(1, 3), a ~ U(0, 2 pi) per sample;
  s: 1-4 Gaussian blobs (sigma U(0.05, 0.1), amplitude U(0.5, 2)). 300 samples. Stored: u, s on the 64 x 64 cells, vx, vy per sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

OUT = Path(r'D:\fieldlint_data_v5\generated')


def q1_solve_batch(k_nod, q_nod, g):
    """Q1 FEM on an (m+1) x (m+1) node grid, h = 1/m, Dirichlet 0 on the boundary. k_nod, q_nod: (N, m+1, m+1). Returns r: (N, m+1, m+1)."""
    n1 = k_nod.shape[-1]
    m = n1 - 1
    h = 1.0 / m
    Ke = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6.0       # unit-k Q1 stiffness (nodes: 00, 10, 11, 01)
    idx = np.arange(n1 * n1).reshape(n1, n1)
    corners = [idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]]                               # (i,j) -> (i+1,j) etc.
    interior = idx[1:-1, 1:-1].ravel()
    out = np.zeros_like(k_nod, dtype=float)
    for s in range(len(k_nod)):
        ke = 0.25 * (k_nod[s][:-1, :-1] + k_nod[s][1:, :-1] + k_nod[s][1:, 1:] + k_nod[s][:-1, 1:])         # element conductivity (m, m)
        rows, cols, vals = [], [], []
        for a in range(4):
            for b in range(4):
                rows.append(corners[a].ravel())
                cols.append(corners[b].ravel())
                vals.append((ke * Ke[a, b]).ravel())
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n1 * n1, n1 * n1)).tocsr()
        if g:
            A = A + g * h * h * sp.identity(n1 * n1)                                                  # lumped mass
        b = (q_nod[s] * h * h).ravel()
        Ai = A[interior][:, interior].tocsc()
        r = np.zeros(n1 * n1)
        r[interior] = spl.spsolve(Ai, b[interior])
        out[s] = r.reshape(n1, n1)
    return out


def rect_field(rng, n1, count, lo, hi, amp):
    f = np.zeros((n1, n1))
    for _ in range(count):
        h, w = rng.integers(lo, hi + 1, size=2)
        y, x = rng.integers(1, n1 - h - 1), rng.integers(1, n1 - w - 1)
        f[y:y + h, x:x + w] += amp(rng)
    return f


def chip_fem_a(seed=20261015, N=300, m=64):
    rng = np.random.default_rng(seed)
    n1 = m + 1
    K, Q = np.empty((N, n1, n1)), np.empty((N, n1, n1))
    for i in range(N):
        k = np.ones((n1, n1))
        for _ in range(rng.integers(2, 5)):
            h, w = rng.integers(8, 25, size=2)
            y, x = rng.integers(0, n1 - h), rng.integers(0, n1 - w)
            k[y:y + h, x:x + w] = np.exp(rng.uniform(np.log(0.2), np.log(10.0)))
        K[i] = k
        Q[i] = rect_field(rng, n1, int(rng.integers(1, 6)), 4, 14, lambda r: r.uniform(0.5, 2.0))
    R = q1_solve_batch(K, Q, 0.0)
    return {'T': 300.0 + 50.0 * R, 'power': Q, 'k': K}


def chip_fem_b(seed=20261016, N=300, m=64):
    rng = np.random.default_rng(seed)
    n1 = m + 1
    yy, xx = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n1), indexing='ij')
    K, Q = np.ones((N, n1, n1)), np.empty((N, n1, n1))
    for i in range(N):
        q = rect_field(rng, n1, 1, 4, 14, lambda r: r.uniform(0.5, 2.0))
        for _ in range(rng.integers(1, 5)):
            cy, cx = rng.uniform(0.1, 0.9, size=2)
            sg = rng.uniform(0.03, 0.08)
            q += rng.uniform(0.5, 2.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg ** 2))
        Q[i] = q
    R = q1_solve_batch(K, Q, 5.0)
    return {'T': 300.0 + 50.0 * R, 'power': Q, 'k': K}


def advdiff(seed=20261017, N=300, n=64, eps=0.02):
    rng = np.random.default_rng(seed)
    h = 1.0 / n
    c = (np.arange(n) + 0.5) * h
    yy, xx = np.meshgrid(c, c, indexing='ij')                                                      # array[i, j]: i = y index, j = x index
    idx = np.arange(n * n).reshape(n, n)
    U, S, VX, VY = np.empty((N, n, n)), np.empty((N, n, n)), np.empty(N), np.empty(N)
    for s in range(N):
        mag, ang = rng.uniform(1.0, 3.0), rng.uniform(0, 2 * np.pi)
        vx, vy = mag * np.cos(ang), mag * np.sin(ang)
        src = np.zeros((n, n))
        for _ in range(rng.integers(1, 5)):
            cy, cx = rng.uniform(0.15, 0.85, size=2)
            sg = rng.uniform(0.05, 0.1)
            src += rng.uniform(0.5, 2.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg ** 2))
        rows, cols, vals = [], [], []
        diag = np.zeros((n, n))
        for (di, dj, vel) in ((0, 1, vx), (0, -1, -vx), (1, 0, vy), (-1, 0, -vy)):                   # neighbour in the +/- x / y direction; vel = outward face velocity
            ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            ni, nj = ii + di, jj + dj
            inside = (ni >= 0) & (ni < n) & (nj >= 0) & (nj < n)
            outflow = max(vel, 0.0) * h                                                             # upwind: outward flux uses the cell's own value
            inflow = min(vel, 0.0) * h                                                              # inward flux uses the neighbour's value (vel < 0 -> coefficient on neighbour)
            diag += (eps + outflow)                                                                 # diffusion (Dirichlet 0 outside uses h/2: factor 2 below) + outflow
            diag[~inside] += eps                                                                    # boundary face: half distance -> twice the diffusive conductance
            rows.append(idx[inside]); cols.append((ni * n + nj)[inside])
            vals.append(np.full(inside.sum(), -eps + inflow))
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)).tocsr() + sp.diags(diag.ravel())
        U[s] = spl.spsolve(A.tocsc(), (src * h * h).ravel()).reshape(n, n)
        S[s], VX[s], VY[s] = src, vx, vy
    return {'u': U, 'source': S, 'vx': VX, 'vy': VY}


GEN = {'chip_fem_a': chip_fem_a, 'chip_fem_b': chip_fem_b, 'advdiff': advdiff}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('name', choices=list(GEN))
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{a.name}.npz'
    if path.exists():
        sys.exit(f'refusing: {path} exists')
    arrays = GEN[a.name]()
    np.savez(path, **arrays)
    man = {'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'shapes': {k: list(v.shape) for k, v in arrays.items()}}
    (out / f'{a.name}.MANIFEST.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
