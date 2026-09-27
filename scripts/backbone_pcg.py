"""Classical baseline for every learned correction (report §9.30): CG on the exact FV system, preconditioned by
the layered DCT x tridiagonal backbone. No training. Prints peak/RMS error vs the direct solve per iteration.
Usage: python scripts/backbone_pcg.py [--geometries geometry4 ...] [--data-dir "data/3d-ice-layout-{g}"] [--n 3] [--iters 30]
"""
import sys, time, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from scipy.fft import dctn, idctn
from src.hybrid import layered_backbone as lb
from src.validation import fv_solver as fv
from scripts.backbone_vs_fv import scenario

def backbone_op(g, kl3, kv3, htc):
    nx, ny, nz = g.shape
    dx = float(np.diff(g.xe)[0]) * fv.UM; dy = float(np.diff(g.ye)[0]) * fv.UM; dz = np.diff(g.ze) * fv.UM
    area = dx * dy; kl = kl3.mean((0, 1)); kv = kv3.mean((0, 1))
    ex = (2 * np.sin(np.pi * np.arange(nx) / (2 * nx))) ** 2; ey = (2 * np.sin(np.pi * np.arange(ny) / (2 * ny))) ** 2
    lat = (kl * dz)[:, None, None] * ((dy / dx) * ex[None, :, None] + (dx / dy) * ey[None, None, :])
    gz = area / (dz[:-1] / (2 * kv[:-1]) + dz[1:] / (2 * kv[1:])); gb = area / (dz[0] / (2 * kv[0]) + 1.0 / htc)
    diag = lat.copy(); diag[:-1] += gz[:, None, None]; diag[1:] += gz[:, None, None]; diag[0] += gb
    lo = np.zeros_like(diag); lo[1:] = -gz[:, None, None]; up = np.zeros_like(diag); up[:-1] = -gz[:, None, None]
    def apply(r):
        rh = np.moveaxis(dctn(r.reshape(g.shape), type=2, axes=(0, 1), norm='ortho'), 2, 0)
        th = lb._thomas(lo, diag, up, rh)
        return idctn(np.moveaxis(th, 0, 2), type=2, axes=(0, 1), norm='ortho').ravel()
    return apply

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--geometries', nargs='*', default=['geometry4', 'geometry5', 'geometry6'])
_ap.add_argument('--data-dir', default='data/3d-ice-layout-{g}')
_ap.add_argument('--n', type=int, default=3)
_ap.add_argument('--iters', type=int, default=30)
ARGS = _ap.parse_args()
CHECK = sorted({1, 2, 3, 5, 10, 20, 30, 50, 100, 200, ARGS.iters} - {i for i in range(ARGS.iters + 1, 10 ** 6)})

for geom_name in ARGS.geometries:
    files = sorted(Path(ARGS.data_dir.format(g=geom_name)).rglob(f'{geom_name}_*.npz'))[:ARGS.n]
    for f in files:
        geom, scen, c, y = scenario(f, geom_name)
        g = fv.make_grid(geom); kl, kv = fv.conductivity(geom, scen, g)
        A, _ = fv.assemble(g, kl, kv, scen['htc']); b = fv.power(geom, scen, g).ravel()
        t = time.time(); exact = fv.solve(geom, scen, g)['T'].ravel() - (scen['t_ambient'] + 273.15); t_full = time.time() - t
        M = backbone_op(g, kl, kv, scen['htc'])
        t0 = time.time(); x = np.zeros_like(b); r = b.copy(); z = M(r); p = z.copy(); rz = r @ z
        hist = []
        for it in range(1, ARGS.iters + 1):
            Ap = A @ p; a = rz / (p @ Ap); x += a * p; r -= a * Ap
            if it in CHECK:
                hist.append((it, abs(x.max() - exact.max()), np.sqrt(np.mean((x - exact) ** 2)), time.time() - t0))
            z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
        bb = M(b)
        print(f'{geom_name} {f.stem[-6:]} cells {b.size:,} | full solve {t_full:.2f}s | backbone |peak| {abs(bb.max()-exact.max()):.2f} K', flush=True)
        print('   ' + '  '.join(f'it{it}: |peak| {pk:.3f} K rms {rm:.3f} ({tt:.2f}s)' for it, pk, rm, tt in hist), flush=True)
