"""PDEBench loaders for the linearity audit: canonical operator-learning benchmarks as (X, Y) pairs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

log = logging.getLogger('pde_data')

CACHE = Path('data/pde-bench')

# PDEBench (Takamoto et al., NeurIPS 2022 Datasets & Benchmarks), DaRUS doi:10.18419/darus-2986.
# Files are 1.3-10 GB each, far more than this audit needs, so we read only the leading N
# samples over HTTP range requests rather than downloading them. Every array in these files is
# contiguous (chunks=None for Darcy/Burgers), so a leading slice is a single byte range.
FILES = {
    'darcy_beta1.0':  '133219',   # 2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5   1.3 GB
    'darcy_beta0.01': '133217',   # 2D/DarcyFlow/2D_DarcyFlow_beta0.01_Train.hdf5  1.3 GB
    'burgers_nu0.01': '281363',   # 1D/Burgers/Train/1D_Burgers_Sols_Nu0.01.hdf5   8.2 GB
    'burgers_nu0.001': '268190',  # 1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5  8.2 GB
    'ns_incom_0':     '133280',   # 2D/NS_incom/ns_incom_inhom_2d_512-0.h5         9.9 GB
}

URL = 'https://darus.uni-stuttgart.de/api/access/datafile/{}'


def _open(key: str):
    import fsspec, h5py
    fs = fsspec.filesystem('http')
    return h5py.File(fs.open(URL.format(FILES[key]), block_size=8 * 1024 * 1024), 'r')


def _cached(name: str, build) -> Tuple[np.ndarray, np.ndarray]:
    """Build once, then reuse. The npz is a few hundred MB at most; data/ is gitignored."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f'{name}.npz'
    if p.exists():
        d = np.load(p)
        log.info('%s: cached %s -> %s', name, d['X'].shape, d['Y'].shape)
        return d['X'], d['Y']
    X, Y = build()
    np.savez_compressed(p, X=X, Y=Y)
    log.info('%s: built and cached %s -> %s (%.0f MB)', name, X.shape, Y.shape,
             p.stat().st_size / 1e6)
    return X, Y


def darcy(n: int = 1000, beta: str = '1.0') -> Tuple[np.ndarray, np.ndarray]:
    """2D Darcy flow: coefficient field a(x) -> steady solution u(x), on a 128x128 grid.

    The closest external analogue to this project's benchmark. -div(a grad u) = f is linear in
    u for fixed a, but the map a -> u is not, and PDEBench varies a per sample -- which is
    exactly the operator variation our fixed-placement dataset lacked (docs/report.md 9.14).
    """
    key = f'darcy_beta{beta}'

    def build():
        with _open(key) as h:
            a = np.asarray(h['nu'][:n], dtype=np.float64)          # (n, 128, 128)
            u = np.asarray(h['tensor'][:n, 0], dtype=np.float64)   # (n, 128, 128)
        return a.reshape(len(a), -1), u.reshape(len(u), -1)

    return _cached(f'{key}_n{n}', build)


def burgers(n: int = 400, nu: str = '0.01', t_out: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """1D Burgers: initial condition u(x, 0) -> solution u(x, t_out), 1024 grid points.

    Genuinely nonlinear (the u du/dx term), so a linear probe should fail here -- a negative
    control for the diagnostic rather than a benchmark we expect to break.

    `t_out` is NOT the final step, deliberately. Viscous dissipation flattens the solution, and
    at the last step (200 of 201) the target is close to uniform: measured spatial std falls
    from 0.42 to 0.055, with 13% of fields under 1e-3 and 41/400 in a larger sample. Detrended
    R^2 divides by that vanishing spatial variance, which produced a meaningless -747 on the
    first run -- and a mean-field predictor scoring -210, the tell that the metric had broken
    rather than the model. At t=20 structure is still 60% of initial and no field is
    degenerate. The same failure mode is flagged in third_party/ic_thermbench/README.md.
    """
    key = f'burgers_nu{nu}'

    def build():
        with _open(key) as h:
            # One contiguous read of the leading n samples, then slice the two times.
            block = np.asarray(h['tensor'][:n], dtype=np.float64)  # (n, 201, 1024)
        y = block[:, t_out, :]
        std = (y - y.mean(1, keepdims=True)).std(1)
        degenerate = int((std < 1e-3).sum())
        if degenerate:
            log.warning('burgers t_out=%d: %d/%d target fields have spatial std < 1e-3; '
                        'detrended R2 is unreliable on those', t_out, degenerate, len(y))
        return block[:, 0, :], y

    return _cached(f'{key}_t{t_out}_n{n}', build)


def navier_stokes(n_pairs: int = 300, stride: int = 2, every: int = 4
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """2D incompressible Navier-Stokes: velocity at t -> velocity at t+stride.

    This file holds 4 trajectories x 1000 timesteps at 512x512x2, so samples are time pairs
    drawn from within trajectories, not independent draws. Spatially subsampled by `every` to
    keep the fit tractable; the subsampling is stated because it changes the operator being
    probed (a coarse view of a fine field), unlike the Darcy and Burgers cases.
    """
    def build():
        with _open('ns_incom_0') as h:
            v = h['velocity']                                       # (4, 1000, 512, 512, 2)
            n_traj, n_t = v.shape[0], v.shape[1]
            per = max(1, n_pairs // n_traj)
            # Start times are spread across the usable window with OVERLAPPING pairs, so the
            # sample count stays ~constant as `stride` grows. Stepping t0 by `stride` instead
            # (the first version) gave 300/76/16 samples at strides 2/50/200, which confounded
            # the effect of the time gap with a collapsing training set -- at stride 200 only
            # ~12 samples remained, far too few to fit.
            usable = n_t - stride
            if usable <= 0:
                raise ValueError(f'stride {stride} exceeds trajectory length {n_t}')
            step = max(1, usable // per)
            Xs, Ys = [], []
            for tr in range(n_traj):
                for i in range(per):
                    t0 = i * step
                    if t0 + stride >= n_t:
                        break
                    Xs.append(np.asarray(v[tr, t0, ::every, ::every], dtype=np.float64).ravel())
                    Ys.append(np.asarray(v[tr, t0 + stride, ::every, ::every],
                                         dtype=np.float64).ravel())
        return np.asarray(Xs), np.asarray(Ys)

    return _cached(f'ns_incom_p{n_pairs}_s{stride}_e{every}', build)
