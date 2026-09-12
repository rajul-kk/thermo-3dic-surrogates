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


def burgers(n: int = 400, nu: str = '0.01') -> Tuple[np.ndarray, np.ndarray]:
    """1D Burgers: initial condition u(x, 0) -> solution u(x, T), 1024 grid points.

    Genuinely nonlinear (the u du/dx term), so a linear probe should fail here. That makes it
    a negative control for the diagnostic rather than a benchmark we expect to break.
    """
    key = f'burgers_nu{nu}'

    def build():
        with _open(key) as h:
            t = h['tensor']
            # Read the leading n samples in one contiguous span, then take first/last times.
            block = np.asarray(t[:n], dtype=np.float64)            # (n, 201, 1024)
        return block[:, 0, :], block[:, -1, :]

    return _cached(f'{key}_n{n}', build)


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
            n_traj = v.shape[0]
            per = max(1, n_pairs // n_traj)
            Xs, Ys = [], []
            for tr in range(n_traj):
                for i in range(per):
                    t0 = i * stride
                    if t0 + stride >= v.shape[1]:
                        break
                    Xs.append(np.asarray(v[tr, t0, ::every, ::every], dtype=np.float64).ravel())
                    Ys.append(np.asarray(v[tr, t0 + stride, ::every, ::every],
                                         dtype=np.float64).ravel())
        return np.asarray(Xs), np.asarray(Ys)

    return _cached(f'ns_incom_p{n_pairs}_s{stride}_e{every}', build)
