"""PDEBench loaders for the linearity audit: canonical operator-learning benchmarks as (X, Y) pairs."""
from __future__ import annotations

import hashlib
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
    'shallow_water':  '133021',   # 2D/shallow-water/2D_rdb_NA_NA.h5              6.6 GB
    'diff_react':     '133017',   # 2D/diffusion-reaction/2D_diff-react_NA_NA.h5 13.2 GB
    # Sibling NS_incom files (same 4-trajectories-x-1000-steps-x-512x512x2 layout), looked up
    # via the DaRUS dataset file listing 2026-09-16 -- used only to get INDEPENDENT simulation
    # runs (different initial/forcing conditions) for a leave-one-file-out robustness check on
    # the §9.16b persistence finding, which was previously measured on a single file's 4
    # trajectories only.
    'ns_incom_10':    '133309',   # ns_incom_inhom_2d_512-10.h5
    'ns_incom_11':    '133336',   # ns_incom_inhom_2d_512-11.h5
    'ns_incom_12':    '133574',   # ns_incom_inhom_2d_512-12.h5
    'ns_incom_13':    '133304',   # ns_incom_inhom_2d_512-13.h5
    'ns_incom_14':    '133281',   # ns_incom_inhom_2d_512-14.h5
    # Extended to 25 files total for the 20-30 file scale-up (2026-09-17): a wider slice of
    # the same DaRUS file listing, chosen only for availability, not for any property of the
    # runs.
    'ns_incom_100':   '133267', 'ns_incom_101': '133289', 'ns_incom_102': '133291',
    'ns_incom_103':   '133294', 'ns_incom_104': '133298', 'ns_incom_105': '133290',
    'ns_incom_106':   '133374', 'ns_incom_107': '133375', 'ns_incom_108': '133376',
    'ns_incom_109':   '133305', 'ns_incom_110': '133313', 'ns_incom_111': '133318',
    'ns_incom_112':   '133324', 'ns_incom_113': '133377', 'ns_incom_114': '133378',
    'ns_incom_115':   '133379', 'ns_incom_116': '133380', 'ns_incom_117': '133381',
    'ns_incom_118':   '133383', 'ns_incom_119': '133385',
}

URL = 'https://darus.uni-stuttgart.de/api/access/datafile/{}'


def _open(key: str, block_mb: int = 8):
    import fsspec, h5py
    fs = fsspec.filesystem('http')
    return h5py.File(fs.open(URL.format(FILES[key]), block_size=block_mb * 1024 * 1024), 'r')


def _grouped_pair(key: str, n: int, t_out: int, name: str):
    """Read N independent trajectories from a group-per-sample file: u(t=0) -> u(t=t_out).

    These files store one HDF5 group per trajectory ('0000'..'0999'), each holding
    data of shape (101, 128, 128, C). Unlike ns_incom (4 trajectories, overlapping time
    pairs) these 1000 trajectories are independent initial conditions, so samples are i.i.d.
    A small HTTP block size is used because only two of 101 timesteps are wanted per group.
    """
    with _open(key, block_mb=1) as h:
        groups = sorted(h)[:n]
        X, Y = [], []
        for g in groups:
            d = h[g]['data']
            X.append(np.asarray(d[0], dtype=np.float64).ravel())
            Y.append(np.asarray(d[t_out], dtype=np.float64).ravel())
    X, Y = np.asarray(X), np.asarray(Y)
    std = (Y - Y.mean(1, keepdims=True)).std(1)
    if int((std < 1e-3).sum()):
        log.warning('%s t_out=%d: %d/%d targets have spatial std < 1e-3; detrended R2 is '
                    'unreliable on those', name, t_out, int((std < 1e-3).sum()), len(Y))
    return X, Y


def shallow_water(n: int = 300, t_out: int = 20):
    """2D shallow water (radial dam break): h(t=0) -> h(t=t_out), 128x128.

    The second time-evolution benchmark, added to test whether the persistence finding of
    docs/report.md 9.16b generalises beyond incompressible Navier-Stokes or is specific to
    it. Structure decays gently here (spatial std 0.173 -> 0.121 by t=20) and input and
    output are on the same scale, so persistence is a genuinely plausible baseline rather
    than a straw man.
    """
    return _cached(f'shallow_water_t{t_out}_n{n}',
                   lambda: _grouped_pair('shallow_water', n, t_out, 'shallow_water'))


def diffusion_reaction(n: int = 300, t_out: int = 20):
    """2D diffusion-reaction (activator/inhibitor, 2 channels): u(t=0) -> u(t=t_out).

    Both channels are flattened together. Note the input is a random initial condition whose
    spatial std (0.998) is ~24x the target's (0.042) -- the field settles into pattern
    formation within two steps -- so persistence is expected to fail badly here, unlike
    shallow water. That contrast is the point of including both.
    """
    return _cached(f'diff_react_t{t_out}_n{n}',
                   lambda: _grouped_pair('diff_react', n, t_out, 'diff_react'))


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


def navier_stokes_multi_file(file_keys, per_traj: int = 5, stride: int = 2, every: int = 4
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same task as navier_stokes(), but pooled across several independent NS_incom files.

    Each file is a separate 3D-ICE-unrelated PDEBench simulation run (different initial/
    forcing conditions), so this -- unlike navier_stokes() -- gives genuinely independent
    samples across files, letting a leave-one-file-out split test the §9.16b persistence
    finding without the single-file time-correlation caveat. Returns (X, Y, group) where
    `group` is an integer file index per sample, for use as CV groups.
    """
    def build():
        Xs, Ys, Gs = [], [], []
        for gi, key in enumerate(file_keys):
            with _open(key) as h:
                v = h['velocity']                                   # (4, 1000, 512, 512, 2)
                n_traj, n_t = v.shape[0], v.shape[1]
                usable = n_t - stride
                if usable <= 0:
                    raise ValueError(f'stride {stride} exceeds trajectory length {n_t}')
                step = max(1, usable // per_traj)
                for tr in range(n_traj):
                    for i in range(per_traj):
                        t0 = i * step
                        if t0 + stride >= n_t:
                            break
                        Xs.append(np.asarray(v[tr, t0, ::every, ::every],
                                             dtype=np.float64).ravel())
                        Ys.append(np.asarray(v[tr, t0 + stride, ::every, ::every],
                                             dtype=np.float64).ravel())
                        Gs.append(gi)
        return np.asarray(Xs), np.asarray(Ys), np.asarray(Gs)

    # Cache key: spelling out every file key overran Windows' 260-char MAX_PATH at 26 files
    # (408 chars) and surfaced as FileNotFoundError at SAVE time, discarding a completed
    # ~30-minute download. Long lists collapse to a stable hash; short ones keep the original
    # spelled-out name so caches built before this change are still found.
    spelled = f'ns_multi_{"_".join(file_keys)}_pt{per_traj}_s{stride}_e{every}'
    if len(str((CACHE / f'{spelled}.npz').resolve())) <= 200:
        key = spelled
    else:
        digest = hashlib.sha1('_'.join(file_keys).encode()).hexdigest()[:10]
        key = f'ns_multi_n{len(file_keys)}_{digest}_pt{per_traj}_s{stride}_e{every}'
    p = CACHE / f'{key}.npz'
    if p.exists():
        d = np.load(p)
        log.info('%s: cached %s -> %s', key, d['X'].shape, d['Y'].shape)
        return d['X'], d['Y'], d['G']

    # Pre-flight: prove the cache path is writable BEFORE fetching anything, so an
    # unwritable target can never again throw away hours of completed downloads.
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        p.touch()
        p.unlink()
    except OSError as exc:
        raise OSError(f'cache path not writable before download: {p} ({exc})') from exc

    X, Y, G = build()
    np.savez_compressed(p, X=X, Y=Y, G=G)
    log.info('%s: built %s -> %s (%d groups)', key, X.shape, Y.shape, len(set(G.tolist())))
    return X, Y, G
