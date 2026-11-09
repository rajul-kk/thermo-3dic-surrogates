"""Conversion of the v5 held-out datasets (derived from convert_v4.py, which is unchanged).

Same rules as v3/v4: values are copied bit for bit (dtype preserved), nothing is rescaled or cropped in space, axis order is the stored order (except
the documented ConDiff index map and the SISSA mesh rasterisation, both labelled), the only selection is by sample/trajectory count fixed in advance
(never by looking at values), and a <output>.MANIFEST.json records sha256 of every source and output file.

Usage:  python -I tools/fieldlint/bench/convert_v5.py <name> [--data-root D:/fieldlint_data_v5]
  names: condiff:<cubic1.0_grid128|...>, helmholtz_staircase, acoustic_inclusions, rayleigh_benard, flowbench, sissa
Needs numpy, h5py, fsspec (+aiohttp), pyarrow, scipy, huggingface_hub. Run with `python -I` so that nothing in the downloaded directories is imported.
Large HDF5 files (The Well) are read by HTTP range requests (only the frames used); the sha256 of such a source is the Hugging Face LFS id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(r'D:\fieldlint_data_v5')
WELL = 'https://huggingface.co/datasets/polymathic-ai/{ds}/resolve/main/data/{split}/{fn}'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def write_manifest(outdir: Path, sources: list[Path], outputs: list[Path], note: str, extra: dict | None = None):
    man = {'note': note, 'sources': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sources},
           'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in outputs}, **(extra or {})}
    (outdir / (outputs[0].stem + '.MANIFEST.json')).write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1)[:3000])


def lfs_ids(repo: str, prefix: str) -> dict:
    from huggingface_hub import HfApi
    out = {}
    for s in HfApi().dataset_info(repo, files_metadata=True).siblings:
        if s.rfilename.startswith(prefix) and s.lfs:
            out[s.rfilename] = {'bytes': s.size, 'sha256_lfs': s.lfs.sha256}
    return out


def open_remote(url):
    import fsspec
    return fsspec.open(url, 'rb', block_size=2 << 20).open()


# ── ConDiff (identical packaging to convert_v4.condiff; files in <root>/condiff_v5) ─────────────────────────────
def condiff(root: Path, name: str):
    import h5py
    src = root / 'condiff_v5' / f'{name}_test.h5'
    out = root / 'condiff_v5' / 'converted'
    out.mkdir(exist_ok=True)
    with h5py.File(src, 'r') as f:
        x, rhs, k = f['x'][:], f['rhs'][:], f['k'][:]
    n = int(round(x.shape[1] ** 0.5))
    arrays = {'u': np.ascontiguousarray(x.reshape(-1, n, n).transpose(0, 2, 1)),
              'source': np.ascontiguousarray(rhs.reshape(-1, n, n).transpose(0, 2, 1)),
              'k_node': k, 'k_cell': 0.25 * (k[:, :-1, :-1] + k[:, :-1, 1:] + k[:, 1:, :-1] + k[:, 1:, 1:])}
    assert np.array_equal(arrays['u'][3].reshape(-1, order='F'), x[3])
    path = out / f'{name}_test.npz'
    np.savez(path, **arrays)
    write_manifest(out, [src], [path], f'{name}_test as in convert_v4.condiff: u, source re-indexed with order F; k_cell derived (NOT lossless)',
                   {'dtypes': {a: str(v.dtype) for a, v in arrays.items()}, 'shapes': {a: list(v.shape) for a, v in arrays.items()}})


# ── The Well helmholtz_staircase: the first 13 trajectories of each of the 16 TRAIN files (sorted by file name), frame t = 0 of pressure_re ──────────
# Decided before any rule ran: 16 omega x 13 = 208 samples (>= 200). Frame 0 of U(t, x) = u(x) exp(-i omega t) is Re u. The source position is NOT stored in the file.
def helmholtz_staircase(root: Path, per_file: int = 13):
    import h5py
    from huggingface_hub import HfApi
    out = root / 'helmholtz_staircase' / 'converted'
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(s.rfilename for s in HfApi().dataset_info('polymathic-ai/helmholtz_staircase').siblings if s.rfilename.startswith('data/train/'))
    assert len(files) == 16, files
    us, om, mask = [], [], None
    for fn in files:
        with open_remote(WELL.format(ds='helmholtz_staircase', split='train', fn=fn.split('/')[-1])) as fh, h5py.File(fh, 'r') as f:
            us.append(f['t0_fields/pressure_re'][:per_file, 0])
            om += [float(f['scalars/omega'][()])] * per_file
            mask = f['t0_fields/mask'][()]
        print(fn, us[-1].shape, flush=True)
    arrays = {'u': np.concatenate(us), 'omega': np.asarray(om), 'mask': mask}
    path = out / 'helmholtz_staircase_train_first13_t0.npz'
    np.savez(path, **arrays)
    write_manifest(out, [], [path], 'pressure_re[:13, 0] of each train file by HTTP range request; omega per sample; mask as stored (stationary)',
                   {'sources_lfs': lfs_ids('polymathic-ai/helmholtz_staircase', 'data/train/'),
                    'dtypes': {k: str(v.dtype) for k, v in arrays.items()}, 'shapes': {k: list(v.shape) for k, v in arrays.items()}})


# ── The Well acoustic_scattering_inclusions: test chunks 36 and 37 (100 trajectories each) ────────────────────────────────────────────────────
def acoustic_inclusions(root: Path):
    import h5py
    out = root / 'acoustic_inclusions' / 'converted'
    out.mkdir(parents=True, exist_ok=True)
    p0, pT, c = [], [], []
    names = [f'acoustic_scattering_inclusions_chunk_{i}.hdf5' for i in (36, 37)]
    for fn in names:
        with open_remote(WELL.format(ds='acoustic_scattering_inclusions', split='test', fn=fn)) as fh, h5py.File(fh, 'r') as f:
            p0.append(f['t0_fields/pressure'][:, 0])
            pT.append(f['t0_fields/pressure'][:, -1])
            c.append(f['t0_fields/speed_of_sound'][()])
        print(fn, p0[-1].shape, flush=True)
    arrays = {'p_first': np.concatenate(p0), 'p_last': np.concatenate(pT), 'c': np.concatenate(c)}
    path = out / 'acoustic_inclusions_test36_37.npz'
    np.savez(path, **arrays)
    write_manifest(out, [], [path], 'pressure[:, 0], pressure[:, -1] (102 stored frames) and speed_of_sound of test chunks 36, 37 by HTTP range request',
                   {'sources_lfs': {k: v for k, v in lfs_ids('polymathic-ai/acoustic_scattering_inclusions', 'data/test/').items() if k.endswith(tuple(names))},
                    'dtypes': {k: str(v.dtype) for k, v in arrays.items()}, 'shapes': {k: list(v.shape) for k, v in arrays.items()}})


# ── The Well rayleigh_benard: first 256 trajectories over test files (sorted by name) then valid files (sorted), buoyancy frames 0 and -1 ──────────
def rayleigh_benard(root: Path, n: int = 256):
    import h5py
    from huggingface_hub import HfApi
    out = root / 'rayleigh_benard' / 'converted'
    out.mkdir(parents=True, exist_ok=True)
    sib = [s.rfilename for s in HfApi().dataset_info('polymathic-ai/rayleigh_benard').siblings]
    files = [(sp, f) for sp in ('test', 'valid') for f in sorted(x for x in sib if x.startswith(f'data/{sp}/'))]
    b0, bT, ra, pr, used = [], [], [], [], []
    for sp, fn in files:
        if sum(len(x) for x in b0) >= n:
            break
        with open_remote(WELL.format(ds='rayleigh_benard', split=sp, fn=fn.split('/')[-1])) as fh, h5py.File(fh, 'r') as f:
            b = f['t0_fields/buoyancy']
            b0.append(b[:, 0])
            bT.append(b[:, -1])
            ra += [float(f['scalars/Rayleigh'][()])] * b.shape[0]
            pr += [float(f['scalars/Prandtl'][()])] * b.shape[0]
        used.append(fn)
        print(fn, b0[-1].shape, flush=True)
    arrays = {'b_first': np.concatenate(b0)[:n], 'b_last': np.concatenate(bT)[:n], 'Rayleigh': np.asarray(ra)[:n], 'Prandtl': np.asarray(pr)[:n]}
    path = out / f'rayleigh_benard_first{n}.npz'
    np.savez(path, **arrays)
    write_manifest(out, [], [path], f'buoyancy frames 0 and -1 (200 stored) of the first {n} trajectories (test files sorted, then valid files sorted) by HTTP range request',
                   {'files_used': used, 'sources_lfs': {k: v for k, v in lfs_ids('polymathic-ai/rayleigh_benard', 'data/').items() if k in used},
                    'dtypes': {k: str(v.dtype) for k, v in arrays.items()}, 'shapes': {k: list(v.shape) for k, v in arrays.items()}})


# ── FlowBench LDC NS+HT constant Re 128 x 128, harmonics: X (1000,3,128,128), Y (1000,5,128,128) ───────────────────────────────────────────────
# Channel semantics are NOT as in the paper's Table 12 (see datasets_v5.md): X = [Gr (constant per sample), a continuous field (SDF), a binary 0/255 field (mask)],
# Y = [5 channels, channel 3 = temperature]. Copied as stored: first N channels kept by index, nothing rescaled.
def flowbench(root: Path):
    d = root / 'flowbench_ldcht_constRe_128_harmonics'
    out = d / 'converted'
    out.mkdir(exist_ok=True)
    X = np.load(d / 'harmonics_lid_driven_cavity_X.npz')['data']
    Y = np.load(d / 'harmonics_lid_driven_cavity_Y.npz')['data']
    arrays = {'theta': Y[:, 3], 'mask': X[:, 2], 'sdf': X[:, 1], 'Gr': X[:, 0, 0, 0]}
    path = out / 'flowbench_ldcht_constRe_128_harmonics_N1000.npz'
    np.savez(path, **arrays)
    write_manifest(out, [d / 'harmonics_lid_driven_cavity_X.npz', d / 'harmonics_lid_driven_cavity_Y.npz'], [path],
                   'Y[:,3] -> theta; X[:,2] -> mask (binary 0/255 as stored); X[:,1] -> sdf; X[:,0,0,0] -> Gr; all 1000 samples, stored axes',
                   {'dtypes': {k: str(v.dtype) for k, v in arrays.items()}, 'shapes': {k: list(v.shape) for k, v in arrays.items()}})


# ── SISSA thermal-conduction: 2_, 3_, 4_stripes.star (100 snapshots each, one fixed unstructured mesh per split) rasterised to 128 x 128 ───────────
# PROCESSED: scipy griddata linear for the steady nodal temperature, nearest for the (piecewise constant) nodal conductivity, on a uniform grid over
# [0, 15]^2 (the mesh bounding box). Order: the three splits concatenated (2, 3, 4 stripes), 100 samples each.
def sissa(root: Path, ngrid: int = 128):
    import pyarrow.parquet as pq
    from scipy.interpolate import griddata
    d = root / 'sissa_thermal_conduction_star'
    out = d / 'converted'
    out.mkdir(exist_ok=True)
    g = np.linspace(0.0, 15.0, ngrid)
    GX, GY = np.meshgrid(g, g, indexing='ij')                       # array[i, j] at (x_i, y_j)
    U, K, S, IN, srcs = [], [], [], [], []
    for ns in (2, 3, 4):
        geo = pq.read_table(d / 'geometry' / f'{ns}_stripes.star-00000-of-00001.parquet').to_pylist()
        snap = pq.read_table(d / 'snapshots' / f'{ns}_stripes.star-00000-of-00001.parquet', columns=['steady']).to_pylist()
        srcs += [d / 'geometry' / f'{ns}_stripes.star-00000-of-00001.parquet', d / 'snapshots' / f'{ns}_stripes.star-00000-of-00001.parquet']
        pts = np.asarray(geo[0]['points'])[:, :2]
        assert all(np.array_equal(np.asarray(r['points'])[:, :2], pts) for r in geo)       # one mesh per split
        for r, s in zip(geo, snap):
            u = np.asarray(s['steady'])
            k = np.asarray(r['conductivity'])
            ul = griddata(pts, u, (GX, GY), method='linear')
            un = griddata(pts, u, (GX, GY), method='nearest')
            inside = np.isfinite(ul)                                  # outside the convex hull of the nodes (e.g. a non-convex domain) the nearest node is used
            U.append(np.where(inside, ul, un))
            IN.append(inside)
            K.append(griddata(pts, k, (GX, GY), method='nearest'))
            S.append(ns)
    arrays = {'u': np.asarray(U), 'k': np.asarray(K), 'stripes': np.asarray(S), 'inside_hull': np.asarray(IN)}
    assert np.isfinite(arrays['u']).all()
    path = out / 'sissa_star_2_3_4_stripes_128.npz'
    np.savez(path, **arrays)
    write_manifest(out, srcs, [path], 'steady nodal temperature (linear) and conductivity (nearest) rasterised to 128 x 128 over [0,15]^2; PROCESSED, not native',
                   {'dtypes': {k: str(v.dtype) for k, v in arrays.items()}, 'shapes': {k: list(v.shape) for k, v in arrays.items()}})


CONVERTERS = {'helmholtz_staircase': helmholtz_staircase, 'acoustic_inclusions': acoustic_inclusions, 'rayleigh_benard': rayleigh_benard,
              'flowbench': flowbench, 'sissa': sissa}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name')
    ap.add_argument('--data-root', default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    root = Path(a.data_root)
    if a.name.startswith('condiff:'):
        condiff(root, a.name.split(':', 1)[1])
    else:
        CONVERTERS[a.name](root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
