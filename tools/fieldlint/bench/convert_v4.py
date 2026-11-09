"""Conversion of the v4 held-out datasets (derived from convert_v3.py, which is unchanged).

Rules, as in v3 (checked by the <output>.MANIFEST.json written next to each output):
  * values are copied bit for bit: dtype is preserved, nothing is resampled, rescaled, normalised or cropped in space;
  * axis order is the stored order, EXCEPT where the data are stored as flat vectors whose documented index map is applied (ConDiff, below);
  * the only selection is by sample: the first N samples in file order (N fixed in advance, never chosen by looking at values);
  * the manifest records the sha256 of every source file and of every output file.
The single deliberately NON-lossless output is ConDiff `k_cell` (a derived array, labelled as such in the manifest and the data card).

Usage:  python -I tools/fieldlint/bench/convert_v4.py <name> [--data-root DATA]
        names: elado:<B1_test|C1_test>, condiff:<poisson_grid128|gaussian0.1_grid128|...>, darcy:<0.1|10.0|100.0>, helmholtz_pdegym, wave_layer
Needs numpy, h5py; zstandard for ELADO; fsspec + huggingface_hub for Wave-Layer; nothing else.
Run with `python -I` so that nothing in the downloaded directories is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(r'D:\Work\3D-ICE Thermal-modelling\Thermo\data\_external_v4')
N_MAX = 1000                      # first N samples in file order, fixed in advance


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
    print(json.dumps(man, indent=1))


# ── ELADO (zstd-compressed HDF5; the adapter reads .h5 directly) ───────────────────────
# The verified source is the ranged download `<name>.h5.zst.part2` (md5 checked against Zenodo). The plain `<name>.h5.zst` files that
# may sit in the same directory are truncated fragments of a failed first attempt and are not used.
def elado(root: Path, name: str):
    import zstandard
    src = root / 'elado' / f'{name}.h5.zst.part2'
    out = root / 'elado' / 'converted'
    out.mkdir(exist_ok=True)
    dst = out / f'{name}.h5'
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        zstandard.ZstdDecompressor().copy_stream(fi, fo)
    write_manifest(out, [src], [dst], f'{name}.h5.zst decompressed byte for byte (zstd is lossless); no other change')


# ── ConDiff (HF condiff/ConDiff; generator github.com/condiff-dataset/ConDiff, condiff.py) ──────────────────────────
# Stored: k (N,129,129) coefficient at the nodes, rhs (N,n*n) and x (N,n*n) flat vectors built with flatten("F")
# (condiff.py lines 151-155), n = 128 cells per side. Outputs:
#   u, source : x and rhs reshaped per sample with order='F' to (n, n), i.e. x.reshape(N,n,n).transpose(0,2,1). A pure re-indexing by the
#               documented index map; every value is copied unchanged.
#   k_node    : k as stored (129 x 129); cannot share a grid with u, kept for completeness.
#   k_cell    : DERIVED, NOT LOSSLESS. The solver's own cell coefficient, the mean of the four corner nodes, i.e. `ac / 4` of fd_mtx2
#               (condiff.py lines 145-148: ac[i,j] = a[i,j] + a[i,j+1] + a[i+1,j] + a[i+1,j+1]); float64 arithmetic.
def condiff(root: Path, name: str):
    import h5py
    src = root / 'condiff' / name / f'{name}_test.h5'
    out = root / 'condiff' / 'converted'
    out.mkdir(exist_ok=True)
    with h5py.File(src, 'r') as f:
        x, rhs = f['x'][:], f['rhs'][:]
        k = f['k'][:] if 'k' in f else None
    n = int(round(x.shape[1] ** 0.5))
    arrays = {'u': np.ascontiguousarray(x.reshape(-1, n, n).transpose(0, 2, 1)),
              'source': np.ascontiguousarray(rhs.reshape(-1, n, n).transpose(0, 2, 1))}
    if k is not None:
        arrays['k_node'] = k
        arrays['k_cell'] = 0.25 * (k[:, :-1, :-1] + k[:, :-1, 1:] + k[:, 1:, :-1] + k[:, 1:, 1:])
    assert np.array_equal(arrays['u'][3].reshape(-1, order='F'), x[3])          # round trip of the index map
    path = out / f'{name}_test.npz'
    np.savez(path, **arrays)
    write_manifest(out, [src], [path], f'{name}_test: u, source = x, rhs re-indexed with order F; k_node as stored; '
                   "k_cell = mean of the four corner nodes of k (derived: the solver's own cell coefficient, NOT lossless)",
                   {'dtypes': {a: str(v.dtype) for a, v in arrays.items()}, 'shapes': {a: list(v.shape) for a, v in arrays.items()}})


# ── PDEBench 2D Darcy (DaRUS 10.18419/DARUS-2986) ──────────────────────────────────────────────────────────────
# The file has `nu` (10000,128,128) and `tensor` (10000,1,128,128). Same packaging as the v1/v2 files data/pde-bench/darcy_beta*_n1000.npz:
# X = nu, Y = tensor, first 1000 samples, each flattened C-order to 16384, dtype kept as stored; the existing `darcy` preset reads it.
def pdebench_darcy(root: Path, beta: str):
    import h5py
    src = root / 'pdebench_darcy' / f'2D_DarcyFlow_beta{beta}_Train.hdf5'
    if (src.parent / (src.name + '.part2')).exists():          # the verified ranged download; a plain-named file may be a truncated fragment
        src = src.parent / (src.name + '.part2')
    out = root / 'pdebench_darcy' / 'converted'
    out.mkdir(exist_ok=True)
    with h5py.File(src, 'r') as f:
        nu = f['nu'][:N_MAX]
        u = f['tensor'][:N_MAX]
    arrays = {'X': nu.reshape(nu.shape[0], -1), 'Y': u.reshape(u.shape[0], -1)}
    path = out / f'darcy_beta{beta}_n{nu.shape[0]}.npz'
    np.savez(path, **arrays)
    write_manifest(out, [src], [path], 'first 1000 samples: X = nu, Y = tensor (channel 0), each flattened C-order to 16384',
                   {'dtypes': {a: str(v.dtype) for a, v in arrays.items()}, 'shapes': {a: list(v.shape) for a, v in arrays.items()}})


# ── PDEgym Helmholtz.h5 (HF camlab-ethz/Helmholtz): groups Sample_<i> with a (128,128), u (128,128), bc (scalar) ──────────
def helmholtz_pdegym(root: Path):
    import h5py
    src = root / 'helmholtz_pdegym' / 'Helmholtz.h5'
    out = root / 'helmholtz_pdegym' / 'converted'
    out.mkdir(exist_ok=True)
    with h5py.File(src, 'r') as f:
        names = sorted((k for k in f.keys() if k.lower().startswith('sample')), key=lambda s: int(re.findall(r'\d+', s)[-1]))[:N_MAX]
        arrays = {'a': np.stack([f[k]['a'][()] for k in names]), 'u': np.stack([f[k]['u'][()] for k in names]),
                  'bc': np.asarray([f[k]['bc'][()] for k in names])}
    path = out / f'helmholtz_first{len(names)}.npz'
    np.savez(path, **arrays)
    write_manifest(out, [src], [path], 'Sample_i/a, u, bc stacked in numeric sample order; stored axes kept',
                   {'groups_first_last': [names[0], names[-1]], 'dtypes': {a: str(v.dtype) for a, v in arrays.items()},
                    'shapes': {a: list(v.shape) for a, v in arrays.items()}})


# ── PDEgym Wave-Layer: range-read subset of the first 256 trajectories (as wave_gauss() in convert_v3) ─────────────────
def wave_layer(root: Path, n: int = 256):
    import fsspec
    import h5py
    from huggingface_hub import HfApi
    out = root / 'wave_layer' / 'converted'
    out.mkdir(parents=True, exist_ok=True)
    base = 'https://huggingface.co/datasets/camlab-ethz/Wave-Layer/resolve/main/'
    arrays, lfs = {}, {}
    for s in HfApi().dataset_info('camlab-ethz/Wave-Layer', files_metadata=True).siblings:
        if s.lfs:
            lfs[s.rfilename] = {'bytes': s.size, 'sha256_lfs': s.lfs.sha256}
    for fn, var in (('solution_0.nc', 'solution'), ('c_0.nc', 'c')):
        with fsspec.open(base + fn, 'rb', block_size=16 << 20).open() as fh, h5py.File(fh, 'r') as f:
            arrays[var] = f[var][:n]
            print(fn, var, f[var].shape, f[var].dtype, '-> kept', arrays[var].shape, flush=True)
    path = out / f'wave_layer_first{n}.npz'
    np.savez(path, **arrays)
    write_manifest(out, [], [path], f'first {n} trajectories of solution_0.nc (sample, time, x, y) and the first {n} rows of c_0.nc (sample, x, y), '
                   'read by HTTP range requests; sources not downloaded in full, so their sha256 is the Hugging Face LFS id',
                   {'sources_lfs': {k: lfs[k] for k in ('solution_0.nc', 'c_0.nc')}, 'dtypes': {k: str(v.dtype) for k, v in arrays.items()},
                    'shapes': {k: list(v.shape) for k, v in arrays.items()}})


CONVERTERS = {
    'helmholtz_pdegym': helmholtz_pdegym,
    'wave_layer': wave_layer,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', help='converter name, or elado:<B1_test>, condiff:<gaussian0.1_grid128>, darcy:<0.1>')
    ap.add_argument('--data-root', default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    root = Path(a.data_root)
    if ':' in a.name:
        kind, arg = a.name.split(':', 1)
        {'elado': elado, 'condiff': condiff, 'darcy': pdebench_darcy}[kind](root, arg)
    else:
        CONVERTERS[a.name](root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
