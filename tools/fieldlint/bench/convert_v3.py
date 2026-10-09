"""Lossless conversion of the v3 held-out datasets to files the frozen fieldlint adapters can read.

Why this exists: bench/results/frozen_thresholds_v2.json hashes src/fieldlint/adapters.py (and rules.py, model.py, runner.py and the
five original presets), so the adapter may not gain a NetCDF / Arrow / HDF5-group / zstd reader. Instead each source file is
re-packaged here into a plain `.npz` (or, for ELADO, decompressed to the `.h5` the adapter already reads).

Rules for every conversion (checked by the <output>.MANIFEST.json written next to each output):
  * values are copied bit for bit: dtype is preserved (no float32 -> float64, no casting), nothing is resampled, rescaled,
    normalised, transposed or cropped in space;
  * axis order is the stored order (so a dataset that stores its inputs transposed against its outputs is NOT repaired);
  * the only selection is by sample: the first N samples in file order (N given below, never chosen by looking at values);
  * the manifest records the sha256 of every source file and of every output file.

Usage:  python tools/fieldlint/bench/convert_v3.py <name> [--data-root DATA]      (names: see CONVERTERS)
Needs numpy, h5py; pyarrow for the DiffusionPDE sets; zstandard for ELADO; nothing else.
Run with `python -I` so that nothing in the downloaded directories is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(r'D:\Work\3D-ICE Thermal-modelling\Thermo\data\_external')
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


# ── PDEgym NetCDF4 (an HDF5 file; h5py reads it) ────────────────────────────────────────
def poisson_gauss(root: Path):
    import h5py
    src = root / 'poisson_gauss' / 'Poisson-Gauss.nc'
    out = root / 'poisson_gauss' / 'converted'
    out.mkdir(exist_ok=True)
    with h5py.File(src, 'r') as f:
        n = min(N_MAX, f['solution'].shape[0])
        arrays = {'source': f['source'][:n], 'solution': f['solution'][:n], 'x': f['x'][:], 'y': f['y'][:]}
    path = out / 'poisson_gauss_first1000.npz'
    np.savez(path, **arrays)
    write_manifest(out, [src], [path], 'first 1000 samples of source/solution as stored; axes (sample, x, y) per the dataset README',
                   {'dtypes': {k: str(v.dtype) for k, v in arrays.items()}, 'shapes': {k: list(v.shape) for k, v in arrays.items()}})


# ── DiffusionPDE-normalized (HF datasets / Arrow IPC stream) ─────────────────────────────
def diffusionpde(root: Path, sub: str):
    import pyarrow as pa
    d = root / 'diffusionpde_norm' / sub
    src = sorted(d.glob('data-00000-of-*.arrow'))[0]
    out = root / 'diffusionpde_norm' / 'converted'
    out.mkdir(exist_ok=True)
    chunks, n = [], 0
    for batch in pa.ipc.open_stream(pa.memory_map(str(src))):
        a = batch.column('data').flatten().flatten().flatten().to_numpy(zero_copy_only=False)
        a = a.reshape(batch.num_rows, 2, 128, 128)
        chunks.append(a[: N_MAX - n])
        n += chunks[-1].shape[0]
        if n >= N_MAX:
            break
    data = np.concatenate(chunks)
    path = out / f'{sub}_first{data.shape[0]}.npz'
    np.savez(path, data=data)                         # (N, 2, 128, 128) float64, channels as stored
    meta = d / 'metadata.json'
    write_manifest(out, [src, meta], [path], f'first {data.shape[0]} rows of {src.name}; data[:, c] = channel c as stored',
                   {'dtype': str(data.dtype), 'shape': list(data.shape), 'metadata.json': json.loads(meta.read_text())})


# ── CNO Representative PDE Benchmarks (HDF5 with one group per sample) ─────────────────
def cno(root: Path, spec: str):
    """spec = <member.h5>[:<regex on group names>]. Reads root/cno_rpb/extracted/data/<member> (= data.zip unpacked, unchanged).
    Writes input (N, H, W) and output (N, H, W) stacked in numeric sample order, plus the file-level scalars as scalar_<name>."""
    import re
    import h5py
    member, _, pat = spec.partition(':')
    src = root / 'cno_rpb' / 'extracted' / 'data' / member
    out = root / 'cno_rpb' / 'converted'
    out.mkdir(exist_ok=True)
    f = h5py.File(src, 'r')
    groups = [k for k in f.keys() if k.lower().startswith('sample') and (not pat or re.search(pat, k))]
    groups.sort(key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)])   # numeric sample order
    groups = groups[:N_MAX]
    arrays = {'input': np.stack([f[g]['input'][()] for g in groups]), 'output': np.stack([f[g]['output'][()] for g in groups])}
    scalars = {k: f[k][()] for k in f.keys() if not k.lower().startswith('sample')}
    tag = Path(member).stem + (re.sub(r'\W+', '', pat) if pat else '')
    path = out / f'{tag}_first{len(groups)}.npz'
    np.savez(path, **arrays, **{('scalar_' + k): np.asarray(v) for k, v in scalars.items()})
    write_manifest(out, [src], [path], f'{member}: Sample_i/input and Sample_i/output stacked in numeric sample order (groups matching '
                   f'{pat!r}); file-level scalars stored as scalar_<name>; the source file is the member of data.zip (Zenodo 10406879) as unzipped',
                   {'groups_first_last': [groups[0], groups[-1]], 'dtypes': {k: str(v.dtype) for k, v in arrays.items()},
                    'shapes': {k: list(v.shape) for k, v in arrays.items()}})


# ── PDEgym Wave-Gauss: range-read subset of the first 256 trajectories (the full chunk files are 3.7 GB / 0.7 GB) ─────
def wave_gauss(root: Path, n: int = 256):
    import fsspec
    import h5py
    out = root / 'wave_gauss' / 'converted'
    out.mkdir(parents=True, exist_ok=True)
    base = 'https://huggingface.co/datasets/camlab-ethz/Wave-Gauss/resolve/main/'
    arrays, lfs = {}, {}
    from huggingface_hub import HfApi
    for s in HfApi().dataset_info('camlab-ethz/Wave-Gauss', files_metadata=True).siblings:
        if s.lfs:
            lfs[s.rfilename] = {'bytes': s.size, 'sha256_lfs': s.lfs.sha256}
    for fn, var in (('solution_0.nc', 'solution'), ('c_0.nc', 'c')):
        with fsspec.open(base + fn, 'rb', block_size=16 << 20).open() as fh, h5py.File(fh, 'r') as f:
            arrays[var] = f[var][:n]
            print(fn, var, f[var].shape, f[var].dtype, '-> kept', arrays[var].shape, flush=True)
    path = out / f'wave_gauss_first{n}.npz'
    np.savez(path, **arrays)
    write_manifest(out, [], [path], f'first {n} trajectories of solution_0.nc (sample, time, x, y) and the first {n} rows of c_0.nc (sample, x, y), '
                   'read by HTTP range requests; the two sources are NOT downloaded in full, so their sha256 is the Hugging Face LFS id',
                   {'sources_lfs': {k: lfs[k] for k in ('solution_0.nc', 'c_0.nc')}, 'dtypes': {k: str(v.dtype) for k, v in arrays.items()},
                    'shapes': {k: list(v.shape) for k, v in arrays.items()}})


# ── ELADO (zstd-compressed HDF5; the adapter reads .h5 directly) ───────────────────────
def elado(root: Path, name: str):
    import zstandard
    src = root / 'elado' / f'{name}.h5.zst'
    out = root / 'elado' / 'converted'
    out.mkdir(exist_ok=True)
    dst = out / f'{name}.h5'
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        zstandard.ZstdDecompressor().copy_stream(fi, fo)
    write_manifest(out, [src], [dst], f'{name}.h5.zst decompressed byte for byte (zstd is lossless); no other change')


# ── UCSD/LLNL MTNN (tar of npz; the adapter reads .npz directly) ─────────────────────────
def ucsd(root: Path, member_suffix: str):
    tpath = root / 'ucsd_pde' / 'bb1852369g_2_1.tar'
    out = root / 'ucsd_pde' / 'converted'
    out.mkdir(exist_ok=True)
    with tarfile.open(tpath) as t:
        ms = [m for m in t.getmembers() if m.name.endswith(member_suffix)]
        if len(ms) != 1:
            sys.exit(f'{member_suffix}: {[m.name for m in ms]}')
        dst = out / Path(ms[0].name).name
        with open(dst, 'wb') as fo:
            fo.write(t.extractfile(ms[0]).read())
    write_manifest(out, [tpath], [dst], f'{ms[0].name} extracted from the tar unchanged')


CONVERTERS = {
    'poisson_gauss': lambda r: poisson_gauss(r),
    'dpde_poisson': lambda r: diffusionpde(r, 'poisson_hf'),
    'dpde_darcy': lambda r: diffusionpde(r, 'darcy_hf'),
    'dpde_helmholtz': lambda r: diffusionpde(r, 'helmholtz_hf'),
    'wave_gauss': lambda r: wave_gauss(r),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', help='converter name, or cno:<member.h5>[:regex], elado:<A1_test>, ucsd:<member suffix>')
    ap.add_argument('--data-root', default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    root = Path(a.data_root)
    if ':' in a.name:
        kind, arg = a.name.split(':', 1)
        {'cno': cno, 'elado': elado, 'ucsd': ucsd}[kind](root, arg)
    else:
        CONVERTERS[a.name](root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
