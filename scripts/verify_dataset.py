"""Check that the dataset on disk is the one the results were computed on, and that 3D-ICE still produces it.

Two parts:
  fingerprint  sha256 of every file's coords and temp arrays, per geometry, written to a JSON file. Compare two
               fingerprints to see whether a regenerated dataset is the same data.
  resolve      re-solve a fixed sample of files per geometry from their own metadata (scripts/resolve_from_metadata.py)
               and compare the new temperatures with the stored ones on the same nodes.

Usage: python scripts/verify_dataset.py <data_root> --out results/3dice_version_check/dataset_verify.json [--per-geometry 2] [--seed 0]
       [--no-resolve] [--geometries geometry1 geometry4]
TSV-map files (geometry2a/5/6) need ICEFORGE_GUARD=0, which reproduces how they were made; the script sets it.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LIVE_GEOMETRIES = ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5', 'geometry6']
TOL_MAX_K = 0.01


def array_digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def fingerprint(data_root: Path, geometries) -> dict:
    out = {}
    for g in geometries:
        files = sorted((data_root / g).glob(f'{g}_*.npz'))
        per = {}
        for f in files:
            with np.load(f, allow_pickle=True) as d:
                per[f.name] = {'coords': array_digest(d['coords']), 'temp': array_digest(d['temp'])}
        whole = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
        out[g] = {'n_files': len(files), 'digest': whole, 'files': per}
    return out


def pick(files, n, seed):
    if len(files) <= n:
        return list(files)
    idx = np.random.default_rng(seed).choice(len(files), n, replace=False)
    return [files[i] for i in sorted(idx)]


def compare(stored: dict, new: dict) -> dict:
    """Temperatures on the same nodes: max and RMS difference, peak difference. Nodes are matched by exact coordinates
    after sorting; a different grid is reported, not interpolated."""
    c0, c1 = stored['coords'].astype(np.float64), new['coords'].astype(np.float64)
    if c0.shape != c1.shape:
        return {'same_grid': False, 'n_stored': len(c0), 'n_new': len(c1)}
    o0, o1 = np.lexsort(c0.T[::-1]), np.lexsort(c1.T[::-1])
    if np.abs(c0[o0] - c1[o1]).max() > 1e-3:
        return {'same_grid': False, 'n_stored': len(c0), 'n_new': len(c1)}
    t0, t1 = stored['temp'].astype(np.float64)[o0], new['temp'].astype(np.float64)[o1]
    d = t1 - t0
    return {'same_grid': True, 'max_abs_K': float(np.abs(d).max()), 'rms_K': float(np.sqrt((d ** 2).mean())),
            'peak_diff_K': float(t1.max() - t0.max()), 'within_tol': bool(np.abs(d).max() <= TOL_MAX_K)}


def resolve_sample(data_root: Path, geometries, per_geometry: int, seed: int) -> dict:
    import scripts.resolve_from_metadata as rfm

    os.environ['ICEFORGE_GUARD'] = '0'
    out = {}
    for g in geometries:
        files = pick(sorted((data_root / g).glob(f'{g}_*.npz')), per_geometry, seed)
        for f in files:
            td = Path(tempfile.mkdtemp())
            try:
                src, dst = td / 'src', td / 'dst'
                src.mkdir()
                shutil.copy2(f, src / f.name)
                t0 = time.time()
                rfm.main([str(src), '--out', str(dst)])
                with np.load(f, allow_pickle=True) as a, np.load(dst / f.name, allow_pickle=True) as b:
                    r = compare({'coords': a['coords'], 'temp': a['temp']}, {'coords': b['coords'], 'temp': b['temp']})
                    r['ice_commit'] = b['metadata'].item().get('ice_commit')
                r['seconds'] = round(time.time() - t0, 1)
            except Exception as e:  # noqa: BLE001  any failure is recorded per file and the run goes on
                r = {'error': f'{type(e).__name__}: {e}'}
            finally:
                shutil.rmtree(td, ignore_errors=True)
            out[f'{g}/{f.name}'] = r
            print(f.name, json.dumps(r), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--per-geometry', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-resolve', action='store_true')
    ap.add_argument('--geometries', nargs='+', default=LIVE_GEOMETRIES)
    a = ap.parse_args(argv)
    res = {'data_root': str(a.data_root), 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'fingerprint': fingerprint(a.data_root, a.geometries)}
    if not a.no_resolve:
        r = resolve_sample(a.data_root, a.geometries, a.per_geometry, a.seed)
        ok = [v for v in r.values() if v.get('same_grid')]
        res['resolve'] = {'tolerance_max_abs_K': TOL_MAX_K, 'per_geometry': a.per_geometry, 'seed': a.seed,
                          'n': len(r), 'n_same_grid': len(ok), 'n_within_tol': sum(v['within_tol'] for v in ok),
                          'files': r}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({g: {k: v[k] for k in ('n_files', 'digest')} for g, v in res['fingerprint'].items()}, indent=1))
    if 'resolve' in res:
        print(json.dumps({k: v for k, v in res['resolve'].items() if k != 'files'}))
    return res


if __name__ == '__main__':
    main()
