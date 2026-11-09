"""Measure the in-plane orientation of IC-ThermBench input power vs output temperature, per scope.

Per sample: Pearson corr of gaussian-smoothed power (sigma 2 cells) with temperature, power as stored vs transposed
in-plane; and the maximum-principle test (hottest node is unpowered) for both. Usage:
  python scripts/icb_orientation_check.py [--root data/ic-thermbench/datasets] [--n 400] [--json out.json]
"""
import argparse, json
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


def check(folder, n):
    with h5py.File(folder / 'input.mat', 'r') as fi, h5py.File(folder / 'output.mat', 'r') as fo:
        P = np.asarray(fi['data'][:n, 0, 0], float)       # raw layout (B,P,Z,Y,X): channel 0 = power, Z=1
        T = np.asarray(fo['data'][:n, 0], float)          # (B,Z,Y,X)
    r = {'stored': [], 'transposed': []}
    v = {'stored': 0, 'transposed': 0}
    for p, t in zip(P, T):
        for name, q in (('stored', p), ('transposed', p.T)):
            r[name].append(np.corrcoef(gaussian_filter(q, 2).ravel(), t.ravel())[0, 1])
            pw = q > 0
            if pw.any() and not pw[np.unravel_index(t.argmax(), t.shape)] and t.max() - t[pw].max() > 0.01:
                v[name] += 1
    a, b = np.array(r['stored']), np.array(r['transposed'])
    return {'n': len(P), 'corr_stored_mean': float(a.mean()), 'corr_transposed_mean': float(b.mean()),
            'transposed_better': int((b > a).sum()), 'maxprinc_viol_stored': v['stored'],
            'maxprinc_viol_transposed': v['transposed'],
            'power_symmetric_samples': int(sum(np.array_equal(p, p.T) for p in P))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/ic-thermbench/datasets'))
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    out = {}
    for s in ('level2', 'level3', 'level4', 'level5'):
        out[s] = check(a.root / f'{s}_steady', a.n)
        print(s, out[s])
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1))
