"""Test all 8 dihedral orientations of IC-ThermBench power (and k) against the stored temperature, per split.

Extends icb_orientation_check.py (stored vs transpose only) to the full dihedral group of the square grid. Per orientation:
  corr_mean   mean Pearson corr of gaussian-smoothed (sigma 2) power with temperature
  maxprinc    samples whose hottest node is unpowered by > 0.01 K (maximum-principle violation)
  f008_r2     median fieldlint F008 operator-residual R^2 (s ~ aL(u)+b(u-c)); the same transform is applied to source and k
  n_identical_to_stored / n_symmetric_ties: how often an orientation gives an array equal to the stored one (cannot be told apart)
Usage: python scripts/icb_orientation_dihedral.py [--root data/_external/ictherm/datasets] [--n 400] [--n-f008 200]
"""
import argparse, json, sys
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'fieldlint' / 'src'))
from fieldlint.rules import operator_r2  # noqa: E402

# (y, x) arrays. 'transpose' is the current fix (p.T).
D8 = {
    'identity': lambda a: a,
    'transpose': lambda a: a.T,
    'flip_x': lambda a: a[:, ::-1],
    'flip_y': lambda a: a[::-1, :],
    'rot90': lambda a: np.rot90(a, 1),
    'rot180': lambda a: np.rot90(a, 2),
    'rot270': lambda a: np.rot90(a, 3),
    'anti_transpose': lambda a: a[::-1, ::-1].T,
}


def run(folder, n, n_f008):
    with h5py.File(folder / 'input.mat', 'r') as fi, h5py.File(folder / 'output.mat', 'r') as fo:
        nch = fi['data'].shape[1]
        P = np.asarray(fi['data'][:n, 0, 0], float)
        K = np.asarray(fi['data'][:n, 3, 0], float) if nch >= 4 else None
        T = np.asarray(fo['data'][:n, 0], float)
    out = {}
    for name, f in D8.items():
        corr, viol, r2s, same = [], 0, [], 0
        for i in range(len(P)):
            p, t = f(P[i]), T[i]
            corr.append(np.corrcoef(gaussian_filter(p, 2).ravel(), t.ravel())[0, 1])
            pw = p > 0
            if pw.any() and not pw[np.unravel_index(t.argmax(), t.shape)] and t.max() - t[pw].max() > 0.01:
                viol += 1
            same += int(np.array_equal(p, P[i]))
            if i < n_f008:
                k = None if K is None else f(K[i])
                r = operator_r2(p, k, t)
                if r is not None:
                    r2s.append(r)
        out[name] = {'corr_mean': float(np.mean(corr)), 'maxprinc_viol': viol,
                     'f008_r2_median': float(np.median(r2s)) if r2s else None,
                     'n_identical_to_stored': same}
    out['_meta'] = {'n': len(P), 'n_f008': min(n_f008, len(P)), 'k_used': K is not None}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/_external/ictherm/datasets'))
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--n-f008', type=int, default=200)
    ap.add_argument('--json', type=Path, default=Path('results/icb_orientation/dihedral.json'))
    a = ap.parse_args()
    res = {}
    for s in ('level2', 'level3', 'level4', 'level5'):
        res[s] = run(a.root / f'{s}_steady', a.n, a.n_f008)
        print(s)
        for k, v in res[s].items():
            print(' ', k, v)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1))
