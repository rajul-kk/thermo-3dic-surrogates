"""Stress-test §9.16b's PDEBench-NS persistence finding beyond one file's 4 trajectories.

§9.16b measured "persistence beats a linear fit at short time gaps" on a single NS_incom
file (4 correlated trajectories, contiguous split) -- flagged there as an optimistic setup
for the linear-probe number. This script re-measures it with a leave-one-FILE-out split
across several independent PDEBench simulation runs, and finds the stride at which linear
catches persistence, to see whether the result is a candidate for a standalone note.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pde_benchmark_data import navier_stokes_multi_file
from scripts.benchmark_linearity_audit import _pca, _ridge, _with_intercept, _spatial_r2, LAMBDAS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('ns_robust')

FILE_KEYS_5 = ['ns_incom_0', 'ns_incom_10', 'ns_incom_11', 'ns_incom_13', 'ns_incom_14']
FILE_KEYS_25 = FILE_KEYS_5 + ['ns_incom_12'] + [f'ns_incom_1{n:02d}' for n in range(0, 20)]
FILE_KEYS = FILE_KEYS_25


def mean_baseline_r2(Ytr, Yte):
    pred = np.broadcast_to(Ytr.mean(0), Yte.shape)
    return _spatial_r2(pred, Yte)


def persistence_r2(Xte, Yte):
    return _spatial_r2(Xte, Yte)


def linear_r2(Xtr, Ytr, Xte, Yte, pca_grid=(2, 4, 8, 16, 32, 64)):
    """Same PCA-ridge search the audit uses, picking width/lambda on a held-out slice
    of the TRAINING groups only -- never touching the held-out file."""
    n = len(Xtr)
    cut = max(1, int(n * 0.8))
    idx = np.random.default_rng(0).permutation(n)
    tr_idx, val_idx = idx[:cut], idx[cut:]
    best = (-np.inf, None, None)
    for k in pca_grid:
        if k >= min(Xtr[tr_idx].shape[0], Xtr.shape[1]):
            continue
        mu, comp = _pca(Xtr[tr_idx], k)
        Ztr = _with_intercept((Xtr[tr_idx] - mu) @ comp.T)
        Zval = _with_intercept((Xtr[val_idx] - mu) @ comp.T)
        for lam in LAMBDAS:
            W = _ridge(Ztr, Ytr[tr_idx], lam)
            r2 = _spatial_r2(Zval @ W, Ytr[val_idx])
            if r2 > best[0]:
                best = (r2, k, lam)
    _, k, lam = best
    if k is None:
        return float('nan')
    mu, comp = _pca(Xtr, k)
    Ztr_full = _with_intercept((Xtr - mu) @ comp.T)
    W = _ridge(Ztr_full, Ytr, lam)
    Zte = _with_intercept((Xte - mu) @ comp.T)
    return _spatial_r2(Zte @ W, Yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strides', type=int, nargs='+', default=[2, 10, 25, 50, 100, 200, 400])
    ap.add_argument('--per-traj', type=int, default=6)
    ap.add_argument('--every', type=int, default=4)
    args = ap.parse_args()

    rows = []
    for stride in args.strides:
        log.info('=== stride %d ===', stride)
        X, Y, G = navier_stokes_multi_file(FILE_KEYS, per_traj=args.per_traj,
                                           stride=stride, every=args.every)
        groups = sorted(set(G.tolist()))
        fold_lin, fold_pers, fold_mean = [], [], []
        for g in groups:
            te = G == g
            tr = ~te
            fold_lin.append(linear_r2(X[tr], Y[tr], X[te], Y[te]))
            fold_pers.append(persistence_r2(X[te], Y[te]))
            fold_mean.append(mean_baseline_r2(Y[tr], Y[te]))
        row = {'stride': stride, 'n': len(X), 'n_groups': len(groups),
               'linear_r2_mean': float(np.mean(fold_lin)), 'linear_r2_std': float(np.std(fold_lin)),
               'persistence_r2_mean': float(np.mean(fold_pers)), 'persistence_r2_std': float(np.std(fold_pers)),
               'mean_r2_mean': float(np.mean(fold_mean))}
        rows.append(row)
        log.info('stride=%d n=%d groups=%d | linear %.4f+/-%.4f | persistence %.4f+/-%.4f | mean-field %.4f',
                 stride, row['n'], row['n_groups'], row['linear_r2_mean'], row['linear_r2_std'],
                 row['persistence_r2_mean'], row['persistence_r2_std'], row['mean_r2_mean'])

    print()
    print('=' * 100)
    print(f"{'stride':>7} {'n':>5} {'groups':>7} {'linear (leave-1-file-out)':>28} "
          f"{'persistence':>18} {'mean-field':>12} {'linear-persistence':>20}")
    print('=' * 100)
    for r in rows:
        gap = r['linear_r2_mean'] - r['persistence_r2_mean']
        print(f"{r['stride']:>7} {r['n']:>5} {r['n_groups']:>7} "
              f"{r['linear_r2_mean']:>14.4f}+/-{r['linear_r2_std']:<10.4f} "
              f"{r['persistence_r2_mean']:>18.4f} {r['mean_r2_mean']:>12.4f} {gap:>+20.4f}")

    out = Path('results/ns_persistence_robustness.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps({'file_keys': FILE_KEYS, 'per_traj': args.per_traj,
                               'every': args.every, 'rows': rows}, indent=2))
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
