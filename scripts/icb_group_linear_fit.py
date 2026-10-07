"""Within-layout-group linear fit on IC-ThermBench S2 (report 9.29 'linearity floor'), orientation-aware.

The original ad hoc script was not committed, so this is a reconstruction: the layout group is the largest set of S2
samples sharing the same three chiplet footprint areas (and therefore the same fixed per-chiplet powers); features are
64 power components (8x8 average pool, or PCA-64 of the power map); least squares to the full temperature field with
an intercept; deterministic 80/20 split. Run with --as-stored and without to compare orientations.
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ic_thermbench_data import load_mat_pair, SCOPES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/_external/ictherm/datasets')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = {}
    for name, fix in (('as_stored', False), ('fixed', True)):
        x, y = load_mat_pair(Path(a.data_root) / 'level2_steady', fix_orientation=fix, channels=SCOPES['level2'])
        P = x[:, :, :, 0, 0].astype(np.float64)
        Y = y[:, :, :, 0].astype(np.float64)
        keys = [tuple(int((P[i] == v).sum()) for v in np.unique(P[i])[1:]) for i in range(len(P))]
        top, n = collections.Counter(keys).most_common(1)[0]
        idx = np.array([i for i, k in enumerate(keys) if k == top])
        rng = np.random.default_rng(0)
        idx = rng.permutation(idx)
        ntr = int(0.8 * len(idx))
        tr, te = idx[:ntr], idx[ntr:]
        out = {'group_size': int(n), 'areas': list(top), 'n_train': int(ntr), 'n_test': int(len(te)),
               'distinct_masks': len({(P[i] > 0).tobytes() for i in idx})}
        pool = P.reshape(len(P), 8, 8, 8, 8).mean((2, 4)).reshape(len(P), 64)
        flat = P.reshape(len(P), -1)
        mu = flat[tr].mean(0)
        _, _, vt = np.linalg.svd(flat[tr] - mu, full_matrices=False)
        pca = (flat - mu) @ vt[:64].T
        Yf = Y.reshape(len(Y), -1)
        for fname, F in (('pool8x8', pool), ('pca64', pca), ('full_ridge', None)):
            if F is None:
                F = flat
                A = np.c_[F[tr], np.ones(len(tr))]
                lam = 1e-3 * np.trace(A.T @ A) / A.shape[1]
                W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Yf[tr])
            else:
                A = np.c_[F[tr], np.ones(len(tr))]
                W = np.linalg.lstsq(A, Yf[tr], rcond=None)[0]
            pred = np.c_[F[te], np.ones(len(te))] @ W
            out[fname] = {'test_rmse_K': float(np.sqrt(((pred - Yf[te]) ** 2).mean())),
                          'train_rmse_K': float(np.sqrt(((np.c_[F[tr], np.ones(len(tr))] @ W - Yf[tr]) ** 2).mean()))}
        res[name] = out
        print(name, json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
