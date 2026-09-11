"""Is this PDE-surrogate benchmark linear-solvable? A portable diagnostic."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('audit')

LAMBDAS = [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4]
MAX_PCA = 256
# PCA width is selected on validation, not fixed. Fixing it broke the first version of this
# audit: 256 components fitted from ~30 training samples is degenerate, and it reported
# geometry1-fixed at linear R^2 0.246 when that dataset's established, repeatedly verified
# value is 0.970. The feature width a linear model gets must scale with the data available
# to fit it, or the audit measures sample size rather than linear solvability.
PCA_GRID = [2, 4, 8, 16, 32, 64, 128, 256]


# ── core diagnostic ────────────────────────────────────────────────────────────────────

def participation_ratio(fields: np.ndarray) -> float:
    """Effective number of degrees of freedom in an ensemble of input fields."""
    centred = fields - fields.mean(0)
    n = min(len(centred), MAX_PCA)
    # Eigenvalues of the sample covariance via the small Gram matrix.
    g = centred @ centred.T / max(len(centred) - 1, 1)
    ev = np.linalg.eigvalsh(g)[::-1][:n]
    ev = ev[ev > 0]
    if ev.size == 0:
        return 1.0
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def _pca(train: np.ndarray, n_comp: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    mu = train.mean(0)
    c = train - mu
    k = min(n_comp, min(c.shape) - 1)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((c.shape[1], k + min(10, c.shape[1] - k)))
    y = c @ omega
    y = c @ (c.T @ y)
    q, _ = np.linalg.qr(y)
    _, _, vt = np.linalg.svd(q.T @ c, full_matrices=False)
    return mu, vt[:k]


def _ridge(A: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ Y)


def _with_intercept(z):
    return np.hstack([z, np.ones((len(z), 1))])


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _spatial_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Detrended (per-sample mean removed) R^2, averaged over samples."""
    vals = []
    for p, t in zip(pred, true):
        dp, dt = p - p.mean(), t - t.mean()
        ss_tot = float((dt ** 2).sum())
        if ss_tot > 0:
            vals.append(1.0 - float(((dp - dt) ** 2).sum()) / ss_tot)
    return float(np.mean(vals)) if vals else float('nan')


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """Standard neural-operator metric: per-sample relative L2, averaged."""
    num = np.linalg.norm(pred - true, axis=1)
    den = np.maximum(np.linalg.norm(true, axis=1), 1e-12)
    return float((num / den).mean())


def audit(X: np.ndarray, Y: np.ndarray, name: str,
          val_frac: float = 0.1, test_frac: float = 0.2) -> Dict[str, float]:
    """
    Run the audit on paired input/output fields. X: (N, P), Y: (N, Q), both flattened.
    Splits are contiguous (no shuffling) so a caller controls ordering.
    """
    t0 = time.time()
    n = len(X)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * (n - n_test))))
    n_tr = n - n_test - n_val
    if n_tr < 4:
        raise ValueError(f'{name}: only {n} samples, too few to audit')

    Xtr, Xva, Xte = X[:n_tr], X[n_tr:n_tr + n_val], X[n_tr + n_val:]
    Ytr, Yva, Yte = Y[:n_tr], Y[n_tr:n_tr + n_val], Y[n_tr + n_val:]

    dof = participation_ratio(Xtr)

    def fit_eval(feat_tr, feat_va, feat_te, tag):
        mu, sd = feat_tr.mean(0), feat_tr.std(0)
        sd[sd < 1e-12] = 1.0
        A = _with_intercept((feat_tr - mu) / sd)
        Av = _with_intercept((feat_va - mu) / sd)
        At = _with_intercept((feat_te - mu) / sd)
        best = None
        for lam in LAMBDAS:
            W = _ridge(A, Ytr, lam)
            score = _spatial_r2(Av @ W, Yva)
            if best is None or score > best[0]:
                best = (score, lam, W)
        _, lam, W = best
        pred = At @ W
        return {f'{tag}_spatial_r2': _spatial_r2(pred, Yte),
                f'{tag}_r2': _pooled_r2(pred, Yte),
                f'{tag}_rel_l2': _rel_l2(pred, Yte),
                f'{tag}_lambda': lam,
                f'{tag}_params': int(W.size)}

    out: Dict[str, float] = {'n_samples': n, 'n_train': n_tr,
                             'in_cells': X.shape[1], 'out_cells': Y.shape[1],
                             'effective_dof': dof}

    # Dense operator: the exact solution form for a linear PDE with a fixed operator.
    if X.shape[1] <= 20000:
        out.update(fit_eval(Xtr, Xva, Xte, 'dense'))
    else:
        log.info('%s: %d input cells, skipping dense fit', name, X.shape[1])

    # PCA-restricted, with the WIDTH selected on validation. Sweeping k is what makes this
    # comparable across benchmarks whose sample counts differ by three orders of magnitude.
    best_pca = None
    for k in PCA_GRID:
        if k > max(2, n_tr // 3):
            break
        mu_p, basis = _pca(Xtr, k)
        proj = lambda M: (M - mu_p) @ basis.T
        cand = fit_eval(proj(Xtr), proj(Xva), proj(Xte), 'pca')
        mu2, sd2 = proj(Xtr).mean(0), proj(Xtr).std(0)
        sd2[sd2 < 1e-12] = 1.0
        Wv = _ridge(_with_intercept((proj(Xtr) - mu2) / sd2), Ytr, cand['pca_lambda'])
        vscore = _spatial_r2(_with_intercept((proj(Xva) - mu2) / sd2) @ Wv, Yva)
        if best_pca is None or vscore > best_pca[0]:
            best_pca = (vscore, k, cand)
    _, k_best, cand = best_pca
    cand['pca_k'] = k_best
    out.update(cand)

    mean_field = Ytr.mean(0)
    out['mean_spatial_r2'] = _spatial_r2(np.repeat(mean_field[None], len(Yte), 0), Yte)
    out['mean_r2'] = _pooled_r2(np.repeat(mean_field[None], len(Yte), 0), Yte)
    out['mean_rel_l2'] = _rel_l2(np.repeat(mean_field[None], len(Yte), 0), Yte)
    out['seconds'] = time.time() - t0

    best_r2 = max(out.get('dense_spatial_r2', -np.inf), out['pca_spatial_r2'])
    out['best_linear_spatial_r2'] = best_r2
    out['linear_solvable'] = bool(best_r2 > 0.95)
    log.info('%-34s DOF=%7.1f  pca_k=%4s  linear spatialR2=%7.4f  relL2=%6.3f  %s',
             name, dof, out.get('pca_k'), best_r2,
             min(out.get('dense_rel_l2', np.inf), out['pca_rel_l2']),
             'LINEAR-SOLVABLE' if out['linear_solvable'] else '')
    return out


# ── dataset adapters ───────────────────────────────────────────────────────────────────

def _ours(root: str, geom: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = sorted(glob.glob(f'{root}/**/{geom}_*.npz', recursive=True))
    if not fs:
        raise FileNotFoundError(f'{root}/{geom}')
    X, Y = [], []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        X.append(d['power'].ravel())
        Y.append(d['temp'].ravel())
    return np.asarray(X, np.float64), np.asarray(Y, np.float64)


def _icthermbench(scope: str) -> Tuple[np.ndarray, np.ndarray]:
    from scripts.ic_thermbench_data import load_scope, spatial_channel_indices
    s = load_scope(Path('data/ic-thermbench/datasets'), scope,
                   split_data=(scope != 'level5'))
    x = s.x_test if scope == 'level5' else np.concatenate([s.x_train, s.x_val, s.x_test])
    y = s.y_test if scope == 'level5' else np.concatenate([s.y_train, s.y_val, s.y_test])
    sp = spatial_channel_indices(s.channels)
    # Power map is the source; it is the field a linear operator would act on.
    X = x[..., sp['chiplet_power']].reshape(len(x), -1)
    return X.astype(np.float64), y.reshape(len(y), -1).astype(np.float64)


REGISTRY: Dict[str, Callable[[], Tuple[np.ndarray, np.ndarray]]] = {
    # This project, fixed placement (the regime §9.14 diagnoses).
    'ours/geometry1-fixed':   lambda: _ours('data/3d-ice', 'geometry1'),
    'ours/geometry4-fixed':   lambda: _ours('data/3d-ice', 'geometry4'),
    'ours/geometry6-fixed':   lambda: _ours('data/3d-ice', 'geometry6'),
    'ours/geometry7-pilot':   lambda: _ours('data/3d-ice-geometry7-pilot', 'geometry7'),
    # This project, placement varied (§9.15).
    'ours/geometry1-translate': lambda: _ours('data/3d-ice-moving-geometry1', 'geometry1'),
    'ours/geometry4-translate': lambda: _ours('data/3d-ice-moving-geometry4', 'geometry4'),
    'ours/geometry1-layout':    lambda: _ours('data/3d-ice-layout-geometry1', 'geometry1'),
    # IC-ThermBench (§9.13).
    'icthermbench/S2': lambda: _icthermbench('level2'),
    'icthermbench/S3': lambda: _icthermbench('level3'),
    'icthermbench/S4': lambda: _icthermbench('level4'),
    'icthermbench/S5': lambda: _icthermbench('level5'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='names or prefixes from the registry (default: all available)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--max-samples', type=int, default=3000,
                    help='cap per dataset, for the large benchmarks')
    ap.add_argument('--output', type=Path, default=Path('results/linearity_audit.json'))
    args = ap.parse_args()

    if args.list:
        for k in REGISTRY:
            print(' ', k)
        return

    wanted = [k for k in REGISTRY
              if args.datasets is None or any(k.startswith(p) for p in args.datasets)]
    results = {}
    for name in wanted:
        try:
            X, Y = REGISTRY[name]()
        except Exception as exc:
            log.warning('%s: unavailable (%s)', name, exc)
            continue
        if len(X) > args.max_samples:
            X, Y = X[:args.max_samples], Y[:args.max_samples]
        try:
            results[name] = audit(X, Y, name)
        except Exception as exc:
            log.error('%s: audit failed (%s)', name, exc)

    print('\n' + '=' * 104)
    print(f'{"benchmark":<30}{"N":>6}{"pca k":>7}{"eff. DOF":>10}'
          f'{"lin spatR2":>12}{"rel L2":>9}{"mean spatR2":>13}{"verdict":>16}')
    print('-' * 104)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]['effective_dof']):
        verdict = 'LINEAR-SOLVABLE' if r['linear_solvable'] else 'discriminative'
        print(f'{name:<30}{r["n_samples"]:>6}{r.get("pca_k",0):>7}{r["effective_dof"]:>10.1f}'
              f'{r["best_linear_spatial_r2"]:>12.4f}'
              f'{min(r.get("dense_rel_l2", np.inf), r["pca_rel_l2"]):>9.3f}'
              f'{r["mean_spatial_r2"]:>13.3f}{verdict:>16}')
    print('=' * 104)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
