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


def _kfold(n: int, k: int, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(n)
    return [np.asarray(p) for p in np.array_split(idx, k)]


def audit(X: np.ndarray, Y: np.ndarray, name: str,
          test_frac: float = 0.2, n_folds: int = 5,
          X_test: np.ndarray = None, Y_test: np.ndarray = None) -> Dict[str, float]:
    """
    Run the audit on paired input/output fields. X: (N, P), Y: (N, Q), both flattened.
    Splits are contiguous (no shuffling) so a caller controls ordering.
    """
    t0 = time.time()
    # Transfer mode: fit entirely on X/Y, evaluate on a DIFFERENT dataset. Needed for
    # IC-ThermBench S5, which is an evaluation-only OOD scope -- auditing it in-distribution
    # (the first version of this script) measures a different and much easier question.
    if X_test is not None:
        Xtr, Ytr, Xte, Yte = X, Y, X_test, Y_test
    else:
        n = len(X)
        n_test = max(1, int(round(test_frac * n)))
        Xtr, Ytr = X[:n - n_test], Y[:n - n_test]
        Xte, Yte = X[n - n_test:], Y[n - n_test:]
    n_tr = len(Xtr)
    if n_tr < 8:
        raise ValueError(f'{name}: only {n_tr} training samples, too few to audit')

    dof = participation_ratio(Xtr)

    folds = _kfold(n_tr, min(n_folds, max(2, n_tr // 4)))

    def cv_select(make_feats):
        """Choose lambda by k-fold CV on the training split. Never touches test."""
        best = None
        for lam in LAMBDAS:
            scores = []
            for f in folds:
                tr_i = np.setdiff1d(np.arange(n_tr), f)
                Ftr, Fva = make_feats(tr_i, f)
                mu, sd = Ftr.mean(0), Ftr.std(0)
                sd[sd < 1e-12] = 1.0
                W = _ridge(_with_intercept((Ftr - mu) / sd), Ytr[tr_i], lam)
                scores.append(_spatial_r2(_with_intercept((Fva - mu) / sd) @ W, Ytr[f]))
            m = float(np.nanmean(scores))
            if best is None or m > best[0]:
                best = (m, lam)
        return best

    def fit_eval(feat_tr, feat_te, lam, tag):
        mu, sd = feat_tr.mean(0), feat_tr.std(0)
        sd[sd < 1e-12] = 1.0
        W = _ridge(_with_intercept((feat_tr - mu) / sd), Ytr, lam)
        pred = _with_intercept((feat_te - mu) / sd) @ W
        return {f'{tag}_spatial_r2': _spatial_r2(pred, Yte),
                f'{tag}_r2': _pooled_r2(pred, Yte),
                f'{tag}_rel_l2': _rel_l2(pred, Yte),
                f'{tag}_lambda': lam,
                f'{tag}_params': int(W.size)}

    # Spatial structure relative to field magnitude. This is why rel_l2 must NOT be compared
    # across benchmarks: thermal fields are ~325 K carrying ~3 K of structure (ratio ~0.01),
    # so a mean-only predictor already scores rel_l2 ~ 0.01, while the same predictor on Darcy
    # (ratio ~0.56) scores ~0.56. Detrended spatial R2 is the comparable quantity, and it is
    # what `linear_solvable` is thresholded on.
    _Yall = np.concatenate([Ytr, Yte]) if len(Yte) else Ytr
    _struct = float((_Yall - _Yall.mean(axis=1, keepdims=True)).std(axis=1).mean())
    _mag = float(abs(_Yall.mean())) or 1.0
    out: Dict[str, float] = {'n_samples': n_tr + len(Xte), 'n_train': n_tr,
                             'in_cells': Xtr.shape[1], 'out_cells': Ytr.shape[1],
                             'effective_dof': dof,
                             'structure_to_mean': _struct / _mag}

    # Dense operator: the exact solution form for a linear PDE with a fixed operator.
    #
    # Gated on three things, not just cell count. The Gram is in_cells^2 float64, so 16k cells
    # is 2.1 GB per fold per lambda -- PDEBench Darcy (16,384 cells) sat just under the old
    # 20,000-cell limit and exhausted 16 GB of RAM. It is also pointless when in_cells greatly
    # exceeds n_train: the system is rank-deficient and the ridge penalty, not the data, picks
    # the solution. The PCA path below is the honest estimator in that regime.
    gram_gb = (Xtr.shape[1] ** 2) * 8 / 1e9
    dense_ok = (Xtr.shape[1] <= 20000 and gram_gb <= 1.0
                and Xtr.shape[1] <= 4 * max(n_tr, 1))
    if dense_ok:
        _, lam_d = cv_select(lambda a, b: (Xtr[a], Xtr[b]))
        out.update(fit_eval(Xtr, Xte, lam_d, 'dense'))
    else:
        log.info('%s: %d input cells vs %d train samples (Gram %.1f GB), skipping dense fit',
                 name, Xtr.shape[1], n_tr, gram_gb)

    # PCA-restricted, with the WIDTH selected on validation. Sweeping k is what makes this
    # comparable across benchmarks whose sample counts differ by three orders of magnitude.
    best_pca = None
    for k in PCA_GRID:
        if k > max(2, n_tr // 3):
            break
        def mk(a, b, _k=k):
            mu_p, basis = _pca(Xtr[a], _k)
            return (Xtr[a] - mu_p) @ basis.T, (Xtr[b] - mu_p) @ basis.T
        cvscore, lam_k = cv_select(mk)
        if best_pca is None or cvscore > best_pca[0]:
            best_pca = (cvscore, k, lam_k)
    _, k_best, lam_k = best_pca
    mu_p, basis = _pca(Xtr, k_best)
    cand = fit_eval((Xtr - mu_p) @ basis.T, (Xte - mu_p) @ basis.T, lam_k, 'pca')
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


def _pde(fn: str, **kw) -> Tuple[np.ndarray, np.ndarray]:
    from scripts import pde_benchmark_data as P
    return getattr(P, fn)(**kw)


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


# Transfer pairs: fit on the source, score on the target without refitting. Mirrors
# IC-ThermBench's own S5 protocol (frozen S4 model, unchanged preprocessing).
TRANSFERS: Dict[str, Tuple[str, str]] = {
    'icthermbench/S4->S5': ('level4', 'level5'),
}

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
    # Shelf layouts (§9.15b): chiplets permuted along x with the slack redistributed, which
    # is the only randomisation that fits the densely packed packages.
    'ours/geometry4-shelf':     lambda: _ours('data/3d-ice-layout-geometry4', 'geometry4'),
    'ours/geometry5-shelf':     lambda: _ours('data/3d-ice-layout-geometry5', 'geometry5'),
    'ours/geometry6-shelf':     lambda: _ours('data/3d-ice-layout-geometry6', 'geometry6'),
    # Canonical operator-learning PDE benchmarks (PDEBench, arXiv:2210.07182). These are the
    # calibration points: what does the diagnostic say about benchmarks the field agrees are
    # discriminative? Darcy is the closest analogue to ours -- it varies the coefficient field
    # (i.e. the operator) per sample, which is exactly what our fixed-placement data lacked.
    'pdebench/darcy-beta1.0':  lambda: _pde('darcy', n=1000, beta='1.0'),
    'pdebench/darcy-beta0.01': lambda: _pde('darcy', n=1000, beta='0.01'),
    'pdebench/burgers-nu0.01': lambda: _pde('burgers', n=400, nu='0.01', t_out=20),
    'pdebench/burgers-final':  lambda: _pde('burgers', n=400, nu='0.01', t_out=200),
    'pdebench/navier-stokes':  lambda: _pde('navier_stokes', n_pairs=300, stride=2, every=4),
    # IC-ThermBench (§9.13).
    'icthermbench/S2': lambda: _icthermbench('level2'),
    'icthermbench/S3': lambda: _icthermbench('level3'),
    'icthermbench/S4': lambda: _icthermbench('level4'),
    # S5 is deliberately absent here: it is an evaluation-only OOD scope, so auditing it
    # in-distribution measures the wrong question. It is handled as a transfer pair below.
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

    for name, (src, tgt) in TRANSFERS.items():
        if args.datasets is not None and not any(name.startswith(p) for p in args.datasets):
            continue
        try:
            Xs, Ys = _icthermbench(src)
            Xt, Yt = _icthermbench(tgt)
        except Exception as exc:
            log.warning('%s: unavailable (%s)', name, exc)
            continue
        if len(Xs) > args.max_samples:
            Xs, Ys = Xs[:args.max_samples], Ys[:args.max_samples]
        if len(Xt) > args.max_samples:
            Xt, Yt = Xt[:args.max_samples], Yt[:args.max_samples]
        try:
            results[name] = audit(Xs, Ys, name, X_test=Xt, Y_test=Yt)
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
    print('verdict thresholds detrended spatial R2 > 0.95. `rel L2` is shown for reference but')
    print('is NOT comparable across benchmarks: it normalises by the field including its mean,')
    print('and structure/mean ranges from ~0.01 (thermal, 325 K fields with 3 K of structure)')
    print('to ~0.56 (Darcy). A mean-only predictor scores rel L2 ~= structure/mean.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Merge with whatever is already on disk rather than overwriting. Running a subset with
    # --datasets used to clobber the full table: a three-dataset shelf run destroyed an
    # eleven-row artifact, which was only noticed later because the console log still had it.
    merged = {}
    if args.output.exists():
        try:
            prev = json.loads(args.output.read_text(encoding='utf-8'))
            if isinstance(prev, dict):
                merged.update(prev)
        except Exception as exc:
            log.warning('could not read existing %s (%s); it will be replaced',
                        args.output, exc)
    merged.update(results if isinstance(results, dict) else {})
    with open(args.output, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'Saved: {args.output}  ({len(merged)} datasets on file, '
          f'{len(results)} from this run)')


if __name__ == '__main__':
    main()
