"""
Non-neural baselines on IC-ThermBench S2-S5.

See docs/ic_thermbench_plan.md. IC-ThermBench's eight baselines are all deep networks;
this supplies the missing closed-form row, scored with their own metric code
(`third_party/ic_thermbench/metrics.py`, vendored verbatim) on their own splits
(`scripts/ic_thermbench_data.py`, verified array-exact against their loader).

Baselines
---------
ridge-green   The physically-motivated linear model: a dense ridge operator from the
              4,096-cell power map (plus the per-cell conductivity map where the scope
              has one, plus per-sample scalars) to the 4,096-cell temperature field.
              Steady-state conduction IS linear in the source, so for a fixed operator
              this is the exact solution form, not an approximation:
                  T(x) = T_amb + sum_x' G(x,x') Q(x')
              The intercept absorbs any fixed offset. This is the number that matters.

ridge-pca     The same idea with the spatial maps compressed to their leading principal
              components. Far fewer parameters; guards against the objection that
              ridge-green only wins because it is large.

knn / nn      Inverse-distance-weighted / single nearest training field, in PCA feature
              space (raw 4,096-d distances over 10,800 training samples are needlessly
              slow and no more meaningful).

mean          Training-set mean field. The floor.

Design notes
------------
- Ridge lambda is selected on THEIR validation split, mirroring the validation-best
  checkpoint selection their models use. Not tuned on test.
- Features are standardised; the intercept is never penalised (same convention as
  scripts/baselines.py).
- `grid_x` / `grid_y` are per-cell geometry encodings that vary per sample (they carry
  layout identity). They are included as PCA components rather than raw 4,096-dim blocks
  by default: including them raw pushes the feature count past the training-set size,
  which is a different and less interpretable regime. `--raw-grid` opts into it.

Usage
-----
    python scripts/ic_thermbench_baselines.py --scope level2
    python scripts/ic_thermbench_baselines.py --scope level2 level3 level4 --output results/ic_thermbench_baselines.json
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ic_thermbench_data import (SCALAR_CHANNELS, load_scope, spatial_channel_indices,
                                        scalar_channel_indices)
from third_party.ic_thermbench.metrics import _compute_additional_test_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('icb_baselines')

LAMBDAS = [1e-4, 1e-2, 1.0, 1e2, 1e4]
REPORT_KEYS = ['rmse', 'mean_absolute_error', 'r2', 'max_absolute_error',
               'max_temperature_error', 'topk50_temperature_difference']


def flatten_fields(x: np.ndarray, ch_idx: int) -> np.ndarray:
    """(B, X, Y, Z, P) -> (B, X*Y*Z) for one channel."""
    return x[..., ch_idx].reshape(len(x), -1).astype(np.float64)


def scalar_values(x: np.ndarray, ch_idx: int) -> np.ndarray:
    """(B, X, Y, Z, P) -> (B,) for a channel that is constant within each sample."""
    return x[..., ch_idx].reshape(len(x), -1).mean(axis=1).astype(np.float64)


def build_scalar_block(x: np.ndarray, channels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Per-sample scalar features, including 1/h -- the term that enters T linearly."""
    cols, names = [], []
    for name, idx in scalar_channel_indices(channels).items():
        v = scalar_values(x, idx)
        cols.append(v)
        names.append(name)
        if name == 'h_w_m2k':
            with np.errstate(divide='ignore'):
                inv = np.where(v > 0, 1.0 / np.maximum(v, 1e-12), 0.0)
            cols.append(inv)
            names.append('inv_h')
    if not cols:
        return np.zeros((len(x), 0)), []
    return np.stack(cols, axis=1), names


def fit_pca(train_fields: np.ndarray, n_comp: int, seed: int = 0
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leading principal components of centred training fields. Returns (mean, basis).

    Randomised range-finder rather than a full SVD: we only ever want the leading
    ~256 directions of a 10,800 x 4,096 matrix, and a full economy SVD costs
    O(n*m*min(n,m)) ~ 1.8e11 flops against ~1.1e10 here. Uses one power iteration,
    which is ample for the strongly-decaying spectra these thermal fields have.
    """
    mu = train_fields.mean(axis=0)
    centred = train_fields - mu
    n_comp = min(n_comp, min(centred.shape) - 1)

    rng = np.random.default_rng(seed)
    oversample = min(centred.shape[1] - n_comp, 10)
    omega = rng.standard_normal((centred.shape[1], n_comp + oversample))
    y = centred @ omega                    # (n_samples, k)
    y = centred @ (centred.T @ y)          # one power iteration, sharpens the range
    q, _ = np.linalg.qr(y)                 # (n_samples, k) orthonormal
    b = q.T @ centred                      # (k, n_cells)
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    return mu, vt[:n_comp]


def pca_cache_get(cache: Dict[str, Tuple[np.ndarray, np.ndarray]], name: str,
                  fields: np.ndarray, n_comp: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit-once-per-channel PCA. ridge-green and ridge-pca share the same bases."""
    if name not in cache:
        t = time.time()
        cache[name] = fit_pca(fields, n_comp)
        log.info('  PCA[%s]: %d comps in %.1fs', name, cache[name][1].shape[0],
                 time.time() - t)
    return cache[name]


def ridge_gram(A: np.ndarray, Y: np.ndarray, chunk: int = 1500
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute the lambda-independent normal-equation blocks, in chunks.

    AtA and AtY are the expensive parts (O(n*p^2) and O(n*p*c)) and neither depends on
    lambda, so they are computed once and reused across the whole lambda sweep. Only the
    p x p solve repeats -- a 5x saving on the dominant cost for a 5-value sweep.

    Accumulated over row chunks so the float64 copy of A never exists in full. At S4's
    8,709 features a full float64 A is ~750 MB on top of everything else, and this
    machine has ~4 GB free; chunking keeps the transient at ~100 MB while giving
    bit-comparable results (the sum order changes, nothing else). A itself may be
    float32 -- the accumulation is done in float64 regardless, which is what matters
    for the conditioning of the normal equations.
    """
    t = time.time()
    p = A.shape[1]
    AtA = np.zeros((p, p), dtype=np.float64)
    AtY = np.zeros((p, Y.shape[1]), dtype=np.float64)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk].astype(np.float64, copy=False)
        AtA += a.T @ a
        AtY += a.T @ Y[i:i + chunk]
    log.info('  gram: AtA %s, AtY %s in %.1fs', AtA.shape, AtY.shape, time.time() - t)
    return AtA, AtY


def ridge_solve_gram(AtA: np.ndarray, AtY: np.ndarray, lam: float) -> np.ndarray:
    """Ridge from precomputed blocks, with an unpenalised trailing intercept column."""
    reg = lam * np.eye(AtA.shape[0])
    reg[-1, -1] = 0.0
    return np.linalg.solve(AtA + reg, AtY)


def standardise(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def with_intercept(z: np.ndarray) -> np.ndarray:
    # float32: these are the largest arrays in the run and only ever feed the chunked
    # float64 Gram accumulation or a prediction matmul. See ridge_gram's docstring.
    return np.hstack([z, np.ones((len(z), 1), dtype=z.dtype)]).astype(np.float32, copy=False)


def geometry_keys(x: np.ndarray, spatial: Dict[str, int]) -> np.ndarray:
    """
    A stable id per sample for its substrate geometry, from the grid_x/grid_y channels.

    These encode the (non-uniform) grid the case is discretised on, and measurement shows
    they take only ~20 distinct values across thousands of samples while the power map
    varies per sample. Hashing the pair therefore recovers the layout/case grouping that
    the released files do not label directly.
    """
    keys = []
    names = [n for n in ('grid_x', 'grid_y') if n in spatial]
    for i in range(len(x)):
        h = hashlib.md5()
        for n in names:
            h.update(np.ascontiguousarray(x[i, ..., spatial[n]]).tobytes())
        keys.append(h.hexdigest())
    return np.array(keys)


def score(pred: np.ndarray, true: np.ndarray, prefix: str) -> Dict[str, float]:
    """Their metric function, on (B, cells) arrays."""
    m = _compute_additional_test_metrics(pred.astype(np.float64),
                                         true.astype(np.float64), prefix)
    return {k.split('/', 1)[1]: v for k, v in m.items()}


class FittedRidge:
    """
    Everything needed to predict on new data with a fitted ridge-green model.

    Kept so S5 can be scored zero-shot: their S5 protocol freezes an S4-trained model
    and applies it to unseen cases with unchanged preprocessing statistics, so the PCA
    bases and standardisation must come from S4 and must NOT be refitted on S5.
    """

    def __init__(self, channels, spatial, raw_names, pca_names, pca_cache, mu, sd, W, lam):
        self.channels = channels
        self.spatial = spatial
        self.raw_names = raw_names
        self.pca_names = pca_names
        self.pca_cache = pca_cache
        self.mu, self.sd, self.W, self.lam = mu, sd, W, lam

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw = [flatten_fields(x, self.spatial[n]) for n in self.raw_names]
        pcs = []
        for n in self.pca_names:
            mu_p, basis = self.pca_cache[n]
            pcs.append((flatten_fields(x, self.spatial[n]) - mu_p) @ basis.T)
        sc, _ = build_scalar_block(x, self.channels)
        F = np.hstack(raw + pcs + [sc])
        return with_intercept((F - self.mu) / self.sd) @ self.W


def run_scope(data_root: Path, scope: str, n_pca: int, raw_grid: bool,
              knn_k: int, return_state: bool = False):
    t0 = time.time()
    split = load_scope(data_root, scope, split_data=(scope != 'level5'))
    log.info('%s', split.summary())

    if scope == 'level5':
        # Pure evaluation set: there is no S5 training data by construction, so an S5
        # row would need an S4-trained model. Handled by --transfer, not here.
        raise SystemExit('level5 is evaluation-only; use --transfer-from level4')

    y_tr = split.y_train.reshape(len(split.y_train), -1).astype(np.float64)
    y_te = split.y_test.reshape(len(split.y_test), -1).astype(np.float64)
    y_va = split.y_val.reshape(len(split.y_val), -1).astype(np.float64)

    spatial = spatial_channel_indices(split.channels)
    log.info('spatial channels: %s | scalar channels: %s',
             list(spatial), [c for c in split.channels if c in SCALAR_CHANNELS])

    # --- feature blocks -----------------------------------------------------------
    # power (and conductivity, where present) enter ridge-green raw: they are the
    # operator's actual arguments. grid_x/grid_y enter as PCA unless --raw-grid.
    raw_names = ['chiplet_power'] + (['local_thermal_k'] if 'local_thermal_k' in spatial else [])
    pca_names = [n for n in spatial if n not in raw_names]
    if raw_grid:
        raw_names += pca_names
        pca_names = []

    pca_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def blocks(x: np.ndarray, is_train: bool):
        raw = [flatten_fields(x, spatial[n]) for n in raw_names]
        pcs = []
        for n in pca_names:
            f = flatten_fields(x, spatial[n])
            if is_train:
                mu, basis = pca_cache_get(pca_cache, n, f, n_pca)
            else:
                mu, basis = pca_cache[n]
            pcs.append((f - mu) @ basis.T)
        sc, _ = build_scalar_block(x, split.channels)
        return raw, pcs, sc

    raw_tr, pcs_tr, sc_tr = blocks(split.x_train, True)
    raw_va, pcs_va, sc_va = blocks(split.x_val, False)
    raw_te, pcs_te, sc_te = blocks(split.x_test, False)

    F_tr = np.hstack(raw_tr + pcs_tr + [sc_tr])
    F_va = np.hstack(raw_va + pcs_va + [sc_va])
    F_te = np.hstack(raw_te + pcs_te + [sc_te])
    del raw_tr, raw_va, raw_te, pcs_tr, pcs_va, pcs_te
    log.info('ridge-green features: %d (raw %s, pca %s, scalars %d)',
             F_tr.shape[1], raw_names, pca_names, sc_tr.shape[1])

    results: Dict[str, Dict[str, float]] = {}

    # --- mean ---------------------------------------------------------------------
    mean_field = y_tr.mean(axis=0)
    results['mean'] = score(np.repeat(mean_field[None], len(y_te), axis=0), y_te, 'mean')
    results['mean']['n_params'] = int(mean_field.size)

    # --- ridge-green --------------------------------------------------------------
    mu, sd = standardise(F_tr)
    A_tr = with_intercept((F_tr - mu) / sd)
    A_va = with_intercept((F_va - mu) / sd)
    A_te = with_intercept((F_te - mu) / sd)

    AtA, AtY = ridge_gram(A_tr, y_tr)
    best = None
    for lam in LAMBDAS:
        W = ridge_solve_gram(AtA, AtY, lam)
        rmse_va = score(A_va @ W, y_va, 'v')['rmse']
        log.info('  ridge-green lambda=%-8g val rmse=%.4f', lam, rmse_va)
        if best is None or rmse_va < best[0]:
            best = (rmse_va, lam, W)
    _, lam_best, W = best
    del AtA, AtY
    results['ridge-green'] = score(A_te @ W, y_te, 'r')
    results['ridge-green']['lambda'] = lam_best
    results['ridge-green']['n_params'] = int(W.size)
    log.info('ridge-green: lambda=%g params=%d test rmse=%.4f',
             lam_best, W.size, results['ridge-green']['rmse'])

    # --- ridge-pca (all spatial maps compressed) ----------------------------------
    pca_all_tr, pca_all_va, pca_all_te = [], [], []
    for n in spatial:
        f_tr = flatten_fields(split.x_train, spatial[n])
        m2, b2 = pca_cache_get(pca_cache, n, f_tr, n_pca)   # reuses ridge-green's fits
        pca_all_tr.append((f_tr - m2) @ b2.T)
        pca_all_va.append((flatten_fields(split.x_val, spatial[n]) - m2) @ b2.T)
        pca_all_te.append((flatten_fields(split.x_test, spatial[n]) - m2) @ b2.T)
    P_tr = np.hstack(pca_all_tr + [sc_tr])
    P_va = np.hstack(pca_all_va + [sc_va])
    P_te = np.hstack(pca_all_te + [sc_te])
    mu2, sd2 = standardise(P_tr)
    B_tr, B_va, B_te = (with_intercept((M - mu2) / sd2) for M in (P_tr, P_va, P_te))

    BtB, BtY = ridge_gram(B_tr, y_tr)
    best = None
    for lam in LAMBDAS:
        W2 = ridge_solve_gram(BtB, BtY, lam)
        rmse_va = score(B_va @ W2, y_va, 'v')['rmse']
        log.info('  ridge-pca   lambda=%-8g val rmse=%.4f', lam, rmse_va)
        if best is None or rmse_va < best[0]:
            best = (rmse_va, lam, W2)
    _, lam2, W2 = best
    del BtB, BtY
    results['ridge-pca'] = score(B_te @ W2, y_te, 'r')
    results['ridge-pca']['lambda'] = lam2
    results['ridge-pca']['n_params'] = int(W2.size)
    log.info('ridge-pca: lambda=%g params=%d test rmse=%.4f',
             lam2, W2.size, results['ridge-pca']['rmse'])

    # --- ridge-per-geom -----------------------------------------------------------
    # Mechanism test. Steady-state conduction is exactly linear in the source for a
    # FIXED operator: T = T_amb + G.Q. But G depends on the geometry, and these scopes
    # vary layout. Measured on level2: the (grid_x, grid_y) pair takes only ~20 distinct
    # values across thousands of samples while the power map varies essentially per
    # sample -- i.e. ~20 discrete operators, each driven by a continuously varying source.
    #
    # So a single global linear map is mis-specified by construction. Fitting one ridge
    # per geometry group tests exactly that: if it closes most of the gap, the physics is
    # linear *within* a layout and what the neural models are buying is the ability to
    # condition on layout, not nonlinearity in the source.
    geom_tr = geometry_keys(split.x_train, spatial)
    geom_te = geometry_keys(split.x_test, spatial)
    groups = {}
    for i, g in enumerate(geom_tr):
        groups.setdefault(g, []).append(i)
    log.info('geometry groups: %d in train (sizes %s...)', len(groups),
             sorted((len(v) for v in groups.values()), reverse=True)[:5])

    # Power (+ conductivity) as PCA features, so each group's system is well-determined.
    per_geom_blocks = [pca_cache_get(pca_cache, n, flatten_fields(split.x_train, spatial[n]),
                                     n_pca)
                       for n in raw_names]
    def pg_feat(x):
        cols = []
        for n, (m_, b_) in zip(raw_names, per_geom_blocks):
            cols.append((flatten_fields(x, spatial[n]) - m_) @ b_.T)
        sc_, _ = build_scalar_block(x, split.channels)
        return np.hstack(cols + [sc_])

    Fg_tr, Fg_te = pg_feat(split.x_train), pg_feat(split.x_test)
    pred_pg = np.zeros_like(y_te)
    global_mu, global_sd = standardise(Fg_tr)
    Ag_tr = with_intercept((Fg_tr - global_mu) / global_sd)
    W_glob = ridge_solve_gram(*ridge_gram(Ag_tr, y_tr), 1.0)
    n_params_pg, n_fallback = 0, 0
    for g, idx in groups.items():
        te_idx = np.where(geom_te == g)[0]
        if len(te_idx) == 0:
            continue
        Fi = Fg_tr[idx]
        mu_i, sd_i = standardise(Fi)
        Ai = with_intercept((Fi - mu_i) / sd_i)
        Wi = ridge_solve_gram(*ridge_gram(Ai, y_tr[idx]), 1.0)
        n_params_pg += int(Wi.size)
        Ate = with_intercept((Fg_te[te_idx] - mu_i) / sd_i)
        pred_pg[te_idx] = Ate @ Wi
    unseen = np.array([i for i, g in enumerate(geom_te) if g not in groups])
    if len(unseen):
        n_fallback = len(unseen)
        pred_pg[unseen] = with_intercept((Fg_te[unseen] - global_mu) / global_sd) @ W_glob
    results['ridge-per-geom'] = score(pred_pg, y_te, 'pg')
    results['ridge-per-geom']['n_params'] = n_params_pg
    results['ridge-per-geom']['n_groups'] = len(groups)
    results['ridge-per-geom']['n_test_fallback'] = n_fallback
    log.info('ridge-per-geom: %d groups, %d params, %d test samples fell back to global, '
             'test rmse=%.4f', len(groups), n_params_pg, n_fallback,
             results['ridge-per-geom']['rmse'])

    # --- nn / knn in PCA feature space --------------------------------------------
    Z_tr = (P_tr - mu2) / sd2
    Z_te = (P_te - mu2) / sd2
    d2 = (
        (Z_te ** 2).sum(1)[:, None] - 2 * Z_te @ Z_tr.T + (Z_tr ** 2).sum(1)[None, :]
    )
    nn_idx = np.argmin(d2, axis=1)
    results['nn'] = score(y_tr[nn_idx], y_te, 'nn')
    results['nn']['n_params'] = 0

    k = min(knn_k, len(y_tr))
    top = np.argpartition(d2, k, axis=1)[:, :k]
    dist = np.sqrt(np.maximum(np.take_along_axis(d2, top, axis=1), 0.0))
    w = 1.0 / np.maximum(dist, 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    knn_pred = np.einsum('bk,bkc->bc', w, y_tr[top])
    results['knn'] = score(knn_pred, y_te, 'knn')
    results['knn']['n_params'] = 0

    log.info('%s done in %.1f min', scope, (time.time() - t0) / 60)
    if return_state:
        state = FittedRidge(split.channels, spatial, raw_names, pca_names,
                            pca_cache, mu, sd, W, lam_best)
        # ridge-pca as a second transferable model. On out-of-distribution scopes the
        # dense raw-feature operator is numerically hopeless -- standardising unseen
        # inputs by training statistics sends rarely-varying cells to huge z-scores,
        # which a 4,096-column operator then amplifies. The PCA variant projects onto
        # directions that actually carry training variance, so it degrades rather than
        # explodes. Both are reported: the blowup is real and worth showing, but the
        # PCA model is the linear baseline's honest best effort out of distribution.
        state_pca = FittedRidge(split.channels, spatial, [], list(spatial),
                                pca_cache, mu2, sd2, W2, lam2)
        return results, state, state_pca, mean_field
    return results


def run_transfer(data_root: Path, source: str, target: str, n_pca: int,
                 raw_grid: bool, knn_k: int) -> Dict[str, Dict[str, float]]:
    """
    Zero-shot transfer: fit on `source`, evaluate on `target` without refitting.

    Mirrors their S5 protocol -- a frozen source-trained model, unchanged preprocessing
    statistics, scored on all target samples.
    """
    log.info('=== transfer %s -> %s (zero-shot) ===', source, target)
    _, state, state_pca, mean_field = run_scope(data_root, source, n_pca, raw_grid,
                                                knn_k, return_state=True)

    tgt = load_scope(data_root, target, split_data=False)
    log.info('%s', tgt.summary())
    if tgt.channels != state.channels:
        raise ValueError(f'channel mismatch: {source}={state.channels} '
                         f'{target}={tgt.channels}')

    y = tgt.y_test.reshape(len(tgt.y_test), -1).astype(np.float64)
    out = {
        'ridge-green': score(state.predict(tgt.x_test), y, 'r'),
        'ridge-pca':   score(state_pca.predict(tgt.x_test), y, 'rp'),
        'mean':        score(np.repeat(mean_field[None], len(y), axis=0), y, 'm'),
    }
    out['ridge-green']['lambda'] = state.lam
    out['ridge-green']['n_params'] = int(state.W.size)
    out['ridge-pca']['lambda'] = state_pca.lam
    out['ridge-pca']['n_params'] = int(state_pca.W.size)
    out['mean']['n_params'] = int(mean_field.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', nargs='+', default=['level2'])
    ap.add_argument('--data-root', type=Path, default=Path('data/ic-thermbench/datasets'))
    ap.add_argument('--n-pca', type=int, default=256)
    ap.add_argument('--knn-k', type=int, default=3)
    ap.add_argument('--raw-grid', action='store_true',
                    help='give ridge-green grid_x/grid_y raw rather than as PCA components')
    ap.add_argument('--transfer', nargs=2, metavar=('SOURCE', 'TARGET'), default=None,
                    help='zero-shot transfer, e.g. --transfer level4 level5 (their S5 protocol)')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    if args.transfer:
        src, tgt = args.transfer
        res = run_transfer(args.data_root, src, tgt, args.n_pca, args.raw_grid, args.knn_k)
        print(f'\n=== {src} -> {tgt} zero-shot (their metrics) ===')
        print('  published S5 zero-shot: Therm-FM 15.51 RMSE (best), U-Net 19.10 (runner-up)')
        print(f'{"baseline":<14}' + ''.join(f'{k[:13]:>15}' for k in REPORT_KEYS))
        for name, m in res.items():
            print(f'{name:<14}' + ''.join(f'{m[k]:>15.4f}' for k in REPORT_KEYS))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump({f'{src}->{tgt}': res}, f, indent=2)
            print(f'\nSaved: {args.output}')
        return

    published = {
        'level2': ('Therm-FM', 0.4427, 'SAU-FNO', 0.7028),
        'level3': ('Therm-FM', 0.7161, 'U-FNO', 0.8016),
        'level4': ('Therm-FM', 0.9334, 'SAU-FNO', 1.2158),
    }

    all_out = {}
    for scope in args.scope:
        res = run_scope(args.data_root, scope, args.n_pca, args.raw_grid, args.knn_k)
        all_out[scope] = res

        print(f'\n=== {scope} (test split, their metrics) ===')
        if scope in published:
            b, bv, r, rv = published[scope]
            print(f'  published: {b} {bv:.4f} RMSE (best), {r} {rv:.4f} (runner-up)')
        hdr = f'{"baseline":<14}' + ''.join(f'{k[:13]:>15}' for k in REPORT_KEYS) + f'{"params":>12}'
        print(hdr)
        for name, m in res.items():
            row = f'{name:<14}' + ''.join(f'{m[k]:>15.4f}' for k in REPORT_KEYS)
            row += f'{int(m["n_params"]):>12,}'
            print(row)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(all_out, f, indent=2)
    if args.output:
        print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
