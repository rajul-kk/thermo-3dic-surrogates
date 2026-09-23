"""Non-neural baselines for the 3D-IC thermal benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
_log = logging.getLogger('baselines')


# ── Data loading ───────────────────────────────────────────────────────────────

def load_scenario(path: Path) -> dict:
    """Load one NPZ into a plain dict (coords, temp, power, metadata)."""
    d = np.load(path, allow_pickle=True)
    return {
        'name':   path.stem,
        'coords': d['coords'].astype(np.float64),
        'temp':   d['temp'].astype(np.float64),
        'power':  d['power'].astype(np.float64),
        'meta':   d['metadata'].item(),
    }


def power_pca_features(train: List[dict], test: List[dict], n_comp: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Project each scenario's full power field onto its leading principal components."""
    P_tr = np.stack([sc['power'] for sc in train])
    mu = P_tr.mean(0)
    # Economy SVD of the centred training fields; components are rows of Vt.
    _, _, Vt = np.linalg.svd(P_tr - mu, full_matrices=False)
    basis = Vt[:min(n_comp, Vt.shape[0])]
    proj = lambda S: np.stack([(sc['power'] - mu) @ basis.T for sc in S])
    return proj(train), proj(test)


def feature_vector(meta: dict, block_keys: List[str],
                   pos_keys: Optional[List[str]] = None) -> np.ndarray:
    """Build the scenario parameter vector."""
    htc = float(meta.get('htc', 0.0))
    feats = [float(meta.get(k, 0.0)) for k in block_keys]
    for k in (pos_keys or []):
        feats.append(float(meta.get(k, 0.0)))
    feats.append(htc)
    feats.append(1.0 / htc if htc > 0 else 0.0)
    feats.append(float(meta.get('t_ambient_kelvin', 298.15)))
    feats.append(float(meta.get('tsv_density', 0.0)))
    return np.asarray(feats, dtype=np.float64)


def collect_position_keys(scenarios: List[dict]) -> List[str]:
    """Union of per-block position metadata keys, sorted for determinism."""
    keys = set()
    for sc in scenarios:
        keys.update(k for k in sc['meta']
                    if k.startswith('block_x_') or k.startswith('block_y_'))
    return sorted(keys)


def collect_block_keys(scenarios: List[dict]) -> List[str]:
    """Union of block-power metadata keys, sorted for determinism."""
    uses_nominal = any(sc['meta'].get('throttle_enabled')
                       or sc['meta'].get('leakage_enabled') for sc in scenarios)
    prefix = 'nominal_block_power_' if uses_nominal else 'block_power_'
    keys = set()
    for sc in scenarios:
        keys.update(k for k in sc['meta'] if k.startswith(prefix))
    if uses_nominal and not keys:
        # throttle/leakage enabled but no nominal_block_power_* present -- an
        # older export predating that field. Fail loud rather than silently
        # falling back to the misleading delivered-power feature.
        raise ValueError(
            "Scenario(s) have throttle_enabled or leakage_enabled=True but no "
            "nominal_block_power_* metadata (pre-dates that export field). "
            "Re-export this data before running baselines on it."
        )
    return sorted(keys)


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(pred: np.ndarray, true: np.ndarray, coords: np.ndarray) -> Dict[str, float]:
    """Raw AND spatially-detrended error."""
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    d_pred, d_true = pred - pred.mean(), true - true.mean()
    d_err = d_pred - d_true
    d_ss_tot = float(np.sum(d_true ** 2))

    i_pred, i_true = int(np.argmax(pred)), int(np.argmax(true))
    loc_err = float(np.linalg.norm(coords[i_pred] - coords[i_true]))

    return {
        'mae_K':              float(np.mean(np.abs(err))),
        'rmse_K':             float(np.sqrt(np.mean(err ** 2))),
        'max_abs_err_K':      float(np.max(np.abs(err))),
        'r2':                 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
        'mae_detrended_K':    float(np.mean(np.abs(d_err))),
        'spatial_r2':         1.0 - float(np.sum(d_err ** 2)) / d_ss_tot
                              if d_ss_tot > 0 else float('nan'),
        'true_spatial_std_K': float(true.std()),
        'hotspot_temp_err_K': float(pred[i_true] - true[i_true]),
        'hotspot_loc_err_um': loc_err,
    }


def aggregate(per_scenario: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean over scenarios, plus worst-case MAE."""
    if not per_scenario:
        return {}
    out = {k: float(np.mean([m[k] for m in per_scenario])) for k in per_scenario[0]}
    out['worst_mae_K'] = float(np.max([m['mae_K'] for m in per_scenario]))
    return out


# ── Baselines ──────────────────────────────────────────────────────────────────

# Linear-on-field baseline settings, copied from the Sec 9.15d notebook
# (kaggle_geometry4_vs_geometry6_fno.ipynb::linear_field_cv) so the numbers are directly
# comparable to the 0.941 / 0.513 it reported. Note lam penalises the intercept here,
# unlike the ridge baseline below which deliberately does not -- kept for fidelity.
FIELD_PCA_K = 8
FIELD_LAMBDA = 1e-2
FIELD_PCA_GRID = (4, 8, 16, 24, 32)       # nested selection grid (docs/report.md 9.17)


def _field_fit(P_tr, Y_tr, pca_k):
    """PCA-ridge from the power field to the temperature field; returns a predictor."""
    f_mu, f_sd = P_tr.mean(0), P_tr.std(0) + 1e-12
    Pc = (P_tr - f_mu) / f_sd
    # Economy SVD of the (n_train x n_cells) matrix: never forms a cell-space Gram, which
    # is 24-152 GB on these grids and is what OOMed the linearity audit on Darcy.
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    B = Vt[:min(pca_k, Vt.shape[0])].T
    A = np.hstack([Pc @ B, np.ones((len(P_tr), 1))])
    W = np.linalg.solve(A.T @ A + FIELD_LAMBDA * np.eye(A.shape[1]), A.T @ Y_tr)
    return lambda P: np.hstack([((P - f_mu) / f_sd) @ B, np.ones((len(P), 1))]) @ W


def select_field_pca_k(P_tr, Y_tr, grid=FIELD_PCA_GRID, folds=4, seed=0):
    """pca_k maximising inner-CV spatially-detrended R², using the training scenarios only."""
    idx = np.random.default_rng(seed).permutation(len(P_tr))
    parts = np.array_split(idx, folds)
    score = []
    for k in grid:
        r2 = []
        for te in parts:
            tr = np.setdiff1d(idx, te)
            pred = _field_fit(P_tr[tr], Y_tr[tr], k)(P_tr[te])
            for yp, yt in zip(pred, Y_tr[te]):
                a, b = yp - yp.mean(), yt - yt.mean()
                r2.append(1.0 - ((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-12))
        score.append(float(np.mean(r2)))
    return grid[int(np.argmax(score))]


def predict_all(train: List[dict], test: List[dict], block_keys: List[str],
                k: int, ridge_lambda: float, power_pca: int = 0,
                include_field_linear: bool = False,
                field_pca_k: Optional[int] = FIELD_PCA_K,
                ) -> List[Dict[str, np.ndarray]]:
    """Fit every baseline on `train` and return raw predicted fields for `test`.
    `include_field_linear` adds a linear fit on the full power field; off by default so §9.1a/§9.12d tables are unchanged.
    `field_pca_k=None` selects its rank by nested CV on `train` (the §9.17 protocol)."""
    # Per-block positions enter the feature vector whenever the dataset carries them.
    # For fixed-placement data they are constant and the zero-variance guard below drops
    # them; for moving-source data (docs/report.md 9.14) they are what tells the baseline
    # where the heat actually went.
    pos_keys = collect_position_keys(train + test)
    X_tr = np.stack([feature_vector(sc['meta'], block_keys, pos_keys) for sc in train])
    X_te = np.stack([feature_vector(sc['meta'], block_keys, pos_keys) for sc in test])

    if power_pca > 0:
        # Append a linear summary of the full power field so ridge sees the actual
        # source, not just its block averages.
        #
        # Cap the component count against the training size. With more features
        # than samples the normal equations are singular and the fit explodes on
        # extrapolation splits -- an unregularised run produced detrended MAE of
        # 3e12 K, which is a degenerate solve rather than a measurement. Keeping
        # features well below n_train makes the baseline numerically honest, and
        # a linear model that needs more components than it has samples has
        # already lost on capacity grounds.
        n_comp = max(1, min(power_pca, (len(train) - X_tr.shape[1]) // 2))
        if n_comp < power_pca:
            _log.info("power PCA reduced %d -> %d components for %d training "
                      "scenarios (keeps the linear system determined)",
                      power_pca, n_comp, len(train))
        A_tr, A_te = power_pca_features(train, test, n_comp)
        X_tr = np.hstack([X_tr, A_tr])
        X_te = np.hstack([X_te, A_te])

    Y_tr = np.stack([sc['temp'] for sc in train])           # (n_train, n_points)

    # Standardise features for distance-based methods (guard zero-variance cols).
    mu, sigma = X_tr.mean(0), X_tr.std(0)
    sigma[sigma < 1e-12] = 1.0
    Z_tr = (X_tr - mu) / sigma

    # Ridge on standardised features + intercept, solved once for all points.
    A = np.hstack([Z_tr, np.ones((len(train), 1))])
    n_feat = A.shape[1]
    reg = ridge_lambda * np.eye(n_feat)
    reg[-1, -1] = 0.0                                        # never penalise intercept
    W = np.linalg.solve(A.T @ A + reg, A.T @ Y_tr)           # (n_feat, n_points)

    mean_field = Y_tr.mean(0)

    field_pred = None
    if include_field_linear:
        P_tr = np.stack([sc['power'] for sc in train])
        P_te = np.stack([sc['power'] for sc in test])
        k_f = field_pca_k if field_pca_k else select_field_pca_k(P_tr, Y_tr)
        field_pred = _field_fit(P_tr, Y_tr, k_f)(P_te)
        predict_all.last_field_pca_k = k_f

    preds: List[Dict[str, np.ndarray]] = []

    for ti in range(len(test)):
        x = (X_te[ti] - mu) / sigma

        d = np.linalg.norm(Z_tr - x, axis=1)
        idx = np.argsort(d)[:min(k, len(train))]
        w = 1.0 / np.maximum(d[idx], 1e-9)
        w /= w.sum()

        row = {
            'mean':  mean_field,
            'nn':    Y_tr[int(np.argmin(d))],
            'knn':   w @ Y_tr[idx],
            'ridge': np.append(x, 1.0) @ W,
        }
        if field_pred is not None:
            row['linear_field'] = field_pred[ti]
        preds.append(row)

    return preds


def fit_predict(train: List[dict], test: List[dict], block_keys: List[str],
                k: int, ridge_lambda: float, power_pca: int = 0
                ) -> Dict[str, List[Dict[str, float]]]:
    """Run every baseline; return {baseline_name: [per-scenario metrics]}."""
    preds = predict_all(train, test, block_keys, k, ridge_lambda, power_pca)
    results: Dict[str, List[Dict[str, float]]] = {n: [] for n in ('mean', 'nn', 'knn', 'ridge')}
    for sc, pred in zip(test, preds):
        for name, field in pred.items():
            results[name].append(metrics(field, sc['temp'], sc['coords']))
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--geometry', required=True, help="e.g. geometry1")
    p.add_argument('--data', type=Path, default=Path('data/3d-ice'))
    p.add_argument('--test-split', default='test',
                   help="Filename split tag for in-distribution evaluation (default: test)")
    p.add_argument('--split-manifest', type=Path, default=None,
                   help="JSON manifest from scripts/make_ood_split.py defining an "
                        "extrapolation split. Overrides --test-split.")
    p.add_argument('--k', type=int, default=3, help="Neighbours for knn (default: 3)")
    p.add_argument('--ridge-lambda', type=float, default=1.0,
                   help="L2 penalty for the ridge baseline (default: 1.0)")
    p.add_argument('--power-pca', type=int, default=0,
                   help="Give ridge/kNN this many principal components of the FULL "
                        "power field as extra features. Required for a fair comparison "
                        "on per-cell power data: block means hide most of the source.")
    p.add_argument('--output', type=Path, default=None, help="Write results JSON here")
    args = p.parse_args()

    geom_dir = args.data / args.geometry
    root = geom_dir if geom_dir.is_dir() else args.data

    if args.split_manifest:
        man = json.loads(args.split_manifest.read_text(encoding='utf-8'))
        if man.get('geometry') != args.geometry:
            _log.error("Manifest is for %s, not %s", man.get('geometry'), args.geometry)
            sys.exit(1)
        train_files = [root / n for n in man['train']]
        test_files = [root / n for n in man['test']]
        split_label = f"ood:{man['axis']}"
        _log.info("OOD split '%s' -- %s", man['axis'], man['rule'])
    else:
        train_files = sorted(root.glob(f'{args.geometry}_train_*.npz'))
        test_files = sorted(root.glob(f'{args.geometry}_{args.test_split}_*.npz'))
        split_label = args.test_split

    missing = [f for f in train_files + test_files if not f.exists()]
    if missing:
        _log.error("Missing files: %s", [f.name for f in missing[:5]])
        sys.exit(1)
    if not train_files or not test_files:
        _log.error("Empty split for %s under %s (train=%d, test=%d)",
                   args.geometry, root, len(train_files), len(test_files))
        sys.exit(1)

    train = [load_scenario(f) for f in train_files]
    test = [load_scenario(f) for f in test_files]

    n_pts = train[0]['temp'].shape[0]
    mismatched = [sc['name'] for sc in train + test if sc['temp'].shape[0] != n_pts]
    if mismatched:
        _log.error("Mesh size mismatch (expected %d points): %s", n_pts, mismatched)
        sys.exit(1)

    block_keys = collect_block_keys(train + test)
    _log.info("%s: %d train, %d %s, %d points, %d power blocks",
              args.geometry, len(train), len(test), split_label, n_pts, len(block_keys))

    raw = fit_predict(train, test, block_keys, args.k, args.ridge_lambda,
                      power_pca=args.power_pca)
    summary = {name: aggregate(m) for name, m in raw.items()}

    spatial_std = float(np.mean([m['true_spatial_std_K'] for m in raw['mean']]))
    print(f"\n{args.geometry}  (train={len(train)}, {split_label}={len(test)})")
    print(f"Mean spatial std of the true test fields: {spatial_std:.3f} K "
          f"-- this is the signal a surrogate must actually predict.")
    print(f"{'baseline':<8} {'MAE':>8} {'RMSE':>8} {'worst':>8} "
          f"{'det.MAE':>9} {'spat.R2':>9} {'hot_dx':>9}")
    print("-" * 66)
    for name in ('mean', 'nn', 'knn', 'ridge'):
        s = summary[name]
        print(f"{name:<8} {s['mae_K']:>8.3f} {s['rmse_K']:>8.3f} {s['worst_mae_K']:>8.3f} "
              f"{s['mae_detrended_K']:>9.3f} {s['spatial_r2']:>9.3f} "
              f"{s['hotspot_loc_err_um']:>9.1f}")
    print("\nMAE/RMSE/worst/det.MAE in K; hot_dx = hotspot location error (um).")
    print("det.MAE and spat.R2 are computed after removing each field's mean, so")
    print("they score spatial structure only. Judge architectures on those two:")
    print("raw MAE is dominated by the per-scenario offset (largely the ambient")
    print("input), which a constant predictor can already match.")
    print("A learned surrogate must beat 'ridge' to justify its cost.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'geometry': args.geometry,
            'test_split': split_label,
            'n_train': len(train), 'n_test': len(test),
            'n_points': n_pts, 'block_keys': block_keys,
            'k': args.k, 'ridge_lambda': args.ridge_lambda,
            'summary': summary,
            'per_scenario': {n: dict(zip([s['name'] for s in test], m))
                             for n, m in raw.items()},
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        _log.info("Wrote %s", args.output)


if __name__ == '__main__':
    main()
