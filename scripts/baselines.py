"""
Non-neural baselines for the 3D-IC thermal benchmark.

Every learned surrogate in this repo must beat these before its architecture can
be credited for anything. They need no GPU and no training loop; the whole suite
runs in seconds.

Baselines
---------
mean   Predict the training-set mean temperature field. The floor: any model
       that does not beat this has learned nothing.

nn     Copy the temperature field of the single nearest training scenario in
       normalised parameter space. Zero-parameter memorisation.

knn    Inverse-distance-weighted blend of the k nearest training scenarios.
       This is "what you get for free" from the dataset without any model.

ridge  Per-point ridge regression on the scenario parameter vector. This is the
       baseline that matters most scientifically. Steady-state conduction with
       fixed k is LINEAR in volumetric power and in ambient temperature:

           T(x) = T_amb + sum_b A_b(x) * Q_b

       where A_b is the (scenario-independent) thermal impedance from block b.
       The only genuine nonlinearities are k(T) and the convective boundary term
       (which enters through 1/h, supplied here as an explicit feature). A ridge
       fit therefore recovers the exact physics of the linear regime, and any
       neural surrogate must beat it to justify its cost. If ridge is already at
       ~1 K MAE, the nonlinear capacity is not what is buying accuracy.

Usage
-----
    python scripts/baselines.py --geometry geometry1 --data data/3d-ice
    python scripts/baselines.py --geometry geometry1 --data data/3d-ice \\
        --test-split ood --output results/baselines_geometry1_ood.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    """
    Project each scenario's full power field onto its leading principal components.

    Needed to keep ridge a FAIR baseline once power becomes a per-cell field. With
    block-scalar power the whole source is 4-13 numbers and ridge can consume it
    directly; with a per-cell map the source has ~10,000 degrees of freedom and
    handing ridge only the block means would rig the comparison by hiding most of
    the input. PCA gives the linear model the best n_comp-dimensional summary of
    the source that exists, so if it still loses, it loses on the merits.
    """
    P_tr = np.stack([sc['power'] for sc in train])
    mu = P_tr.mean(0)
    # Economy SVD of the centred training fields; components are rows of Vt.
    _, _, Vt = np.linalg.svd(P_tr - mu, full_matrices=False)
    basis = Vt[:min(n_comp, Vt.shape[0])]
    proj = lambda S: np.stack([(sc['power'] - mu) @ basis.T for sc in S])
    return proj(train), proj(test)


def feature_vector(meta: dict, block_keys: List[str]) -> np.ndarray:
    """
    Build the scenario parameter vector.

    Includes 1/htc alongside htc: convective thermal resistance is proportional
    to 1/h, so 1/h is the term that enters the temperature field linearly. Giving
    the linear models this feature is what makes `ridge` a fair — rather than
    strawman — baseline.
    """
    htc = float(meta.get('htc', 0.0))
    feats = [float(meta.get(k, 0.0)) for k in block_keys]
    feats.append(htc)
    feats.append(1.0 / htc if htc > 0 else 0.0)
    feats.append(float(meta.get('t_ambient_kelvin', 298.15)))
    feats.append(float(meta.get('tsv_density', 0.0)))
    return np.asarray(feats, dtype=np.float64)


def collect_block_keys(scenarios: List[dict]) -> List[str]:
    """
    Union of block-power metadata keys, sorted for determinism.

    Prefers `nominal_block_power_*` (the requested power) over
    `block_power_*` (the delivered power after any closed-loop feedback)
    whenever any scenario carries throttle OR leakage metadata. Both
    mechanisms make delivered power a function of the temperature being
    solved for, so both must hand the baseline the request, not the outcome. Using the delivered power as ridge's
    feature hands it the already-resolved answer for throttled scenarios --
    it no longer has to represent the closed feedback loop at all, and scores
    a misleadingly high R^2 that has nothing to do with whether the map is
    actually linear. Found 2026-08-09: ridge scored *better* on throttled
    data (spatial R^2 0.989) than on the same geometry without throttling
    (0.970) until this was fixed -- using nominal power instead correctly
    shows real degradation (0.890).
    """
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
    """
    Raw AND spatially-detrended error.

    Detrending matters on this dataset. Each scenario's field is dominated by a
    near-uniform offset set by ambient temperature and total power, while the
    spatial gradient within a scenario is ~1 K. Raw MAE therefore mostly scores
    how well a model copies the ambient input, and a model that predicts a
    constant per scenario can post a respectable MAE having learned no spatial
    structure whatsoever.

    `mae_detrended_K` removes each field's own mean from both prediction and
    truth, so it measures only the spatial structure -- the part a thermal
    surrogate actually exists to predict. `spatial_r2` is R^2 on that same
    detrended field. Report both; judge architectures on the detrended pair.
    """
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

def fit_predict(train: List[dict], test: List[dict], block_keys: List[str],
                k: int, ridge_lambda: float, power_pca: int = 0
                ) -> Dict[str, List[Dict[str, float]]]:
    """Run every baseline; return {baseline_name: [per-scenario metrics]}."""
    X_tr = np.stack([feature_vector(sc['meta'], block_keys) for sc in train])
    X_te = np.stack([feature_vector(sc['meta'], block_keys) for sc in test])

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
    results: Dict[str, List[Dict[str, float]]] = {n: [] for n in ('mean', 'nn', 'knn', 'ridge')}

    for ti, sc in enumerate(test):
        x = (X_te[ti] - mu) / sigma
        true, coords = sc['temp'], sc['coords']

        results['mean'].append(metrics(mean_field, true, coords))

        d = np.linalg.norm(Z_tr - x, axis=1)
        results['nn'].append(metrics(Y_tr[int(np.argmin(d))], true, coords))

        idx = np.argsort(d)[:min(k, len(train))]
        w = 1.0 / np.maximum(d[idx], 1e-9)
        w /= w.sum()
        results['knn'].append(metrics(w @ Y_tr[idx], true, coords))

        results['ridge'].append(metrics(np.append(x, 1.0) @ W, true, coords))

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
