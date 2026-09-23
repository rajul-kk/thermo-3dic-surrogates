"""Hotspot-focused evaluation, scored with k-fold cross-validation."""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.baselines import load_scenario, collect_block_keys, predict_all

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger('hotspot_eval')

TOP_FRAC = 0.01      # "top 1% hottest cells" for the recall metric
FLATNESS_N = 100     # how many hottest cells the flatness diagnostic looks at


def hotspot_metrics(pred: np.ndarray, true: np.ndarray, coords: np.ndarray) -> Dict[str, float]:
    """Hotspot-specific scores. See module docstring for what each one is for."""
    i_true, i_pred = int(np.argmax(true)), int(np.argmax(pred))
    loc_err = float(np.linalg.norm(coords[i_pred] - coords[i_true]))

    n_top = max(1, int(round(TOP_FRAC * true.size)))
    top_true = np.argpartition(true, -n_top)[-n_top:]
    top_pred = np.argpartition(pred, -n_top)[-n_top:]
    recall = len(np.intersect1d(top_true, top_pred, assume_unique=False)) / float(n_top)

    return {
        'peak_temp_err_K':     float(pred.max() - true.max()),
        'abs_peak_temp_err_K': float(abs(pred.max() - true.max())),
        'hotspot_temp_err_K':  float(pred[i_true] - true[i_true]),
        'hotspot_loc_err_um':  loc_err,
        'top1pct_recall':      float(recall),
        'hit_within_1mm':      float(loc_err <= 1000.0),
        'hit_within_2mm':      float(loc_err <= 2000.0),
    }


def flatness_diagnostic(scenarios: List[dict]) -> Dict[str, float]:
    """How well-posed is 'the hotspot' on this geometry? See module docstring."""
    spreads, gaps = [], []
    for sc in scenarios:
        T, C = sc['temp'], sc['coords']
        order = np.argsort(T)[::-1][:FLATNESS_N]
        spreads.append(float(np.median(np.linalg.norm(C[order] - C[order[0]], axis=1))))
        gaps.append(float(T[order[0]] - T[order[-1]]))
    return {
        'peak_spread_um_median': float(np.median(spreads)),
        'peak_spread_um_max':    float(np.max(spreads)),
        'dT_top100_K_median':    float(np.median(gaps)),
    }


def kfold_indices(n: int, folds: int, seed: int) -> List[np.ndarray]:
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [np.asarray(part) for part in np.array_split(idx, folds)]


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=float)
        if key.startswith('hit_within'):
            out[key] = float(vals.mean())            # a rate, not a distribution
        else:
            out[f'{key}_mean'] = float(vals.mean())
            out[f'{key}_median'] = float(np.median(vals))
            out[f'{key}_p90'] = float(np.percentile(vals, 90))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', nargs='+', required=True)
    ap.add_argument('--data', type=Path, default=Path('data/3d-ice'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--k', type=int, default=3, help='kNN neighbour count')
    ap.add_argument('--ridge-lambda', type=float, default=1.0)
    ap.add_argument('--power-pca', type=int, default=0)
    ap.add_argument('--field-linear', action='store_true',
                    help='Add a linear fit on the full per-cell power field. Needed on '
                         'layout-varying data, where the compact block vector is p>n and '
                         'ridge is confounded (docs/report.md 9.15c/9.15d).')
    ap.add_argument('--field-pca-k', type=int, default=0,
                    help='Rank of the field fit; 0 = choose it by nested CV inside each training fold.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    all_results = {}
    for geom in args.geometry:
        files = sorted(args.data.rglob(f'{geom}_*.npz'))
        if not files:
            log.error('No .npz files for %s under %s', geom, args.data)
            sys.exit(1)
        scenarios = [load_scenario(f) for f in files]
        block_keys = collect_block_keys(scenarios)
        diag = flatness_diagnostic(scenarios)

        log.info('%s: %d scenarios pooled (train+test), %d-fold CV, %d block features',
                 geom, len(scenarios), args.folds, len(block_keys))
        log.info('%s: peak region spread (median of top-%d cells) = %.0f um; '
                 'dT across top-%d = %.3f K',
                 geom, FLATNESS_N, diag['peak_spread_um_median'],
                 FLATNESS_N, diag['dT_top100_K_median'])

        per_baseline: Dict[str, List[Dict[str, float]]] = {}
        pca_ks: List[int] = []
        for fold_idx in kfold_indices(len(scenarios), args.folds, args.seed):
            test_set = [scenarios[i] for i in fold_idx]
            train_set = [scenarios[i] for i in range(len(scenarios)) if i not in set(fold_idx.tolist())]
            preds = predict_all(train_set, test_set, block_keys,
                                args.k, args.ridge_lambda, args.power_pca,
                                include_field_linear=args.field_linear,
                                field_pca_k=args.field_pca_k or None)
            if args.field_linear:
                pca_ks.append(predict_all.last_field_pca_k)
            for sc, pred in zip(test_set, preds):
                for name, field in pred.items():
                    per_baseline.setdefault(name, []).append(
                        hotspot_metrics(field, sc['temp'], sc['coords']))

        all_results[geom] = {
            'n_scenarios': len(scenarios),
            'folds': args.folds,
            'flatness': diag,
            'field_pca_k_chosen': pca_ks,
            'baselines': {name: aggregate(rows) for name, rows in per_baseline.items()},
        }

        print(f'\n=== {geom}  ({len(scenarios)} scenarios, {args.folds}-fold CV) ===')
        print(f'  hotspot well-posedness: top-{FLATNESS_N} cells span '
              f'{diag["peak_spread_um_median"]:.0f} um (median), '
              f'dT = {diag["dT_top100_K_median"]:.3f} K'
              f'{"   [SHARP: argmax distance is meaningful]" if diag["peak_spread_um_median"] < 1000 else "   [DIFFUSE: read argmax distance with caution]"}')
        print(f'  {"baseline":<10} {"|peak err| K":>13} {"loc err um":>12} {"loc p90":>10} '
              f'{"top1% recall":>13} {"<=1mm":>7} {"<=2mm":>7}')
        for name, agg in all_results[geom]['baselines'].items():
            print(f'  {name:<10} {agg["abs_peak_temp_err_K_mean"]:>13.3f} '
                  f'{agg["hotspot_loc_err_um_median"]:>12.0f} '
                  f'{agg["hotspot_loc_err_um_p90"]:>10.0f} '
                  f'{agg["top1pct_recall_mean"]:>13.3f} '
                  f'{agg["hit_within_1mm"]:>7.2f} {agg["hit_within_2mm"]:>7.2f}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
