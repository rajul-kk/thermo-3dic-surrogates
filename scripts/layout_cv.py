"""K-fold CV of the standard baselines on layout-randomised datasets (docs/report.md 9.15b)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.baselines import collect_block_keys, load_scenario, predict_all
from scripts.hotspot_eval import kfold_indices

log = logging.getLogger('layout_cv')


def per_scenario_stats(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Detrended (mean-removed) scores for one field, so only spatial structure is judged."""
    dy, dp = y - y.mean(), p - p.mean()
    ss = float(np.sum(dy ** 2))
    return {'det_mae': float(np.mean(np.abs(dy - dp))),
            'sigma': float(dy.std()),
            'r2': float(1.0 - np.sum((dy - dp) ** 2) / ss) if ss > 0 else float('nan'),
            'corr': float(np.corrcoef(dp, dy)[0, 1]) if dp.std() > 0 and dy.std() > 0 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--ridge-lambda', type=float, default=1.0)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    g = args.geometry
    root = args.data / g if (args.data / g).is_dir() else args.data
    files = sorted(root.glob(f'{g}_*.npz'))
    if not files:
        log.error('no scenarios under %s', root)
        sys.exit(1)
    scen = [load_scenario(f) for f in files]
    bk = collect_block_keys(scen)
    log.info('%s: %d scenarios pooled, %d-fold CV, %d block features', g, len(scen), args.folds, len(bk))

    rows: Dict[str, List[Dict[str, float]]] = {}
    for fold in kfold_indices(len(scen), args.folds, args.seed):
        keep = set(fold.tolist())
        test = [scen[i] for i in fold]
        train = [scen[i] for i in range(len(scen)) if i not in keep]
        preds = predict_all(train, test, bk, args.k, args.ridge_lambda)
        for j, sc in enumerate(test):
            for name, p in preds[j].items():
                rows.setdefault(name, []).append(per_scenario_stats(sc['temp'], p))

    n = sum(len(v) for v in rows.values()) // max(len(rows), 1)
    sigma = float(np.mean([r['sigma'] for r in next(iter(rows.values()))]))
    print(f'\n=== {g}  ({len(scen)} scenarios, {args.folds}-fold CV, n={n} held-out) ===')
    print(f'Mean spatial sigma of true fields: {sigma:.3f} K')
    print(f"{'baseline':9s} {'det.MAE':>8s} {'norm err':>9s} {'R2 mean':>9s} "
          f"{'R2 med':>8s} {'corr':>7s} {'R2<0':>6s}")
    out = {}
    for name, rs in rows.items():
        r2 = np.array([r['r2'] for r in rs])
        dm = float(np.mean([r['det_mae'] for r in rs]))
        out[name] = {'det_mae': dm, 'norm_err': dm / sigma, 'r2_mean': float(r2.mean()),
                     'r2_median': float(np.median(r2)), 'r2_std': float(r2.std()),
                     'corr': float(np.mean([r['corr'] for r in rs])),
                     'n_negative_r2': int((r2 < 0).sum()), 'n': len(rs)}
        o = out[name]
        print(f"{name:9s} {dm:8.3f} {o['norm_err']:9.3f} {o['r2_mean']:9.3f} "
              f"{o['r2_median']:8.3f} {o['corr']:+7.3f} {o['n_negative_r2']:4d}/{len(rs)}")
    print('\nR2<0 counts scenarios where the model is worse than that field\'s own mean.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'geometry': g, 'sigma_K': sigma,
                                           'folds': args.folds, 'results': out}, indent=2),
                               encoding='utf-8')


if __name__ == '__main__':
    main()
