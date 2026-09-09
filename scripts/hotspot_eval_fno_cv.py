"""
k-fold cross-validated FNO vs. ridge/kNN on hotspot metrics.

Companion to scripts/hotspot_eval.py, which does the same thing for the non-neural
baselines only. This one trains an FNO per fold, so it is expensive (hours on CPU) and
must be run in the background.

Why: docs/report.md Sec 9.12b recorded two CPU-budget FNOs beating ridge on hotspot
localisation -- the first metric in this project on which a neural operator wins -- but
on only 5 test scenarios, and scored on argmax distance alone. Re-scoring those same
checkpoints with the fuller metric set (2026-09-09) showed the win is real but much
narrower than it looked:

    model                   |peak err| K   loc err um   top1% recall   <=2mm
    ridge                          2.132         7169          0.032    0.00
    kNN                            1.777         7616          0.038    0.00
    FNO-default                   14.860         4469          0.025    0.40
    FNO-tierB-wide-narrow          8.060         3324          0.054    0.20

i.e. FNO localises the hotspot better (3.3-4.5 mm vs ridge's 7.2 mm, and the only
non-zero hit rates within 2 mm) while being 4-7x WORSE at predicting the peak
temperature. Knowing where the hotspot is but being 15 K wrong about how hot it is, is
not straightforwardly more useful than the reverse. Both halves need saying.

Those numbers are n=5. This script gets the sample size up by training one FNO per fold
on the SAME fold split scripts/hotspot_eval.py uses (same seed, same kfold_indices call),
so FNO and the baselines are scored on identical held-out scenarios.

Usage (expect ~6 h on CPU for the defaults; run it in the background):
    python scripts/hotspot_eval_fno_cv.py --geometry geometry1 --folds 5 --epochs 150
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.baselines import load_scenario, collect_block_keys, predict_all
from scripts.hotspot_eval import hotspot_metrics, kfold_indices, aggregate, flatness_diagnostic

from src.reproducibility import set_seed
from src.pinn.data_loader import NormStats, compute_norm_stats
from src.fno.model import build_fno
from src.fno.data_loader import FNODataset, predict_to_flat
from src.fno.trainer import FNOTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('hotspot_cv')
warnings.filterwarnings('ignore', message='FNO3d: modes.*clamped')

CONFIGS = {
    'default':           (32, (16, 16, 12)),
    'tierB-wide-narrow': (44, (12, 12, 8)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', default='geometry1')
    ap.add_argument('--data', type=Path, default=Path('data/3d-ice'))
    ap.add_argument('--config', default='default', choices=list(CONFIGS))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=Path,
                    default=Path('results/hotspot_eval_fno_cv.json'))
    args = ap.parse_args()

    files = sorted(args.data.rglob(f'{args.geometry}_*.npz'))
    assert files, f'No .npz for {args.geometry} under {args.data}'
    scenarios = [load_scenario(f) for f in files]
    block_keys = collect_block_keys(scenarios)
    diag = flatness_diagnostic(scenarios)
    log.info('%s: %d scenarios, %d-fold CV, config=%s, epochs=%d',
             args.geometry, len(scenarios), args.folds, args.config, args.epochs)
    log.info('hotspot well-posedness: top-100 spread %.0f um (median), dT %.3f K',
             diag['peak_spread_um_median'], diag['dT_top100_K_median'])

    channels, modes = CONFIGS[args.config]
    # IDENTICAL folds to scripts/hotspot_eval.py -- same helper, same seed. Without this
    # the FNO and baseline numbers would not be comparable scenario-for-scenario.
    folds = kfold_indices(len(scenarios), args.folds, args.seed)

    # Grid from data, as everywhere else in this project (3D-ICE z-grid is adaptive).
    from src.core.geometry_builders import get_geometry_by_name
    geometry = get_geometry_by_name(args.geometry)
    grid = list(geometry.mesh_resolution)
    n_pts = scenarios[0]['coords'].shape[0]
    if n_pts % (grid[0] * grid[1]) == 0:
        grid[2] = n_pts // (grid[0] * grid[1])
    grid_shape = tuple(grid)
    log.info('grid_shape=%s', grid_shape)

    rows: Dict[str, List[Dict[str, float]]] = {}
    out_root = args.output.parent / f'hotspot_cv_{args.geometry}_{args.config}'
    out_root.mkdir(parents=True, exist_ok=True)

    for fi, fold_idx in enumerate(folds, 1):
        test_i = set(fold_idx.tolist())
        train_i = [i for i in range(len(scenarios)) if i not in test_i]
        train_sc = [scenarios[i] for i in train_i]
        test_sc = [scenarios[i] for i in sorted(test_i)]
        train_f = [files[i] for i in train_i]
        test_f = [files[i] for i in sorted(test_i)]

        # --- baselines on this fold (cheap, exact same implementations as Sec 9.1a)
        for sc, pred in zip(test_sc, predict_all(train_sc, test_sc, block_keys, 3, 1.0, 0)):
            for name, field in pred.items():
                rows.setdefault(name, []).append(
                    hotspot_metrics(field, sc['temp'], sc['coords']))

        # --- FNO on this fold
        set_seed(args.seed, deterministic=True)
        fold_dir = out_root / f'fold{fi}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        norm_stats = compute_norm_stats(train_f, {args.geometry: geometry})
        train_ds = FNODataset(train_f, norm_stats, grid_shape)
        val_ds = FNODataset(test_f, norm_stats, grid_shape)

        model = build_fno(grid_shape=grid_shape, modes=modes, hidden_ch=channels,
                          n_blocks=args.blocks, device=torch.device('cpu'))
        log.info('fold %d/%d: train=%d test=%d params=%d',
                 fi, args.folds, len(train_f), len(test_f), model.n_parameters)
        t0 = time.time()
        FNOTrainer(
            model=model, norm_stats=norm_stats, train_data=train_ds, val_data=val_ds,
            output_dir=fold_dir, batch_size=args.batch_size, epochs=args.epochs,
            lr=args.lr, device=torch.device('cpu'),
            geometry_name=f'{args.geometry}_fold{fi}', pde_weight=0.0, flux_weight=0.0,
            geometry=geometry, log_interval=max(1, args.epochs // 3),
        ).train()

        model.eval()
        for sc, item in zip(test_sc, val_ds.items):
            P, T = predict_to_flat(model, item, torch.device('cpu'), norm_stats)
            rows.setdefault(f'FNO-{args.config}', []).append(
                hotspot_metrics(P, T, sc['coords']))
        log.info('fold %d done in %.1f min', fi, (time.time() - t0) / 60)

        # Incremental save so a partial run is still usable.
        with open(args.output, 'w') as f:
            json.dump({'geometry': args.geometry, 'config': args.config,
                       'folds_completed': fi, 'folds': args.folds,
                       'epochs': args.epochs, 'flatness': diag,
                       'results': {k: aggregate(v) for k, v in rows.items()}}, f, indent=2)

    print(f'\n=== {args.geometry}: {args.folds}-fold CV, FNO config={args.config}, '
          f'{args.epochs} epochs ===')
    print(f'{"model":<22} {"|peak err| K":>13} {"loc err um":>12} {"loc p90":>10} '
          f'{"top1% recall":>13} {"<=1mm":>7} {"<=2mm":>7}')
    for name, rs in rows.items():
        a = aggregate(rs)
        print(f'{name:<22} {a["abs_peak_temp_err_K_mean"]:>13.3f} '
              f'{a["hotspot_loc_err_um_median"]:>12.0f} {a["hotspot_loc_err_um_p90"]:>10.0f} '
              f'{a["top1pct_recall_mean"]:>13.3f} '
              f'{a["hit_within_1mm"]:>7.2f} {a["hit_within_2mm"]:>7.2f}')
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
