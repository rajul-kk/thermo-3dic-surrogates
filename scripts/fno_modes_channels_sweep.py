"""Modes-vs-channels compute-optimal sweep for baseline FNO."""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.core.geometry_builders import get_geometry_by_name
from src.reproducibility import set_seed
from src.pinn.data_loader import NormStats, compute_norm_stats
from src.fno.model import build_fno
from src.fno.data_loader import FNODataset, predict_to_flat
from src.fno.trainer import FNOTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                     datefmt='%H:%M:%S')
log = logging.getLogger('fno_sweep')
warnings.filterwarnings('ignore', message='FNO3d: modes.*clamped')

# (label, channels, modes) -- see module docstring for how these were chosen.
# "default" is this project's existing scripts/train_fno.py hardcoded default,
# included as the point everything else is being checked against.
CANDIDATES = [
    ('cpu-fast-preset',   16, (8, 8, 6)),     # this project's own --cpu-fast preset
    ('tierA-narrow-wide', 16, (28, 28, 10)),  # ~4.8M params
    ('tierA-mid',         20, (22, 22, 10)),  # ~4.6M params
    ('tierB-narrow-wide', 18, (28, 28, 10)),  # ~6.1M params
    ('default',           32, (16, 16, 12)),  # ~6.3M params -- CURRENT PROJECT DEFAULT
    ('tierB-wide-narrow', 44, (12, 12, 8)),   # ~6.7M params
]

# geometry1 baselines, re-measured 2026-09-09 on the CURRENT dataset
# (regenerate: python scripts/baselines.py --geometry geometry1 --data data/3d-ice;
# see docs/report.md Sec 9.1a).
#
# CORRECTION 2026-09-09: this dict originally carried docs/report.md Sec 9.1's
# 2026-07-31 numbers (ridge det.MAE 0.009, spatial R^2 0.999). Those describe a
# superseded pre-regime-fix dataset -- Sec 9.5 had already said so -- and quoting them
# here made this sweep's first write-up claim a ~60x FNO-vs-ridge gap when the real
# gap on current data is ~2.7x. Re-measure with scripts/baselines.py rather than
# copying numbers out of the report if this is ever pointed at a different dataset.
REFERENCE = {
    'mean':              {'mae_detrended_K': 2.308, 'spatial_r2': -0.540, 'hotspot_loc_err_um': 5880.3},
    'nearest-neighbour': {'mae_detrended_K': 0.733, 'spatial_r2': 0.870, 'hotspot_loc_err_um': 7734.2},
    'kNN (k=3)':         {'mae_detrended_K': 0.465, 'spatial_r2': 0.962, 'hotspot_loc_err_um': 7100.6},
    'ridge':             {'mae_detrended_K': 0.407, 'spatial_r2': 0.970, 'hotspot_loc_err_um': 7399.5},
}


def metrics(pred: np.ndarray, true: np.ndarray, coords: np.ndarray) -> dict:
    """Byte-for-byte identical to scripts/baselines.py::metrics."""
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    d_pred, d_true = pred - pred.mean(), true - true.mean()
    d_err = d_pred - d_true
    d_ss_tot = float(np.sum(d_true ** 2))
    spatial_r2 = 1.0 - float(np.sum(d_err ** 2)) / d_ss_tot if d_ss_tot > 0 else float('nan')

    i_pred, i_true = int(np.argmax(pred)), int(np.argmax(true))
    return {
        'mae_K': float(np.mean(np.abs(err))),
        'mae_detrended_K': float(np.mean(np.abs(d_err))),
        'r2': r2,
        'spatial_r2': spatial_r2,
        'hotspot_loc_err_um': float(np.linalg.norm(coords[i_pred] - coords[i_true])),
    }


def collect_files(data_dir: Path, geom: str, split: str):
    files = sorted(data_dir.rglob(f'{geom}_{split}_*.npz'))
    return files or sorted(data_dir.glob(f'{geom}_{split}_*.npz'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', default='geometry1')
    ap.add_argument('--data', type=Path, default=Path('data/3d-ice'))
    ap.add_argument('--output', type=Path, default=Path('results/fno_modes_channels_sweep'))
    ap.add_argument('--only', nargs='+', default=None, metavar='LABEL',
                   help='Run only these candidate labels (see CANDIDATES) instead of '
                        'the full sweep -- e.g. for a longer confirmatory run on the '
                        'two most informative configs from a prior short sweep.')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    geometry = get_geometry_by_name(args.geometry)
    geometries = {args.geometry: geometry}
    grid_shape = geometry.mesh_resolution

    train_files = collect_files(args.data, args.geometry, 'train')
    test_files = collect_files(args.data, args.geometry, 'test')
    assert train_files, f'No training files found for {args.geometry} in {args.data}'
    assert test_files, f'No test files found for {args.geometry} in {args.data}'
    log.info('Files: %d train, %d test', len(train_files), len(test_files))

    # Grid from data, not the declared mesh -- same correction every other script in
    # this project applies (3D-ICE's z-grid is adaptive).
    d0 = np.load(train_files[0], allow_pickle=True)
    n_pts = d0['coords'].shape[0]
    nx, ny = grid_shape[0], grid_shape[1]
    if nx * ny > 0 and n_pts % (nx * ny) == 0:
        data_grid = (nx, ny, n_pts // (nx * ny))
        if data_grid != tuple(grid_shape):
            log.info('Grid from data: %s (geometry declares %s)', data_grid, tuple(grid_shape))
            grid_shape = data_grid

    norm_path = args.output / 'norm_stats.json'
    if norm_path.exists():
        norm_stats = NormStats.load(norm_path)
    else:
        norm_stats = compute_norm_stats(train_files, geometries)
        norm_stats.save(norm_path)

    nx, ny, nz = grid_shape
    xs = np.linspace(0, geometry.die_width, nx)
    ys = np.linspace(0, geometry.die_length, ny)
    zs = np.linspace(0, geometry.get_total_height(), nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    device = torch.device('cpu')
    candidates = CANDIDATES
    if args.only:
        wanted = set(args.only)
        candidates = [c for c in CANDIDATES if c[0] in wanted]
        missing = wanted - {c[0] for c in candidates}
        if missing:
            log.error('Unknown --only label(s): %s. Valid labels: %s',
                      sorted(missing), [c[0] for c in CANDIDATES])
            sys.exit(1)
    log.info('Sweep: %d candidates, %d epochs each, device=%s', len(candidates), args.epochs, device)

    results = {}
    for label, channels, modes in candidates:
        # Same seed for every candidate -- isolates (channels, modes) as the only
        # variable, same convention as kaggle_pinn_sampling_comparison.ipynb.
        set_seed(args.seed, deterministic=True)

        train_dataset = FNODataset(train_files, norm_stats, grid_shape)
        val_dataset = FNODataset(test_files, norm_stats, grid_shape)

        model = build_fno(grid_shape=grid_shape, modes=modes, hidden_ch=channels,
                          n_blocks=args.blocks, device=device)
        n_params = model.n_parameters
        log.info('[%s] channels=%d modes=%s params=%d', label, channels, modes, n_params)

        out_dir = args.output / label
        t0 = time.time()
        trainer = FNOTrainer(
            model=model, norm_stats=norm_stats, train_data=train_dataset, val_data=val_dataset,
            output_dir=out_dir, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            device=device, geometry_name=f'{args.geometry}_{label}', pde_weight=0.0, flux_weight=0.0,
            geometry=geometry, log_interval=max(1, args.epochs // 5),
        )
        trainer.train()
        elapsed = time.time() - t0

        model.eval()
        all_m = []
        with torch.no_grad():
            for item in val_dataset.items:
                T_pred_K, T_true_K = predict_to_flat(model, item, device, norm_stats)
                all_m.append(metrics(T_pred_K, T_true_K, coords))
        agg = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}
        agg['n_params'] = n_params
        agg['channels'] = channels
        agg['modes'] = list(modes)
        agg['train_time_s'] = elapsed
        agg['best_val_mae_K'] = trainer.best_val_mae
        results[label] = agg
        log.info('[%s] det.MAE=%.4f spatial_R2=%.4f hotspot_err=%.0fum time=%.1fmin',
                 label, agg['mae_detrended_K'], agg['spatial_r2'], agg['hotspot_loc_err_um'],
                 elapsed / 60)

        # Save incrementally so a partial run is still readable if interrupted.
        with open(args.output / 'sweep_results.json', 'w') as f:
            json.dump({'reference': REFERENCE, 'results': results,
                       'config': {'geometry': args.geometry, 'epochs': args.epochs,
                                  'batch_size': args.batch_size, 'seed': args.seed}}, f, indent=2)

    print('\n' + '=' * 100)
    print(f'{"config":<20} {"channels":>8} {"modes":>16} {"params":>10} '
          f'{"det.MAE(K)":>11} {"spatial R2":>11} {"hotspot(um)":>12} {"train(min)":>11}')
    print('-' * 100)
    for name, r in REFERENCE.items():
        print(f'{name:<20} {"":>8} {"":>16} {"":>10} {r["mae_detrended_K"]:>11.3f} '
              f'{r["spatial_r2"]:>11.3f} {r["hotspot_loc_err_um"]:>12.0f} {"":>11}')
    print('-' * 100)
    best_label = min(results, key=lambda k: results[k]['mae_detrended_K'])
    for label, r in results.items():
        marker = '  <-- BEST' if label == best_label else ''
        marker += '  (project default)' if label == 'default' else ''
        print(f'{label:<20} {r["channels"]:>8} {str(r["modes"]):>16} {r["n_params"]:>10,} '
              f'{r["mae_detrended_K"]:>11.4f} {r["spatial_r2"]:>11.4f} '
              f'{r["hotspot_loc_err_um"]:>12.0f} {r["train_time_s"]/60:>11.1f}{marker}')
    print('=' * 100)
    print(f'\nBest config: {best_label}. Project default is "default" '
          f'(channels=32, modes=(16,16,12)).')
    print(f'Full results: {args.output / "sweep_results.json"}')


if __name__ == '__main__':
    main()
