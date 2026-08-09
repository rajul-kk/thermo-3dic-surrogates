"""
Track B validation gate: leave-one-geometry-out.

Train FNO on 5 geometries (resampled onto a common grid), test zero-shot on
the 6th, with and without the distance-to-power-block geometry-aware field
(src/core/mesh.py:generate_distance_to_power_block_field). This is the actual
test goal.md's Track B specified before committing to the mechanism being
useful -- the field existing and being wired in (already done) doesn't by
itself demonstrate it helps zero-shot generalization.

Usage:
    python scripts/experiment_geometry_aware.py --holdout geometry4 --epochs 60
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.core.geometry_builders import get_geometry_by_name
from src.fno.data_loader import FNODataset, predict_to_flat
from src.fno.model import build_fno
from src.fno.trainer import FNOTrainer
from src.pinn.data_loader import compute_norm_stats
from src.reproducibility import set_seed
from scripts.baselines import metrics as compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

ALL_GEOMS = ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5', 'geometry6']


def _files_for(geom_name, data_root, split):
    return sorted((data_root / geom_name).glob(f'{geom_name}_{split}_*.npz'))


def run_one(use_geometry_field: bool, train_geoms, holdout, data_root, common_grid,
           epochs, channels, modes, blocks, seed):
    set_seed(seed)
    geometries = {g: get_geometry_by_name(g) for g in ALL_GEOMS}

    train_files, val_files = [], []
    for g in train_geoms:
        train_files += _files_for(g, data_root, 'train')
        val_files += _files_for(g, data_root, 'test')
    holdout_files = _files_for(holdout, data_root, 'test')

    ns = compute_norm_stats(train_files, geometries)
    ds_train = FNODataset(train_files, ns, expected_grid=(1, 1, 1),
                          target_grid=common_grid, geometries=geometries)
    ds_val = FNODataset(val_files, ns, expected_grid=(1, 1, 1),
                        target_grid=common_grid, geometries=geometries)
    ds_holdout = FNODataset(holdout_files, ns, expected_grid=(1, 1, 1),
                            target_grid=common_grid, geometries=geometries)

    model = build_fno(grid_shape=common_grid, modes=modes, hidden_ch=channels,
                      n_blocks=blocks, use_geometry_field=use_geometry_field)
    log.info("model params=%d use_geometry_field=%s", model.n_parameters, use_geometry_field)

    tag = "geomfield" if use_geometry_field else "baseline"
    out_dir = Path('checkpoints/fno') / f'leaveout_{holdout}_{tag}_seed{seed}'
    trainer = FNOTrainer(
        model=model, norm_stats=ns, train_data=ds_train, val_data=ds_val,
        output_dir=out_dir, epochs=epochs, geometry_name=out_dir.name,
        log_interval=max(1, epochs // 5),
    )
    trainer.train()

    # Evaluate zero-shot on the held-out geometry with the SAME detrended
    # metrics scripts/baselines.py uses for ridge, for a direct comparison.
    T_range = ns.T_max - ns.T_min
    all_m = []
    model.eval()
    with torch.no_grad():
        for item in ds_holdout.items:
            T_pred_K, T_true_K = predict_to_flat(model, item, torch.device('cpu'), ns)
            nx, ny, nz = common_grid
            # Real physical extents of the HELD-OUT geometry, so hotspot_loc_err_um
            # is actually in micrometers (a normalised unit cube previously gave a
            # dimensionless quantity mislabeled "_um" -- harmless for a same-run
            # baseline-vs-field comparison since both sides shared it, but wrong if
            # ever compared against real-µm numbers like the ridge baselines).
            holdout_geom = geometries[holdout]
            xs = np.linspace(0, holdout_geom.die_width, nx)
            ys = np.linspace(0, holdout_geom.die_length, ny)
            zs = np.linspace(0, holdout_geom.get_total_height(), nz)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
            coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            m = compute_metrics(T_pred_K, T_true_K, coords)
            m['name'] = item['name']
            all_m.append(m)

    agg = {k: float(np.mean([m[k] for m in all_m]))
          for k in ('mae_K', 'mae_detrended_K', 'spatial_r2', 'hotspot_loc_err_um')}
    return agg


_METRICS = ('mae_K', 'mae_detrended_K', 'spatial_r2', 'hotspot_loc_err_um')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', default='geometry4', choices=ALL_GEOMS)
    ap.add_argument('--holdouts', nargs='+', choices=ALL_GEOMS, default=None,
                    help='Repeat the whole experiment for each holdout geometry '
                         '(overrides --holdout). Use to check the B2 result is not '
                         'an artifact of the specific geometry4 choice.')
    ap.add_argument('--data', type=Path, default=Path('data/3d-ice'))
    ap.add_argument('--common-grid', nargs=3, type=int, default=[32, 32, 8])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--channels', type=int, default=16)
    ap.add_argument('--modes', nargs=3, type=int, default=[8, 8, 4])
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='Repeat the whole experiment for each seed (overrides '
                         '--seed). Use to check the B2 result is not a single-run '
                         'artifact.')
    args = ap.parse_args()

    holdouts = args.holdouts or [args.holdout]
    seeds = args.seeds or [args.seed]
    common_grid = tuple(args.common_grid)
    modes = tuple(args.modes)

    # {holdout: {'baseline': [agg,...], 'geomfield': [agg,...]}}
    results: dict = {h: {'baseline': [], 'geomfield': []} for h in holdouts}

    for holdout in holdouts:
        train_geoms = [g for g in ALL_GEOMS if g != holdout]
        for seed in seeds:
            log.info("=== holdout=%s seed=%d: baseline (no geometry-aware field) ===",
                     holdout, seed)
            baseline = run_one(False, train_geoms, holdout, args.data, common_grid,
                               args.epochs, args.channels, modes, args.blocks, seed)
            log.info("holdout=%s seed=%d baseline: %s", holdout, seed, baseline)
            results[holdout]['baseline'].append(baseline)

            log.info("=== holdout=%s seed=%d: with geometry-aware field ===",
                     holdout, seed)
            geomfield = run_one(True, train_geoms, holdout, args.data, common_grid,
                                args.epochs, args.channels, modes, args.blocks, seed)
            log.info("holdout=%s seed=%d geom-field: %s", holdout, seed, geomfield)
            results[holdout]['geomfield'].append(geomfield)

    print("\n=== RESULT (mean over %d seed(s): %s) ===" % (len(seeds), seeds))
    for holdout in holdouts:
        b_runs = results[holdout]['baseline']
        g_runs = results[holdout]['geomfield']
        print(f"\n--- holdout={holdout} ---")
        print(f"{'metric':<20} {'baseline (mean±std)':>24} {'geom-field (mean±std)':>24} {'wins':>6}")
        for k in _METRICS:
            b_vals = np.array([r[k] for r in b_runs])
            g_vals = np.array([r[k] for r in g_runs])
            b_str = f"{b_vals.mean():.4f}±{b_vals.std():.4f}"
            g_str = f"{g_vals.mean():.4f}±{g_vals.std():.4f}"
            better_is_higher = (k == 'spatial_r2')
            g_wins = (g_vals.mean() > b_vals.mean()) if better_is_higher else (g_vals.mean() < b_vals.mean())
            print(f"{k:<20} {b_str:>24} {g_str:>24} {'field' if g_wins else 'base':>6}")

    if len(seeds) > 1:
        print("\nPer-seed raw values:")
        for holdout in holdouts:
            for cond, runs in results[holdout].items():
                for seed, r in zip(seeds, runs):
                    print(f"  {holdout} {cond} seed={seed}: " +
                         ", ".join(f"{k}={r[k]:.4f}" for k in _METRICS))


if __name__ == '__main__':
    main()
