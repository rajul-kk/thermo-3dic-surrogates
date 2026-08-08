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

    out_dir = Path('checkpoints/fno') / f'leaveout_{holdout}_{"geomfield" if use_geometry_field else "baseline"}'
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
            # Coords only used for the hotspot-distance term; a normalised unit
            # cube is fine here since we only compare relative locations within
            # this same resampled grid.
            xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny); zs = np.linspace(0, 1, nz)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
            coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            m = compute_metrics(T_pred_K, T_true_K, coords)
            m['name'] = item['name']
            all_m.append(m)

    agg = {k: float(np.mean([m[k] for m in all_m]))
          for k in ('mae_K', 'mae_detrended_K', 'spatial_r2', 'hotspot_loc_err_um')}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', default='geometry4', choices=ALL_GEOMS)
    ap.add_argument('--data', type=Path, default=Path('data/3d-ice'))
    ap.add_argument('--common-grid', nargs=3, type=int, default=[32, 32, 8])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--channels', type=int, default=16)
    ap.add_argument('--modes', nargs=3, type=int, default=[8, 8, 4])
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    train_geoms = [g for g in ALL_GEOMS if g != args.holdout]
    common_grid = tuple(args.common_grid)
    modes = tuple(args.modes)

    log.info("Leave-one-geometry-out: train=%s  holdout=%s  grid=%s",
            train_geoms, args.holdout, common_grid)

    log.info("=== Baseline (no geometry-aware field) ===")
    baseline = run_one(False, train_geoms, args.holdout, args.data, common_grid,
                       args.epochs, args.channels, modes, args.blocks, args.seed)
    log.info("Baseline zero-shot on %s: %s", args.holdout, baseline)

    log.info("=== With geometry-aware field ===")
    geomfield = run_one(True, train_geoms, args.holdout, args.data, common_grid,
                        args.epochs, args.channels, modes, args.blocks, args.seed)
    log.info("Geometry-field zero-shot on %s: %s", args.holdout, geomfield)

    print("\n=== RESULT ===")
    print(f"{'metric':<20} {'baseline':>12} {'geom-field':>12}")
    for k in ('mae_K', 'mae_detrended_K', 'spatial_r2', 'hotspot_loc_err_um'):
        print(f"{k:<20} {baseline[k]:>12.4f} {geomfield[k]:>12.4f}")


if __name__ == '__main__':
    main()
