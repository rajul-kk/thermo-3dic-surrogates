"""CLI entry point for PINN evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.core.geometry_builders import get_geometry_by_name
from src.pinn.data_loader import ThermalDataset, NormStats
from src.pinn.model import FourierPINN, build_model
from src.pinn.evaluate import (
    evaluate_dataset, predict_scenario, save_metrics,
    plot_comparison, plot_z_profile,
)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate trained PINN')
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Path to .pt checkpoint from train_pinn.py')
    p.add_argument('--data', type=Path, required=True,
                   help='Root directory containing .npz test files')
    p.add_argument('--output', type=Path, default=Path('results'),
                   help='Output directory for metrics and plots')
    p.add_argument('--split', default='test', choices=['train', 'test'],
                   help='Dataset split to evaluate')
    p.add_argument('--plots', action='store_true',
                   help='Generate comparison plots')
    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    geom_name = ckpt['geometry_name']
    geometry = get_geometry_by_name(geom_name)
    geometries = {geom_name: geometry}

    ns_dict = ckpt['norm_stats']
    norm_stats = NormStats(
        T_min=ns_dict['T_min'],
        T_max=ns_dict['T_max'],
        power_mean=ns_dict['power_mean'],
        power_std=ns_dict['power_std'],
        htc_min=ns_dict.get('htc_min', 500.0),
        htc_max=ns_dict.get('htc_max', 10000.0),
        t_amb_min=ns_dict.get('t_amb_min', 25.0),
        t_amb_max=ns_dict.get('t_amb_max', 85.0),
        geom_extents=ns_dict['geom_extents'],
    )

    n_layers = len(geometry.layers)
    model = build_model(n_layers=n_layers)
    model.load_state_dict(ckpt['model_state'])

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    model.to(device)
    logging.info("Loaded checkpoint from epoch %d (val MAE: %.3f K)",
                 ckpt['epoch'], ckpt['val_mae_K'])

    # Data
    npz_files = sorted(args.data.rglob(f'{geom_name}_{args.split}_*.npz'))
    if not npz_files:
        npz_files = sorted(args.data.glob(f'{geom_name}_{args.split}_*.npz'))

    if not npz_files:
        logging.error("No %s files found for %s in %s", args.split, geom_name, args.data)
        sys.exit(1)

    dataset = ThermalDataset(npz_files, geometries, norm_stats)
    dataset.to_device(device)

    # Evaluate
    results = evaluate_dataset(model, dataset, norm_stats, device)
    metrics_path = args.output / 'metrics.json'
    save_metrics(results, metrics_path)
    logging.info("Metrics saved to %s", metrics_path)

    agg = results['_aggregate']
    print(f"\n=== {geom_name} evaluation ({args.split} set) ===")
    print(f"  Mean MAE:    {agg['mean_mae_K']:.3f} K")
    print(f"  Max  MAE:    {agg['max_mae_K']:.3f} K")
    print(f"  Mean RMSE:   {agg['mean_rmse_K']:.3f} K")
    print(f"  Mean R²:     {agg['mean_r2']:.5f}")

    # Plots
    if args.plots:
        logging.info("Generating comparison plots...")
        for sc in dataset.scenarios:
            T_pred_K, T_true_K = predict_scenario(model, sc, device, norm_stats)

            plot_comparison(
                T_pred_K, T_true_K,
                sc.coords.cpu().numpy(),
                sc.geom_extents,
                sc.name,
                args.output / f'comparison_{sc.name}.png',
            )
            plot_z_profile(
                T_pred_K, T_true_K,
                sc.coords.cpu().numpy(),
                sc.geom_extents,
                geometry, sc.name,
                args.output / f'z_profile_{sc.name}.png',
            )
        logging.info("Plots saved to %s", args.output)


if __name__ == '__main__':
    main()
