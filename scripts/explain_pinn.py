"""
CLI entry point for PINN explainability analysis.

Runs three complementary explainability methods on a trained checkpoint:

  Option 1 — PDE residual maps: where does the model violate the heat equation?
  Option 2 — Engineering sensitivity maps: power block influence + HTC sensitivity
  Option 5 — MC Dropout uncertainty: where is the model uncertain?

Usage:
    python scripts/explain_pinn.py \\
        --checkpoint checkpoints/geometry1/geometry1_best.pt \\
        --data data/3d-ice \\
        --output results/explain/geometry1 \\
        --split test

    # Only specific methods:
    python scripts/explain_pinn.py \\
        --checkpoint checkpoints/geometry1/geometry1_best.pt \\
        --data data/3d-ice \\
        --output results/explain/geometry1 \\
        --methods residual sensitivity

    # Control MC Dropout samples (fewer = faster but noisier):
    python scripts/explain_pinn.py ... --mc-samples 50

Outputs per scenario (--split test, default):
    pde_residual_<scenario>.png        PDE residual at die z-slice
    sensitivity_block_<b>_<scenario>.png   per-block dT/dQ influence map
    sensitivity_htc_<scenario>.png     cooling effectiveness map
    uncertainty_<scenario>.png         MC Dropout predictive std
    explain_summary.json               aggregate statistics

Notes:
    MC Dropout requires the model to have been trained with dropout_p > 0
    (the default in build_model). Models trained with dropout_p=0 will
    produce near-zero uncertainty estimates.
"""

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

import numpy as np
import torch

from src.core.geometry_builders import get_geometry_by_name
from src.pinn.data_loader import ThermalDataset, NormStats
from src.pinn.model import build_model
from src.pinn.explain import (
    residual_map, plot_residual_map,
    power_block_sensitivity, htc_sensitivity_map, plot_sensitivity_map,
    mc_dropout_uncertainty, plot_uncertainty_map,
    hotspot_ig, plot_ig_attribution,
)

_ALL_METHODS = ['residual', 'sensitivity', 'ig', 'uncertainty']


def parse_args():
    p = argparse.ArgumentParser(
        description='PINN explainability: PDE residual, sensitivity, and uncertainty maps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Path to .pt checkpoint')
    p.add_argument('--data', type=Path, required=True,
                   help='Root directory containing .npz files')
    p.add_argument('--output', type=Path, default=Path('results/explain'),
                   help='Output directory')
    p.add_argument('--split', default='test', choices=['train', 'test'])
    p.add_argument('--methods', nargs='+', choices=_ALL_METHODS,
                   default=_ALL_METHODS,
                   help='Which explainability methods to run')
    p.add_argument('--mc-samples', type=int, default=100,
                   help='MC Dropout forward passes (fewer = faster but noisier)')
    p.add_argument('--ig-steps', type=int, default=50,
                   help='IG path integration steps (50=standard, 100=publication quality)')
    p.add_argument('--z-fraction', type=float, default=0.95,
                   help='Z-slice position as fraction of domain height for plots')
    p.add_argument('--device', default=None)
    return p.parse_args()


def _load_norm_stats(ns_dict: dict) -> NormStats:
    return NormStats(
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


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    geom_name = ckpt['geometry_name']
    geometry = get_geometry_by_name(geom_name)
    geometries = {geom_name: geometry}
    norm_stats = _load_norm_stats(ckpt['norm_stats'])

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    logging.info("Device: %s | Geometry: %s | Methods: %s",
                 device, geom_name, args.methods)

    n_layers = len(geometry.layers)
    model = build_model(n_layers=n_layers)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    logging.info("Loaded checkpoint epoch=%d val_MAE=%.3f K",
                 ckpt['epoch'], ckpt['val_mae_K'])

    # ── Load data ────────────────────────────────────────────────────────────
    npz_files = sorted(args.data.rglob(f'{geom_name}_{args.split}_*.npz'))
    if not npz_files:
        npz_files = sorted(args.data.glob(f'{geom_name}_{args.split}_*.npz'))
    if not npz_files:
        logging.error("No %s files for %s in %s", args.split, geom_name, args.data)
        sys.exit(1)

    dataset = ThermalDataset(npz_files, geometries, norm_stats)
    dataset.to_device(device)
    logging.info("Loaded %d scenarios", len(dataset))

    # ── Per-scenario analysis ─────────────────────────────────────────────────
    summary = {'geometry': geom_name, 'scenarios': {}}

    for sc in dataset.scenarios:
        logging.info("  Processing %s ...", sc.name)
        sc_summary: dict = {}
        coords_np = sc.coords.cpu().numpy()

        # ── Option 1: PDE residual map ──────────────────────────────────────
        if 'residual' in args.methods:
            res = residual_map(model, sc, geometry, norm_stats, device)
            sc_summary['pde_residual_mean'] = float(res.mean())
            sc_summary['pde_residual_max']  = float(res.max())
            sc_summary['pde_residual_p95']  = float(np.percentile(res, 95))

            plot_residual_map(
                res, coords_np, sc.geom_extents,
                sc.name,
                args.output / f'pde_residual_{sc.name}.png',
                z_fraction=args.z_fraction,
            )
            logging.info("    residual: mean=%.3e max=%.3e", res.mean(), res.max())

        # ── Option 2: Sensitivity maps ──────────────────────────────────────
        if 'sensitivity' in args.methods:
            # Power block influence
            block_sens = power_block_sensitivity(
                model, sc, geometry, norm_stats, device
            )
            sc_summary['power_sensitivity'] = {}
            for bname, sens in block_sens.items():
                peak_k_per_wcm2 = float(np.abs(sens).max())
                sc_summary['power_sensitivity'][bname] = {
                    'peak_dT_K_per_wcm2': peak_k_per_wcm2,
                    'mean_dT_K_per_wcm2': float(sens.max()),  # max of signed
                }
                plot_sensitivity_map(
                    sens, coords_np, sc.geom_extents,
                    title=f'dT/dQ — block {bname} — {sc.name}',
                    output_path=args.output / f'sensitivity_block_{bname}_{sc.name}.png',
                    z_fraction=args.z_fraction,
                    clabel='dT/dQ (K per W/cm²)',
                )
                logging.info("    block %s: peak=%.2f K/(W/cm²)", bname, peak_k_per_wcm2)

            # HTC sensitivity (cooling effectiveness)
            htc_sens = htc_sensitivity_map(model, sc, norm_stats, device)
            sc_summary['htc_sensitivity_mean_K_per_wm2k'] = float(htc_sens.mean())
            sc_summary['htc_sensitivity_min_K_per_wm2k']  = float(htc_sens.min())
            plot_sensitivity_map(
                htc_sens, coords_np, sc.geom_extents,
                title=f'dT/dHTC — {sc.name}',
                output_path=args.output / f'sensitivity_htc_{sc.name}.png',
                z_fraction=args.z_fraction,
                clabel='dT/dHTC (K per W/m²·K)',
            )
            logging.info("    HTC sensitivity: mean=%.4f K/(W/m²·K)", htc_sens.mean())

        # ── Option 3: Integrated Gradients at hotspot ───────────────────────
        if 'ig' in args.methods:
            ig_result = hotspot_ig(
                model, sc, norm_stats, device,
                n_steps=args.ig_steps,
                include_z_profile=True,
            )
            sc_summary['ig_hotspot_T_K']        = ig_result['hotspot_T_K']
            sc_summary['ig_baseline_T_K']       = ig_result['baseline_T_K']
            sc_summary['ig_completeness_error'] = ig_result['completeness_error']
            sc_summary['ig_hotspot_attrs']      = ig_result['hotspot_attrs']

            plot_ig_attribution(
                ig_result, sc.name,
                args.output / f'ig_attribution_{sc.name}.png',
                geometry=geometry,
            )
            top_feat = max(ig_result['hotspot_attrs'], key=lambda k: abs(ig_result['hotspot_attrs'][k]))
            logging.info(
                "    IG: T_hs=%.1f K  top_feature=%s (%.1f K)  completeness_err=%.3f",
                ig_result['hotspot_T_K'],
                top_feat, ig_result['hotspot_attrs'][top_feat],
                ig_result['completeness_error'],
            )

        # ── Option 5: MC Dropout uncertainty ────────────────────────────────
        if 'uncertainty' in args.methods:
            T_mean_K, T_std_K = mc_dropout_uncertainty(
                model, sc, norm_stats, device, n_samples=args.mc_samples
            )
            sc_summary['mc_dropout_mean_std_K'] = float(T_std_K.mean())
            sc_summary['mc_dropout_max_std_K']  = float(T_std_K.max())
            sc_summary['mc_dropout_p95_std_K']  = float(np.percentile(T_std_K, 95))

            plot_uncertainty_map(
                T_std_K, coords_np, sc.geom_extents,
                sc.name,
                args.output / f'uncertainty_{sc.name}.png',
                z_fraction=args.z_fraction,
            )
            logging.info("    MC Dropout: mean_std=%.3f K max_std=%.3f K",
                         T_std_K.mean(), T_std_K.max())

        summary['scenarios'][sc.name] = sc_summary

    # ── Aggregate summary ────────────────────────────────────────────────────
    all_sc = list(summary['scenarios'].values())
    if 'residual' in args.methods and all_sc:
        summary['aggregate_residual'] = {
            'mean_residual_mean': float(np.mean([s['pde_residual_mean'] for s in all_sc])),
            'mean_residual_max':  float(np.mean([s['pde_residual_max']  for s in all_sc])),
        }
    if 'uncertainty' in args.methods and all_sc:
        summary['aggregate_uncertainty'] = {
            'mean_std_K': float(np.mean([s['mc_dropout_mean_std_K'] for s in all_sc])),
            'max_std_K':  float(np.max( [s['mc_dropout_max_std_K']  for s in all_sc])),
        }

    summary_path = args.output / 'explain_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info("Summary saved to %s", summary_path)

    print(f"\n=== Explainability summary — {geom_name} ({args.split}) ===")
    if 'residual' in args.methods and 'aggregate_residual' in summary:
        r = summary['aggregate_residual']
        print(f"  PDE residual (mean over scenarios): {r['mean_residual_mean']:.3e}")
    if 'uncertainty' in args.methods and 'aggregate_uncertainty' in summary:
        u = summary['aggregate_uncertainty']
        print(f"  MC Dropout uncertainty: mean={u['mean_std_K']:.3f} K  max={u['max_std_K']:.3f} K")
    print(f"  Outputs written to {args.output}/")


if __name__ == '__main__':
    main()
