"""
CLI entry point for PINN training.

Usage:
    python scripts/train_pinn.py \\
        --geometry geometry1 \\
        --data data/3d-ice \\
        --output checkpoints/ \\
        --epochs 8000 \\
        --n-col 20000 \\
        --fourier-sigma 10.0 \\
        [--device cuda]

The script:
  1. Loads train/test .npz files for the given geometry
  2. Computes or loads normalization statistics
  3. Trains a FourierPINN with curriculum staging
  4. Saves best checkpoint + norm_stats.json to --output
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.core.geometry_builders import get_geometry_by_name
from src.reproducibility import set_seed, add_seed_args
from src.pinn.data_loader import ThermalDataset, NormStats, compute_norm_stats
from src.pinn.model import build_model
from src.pinn.trainer import Trainer


_ALL_GEOMS = ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5']

# CPU-fast preset: ~4× fewer parameters, 4× fewer collocation points, shorter training.
# Target time on i7 (no GPU): ~1.5–3 hr per geometry vs ~6–10 hr at full defaults.
# RAR disabled on CPU (100k-candidate residual eval is too expensive per-update).
_CPU_FAST = {
    'hidden_dim':   128,
    'n_res_blocks': 4,
    'n_col':        5000,
    'epochs':       3000,
    'rar_interval': 0,
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Train PINN thermal surrogate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--geometry', required=True, choices=_ALL_GEOMS,
                   help='Target geometry to train on')
    p.add_argument('--data', type=Path, required=True,
                   help='Root directory containing .npz files')
    p.add_argument('--output', type=Path, default=Path('checkpoints'),
                   help='Directory for checkpoints and logs')

    # Architecture — individual flags override --cpu-fast
    p.add_argument('--hidden-dim', type=int, default=256,
                   help='PINN hidden layer width (256=full, 128=cpu-fast)')
    p.add_argument('--n-res-blocks', type=int, default=6,
                   help='Number of residual blocks (6=full, 4=cpu-fast)')
    p.add_argument('--fourier-sigma', type=float, default=10.0,
                   help='Fourier feature bandwidth (10 for geometry1/3, 20 for geometry2)')

    # Training
    p.add_argument('--epochs', type=int, default=8000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-col', type=int, default=20000,
                   help='Collocation points for PDE loss per training step')

    # RAR-D adaptive collocation sampling
    p.add_argument('--rar-interval', type=int, default=500,
                   help='Epochs between RAR updates (0 = disable RAR)')
    p.add_argument('--rar-add-n', type=int, default=1000,
                   help='Collocation points added per RAR update')
    p.add_argument('--rar-max-col', type=int, default=30000,
                   help='Cap on total collocation set size')
    p.add_argument('--rar-epsilon', type=float, default=0.2,
                   help='RAR floor fraction (0=pure residual-weighted, 1=pure uniform)')
    p.add_argument('--sampling-strategy', default='rar',
                   choices=['rar', 'hessian', 'importance', 'curriculum'],
                   help='Adaptive collocation strategy for RAR updates. '
                        'rar=PDE-residual-proportional (default). '
                        'hessian=curvature(Laplacian-trace)-proportional. '
                        'importance=softmax-temperature resampling of residual. '
                        'curriculum=uniform->residual blend growing over training. '
                        'See src/pinn/sampling.py for literature basis.')
    p.add_argument('--sampling-temperature', type=float, default=1.0,
                   help='Softmax temperature for importance/curriculum strategies '
                        '(lower=sharper toward high-signal points)')
    p.add_argument('--curriculum-warmup-frac', type=float, default=0.2,
                   help='Fraction of epochs to stay at pure uniform sampling '
                        'before ramping toward adaptive (curriculum strategy only)')

    # Convenience presets
    p.add_argument('--cpu-fast', action='store_true',
                   help='Apply CPU-optimised defaults: hidden=128, blocks=4, '
                        'n_col=5000, epochs=3000, rar_interval=0. Individual flags override.')

    p.add_argument('--device', default=None,
                   help='Training device: cuda, cpu (default: auto-detect)')
    p.add_argument('--norm-stats', type=Path, default=None,
                   help='Path to existing norm_stats.json (skip recomputation)')

    # GPU efficiency upgrades
    p.add_argument('--compile', action='store_true',
                   help='Apply torch.compile(model, mode=reduce-overhead). '
                        '2-3x speedup on CUDA via kernel fusion. '
                        'Ignored on CPU. Requires PyTorch 2.0+.')
    p.add_argument('--func-pde', action='store_true',
                   help='Use torch.func vmap+grad for PDE residual instead of '
                        'autograd create_graph=True. Eliminates retained '
                        'computation graphs: ~1.5-2x memory reduction and '
                        '1.5-2x faster PDE step on CUDA. Requires PyTorch 2.0+.')
    add_seed_args(p)
    return p.parse_args()


def _apply_cpu_fast(args) -> None:
    """Apply --cpu-fast defaults for any flags still at their argparse defaults."""
    defaults = {
        'hidden_dim': 256, 'n_res_blocks': 6, 'n_col': 20000,
        'epochs': 8000, 'rar_interval': 500,
    }
    for attr, cpu_val in _CPU_FAST.items():
        if getattr(args, attr) == defaults[attr]:
            setattr(args, attr, cpu_val)
            logging.info("--cpu-fast: %s → %s", attr, cpu_val)


def collect_npz_files(data_dir: Path, geom_name: str, split: str) -> list:
    """Find all .npz files for a geometry and split (train/test)."""
    candidates = list(data_dir.rglob(f'{geom_name}_{split}_*.npz'))
    if not candidates:
        # Also try flat directory
        candidates = list(data_dir.glob(f'{geom_name}_{split}_*.npz'))
    return sorted(candidates)


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    if args.cpu_fast:
        _apply_cpu_fast(args)
    geom_name = args.geometry
    geometry = get_geometry_by_name(geom_name)
    geometries = {geom_name: geometry}

    output_dir = args.output / geom_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data files
    train_files = collect_npz_files(args.data, geom_name, 'train')
    test_files = collect_npz_files(args.data, geom_name, 'test')

    if not train_files:
        logging.error("No training files found in %s for geometry %s", args.data, geom_name)
        sys.exit(1)

    logging.info("Found %d train + %d test scenarios", len(train_files), len(test_files))

    # Normalization stats
    norm_stats_path = output_dir / 'norm_stats.json'
    if args.norm_stats and args.norm_stats.exists():
        norm_stats = NormStats.load(args.norm_stats)
        logging.info("Loaded norm stats from %s", args.norm_stats)
    elif norm_stats_path.exists():
        norm_stats = NormStats.load(norm_stats_path)
        logging.info("Loaded existing norm stats from %s", norm_stats_path)
    else:
        logging.info("Computing normalization statistics over %d training files...", len(train_files))
        norm_stats = compute_norm_stats(train_files, geometries)
        norm_stats.save(norm_stats_path)
        logging.info("Norm stats saved to %s", norm_stats_path)

    logging.info(
        "Norm stats: T=[%.1f, %.1f] K, power_mean=%.2e, power_std=%.2e",
        norm_stats.T_min, norm_stats.T_max, norm_stats.power_mean, norm_stats.power_std
    )

    # Datasets
    train_dataset = ThermalDataset(train_files, geometries, norm_stats)
    val_dataset = ThermalDataset(test_files or train_files[-3:], geometries, norm_stats)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)

    # Model
    n_layers = len(geometry.layers)
    model = build_model(
        n_layers=n_layers,
        fourier_sigma=args.fourier_sigma,
        hidden_dim=args.hidden_dim,
        n_res_blocks=args.n_res_blocks,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(
        "FourierPINN: %d params | hidden=%d blocks=%d n_col=%d epochs=%d",
        n_params, args.hidden_dim, args.n_res_blocks, args.n_col, args.epochs,
    )

    # Train
    trainer = Trainer(
        model=model,
        geometry=geometry,
        norm_stats=norm_stats,
        train_data=train_dataset,
        val_data=val_dataset,
        output_dir=output_dir,
        n_col=args.n_col,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        rar_interval=args.rar_interval,
        rar_add_n=args.rar_add_n,
        rar_max_col=args.rar_max_col,
        rar_epsilon=args.rar_epsilon,
        sampling_strategy=args.sampling_strategy,
        sampling_temperature=args.sampling_temperature,
        curriculum_warmup_frac=args.curriculum_warmup_frac,
        use_compile=args.compile,
        use_func_pde=args.func_pde,
    )
    best_ckpt = trainer.train()
    logging.info("Done. Best checkpoint: %s", best_ckpt)


if __name__ == '__main__':
    main()
