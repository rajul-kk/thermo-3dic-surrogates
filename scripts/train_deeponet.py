"""
CLI entry point for PI-DeepONet training.

One model — all 5 uniform-stack geometries simultaneously.
geometry4/5 (2.5D chiplet assemblies with lateral conductivity variation) are
excluded: the trunk's (x,y,z,layer_id) coordinates cannot represent sharp
temperature gradients at chiplet boundaries without explicit region encoding.
geometry4/5 use per-geometry CNO-FNO models instead.

Usage
-----
# Train cross-geometry PI-DeepONet on all 5 stack geometries (default)
python scripts/train_deeponet.py \\
    --data data/3d-ice \\
    --output checkpoints/deeponet/ \\
    --epochs 1000 \\
    --pde-weight 0.1

# Data-only baseline (no physics loss) — pure DeepONet
python scripts/train_deeponet.py \\
    --data data/3d-ice \\
    --output checkpoints/deeponet/ \\
    --pde-weight 0.0

# Subset of geometries (e.g. to compare against single-geometry FNO)
python scripts/train_deeponet.py \\
    --geometries geometry1 geometry3 \\
    --data data/3d-ice \\
    --output checkpoints/deeponet/

# CPU-fast (smaller model, fewer epochs)
python scripts/train_deeponet.py \\
    --data data/3d-ice \\
    --output checkpoints/deeponet/ \\
    --cpu-fast

Key advantages over per-geometry FNO
-------------------------------------
- Single model: train once, evaluate on all 5 stack geometries
- Arbitrary query resolution: trunk evaluates at any (x,y,z), not fixed grid
- Better parametric extrapolation via physics loss on trunk gradients
- ~3x cheaper per epoch than FNO (no 3D FFT overhead)
"""

import argparse
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
from src.reproducibility import set_seed, add_seed_args
from src.pinn.data_loader import NormStats, compute_norm_stats
from src.deeponet.model import build_deeponet
from src.deeponet.cno_model import build_cno_deeponet
from src.deeponet.data_loader import MultiGeomDataset
from src.deeponet.trainer import DeepONetTrainer

# MLP branch: uniform-stack geometries (trunk assumes layer-uniform T in x,y)
ALL_GEOMS_MLP = ['geometry1', 'geometry2a', 'geometry3']
# CNO branch: all 6 geometries (spatial encoder captures lateral variation)
ALL_GEOMS_CNO = ['geometry1', 'geometry2a',
                 'geometry3', 'geometry4', 'geometry5', 'geometry6']
_ALL_GEOMS_WITH_2P5D = ALL_GEOMS_MLP + ['geometry4', 'geometry5', 'geometry6']

ALL_GEOMS = ALL_GEOMS_MLP  # backward-compat default

_CPU_FAST_MLP = {
    'n_basis':       64,
    'branch_hidden': 128,
    'trunk_hidden':  128,
    'epochs':        300,
    'n_data':        2048,
    'n_col':         1024,
}
_CPU_FAST_CNO = {
    'n_basis':    64,
    'ch':         32,
    'trunk_hidden': 128,
    'epochs':     300,
    'n_data':     2048,
    'n_col':      1024,
}
_CPU_FAST = _CPU_FAST_MLP  # resolved in main() based on --model


def parse_args():
    p = argparse.ArgumentParser(
        description='Train PI-DeepONet across multiple 3D-IC geometries',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model', choices=['mlp', 'cno'], default='mlp',
                   help='Branch architecture. '
                        '"mlp" = original PI-DeepONet (fixed 16×16 sensor MLP, 5 geometries). '
                        '"cno" = PI-CNO-DeepONet (full 3D spatial encoder, all 8 geometries). '
                        'cno is recommended: no sensor information loss, cross-geometry '
                        'including g4/5/6 lateral-variation chiplet stacks.')
    p.add_argument('--geometries', nargs='+', default=None,
                   choices=_ALL_GEOMS_WITH_2P5D, metavar='GEOM',
                   help='Geometries to include. Default: 5 stack geoms for --model mlp, '
                        'all 8 for --model cno.')
    p.add_argument('--include-chiplets', action='store_true',
                   help='(--model mlp only) Add geometry4/5/6 to the geometry list')
    p.add_argument('--data',   type=Path, required=True,
                   help='Root data directory containing .npz files')
    p.add_argument('--output', type=Path, default=Path('checkpoints/deeponet'),
                   help='Output directory for checkpoints')
    p.add_argument('--name',   type=str, default='deeponet',
                   help='Model name prefix for checkpoint files')

    # Architecture — shared
    p.add_argument('--n-basis',       type=int, default=128,
                   help='DeepONet basis dimension (branch/trunk output dim)')
    p.add_argument('--trunk-hidden',  type=int, default=256,
                   help='Trunk MLP hidden width')
    p.add_argument('--trunk-layers',  type=int, default=4)
    p.add_argument('--fourier-sigma', type=float, default=10.0,
                   help='Trunk Fourier encoding bandwidth')
    # Architecture — MLP branch only
    p.add_argument('--branch-hidden', type=int, default=256,
                   help='(--model mlp) Branch MLP hidden width')
    p.add_argument('--branch-layers', type=int, default=4,
                   help='(--model mlp) Branch MLP depth')
    # Architecture — CNO branch only
    p.add_argument('--ch',          type=int, default=64,
                   help='(--model cno) Branch channel width (default 64)')
    p.add_argument('--fno-blocks',  type=int, default=4,
                   help='(--model cno) FiLM-FNO blocks in branch latent')
    p.add_argument('--cno-layers',  type=int, default=2,
                   help='(--model cno) CNN encoder downsampling stages')

    # Training
    p.add_argument('--epochs',     type=int,   default=1000)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--batch-size', type=int,   default=4,
                   help='Scenarios per gradient step')
    p.add_argument('--n-data',     type=int,   default=4096,
                   help='Data points per scenario per step (random subset)')
    p.add_argument('--n-col',      type=int,   default=2048,
                   help='Collocation points per scenario per step (PDE loss)')

    # Physics
    p.add_argument('--pde-weight', type=float, default=0.1,
                   help='Weight for PDE residual loss (0=pure data-driven)')

    # Convenience
    p.add_argument('--cpu-fast', action='store_true',
                   help='Apply CPU-optimised defaults: n_basis=64, hidden=128, '
                        'epochs=300, n_data=2048. Individual flags override.')
    p.add_argument('--device',    default=None,
                   help='Training device: cuda, cpu (default: auto-detect)')
    p.add_argument('--norm-stats', type=Path, default=None,
                   help='Path to existing norm_stats.json (skip recomputation)')
    add_seed_args(p)
    return p.parse_args()


def _apply_cpu_fast(args) -> None:
    cpu_fast = _CPU_FAST_CNO if args.model == 'cno' else _CPU_FAST_MLP
    defaults = {'n_basis': 128, 'branch_hidden': 256, 'trunk_hidden': 256,
                'epochs': 1000, 'n_data': 4096, 'n_col': 2048, 'ch': 64}
    for attr, val in cpu_fast.items():
        if getattr(args, attr, None) == defaults.get(attr):
            setattr(args, attr, val)
            logging.info("--cpu-fast: %s -> %s", attr, val)


def collect_files(data_dir: Path, geom_name: str, split: str) -> list:
    files = list(data_dir.rglob(f'{geom_name}_{split}_*.npz'))
    if not files:
        files = list(data_dir.glob(f'{geom_name}_{split}_*.npz'))
    return sorted(files)


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    # Resolve geometry list based on model type and flags
    if args.geometries is None:
        args.geometries = ALL_GEOMS_CNO if args.model == 'cno' else ALL_GEOMS_MLP
    if args.include_chiplets and args.model == 'mlp':
        for g in ['geometry4', 'geometry5', 'geometry6']:
            if g not in args.geometries:
                args.geometries = args.geometries + [g]
        logging.info("--include-chiplets: %s", args.geometries)

    if args.cpu_fast:
        _apply_cpu_fast(args)

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    logging.info("Model: %s | Device: %s", args.model, device)

    # Load all requested geometries
    geometries = {n: get_geometry_by_name(n) for n in args.geometries}
    logging.info("Geometries (%d): %s", len(geometries), list(geometries.keys()))

    # Collect files
    train_files, test_files = [], []
    for name in args.geometries:
        train_files += collect_files(args.data, name, 'train')
        test_files  += collect_files(args.data, name, 'test')

    if not train_files:
        logging.error("No training files found in %s", args.data)
        sys.exit(1)
    logging.info("Files: %d train + %d test", len(train_files), len(test_files))

    # Norm stats (computed over all geometries together for a shared scale)
    output_dir = args.output / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_path = output_dir / 'norm_stats.json'

    if args.norm_stats and Path(args.norm_stats).exists():
        norm_stats = NormStats.load(args.norm_stats)
        logging.info("Loaded norm stats from %s", args.norm_stats)
    elif norm_path.exists():
        norm_stats = NormStats.load(norm_path)
        logging.info("Loaded existing norm stats from %s", norm_path)
    else:
        logging.info("Computing norm stats over %d train files ...", len(train_files))
        norm_stats = compute_norm_stats(train_files, geometries)
        norm_stats.save(norm_path)
        logging.info("Norm stats saved to %s", norm_path)

    logging.info(
        "Norm stats: T=[%.1f, %.1f] K  power_std=%.2e",
        norm_stats.T_min, norm_stats.T_max, norm_stats.power_std,
    )

    # Datasets
    train_dataset = MultiGeomDataset(train_files, geometries, norm_stats)
    val_dataset   = MultiGeomDataset(test_files or train_files[-5:], geometries, norm_stats)

    # Model
    if args.model == 'cno':
        model = build_cno_deeponet(
            n_basis      = args.n_basis,
            ch           = args.ch,
            n_fno_blocks = args.fno_blocks,
            n_cno_layers = args.cno_layers,
            trunk_hidden = args.trunk_hidden,
            trunk_layers = args.trunk_layers,
            fourier_sigma= args.fourier_sigma,
            device       = device,
        )
        logging.info(
            "PICNODeepONet: n_basis=%d ch=%d fno_blocks=%d trunk=%dx%d "
            "params=%d pde_weight=%.3f geoms=%d",
            args.n_basis, args.ch, args.fno_blocks,
            args.trunk_hidden, args.trunk_layers,
            model.n_parameters, args.pde_weight, len(geometries),
        )
    else:
        model = build_deeponet(
            n_basis       = args.n_basis,
            branch_hidden = args.branch_hidden,
            trunk_hidden  = args.trunk_hidden,
            branch_layers = args.branch_layers,
            trunk_layers  = args.trunk_layers,
            fourier_sigma = args.fourier_sigma,
            device        = device,
        )
        logging.info(
            "PIDeepONet: n_basis=%d branch=%dx%d trunk=%dx%d params=%d pde_weight=%.3f",
            args.n_basis, args.branch_hidden, args.branch_layers,
            args.trunk_hidden, args.trunk_layers,
            model.n_parameters, args.pde_weight,
        )

    # Train
    trainer = DeepONetTrainer(
        model        = model,
        geometries   = geometries,
        norm_stats   = norm_stats,
        train_data   = train_dataset,
        val_data     = val_dataset,
        output_dir   = output_dir,
        batch_size   = args.batch_size,
        epochs       = args.epochs,
        lr           = args.lr,
        pde_weight   = args.pde_weight,
        n_data       = args.n_data,
        n_col        = args.n_col,
        device       = device,
        model_name   = args.name,
    )
    best_ckpt = trainer.train()
    logging.info("Best checkpoint: %s", best_ckpt)


if __name__ == '__main__':
    main()
