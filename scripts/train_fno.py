"""
CLI entry point for FNO training.

Model variants (--model):
    fno       Baseline FNO3d. Fast, data-only, no physics or BC conditioning.
    cond-fno  FiLM-conditioned FNO. HTC/T_amb/TSV_frac modulate spectral
              filters via a hypernetwork. Better BC extrapolation than fno.
    cno-fno   CNO-FNO hybrid (default). CNN encoder/decoder around a
              FiLM-FNO in latent space. Best accuracy: sharp interfaces
              (CNO) + global spreading (FNO) + BC conditioning (FiLM).
              Physics loss enabled by default.

Usage:
    # Recommended (CNO-FNO + physics, default model):
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice

    # Higher capacity (publication runs):
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --channels 64

    # Baseline FNO for ablation:
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --model fno

    # All geometries:
    python scripts/train_fno.py --all-geometries --data data/3d-ice

    # CPU-only (reduced defaults):
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --cpu-fast
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

from src.core.geometry_builders import get_geometry_by_name, build_all_geometries
from src.reproducibility import set_seed, add_seed_args
from src.pinn.data_loader import NormStats, compute_norm_stats
from src.fno.model import build_fno, build_cond_fno, build_cno_fno
from src.fno.whno import build_whno
from src.fno.data_loader import FNODataset
from src.fno.trainer import FNOTrainer

ALL_GEOMS = ['geometry1', 'geometry2a',
             'geometry3', 'geometry4', 'geometry5', 'geometry6']

# Per-model defaults
_MODEL_DEFAULTS = {
    'fno':      {'epochs': 500, 'physics': False},
    'cond-fno': {'epochs': 500, 'physics': False},
    'cno-fno':  {'epochs': 400, 'physics': True},
    'whno':     {'epochs': 500, 'physics': False},
}

_PHYSICS_WEIGHT = 0.1   # applied when physics is enabled

# CPU-fast preset — reduces compute to ~8-15h per geometry on CPU.
# Prefer PINN on CPU; FNO cpu-fast is for weekend ablation runs only.
_CPU_FAST = {
    'channels': 16,
    'modes':    [8, 8, 6],
    'blocks':   3,
    'epochs':   100,
    'batch_size': 2,
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Train FNO thermal surrogate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Geometry selection ---
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--geometry', nargs='+', choices=ALL_GEOMS, metavar='GEOM',
                     help='One or more geometries (must share a grid shape unless --common-grid is given)')
    grp.add_argument('--all-geometries', action='store_true',
                     help='Train one model per geometry sequentially')

    # --- Required paths ---
    p.add_argument('--data',   type=Path, required=True,
                   help='Root directory containing .npz training files')
    p.add_argument('--output', type=Path, default=Path('checkpoints/fno'),
                   help='Output directory for checkpoints')
    p.add_argument('--name',   type=str, default=None,
                   help='Model name override (default: geometry name)')

    # --- Model architecture ---
    p.add_argument('--model', choices=['fno', 'cond-fno', 'cno-fno', 'whno'], default='cno-fno',
                   help='Model variant: fno (baseline), cond-fno (FiLM conditioning), '
                        'cno-fno (CNN+FiLM+latent-FNO, default and recommended), '
                        'whno (Walsh-Hadamard spectral basis — no Gibbs ringing at '
                        'sharp material-interface discontinuities, trade-off: no '
                        'smooth-frequency prior for bulk regions; see src/fno/whno.py)')
    p.add_argument('--channels', type=int, default=32,
                   help='Hidden channel width. 32=default, 64=publication accuracy '
                        '(~4x more spectral parameters). Ignored by fno/cond-fno '
                        'which use --modes for capacity instead.')
    p.add_argument('--modes', type=int, nargs=3, default=[16, 16, 12],
                   metavar=('MX', 'MY', 'MZ'),
                   help='Spectral modes per dimension (fno/cond-fno only; '
                        'cno-fno computes modes from its latent grid size)')
    p.add_argument('--blocks', type=int, default=4,
                   help='Number of FNO blocks')
    p.add_argument('--attention', action='store_true', default=False,
                   help='(cno-fno only) Add axial self-attention (SAU-FNO style) to '
                        'each latent FiLM-FNO block. Uses x->y->z factorised attention '
                        '(~112 KB/sample) + FlashAttention on CUDA via PyTorch 2.x. '
                        'Adds ~1M params and ~20%% compute overhead.')
    p.add_argument('--n-heads', type=int, default=4,
                   help='(--attention only) Number of attention heads. '
                        'Must divide --channels evenly (default 4 -> head_dim = ch/4).')

    # --- Physics loss ---
    p.add_argument('--physics', action=argparse.BooleanOptionalAction, default=None,
                   help='Enable finite-difference PDE residual loss '
                        '(default: on for cno-fno, off for fno/cond-fno). '
                        'Use --no-physics to override.')
    p.add_argument('--flux-weight', type=float, default=0.0,
                   help='Weight for interface-isolated flux-continuity loss '
                        '(0=disabled). Independent of --physics/--pde-weight -- '
                        'directly checks k1*dT/dz == k2*dT/dz at material '
                        'interfaces using one-sided differences on each side, '
                        'rather than the whole-volume harmonic-mean stencil. '
                        'See interface_flux_loss() in src/fno/physics.py.')

    # --- Training hyperparameters ---
    p.add_argument('--epochs',     type=int,   default=None,
                   help='Training epochs (default: 400 for cno-fno, 500 for others)')
    p.add_argument('--batch-size', type=int,   default=4)
    p.add_argument('--lr',         type=float, default=1e-3)

    # --- Convenience presets ---
    p.add_argument('--cpu-fast', action='store_true',
                   help='CPU-optimised preset: channels=16, modes=8,8,6, '
                        'blocks=3, epochs=100, batch=2')

    # --- Acceleration ---
    p.add_argument('--compile', action='store_true', default=False,
                   help='Apply torch.compile (PyTorch 2.x) for ~30-40%% speedup on T4. '
                        'Requires CUDA. Uses fullgraph=False to handle dynamic branches.')
    p.add_argument('--patience', type=int, default=0,
                   help='Early stopping patience in epochs (0 = disabled). '
                        'Stops training if val_MAE does not improve for this many '
                        'log-interval checks. Saves significant time when geometry '
                        'converges before the epoch budget.')

    # --- Device / paths ---
    p.add_argument('--device',     default=None,
                   help='Training device (default: auto-detect cuda/cpu). '
                        'Use cuda:0 / cuda:1 to target a specific GPU on multi-GPU nodes.')
    p.add_argument('--norm-stats', type=Path, default=None,
                   help='Path to existing norm_stats.json (skip recomputation)')

    p.add_argument('--common-grid', nargs=3, type=int, metavar=('NX','NY','NZ'),
                   default=None,
                   help='Resample all geometries onto this grid so one FNO can train '
                        'across mesh shapes, e.g. --common-grid 64 64 16. Physical '
                        'extents are passed as conditioning so geometries stay '
                        'distinguishable. Round-trip error measured at <0.07 K RMSE.')
    add_seed_args(p)
    return p.parse_args()


def _resolve_defaults(args) -> None:
    """Fill in model-specific defaults for flags the user left unset."""
    model_defs = _MODEL_DEFAULTS[args.model]

    if args.epochs is None:
        args.epochs = model_defs['epochs']

    if args.physics is None:
        args.physics = model_defs['physics']

    if args.cpu_fast:
        cpu_defs = {
            'channels':   32,
            'modes':      [16, 16, 12],
            'blocks':     4,
            'epochs':     model_defs['epochs'],
            'batch_size': 4,
        }
        for attr, cpu_val in _CPU_FAST.items():
            if getattr(args, attr) == cpu_defs.get(attr, getattr(args, attr)):
                setattr(args, attr, cpu_val)
                logging.info("--cpu-fast: %s -> %s", attr, cpu_val)


def collect_files(data_dir: Path, geom_name: str, split: str) -> list:
    files = list(data_dir.rglob(f'{geom_name}_{split}_*.npz'))
    if not files:
        files = list(data_dir.glob(f'{geom_name}_{split}_*.npz'))
    return sorted(files)


def run_single(geom_names: list, args, device: torch.device, model_name: str) -> None:
    geometries = {n: get_geometry_by_name(n) for n in geom_names}

    grid_shapes = {g.mesh_resolution for g in geometries.values()}
    if args.common_grid:
        # Resample every geometry onto one grid so a single FNO can span them.
        # Physical extents ride along as conditioning (FNODataset.geom_extent_norm),
        # so geometries stay distinguishable after resampling.
        grid_shape = tuple(args.common_grid)
        target_grid = grid_shape
        logging.info("Resampling %d geometries onto common grid %s: %s",
                     len(geometries), grid_shape, sorted(geometries))
    elif len(grid_shapes) > 1:
        logging.error(
            "Geometries have different mesh resolutions: %s. "
            "Pass --common-grid NX NY NZ to resample them onto a shared grid, "
            "or train per-geometry.",
            {n: g.mesh_resolution for n, g in geometries.items()}
        )
        sys.exit(1)
    else:
        grid_shape = next(iter(grid_shapes))
        target_grid = None
    output_dir = args.output / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = [], []
    for name in geom_names:
        train_files += collect_files(args.data, name, 'train')
        test_files  += collect_files(args.data, name, 'test')

    if not train_files:
        logging.error("No training files found for %s in %s", geom_names, args.data)
        sys.exit(1)
    logging.info("Files: %d train, %d test", len(train_files), len(test_files))

    if target_grid is None:
        # Take the z-depth from the DATA, not from geometry.mesh_resolution. 3D-ICE
        # emits one value per stack element, which has never equalled the declared
        # mesh nz (40 declared vs 6 emitted originally, 10 after sub-layer
        # discretisation). Trusting the declaration made the loader reject every
        # file once subdivision landed.
        import numpy as _np
        _d = _np.load(train_files[0], allow_pickle=True)
        _n = _d['coords'].shape[0]
        _nx, _ny = grid_shape[0], grid_shape[1]
        if _nx * _ny > 0 and _n % (_nx * _ny) == 0:
            data_grid = (_nx, _ny, _n // (_nx * _ny))
            if data_grid != tuple(grid_shape):
                logging.info("Grid from data: %s (geometry declares %s)",
                             data_grid, tuple(grid_shape))
                grid_shape = data_grid

    norm_path = output_dir / 'norm_stats.json'
    if args.norm_stats and args.norm_stats.exists():
        norm_stats = NormStats.load(args.norm_stats)
        logging.info("Loaded norm stats from %s", args.norm_stats)
    elif norm_path.exists():
        norm_stats = NormStats.load(norm_path)
        logging.info("Loaded existing norm stats from %s", norm_path)
    else:
        norm_stats = compute_norm_stats(train_files, geometries)
        norm_stats.save(norm_path)
        logging.info("Saved norm stats to %s", norm_path)

    train_dataset = FNODataset(train_files, norm_stats, grid_shape, target_grid=target_grid)
    val_dataset   = FNODataset(test_files or train_files[-3:], norm_stats, grid_shape,
                               target_grid=target_grid)

    if args.model == 'cno-fno':
        model = build_cno_fno(
            grid_shape=grid_shape,
            ch=args.channels,
            n_fno_blocks=args.blocks,
            use_attention=args.attention,
            n_heads=args.n_heads,
            device=device,
        )
        latent = model._latent_shape
        logging.info(
            "cno-fno: grid=%s latent=%s modes=%s channels=%d blocks=%d "
            "attention=%s params=%d physics=%s",
            grid_shape, latent, model.modes, args.channels, args.blocks,
            args.attention, model.n_parameters, args.physics,
        )
    elif args.model == 'cond-fno':
        model = build_cond_fno(
            grid_shape=grid_shape,
            modes=tuple(args.modes),
            hidden_ch=args.channels,
            n_blocks=args.blocks,
            device=device,
        )
        logging.info(
            "cond-fno: grid=%s modes=%s channels=%d blocks=%d params=%d physics=%s",
            grid_shape, tuple(args.modes), args.channels, args.blocks,
            model.n_parameters, args.physics,
        )
    elif args.model == 'whno':
        model = build_whno(
            grid_shape=grid_shape,
            modes=tuple(args.modes),
            hidden_ch=args.channels,
            n_blocks=args.blocks,
            device=device,
        )
        logging.info(
            "whno: grid=%s modes=%s channels=%d blocks=%d params=%d physics=%s",
            grid_shape, tuple(args.modes), args.channels, args.blocks,
            model.n_parameters, args.physics,
        )
    else:  # fno
        model = build_fno(
            grid_shape=grid_shape,
            modes=tuple(args.modes),
            hidden_ch=args.channels,
            n_blocks=args.blocks,
            device=device,
        )
        logging.info(
            "fno: grid=%s modes=%s channels=%d blocks=%d params=%d physics=%s",
            grid_shape, tuple(args.modes), args.channels, args.blocks,
            model.n_parameters, args.physics,
        )

    # torch.compile: ~30-40% speedup on T4 via kernel fusion of spectral + CNN ops.
    # fullgraph=False because CNOFNOHybrid._bcast uses dynamic shapes.
    if getattr(args, 'compile', False):
        if not torch.cuda.is_available():
            logging.warning('--compile requested but no CUDA device found; skipping.')
        else:
            model = torch.compile(model, fullgraph=False)
            logging.info('torch.compile applied (fullgraph=False)')

    geometry = next(iter(geometries.values())) if len(geometries) == 1 else None

    trainer = FNOTrainer(
        model=model,
        norm_stats=norm_stats,
        train_data=train_dataset,
        val_data=val_dataset,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        geometry_name=model_name,
        pde_weight=_PHYSICS_WEIGHT if args.physics else 0.0,
        flux_weight=args.flux_weight,
        geometry=geometry,
        patience=getattr(args, 'patience', 0),
    )
    trainer.train()


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    _resolve_defaults(args)

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    logging.info("Model: %s | device: %s | epochs: %d | physics: %s",
                 args.model, device, args.epochs, args.physics)

    if args.all_geometries:
        for name in ALL_GEOMS:
            logging.info("--- %s ---", name)
            run_single([name], args, device, model_name=name)
    else:
        model_name = args.name or (
            '_'.join(args.geometry) if len(args.geometry) > 1 else args.geometry[0]
        )
        run_single(args.geometry, args, device, model_name=model_name)


if __name__ == '__main__':
    main()
