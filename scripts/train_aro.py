"""
Train the Autoregressive Operator (ARO) thermal surrogate.

Multi-fidelity training pipeline:
  1. Pre-train on LF (analytical) + HF (3D-ICE) data with teacher forcing
  2. Fine-tune on HF only with teacher forcing disabled

Example
-------
# All geometries, multi-fidelity (LF pretrain + HF fine-tune):
python scripts/train_aro.py \
    --hf-data  data/3d-ice \
    --lf-data  data/lf \
    --geometries geometry1 geometry2a geometry2b geometry2c geometry3 \
    --pretrain-epochs 200 \
    --finetune-epochs 100 \
    --output checkpoints/aro

# HF only (no LF):
python scripts/train_aro.py \
    --hf-data  data/3d-ice \
    --geometries geometry1 \
    --pretrain-epochs 300 \
    --finetune-epochs 0 \
    --output checkpoints/aro_hf_only
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reproducibility import set_seed
from src.aro.model import build_aro
from src.aro.data_loader import ARODataset
from src.aro.trainer import AROTrainer
from src.pinn.data_loader import NormStats


_log = logging.getLogger(__name__)

ALL_GEOMS = [
    'geometry1',
    'geometry2a', 'geometry2b', 'geometry2c',
    'geometry3',
    'geometry4',
    'geometry5',
    'geometry6',
]


def _find_npz(root: Path, geometries: list, split: str = 'train') -> list:
    """Find NPZ files matching the given geometries and split tag."""
    files = []
    for geom in geometries:
        # Files live in a per-geometry subdirectory (data/3d-ice/geometry1/...).
        # Globbing only the root found nothing, so this script loaded zero files.
        for base in (root / geom, root):
            found = sorted(base.glob(f"{geom}_{split}_*.npz"))
            if not found:
                found = sorted(base.glob(f"{geom}*.npz"))
            if found:
                files.extend(found)
                break
    return files


def _load_geometries(geometry_names: list):
    """Load Geometry objects from the project's geometry factory."""
    from src.core.geometry_builders import build_all_geometries
    # build_all_geometries() returns a LIST; indexing it by name silently yielded
    # nothing, so this script could never load a geometry.
    all_geoms = {g.name: g for g in build_all_geometries()}
    return {name: all_geoms[name] for name in geometry_names if name in all_geoms}


def main():
    parser = argparse.ArgumentParser(description="Train ARO thermal surrogate")
    parser.add_argument('--hf-data',  required=True, type=Path,
                        help="Directory containing real 3D-ICE NPZ files")
    parser.add_argument('--lf-data',  type=Path, default=None,
                        help="Directory containing LF analytical NPZ files (optional)")
    parser.add_argument('--output',   required=True, type=Path,
                        help="Checkpoint output directory")
    parser.add_argument('--geometries', nargs='+', default=ALL_GEOMS,
                        choices=ALL_GEOMS, metavar='GEOM',
                        help="Geometries to include (default: all 8)")
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--finetune-epochs', type=int, default=100)
    parser.add_argument('--lf-only-pretrain', action='store_true',
                        help="Pre-train on LF data only (faster, noisier)")
    parser.add_argument('--hidden',   type=int, default=32,
                        help="ARO hidden channel width (32=0.85M, 64=3.3M params)")
    parser.add_argument('--modes',    type=int, default=16,
                        help="Fourier modes in each spatial direction")
    parser.add_argument('--n-fno-blocks', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr',       type=float, default=3e-4)
    parser.add_argument('--no-amp',   action='store_true', help="Disable mixed precision")
    parser.add_argument('--rno-weight', type=float, default=0.5,
                        help="Loss weight for RNO windowed self-rollout pass (0=disable RNO, pure teacher forcing)")
    parser.add_argument('--rno-window-min', type=int, default=2,
                        help="Self-rollout window length at epoch 1 (curriculum start)")
    parser.add_argument('--rno-window-max', type=int, default=None,
                        help="Self-rollout window length at final epoch (default: n_layers, i.e. full stack)")
    parser.add_argument('--val-split', type=str, default='test',
                        help="Validation NPZ split tag (default: 'test')")
    parser.add_argument('--seed',     type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log.info("Device: %s", device)

    # ---- Geometries ----
    _log.info("Loading geometries: %s", args.geometries)
    geometries = _load_geometries(args.geometries)
    if not geometries:
        _log.error("No geometries loaded — check build_all_geometries() output")
        sys.exit(1)

    # ---- Norm stats ----
    norm_stats = NormStats()

    # ---- HF files ----
    hf_train = _find_npz(args.hf_data, args.geometries, 'train')
    hf_val   = _find_npz(args.hf_data, args.geometries, args.val_split)
    _log.info("HF: %d train, %d val NPZ files", len(hf_train), len(hf_val))

    # ---- LF files ----
    lf_train = []
    if args.lf_data is not None:
        lf_train = sorted(args.lf_data.glob('*.npz'))
        _log.info("LF: %d files from %s", len(lf_train), args.lf_data)

    # ---- Datasets ----
    train_ds = ARODataset(
        hf_files=hf_train,
        geometries=geometries,
        norm_stats=norm_stats,
        lf_files=lf_train or None,
    )
    val_ds = ARODataset(
        hf_files=hf_val,
        geometries=geometries,
        norm_stats=norm_stats,
    )

    # ---- Determine max n_layers across loaded geometries ----
    n_layers_max = max(len(g.layers) for g in geometries.values())
    _log.info("n_layers_max across loaded geometries: %d", n_layers_max)

    # ---- Model ----
    model = build_aro(
        n_layers=n_layers_max,
        hidden=args.hidden,
        modes1=args.modes,
        modes2=args.modes,
        n_fno_blocks=args.n_fno_blocks,
        cond_dim=4,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info("ARO params: {:,}".format(n_params))

    # ---- Trainer ----
    trainer = AROTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=args.output,
        batch_size=args.batch_size,
        lr=args.lr,
        amp=not args.no_amp,
        rno_weight=args.rno_weight,
        rno_window_min=args.rno_window_min,
        rno_window_max=args.rno_window_max,
        device=device,
    )

    trainer.train(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lf_only_pretrain=args.lf_only_pretrain,
    )

    _log.info("Done. Best val MAE = %.4f K. Checkpoints saved to %s",
              trainer._best_val_mae, args.output)


if __name__ == '__main__':
    main()
