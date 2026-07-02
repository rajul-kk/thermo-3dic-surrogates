"""
Therm-FM: few-shot fine-tuning of a pre-trained CNOFNOHybrid for a new geometry.

Strategy
--------
Given a CNOFNOHybrid pre-trained on 6-7 geometries (encoder has learned
structural features: conduction paths, TSV influence, boundary layers), we
freeze ~90% of parameters and only update:
  - FiLM generator (maps scenario BCs to per-block gamma/beta modulations)
  - Last 2 FNO blocks of the latent trunk (fine spatial adjustment)
  - Decoder (reconstructs T from the FNO latent)
  - Final projection layer (if any)

This is ~10% of total parameters, allowing 5-20 shot fine-tuning in
<1 minute on a T4 without catastrophic forgetting of the source geometries.

Usage
-----
# 10-shot fine-tune for a new geometry:
python scripts/finetune_therm_fm.py \
    --pretrained  checkpoints/cno_fno/cno_fno_best.pt \
    --new-data    data/new_geometry \
    --geometry    geometry_new \
    --output      checkpoints/therm_fm \
    --shots       10 \
    --epochs      100

# All 5 shots (fast iteration):
python scripts/finetune_therm_fm.py \
    --pretrained checkpoints/cno_fno/cno_fno_best.pt \
    --new-data   data/new_geometry \
    --geometry   geometry_new \
    --shots      5  --epochs 50 --output checkpoints/therm_fm
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fno.model import CNOFNOHybrid, build_cno_fno
from src.fno.data_loader import FNODataset
from src.pinn.data_loader import NormStats

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter freeze / unfreeze helpers
# ---------------------------------------------------------------------------

# CNOFNOHybrid's actual top-level parameter groups (verified via named_parameters()):
#   lift, enc_res, enc_down   -- CNN encoder (downsampling path)
#   latent_blocks             -- FiLM-FNO blocks operating at the pooled latent resolution
#   film_gen                  -- FiLM generator (BCs -> per-block gamma/beta)
#   dec_res, dec_fuse         -- CNN decoder (upsampling + skip-connection fusion path)
#   proj                      -- final projection head
_ENCODER_PREFIXES = ('lift', 'enc_res', 'enc_down')
_DECODER_PREFIXES = ('dec_res', 'dec_fuse')
_TAIL_BLOCK_ATTR  = 'latent_blocks'   # NOT 'fno_blocks' -- that attribute doesn't exist


def freeze_encoder(model: CNOFNOHybrid) -> int:
    """Freeze encoder weights (lift + enc_res + enc_down). Returns number of frozen parameters."""
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(_ENCODER_PREFIXES):
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen


def unfreeze_film_and_tail(model: CNOFNOHybrid, n_tail_blocks: int = 2) -> int:
    """Unfreeze FiLM generator + last n_tail_blocks of latent FNO blocks + decoder + proj."""
    unfrozen = 0
    n_latent = len(getattr(model, _TAIL_BLOCK_ATTR))
    tail_start = max(0, n_latent - n_tail_blocks)

    for name, param in model.named_parameters():
        is_film    = name.startswith('film_gen')
        is_decoder = name.startswith(_DECODER_PREFIXES)
        is_proj    = name.startswith('proj')

        is_tail_latent = False
        for i in range(tail_start, n_latent):
            if name.startswith(f'{_TAIL_BLOCK_ATTR}.{i}'):
                is_tail_latent = True
                break

        if is_film or is_decoder or is_proj or is_tail_latent:
            param.requires_grad_(True)
            unfrozen += param.numel()
        else:
            param.requires_grad_(False)

    return unfrozen


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    model:        CNOFNOHybrid,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    output_dir:   Path,
    epochs:       int,
    lr:           float,
    grad_clip:    float,
    amp:          bool,
    device:       torch.device,
    log_interval: int = 10,
    val_interval: int = 20,
) -> float:
    """Run the fine-tune loop. Returns best val MAE (K)."""
    opt   = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler() if (amp and device.type == 'cuda') else None
    criterion = nn.MSELoss()

    best_val_mae = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            if batch is None:
                continue
            # FNODataset batches are dicts with Q_grid, T_grid, cond
            Q     = batch['Q_grid'].to(device)
            T_gt  = batch['T_grid'].to(device)
            htc   = batch['htc_norm'].to(device)
            tamb  = batch['t_amb_norm'].to(device)
            tsv   = batch['tsv_frac'].to(device)

            opt.zero_grad()
            if scaler is not None:
                with autocast():
                    T_pred = model(Q, htc, tamb, tsv)
                    loss   = criterion(T_pred, T_gt)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                scaler.step(opt)
                scaler.update()
            else:
                T_pred = model(Q, htc, tamb, tsv)
                loss   = criterion(T_pred, T_gt)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()

        if epoch % log_interval == 0 or epoch == epochs:
            _log.info("Epoch %d/%d  train_mse=%.5f", epoch, epochs, epoch_loss / max(n_batches, 1))

        if epoch % val_interval == 0 or epoch == epochs:
            val_mae = _evaluate(model, val_loader, device)
            _log.info("Epoch %d  val_MAE=%.4f K", epoch, val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                ckpt = output_dir / 'therm_fm_best.pt'
                torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt)
                _log.info("  -> saved best (%.4f K): %s", val_mae, ckpt.name)

    return best_val_mae


def _evaluate(model, loader, device) -> float:
    model.eval()
    total_mae = 0.0
    n_pts     = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            Q    = batch['Q_grid'].to(device)
            T_gt = batch['T_grid'].to(device)
            htc  = batch['htc_norm'].to(device)
            tamb = batch['t_amb_norm'].to(device)
            tsv  = batch['tsv_frac'].to(device)
            T_pred = model(Q, htc, tamb, tsv)
            total_mae += (T_pred - T_gt).abs().sum().item()
            n_pts     += T_gt.numel()
    return total_mae / max(n_pts, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Therm-FM: few-shot fine-tuning of CNO-FNO")
    parser.add_argument('--pretrained', required=True, type=Path,
                        help="Path to pre-trained CNOFNOHybrid checkpoint (.pt)")
    parser.add_argument('--new-data',  required=True, type=Path,
                        help="Directory with new-geometry NPZ files")
    parser.add_argument('--geometry',  required=True,
                        help="Geometry name (must match NPZ metadata)")
    parser.add_argument('--output',    required=True, type=Path)
    parser.add_argument('--shots',     type=int, default=10,
                        help="Number of training NPZ files to use (few-shot count)")
    parser.add_argument('--epochs',    type=int, default=100)
    parser.add_argument('--lr',        type=float, default=1e-4,
                        help="Fine-tune learning rate (lower than pre-train)")
    parser.add_argument('--tail-blocks', type=int, default=2,
                        help="Number of tail FNO blocks to unfreeze (default 2)")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--no-amp',    action='store_true')
    parser.add_argument('--val-split', type=str, default='test')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log.info("Device: %s", device)

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- Load geometry ----
    from src.core.geometry import build_all_geometries
    all_geoms = build_all_geometries()
    if args.geometry not in all_geoms:
        _log.error("Geometry '%s' not found. Available: %s", args.geometry, list(all_geoms))
        sys.exit(1)
    geometry = all_geoms[args.geometry]

    # ---- Load pre-trained model ----
    _log.info("Loading pre-trained checkpoint: %s", args.pretrained)
    ckpt = torch.load(args.pretrained, map_location='cpu')
    # Checkpoint key convention differs by producer: FNOTrainer._save() uses
    # 'model_state' (src/fno/trainer.py), this script's own finetune() saves
    # use 'model'. Accept either so a checkpoint from CNO-FNO pretraining
    # (FNOTrainer) loads correctly here, not just checkpoints re-saved by
    # this script itself.
    state_dict = ckpt.get('model') or ckpt.get('model_state') or ckpt

    # Reconstruct model — try to infer channel width from checkpoint key shapes
    ch = 32
    for key, val in state_dict.items():
        if 'encoder' in key and val.dim() >= 1 and val.shape[0] > ch:
            ch = val.shape[0]
            break
    _log.info("Inferred CNO-FNO channel width from checkpoint: ch=%d", ch)

    nx, ny, nz = geometry.mesh_resolution
    model = build_cno_fno(nx=nx, ny=ny, nz=nz, ch=ch, device=device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _log.warning("Missing keys when loading checkpoint: %s", missing[:5])
    if unexpected:
        _log.warning("Unexpected keys: %s", unexpected[:5])

    # ---- Partial freeze ----
    frozen   = freeze_encoder(model)
    unfrozen = unfreeze_film_and_tail(model, n_tail_blocks=args.tail_blocks)
    total    = sum(p.numel() for p in model.parameters())
    _log.info("Freeze: %d params frozen, %d unfrozen (%.1f%% of %d total)",
              frozen, unfrozen, 100.0 * unfrozen / total, total)

    # ---- Datasets ----
    norm_stats = NormStats()

    train_files = sorted(args.new_data.glob(f"{args.geometry}_train_*.npz"))
    val_files   = sorted(args.new_data.glob(f"{args.geometry}_{args.val_split}_*.npz"))

    if not train_files:
        train_files = sorted(args.new_data.glob(f"{args.geometry}*.npz"))
        # Reserve last file for val if no explicit val split
        if len(train_files) > 1:
            val_files   = train_files[-1:]
            train_files = train_files[:-1]

    if not train_files:
        _log.error("No training NPZ files found in %s for geometry '%s'",
                   args.new_data, args.geometry)
        sys.exit(1)

    # Few-shot: use only `shots` training files
    train_files = train_files[:args.shots]
    _log.info("Few-shot: %d train files, %d val files", len(train_files), len(val_files))

    train_ds = FNODataset(
        npz_files=train_files,
        geometry=geometry,
        norm_stats=norm_stats,
    )
    val_ds = FNODataset(
        npz_files=val_files if val_files else train_files[-1:],
        geometry=geometry,
        norm_stats=norm_stats,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # ---- Fine-tune ----
    best_mae = finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=1.0,
        amp=not args.no_amp,
        device=device,
    )

    _log.info("Fine-tuning complete. Best val MAE = %.4f K", best_mae)
    torch.save({'model': model.state_dict()},
               args.output / f'therm_fm_{args.geometry}_final.pt')
    _log.info("Final model saved to %s", args.output / f'therm_fm_{args.geometry}_final.pt')


if __name__ == '__main__':
    main()
