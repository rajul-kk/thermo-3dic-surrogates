"""ARO Trainer — multi-fidelity RNO-style training loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

from .model import ARO
from .data_loader import ARODataset

_log = logging.getLogger(__name__)


def _collate_filter(batch: list, hf_only: bool) -> Optional[dict]:
    """Collate batch, optionally keeping only HF items."""
    if hf_only:
        batch = [it for it in batch if not it['is_lf']]
    if not batch:
        return None

    keys = ['Q_stack', 'T_stack', 'k_norms', 'cond']
    out = {k: torch.stack([it[k] for it in batch], dim=0) for k in keys}
    out['is_lf'] = torch.tensor([it['is_lf'] for it in batch], dtype=torch.bool)
    return out


class AROTrainer:
    """Trainer for the Autoregressive Operator."""

    def __init__(
        self,
        model:                ARO,
        train_dataset:        ARODataset,
        val_dataset:          ARODataset,
        output_dir:           Path,
        batch_size:           int   = 8,
        lr:                   float = 3e-4,
        lr_finetune_factor:   float = 0.1,
        val_interval:         int   = 20,
        log_interval:         int   = 10,
        grad_clip:            float = 1.0,
        amp:                  bool  = True,
        rno_weight:           float = 0.5,   # loss weight for the windowed self-rollout pass
        rno_window_min:       int   = 2,     # window length at epoch 1
        rno_window_max:       int   = None,  # window length at final epoch (None -> n_layers)
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_ds = train_dataset
        self.val_ds   = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size         = batch_size
        self.lr                 = lr
        self.lr_finetune_factor = lr_finetune_factor
        self.val_interval       = val_interval
        self.log_interval       = log_interval
        self.grad_clip          = grad_clip
        self.amp                = amp
        self.rno_weight         = rno_weight
        self.rno_window_min     = rno_window_min
        self.rno_window_max     = rno_window_max or model.n_layers

        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model = self.model.to(self.device)

        self._best_val_mae = float('inf')
        self._scaler = GradScaler() if (amp and self.device.type == 'cuda') else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        pretrain_epochs:  int   = 200,
        finetune_epochs:  int   = 100,
        lf_only_pretrain: bool  = False,
        tf_decay:         float = 0.995,   # teacher-forcing ratio decay per epoch
    ) -> None:
        """Run pretrain → fine-tune pipeline."""
        _log.info("ARO pretrain: %d epochs (lf_only=%s)", pretrain_epochs, lf_only_pretrain)
        self._stage_train(
            n_epochs=pretrain_epochs,
            hf_only=False,
            lf_only=lf_only_pretrain,
            tf_ratio_start=1.0,
            tf_decay=tf_decay,
            stage='pretrain',
        )

        if finetune_epochs > 0:
            _log.info("ARO fine-tune: %d epochs on HF only, tf_ratio=0, rno_window=full", finetune_epochs)
            self._adjust_lr(self.lr * self.lr_finetune_factor)
            # Fine-tune rolls out at FULL window length throughout — matches
            # inference exactly (no curriculum ramp; pretrain already did that).
            saved_min = self.rno_window_min
            self.rno_window_min = self.rno_window_max
            self._stage_train(
                n_epochs=finetune_epochs,
                hf_only=True,
                lf_only=False,
                tf_ratio_start=0.0,
                tf_decay=1.0,
                stage='finetune',
            )
            self.rno_window_min = saved_min

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _stage_train(
        self,
        n_epochs:        int,
        hf_only:         bool,
        lf_only:         bool,
        tf_ratio_start:  float,
        tf_decay:        float,
        stage:           str,
    ) -> None:
        """Each step optimises a combined loss:"""
        opt = Adam(self.model.parameters(), lr=self.lr)
        sched = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=self.lr * 0.01)

        # Filter dataset if needed
        if lf_only:
            indices = [i for i, it in enumerate(self.train_ds.items) if it['is_lf']]
        elif hf_only:
            indices = [i for i, it in enumerate(self.train_ds.items) if not it['is_lf']]
        else:
            indices = list(range(len(self.train_ds)))

        from torch.utils.data import Subset
        subset = Subset(self.train_ds, indices)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: _collate_filter(b, hf_only=False),
            drop_last=len(subset) >= self.batch_size,
        )

        tf_ratio = tf_ratio_start
        criterion = nn.MSELoss()

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0

            # Curriculum: window grows linearly across the stage.
            progress = (epoch - 1) / max(n_epochs - 1, 1)
            window = int(round(
                self.rno_window_min + progress * (self.rno_window_max - self.rno_window_min)
            ))
            window = max(2, min(window, self.model.n_layers))

            for batch in loader:
                if batch is None:
                    continue
                Q  = batch['Q_stack'].to(self.device)   # (B, n_layers, H, W)
                T  = batch['T_stack'].to(self.device)
                k  = batch['k_norms'].to(self.device)
                c  = batch['cond'].to(self.device)

                opt.zero_grad()
                if self._scaler is not None:
                    with autocast():
                        T_pred = self.model(Q, k, c, T_gt_stack=T, tf_ratio=tf_ratio)
                        loss_tf = criterion(T_pred, T)

                        T_win, start, win_len = self.model.forward_windowed_rollout(
                            Q, k, c, T_gt_stack=T, window=window,
                        )
                        loss_rno = criterion(T_win, T[:, start:start + win_len, :, :])

                        loss = loss_tf + self.rno_weight * loss_rno
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self._scaler.step(opt)
                    self._scaler.update()
                else:
                    T_pred = self.model(Q, k, c, T_gt_stack=T, tf_ratio=tf_ratio)
                    loss_tf = criterion(T_pred, T)

                    T_win, start, win_len = self.model.forward_windowed_rollout(
                        Q, k, c, T_gt_stack=T, window=window,
                    )
                    loss_rno = criterion(T_win, T[:, start:start + win_len, :, :])

                    loss = loss_tf + self.rno_weight * loss_rno
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    opt.step()

                epoch_loss += loss.item()
                n_batches  += 1

            tf_ratio = max(0.0, tf_ratio * tf_decay)
            sched.step()

            if epoch % self.log_interval == 0 or epoch == n_epochs:
                avg_loss = epoch_loss / max(n_batches, 1)
                _log.info("[%s] Epoch %d/%d  train_mse=%.5f  tf=%.3f  rno_window=%d",
                          stage, epoch, n_epochs, avg_loss, tf_ratio, window)

            if epoch % self.val_interval == 0 or epoch == n_epochs:
                val_mae = self._validate()
                _log.info("[%s] Epoch %d  val_MAE=%.4f K", stage, epoch, val_mae)
                if val_mae < self._best_val_mae:
                    self._best_val_mae = val_mae
                    ckpt = self.output_dir / 'aro_best.pt'
                    torch.save({'epoch': epoch, 'stage': stage,
                                'model': self.model.state_dict()}, ckpt)
                    _log.info("  -> saved best checkpoint: %s (MAE=%.4f K)", ckpt.name, val_mae)

        torch.save({'stage': stage, 'model': self.model.state_dict()},
                   self.output_dir / f'aro_{stage}_final.pt')

    def _validate(self) -> float:
        self.model.eval()
        loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=lambda b: _collate_filter(b, hf_only=True))
        total_mae = 0.0
        n_pts     = 0

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                Q = batch['Q_stack'].to(self.device)
                T = batch['T_stack'].to(self.device)
                k = batch['k_norms'].to(self.device)
                c = batch['cond'].to(self.device)
                T_pred = self.model(Q, k, c, T_gt_stack=None, tf_ratio=0.0)
                total_mae += (T_pred - T).abs().sum().item()
                n_pts     += T.numel()

        return total_mae / max(n_pts, 1)

    def _adjust_lr(self, new_lr: float) -> None:
        self.lr = new_lr
        _log.info("ARO: adjusting LR to %.2e for fine-tune stage", new_lr)
