"""FNO training loop."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_loader import FNODataset, predict_to_flat
from .model import FNO3d
from .physics import pde_loss_fd, interface_flux_loss, build_k_grid, grid_spacings
from ..pinn.data_loader import NormStats
from ..pinn.trainer import _build_layer_tensors

_log = logging.getLogger(__name__)


def relative_l2_loss(T_pred: torch.Tensor, T_true: torch.Tensor) -> torch.Tensor:
    """Relative L2 loss per sample, averaged over batch."""
    diff_norm = (T_pred - T_true).flatten(1).norm(dim=1)           # (B,)
    true_norm = T_true.flatten(1).norm(dim=1).clamp(1e-8)         # (B,)
    return (diff_norm / true_norm).mean()


def mae_kelvin(
    T_pred: torch.Tensor,
    T_true: torch.Tensor,
    T_min: float,
    T_max: float,
) -> float:
    T_range = T_max - T_min
    return float((T_pred - T_true).abs().mean().item() * T_range)


class FNOTrainer:
    """Trains FNO3d (or CondFNO3d) on a fixed-geometry dataset."""

    def __init__(
        self,
        model: FNO3d,
        norm_stats: NormStats,
        train_data: FNODataset,
        val_data: FNODataset,
        output_dir: Path,
        batch_size: int = 4,
        epochs: int = 500,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        log_interval: int = 10,
        geometry_name: str = 'fno',
        pde_weight: float = 0.0,
        flux_weight: float = 0.0,
        geometry=None,
        patience: int = 0,
        use_amp: Optional[bool] = None,
    ):
        self.model = model
        self.norm_stats = norm_stats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.log_interval = log_interval
        self.geometry_name = geometry_name
        self.pde_weight = pde_weight
        self.flux_weight = flux_weight
        # Early stopping: stop when val_MAE hasn't improved for `patience` checks.
        # patience=0 disables it. Each check is every log_interval epochs.
        self.patience = patience
        self._no_improve_count = 0

        # PI-FNO / interface-flux: pre-compute shared physics tensors once
        self._use_pi = pde_weight > 0.0 and geometry is not None
        self._use_flux = flux_weight > 0.0 and geometry is not None
        if self._use_pi or self._use_flux:
            self._dx, self._dy, self._dz = grid_spacings(geometry)
            layer_k, _ = _build_layer_tensors(geometry, torch.device('cpu'))
            self._layer_k = layer_k
            self._n_layers = len(geometry.layers)
            _log.info(
                "PI-FNO: pde_weight=%.3f flux_weight=%.3f  dx=%.1fµm dy=%.1fµm dz=%.1fµm",
                pde_weight, flux_weight,
                self._dx * 1e6, self._dy * 1e6, self._dz * 1e6,
            )
        if pde_weight > 0.0 and geometry is None:
            _log.warning("pde_weight=%.3f but geometry=None — PI loss disabled", pde_weight)
        if flux_weight > 0.0 and geometry is None:
            _log.warning("flux_weight=%.3f but geometry=None — flux loss disabled", flux_weight)

        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model.to(self.device)

        # DataLoader with pin_memory disabled on Windows (driver reliability)
        pin = self.device.type == 'cuda' and not _is_windows()
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            pin_memory=pin, num_workers=0,
        )
        self.val_loader = DataLoader(
            val_data, batch_size=1, shuffle=False,
            pin_memory=pin, num_workers=0,
        )
        self.val_data = val_data   # kept for item-level eval

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 1e-3
        )
        # cuFFT's half-precision path only supports power-of-two FFT sizes; the spectral
        # conv's rfftn crashes under autocast on any grid that isn't (e.g. geometry4's
        # 100x56x10, geometry6's 56x168x15 -- most of this project's geometries). Pass
        # use_amp=False explicitly for those rather than relying on the CUDA-available
        # default, since the failure only appears at runtime inside the FFT, not at
        # construction time.
        self.use_amp = (self.device.type == 'cuda') if use_amp is None else use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.best_val_mae = float('inf')
        self.history: Dict[str, List] = {
            'epoch': [], 'train_rl2': [], 'val_mae_K': [],
        }

        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.output_dir / 'tb_logs'))
        except ImportError:
            pass

    def train(self) -> Path:
        patience_str = f' | patience={self.patience}' if self.patience else ''
        _log.info(
            "FNO training: %s | device=%s | epochs=%d | train=%d val=%d%s",
            self.geometry_name, self.device, self.epochs,
            len(self.train_loader.dataset), len(self.val_loader.dataset),
            patience_str,
        )
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            train_rl2 = self._train_epoch()
            self.scheduler.step()

            if epoch % self.log_interval == 0:
                val_mae = self._validate()
                elapsed = time.time() - t0
                _log.info(
                    "Epoch %4d/%d  rL2=%.4f  val_MAE=%.3f K  %.0fs",
                    epoch, self.epochs, train_rl2, val_mae, elapsed,
                )
                self.history['epoch'].append(epoch)
                self.history['train_rl2'].append(train_rl2)
                self.history['val_mae_K'].append(val_mae)

                if self._writer:
                    self._writer.add_scalar('train/rL2', train_rl2, epoch)
                    self._writer.add_scalar('val/mae_K', val_mae, epoch)

                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    self._no_improve_count = 0
                    self._save(epoch, val_mae, best=True)
                elif self.patience > 0:
                    self._no_improve_count += 1
                    if self._no_improve_count >= self.patience:
                        _log.info(
                            "Early stopping at epoch %d (no improvement for %d checks = %d epochs)",
                            epoch, self.patience, self.patience * self.log_interval,
                        )
                        break

        self._save(epoch, self.best_val_mae, best=False)
        if self._writer:
            self._writer.close()

        ckpt = self.output_dir / f'{self.geometry_name}_best.pt'
        _log.info("Done. Best val MAE: %.3f K  →  %s", self.best_val_mae, ckpt)
        return ckpt

    def _train_epoch(self) -> float:
        self.model.train()
        total_rl2 = 0.0

        for batch in self.train_loader:
            Q = batch['Q_norm'].to(self.device)
            L = batch['layer_id_norm'].to(self.device)
            T_true = batch['T_norm'].to(self.device)
            htc = batch['htc_norm'].to(self.device)
            tamb = batch['t_amb_norm'].to(self.device)
            tsv = batch['tsv_frac'].to(self.device)
            # Track B: only present in batches from an FNODataset built with
            # `geometries=`, only consumed by a model built with
            # use_geometry_field=True -- both default off, so this is a no-op
            # for every model/dataset combination that predates it.
            dist_kwargs = {}
            if getattr(self.model, 'use_geometry_field', False):
                dist_kwargs['dist_to_block'] = batch['dist_to_block_norm'].to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.use_amp else torch.float32,
            ):
                T_pred = self.model(Q, L, htc, tamb, tsv, **dist_kwargs)
                loss = relative_l2_loss(T_pred, T_true)

            # PI loss: finite-difference PDE residual (always FP32, no AMP)
            #
            # NOTE: previously called with T_pred.detach(), which severed the
            # gradient connection to the model entirely -- the term was added
            # to `loss` but contributed ZERO gradient at backward(), making it
            # a pure logging artifact rather than a training signal (despite
            # the "serves as a regulariser" claim in physics.py's docstring).
            # Fixed: T_pred is passed through WITHOUT detaching, so the PDE
            # residual now actually shapes the learned field.
            if self._use_pi:
                layer_k_dev = self._layer_k.to(self.device)
                pi = pde_loss_fd(
                    T_norm=T_pred.float(),
                    Q_norm=Q.float(),
                    layer_id_norm=L.float(),
                    layer_k=layer_k_dev,
                    n_layers=self._n_layers,
                    T_min=self.norm_stats.T_min,
                    T_max=self.norm_stats.T_max,
                    power_std=self.norm_stats.power_std,
                    dx=self._dx, dy=self._dy, dz=self._dz,
                )
                loss = loss + self.pde_weight * pi

            # Interface-isolated flux-continuity loss (independent of pde_weight)
            if self._use_flux:
                layer_k_dev = self._layer_k.to(self.device)
                flux = interface_flux_loss(
                    T_norm=T_pred.float(),
                    layer_id_norm=L.float(),
                    layer_k=layer_k_dev,
                    n_layers=self._n_layers,
                    T_min=self.norm_stats.T_min,
                    T_max=self.norm_stats.T_max,
                    dz=self._dz,
                )
                loss = loss + self.flux_weight * flux

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # FNO does not suffer gradient spikes like the PINN PDE loss,
                # but clip anyway to guard against occasional bad batches
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_rl2 += loss.item()

        return total_rl2 / len(self.train_loader)

    def _validate(self) -> float:
        self.model.eval()
        T_range = self.norm_stats.T_max - self.norm_stats.T_min
        all_mae = []

        with torch.no_grad():
            for item in self.val_data.items:
                T_pred_K, T_true_K = predict_to_flat(
                    self.model, item, self.device, self.norm_stats
                )
                all_mae.append(float(np.abs(T_pred_K - T_true_K).mean()))

        return float(np.mean(all_mae))

    def _save(self, epoch: int, val_mae: float, best: bool) -> None:
        name = f'{self.geometry_name}_best.pt' if best else f'{self.geometry_name}_final.pt'
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_mae_K': val_mae,
            'grid_shape': self.model.grid_shape,
            'modes': self.model.modes,
            'hidden_ch': self.model.hidden_ch,
            'geometry_name': self.geometry_name,
            'norm_stats': {
                'T_min': self.norm_stats.T_min,
                'T_max': self.norm_stats.T_max,
                'power_mean': self.norm_stats.power_mean,
                'power_std': self.norm_stats.power_std,
                'htc_min': self.norm_stats.htc_min,
                'htc_max': self.norm_stats.htc_max,
                't_amb_min': self.norm_stats.t_amb_min,
                't_amb_max': self.norm_stats.t_amb_max,
                'geom_extents': self.norm_stats.geom_extents,
            },
            'history': self.history,
            'pde_weight': self.pde_weight,
            'model_variant': type(self.model).__name__,
        }, self.output_dir / name)


def _is_windows() -> bool:
    import sys
    return sys.platform == 'win32'
