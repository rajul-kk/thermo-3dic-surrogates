"""PI-DeepONet training loop."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import PIDeepONet
from .cno_model import PICNODeepONet
from .data_loader import MultiGeomDataset

_AnyDeepONet = (PIDeepONet, PICNODeepONet)
from ..pinn.data_loader import NormStats
from ..pinn.trainer import _build_layer_tensors

_log = logging.getLogger(__name__)


def _pde_residual_deeponet(
    model: 'PIDeepONet | PICNODeepONet',
    branch_coeffs: torch.Tensor,    # (1, n_basis) — pre-computed, detached
    col_coords: torch.Tensor,       # (N_col, 4) with requires_grad=True
    layer_k: torch.Tensor,          # (n_layers,)
    si_mask: torch.Tensor,          # (n_layers,)
    power_col: torch.Tensor,        # (N_col,) normalised
    T_min: float,
    T_max: float,
    geom_scale: tuple,              # (Lx, Ly, Lz) in µm
    power_std: float,
) -> torch.Tensor:
    """
    Compute PDE residual ∇·(k∇T) + Q = 0 at collocation points via autograd through the trunk network only.
    """
    b = branch_coeffs                              # (1, n_basis)
    t = model.trunk(col_coords)                    # (N_col, n_basis)
    T_norm = (b @ t.T).squeeze(0) + model.bias     # (N_col,)

    T_range = T_max - T_min
    T_phys  = T_norm * T_range + T_min             # [K]

    # Gradients of T w.r.t. normalised coordinates
    Lx, Ly, Lz = [s * 1e-6 for s in geom_scale]   # µm → m

    grad = torch.autograd.grad(
        T_phys, col_coords,
        grad_outputs=torch.ones_like(T_phys),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N_col, 4)

    # Physical gradients: dT/dx_phys = (dT/dx_norm) / Lx
    dTdx = grad[:, 0] / Lx
    dTdy = grad[:, 1] / Ly
    dTdz = grad[:, 2] / Lz

    # Layer conductivity at collocation points
    layer_idx = (col_coords[:, 3] * max(len(layer_k) - 1, 1)).round().long()
    layer_idx  = layer_idx.clamp(0, len(layer_k) - 1)
    k_val      = layer_k[layer_idx]
    is_si      = si_mask[layer_idx]

    # k(T) for Si layers (detached — quasi-linearisation as in PINN)
    k_base = k_val.clone().float()
    if is_si.any():
        T_K = T_phys.detach()[is_si].clamp(200, 1200)
        k_base[is_si] = 148.0 * (300.0 / T_K) ** 1.3

    kTx = k_base * dTdx
    kTy = k_base * dTdy
    kTz = k_base * dTdz

    # Divergence: retain_graph for first two passes; release on last
    div_x = torch.autograd.grad(kTx, col_coords,
                                grad_outputs=torch.ones_like(kTx),
                                create_graph=False,
                                retain_graph=True)[0][:, 0] / Lx
    div_y = torch.autograd.grad(kTy, col_coords,
                                grad_outputs=torch.ones_like(kTy),
                                create_graph=False,
                                retain_graph=True)[0][:, 1] / Ly
    div_z = torch.autograd.grad(kTz, col_coords,
                                grad_outputs=torch.ones_like(kTz),
                                create_graph=False,
                                retain_graph=False)[0][:, 2] / Lz

    div_total = div_x + div_y + div_z

    # RHS: Q in physical units
    Q_phys = power_col * power_std              # [W/m³]

    residual = (div_total + Q_phys) / (power_std + 1e-8)
    return (residual ** 2).mean()


class DeepONetTrainer:
    """Trains PIDeepONet across multiple geometries simultaneously."""

    def __init__(
        self,
        model: 'PIDeepONet | PICNODeepONet',
        geometries: Dict,
        norm_stats: NormStats,
        train_data: MultiGeomDataset,
        val_data: MultiGeomDataset,
        output_dir: Path,
        batch_size: int = 4,
        epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pde_weight: float = 0.0,
        n_data: int = 4096,
        n_col: int = 2048,
        device: Optional[torch.device] = None,
        log_interval: int = 20,
        model_name: str = 'deeponet',
    ):
        self.model      = model
        self.norm_stats = norm_stats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs       = epochs
        self.pde_weight   = pde_weight
        self.n_data       = n_data
        self.n_col        = n_col
        self.log_interval = log_interval
        self.model_name   = model_name

        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model.to(self.device)

        # Pre-compute per-geometry layer tensors (layer_k, si_mask)
        self._layer_tensors: Dict[str, tuple] = {}
        for name, geom in geometries.items():
            lk, sm = _build_layer_tensors(geom, self.device)
            scale = (geom.die_width, geom.die_length, geom.get_total_height())
            self._layer_tensors[name] = (lk, sm, scale)

        pin = self.device.type == 'cuda' and not _is_windows()
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            pin_memory=False, num_workers=0,    # items have variable-N tensors
            collate_fn=_identity_collate,
        )
        self.val_data = val_data

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Reduce LR on plateau — DeepONet convergence is less smooth than FNO
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5, min_lr=lr * 1e-3
        )
        self.use_amp = self.device.type == 'cuda'
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.best_val_mae = float('inf')
        self.history: Dict[str, List] = {
            'epoch': [], 'train_loss': [], 'val_mae_K': []
        }

        _log.info(
            "PIDeepONet: %d params | pde_weight=%.3f | n_data=%d n_col=%d",
            model.n_parameters, pde_weight, n_data, n_col,
        )

    def train(self) -> Path:
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)

            if epoch % self.log_interval == 0:
                val_mae = self._validate()
                self.scheduler.step(val_mae)
                elapsed = time.time() - t0
                lr_now = self.optimizer.param_groups[0]['lr']
                _log.info(
                    "Epoch %4d/%d  loss=%.4f  val_MAE=%.2f K  lr=%.2e  %.0fs",
                    epoch, self.epochs, train_loss, val_mae, lr_now, elapsed,
                )
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['val_mae_K'].append(val_mae)

                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    self._save(epoch, val_mae, best=True)

        self._save(self.epochs, self.best_val_mae, best=False)
        ckpt = self.output_dir / f'{self.model_name}_best.pt'
        _log.info("Done. Best val MAE: %.3f K  ->  %s", self.best_val_mae, ckpt)
        return ckpt

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            step_loss = torch.zeros(1, device=self.device)

            for item in batch:
                geom_name  = item['geom_name']
                coords_all = item['coords_norm'].to(self.device)   # (N, 4)
                T_true_all = item['T_norm'].to(self.device)        # (N,)
                Q_all      = item['power_norm'].to(self.device)    # (N,)
                N = coords_all.shape[0]

                # --- Branch encoding (model-agnostic via encode_branch_from_item) ---
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16 if self.use_amp else torch.float32,
                ):
                    b = self.model.encode_branch_from_item(item, self.device)  # (1, n_basis)

                # --- Data loss: random subset of grid points ---
                n_d = min(self.n_data, N)
                idx = torch.randperm(N, device=self.device)[:n_d]
                coords_d = coords_all[idx]
                T_true_d = T_true_all[idx]

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16 if self.use_amp else torch.float32,
                ):
                    t = self.model.trunk(coords_d)                         # (n_d, n_basis)
                    T_pred = (b @ t.T).squeeze(0) + self.model.bias        # (n_d,)
                    data_loss = F.mse_loss(T_pred, T_true_d)

                step_loss = step_loss + data_loss

                # --- PDE loss: separate collocation subset (trunk autograd only) ---
                if self.pde_weight > 0 and geom_name in self._layer_tensors:
                    layer_k, si_mask, geom_scale = self._layer_tensors[geom_name]
                    n_c = min(self.n_col, N)
                    col_idx = torch.randperm(N, device=self.device)[:n_c]

                    col_coords = coords_all[col_idx].detach().clone().requires_grad_(True)
                    power_col  = Q_all[col_idx]

                    pi = _pde_residual_deeponet(
                        model          = self.model,
                        branch_coeffs  = b.detach().float(),
                        col_coords     = col_coords,
                        layer_k        = layer_k,
                        si_mask        = si_mask,
                        power_col      = power_col,
                        T_min          = self.norm_stats.T_min,
                        T_max          = self.norm_stats.T_max,
                        geom_scale     = geom_scale,
                        power_std      = self.norm_stats.power_std,
                    )
                    step_loss = step_loss + self.pde_weight * pi

            # Backward
            if self.use_amp:
                self.scaler.scale(step_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                step_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += step_loss.item()
            n_steps += 1

        return total_loss / max(n_steps, 1)

    def _validate(self) -> float:
        self.model.eval()
        T_range = self.norm_stats.T_max - self.norm_stats.T_min
        all_mae: List[float] = []

        with torch.no_grad():
            for item in self.val_data.items:
                coords_all = item['coords_norm'].to(self.device)
                T_true     = item['T_norm'].cpu().numpy()

                b = self.model.encode_branch_from_item(item, self.device)
                # Evaluate trunk in chunks to avoid OOM on large grids
                chunks, step = [], 8192
                for start in range(0, len(coords_all), step):
                    t = self.model.trunk(coords_all[start:start+step])
                    chunks.append((b @ t.T).squeeze(0))
                T_pred_norm = torch.cat(chunks).cpu().numpy()

                T_pred_K = T_pred_norm * T_range + self.norm_stats.T_min
                T_true_K = T_true     * T_range + self.norm_stats.T_min
                all_mae.append(float(np.abs(T_pred_K - T_true_K).mean()))

        return float(np.mean(all_mae))

    def _save(self, epoch: int, val_mae: float, best: bool) -> None:
        name = f'{self.model_name}_best.pt' if best else f'{self.model_name}_final.pt'
        torch.save({
            'epoch':        epoch,
            'model_state':  self.model.state_dict(),
            'val_mae_K':    val_mae,
            'n_basis':      self.model.n_basis,
            'pde_weight':   self.pde_weight,
            'history':      self.history,
            'norm_stats': {
                'T_min':      self.norm_stats.T_min,
                'T_max':      self.norm_stats.T_max,
                'power_mean': self.norm_stats.power_mean,
                'power_std':  self.norm_stats.power_std,
                'htc_min':    self.norm_stats.htc_min,
                'htc_max':    self.norm_stats.htc_max,
                't_amb_min':  self.norm_stats.t_amb_min,
                't_amb_max':  self.norm_stats.t_amb_max,
            },
        }, self.output_dir / name)


def _identity_collate(batch):
    """Return list of items — avoids default_collate which requires same-shape tensors."""
    return batch


def _is_windows() -> bool:
    import sys
    return sys.platform == 'win32'
