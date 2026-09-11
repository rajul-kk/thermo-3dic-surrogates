"""Loss functions for the thermal PINN."""

from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn as nn


@dataclass
class LossWeights:
    """Current loss weights (updated per epoch during adaptive phase)."""
    data: float = 1.0
    pde: float = 0.0
    bc_top: float = 0.0
    bc_sides: float = 0.0
    interface: float = 0.0

    # EMA smoothing factor for NTK-based adaptation
    ema_alpha: float = 0.9

    # Stored for EMA update
    _prev: Dict[str, float] = field(default_factory=lambda: {
        'pde': 0.1, 'bc_top': 0.5, 'bc_sides': 0.1, 'interface': 0.1
    })

    def curriculum_stage(self, epoch: int) -> int:
        if epoch < 1000:
            return 1
        if epoch < 3000:
            return 2
        return 3

    # Target weights for Stage 2 (fully ramped)
    _STAGE2_WEIGHTS = {'pde': 0.1, 'bc_top': 0.5, 'bc_sides': 0.05, 'interface': 0.1}
    # Ramp starts _RAMP_START epochs before Stage 2 boundary to avoid a hard step
    _RAMP_START = 900   # begin pre-activating 100 epochs before Stage 2 at 1000

    def apply_curriculum(self, epoch: int) -> None:
        stage = self.curriculum_stage(epoch)
        if stage == 1:
            # Linear pre-ramp during [_RAMP_START, 1000) so physics losses
            # ease in rather than activating as a hard step at epoch 1000.
            ramp = max(0.0, (epoch - self._RAMP_START) / (1000 - self._RAMP_START))
            for key, target in self._STAGE2_WEIGHTS.items():
                setattr(self, key, target * ramp)
        elif stage == 2:
            for key, target in self._STAGE2_WEIGHTS.items():
                setattr(self, key, target)

    def update_ntk(
        self,
        model: nn.Module,
        loss_dict: Dict[str, torch.Tensor],
    ) -> None:
        """NTK-based adaptive weight update (Wang et al. 2022)."""
        grad_norms: Dict[str, float] = {}
        for name, loss in loss_dict.items():
            if loss.requires_grad:
                grads = torch.autograd.grad(
                    loss, model.parameters(),
                    retain_graph=True, allow_unused=True
                )
                norm = sum(
                    g.norm().item() ** 2
                    for g in grads if g is not None
                ) ** 0.5
                grad_norms[name] = max(norm, 1e-8)

        if not grad_norms:
            return

        max_norm = max(grad_norms.values())
        new_weights = {k: max_norm / v for k, v in grad_norms.items()}

        a = self.ema_alpha
        for key, new_val in new_weights.items():
            old_val = self._prev.get(key, new_val)
            smoothed = a * old_val + (1 - a) * new_val
            self._prev[key] = smoothed
            if key == 'pde':
                self.pde = smoothed
            elif key == 'bc_top':
                self.bc_top = smoothed
            elif key == 'bc_sides':
                self.bc_sides = smoothed
            elif key == 'interface':
                self.interface = smoothed


def data_loss(T_pred: torch.Tensor, T_true: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and simulator temperatures (normalised)."""
    return torch.mean((T_pred - T_true) ** 2)


def pde_loss(residual: torch.Tensor) -> torch.Tensor:
    """Mean squared PDE residual."""
    return torch.mean(residual ** 2)


def bc_loss(residual: torch.Tensor) -> torch.Tensor:
    """Mean squared BC residual."""
    return torch.mean(residual ** 2)


def interface_loss(
    T_above: torch.Tensor,
    T_below: torch.Tensor,
) -> torch.Tensor:
    """Temperature continuity at layer interfaces: MSE(T_above - T_below)."""
    return torch.mean((T_above - T_below) ** 2)


def total_loss(
    losses: Dict[str, torch.Tensor],
    weights: LossWeights,
) -> torch.Tensor:
    """Weighted sum of all loss components."""
    L = weights.data * losses['data']
    if weights.pde > 0 and 'pde' in losses:
        L = L + weights.pde * losses['pde']
    if weights.bc_top > 0 and 'bc_top' in losses:
        L = L + weights.bc_top * losses['bc_top']
    if weights.bc_sides > 0 and 'bc_sides' in losses:
        L = L + weights.bc_sides * losses['bc_sides']
    if weights.interface > 0 and 'interface' in losses:
        L = L + weights.interface * losses['interface']
    return L
