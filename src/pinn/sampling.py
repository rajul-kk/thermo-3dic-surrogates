"""Adaptive collocation-sampling strategies for PINN training."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..core.geometry import Geometry
from .data_loader import sample_collocation_stratified, power_at_colloc_points
from .physics import pde_residual


# ---------------------------------------------------------------------------
# Curvature (Hessian-trace) weighting
# ---------------------------------------------------------------------------

def hessian_trace_weights(
    model: torch.nn.Module,
    coords: torch.Tensor,          # (N, 3) requires_grad will be set internally
    layer_ids: torch.Tensor,
    power: torch.Tensor,
    htc_norm: torch.Tensor,
    t_amb_norm: torch.Tensor,
    tsv_frac: torch.Tensor,
    region_ids: Optional[torch.Tensor] = None,
    tim_k_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return (N,) curvature weights = trace of the Hessian of T_hat w.r.t. coords, i.e. the Laplacian |d2T/dx2 + d2T/dy2 + d2T/dz2| — a cheap proxy for full
    """
    coords = coords.clone().requires_grad_(True)
    T_hat = model(coords, layer_ids, power, htc_norm, t_amb_norm, tsv_frac,
                  region_ids=region_ids, tim_k_norm=tim_k_norm)

    grad_T = torch.autograd.grad(
        T_hat, coords, grad_outputs=torch.ones_like(T_hat),
        create_graph=True, retain_graph=True,
    )[0]  # (N, 3)

    laplacian = torch.zeros(coords.shape[0], device=coords.device)
    for d in range(3):
        grad2 = torch.autograd.grad(
            grad_T[:, d], coords, grad_outputs=torch.ones_like(grad_T[:, d]),
            create_graph=False, retain_graph=True,
        )[0][:, d]
        laplacian = laplacian + grad2

    return laplacian.detach().abs()


# ---------------------------------------------------------------------------
# Importance / adversarial-proxy resampling
# ---------------------------------------------------------------------------

def importance_resample(
    weights: torch.Tensor,     # (M,) unnormalised importance signal (residual, curvature, etc.)
    n_new: int,
    temperature: float = 1.0,
    uniform_floor: float = 0.1,
) -> torch.Tensor:
    """
    Sample n_new indices from [0, M) via temperature-controlled softmax resampling, with a uniform floor to prevent collapse onto a single mode
    """
    w = weights.float()
    w = w / (w.std() + 1e-12)             # scale-invariant before softmax
    probs = torch.softmax(w / max(temperature, 1e-6), dim=0)
    probs = (1.0 - uniform_floor) * probs + uniform_floor / len(probs)
    probs = probs / probs.sum()
    return torch.multinomial(probs, n_new, replacement=False)


# ---------------------------------------------------------------------------
# Curriculum-enhanced adaptive sampling
# ---------------------------------------------------------------------------

def curriculum_blend_factor(
    epoch: int,
    total_epochs: int,
    warmup_frac: float = 0.2,
) -> float:
    """
    Blend factor in [0, 1]: 0 = pure uniform/stratified coverage, 1 = pure adaptive (residual/curvature) weighting.
    """
    warmup_epochs = warmup_frac * total_epochs
    if epoch <= warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min(1.0, progress)


def curriculum_enhanced_resample(
    adaptive_weights: torch.Tensor,   # (M,) residual or curvature signal
    n_new: int,
    blend: float,                      # from curriculum_blend_factor()
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Blend uniform sampling with adaptive-weighted sampling according to `blend` (0=uniform, 1=fully adaptive), then draw n_new indices.
    """
    M = len(adaptive_weights)
    w = adaptive_weights.float()
    w = w / (w.std() + 1e-12)
    adaptive_probs = torch.softmax(w / max(temperature, 1e-6), dim=0)
    uniform_probs = torch.full((M,), 1.0 / M, device=adaptive_weights.device)

    probs = (1.0 - blend) * uniform_probs + blend * adaptive_probs
    probs = probs / probs.sum()
    return torch.multinomial(probs, n_new, replacement=False)


# ---------------------------------------------------------------------------
# Strategy dispatch — used by Trainer._rar_update
# ---------------------------------------------------------------------------

SAMPLING_STRATEGIES = ('rar', 'hessian', 'importance', 'curriculum')


def compute_adaptive_weights(
    strategy: str,
    model: torch.nn.Module,
    cand_coords: torch.Tensor,
    cand_ids: torch.Tensor,
    cand_power: torch.Tensor,
    htc_t: torch.Tensor,
    tamb_t: torch.Tensor,
    tsv_t: torch.Tensor,
    layer_k: torch.Tensor,
    si_mask: torch.Tensor,
    T_min: float,
    T_max: float,
    geom_scale: Tuple[float, float, float],
    power_scale: float,
    col_k_lateral: Optional[torch.Tensor] = None,
    col_si_lateral: Optional[torch.Tensor] = None,
    region_ids: Optional[torch.Tensor] = None,
    tim_k_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the (M,) importance signal for a candidate pool according to `strategy`. 'rar' (PDE residual magnitude) is the pre-existing default;
    """
    if strategy == 'hessian':
        model.eval()
        weights = hessian_trace_weights(
            model, cand_coords, cand_ids, cand_power, htc_t, tamb_t, tsv_t,
            region_ids=region_ids, tim_k_norm=tim_k_norm,
        )
        model.train()
        return weights

    # 'rar', 'importance', 'curriculum' all use PDE residual magnitude
    # NOTE: pde_residual() calls coords_col.requires_grad_(True) IN-PLACE on
    # whatever tensor it's given. If we pass cand_coords directly, the
    # caller's tensor is permanently mutated to requires_grad=True, which
    # then poisons the persistent collocation buffer it gets concatenated
    # into (self._col_coords becomes a non-leaf tensor with stale autograd
    # history, corrupting the NEXT epoch's backward pass). Pass a clone so
    # the caller's cand_coords is never touched.
    model.eval()
    res = pde_residual(
        model, cand_coords.clone(), cand_ids, cand_power,
        htc_t, tamb_t, tsv_t,
        layer_k, si_mask, T_min, T_max, geom_scale, power_scale,
        col_k_lateral=col_k_lateral, col_si_lateral=col_si_lateral,
        region_ids_col=region_ids, tim_k_norm=tim_k_norm,
    ).detach().abs()
    model.train()
    return res
