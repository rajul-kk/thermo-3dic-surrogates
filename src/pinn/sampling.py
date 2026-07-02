"""
Adaptive collocation-sampling strategies for PINN training.

Extends the plain residual-proportional RAR already in trainer._rar_update
with three additional strategies drawn from the 2025 adaptive-sampling
literature. All operate on the SAME interface — given a candidate point
pool, return per-point importance weights (or directly resampled points) —
so they are interchangeable via `Trainer(sampling_strategy=...)`.

Strategies
----------
hessian_weighted
    Sample proportional to |Laplacian(T_hat)| (trace of the output Hessian),
    a curvature proxy. Distinct from RAR's PDE-residual weighting: residual
    measures physics violation (∇·(k∇T)+Q != 0), while curvature measures
    where the network's own output is changing sharply — the model can have
    near-zero residual (physics satisfied) at a point with immense curvature
    if it also fits Q correctly there. Curvature-based sampling targets
    representation difficulty, not physics violation, so it is a genuinely
    different signal.  Motivated by "Provably Accurate Adaptive Sampling for
    Collocation Points in PINNs" (ECML PKDD 2025), which used a Hessian-based
    quadrature bound; this is a practical (trace-only, not full Hessian)
    approximation of that idea — full per-point Hessian is O(N) double-
    backward calls and too slow for the >20k point pools used here.

importance_adversarial
    Residual-magnitude softmax resampling. This is a SIMPLIFIED PROXY for
    "Adversarial Adaptive Sampling" (Tang et al. 2024), which trains a deep
    generative model + optimal-transport (Wasserstein) map to *synthesize*
    new sample locations. Implementing the full GAN+OT machinery is out of
    scope here; this keeps the adversarial paper's core intuition — sample
    from a distribution shaped by the residual field rather than a hard
    top-k cutoff — via temperature-controlled softmax resampling, which is
    a defensible but honestly-labelled simplification, not a re-implementation
    of Tang et al.'s method.

curriculum_enhanced
    Blends uniform/stratified coverage (early epochs) with adaptive
    (residual- or curvature-weighted) sampling (late epochs) via a linearly
    growing blend factor. Directly follows "Curriculum-Enhanced Adaptive
    Sampling for PINNs: A Robust Framework for Stiff PDEs" (MDPI, Dec 2025),
    which argues plain RAR over-exploits high-residual regions too early,
    before the network has learned enough of the bulk solution to produce
    meaningful residual signal there. The stiffness they target (steep
    gradients) maps directly onto this repo's material-interface jumps
    (Si k=148 vs TIM k=4, a 37x ratio).
"""

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
    Return (N,) curvature weights = trace of the Hessian of T_hat w.r.t. coords,
    i.e. the Laplacian |d2T/dx2 + d2T/dy2 + d2T/dz2| — a cheap proxy for full
    per-point Hessian eigenvalue analysis (O(N) single Laplacian evaluation
    instead of O(N*3) for the full 3x3 Hessian).

    Uses double-backward (create_graph=True on first derivative). Caller
    should call model.eval() beforehand for consistent dropout state, matching
    the convention in trainer._rar_update.
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
    Sample n_new indices from [0, M) via temperature-controlled softmax
    resampling, with a uniform floor to prevent collapse onto a single mode
    (the failure mode plain RAR is documented to have — "always picking the
    largest residual locations, reducing exploration of other regions").

    temperature < 1 sharpens toward the highest-weight points (more like
    hard top-k RAR); temperature > 1 flattens toward uniform (more
    exploration). temperature=1 is the direct softmax of the raw signal.

    Returns: (n_new,) LongTensor of indices into the weights array.
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
    Blend factor in [0, 1]: 0 = pure uniform/stratified coverage,
    1 = pure adaptive (residual/curvature) weighting.

    Stays at 0 for the first `warmup_frac` of training (let the network learn
    the bulk solution before trusting its residual/curvature signal — this is
    the core argument in Curriculum-Enhanced Adaptive Sampling: early-training
    residuals are dominated by random init noise, not real physics difficulty),
    then ramps linearly to 1 by the end of training.
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
    Blend uniform sampling with adaptive-weighted sampling according to
    `blend` (0=uniform, 1=fully adaptive), then draw n_new indices.

    Implemented as a blended probability distribution (not a coin-flip
    between two separate samplers) so that intermediate blend values produce
    a smooth interpolation rather than a discontinuous switch.
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
    Compute the (M,) importance signal for a candidate pool according to
    `strategy`. 'rar' (PDE residual magnitude) is the pre-existing default;
    'hessian' and the raw signal for 'importance'/'curriculum' are new.

    For 'importance' and 'curriculum', the underlying signal is still the PDE
    residual (matching RAR) — 'importance' changes HOW points are drawn from
    that signal (softmax temperature vs residual-proportional), 'curriculum'
    changes WHEN the signal is trusted (blend factor). Pass strategy='hessian'
    if you want curvature instead of residual as the underlying signal for
    importance/curriculum too — see compute_adaptive_weights_for.
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
