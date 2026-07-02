"""
Spectral mode-importance XAI for FNO / WHNO / CNO-FNO.

Unlike the PINN's post-hoc XAI (gradient-based, requires forward passes),
an operator with a spectral basis has interpretability NATIVE to its
architecture: the learned `weight` tensor of each spectral block IS a
per-mode importance map, readable directly off the trained parameters with
zero forward passes.

Both FNO's SpectralConv3d and WHNO's WalshConv3d store their truncated
`weight` as (in_ch, out_ch, mx, my, mz), already ordered low-index=smooth
to high-index=oscillatory (FFT's rfft ordering for Fourier; sequency
ordering — see whno.hadamard_to_sequency_perm — for Walsh, already applied
before truncation). So mode index 0 along any axis is the smoothest
component retained, with no extra basis-specific conversion needed to
compare "does this model emphasise fine detail or bulk smoothness" between
the two bases.

Use case: cross-reference this against known material-interface locations
(see src/fno/physics.py's interface z-index detection) to test the
hypothesis that WHNO concentrates more importance in finer z-bands (because
interfaces are piecewise-constant discontinuities the Walsh basis
represents natively) while FNO's importance decays faster and is truncated
before it can represent the interface (the Gibbs-ringing argument, already
validated on a synthetic step function in whno.py's docstring — this module
lets you check whether that pattern shows up in ACTUAL TRAINED WEIGHTS, not
just the toy 1D demo).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _get_spectral_blocks(model: nn.Module) -> List[nn.Module]:
    """
    Locate the sequence of spectral-conv-containing blocks on a model.

    Supports FNO3d/WHNO3d (`.blocks`, each with `.spectral`) and
    CNOFNOHybrid (`.latent_blocks`, each with `.spectral` OR an axial-
    attention wrapper around one -- see AxialAttentionFiLMBlock).
    """
    if hasattr(model, 'blocks'):
        return list(model.blocks)
    if hasattr(model, 'latent_blocks'):
        return list(model.latent_blocks)
    raise ValueError(
        f"{type(model).__name__}: no '.blocks' or '.latent_blocks' attribute found — "
        "spectral mode importance requires a model with a sequence of spectral-conv blocks."
    )


def _get_spectral_weight(block: nn.Module) -> torch.Tensor:
    """Extract the (in_ch, out_ch, mx, my, mz) weight tensor from a block."""
    target = block
    # AxialAttentionFiLMBlock and FiLMFNOBlock wrap an inner spectral conv;
    # try common attribute names in priority order.
    for attr in ('spectral',):
        if hasattr(target, attr):
            inner = getattr(target, attr)
            if hasattr(inner, 'weight'):
                return inner.weight
    raise ValueError(f"{type(block).__name__}: no '.spectral.weight' found")


def block_mode_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    Per-mode importance for a single spectral block's weight tensor.

    weight: (in_ch, out_ch, mx, my, mz), real or complex.
    Returns (mx, my, mz) real-valued Frobenius-norm-over-channels importance
    map. Works identically for FNO's complex weights (via .abs()) and WHNO's
    real weights.
    """
    return weight.detach().abs().pow(2).sum(dim=(0, 1)).sqrt()


def model_mode_importance(model: nn.Module) -> List[torch.Tensor]:
    """Return one (mx,my,mz) importance tensor per spectral block in the model."""
    blocks = _get_spectral_blocks(model)
    return [block_mode_importance(_get_spectral_weight(b)) for b in blocks]


def marginal_importance(importance_3d: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Collapse a (mx,my,mz) importance map to a 1D profile along one axis by
    summing over the other two — "how much total energy sits at each mode
    index along this axis, regardless of the other two axes."

    axis: 0=x, 1=y, 2=z (z is the axis carrying material-interface structure
    in these layered-stack geometries — most informative for the interface
    hypothesis).
    """
    dims = [0, 1, 2]
    dims.remove(axis)
    return importance_3d.sum(dim=tuple(dims))


def z_axis_decay_profile(model: nn.Module, block_idx: int = 0) -> torch.Tensor:
    """Convenience: z-axis marginal importance for one block (default: first)."""
    imp = model_mode_importance(model)[block_idx]
    return marginal_importance(imp, axis=2)


def compare_z_decay(
    model_a: nn.Module,
    model_b: nn.Module,
    block_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (decay_a, decay_b) z-axis marginal importance profiles for two
    models (e.g. FNO3d vs WHNO3d trained on the SAME geometry) so their
    energy-vs-mode-index decay curves can be compared directly.

    Only meaningful on trained checkpoints — on randomly-initialised models
    both profiles will be roughly flat/noisy since there's no learned signal
    yet.
    """
    return z_axis_decay_profile(model_a, block_idx), z_axis_decay_profile(model_b, block_idx)


# ---------------------------------------------------------------------------
# Ablation-based (causal) importance — optional, more expensive
# ---------------------------------------------------------------------------

def ablate_and_measure(
    model: nn.Module,
    forward_fn,                     # callable() -> scalar val loss (e.g. MAE on val set)
    block_idx: int,
    n_bands: int = 8,
) -> List[float]:
    """
    Causal importance: zero out successive low-sequency/low-frequency BANDS
    of one block's spectral weight (band 0 = lowest modes ... band n-1 =
    highest retained modes), re-measure `forward_fn()` after each ablation,
    restore the original weight before returning.

    Returns a list of `val_loss_after_ablating_band_i` for i in [0, n_bands).
    Compare against the baseline (unablated) forward_fn() value to see which
    bands are load-bearing vs redundant -- a causal complement to the
    magnitude-based importance above (magnitude tells you where the model
    PUT weight; ablation tells you where it actually MATTERS for accuracy).

    Cost: n_bands forward passes over the val set. Cheap for FNO/WHNO
    (inference is fast — seconds, not hours) but O(n_bands) wall time.
    """
    blocks = _get_spectral_blocks(model)
    weight = _get_spectral_weight(blocks[block_idx])
    original = weight.data.clone()

    mx = weight.shape[2]
    band_edges = torch.linspace(0, mx, n_bands + 1).round().long()

    results = []
    try:
        for i in range(n_bands):
            lo, hi = int(band_edges[i]), int(band_edges[i + 1])
            weight.data.copy_(original)
            weight.data[:, :, lo:hi, :, :] = 0
            results.append(float(forward_fn()))
    finally:
        weight.data.copy_(original)   # always restore, even if forward_fn raises

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mode_importance_comparison(
    profiles: Dict[str, torch.Tensor],   # {label: 1D importance profile}
    output_path,
    title: str = 'Spectral mode importance (z-axis, low=smooth -> high=oscillatory)',
) -> None:
    """Overlay multiple models' 1D mode-importance-vs-index profiles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, profile in profiles.items():
        p = profile.detach().cpu().numpy()
        p_norm = p / (p.sum() + 1e-12)   # normalise so different mode counts are comparable
        ax.plot(range(len(p_norm)), p_norm, marker='o', markersize=3, label=label)
    ax.set_xlabel('mode index (0 = smoothest retained mode)')
    ax.set_ylabel('fraction of total block energy')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
