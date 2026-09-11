"""Spectral mode-importance XAI for FNO / WHNO / CNO-FNO."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _get_spectral_blocks(model: nn.Module) -> List[nn.Module]:
    """Locate the sequence of spectral-conv-containing blocks on a model."""
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
    """Per-mode importance for a single spectral block's weight tensor."""
    return weight.detach().abs().pow(2).sum(dim=(0, 1)).sqrt()


def model_mode_importance(model: nn.Module) -> List[torch.Tensor]:
    """Return one (mx,my,mz) importance tensor per spectral block in the model."""
    blocks = _get_spectral_blocks(model)
    return [block_mode_importance(_get_spectral_weight(b)) for b in blocks]


def marginal_importance(importance_3d: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Collapse a (mx,my,mz) importance map to a 1D profile along one axis by summing over the other two — "how much total energy sits at each mode
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
    Return (decay_a, decay_b) z-axis marginal importance profiles for two models (e.g. FNO3d vs WHNO3d trained on the SAME geometry) so their
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
    Causal importance: zero out successive low-sequency/low-frequency BANDS of one block's spectral weight (band 0 = lowest modes ... band n-1 =
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
