"""Spatially varying TSV density, and the anisotropic conductivity it implies."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .power_maps import _normalise, _smooth

# Bulk conductivities [W/m.K]. Silicon is the 300 K value; k(T) is applied by the
# PINN physics, not baked into the ground-truth material.
K_SI = 148.0
K_CU = 400.0


def generate_tsv_density_map(
    n_l: int,
    n_w: int,
    mean_density: float = 0.05,
    seed: int = 0,
    corr_frac: float = 0.06,
    contrast: float = 0.8,
    max_density: float = 0.35,
) -> np.ndarray:
    """Build a spatially varying TSV area-fraction field."""
    if not 0.0 <= mean_density <= max_density:
        raise ValueError(f"mean_density {mean_density} outside [0, {max_density}]")
    if mean_density == 0.0:
        return np.zeros((n_l, n_w))

    rng = np.random.default_rng(seed)
    field = _normalise(_smooth(rng.standard_normal((n_l, n_w)),
                               max(corr_frac * max(n_l, n_w), 0.75)), 0.0)

    # Blend between uniform and clustered, then rescale to the requested mean.
    phi = (1.0 - contrast) + contrast * field
    phi *= mean_density / phi.mean()
    return np.clip(phi, 0.0, max_density)


def tsv_effective_k(phi: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
    """Anisotropic effective conductivity for a TSV region."""
    phi = np.asarray(phi, dtype=np.float64)
    k_vertical = (1.0 - phi) * K_SI + phi * K_CU
    k_lateral = 1.0 / ((1.0 - phi) / K_SI + phi / K_CU)
    return k_lateral, k_vertical


def quantise_materials(phi: np.ndarray, n_levels: int = 12
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a continuous density field to a small set of material levels."""
    lo, hi = float(phi.min()), float(phi.max())
    if hi - lo < 1e-9:
        return np.zeros_like(phi, dtype=int), np.array([lo])
    edges = np.linspace(lo, hi, n_levels + 1)
    idx = np.clip(np.digitize(phi, edges[1:-1]), 0, n_levels - 1)
    centres = np.array([0.5 * (edges[i] + edges[i + 1]) for i in range(n_levels)])
    return idx, centres
