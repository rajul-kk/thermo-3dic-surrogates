"""
Spatially varying TSV density, and the anisotropic conductivity it implies.

Why
---
TSV density was a single scalar per geometry taking four discrete values
(0, 3%, 5%, 10%). That is a categorical variant axis, not a parameter: with three
non-zero points no interpolation can be demonstrated, and the paper's claim of a
"continuous TSV parameter" was simply false (docs/report.md).

Real TSV arrays are not uniform. They cluster around the interconnect they serve,
leaving sparse regions between. Making density a FIELD phi(x, y) turns the
material coefficient into a per-scenario spatial input, which is the canonical
operator-learning benchmark structure -- the thermal analogue of Darcy flow with a
Gaussian-random-field permeability. It is also the part of the solution operator
that is genuinely nonlinear: the Green's function depends nonlinearly on the
coefficients, while it is merely linear in the source.

This requires 3D-ICE 4.0, which supports per-floorplan-element materials and
anisotropic conductivity. 3.0.0 allowed one isotropic material per layer.

Anisotropy
----------
A TSV is a copper cylinder through silicon, so conduction is directional:

  vertical (kz)   heat runs ALONG the copper -- the phases act in parallel, so the
                  arithmetic mean (rule of mixtures) applies:
                      kz = (1 - phi) k_Si + phi k_Cu
  lateral (kx,ky) heat must cross alternating Si/Cu boundaries -- the phases act in
                  series, so the harmonic mean applies:
                      1/kxy = (1 - phi)/k_Si + phi/k_Cu

The old model used the arithmetic mean in ALL directions, which
`docs/references.md` already flagged as an upper bound that "slightly overestimates
lateral heat spreading from TSVs". At phi = 0.10 the two differ by 17%
(183.8 vs 152.4 W/m.K). Anisotropic materials let us stop making that
approximation, so this fixes a documented simplification rather than adding one.
"""

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
    """
    Build a spatially varying TSV area-fraction field.

    Args:
        n_l, n_w:      Map resolution (along die length, along die width).
        mean_density:  Target mean area fraction, i.e. the scalar this replaces.
        seed:          Per-scenario seed.
        corr_frac:     Cluster size as a fraction of the die edge. ~0.06 gives
                       ~600 um clusters on a 10 mm die, the scale of a real TSV farm.
        contrast:      0 = uniform (reproduces the old scalar), 1 = maximum
                       clustering. Lets a scenario sweep from uniform to highly
                       clustered.
        max_density:   Hard ceiling. Real arrays do not approach 100% copper;
                       above ~35% the rule of mixtures stops being credible and
                       the layer is better modelled as bulk copper.

    Returns:
        (n_l, n_w) array of area fractions in [0, max_density], mean ~= mean_density.
    """
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
    """
    Anisotropic effective conductivity for a TSV region.

    Returns (k_lateral, k_vertical) in W/m.K. See the module docstring: vertical
    conduction runs along the copper (parallel, arithmetic mean) while lateral
    conduction crosses the phases (series, harmonic mean).
    """
    phi = np.asarray(phi, dtype=np.float64)
    k_vertical = (1.0 - phi) * K_SI + phi * K_CU
    k_lateral = 1.0 / ((1.0 - phi) / K_SI + phi / K_CU)
    return k_lateral, k_vertical


def quantise_materials(phi: np.ndarray, n_levels: int = 12
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce a continuous density field to a small set of material levels.

    3D-ICE needs a named material per distinct conductivity, so a 100x100 field of
    unique values would mean 10,000 material declarations. Quantising to ~12 levels
    keeps the stack file tractable while preserving the spatial structure that
    matters. The error this introduces is far below the modelling error in the rule
    of mixtures itself.

    Returns (level_index_map, level_phi_values).
    """
    lo, hi = float(phi.min()), float(phi.max())
    if hi - lo < 1e-9:
        return np.zeros_like(phi, dtype=int), np.array([lo])
    edges = np.linspace(lo, hi, n_levels + 1)
    idx = np.clip(np.digitize(phi, edges[1:-1]), 0, n_levels - 1)
    centres = np.array([0.5 * (edges[i] + edges[i + 1]) for i in range(n_levels)])
    return idx, centres
