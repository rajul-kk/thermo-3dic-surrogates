"""
Physics helpers for the thermal PINN.

Provides:
  - thermal_conductivity(): layer-wise k(T) with silicon temperature dependence
  - pde_residual(): ∇·(k(T)∇T) + Q using autograd second derivatives
  - bc_residual_top(): convective BC at top surface: -k dT/dz = h(T - T_amb)
  - bc_residual_adiabatic(): dT/dn = 0 on side/bottom faces

All tensors in normalised units unless stated otherwise. Temperatures are
denormalised to Kelvin internally for k(T) evaluation, then returned to
normalised space for the PDE residual.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


# Silicon layer label — layer index 4 in geometry1, 4 and 8 in geometry2.
# Identified at training time by checking layer.material == 'silicon'.
_SILICON_K300 = 148.0   # W/m·K
_SILICON_ALPHA = 1.3    # power-law exponent from Glassbrenner & Slack (1964)
_T_REF = 300.0          # K


def thermal_conductivity(
    T_norm: torch.Tensor,
    layer_ids: torch.Tensor,
    layer_k: torch.Tensor,                        # (n_layers,) float, base k values in W/m·K
    si_layer_mask: torch.Tensor,                  # (n_layers,) bool, True for silicon layers
    T_min: float,
    T_max: float,
    col_k_lateral: Optional[torch.Tensor] = None, # (N,) per-point k override for 2p5d lateral variation
    col_si_lateral: Optional[torch.Tensor] = None, # (N,) per-point si mask override
) -> torch.Tensor:
    """
    Compute pointwise thermal conductivity k(T) in W/m·K.

    Silicon layers: k(T) = k_base * (300/T_K)^1.3  (temperature-dependent)
    All other layers: k = k_base  (constant)

    Args:
        T_norm:        (N,) normalised temperature [0,1]
        layer_ids:     (N,) int, layer index for each point
        layer_k:       (n_layers,) base thermal conductivity per layer
        si_layer_mask: (n_layers,) bool, True where material == silicon
        T_min, T_max:  normalisation bounds in Kelvin

    Returns:
        (N,) k values in W/m·K
    """
    T_K = T_norm * (T_max - T_min) + T_min   # denormalise to Kelvin
    T_K = T_K.clamp(100.0, 1685.0)           # physical bounds for silicon model

    # 2p5d_stack: use per-point lateral k/si overrides; otherwise standard layer lookup
    if col_k_lateral is not None:
        k_base = col_k_lateral
        is_si  = col_si_lateral
    else:
        k_base = layer_k[layer_ids]           # (N,) base k
        is_si  = si_layer_mask[layer_ids]     # (N,) bool

    k_si = k_base * (_T_REF / T_K) ** _SILICON_ALPHA
    return torch.where(is_si, k_si, k_base)


def _grad(output: torch.Tensor, inputs: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute gradient of sum(output) w.r.t. inputs."""
    return torch.autograd.grad(
        output.sum(), inputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def pde_residual(
    model: nn.Module,
    coords_col: torch.Tensor,       # (N_col, 3) normalised, requires_grad=True
    layer_ids_col: torch.Tensor,    # (N_col,) int
    power_col: torch.Tensor,        # (N_col,) normalised power density
    htc_norm: torch.Tensor,
    t_amb_norm: torch.Tensor,
    tsv_frac: torch.Tensor,
    layer_k: torch.Tensor,
    si_layer_mask: torch.Tensor,
    T_min: float,
    T_max: float,
    geom_scale: Tuple[float, float, float],           # (L_x, L_y, L_z) in µm for chain rule
    power_scale: float,                                # W/m³ per unit normalised power
    col_k_lateral: Optional[torch.Tensor] = None,     # (N_col,) per-point k for 2p5d lateral variation
    col_si_lateral: Optional[torch.Tensor] = None,    # (N_col,) per-point si mask for 2p5d
    region_ids_col: Optional[torch.Tensor] = None,    # (N_col,) chiplet region ids; passed to model
    tim_k_norm: Optional[torch.Tensor] = None,        # scalar TIM conductivity norm; passed to model
) -> torch.Tensor:
    """
    Compute PDE residual r = ∇·(k(T)∇T) + Q at collocation points.

    The network outputs T̂ (normalised). Spatial coordinates are also normalised
    to [0,1]. The chain rule converts derivatives from normalised to physical space:
        dT/dx_phys = (dT̂/dx̂) * (T_range) / L_x

    Returns:
        (N_col,) residual values (should be ~0 everywhere inside the domain)
    """
    coords_col = coords_col.requires_grad_(True)

    T_hat = model(
        coords_col, layer_ids_col, power_col, htc_norm, t_amb_norm, tsv_frac,
        region_ids=region_ids_col, tim_k_norm=tim_k_norm,
    )

    # First derivatives in normalised space
    dT_hat = _grad(T_hat, coords_col)          # (N_col, 3)  d(T̂)/d(x̂,ŷ,ẑ)

    T_range = T_max - T_min                    # K
    L_x, L_y, L_z = geom_scale                # µm

    # Convert to physical derivatives (K/µm)
    dT_dx = dT_hat[:, 0] * T_range / L_x
    dT_dy = dT_hat[:, 1] * T_range / L_y
    dT_dz = dT_hat[:, 2] * T_range / L_z

    # Quasi-linearisation: k is frozen to the current T_hat prediction so the
    # PDE residual is linear in T for this gradient step. k implicitly updates
    # each epoch as T_hat improves (Picard/fixed-point iteration).
    # Avoids ~30% extra memory/time vs differentiating through k(T) with no
    # meaningful accuracy loss at PDE weight=0.1.
    k = thermal_conductivity(T_hat.detach(), layer_ids_col, layer_k, si_layer_mask, T_min, T_max,
                             col_k_lateral=col_k_lateral, col_si_lateral=col_si_lateral)

    # k * grad(T) in physical space (W/m·K * K/µm = W/m·µm = 1e-6 W/m²)
    # We keep units consistent by working in µm throughout; the PDE becomes:
    #   ∇·(k ∇T) + Q_µm = 0  where Q_µm = Q [W/m³] * 1e-18 [m³/µm³]
    kTx = k * dT_dx   # (N_col,)
    kTy = k * dT_dy
    kTz = k * dT_dz

    # Second derivatives: d(kTx)/dx via autograd
    # We need to differentiate through k*dT_dx, which itself contains T_hat
    # Use create_graph=True so higher-order gradients flow back
    div_kT = (
        _grad(kTx, coords_col)[:, 0] / L_x
        + _grad(kTy, coords_col)[:, 1] / L_y
        + _grad(kTz, coords_col)[:, 2] / L_z
    ) * T_range  # chain rule back to physical units

    # Power source term Q in W/m³, scaled to match div_kT units (W/µm³ * 1e18)
    Q_phys = power_col * power_scale * 1e-18   # W/µm³

    return div_kT + Q_phys   # should be 0


def bc_residual_top(
    model: nn.Module,
    coords_top: torch.Tensor,       # (N_bc, 3) normalised, on the convective face
    layer_ids_top: torch.Tensor,
    power_top: torch.Tensor,
    htc_norm: torch.Tensor,
    t_amb_norm: torch.Tensor,
    tsv_frac: torch.Tensor,
    layer_k: torch.Tensor,
    si_layer_mask: torch.Tensor,
    T_min: float,
    T_max: float,
    L_z: float,          # physical domain height in µm
    htc: float,          # dimensional HTC in W/m²·K = W/(µm²·K) * 1e12
    T_amb_K: float,      # dimensional ambient temperature in K
    region_ids_top: Optional[torch.Tensor] = None,
    tim_k_norm: Optional[torch.Tensor] = None,
    outward_normal_sign: float = 1.0,   # +1 at z=1 (outward normal = +z); -1 at z=0 (outward normal = -z)
) -> torch.Tensor:
    """
    Convective BC residual: -k * dT/dn = h * (T - T_amb), where n is the
    OUTWARD surface normal.

    Despite the name (kept for backward compatibility — originally this was
    only ever called at z=1), this function works at ANY z-face; pass
    outward_normal_sign=-1.0 when evaluating at z=0 (bottom), since the
    outward normal there points in -z, flipping the sign of dT/dn relative
    to dT/dz. Ground truth for this dataset (see ice_simulator.py's "bottom
    heat sink" directive) applies convective cooling at z=0, layer index 0
    — callers should use outward_normal_sign=-1.0 in the normal case here,
    not the historical z=1/+1.0 default.

    Returns (N_bc,) residual (should be ~0).
    """
    coords_top = coords_top.requires_grad_(True)
    T_hat = model(
        coords_top, layer_ids_top, power_top, htc_norm, t_amb_norm, tsv_frac,
        region_ids=region_ids_top, tim_k_norm=tim_k_norm,
    )

    T_range = T_max - T_min
    dT_hat_dz = _grad(T_hat, coords_top)[:, 2]   # d(T̂)/dẑ
    dT_dz_phys = dT_hat_dz * T_range / L_z        # K/µm

    k = thermal_conductivity(T_hat.detach(), layer_ids_top, layer_k, si_layer_mask, T_min, T_max)

    T_K = T_hat * T_range + T_min
    htc_um = htc * 1e-12   # W/(m²·K) → W/(µm²·K)

    # -k [W/(m·K)] * dT/dn [K/µm] = -k*1e-6 [W/(µm·K)] * (sign * dT/dz) [K/µm]  →  W/µm²
    # h [W/(µm²·K)] * (T - T_amb) [K]  →  W/µm²
    lhs = -outward_normal_sign * (k * 1e-6) * dT_dz_phys      # W/µm²
    rhs = htc_um * (T_K - T_amb_K)      # W/µm²
    return lhs - rhs


def bc_residual_adiabatic(
    model: nn.Module,
    coords_side: torch.Tensor,      # (N_bc, 3) normalised, on a side/bottom face
    layer_ids_side: torch.Tensor,
    power_side: torch.Tensor,
    htc_norm: torch.Tensor,
    t_amb_norm: torch.Tensor,
    tsv_frac: torch.Tensor,
    normal_dim: int,                # 0=x, 1=y, 2=z (which face normal)
    geom_scale: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Adiabatic BC residual: dT/dn = 0.

    Returns (N_bc,) residual.
    """
    coords_side = coords_side.requires_grad_(True)
    T_hat = model(coords_side, layer_ids_side, power_side, htc_norm, t_amb_norm, tsv_frac)

    dT_hat = _grad(T_hat, coords_side)                    # (N_bc, 3)
    L = geom_scale[normal_dim]
    T_range = 1.0   # gradient in normalised space (residual = 0 in any scaling)
    return dT_hat[:, normal_dim] * T_range / L            # dT/dn (K/µm)
