"""torch.func-based PDE residual for PINN training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def _has_func() -> bool:
    try:
        from torch.func import vmap, grad, functional_call   # noqa: F401
        return True
    except ImportError:
        return False


def pde_residual_func(
    model,
    col_coords: torch.Tensor,          # (N, 3)  requires_grad not needed
    col_ids: torch.Tensor,             # (N,)  int
    col_power: torch.Tensor,           # (N,)  normalised
    htc_norm: torch.Tensor,            # scalar
    t_amb_norm: torch.Tensor,          # scalar
    tsv_frac: torch.Tensor,            # scalar
    layer_k: torch.Tensor,             # (n_layers,) [W/m·K]
    si_mask: torch.Tensor,             # (n_layers,)  bool
    T_min: float,
    T_max: float,
    geom_scale: Tuple[float, float, float],   # (Lx, Ly, Lz) [µm]
    power_std: float,
    col_k_lateral: Optional[torch.Tensor] = None,
    col_si_lateral: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-point PDE residual |∇·(k∇T) + Q| via vmap + grad."""
    from torch.func import vmap, grad, functional_call

    params  = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    Lx, Ly, Lz = [s * 1e-6 for s in geom_scale]
    T_range = float(T_max - T_min)

    # ------------------------------------------------------------------
    # Build per-point layer conductivities (same quasi-linearisation as
    # the existing autograd path — k is not differentiated through).
    # ------------------------------------------------------------------
    if col_k_lateral is not None:
        k_per_pt  = col_k_lateral.float()   # (N,)
        is_si_pt  = col_si_lateral.bool()   # (N,)
    else:
        k_per_pt  = layer_k[col_ids]        # (N,)
        is_si_pt  = si_mask[col_ids]        # (N,)

    # ------------------------------------------------------------------
    # Single-point scalar forward (for grad differentiation).
    # Dropout is put in eval mode inside vmap to guarantee determinism.
    # ------------------------------------------------------------------
    def T_scalar(coord: torch.Tensor,          # (3,)
                 layer_id: torch.Tensor,       # ()  int
                 power_val: torch.Tensor,      # ()
                 ) -> torch.Tensor:            # scalar
        # functional_call with eval-style dropout (training=False forces
        # Dropout to identity, keeping the computation graph clean)
        out = functional_call(
            model, (params, buffers),
            args=(coord.unsqueeze(0),
                  layer_id.view(1),
                  power_val.view(1),
                  htc_norm,
                  t_amb_norm,
                  tsv_frac),
            kwargs={'training': False},       # disables Dropout stochasticity
        )
        return out.squeeze()                  # scalar

    # ------------------------------------------------------------------
    # First spatial derivatives via vmap(grad(T_scalar)).
    # grad differentiates w.r.t. the first positional argument (coord).
    # vmap vectorises over the N-point axis.
    # ------------------------------------------------------------------
    dT_norm_dcoord_norm = vmap(
        grad(T_scalar, argnums=0),
        in_dims=(0, 0, 0),
    )(col_coords, col_ids, col_power)           # (N, 3)

    # Physical gradients [K/m]
    dTdx = dT_norm_dcoord_norm[:, 0] * T_range / Lx
    dTdy = dT_norm_dcoord_norm[:, 1] * T_range / Ly
    dTdz = dT_norm_dcoord_norm[:, 2] * T_range / Lz

    # k(T) correction for Si (quasi-linearised: use T from a no-grad pass)
    with torch.no_grad():
        T_norm_no_grad = vmap(T_scalar, in_dims=(0, 0, 0))(
            col_coords, col_ids, col_power
        )
    T_phys_detached = (T_norm_no_grad * T_range + T_min).clamp(200.0, 1600.0)
    k_base = k_per_pt.clone().float()
    if is_si_pt.any():
        k_base[is_si_pt] = 148.0 * (300.0 / T_phys_detached[is_si_pt]) ** 1.3

    # Heat flux components [W/m²]
    kTx = k_base * dTdx
    kTy = k_base * dTdy
    kTz = k_base * dTdz

    # ------------------------------------------------------------------
    # Second derivatives (divergence) via a second vmap + grad layer.
    # We compute ∂(k·∂T/∂x)/∂x + ∂(k·∂T/∂y)/∂y + ∂(k·∂T/∂z)/∂z.
    #
    # For each point i: div_i = sum_d ∂(k_i · ∂T/∂x_d) / ∂x_d
    # k_i is frozen (quasi-linearised) so we only differentiate through T.
    # ------------------------------------------------------------------
    def flux_sum_scalar(coord, layer_id, power_val, k_i):
        """kTx + kTy + kTz at one point — scalar so grad can differentiate it."""
        g = grad(T_scalar, argnums=0)(coord, layer_id, power_val)  # (3,)
        g_phys = g * T_range * torch.tensor([1/Lx, 1/Ly, 1/Lz],
                                             device=coord.device, dtype=g.dtype)
        return (k_i * g_phys).sum()   # scalar: k*(dT/dx + dT/dy + dT/dz)

    # We want ∂(k_i·∂T/∂x_d)/∂x_d summed over d.
    # This equals grad_coord(flux_sum_scalar)[0] + [1] + [2]
    # = sum(grad of flux_sum w.r.t. coord)
    div_vals = vmap(
        lambda c, lid, pv, ki:
            grad(lambda cc: flux_sum_scalar(cc, lid, pv, ki), argnums=0)(c).sum(),
        in_dims=(0, 0, 0, 0),
    )(col_coords, col_ids, col_power, k_base)        # (N,)
    # div_vals[i] = ∂(kTx)/∂x + ∂(kTy)/∂y + ∂(kTz)/∂z at point i

    # PDE residual: ∇·(k∇T) + Q = 0  →  residual = div + Q_phys
    Q_phys = col_power * power_std                   # [W/m³]
    residual = div_vals + Q_phys

    return residual   # (N,)  [W/m³] unnormalised
