"""PI-FNO: physics residual via finite differences on the FNO output grid."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def build_k_grid(
    layer_id_norm: torch.Tensor,      # (B, nx, ny, nz)  in [0, 1]
    n_layers: int,
    layer_k: torch.Tensor,            # (n_layers,)  W/m·K
    col_k_lateral: Optional[torch.Tensor] = None,  # (nx*ny*nz,) for one sample
) -> torch.Tensor:
    """Map layer-id grid to conductivity grid (B, nx, ny, nz) in W/m·K."""
    layer_idx = (layer_id_norm * max(n_layers - 1, 1)).round().long()   # (B, nx, ny, nz)
    layer_idx = layer_idx.clamp(0, len(layer_k) - 1)
    k_grid = layer_k[layer_idx]                                          # (B, nx, ny, nz)
    return k_grid


def _harmonic_mean(ka: torch.Tensor, kb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Two-point harmonic mean face conductivity: 2*ka*kb / (ka+kb)."""
    return 2.0 * ka * kb / (ka + kb + eps)


def fd_divergence(
    T_phys: torch.Tensor,   # (B, nx, ny, nz)  physical temperature [K]
    k: torch.Tensor,        # (B, nx, ny, nz)  conductivity [W/m·K]
    dx: float,              # [m]
    dy: float,              # [m]
    dz: float,              # [m]
) -> torch.Tensor:
    """Compute ∇·(k ∇T) via central FV differences on a regular grid."""
    # Pad T and k by 1 on each spatial side using replicate (extends last real value)
    T = F.pad(T_phys, (1, 1, 1, 1, 1, 1), mode='replicate')  # (B, nx+2, ny+2, nz+2)
    k_pad = F.pad(k,   (1, 1, 1, 1, 1, 1), mode='replicate')

    # --- x direction ---
    k_xp = _harmonic_mean(k_pad[:, 2:, 1:-1, 1:-1], k_pad[:, 1:-1, 1:-1, 1:-1])  # k at i+½
    k_xm = _harmonic_mean(k_pad[:, :-2, 1:-1, 1:-1], k_pad[:, 1:-1, 1:-1, 1:-1])  # k at i-½
    div_x = (k_xp * (T[:, 2:, 1:-1, 1:-1] - T[:, 1:-1, 1:-1, 1:-1])
           - k_xm * (T[:, 1:-1, 1:-1, 1:-1] - T[:, :-2, 1:-1, 1:-1])) / dx**2

    # --- y direction ---
    k_yp = _harmonic_mean(k_pad[:, 1:-1, 2:, 1:-1], k_pad[:, 1:-1, 1:-1, 1:-1])
    k_ym = _harmonic_mean(k_pad[:, 1:-1, :-2, 1:-1], k_pad[:, 1:-1, 1:-1, 1:-1])
    div_y = (k_yp * (T[:, 1:-1, 2:, 1:-1] - T[:, 1:-1, 1:-1, 1:-1])
           - k_ym * (T[:, 1:-1, 1:-1, 1:-1] - T[:, 1:-1, :-2, 1:-1])) / dy**2

    # --- z direction ---
    k_zp = _harmonic_mean(k_pad[:, 1:-1, 1:-1, 2:], k_pad[:, 1:-1, 1:-1, 1:-1])
    k_zm = _harmonic_mean(k_pad[:, 1:-1, 1:-1, :-2], k_pad[:, 1:-1, 1:-1, 1:-1])
    div_z = (k_zp * (T[:, 1:-1, 1:-1, 2:] - T[:, 1:-1, 1:-1, 1:-1])
           - k_zm * (T[:, 1:-1, 1:-1, 1:-1] - T[:, 1:-1, 1:-1, :-2])) / dz**2

    return div_x + div_y + div_z   # (B, nx, ny, nz)


def pde_loss_fd(
    T_norm: torch.Tensor,         # (B, nx, ny, nz)  FNO output  [0,1]
    Q_norm: torch.Tensor,         # (B, nx, ny, nz)  power density, normalised
    layer_id_norm: torch.Tensor,  # (B, nx, ny, nz)  layer id / (n_layers-1)
    layer_k: torch.Tensor,        # (n_layers,)  [W/m·K]
    n_layers: int,
    T_min: float,                 # [K]
    T_max: float,                 # [K]
    power_std: float,             # [W/m³]  normalisation denominator
    dx: float,                    # [m]  physical cell width
    dy: float,                    # [m]  physical cell height
    dz: float,                    # [m]  mean physical cell depth
) -> torch.Tensor:
    """Finite-difference PDE residual loss for PI-FNO."""
    T_range = T_max - T_min

    # Convert to physical
    T_phys = T_norm.float() * T_range + T_min   # [K]
    Q_phys = Q_norm.float() * power_std         # [W/m³]

    # Build conductivity field
    k_grid = build_k_grid(layer_id_norm, n_layers, layer_k.float())  # [W/m·K]

    # Finite-difference divergence
    div_kgradT = fd_divergence(T_phys, k_grid, dx, dy, dz)           # [W/m³]

    # PDE residual: ∇·(k∇T) + Q = 0
    residual = div_kgradT + Q_phys                                     # [W/m³]

    # Normalise residual by power_std so loss is O(1) regardless of geometry
    residual_norm = residual / (power_std + 1e-8)

    return (residual_norm ** 2).mean()


def interface_flux_loss(
    T_norm: torch.Tensor,         # (B, nx, ny, nz)  FNO output, [0,1] normalised
    layer_id_norm: torch.Tensor,  # (B, nx, ny, nz)  layer id / (n_layers-1)
    layer_k: torch.Tensor,        # (n_layers,)  [W/m·K]
    n_layers: int,
    T_min: float,                 # [K]
    T_max: float,                 # [K]
    dz: float,                    # [m]  physical z cell spacing
) -> torch.Tensor:
    """
    Flux-continuity residual at each material interface, using one-sided finite differences computed INDEPENDENTLY on each side of the boundary
    """
    B, nx, ny, nz = T_norm.shape
    T_range = T_max - T_min
    T_phys = T_norm.float() * T_range + T_min   # (B, nx, ny, nz)  [K]

    layer_idx_z = (layer_id_norm[:, 0, 0, :].float() * max(n_layers - 1, 1)).round().long()  # (B, nz)

    sq_terms = []
    for b in range(B):
        idx = layer_idx_z[b]
        change_pts = (idx[1:] != idx[:-1]).nonzero(as_tuple=True)[0] + 1  # z-index where new layer starts
        for iz in change_pts.tolist():
            if iz - 2 < 0 or iz + 1 >= nz:
                continue   # not enough cells on one side to form a one-sided difference
            lower_layer = int(idx[iz - 1])
            upper_layer = int(idx[iz])
            if int(idx[iz - 2]) != lower_layer or int(idx[iz + 1]) != upper_layer:
                continue   # adjacent layer is only 1 cell thick — skip (too coarse to resolve)

            k_lo = layer_k[lower_layer]
            k_hi = layer_k[upper_layer]
            flux_lo = k_lo * (T_phys[b, :, :, iz - 1] - T_phys[b, :, :, iz - 2]) / dz   # (nx, ny)
            flux_hi = k_hi * (T_phys[b, :, :, iz + 1] - T_phys[b, :, :, iz]) / dz         # (nx, ny)
            resid = flux_lo - flux_hi
            sq_terms.append((resid ** 2).mean())

    if not sq_terms:
        return torch.zeros((), device=T_norm.device, dtype=T_norm.dtype)
    return torch.stack(sq_terms).mean()


def grid_spacings(geometry) -> Tuple[float, float, float]:
    """Return (dx, dy, dz) in metres for a geometry's Cartesian mesh."""
    nx, ny, nz = geometry.mesh_resolution
    dx = (geometry.die_width  * 1e-6) / nx    # µm → m
    dy = (geometry.die_length * 1e-6) / ny
    dz = (geometry.get_total_height() * 1e-6) / nz
    return dx, dy, dz
