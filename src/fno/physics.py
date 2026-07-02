"""
PI-FNO: physics residual via finite differences on the FNO output grid.

Unlike the PINN which uses autograd second derivatives through a point-wise MLP,
the FNO already produces a full 3D grid at each forward pass.  We can therefore
evaluate the heat equation cheaply with central finite differences on that grid
— no computation graph retention, no third-order autograd, just tensor ops.

PDE (steady-state heat):   ∇·(k ∇T) + Q = 0

Finite-difference approximation on a regular Cartesian mesh (dx, dy uniform;
dz uniform approximation using mean layer thickness):

    ∂/∂x (k ∂T/∂x) ≈  [k_{i+½}(T_{i+1}-T_i) - k_{i-½}(T_i-T_{i-1})] / dx²

    where  k_{i+½} = (k_i + k_{i+1}) / 2  (arithmetic mean at face)

This conservative finite-volume stencil preserves energy balance at material
interfaces (k discontinuities between layers).

Coordinate convention (physical, metres):
    dx = die_width_m  / nx
    dy = die_length_m / ny
    dz = total_height_m / nz   (mean cell; valid approximation for regularised
                                  meshes where adaptive z-spacing is < 2× uniform)
"""

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
    """
    Map layer-id grid to conductivity grid (B, nx, ny, nz) in W/m·K.

    layer_id_norm ∈ [0,1] → layer index = round(layer_id_norm * (n_layers-1))
    """
    layer_idx = (layer_id_norm * max(n_layers - 1, 1)).round().long()   # (B, nx, ny, nz)
    layer_idx = layer_idx.clamp(0, len(layer_k) - 1)
    k_grid = layer_k[layer_idx]                                          # (B, nx, ny, nz)
    return k_grid


def _harmonic_mean(ka: torch.Tensor, kb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Two-point harmonic mean face conductivity: 2*ka*kb / (ka+kb).

    This is the standard finite-volume face-conductivity choice for
    heterogeneous diffusion (Patankar, "Numerical Heat Transfer and Fluid
    Flow", 1980) — it exactly reproduces flux continuity for a piecewise-
    constant conductivity field with a two-point flux approximation.
    Arithmetic mean (ka+kb)/2, used previously, has O(1) relative error
    when ka/kb is large — exactly the regime at every material interface in
    this dataset (Si k=148 vs TIM k=4 W/m·K, a 37x ratio).
    """
    return 2.0 * ka * kb / (ka + kb + eps)


def fd_divergence(
    T_phys: torch.Tensor,   # (B, nx, ny, nz)  physical temperature [K]
    k: torch.Tensor,        # (B, nx, ny, nz)  conductivity [W/m·K]
    dx: float,              # [m]
    dy: float,              # [m]
    dz: float,              # [m]
) -> torch.Tensor:
    """
    Compute ∇·(k ∇T) via central FV differences on a regular grid.

    Returns (B, nx, ny, nz) [W/m³].  Interior cells only — boundary rows
    use one-sided differences (forward/backward) so the output has the same
    shape as the input without cropping.

    Face conductivity uses the HARMONIC mean of adjacent cells (see
    _harmonic_mean) — required for flux accuracy across the large
    conductivity contrasts at material interfaces in this dataset.
    """
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
    """
    Finite-difference PDE residual loss for PI-FNO.

    L_pde = mean_over_batch_and_grid( (∇·(k∇T) + Q)² )

    All quantities are converted to physical units before differencing so the
    loss has units of (W/m³)² — consistent across geometries.

    Gradient is NOT tracked through this loss (no create_graph) — it serves
    purely as a regulariser. AMP autocast is compatible: operates in float32
    because F.pad and float arithmetic stay in fp32 under autocast.
    """
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
    Flux-continuity residual at each material interface, using one-sided
    finite differences computed INDEPENDENTLY on each side of the boundary
    (not the blended harmonic-mean face conductivity fd_divergence uses for
    the bulk PDE residual). This directly checks the interface condition
    k_lower * dT/dz|- == k_upper * dT/dz|+ using two separately-estimated
    one-sided derivatives, rather than a single symmetric stencil straddling
    the boundary — a more direct, independently-reportable diagnostic of
    interface behaviour, and lets flux-continuity be weighted separately
    from (and typically much higher than) the diluted whole-volume PDE loss.

    Interface z-indices are located PER-SAMPLE from layer_id_norm at runtime
    (not hardcoded), by scanning where the layer index changes along z.
    layer_id_norm is assumed constant across (x,y) within a z-slice, which
    holds for all layered-stack geometries in this dataset (2d_stack,
    3d_stack, 2p5d_stack all assign layer index purely by z-range; lateral
    heterogeneity in 2p5d_stack geometries is a separate k-override
    mechanism, not a layer-index change).

    An interface is skipped (not scored) if either adjacent layer has fewer
    than 2 z-cells in the mesh (too thin to form a one-sided difference) —
    this can happen for very thin layers (e.g. a 5µm hybrid-bonding layer)
    at coarse z-resolution.

    Returns a scalar (mean squared flux mismatch, W²/m⁴, batch- and
    interface-averaged). NOT detached — gradients flow through to T_norm's
    source (the model), unlike pde_loss_fd's current call site.
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
    """
    Return (dx, dy, dz) in metres for a geometry's Cartesian mesh.

    dx = die_width  / nx
    dy = die_length / ny
    dz = total_height / nz   (mean cell — valid approximation for our
                               adaptive-z meshes where variation is < 2×)
    """
    nx, ny, nz = geometry.mesh_resolution
    dx = (geometry.die_width  * 1e-6) / nx    # µm → m
    dy = (geometry.die_length * 1e-6) / ny
    dz = (geometry.get_total_height() * 1e-6) / nz
    return dx, dy, dz
