"""
PI-DeepONet: Physics-Informed Deep Operator Network.

Architecture
------------
T(x) = Σ_i  branch_i(u)  ×  trunk_i(x)   +  bias

  Branch network  — encodes the input *function* (power distribution + BCs
                    + geometry descriptor) into N_basis coefficients.

  Trunk network   — evaluates N_basis spatial basis functions at any query
                    point (x_norm, y_norm, z_norm, layer_id_norm).

  Bias            — learnable scalar added after the dot product.

The trunk takes CONTINUOUS coordinates, so a single trained model can
evaluate temperatures at arbitrary (x,y,z) across all 5 uniform-stack
geometries (geometry1/2a/2b/2c/3) without retraining.  This is the key
architectural advantage over FNO (which requires separate models per grid).
geometry4/5 (2.5D chiplet assemblies) are excluded: their per-layer thermal
conductivity varies laterally by chiplet region — a signal absent from the
trunk's (x,y,z,layer_id) coordinate space. FNO per-geometry covers those.

Branch input layout (BRANCH_DIM = 534)
--------------------------------------
  [0:512]    Power sensor values: 2 active layers × 16×16 bilinear
             downsampling of the Q_norm grid, zero-padded to 512 if only
             1 active layer (geometry1/3/4).
  [512:515]  Scenario scalars: htc_norm, t_amb_norm, tsv_frac
  [515:534]  Geometry descriptor (19 features):
               - 8 × layer_thickness_norm  (µm / max_thickness)
               - 8 × layer_k_norm          (W/m·K / 400)
               - underfill_k_norm           (k / 400; 0 for non-2p5d)
               - die_width_norm             (µm / 30000)
               - die_length_norm            (µm / 30000)

Trunk input (4 features, Fourier-encoded)
-----------------------------------------
  (x_norm, y_norm, z_norm, layer_id_norm)  ∈ [0, 1]
  Fourier encoding: 2 × N_freq × 4 = 128 features (N_freq=16, sigma=10)

Physics loss
------------
  ∂T/∂x, ∂T/∂y, ∂T/∂z are computed via autograd through the TRUNK only
  (branch coefficients are fixed per scenario).  With N_basis=128 and a
  2-layer trunk MLP, differentiating the trunk is O(N_col × trunk_cost)
  — far cheaper than differentiating through a full PINN.

  PDE:  ∇·(k ∇T) + Q = 0   →  L_pde = mean[(∇·(k∇T) + Q)²]
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ACTIVE_LAYERS = 2       # geometry2/5 have 2; geometry1/3/4 have 1
SENSOR_GRID       = 16      # 16×16 bilinear downsample per active layer
SENSOR_DIM        = MAX_ACTIVE_LAYERS * SENSOR_GRID * SENSOR_GRID   # 512
SCENARIO_DIM      = 3       # htc_norm, t_amb_norm, tsv_frac
GEOM_LAYERS_MAX   = 11      # geometry5/6 have 11 layers (max across all 8)
GEOM_DESC_DIM     = GEOM_LAYERS_MAX * 2 + 3   # 22 + 3 = 25
BRANCH_DIM        = SENSOR_DIM + SCENARIO_DIM + GEOM_DESC_DIM        # 540

TRUNK_COORD_DIM   = 4       # (x, y, z, layer_id)  all normalised [0,1]
N_FREQ            = 16      # Fourier feature frequencies per coordinate
FOURIER_DIM       = 2 * N_FREQ * TRUNK_COORD_DIM   # 128


# ---------------------------------------------------------------------------
# Geometry descriptor helper
# ---------------------------------------------------------------------------

def geometry_descriptor(geometry) -> torch.Tensor:
    """
    Build a fixed-size (GEOM_DESC_DIM,) float32 tensor from a Geometry object.

    Padding to GEOM_LAYERS_MAX layers with zeros for shorter stacks.
    Normalisation keeps all features in ~[0, 1].
    """
    max_thick = max(l.thickness for l in geometry.layers)  # µm

    thick_norm = []
    k_norm = []
    for layer in geometry.layers:
        thick_norm.append(layer.thickness / max(max_thick, 1.0))
        k_norm.append(layer.k_thermal / 400.0)           # Cu k = 400 W/m·K

    n = len(geometry.layers)
    # Pad to GEOM_LAYERS_MAX
    thick_norm += [0.0] * (GEOM_LAYERS_MAX - n)
    k_norm     += [0.0] * (GEOM_LAYERS_MAX - n)

    underfill_k = getattr(geometry, 'underfill_k', 0.0) / 400.0
    die_w = geometry.die_width  / 30000.0
    die_l = geometry.die_length / 30000.0

    desc = thick_norm + k_norm + [underfill_k, die_w, die_l]
    return torch.tensor(desc, dtype=torch.float32)   # (19,)


def extract_sensors(
    Q_norm_grid: torch.Tensor,          # (nx, ny, nz) or (B, nx, ny, nz)
    geometry,
    device: torch.device,
) -> torch.Tensor:
    """
    Bilinear-downsample each active-layer power field to 16×16, concatenate,
    zero-pad to SENSOR_DIM=512.  Returns (SENSOR_DIM,) or (B, SENSOR_DIM).
    """
    batched = Q_norm_grid.dim() == 4
    if not batched:
        Q_norm_grid = Q_norm_grid.unsqueeze(0)   # (1, nx, ny, nz)

    B, nx, ny, nz = Q_norm_grid.shape

    active_layer_indices = [
        i for i, l in enumerate(geometry.layers) if l.is_active
    ]

    patches = []
    for layer_i in active_layer_indices[:MAX_ACTIVE_LAYERS]:
        # Take mean along z within this layer's z-indices
        # Find z-indices for this layer from its z-range
        layer = geometry.layers[layer_i]
        total_h = geometry.get_total_height()
        z_lo = layer.z_bottom / total_h
        z_hi = layer.z_top   / total_h
        z_lo_idx = max(0,    int(z_lo * nz))
        z_hi_idx = min(nz-1, int(z_hi * nz) + 1)
        # Average Q over the layer's z-extent
        q_slice = Q_norm_grid[:, :, :, z_lo_idx:z_hi_idx].mean(dim=-1)  # (B, nx, ny)
        # Bilinear downsample to (SENSOR_GRID, SENSOR_GRID)
        q_down = F.interpolate(
            q_slice.unsqueeze(1),                   # (B, 1, nx, ny)
            size=(SENSOR_GRID, SENSOR_GRID),
            mode='bilinear', align_corners=False,
        ).squeeze(1)                                 # (B, 16, 16)
        patches.append(q_down.flatten(1))            # (B, 256)

    # Zero-pad if fewer than MAX_ACTIVE_LAYERS active layers
    while len(patches) < MAX_ACTIVE_LAYERS:
        patches.append(torch.zeros(B, SENSOR_GRID**2, device=device))

    sensors = torch.cat(patches, dim=1)              # (B, 512)
    if not batched:
        sensors = sensors.squeeze(0)                 # (512,)
    return sensors


# ---------------------------------------------------------------------------
# Branch network
# ---------------------------------------------------------------------------

class BranchNet(nn.Module):
    """
    MLP that encodes (power sensors + scenario BCs + geometry descriptor)
    into N_basis coefficient scalars.

    input_dim:  BRANCH_DIM = 534
    output_dim: N_basis
    """

    def __init__(self, n_basis: int = 128, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(BRANCH_DIM, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, n_basis))
        self.net = nn.Sequential(*layers)

        # Initialise last layer small so early training is stable
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, BRANCH_DIM)  →  (B, N_basis)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Trunk network (with Fourier encoding)
# ---------------------------------------------------------------------------

class FourierEncoding4D(nn.Module):
    """
    Random Fourier feature encoding for 4D trunk input (x, y, z, layer_id).

    B matrix: (4, N_freq) sampled from N(0, σ²) once and fixed.
    Output dim: 2 * N_freq * 4 = 128 for N_freq=16.
    """

    def __init__(self, n_freq: int = 16, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(TRUNK_COORD_DIM, n_freq) * sigma
        self.register_buffer('B', B)
        self.output_dim = 2 * n_freq * TRUNK_COORD_DIM

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 4)  →  (N, 128)"""
        # For each coordinate dimension i, project and Fourier-encode independently.
        # coords[:, i:i+1] : (N, 1)   self.B[i:i+1, :] : (1, n_freq)
        # product: (N, n_freq) — separate frequency bank per coordinate.
        parts = []
        for i in range(TRUNK_COORD_DIM):
            p = 2 * math.pi * coords[:, i:i+1] * self.B[i:i+1, :]  # (N, n_freq)
            parts.append(torch.cos(p))
            parts.append(torch.sin(p))
        return torch.cat(parts, dim=-1)   # (N, 2*n_freq*TRUNK_COORD_DIM)


class TrunkNet(nn.Module):
    """
    MLP that maps (x_norm, y_norm, z_norm, layer_id_norm) to N_basis spatial
    basis functions.

    Fourier encoding of the 4 coordinates gives the trunk good spectral
    properties for representing smooth thermal fields.

    Inputs flow through the trunk with requires_grad=True so autograd can
    compute ∂trunk/∂x for the PDE physics loss.
    """

    def __init__(self, n_basis: int = 128, hidden_dim: int = 256, n_layers: int = 4,
                 n_freq: int = 16, sigma: float = 10.0):
        super().__init__()
        self.fourier = FourierEncoding4D(n_freq, sigma)
        in_dim = self.fourier.output_dim + TRUNK_COORD_DIM   # 128 + 4 = 132

        layers: List[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),   # Tanh in trunk gives smoother basis functions than GELU
        ]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, n_basis))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 4)  →  (N, N_basis)"""
        fourier_feats = self.fourier(coords)                  # (N, 128)
        x = torch.cat([coords, fourier_feats], dim=-1)        # (N, 132)
        return self.net(x)                                     # (N, N_basis)


# ---------------------------------------------------------------------------
# PI-DeepONet
# ---------------------------------------------------------------------------

class PIDeepONet(nn.Module):
    """
    Physics-Informed Deep Operator Network.

    T(x) = branch(u) · trunk(x)  +  bias

    Args:
        n_basis:       Number of basis functions (depth of dot product)
        branch_hidden: Width of branch MLP
        trunk_hidden:  Width of trunk MLP
        branch_layers: Depth of branch MLP
        trunk_layers:  Depth of trunk MLP
        fourier_sigma: Bandwidth for trunk Fourier encoding
    """

    def __init__(
        self,
        n_basis: int = 128,
        branch_hidden: int = 256,
        trunk_hidden: int = 256,
        branch_layers: int = 4,
        trunk_layers: int = 4,
        fourier_sigma: float = 10.0,
    ):
        super().__init__()
        self.n_basis = n_basis
        self.branch = BranchNet(n_basis, branch_hidden, branch_layers)
        self.trunk  = TrunkNet(n_basis, trunk_hidden, trunk_layers,
                               n_freq=N_FREQ, sigma=fourier_sigma)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_input: torch.Tensor,   # (B, BRANCH_DIM)
        trunk_coords: torch.Tensor,   # (N, 4)  — query points for ONE scenario
    ) -> torch.Tensor:
        """
        Evaluate T at N query points for a BATCH of B scenarios.

        Returns (B, N) — temperature at all query points for all scenarios.
        Each scenario gets the same query locations but different branch coefficients.
        """
        b = self.branch(branch_input)      # (B, n_basis)
        t = self.trunk(trunk_coords)       # (N, n_basis)
        # Outer product: (B, N) = (B, n_basis) @ (n_basis, N)
        return b @ t.T + self.bias         # (B, N)

    def forward_single(
        self,
        branch_input: torch.Tensor,   # (BRANCH_DIM,) single scenario
        trunk_coords: torch.Tensor,   # (N, 4)
    ) -> torch.Tensor:
        """Evaluate T for a single scenario. Returns (N,)."""
        b = self.branch(branch_input.unsqueeze(0))  # (1, n_basis)
        t = self.trunk(trunk_coords)                # (N, n_basis)
        return (b @ t.T).squeeze(0) + self.bias     # (N,)

    def encode_branch_from_item(
        self,
        item: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Unified branch encoding interface used by DeepONetTrainer.
        Returns (1, n_basis) branch coefficients for a single dataset item.
        Trainer calls this on any DeepONet variant — subclasses override.
        """
        return self.branch(item['branch_input'].to(device).unsqueeze(0))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------

def build_deeponet(
    n_basis: int = 128,
    branch_hidden: int = 256,
    trunk_hidden: int = 256,
    branch_layers: int = 4,
    trunk_layers: int = 4,
    fourier_sigma: float = 10.0,
    device: Optional[torch.device] = None,
) -> PIDeepONet:
    model = PIDeepONet(
        n_basis=n_basis,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
        branch_layers=branch_layers,
        trunk_layers=trunk_layers,
        fourier_sigma=fourier_sigma,
    )
    if device is not None:
        model = model.to(device)
    return model
