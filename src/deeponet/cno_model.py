"""
PI-CNO-DeepONet: CNO-FNO spatial branch encoder + coordinate trunk.

Replaces the fixed 16×16 sensor MLP branch of PI-DeepONet with a full
spatial CNO-FNO encoder, eliminating information loss from point sampling.

Architecture
------------
  Branch  —  CNOBranchEncoder
    Lift (5→ch) → CNN encoder (stride-2 × 2) → AdaptiveAvgPool3d to
    fixed (LF, LF, LF) latent → FiLM-FNO blocks → global pool → Linear
    → n_basis coefficients.

    FiLM is conditioned on (htc_norm, t_amb_norm, tsv_frac, geom_desc),
    giving the branch full geometry + scenario awareness via two pathways:
      (a) spatial Q_grid input (power field structure)
      (b) FiLM scalar conditioning (geometry descriptor + BCs)

    AdaptiveAvgPool3d normalises variable grid shapes (g1: 100×100×40,
    g6: 56×168×50, etc.) to a fixed (8,8,8) latent before the spectral
    blocks, making ONE branch work across all 8 geometries.

  Trunk  —  TrunkNet (reused from model.py)
    Fourier encoding of (x_norm, y_norm, z_norm, layer_id_norm) → n_basis.
    Trunk-only autograd for PDE loss: ∂T/∂x_i = branch · ∂trunk/∂x_i.

  Output
    T(x) = branch(Q_grid) · trunk(x) + bias   shape: (B, N)

Advantages over PI-DeepONet
----------------------------
  • No sensor information loss: full 3D power field, not 16×16 samples
  • Cross-geometry: AdaptiveAvgPool3d handles any (nx, ny, nz) → one model
  • geometry4/5/6 support: lateral k variation is implicitly encoded in the
    Q_grid spatial structure + layer_id channel
  • PDE cost identical: branch coefficients are fixed per scenario, so only
    the trunk is differentiated during the physics loss
  • ~3-5× better branch expressiveness vs MLP on the same n_basis

Parameter count (ch=64, n_basis=128, n_fno_blocks=4)
------------------------------------------------------
  CNN encoder (~640k)  +  latent FNO (~1.4M)  +  FiLM (~0.2M)
  + proj head (~16k)   +  trunk (~0.5M)
  Total: ~2.8M  vs  ~0.5M for PIDeepONet (richer, justified by 275 train scenarios)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .model import TrunkNet, FourierEncoding4D, N_FREQ, GEOM_DESC_DIM
from ..fno.model import _CNOResBlock, FiLMFNOBlock, FiLMGenerator, IN_CH

# Fixed latent spatial size after adaptive pooling — same for all geometries
_LATENT_FIXED = 8
_LATENT_MODES = (4, 4, 4)

# FiLM input: 3 scenario scalars + geometry descriptor
_FILM_DIM = 3 + GEOM_DESC_DIM


class CNOBranchEncoder(nn.Module):
    """
    Cross-geometry CNO-FNO encoder: (B,5,nx,ny,nz) → (B, n_basis).

    The 5 input channels mirror CNOFNOHybrid (Q_norm, layer_id_norm,
    htc, t_amb, tsv broadcast) so the branch has the same physics signals.

    Args:
        ch:           channel width (recommend 64 for cross-geometry training)
        n_cno_layers: stride-2 downsampling stages (2 → 4× spatial reduction)
        n_fno_blocks: FiLM-FNO blocks at the fixed (8,8,8) latent
        n_basis:      branch output dimension (must match trunk)
    """

    def __init__(
        self,
        ch: int = 64,
        n_cno_layers: int = 2,
        n_fno_blocks: int = 4,
        n_basis: int = 128,
    ):
        super().__init__()
        self.ch = ch
        LF    = _LATENT_FIXED
        modes = _LATENT_MODES

        self.lift     = nn.Conv3d(IN_CH, ch, kernel_size=1)
        self.enc_res  = nn.ModuleList([_CNOResBlock(ch) for _ in range(n_cno_layers)])
        self.enc_down = nn.ModuleList([
            nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1)
            for _ in range(n_cno_layers)
        ])

        # Pool variable latent to fixed (LF, LF, LF) before spectral blocks
        self.latent_pool   = nn.AdaptiveAvgPool3d((LF, LF, LF))
        self.latent_blocks = nn.ModuleList(
            [FiLMFNOBlock(ch, modes) for _ in range(n_fno_blocks)]
        )
        # FiLM conditioned on scenario scalars + full geometry descriptor
        self.film_gen = FiLMGenerator(n_fno_blocks, ch, param_dim=_FILM_DIM)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Linear(ch, ch * 2),
            nn.GELU(),
            nn.Linear(ch * 2, n_basis),
        )
        # Small init so trunk dominates early training (standard for branch nets)
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        Q_norm: torch.Tensor,         # (B, nx, ny, nz)
        layer_id_norm: torch.Tensor,  # (B, nx, ny, nz)
        htc_norm: torch.Tensor,       # (B,)
        t_amb_norm: torch.Tensor,     # (B,)
        tsv_frac: torch.Tensor,       # (B,)
        geom_params: torch.Tensor,    # (B, GEOM_DESC_DIM)
    ) -> torch.Tensor:                # (B, n_basis)
        B, nx, ny, nz = Q_norm.shape

        def _bcast(t: torch.Tensor) -> torch.Tensor:
            v = t.view(-1) if t.dim() >= 1 else t.unsqueeze(0)
            v = v[:B] if v.shape[0] >= B else v.expand(B)
            return v.view(B, 1, 1, 1, 1).expand(B, 1, nx, ny, nz)

        x = torch.stack([
            Q_norm, layer_id_norm,
            _bcast(htc_norm).squeeze(1),
            _bcast(t_amb_norm).squeeze(1),
            _bcast(tsv_frac).squeeze(1),
        ], dim=1)   # (B, 5, nx, ny, nz)

        # FiLM params: [htc, t_amb, tsv, geom_desc]
        scl = torch.stack([
            htc_norm.view(B)   if htc_norm.numel()   >= B else htc_norm.expand(B),
            t_amb_norm.view(B) if t_amb_norm.numel() >= B else t_amb_norm.expand(B),
            tsv_frac.view(B)   if tsv_frac.numel()   >= B else tsv_frac.expand(B),
        ], dim=1).float()
        film_in = torch.cat([scl, geom_params.float()], dim=1)  # (B, _FILM_DIM)
        gamma, beta = self.film_gen(film_in)  # (B, n_fno_blocks, ch)

        # CNN encode
        x = self.lift(x)
        for res, down in zip(self.enc_res, self.enc_down):
            x = res(x)
            x = down(x)

        # Normalise to fixed latent, apply FNO blocks, global pool
        x = self.latent_pool(x)
        for i, block in enumerate(self.latent_blocks):
            x = block(x, gamma[:, i, :], beta[:, i, :])
        return self.proj(self.global_pool(x).flatten(1))  # (B, n_basis)


class PICNODeepONet(nn.Module):
    """
    PI-CNO-DeepONet: full spatial CNO-FNO branch + coordinate trunk.

    T(x) = CNOBranch(Q_grid) · Trunk(x) + bias

    Drop-in replacement for PIDeepONet — same trainer, same PDE loss,
    same trunk. Only the branch changes: from a 540D MLP to a spatial
    encoder that sees the full 3D power field.

    Args:
        n_basis:       dot-product dimension (default 128)
        ch:            branch channel width (default 64)
        n_fno_blocks:  FiLM-FNO blocks in branch latent
        n_cno_layers:  CNN encoder downsampling stages
        trunk_hidden:  trunk MLP width
        trunk_layers:  trunk MLP depth
        fourier_sigma: trunk Fourier encoding bandwidth
    """

    def __init__(
        self,
        n_basis: int = 128,
        ch: int = 64,
        n_fno_blocks: int = 4,
        n_cno_layers: int = 2,
        trunk_hidden: int = 256,
        trunk_layers: int = 4,
        fourier_sigma: float = 10.0,
    ):
        super().__init__()
        self.n_basis = n_basis
        self.branch  = CNOBranchEncoder(ch, n_cno_layers, n_fno_blocks, n_basis)
        self.trunk   = TrunkNet(n_basis, trunk_hidden, trunk_layers, N_FREQ, fourier_sigma)
        self.bias    = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        Q_norm: torch.Tensor,         # (B, nx, ny, nz)
        layer_id_norm: torch.Tensor,  # (B, nx, ny, nz)
        htc_norm: torch.Tensor,       # (B,)
        t_amb_norm: torch.Tensor,
        tsv_frac: torch.Tensor,
        geom_params: torch.Tensor,    # (B, GEOM_DESC_DIM)
        trunk_coords: torch.Tensor,   # (N, 4)
    ) -> torch.Tensor:               # (B, N)
        b = self.branch(Q_norm, layer_id_norm, htc_norm, t_amb_norm, tsv_frac, geom_params)
        t = self.trunk(trunk_coords)
        return b @ t.T + self.bias

    def encode_branch(
        self,
        Q_norm: torch.Tensor,
        layer_id_norm: torch.Tensor,
        htc_norm: torch.Tensor,
        t_amb_norm: torch.Tensor,
        tsv_frac: torch.Tensor,
        geom_params: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-compute (B, n_basis) branch coefficients — reuse across all trunk calls."""
        return self.branch(Q_norm, layer_id_norm, htc_norm, t_amb_norm, tsv_frac, geom_params)

    def encode_branch_from_item(
        self,
        item: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """Unified interface called by DeepONetTrainer. Returns (1, n_basis)."""
        return self.encode_branch(
            item['Q_grid_3d'].to(device).unsqueeze(0),
            item['layer_id_grid'].to(device).unsqueeze(0),
            item['htc_norm'].to(device).unsqueeze(0),
            item['t_amb_norm'].to(device).unsqueeze(0),
            item['tsv_frac'].to(device).unsqueeze(0),
            item['geom_desc'].to(device).unsqueeze(0),
        )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_cno_deeponet(
    n_basis: int = 128,
    ch: int = 64,
    n_fno_blocks: int = 4,
    n_cno_layers: int = 2,
    trunk_hidden: int = 256,
    trunk_layers: int = 4,
    fourier_sigma: float = 10.0,
    device: Optional[torch.device] = None,
) -> PICNODeepONet:
    model = PICNODeepONet(
        n_basis=n_basis, ch=ch, n_fno_blocks=n_fno_blocks,
        n_cno_layers=n_cno_layers, trunk_hidden=trunk_hidden,
        trunk_layers=trunk_layers, fourier_sigma=fourier_sigma,
    )
    if device is not None:
        model = model.to(device)
    return model
