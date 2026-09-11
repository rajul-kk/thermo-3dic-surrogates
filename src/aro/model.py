"""Autoregressive Operator (ARO) for 3D-IC thermal prediction."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral convolution (2D)
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """2D Fourier integral operator — truncated spectral domain."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bixy,ioxy->boxy', x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)
        out[:, :, :m1, :m2] = self._compl_mul2d(x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W))


# ---------------------------------------------------------------------------
# FiLM-conditioned 2D FNO block
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """FiLM: Feature-wise Linear Modulation — condition on a scalar vector."""

    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.film_gen = nn.Linear(cond_dim, 2 * n_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W); cond: (B, cond_dim) → modulated x."""
        gamma_beta = self.film_gen(cond)          # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class AROBlock(nn.Module):
    """Shared 2D FiLM-FNO block, weight-tied across all z-layers."""

    def __init__(
        self,
        in_ch:   int = 3,
        hidden:  int = 32,
        out_ch:  int = 1,
        modes1:  int = 16,
        modes2:  int = 16,
        n_fno_blocks: int = 4,
        cond_dim: int = 4,   # (htc_norm, t_amb_norm, tsv_frac, tim_k_norm)
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, hidden, 1)

        self.fno_blocks = nn.ModuleList([
            SpectralConv2d(hidden, hidden, modes1, modes2) for _ in range(n_fno_blocks)
        ])
        self.bypass_convs = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 1) for _ in range(n_fno_blocks)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(cond_dim, hidden) for _ in range(n_fno_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(min(8, hidden), hidden) for _ in range(n_fno_blocks)
        ])

        self.project = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, in_ch, H, W)
        cond: torch.Tensor,        # (B, cond_dim) per-sample condition
    ) -> torch.Tensor:             # (B, 1, H, W)
        x = self.lift(x)
        for spec, bypass, film, norm in zip(
            self.fno_blocks, self.bypass_convs, self.film_layers, self.norms
        ):
            h = spec(x) + bypass(x)
            h = film(h, cond)
            h = norm(h)
            x = F.gelu(x + h)
        return self.project(x)


# ---------------------------------------------------------------------------
# ARO: Autoregressive Operator
# ---------------------------------------------------------------------------

class ARO(nn.Module):
    """Autoregressive Operator: shared AROBlock applied sequentially per z-layer."""

    def __init__(
        self,
        n_layers: int = 11,
        hidden:   int = 32,
        modes1:   int = 16,
        modes2:   int = 16,
        n_fno_blocks: int = 4,
        cond_dim: int = 4,
        n_regions: int = 0,           # 0 = no region embedding; >0 adds region channel
        region_emb_dim: int = 0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_regions = n_regions

        in_ch = 3  # Q[i], T[i-1], k[i]
        if region_emb_dim > 0:
            self.region_emb = nn.Embedding(n_regions, region_emb_dim)
            in_ch += region_emb_dim
        else:
            self.region_emb = None

        self.block = AROBlock(
            in_ch=in_ch,
            hidden=hidden,
            out_ch=1,
            modes1=modes1,
            modes2=modes2,
            n_fno_blocks=n_fno_blocks,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        Q_stack:      torch.Tensor,                   # (B, n_layers, H, W)
        k_norms:      torch.Tensor,                   # (B, n_layers)
        cond:         torch.Tensor,                   # (B, cond_dim)
        region_map:   Optional[torch.Tensor] = None,  # (B, H, W) int or None
        T_gt_stack:   Optional[torch.Tensor] = None,  # (B, n_layers, H, W) for teacher forcing
        tf_ratio:     float = 1.0,                    # teacher-forcing probability (1=full, 0=none)
    ) -> torch.Tensor:                                # (B, n_layers, H, W)
        B, n_layers, H, W = Q_stack.shape
        device = Q_stack.device

        T_prev = torch.zeros(B, 1, H, W, device=device, dtype=Q_stack.dtype)
        outputs = []

        # Region embedding: (B, region_emb_dim, H, W)
        if self.region_emb is not None and region_map is not None:
            B_, H_, W_ = region_map.shape
            reg = self.region_emb(region_map.view(-1)).view(B_, H_, W_, -1)
            reg = reg.permute(0, 3, 1, 2)   # (B, emb_dim, H, W)
        else:
            reg = None

        for i in range(n_layers):
            Q_i = Q_stack[:, i:i+1, :, :]            # (B, 1, H, W)
            k_i = k_norms[:, i].view(B, 1, 1, 1).expand(B, 1, H, W)  # (B, 1, H, W)

            parts = [Q_i, T_prev, k_i]
            if reg is not None:
                parts.append(reg)

            inp = torch.cat(parts, dim=1)             # (B, in_ch, H, W)
            T_i = self.block(inp, cond)               # (B, 1, H, W)
            outputs.append(T_i)

            # Teacher forcing: use GT T[i] as input to next layer during training
            if T_gt_stack is not None and torch.rand(1).item() < tf_ratio:
                T_prev = T_gt_stack[:, i:i+1, :, :]
            else:
                T_prev = T_i.detach()

        return torch.cat(outputs, dim=1)              # (B, n_layers, H, W)

    def forward_windowed_rollout(
        self,
        Q_stack:    torch.Tensor,                   # (B, n_layers, H, W)
        k_norms:    torch.Tensor,                   # (B, n_layers)
        cond:       torch.Tensor,                   # (B, cond_dim)
        T_gt_stack: torch.Tensor,                   # (B, n_layers, H, W) ground truth
        window:     int,                             # number of consecutive self-fed layers
        region_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """RNO-style training pass (Recurrent Neural Operators, Yang et al. 2025):"""
        B, n_layers, H, W = Q_stack.shape
        device = Q_stack.device
        window = min(window, n_layers)

        start = int(torch.randint(0, n_layers - window + 1, (1,)).item())

        if start == 0:
            T_prev = torch.zeros(B, 1, H, W, device=device, dtype=Q_stack.dtype)
        else:
            # Anchor the window on ground truth at the boundary — only the
            # INTERIOR of the window is self-fed, matching RNO's "recurrent
            # training over a window" rather than unrolling from t=0 every step.
            T_prev = T_gt_stack[:, start - 1 : start, :, :]

        if self.region_emb is not None and region_map is not None:
            B_, H_, W_ = region_map.shape
            reg = self.region_emb(region_map.view(-1)).view(B_, H_, W_, -1)
            reg = reg.permute(0, 3, 1, 2)
        else:
            reg = None

        outputs = []
        for i in range(start, start + window):
            Q_i = Q_stack[:, i:i+1, :, :]
            k_i = k_norms[:, i].view(B, 1, 1, 1).expand(B, 1, H, W)

            parts = [Q_i, T_prev, k_i]
            if reg is not None:
                parts.append(reg)

            inp = torch.cat(parts, dim=1)
            T_i = self.block(inp, cond)
            outputs.append(T_i)

            # No detach here — gradients flow through the self-fed chain within
            # the window. This is the key difference from scheduled-sampling
            # teacher forcing: the loss on T[i+1] can backprop through T[i]'s
            # error, teaching the model to be robust to its own mistakes.
            T_prev = T_i

        return torch.cat(outputs, dim=1), start, window


def build_aro(
    n_layers:       int   = 11,
    hidden:         int   = 32,
    modes1:         int   = 16,
    modes2:         int   = 16,
    n_fno_blocks:   int   = 4,
    cond_dim:       int   = 4,
    n_regions:      int   = 0,
    region_emb_dim: int   = 0,
    device: Optional[torch.device] = None,
) -> ARO:
    """Construct an ARO model with sensible defaults for a T4 (16 GB)."""
    model = ARO(
        n_layers=n_layers,
        hidden=hidden,
        modes1=modes1,
        modes2=modes2,
        n_fno_blocks=n_fno_blocks,
        cond_dim=cond_dim,
        n_regions=n_regions,
        region_emb_dim=region_emb_dim,
    )
    if device is not None:
        model = model.to(device)
    return model
