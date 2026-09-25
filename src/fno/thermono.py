"""ThermoNO: a neural operator shaped by the physics of layered chip conduction (docs/report.md 9.26).
Predicts a correction to the layered spectral backbone that is exactly linear in power and nonlinear in geometry.
"""
# Components and where each comes from (docs/references.md, 2026-09-24/25):
#  - residual on the classical layered backbone (report 9.25; analytic-backbone + residual, PPDNO)
#  - exact linearity in the source, geometry-dependent operator (Neural Green's Functions, 2025)
#  - DCT-II spectral convolution: the Neumann eigenbasis for adiabatic walls (SPFNO; Hartley NO)
#  - dense learned z-mixing instead of spectral z: the discrete transfer-matrix coupling of layers
#  - optional CNO multiscale geometry encoder (+ axial attention), reused from this repo's
#    CNOFNOHybrid, only on the power-free gate path, so linearity in power is untouched
#  - per-cell log-conductivity inputs: the lateral heterogeneity the backbone averages away
#  - optional peak-weighted and energy-norm losses (two-head HBM work; DeepOHeat-v2)
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix: row p is mode p sampled at cell centres."""
    k = torch.arange(n, dtype=torch.float64)
    m = torch.cos(math.pi * (k[None, :] + 0.5) * k[:, None] / n)
    m[0] /= math.sqrt(2.0)
    return (m * math.sqrt(2.0 / n)).to(dtype)


class DCTSpectralConv(nn.Module):
    """Linear map: DCT in x and y, per-mode channel mixing, dense mixing across z, inverse DCT."""

    def __init__(self, ch: int, nx: int, ny: int, nz: int, mx: int, my: int):
        super().__init__()
        self.mx, self.my = min(mx, nx), min(my, ny)
        self.register_buffer('Dx', dct_matrix(nx)[:self.mx])      # (mx, nx)
        self.register_buffer('Dy', dct_matrix(ny)[:self.my])      # (my, ny)
        scale = 1.0 / ch
        self.w = nn.Parameter(scale * torch.randn(ch, ch, self.mx, self.my))
        self.zmix = nn.Parameter(torch.eye(nz) + 0.01 * torch.randn(nz, nz))   # transfer-like coupling

    def forward(self, x):                                        # x: (B, C, nx, ny, nz)
        xh = torch.einsum('px,bcxyz->bcpyz', self.Dx, x)
        xh = torch.einsum('qy,bcpyz->bcpqz', self.Dy, xh)
        xh = torch.einsum('bcpqz,copq->bopqz', xh, self.w)
        xh = torch.einsum('bopqz,zw->bopqw', xh, self.zmix)
        xh = torch.einsum('qy,bopqz->bopyz', self.Dy, xh)
        return torch.einsum('px,bopyz->boxyz', self.Dx, xh)


class GeometryEncoder(nn.Module):
    """Nonlinear per-cell features of the stack (never sees power), used to gate the linear path."""

    def __init__(self, in_ch: int, ch: int, n_gates: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, ch, 1), nn.GELU(),
            nn.Conv3d(ch, ch, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv3d(ch, n_gates * ch, 1))
        self.n_gates, self.ch = n_gates, ch

    def forward(self, geo):
        return torch.chunk(self.net(geo), self.n_gates, dim=1)


class _GeoResBlock(nn.Module):
    """Single-resolution residual Conv3d block (CNOFNOHybrid's _CNOResBlock, reused verbatim)."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.Conv3d(ch, ch, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv3d(ch, ch, 3, padding=1, padding_mode='replicate'))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class AxialAttention3d(nn.Module):
    """Self-attention along x, then y, then z (CNOFNOHybrid's AxialAttentionFiLMBlock, without FiLM
    -- the geometry encoder has no per-scenario scalar to condition on beyond its own channels)."""

    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        if ch % n_heads != 0:
            n_heads = 1
        self.attn_x = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.attn_y = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.attn_z = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.norm = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):                                        # (B, C, lx, ly, lz)
        B, C, lx, ly, lz = x.shape
        # x-axis: group (B, ly, lz) as batch, attend over lx
        t = x.permute(0, 3, 4, 2, 1).reshape(B * ly * lz, lx, C)
        t, _ = self.attn_x(t, t, t)
        x = x + t.reshape(B, ly, lz, lx, C).permute(0, 4, 3, 1, 2)
        # y-axis: group (B, lx, lz) as batch, attend over ly
        t = x.permute(0, 2, 4, 3, 1).reshape(B * lx * lz, ly, C)
        t, _ = self.attn_y(t, t, t)
        x = x + t.reshape(B, lx, lz, ly, C).permute(0, 4, 1, 3, 2)
        # z-axis: group (B, lx, ly) as batch, attend over lz
        t = x.permute(0, 2, 3, 4, 1).reshape(B * lx * ly, lz, C)
        t, _ = self.attn_z(t, t, t)
        x = x + t.reshape(B, lx, ly, lz, C).permute(0, 4, 1, 2, 3)
        return self.norm(x)


class CNOGeometryEncoder(nn.Module):
    """CNO-style multiscale encoder/decoder (+ optional axial attention) for the geometry gates.

    Mirrors CNOFNOHybrid's stride-2 encoder / trilinear-upsample decoder (src/fno/model.py), moved
    from a full field predictor into this repo's residual-gate role: local conv layers see the
    sharp die/underfill/TSV boundaries the plain 1x1+3x3 GeometryEncoder can blur, and attention at
    the bottleneck lets a gate depend on the whole stack, not just a 3x3x3 neighbourhood.
    """

    def __init__(self, in_ch: int, ch: int, n_gates: int, grid: Tuple[int, int, int],
                 n_layers: int = 2, use_attention: bool = False, n_heads: int = 4):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, ch, 3, padding=1, padding_mode='replicate')
        self.enc_res = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        cur = tuple(grid)
        for _ in range(n_layers):
            self.enc_res.append(_GeoResBlock(ch))
            stride = tuple(2 if d >= 4 else 1 for d in cur)
            self.enc_down.append(nn.Conv3d(ch, ch, 3, stride=stride, padding=1))
            cur = tuple((d + 2 - 3) // s + 1 for d, s in zip(cur, stride))
        self.attn = AxialAttention3d(ch, n_heads) if use_attention else nn.Identity()
        self.dec_fuse = nn.ModuleList(nn.Conv3d(2 * ch, ch, 1) for _ in range(n_layers))
        self.dec_res = nn.ModuleList(_GeoResBlock(ch) for _ in range(n_layers))
        self.head = nn.Conv3d(ch, n_gates * ch, 1)
        self.n_gates, self.ch = n_gates, ch

    def forward(self, geo):
        x = self.stem(geo)
        skips = []
        for res, down in zip(self.enc_res, self.enc_down):
            x = res(x)
            skips.append(x)
            x = down(x)
        x = self.attn(x)
        for i in reversed(range(len(self.dec_fuse))):
            x = F.interpolate(x, size=skips[i].shape[2:], mode='trilinear', align_corners=False)
            x = self.dec_fuse[i](torch.cat([x, skips[i]], dim=1))
            x = self.dec_res[i](x)
        return torch.chunk(self.head(x), self.n_gates, dim=1)


class ThermoNO(nn.Module):
    """Correction field dT = G_theta[power, backbone theta]: exactly linear in its power inputs."""

    def __init__(self, grid, ch: int = 16, n_blocks: int = 3, modes=(16, 16),
                 lin_in: int = 2, geo_in: int = 4, geo_arch: str = 'mlp', geo_attn: bool = False):
        super().__init__()
        nx, ny, nz = grid
        # No bias anywhere on the linear path: a bias would break dT(a*q) = a*dT(q).
        self.lift = nn.Conv3d(lin_in, ch, 1, bias=False)
        self.spec = nn.ModuleList([DCTSpectralConv(ch, nx, ny, nz, *modes) for _ in range(n_blocks)])
        self.local = nn.ModuleList([nn.Conv3d(ch, ch, 1, bias=False) for _ in range(n_blocks)])
        # geo_arch='cno' swaps the plain conv gate encoder for CNOFNOHybrid's multiscale
        # encoder/decoder (+ axial attention if geo_attn): sharper at chiplet/underfill edges.
        if geo_arch == 'mlp':
            self.geo = GeometryEncoder(geo_in, ch, 2 * n_blocks + 1)
        elif geo_arch == 'cno':
            self.geo = CNOGeometryEncoder(geo_in, ch, 2 * n_blocks + 1, grid, use_attention=geo_attn)
        else:
            raise ValueError(f'unknown geo_arch {geo_arch!r}; expected "mlp" or "cno"')
        self.proj = nn.Conv3d(ch, 1, 1, bias=False)

    def forward(self, lin, geo):
        """lin: (B, 2, ...) = [power, backbone theta], both linear in power; geo: (B, 4, ...) power-free."""
        gates = self.geo(geo)
        x = self.lift(lin) * gates[-1]
        for b, (spec, local) in enumerate(zip(self.spec, self.local)):
            # gated sum of two linear maps: still linear in x, nonlinear in geometry
            x = x + spec(x) * gates[2 * b] + local(x) * gates[2 * b + 1]
        return self.proj(x)[:, 0]


def peak_weighted_mse(pred, target, mask, top_frac: float = 0.01, w_top: float = 4.0):
    """Masked MSE plus extra weight on each sample's hottest cells."""
    err = (pred - target) ** 2
    base = (err * mask).sum() / mask.sum()
    loss = base
    if w_top > 0:
        flat_t = torch.where(mask > 0, target, torch.full_like(target, -1e9)).flatten(1)
        k = max(1, int(top_frac * mask[0].sum().item()))
        idx = flat_t.topk(k, dim=1).indices
        loss = loss + w_top * err.flatten(1).gather(1, idx).mean()
    return loss


def energy_norm_loss(theta_hat: torch.Tensor, A: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Discrete energy 1/2 th^T A th - q^T th, minimised by the FV solution (DeepOHeat-v2-style, label-free)."""
    th = theta_hat.reshape(-1)
    return 0.5 * torch.dot(th, torch.sparse.mm(A, th[:, None])[:, 0]) - torch.dot(q.reshape(-1), th)
