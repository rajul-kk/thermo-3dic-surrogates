"""ThermoNO: a neural operator shaped by the physics of layered chip conduction (docs/report.md 9.26).
Predicts a correction to the layered spectral backbone that is exactly linear in power and nonlinear in geometry.
"""
# Components and where each comes from (docs/references.md, 2026-09-24/25):
#  - residual on the classical layered backbone (report 9.25; analytic-backbone + residual, PPDNO)
#  - exact linearity in the source, geometry-dependent operator (Neural Green's Functions, 2025)
#  - DCT-II spectral convolution: the Neumann eigenbasis for adiabatic walls (SPFNO; Hartley NO)
#  - dense learned z-mixing instead of spectral z: the discrete transfer-matrix coupling of layers
#  - per-cell log-conductivity inputs: the lateral heterogeneity the backbone averages away
#  - optional peak-weighted and energy-norm losses (two-head HBM work; DeepOHeat-v2)
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


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


class ThermoNO(nn.Module):
    """Correction field dT = G_theta[power, backbone theta]: exactly linear in its power inputs."""

    def __init__(self, grid, ch: int = 16, n_blocks: int = 3, modes=(16, 16),
                 lin_in: int = 2, geo_in: int = 4):
        super().__init__()
        nx, ny, nz = grid
        # No bias anywhere on the linear path: a bias would break dT(a*q) = a*dT(q).
        self.lift = nn.Conv3d(lin_in, ch, 1, bias=False)
        self.spec = nn.ModuleList([DCTSpectralConv(ch, nx, ny, nz, *modes) for _ in range(n_blocks)])
        self.local = nn.ModuleList([nn.Conv3d(ch, ch, 1, bias=False) for _ in range(n_blocks)])
        self.geo = GeometryEncoder(geo_in, ch, 2 * n_blocks + 1)
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
