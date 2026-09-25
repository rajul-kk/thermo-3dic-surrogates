"""Layer-Transfer FNO (LT-FNO): FNO with spectral-in-z replaced by learned per-mode layer coupling.
One-component change to FNO3d (docs/report.md 9.27): FFT in x and y as before; z is coupled densely.
"""
# Why z is the component to change. FNO3d takes a truncated FFT along z, which assumes a uniform,
# periodic axis. A layered stack is neither: 3D-ICE's z-nodes vary in spacing by up to ~90x, the
# bottom face convects and the top is adiabatic, and the grid has only 10-15 nodes, so FNO3d keeps
# 4-6 z-modes. For laterally uniform layers the exact solution couples each lateral Fourier/cosine
# mode through a small dense system in z (the transfer-matrix method; src/hybrid/layered_backbone.py
# solves it). LT-FNO learns that coupling: for every retained lateral mode (p, q), a real nz x nz
# matrix mixes the layers, at full z-rank, with no periodicity and no dependence on node spacing.
# Everything else (lifting, local path, norm, activation, projection, inputs) is FNO3d unchanged,
# so FNO3d vs LTFNO3d isolates this one design choice, the way FNO vs WHNO isolates the basis.
# Prior art: none found for a learned per-mode layer-coupling operator inside an FNO (one search
# pass, docs/references.md 2026-09-24/25). The transfer-matrix method itself is classical.
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .model import FNO3d


class LayerTransferConv3d(nn.Module):
    """Truncated FFT in x and y, channel mixing per lateral mode, dense learned z-coupling per mode."""

    def __init__(self, in_ch: int, out_ch: int, modes_xy: Tuple[int, int], nz: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.mx, self.my = modes_xy
        self.nz = nz
        scale = 1.0 / (in_ch * out_ch) ** 0.5
        self.weight = nn.Parameter(scale * torch.rand(in_ch, out_ch, self.mx, self.my, dtype=torch.cfloat))
        # identity plus a small perturbation: starts as independent per-layer 2D spectral convs
        self.zcouple = nn.Parameter(torch.eye(nz).expand(self.mx, self.my, nz, nz).clone()
                                    + 0.01 * torch.randn(self.mx, self.my, nz, nz))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, nx, ny, nz = x.shape
        assert C == self.in_ch and nz == self.nz, (C, nz)
        assert self.mx <= nx // 2 and self.my <= ny // 2 + 1, (self.mx, self.my, nx, ny)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2))                      # (B, C, nx, ny//2+1, nz)
        y = torch.einsum('bipqz,iopq->bopqz', x_ft[:, :, :self.mx, :self.my, :], self.weight)
        y = torch.einsum('bopqz,pqzw->bopqw', y, self.zcouple.to(y.dtype))
        out_ft = torch.zeros(B, self.out_ch, nx, ny // 2 + 1, nz, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.mx, :self.my, :] = y
        return torch.fft.irfftn(out_ft, s=(nx, ny), dim=(-3, -2))


class LTFNOBlock(nn.Module):
    """FNOBlock with the spectral path swapped for LayerTransferConv3d; local path, norm, act unchanged."""

    def __init__(self, channels: int, modes_xy: Tuple[int, int], nz: int):
        super().__init__()
        self.spectral = LayerTransferConv3d(channels, channels, modes_xy, nz)
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.local(x)))


class LTFNO3d(FNO3d):
    """FNO3d with every block's spectral convolution replaced by LayerTransferConv3d."""

    def __init__(self, grid_shape, modes=(16, 16, 12), hidden_ch: int = 32, n_blocks: int = 4,
                 use_geometry_field: bool = False):
        super().__init__(grid_shape, modes, hidden_ch, n_blocks, use_geometry_field)
        nx, ny, nz = grid_shape
        mxy = (self.modes[0], min(modes[1], ny // 2 + 1))
        self.blocks = nn.Sequential(*[LTFNOBlock(hidden_ch, mxy, nz) for _ in range(n_blocks)])


def build_lt_fno(grid_shape, modes=(16, 16, 12), hidden_ch: int = 32, n_blocks: int = 4,
                 device: torch.device = None, use_geometry_field: bool = False) -> LTFNO3d:
    model = LTFNO3d(grid_shape, modes, hidden_ch, n_blocks, use_geometry_field=use_geometry_field)
    return model.to(device) if device is not None else model
