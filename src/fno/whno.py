"""Walsh-Hadamard Neural Operator (WHNO) for 3D thermal field prediction."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .model import IN_CH, OUT_CH


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform primitives
# ---------------------------------------------------------------------------

def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised Fast Walsh-Hadamard Transform along the LAST dimension."""
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n & (n - 1) == 0, f"fwht_last_dim: length {n} is not a power of 2"

    xf = x.reshape(-1, n).contiguous()
    M = xf.shape[0]
    h = 1
    while h < n:
        xf = xf.view(M, n // (2 * h), 2, h)
        a = xf[:, :, 0, :]
        b = xf[:, :, 1, :]
        xf = torch.stack((a + b, a - b), dim=2).view(M, n)
        h *= 2
    return xf.view(orig_shape)


def fwht_along_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply fwht_last_dim along an arbitrary dimension."""
    x = x.transpose(dim, -1)
    x = fwht_last_dim(x)
    return x.transpose(dim, -1)


def hadamard_to_sequency_perm(n: int, device=None) -> torch.Tensor:
    """
    Permutation mapping natural (Hadamard-ordered) FWHT coefficient index -> sequency-ordered index (ordered by number of sign changes in the
    """
    eye = torch.eye(n, device=device)
    H = fwht_last_dim(eye)   # row i of H == i-th natural-order Walsh function
    sign_changes = (H[:, 1:] * H[:, :-1] < 0).sum(dim=1)   # (n,) count of sign flips
    # stable sort so ties (rare, only at exact equal counts) keep natural order
    perm = torch.argsort(sign_changes, stable=True)
    return perm


# ---------------------------------------------------------------------------
# Walsh-Hadamard spectral convolution (3D)
# ---------------------------------------------------------------------------

class WalshConv3d(nn.Module):
    """3D "spectral" convolution via truncated Walsh-Hadamard transform."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        grid_shape: Tuple[int, int, int],   # (nx, ny, nz) — UNPADDED input size
        modes: Tuple[int, int, int],        # (mx, my, mz) low-sequency modes kept
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.grid_shape = grid_shape

        self.pad_shape = tuple(next_pow2(d) for d in grid_shape)
        self.mx, self.my, self.mz = (
            min(modes[0], self.pad_shape[0]),
            min(modes[1], self.pad_shape[1]),
            min(modes[2], self.pad_shape[2]),
        )

        for axis, n in enumerate(self.pad_shape):
            perm = hadamard_to_sequency_perm(n)
            inv_perm = torch.argsort(perm)
            self.register_buffer(f'_perm_{axis}', perm, persistent=False)
            self.register_buffer(f'_inv_perm_{axis}', inv_perm, persistent=False)

        scale = 1.0 / (in_ch * out_ch) ** 0.5
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, self.mx, self.my, self.mz)
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = self.grid_shape
        px, py, pz = self.pad_shape
        return torch.nn.functional.pad(x, (0, pz - nz, 0, py - ny, 0, px - nx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, nx, ny, nz)
        B = x.shape[0]
        x = self._pad(x)   # (B, in_ch, px, py, pz)

        # Forward WHT along each spatial axis (dims 2,3,4), then permute to
        # sequency order along each axis.
        for axis, dim in enumerate((2, 3, 4)):
            x = fwht_along_dim(x, dim)
            perm = getattr(self, f'_perm_{axis}')
            x = x.index_select(dim, perm)

        # Truncate to low-sequency block, mix channels, zero-pad back
        x_trunc = x[:, :, :self.mx, :self.my, :self.mz]
        out_trunc = torch.einsum('bixyz,ioxyz->boxyz', x_trunc, self.weight)

        px, py, pz = self.pad_shape
        out = torch.zeros(B, self.out_ch, px, py, pz, dtype=x.dtype, device=x.device)
        out[:, :, :self.mx, :self.my, :self.mz] = out_trunc

        # Inverse permute (sequency -> natural) then inverse WHT (self-inverse
        # up to 1/n scale) along each axis, in reverse order.
        for axis, dim in reversed(list(enumerate((2, 3, 4)))):
            inv_perm = getattr(self, f'_inv_perm_{axis}')
            out = out.index_select(dim, inv_perm)
            out = fwht_along_dim(out, dim) / out.shape[dim]

        nx, ny, nz = self.grid_shape
        return out[:, :, :nx, :ny, :nz]


class WHNOBlock(nn.Module):
    """Walsh-Hadamard analogue of FNOBlock: spectral(Walsh) + local + norm + GELU."""

    def __init__(self, channels: int, grid_shape: Tuple[int, int, int], modes: Tuple[int, int, int]):
        super().__init__()
        self.spectral = WalshConv3d(channels, channels, grid_shape, modes)
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.local(x)))


class WHNO3d(nn.Module):
    """Walsh-Hadamard Neural Operator — drop-in alternative to FNO3d."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        modes: Tuple[int, int, int] = (16, 16, 12),
        hidden_ch: int = 32,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.modes = modes
        self.hidden_ch = hidden_ch

        self.lift = nn.Conv3d(IN_CH, hidden_ch, kernel_size=1)
        self.blocks = nn.Sequential(
            *[WHNOBlock(hidden_ch, grid_shape, modes) for _ in range(n_blocks)]
        )
        self.proj = nn.Sequential(
            nn.Conv3d(hidden_ch, hidden_ch * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_ch * 2, OUT_CH, kernel_size=1),
        )

    def forward(
        self,
        Q_norm: torch.Tensor,
        layer_id_norm: torch.Tensor,
        htc_norm: torch.Tensor,
        t_amb_norm: torch.Tensor,
        tsv_frac: torch.Tensor,
    ) -> torch.Tensor:
        B, nx, ny, nz = Q_norm.shape
        assert (nx, ny, nz) == self.grid_shape, (
            f"Input grid {(nx, ny, nz)} != model grid {self.grid_shape}"
        )

        def _broadcast(t: torch.Tensor) -> torch.Tensor:
            v = t.view(-1) if t.dim() >= 1 else t.unsqueeze(0)
            v = v[:B] if v.shape[0] >= B else v.expand(B)
            return v.view(B, 1, 1, 1, 1).expand(B, 1, nx, ny, nz)

        x = torch.stack([
            Q_norm,
            layer_id_norm,
            _broadcast(htc_norm).squeeze(1),
            _broadcast(t_amb_norm).squeeze(1),
            _broadcast(tsv_frac).squeeze(1),
        ], dim=1)

        x = self.lift(x)
        x = self.blocks(x)
        x = self.proj(x)
        return x.squeeze(1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_whno(
    grid_shape: Tuple[int, int, int],
    modes: Tuple[int, int, int] = (16, 16, 12),
    hidden_ch: int = 32,
    n_blocks: int = 4,
    device: torch.device = None,
) -> WHNO3d:
    model = WHNO3d(grid_shape, modes, hidden_ch, n_blocks)
    if device is not None:
        model = model.to(device)
    return model
