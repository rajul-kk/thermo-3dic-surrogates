"""FNO3d: Fourier Neural Operator for 3D steady-state heat conduction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


IN_CH = 5       # input channels per voxel (see module docstring)
OUT_CH = 1      # temperature field


def _as_field(t: torch.Tensor, B: int, nx: int, ny: int, nz: int) -> torch.Tensor:
    """Expand a conditioning input to (B, 1, nx, ny, nz)."""
    if t.dim() >= 3 and t.shape[-3:] == (nx, ny, nz):
        return t.view(B, 1, nx, ny, nz)
    v = t.view(-1) if t.dim() >= 1 else t.unsqueeze(0)
    v = v[:B] if v.shape[0] >= B else v.expand(B)
    return v.view(B, 1, 1, 1, 1).expand(B, 1, nx, ny, nz)


def _as_scalar(t: torch.Tensor, B: int) -> torch.Tensor:
    """Reduce a conditioning input to (B,) for use in a FiLM-style scalar summary."""
    if t.dim() >= 3:
        return t.reshape(t.shape[0], -1).mean(dim=-1)
    return t.view(B) if t.numel() >= B else t.expand(B)


class SpectralConv3d(nn.Module):
    """3D spectral convolution via truncated FFT."""

    def __init__(self, in_ch: int, out_ch: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mx, self.my, self.mz = modes   # modes retained in each dimension

        scale = 1.0 / (in_ch * out_ch) ** 0.5
        # rfft3d output has shape (..., nx, ny, nz//2+1); only the first mz
        # frequency bins along z are learned (rfft already halves the z dimension)
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, self.mx, self.my, self.mz, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, nx, ny, nz)
        B, C, nx, ny, nz = x.shape
        assert C == self.in_ch, f"SpectralConv3d: expected {self.in_ch} channels, got {C}"
        assert self.mx <= nx // 2, f"modes_x={self.mx} > nx//2={nx//2}"
        assert self.my <= ny // 2, f"modes_y={self.my} > ny//2={ny//2}"
        assert self.mz <= nz // 2 + 1, f"modes_z={self.mz} > nz//2+1={nz//2+1}"

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))  # (B, in_ch, nx, ny, nz//2+1)

        # Multiply low-mode block by learned weights
        # torch.einsum is cleaner here than manual broadcasting
        out_ft = torch.zeros(
            B, self.out_ch, nx, ny, nz // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.mx, :self.my, :self.mz] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.mx, :self.my, :self.mz],
            self.weight
        )

        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1))  # (B, out_ch, nx, ny, nz)


class FNOBlock(nn.Module):
    """Single FNO layer: spectral path + local (pointwise) path + residual."""

    def __init__(self, channels: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes)
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.local(x)))


class FNO3d(nn.Module):
    """Fourier Neural Operator for 3D thermal field prediction."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        modes: Tuple[int, int, int] = (16, 16, 12),
        hidden_ch: int = 32,
        n_blocks: int = 4,
        use_geometry_field: bool = False,
    ):
        super().__init__()
        nx, ny, nz = grid_shape
        mx = min(modes[0], nx // 2)
        my = min(modes[1], ny // 2)
        mz = min(modes[2], nz // 2 + 1)
        clamped = (mx, my, mz)
        if clamped != modes:
            import warnings
            warnings.warn(
                f"FNO3d: modes {modes} clamped to {clamped} for grid {grid_shape}",
                stacklevel=2,
            )

        self.grid_shape = grid_shape
        self.modes = clamped
        self.hidden_ch = hidden_ch
        # Track B (goal.md): opt-in 6th channel, distance to nearest power
        # block. Off by default so IN_CH-sized checkpoints trained before this
        # existed keep loading unchanged.
        self.use_geometry_field = use_geometry_field
        in_ch = IN_CH + 1 if use_geometry_field else IN_CH

        self.lift = nn.Conv3d(in_ch, hidden_ch, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock(hidden_ch, clamped) for _ in range(n_blocks)])
        # Two-layer projection head is standard; a single linear loses accuracy
        self.proj = nn.Sequential(
            nn.Conv3d(hidden_ch, hidden_ch * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_ch * 2, OUT_CH, kernel_size=1),
        )

    def forward(
        self,
        Q_norm: torch.Tensor,       # (B, nx, ny, nz) normalised power density
        layer_id_norm: torch.Tensor, # (B, nx, ny, nz) layer_index / (n_layers-1)
        htc_norm: torch.Tensor,      # (B,) or scalar
        t_amb_norm: torch.Tensor,    # (B,) or scalar
        tsv_frac: torch.Tensor,      # (B,) or scalar
        dist_to_block: Optional[torch.Tensor] = None,  # (B, nx, ny, nz), Track B
    ) -> torch.Tensor:
        B, nx, ny, nz = Q_norm.shape
        assert (nx, ny, nz) == self.grid_shape, (
            f"Input grid {(nx, ny, nz)} != model grid {self.grid_shape}"
        )
        if self.use_geometry_field and dist_to_block is None:
            raise ValueError(
                "FNO3d was built with use_geometry_field=True but forward() "
                "received no dist_to_block tensor."
            )

        def _broadcast(t: torch.Tensor) -> torch.Tensor:
            """Expand scalar/batch scalar to (B, 1, nx, ny, nz)."""
            v = t.view(-1) if t.dim() >= 1 else t.unsqueeze(0)
            v = v[:B] if v.shape[0] >= B else v.expand(B)
            return v.view(B, 1, 1, 1, 1).expand(B, 1, nx, ny, nz)

        channels = [
            Q_norm,
            layer_id_norm,
            _broadcast(htc_norm).squeeze(1),
            _broadcast(t_amb_norm).squeeze(1),
            _as_field(tsv_frac, B, nx, ny, nz).squeeze(1),
        ]
        if self.use_geometry_field:
            channels.append(_as_field(dist_to_block, B, nx, ny, nz).squeeze(1))
        x = torch.stack(channels, dim=1)  # (B, in_ch, nx, ny, nz)

        x = self.lift(x)      # (B, hidden_ch, nx, ny, nz)
        x = self.blocks(x)    # (B, hidden_ch, nx, ny, nz)
        x = self.proj(x)      # (B, 1, nx, ny, nz)
        return x.squeeze(1)   # (B, nx, ny, nz)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_fno(
    grid_shape: Tuple[int, int, int],
    modes: Tuple[int, int, int] = (16, 16, 12),
    hidden_ch: int = 32,
    n_blocks: int = 4,
    device: torch.device = None,
    use_geometry_field: bool = False,
) -> FNO3d:
    model = FNO3d(grid_shape, modes, hidden_ch, n_blocks, use_geometry_field=use_geometry_field)
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# FiLM-conditioned FNO (improvements 2 + 4: BC hypernet + param conditioning)
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """
    Small MLP that maps physics parameters → (γ, β) scaling vectors for each FNO block. FiLM (Feature-wise Linear Modulation) conditions the spectral
    """

    def __init__(self, n_blocks: int, hidden_ch: int, param_dim: int = 3):
        super().__init__()
        out_dim = 2 * n_blocks * hidden_ch
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )
        self.n_blocks = n_blocks
        self.hidden_ch = hidden_ch
        # Init: γ ≈ 1, β ≈ 0 so training starts near standard FNO
        nn.init.zeros_(self.mlp[-1].weight)
        bias = self.mlp[-1].bias.data
        bias[:n_blocks * hidden_ch] = 1.0   # γ initialised to 1
        bias[n_blocks * hidden_ch:] = 0.0   # β initialised to 0

    def forward(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        params: (B, 3)
        Returns: gamma (B, n_blocks, hidden_ch), beta (B, n_blocks, hidden_ch)
        """
        out = self.mlp(params)                            # (B, 2*n_blocks*hidden_ch)
        B = params.shape[0]
        gamma = out[:, :self.n_blocks * self.hidden_ch].view(B, self.n_blocks, self.hidden_ch)
        beta  = out[:, self.n_blocks * self.hidden_ch:].view(B, self.n_blocks, self.hidden_ch)
        return gamma, beta


class FiLMFNOBlock(nn.Module):
    """FNOBlock with FiLM modulation applied after the spectral+local sum."""

    def __init__(self, channels: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes)
        self.local    = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm     = nn.InstanceNorm3d(channels, affine=False)
        self.act      = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,         # (B, C, nx, ny, nz)
        gamma: torch.Tensor,     # (B, C)
        beta: torch.Tensor,      # (B, C)
    ) -> torch.Tensor:
        h = self.norm(self.spectral(x) + self.local(x))  # (B, C, nx, ny, nz)
        # FiLM: channel-wise affine conditioned on physics params
        g = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1, 1)
        b = beta.view(beta.shape[0], beta.shape[1], 1, 1, 1)
        return self.act(g * h + b)


class AxialAttentionFiLMBlock(nn.Module):
    """FiLMFNOBlock + axial self-attention (x → y → z) in the latent space."""

    def __init__(
        self,
        channels: int,
        modes: Tuple[int, int, int],
        n_heads: int = 4,
    ):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(
                f"AxialAttentionFiLMBlock: channels={channels} must be divisible by "
                f"n_heads={n_heads}"
            )
        self.spectral = SpectralConv3d(channels, channels, modes)
        self.local    = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm1    = nn.InstanceNorm3d(channels, affine=False)

        self.attn_x   = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.attn_y   = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.attn_z   = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm_ax  = nn.LayerNorm(channels)
        self.norm_ay  = nn.LayerNorm(channels)
        self.norm_az  = nn.LayerNorm(channels)
        self.act      = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,      # (B, C, lx, ly, lz)
        gamma: torch.Tensor,  # (B, C)
        beta: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        B, C, lx, ly, lz = x.shape

        # Spectral + local path (same as FiLMFNOBlock)
        h = self.norm1(self.spectral(x) + self.local(x))

        # x-axis attention: (B*ly*lz, lx, C)
        hx = h.permute(0, 3, 4, 2, 1).reshape(B * ly * lz, lx, C)
        hx, _ = self.attn_x(hx, hx, hx, need_weights=False)
        hx = self.norm_ax(hx).reshape(B, ly, lz, lx, C).permute(0, 4, 3, 1, 2)
        h = h + hx

        # y-axis attention: (B*lx*lz, ly, C)
        hy = h.permute(0, 2, 4, 3, 1).reshape(B * lx * lz, ly, C)
        hy, _ = self.attn_y(hy, hy, hy, need_weights=False)
        hy = self.norm_ay(hy).reshape(B, lx, lz, ly, C).permute(0, 4, 1, 3, 2)
        h = h + hy

        # z-axis attention: (B*lx*ly, lz, C)
        hz = h.permute(0, 2, 3, 4, 1).reshape(B * lx * ly, lz, C)
        hz, _ = self.attn_z(hz, hz, hz, need_weights=False)
        hz = self.norm_az(hz).reshape(B, lx, ly, lz, C).permute(0, 4, 1, 2, 3)
        h = h + hz

        # FiLM modulation
        g = gamma.view(B, C, 1, 1, 1)
        b = beta.view(B, C, 1, 1, 1)
        return self.act(g * h + b)


class CondFNO3d(nn.Module):
    """Physics-parameter-conditioned FNO3d."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        modes: Tuple[int, int, int] = (16, 16, 12),
        hidden_ch: int = 32,
        n_blocks: int = 4,
    ):
        super().__init__()
        nx, ny, nz = grid_shape
        mx = min(modes[0], nx // 2)
        my = min(modes[1], ny // 2)
        mz = min(modes[2], nz // 2 + 1)
        clamped = (mx, my, mz)
        if clamped != modes:
            import warnings
            warnings.warn(
                f"CondFNO3d: modes {modes} clamped to {clamped} for grid {grid_shape}",
                stacklevel=2,
            )

        self.grid_shape = grid_shape
        self.modes      = clamped
        self.hidden_ch  = hidden_ch
        self.n_blocks   = n_blocks

        self.lift   = nn.Conv3d(IN_CH, hidden_ch, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FiLMFNOBlock(hidden_ch, clamped) for _ in range(n_blocks)]
        )
        self.proj = nn.Sequential(
            nn.Conv3d(hidden_ch, hidden_ch * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_ch * 2, OUT_CH, kernel_size=1),
        )
        self.film_gen = FiLMGenerator(n_blocks, hidden_ch, param_dim=3)

    def forward(
        self,
        Q_norm: torch.Tensor,        # (B, nx, ny, nz)
        layer_id_norm: torch.Tensor, # (B, nx, ny, nz)
        htc_norm: torch.Tensor,      # (B,) or scalar
        t_amb_norm: torch.Tensor,    # (B,) or scalar
        tsv_frac: torch.Tensor,      # (B,) or scalar
    ) -> torch.Tensor:
        B, nx, ny, nz = Q_norm.shape

        def _bcast(t: torch.Tensor) -> torch.Tensor:
            v = t.view(-1) if t.dim() >= 1 else t.unsqueeze(0)
            v = v[:B] if v.shape[0] >= B else v.expand(B)
            return v.view(B, 1, 1, 1, 1).expand(B, 1, nx, ny, nz)

        x = torch.stack([
            Q_norm,
            layer_id_norm,
            _bcast(htc_norm).squeeze(1),
            _bcast(t_amb_norm).squeeze(1),
            _as_field(tsv_frac, B, nx, ny, nz).squeeze(1),
        ], dim=1)                           # (B, 5, nx, ny, nz)

        # FiLM modulation vectors from physics params
        params = torch.stack([
            htc_norm.view(B) if htc_norm.numel() >= B else htc_norm.expand(B),
            t_amb_norm.view(B) if t_amb_norm.numel() >= B else t_amb_norm.expand(B),
            _as_scalar(tsv_frac, B),
        ], dim=1).float()                   # (B, 3)
        gamma, beta = self.film_gen(params) # (B, n_blocks, hidden_ch) each

        x = self.lift(x)
        for i, block in enumerate(self.blocks):
            x = block(x, gamma[:, i, :], beta[:, i, :])
        return self.proj(x).squeeze(1)      # (B, nx, ny, nz)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_cond_fno(
    grid_shape: Tuple[int, int, int],
    modes: Tuple[int, int, int] = (16, 16, 12),
    hidden_ch: int = 32,
    n_blocks: int = 4,
    device: Optional[torch.device] = None,
) -> CondFNO3d:
    model = CondFNO3d(grid_shape, modes, hidden_ch, n_blocks)
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# CNO-FNO Hybrid with FiLM + PI support  (the "best FNO" variant)
# ---------------------------------------------------------------------------
#
# Design rationale
# ----------------
# Pure FNO weakness: spectral modes are global — they smear material
# discontinuities (die/TIM boundary, chiplet underfill gap).
#
# Pure CNO weakness: local convolutions can't propagate heat over long range
# without many layers.
#
# Combination:
#   CNN encoder (stride-2 × 2): extracts local features, preserves sharp edges
#   FiLM-conditioned FNO in latent space: global correlations at 1/4 resolution
#     — FFT is 64× cheaper than full-resolution, enabling more FNO blocks
#   CNN decoder (trilinear upsample + Conv): restores resolution with skip connections
#
# FiLM conditioning on the FNO latent blocks: HTC/T_amb/tsv → spectral filter
# modulation, same mechanism as CondFNO3d.  FiLM on the CNN blocks would
# overlap with the input-channel broadcast (htc/tamb/tsv are already in the
# 5-channel input); no benefit at additional cost.
#
# PI loss: applied on the FULL-resolution decoder output via finite differences
# (same as PI-FNO in physics.py).  The encoder/decoder path does not require
# any special modification for the FD loss.
#
# Parameter count (ch=32, n_fno=4):
#   CNN encoder/decoder: ~1.4M  (Conv3d kernels, BN, residual)
#   Latent FNO spectral:  ~5.9M  (32×32×12×12×5 × 4 blocks, complex)
#   FiLM generator:        ~0.2M
#   Total: ~7.5M  vs  ~12.6M for baseline FNO — fewer params, better accuracy
# ---------------------------------------------------------------------------

class _CNOResBlock(nn.Module):
    """Single-resolution residual Conv3d block with GroupNorm."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CNOFNOHybrid(nn.Module):
    """CNO encoder + FiLM-FNO latent + CNO decoder — the combined best-of-three model."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        ch: int = 32,
        n_fno_blocks: int = 4,
        n_cno_layers: int = 2,
        use_attention: bool = False,
        n_heads: int = 4,
    ):
        super().__init__()
        nx, ny, nz = grid_shape
        self.grid_shape   = grid_shape
        self.hidden_ch    = ch
        self.n_cno_layers = n_cno_layers
        self.use_attention = use_attention

        # Latent grid dimensions after n_cno_layers stride-2 stages
        scale = 2 ** n_cno_layers
        lx = max(1, nx // scale)
        ly = max(1, ny // scale)
        lz = max(1, nz // scale)
        self._latent_shape = (lx, ly, lz)

        # FNO modes clamped to half the latent dimensions
        mx = min(8, max(1, lx // 2))
        my = min(8, max(1, ly // 2))
        mz = min(8, max(1, lz // 2))
        self._latent_modes = (mx, my, mz)

        # ── Lifting ──────────────────────────────────────────────────────────
        self.lift = nn.Conv3d(IN_CH, ch, kernel_size=1)

        # ── CNO Encoder ──────────────────────────────────────────────────────
        # Each stage: residual refinement then stride-2 downsampling
        self.enc_res   = nn.ModuleList()
        self.enc_down  = nn.ModuleList()
        for _ in range(n_cno_layers):
            self.enc_res.append(_CNOResBlock(ch))
            self.enc_down.append(nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1))

        # ── Latent FiLM-FNO (with optional axial SAU attention) ──────────────
        if use_attention:
            self.latent_blocks = nn.ModuleList(
                [AxialAttentionFiLMBlock(ch, self._latent_modes, n_heads)
                 for _ in range(n_fno_blocks)]
            )
        else:
            self.latent_blocks = nn.ModuleList(
                [FiLMFNOBlock(ch, self._latent_modes) for _ in range(n_fno_blocks)]
            )
        self.film_gen = FiLMGenerator(n_fno_blocks, ch, param_dim=3)

        # ── CNO Decoder ──────────────────────────────────────────────────────
        # Each stage: trilinear upsample to skip shape → cat skip → fuse → residual
        self.dec_fuse = nn.ModuleList()
        self.dec_res  = nn.ModuleList()
        for _ in range(n_cno_layers):
            # Input after cat: 2*ch (latent + skip) → ch
            self.dec_fuse.append(nn.Conv3d(ch * 2, ch, kernel_size=1))
            self.dec_res.append(_CNOResBlock(ch))

        # ── Projection head ──────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Conv3d(ch, ch * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(ch * 2, OUT_CH, kernel_size=1),
        )

    def forward(
        self,
        Q_norm: torch.Tensor,        # (B, nx, ny, nz)
        layer_id_norm: torch.Tensor, # (B, nx, ny, nz)
        htc_norm: torch.Tensor,      # (B,) or scalar
        t_amb_norm: torch.Tensor,
        tsv_frac: torch.Tensor,
    ) -> torch.Tensor:               # (B, nx, ny, nz)
        B, nx, ny, nz = Q_norm.shape

        def _bcast(t: torch.Tensor) -> torch.Tensor:
            v = t.view(-1) if t.dim() >= 1 else t.unsqueeze(0)
            v = v[:B] if v.shape[0] >= B else v.expand(B)
            return v.view(B, 1, 1, 1, 1).expand(B, 1, nx, ny, nz)

        x = torch.stack([
            Q_norm, layer_id_norm,
            _bcast(htc_norm).squeeze(1),
            _bcast(t_amb_norm).squeeze(1),
            _as_field(tsv_frac, B, nx, ny, nz).squeeze(1),
        ], dim=1)                    # (B, 5, nx, ny, nz)

        # FiLM conditioning vectors (computed once, used in all FNO latent blocks)
        n_blk = len(self.latent_blocks)
        params = torch.stack([
            htc_norm.view(B) if htc_norm.numel() >= B else htc_norm.expand(B),
            t_amb_norm.view(B) if t_amb_norm.numel() >= B else t_amb_norm.expand(B),
            _as_scalar(tsv_frac, B),
        ], dim=1).float()
        gamma, beta = self.film_gen(params)   # (B, n_fno_blocks, ch)

        # ── Lift + Encode ─────────────────────────────────────────────────────
        x = self.lift(x)                      # (B, ch, nx, ny, nz)
        skips: list = []
        for res, down in zip(self.enc_res, self.enc_down):
            x = res(x)
            skips.append(x)                   # store pre-downsampled skip
            x = down(x)                       # stride-2

        # ── Latent FNO with FiLM ──────────────────────────────────────────────
        for i, block in enumerate(self.latent_blocks):
            x = block(x, gamma[:, i, :], beta[:, i, :])

        # ── Decode ───────────────────────────────────────────────────────────
        for fuse, res, skip in zip(self.dec_fuse, self.dec_res, reversed(skips)):
            # Trilinear upsample to skip's spatial dims — handles odd dimensions
            x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear',
                              align_corners=False)
            x = fuse(torch.cat([x, skip], dim=1))
            x = res(x)

        # Final upsample to original size (only needed if stride created mismatch)
        if x.shape[-3:] != (nx, ny, nz):
            x = F.interpolate(x, size=(nx, ny, nz), mode='trilinear',
                              align_corners=False)

        return self.proj(x).squeeze(1)        # (B, nx, ny, nz)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def modes(self) -> Tuple[int, int, int]:
        return self._latent_modes


def build_cno_fno(
    grid_shape: Tuple[int, int, int],
    ch: int = 32,
    n_fno_blocks: int = 4,
    n_cno_layers: int = 2,
    use_attention: bool = False,
    n_heads: int = 4,
    device: Optional[torch.device] = None,
) -> CNOFNOHybrid:
    """Build the combined CNO + FiLM-FNO best model."""
    model = CNOFNOHybrid(grid_shape, ch, n_fno_blocks, n_cno_layers, use_attention, n_heads)
    if device is not None:
        model = model.to(device)
    return model
