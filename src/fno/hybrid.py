"""Hybrid FNO + PINN correction model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

from .model import FNO3d, CNOFNOHybrid
from ..pinn.model import FourierFeatureEmbedding, ResBlock


# Additional input to CorrectionPINN beyond the base PINN features:
#   T_FNO_norm (1,) — interpolated normalised FNO temperature
_EXTRA_CORRECTION_FEATURES = 1


class CorrectionPINN(nn.Module):
    """Point-wise residual network that predicts δT = T_true - T_FNO."""

    def __init__(
        self,
        n_layers: int = 9,
        n_freq: int = 16,
        fourier_sigma: float = 10.0,
        layer_emb_dim: int = 8,
        hidden_dim: int = 256,
        n_res_blocks: int = 4,
    ):
        super().__init__()
        self.fourier = FourierFeatureEmbedding(n_freq, fourier_sigma)
        self.layer_emb = nn.Embedding(n_layers, layer_emb_dim)

        # 4 scenario scalars + 1 T_FNO interpolated
        input_dim = self.fourier.output_dim + layer_emb_dim + 4 + _EXTRA_CORRECTION_FEATURES

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.output_head = nn.Linear(hidden_dim, 1)

        # Initialise output head near zero so early training starts from T_FNO
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        coords: torch.Tensor,       # (N, 3) normalised [0,1]
        layer_ids: torch.Tensor,    # (N,) int
        power: torch.Tensor,        # (N, 1)
        htc_norm: torch.Tensor,     # (N, 1)
        t_amb_norm: torch.Tensor,   # (N, 1)
        tsv_frac: torch.Tensor,     # (N, 1)
        t_fno_norm: torch.Tensor,   # (N, 1) interpolated FNO prediction
    ) -> torch.Tensor:
        N = coords.shape[0]

        def _expand(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
                return t.expand(N, 1)
            return t.view(N, 1)

        fourier_feats = self.fourier(coords)         # (N, 32)
        layer_feats = self.layer_emb(layer_ids)      # (N, 8)

        x = torch.cat([
            fourier_feats,
            layer_feats,
            _expand(power),
            _expand(htc_norm),
            _expand(t_amb_norm),
            _expand(tsv_frac),
            t_fno_norm.view(N, 1),
        ], dim=-1)  # (N, 45)

        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x).squeeze(-1)   # (N,) normalised δT

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HybridModel(nn.Module):
    """CNOFNOHybrid (frozen, preferred) or FNO3d (frozen) + CorrectionPINN (trained)."""

    def __init__(
        self,
        fno: Union[FNO3d, CNOFNOHybrid],
        n_layers: int = 9,
        fourier_sigma: float = 10.0,
        hidden_dim: int = 256,
        n_res_blocks: int = 4,
    ):
        super().__init__()
        self.fno = fno
        self.correction = CorrectionPINN(
            n_layers=n_layers,
            fourier_sigma=fourier_sigma,
            hidden_dim=hidden_dim,
            n_res_blocks=n_res_blocks,
        )
        # Freeze FNO — its weights are never updated during hybrid training
        for param in self.fno.parameters():
            param.requires_grad_(False)
        self.fno.eval()

    def forward_grid(
        self,
        Q_norm: torch.Tensor,        # (B, nx, ny, nz)
        layer_id_norm: torch.Tensor, # (B, nx, ny, nz)
        htc_norm: torch.Tensor,      # (B,)
        t_amb_norm: torch.Tensor,    # (B,)
        tsv_frac: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        """Return FNO prediction on the full grid: (B, nx, ny, nz)."""
        with torch.no_grad():
            return self.fno(Q_norm, layer_id_norm, htc_norm, t_amb_norm, tsv_frac)

    def interpolate_fno(
        self,
        fno_grid: torch.Tensor,   # (1, nx, ny, nz) FNO output for one scenario
        coords_norm: torch.Tensor, # (N, 3) query points in normalised [0,1]³
    ) -> torch.Tensor:
        """Trilinear interpolation of the FNO grid at arbitrary normalised coords."""
        nx, ny, nz = fno_grid.shape[-3:]

        # grid_sample input: (1, 1, N_d, N_h, N_w) or (1, C, D, H, W)
        # We want per-point interpolation: reshape coords as single-row "grid"
        # grid_sample grid: (1, 1, 1, N, 3) with (x,y,z) order (W,H,D in torch)
        N = coords_norm.shape[0]
        # Remap [0,1] to [-1,1]
        grid_xy_z = coords_norm * 2.0 - 1.0          # (N, 3)
        # torch grid_sample uses (x, y, z) = (W, H, D) indexing
        grid = grid_xy_z[:, [0, 1, 2]].view(1, 1, 1, N, 3)  # (1,1,1,N,3)

        # fno_grid: (1, nx, ny, nz) -> add channel dim: (1, 1, nx, ny, nz)
        vol = fno_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, nx, ny, nz)
        # grid_sample: (N_batch, C, D, H, W) with grid (N_batch, D_out, H_out, W_out, 3)
        vol = fno_grid.unsqueeze(0)   # (1, 1, nx, ny, nz)  — fno_grid already (1,nx,ny,nz)
        interp = F.grid_sample(
            vol,         # (1, 1, nx, ny, nz) -- batch=1, channel=1, D=nx, H=ny, W=nz
            grid,        # (1, 1, 1, N, 3)
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # (1, 1, 1, 1, N)
        return interp.view(N)   # (N,)

    def forward_points(
        self,
        coords_norm: torch.Tensor,   # (N, 3) normalised query coords
        layer_ids: torch.Tensor,     # (N,) int
        power: torch.Tensor,         # (N,)
        htc_norm: torch.Tensor,      # scalar
        t_amb_norm: torch.Tensor,    # scalar
        tsv_frac: torch.Tensor,      # scalar
        fno_grid: torch.Tensor,      # (1, nx, ny, nz) pre-computed FNO grid output
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (T_total_norm, T_fno_interp_norm), both shape (N,)."""
        t_fno_interp = self.interpolate_fno(fno_grid, coords_norm)   # (N,)

        N = coords_norm.shape[0]

        def _expand(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
                return t.expand(N, 1)
            return t.view(N, 1)

        delta_T = self.correction(
            coords=coords_norm,
            layer_ids=layer_ids,
            power=_expand(power),
            htc_norm=_expand(htc_norm),
            t_amb_norm=_expand(t_amb_norm),
            tsv_frac=_expand(tsv_frac),
            t_fno_norm=t_fno_interp.unsqueeze(-1),
        )  # (N,)

        return t_fno_interp + delta_T, t_fno_interp

    def train(self, mode: bool = True):
        """Keep FNO in eval mode regardless of outer train/eval switch."""
        super().train(mode)
        self.fno.eval()
        return self

    @property
    def n_parameters(self) -> int:
        return self.correction.n_parameters


# ---------------------------------------------------------------------------
# Hybrid loss
# ---------------------------------------------------------------------------

def compute_hybrid_loss(
    model: HybridModel,
    fno_grid: torch.Tensor,       # (1, nx, ny, nz) FNO output (pre-computed, no grad)
    T_true_grid: torch.Tensor,    # (1, nx, ny, nz) ground truth on FNO grid
    coords_norm: torch.Tensor,    # (N, 3) query points for PINN losses
    layer_ids: torch.Tensor,      # (N,) int
    power_norm: torch.Tensor,     # (N,)
    htc_norm: torch.Tensor,       # scalar
    t_amb_norm: torch.Tensor,     # scalar
    tsv_frac: torch.Tensor,       # scalar
    T_true_norm: torch.Tensor,    # (N,) ground truth at query points
    # Physics quantities (for PDE/BC losses when weights > 0)
    layer_k: Optional[torch.Tensor] = None,   # (n_layers,) conductivity W/m·K
    si_mask: Optional[torch.Tensor] = None,   # (n_layers,) bool, Si layers
    T_min: float = 298.0,
    T_max: float = 600.0,
    htc_physical: float = 5000.0,            # W/m²·K (un-normalised)
    t_amb_K: float = 313.0,                  # ambient temp in K
    power_physical: Optional[torch.Tensor] = None,  # (N,) W/m³ for PDE
    w_correction: float = 1.0,
    w_pde: float = 0.0,
    w_bc: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute hybrid loss components."""
    T_total_norm, _ = model.forward_points(
        coords_norm=coords_norm,
        layer_ids=layer_ids,
        power=power_norm,
        htc_norm=htc_norm,
        t_amb_norm=t_amb_norm,
        tsv_frac=tsv_frac,
        fno_grid=fno_grid,
    )

    losses: Dict[str, torch.Tensor] = {}

    # Data fidelity: T_total at labelled query points
    losses['correction_data'] = F.mse_loss(T_total_norm, T_true_norm)

    total = w_correction * losses['correction_data']

    if w_pde > 0 and layer_k is not None and power_physical is not None:
        from ..pinn.physics import pde_residual
        coords_col = coords_norm.detach().requires_grad_(True)
        # Rebuild T_total with graph through coords_col
        t_fno_col = model.interpolate_fno(fno_grid, coords_col)
        N = coords_col.shape[0]

        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.expand(N, 1) if (t.dim() == 0 or t.shape[0] == 1) else t.view(N, 1)

        delta_col = model.correction(
            coords=coords_col,
            layer_ids=layer_ids,
            power=_expand(power_norm),
            htc_norm=_expand(htc_norm),
            t_amb_norm=_expand(t_amb_norm),
            tsv_frac=_expand(tsv_frac),
            t_fno_norm=(t_fno_col + delta_col.detach()).unsqueeze(-1),
        )
        T_total_col = t_fno_col + delta_col

        residual = pde_residual(
            T_norm=T_total_col,
            coords_norm=coords_col,
            layer_ids=layer_ids,
            layer_k=layer_k,
            si_layer_mask=si_mask,
            T_min=T_min,
            T_max=T_max,
            Q_physical=power_physical,
        )
        losses['pde'] = (residual ** 2).mean()
        total = total + w_pde * losses['pde']

    if w_bc > 0 and layer_k is not None:
        from ..pinn.physics import bc_residual_top
        # Assume coords_norm[:, 2] == 1.0 for top-surface points
        # Caller is responsible for passing only top-surface points here
        bc_res = bc_residual_top(
            T_norm=T_total_norm,
            coords_norm=coords_norm,
            layer_ids=layer_ids,
            layer_k=layer_k,
            T_min=T_min,
            T_max=T_max,
            htc=htc_physical,
            t_amb_K=t_amb_K,
        )
        losses['bc'] = (bc_res ** 2).mean()
        total = total + w_bc * losses['bc']

    losses['total'] = total
    return losses
