"""FourierPINN architecture for 3D-IC thermal surrogate modelling."""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier feature encoding for spatial coordinates."""

    def __init__(self, n_freq: int = 16, sigma: Union[float, Tuple] = 10.0):
        super().__init__()
        if isinstance(sigma, (int, float)):
            s = torch.tensor([sigma, sigma, sigma], dtype=torch.float32)
        else:
            s = torch.tensor(list(sigma), dtype=torch.float32)
        B = torch.randn(3, n_freq) * s.unsqueeze(1)   # (3, n_freq)
        self.register_buffer('B', B)
        self.output_dim = 2 * n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3) normalised coordinates
        proj = 2 * math.pi * (x @ self.B)   # (N, n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*n_freq)


class ResBlock(nn.Module):
    """Residual block: LayerNorm -> Linear -> SiLU -> Dropout -> LayerNorm -> Linear + skip."""

    def __init__(self, dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FourierPINN(nn.Module):
    """Physics-Informed Neural Network for 3D-IC thermal prediction."""

    def __init__(
        self,
        n_layers: int = 11,                 # geometry5/6 have 11 layers; always pass explicitly
        n_freq: int = 16,                   # Fourier frequencies (output: 2*n_freq = 32)
        fourier_sigma: Union[float, Tuple] = 10.0,  # float or (sx,sy,sz) tuple
        layer_emb_dim: int = 8,
        hidden_dim: int = 256,
        n_res_blocks: int = 6,
        dropout_p: float = 0.1,
        n_regions: int = 1,                 # 1=disabled; 3 for chiplet (underfill/chipA/chipB)
        region_emb_dim: int = 0,            # 0=disabled; 4 for chiplet geometries
        tim_k_input: bool = False,          # True for g5/g6 TIM k-sweep scenarios
        hard_adiabatic: bool = True,        # cosine fold → exact dT/dn=0 at lateral walls
    ):
        super().__init__()
        self.hard_adiabatic = hard_adiabatic
        self.tim_k_input = tim_k_input

        self.fourier = FourierFeatureEmbedding(n_freq, fourier_sigma)
        self.layer_emb = nn.Embedding(n_layers, layer_emb_dim)

        self.region_emb = (
            nn.Embedding(n_regions, region_emb_dim) if region_emb_dim > 0 else None
        )

        # 4 base scalars: power, htc_norm, t_amb_norm, tsv_frac; +1 if tim_k_input
        n_scalars = 5 if tim_k_input else 4
        region_dim = region_emb_dim if self.region_emb is not None else 0
        input_dim = self.fourier.output_dim + layer_emb_dim + region_dim + n_scalars

        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout_p) for _ in range(n_res_blocks)]
        )
        self.output_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _hard_adiabatic_transform(coords: torch.Tensor) -> torch.Tensor:
        """Cosine coordinate fold enforcing dT/dx=dT/dy=0 at x={0,1} and y={0,1}."""
        xc = 0.5 * (1.0 - torch.cos(math.pi * coords[:, 0]))
        yc = 0.5 * (1.0 - torch.cos(math.pi * coords[:, 1]))
        return torch.stack([xc, yc, coords[:, 2]], dim=1)

    def forward(
        self,
        coords: torch.Tensor,                       # (N, 3) normalised
        layer_ids: torch.Tensor,                    # (N,) int
        power: torch.Tensor,                        # (N,) or (N, 1)
        htc_norm: torch.Tensor,
        t_amb_norm: torch.Tensor,
        tsv_frac: torch.Tensor,
        region_ids: Optional[torch.Tensor] = None,  # (N,) int; None → zeros
        tim_k_norm: Optional[torch.Tensor] = None,  # scalar or (N,); None → 0.05
    ) -> torch.Tensor:
        if self.hard_adiabatic:
            coords = self._hard_adiabatic_transform(coords)

        fourier_feats = self.fourier(coords)        # (N, 32)
        layer_feats   = self.layer_emb(layer_ids)   # (N, 8)

        N = coords.shape[0]

        def _expand(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
                return t.expand(N, 1)
            return t.view(N, 1)

        parts = [fourier_feats, layer_feats]

        if self.region_emb is not None:
            if region_ids is None:
                region_ids = torch.zeros(N, dtype=torch.long, device=coords.device)
            parts.append(self.region_emb(region_ids))   # (N, region_emb_dim)

        parts.extend([_expand(power), _expand(htc_norm), _expand(t_amb_norm), _expand(tsv_frac)])

        if self.tim_k_input:
            if tim_k_norm is None:
                tim_k_norm = torch.tensor(0.05, device=coords.device, dtype=coords.dtype)
            parts.append(_expand(tim_k_norm))

        x = torch.cat(parts, dim=-1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x).squeeze(-1)   # (N,)

    def forward_batched_scenarios(
        self,
        coords: torch.Tensor,          # (N, 3)  shared across all scenarios
        layer_ids: torch.Tensor,       # (N,)    shared
        power: torch.Tensor,           # (S, N)  per-scenario power
        htc_norm: torch.Tensor,        # (S,)
        t_amb_norm: torch.Tensor,      # (S,)
        tsv_frac: torch.Tensor,        # (S,)
        region_ids: Optional[torch.Tensor] = None,   # (N,) shared across S; None → zeros
        tim_k_norms: Optional[torch.Tensor] = None,  # (S,) per-scenario; None → 0.05
    ) -> torch.Tensor:                 # (S, N)
        """Process S scenarios in a single forward pass."""
        if self.hard_adiabatic:
            coords = self._hard_adiabatic_transform(coords)

        S = power.shape[0]
        N = coords.shape[0]

        fourier_feats = self.fourier(coords)                            # (N, 32)
        layer_feats   = self.layer_emb(layer_ids)                      # (N, 8)
        fourier_exp   = fourier_feats.unsqueeze(0).expand(S, -1, -1)  # (S, N, 32)
        layer_exp     = layer_feats.unsqueeze(0).expand(S, -1, -1)    # (S, N, 8)

        parts = [fourier_exp, layer_exp]

        if self.region_emb is not None:
            if region_ids is None:
                region_ids = torch.zeros(N, dtype=torch.long, device=coords.device)
            reg_exp = self.region_emb(region_ids).unsqueeze(0).expand(S, -1, -1)
            parts.append(reg_exp)                                       # (S, N, region_emb_dim)

        p  = power.unsqueeze(-1)                              # (S, N, 1)
        h  = htc_norm.view(S, 1, 1).expand(S, N, 1)
        ta = t_amb_norm.view(S, 1, 1).expand(S, N, 1)
        tv = tsv_frac.view(S, 1, 1).expand(S, N, 1)
        parts.extend([p, h, ta, tv])

        if self.tim_k_input:
            if tim_k_norms is None:
                tim_k_norms = torch.full((S,), 0.05, device=coords.device, dtype=coords.dtype)
            tk = tim_k_norms.view(S, 1, 1).expand(S, N, 1)
            parts.append(tk)

        x = torch.cat(parts, dim=-1)    # (S, N, input_dim)
        x = x.reshape(S * N, -1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_head(x).squeeze(-1)
        return x.view(S, N)


def build_model(
    n_layers: int = 11,
    fourier_sigma: Union[float, Tuple] = 10.0,
    hidden_dim: int = 256,
    n_res_blocks: int = 6,
    dropout_p: float = 0.1,
    n_regions: int = 1,
    region_emb_dim: int = 0,
    tim_k_input: bool = False,
    hard_adiabatic: bool = True,
    device: Optional[torch.device] = None,
) -> FourierPINN:
    """Construct a FourierPINN."""
    model = FourierPINN(
        n_layers=n_layers,
        fourier_sigma=fourier_sigma,
        hidden_dim=hidden_dim,
        n_res_blocks=n_res_blocks,
        dropout_p=dropout_p,
        n_regions=n_regions,
        region_emb_dim=region_emb_dim,
        tim_k_input=tim_k_input,
        hard_adiabatic=hard_adiabatic,
    )
    if device is not None:
        model = model.to(device)
    return model
