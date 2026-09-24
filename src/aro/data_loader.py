"""Dataset for Autoregressive Operator (ARO) training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..pinn.data_loader import NormStats

_log = logging.getLogger(__name__)

# Conductivity normalisation reference (Si at 300 K ≈ 148 W/m·K)
_K_REF = 200.0


def _k_norm(k_thermal: float) -> float:
    return k_thermal / _K_REF


class ARODataset(Dataset):
    """Dataset of layered (Q, T, k) stacks for ARO training."""

    def __init__(
        self,
        hf_files:   List[Path],
        geometries: Dict[str, object],   # {name: Geometry}
        norm_stats: NormStats,
        lf_files:   Optional[List[Path]] = None,
    ):
        self.items: List[Dict] = []

        for path in hf_files:
            item = self._load_npz(path, geometries, norm_stats, is_lf=False)
            if item is not None:
                self.items.append(item)

        if lf_files:
            for path in lf_files:
                item = self._load_npz(path, geometries, norm_stats, is_lf=True)
                if item is not None:
                    self.items.append(item)

        if not self.items:
            raise ValueError(
                "ARODataset: no valid items loaded from "
                f"{len(hf_files)} HF + {len(lf_files or [])} LF files"
            )
        hf_n = sum(1 for it in self.items if not it['is_lf'])
        lf_n = sum(1 for it in self.items if it['is_lf'])
        _log.info("ARODataset: %d HF + %d LF = %d total items", hf_n, lf_n, len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]

    # ------------------------------------------------------------------

    def _load_npz(
        self,
        path: Path,
        geometries: Dict[str, object],
        norm_stats: NormStats,
        is_lf: bool,
    ) -> Optional[Dict]:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as exc:
            _log.warning("ARODataset: cannot load %s: %s", path.name, exc)
            return None

        meta = dict(data['metadata'][0])
        geom_name = meta.get('geometry_name', '') or _infer_geom(path.stem)
        if geom_name not in geometries:
            _log.warning("ARODataset: skipping %s (geometry '%s' unknown)", path.name, geom_name)
            data.close()
            return None

        geometry  = geometries[geom_name]
        nx, ny, _ = geometry.mesh_resolution
        n_layers  = len(geometry.layers)
        Lx = float(geometry.die_width)
        Ly = float(geometry.die_length)

        coords    = data['coords'].astype(np.float32)       # (N, 3)
        layer_ids = data['layer'].astype(np.int32)          # (N,)
        T_flat    = data['temp'].astype(np.float32)         # (N,) K
        Q_flat    = data['power'].astype(np.float32)        # (N,) W/m³

        T_norm = norm_stats.norm_temp(T_flat)
        Q_norm = norm_stats.norm_power(Q_flat)

        # One slice per geometry layer, averaged over its z sub-layers. The previous
        # per-layer reshape skipped every layer that sub-layer splitting gave more than
        # one z-node, leaving zeros (docs/report.md 9.24).
        from ..core.mesh import points_to_grid
        Q_grid, T_grid, L_grid = points_to_grid(coords, Q_norm, T_norm, layer_ids)
        if Q_grid.shape[:2] != (nx, ny):
            raise ValueError(f'{path.name}: grid {Q_grid.shape[:2]} != mesh {(nx, ny)}')
        z_layer = L_grid[0, 0, :]
        Q_stack = np.zeros((n_layers, nx, ny), dtype=np.float32)
        T_stack = np.zeros((n_layers, nx, ny), dtype=np.float32)
        for lid in range(n_layers):
            ks = np.nonzero(z_layer == lid)[0]
            if ks.size:
                Q_stack[lid] = Q_grid[:, :, ks].mean(axis=2)
                T_stack[lid] = T_grid[:, :, ks].mean(axis=2)

        # Per-layer normalised k
        k_norms = np.array(
            [_k_norm(layer.k_thermal) for layer in geometry.layers],
            dtype=np.float32,
        )

        # BCs
        htc       = float(meta.get('htc', 5000.0))
        t_amb     = float(meta.get('t_ambient_celsius', 40.0))
        # Actual per-scenario mean of the simulated TSV field where available,
        # falling back to the geometry-constant metadata scalar otherwise.
        tsv_f     = (float(data['tsv_frac'].mean())
                    if 'tsv_frac' in data.files and data['tsv_frac'].size
                    else float(meta.get('tsv_density', 0.0)))
        tim_k_raw = float(meta.get('tim_top_k', 4.0))
        cond = np.array([
            norm_stats.norm_htc(htc),
            norm_stats.norm_t_amb(t_amb),
            tsv_f,
            tim_k_raw / 80.0,
        ], dtype=np.float32)

        data.close()

        return {
            'Q_stack':      torch.from_numpy(Q_stack),      # (n_layers, H, W)
            'T_stack':      torch.from_numpy(T_stack),      # (n_layers, H, W)
            'k_norms':      torch.from_numpy(k_norms),      # (n_layers,)
            'cond':         torch.from_numpy(cond),         # (4,)
            'is_lf':        is_lf,
            'geom_name':    geom_name,
            'scenario_name': meta.get('scenario_name', path.stem),
            'n_layers':     n_layers,
        }


def _infer_geom(stem: str) -> str:
    for g in ['geometry6', 'geometry5', 'geometry4', 'geometry2a',
              'geometry3', 'geometry1']:
        if stem.startswith(g) or g in stem:
            return g
    return ''
