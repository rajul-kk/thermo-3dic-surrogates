"""Multi-geometry dataset for PI-DeepONet training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..pinn.data_loader import NormStats
from .model import geometry_descriptor, extract_sensors, BRANCH_DIM, GEOM_LAYERS_MAX, GEOM_DESC_DIM

_log = logging.getLogger(__name__)


class MultiGeomDataset(Dataset):
    """Dataset covering multiple geometries for cross-geometry DeepONet training."""

    def __init__(
        self,
        npz_files: List[Path],
        geometries: Dict[str, object],          # {geom_name: Geometry}
        norm_stats: NormStats,
    ):
        self.items: List[Dict] = []

        # Pre-compute per-geometry descriptors (constant across scenarios)
        geom_descs = {
            name: geometry_descriptor(g) for name, g in geometries.items()
        }

        for path in npz_files:
            data = np.load(path, allow_pickle=True)
            meta = dict(data['metadata'][0])
            geom_name = meta.get('geometry_name', '') or _infer_geom(path.stem)

            if geom_name not in geometries:
                _log.warning("Skipping %s: geometry '%s' not in provided dict", path.name, geom_name)
                data.close()
                continue

            geometry = geometries[geom_name]

            # --- Flat coordinate arrays ---
            coords = data['coords'].astype(np.float32)           # (N, 3) physical µm
            layer_ids = data['layer'].astype(np.int32)           # (N,)
            T_flat  = data['temp'].astype(np.float32)            # (N,) K
            Q_flat  = data['power'].astype(np.float32)           # (N,) W/m³

            N = coords.shape[0]

            # Normalise coords to [0,1]
            Lx = float(geometry.die_width)
            Ly = float(geometry.die_length)
            Lz = float(geometry.get_total_height())
            n_layers = len(geometry.layers)

            coords_norm = np.stack([
                coords[:, 0] / Lx,
                coords[:, 1] / Ly,
                coords[:, 2] / Lz,
                layer_ids.astype(np.float32) / max(n_layers - 1, 1),
            ], axis=1)                                            # (N, 4)

            T_norm = norm_stats.norm_temp(T_flat)                # (N,)
            Q_norm = norm_stats.norm_power(Q_flat)               # (N,)

            # --- Branch input ---
            htc    = float(meta.get('htc', 5000.0))
            t_amb  = float(meta.get('t_ambient_celsius', 40.0))
            # Actual per-scenario mean of the simulated TSV field where available
            # (varies scenario-to-scenario), falling back to the geometry-constant
            # metadata scalar for files predating the field export.
            tsv_f  = (float(data['tsv_frac'].mean())
                     if 'tsv_frac' in data.files and data['tsv_frac'].size
                     else float(meta.get('tsv_density', 0.0)))

            # Power sensor extraction: need Q on the FNO grid (nx,ny,nz)
            nx, ny, nz = geometry.mesh_resolution
            try:
                Q_grid = Q_norm.reshape(nx, ny, nz)
            except ValueError:
                # Grid size mismatch (e.g. adaptive z gave different count)
                # Approximate: use zeros as sensors (scenario still in dataset)
                Q_grid = np.zeros((nx, ny, nz), dtype=np.float32)
                _log.debug("Sensor extraction: reshape mismatch for %s, using zeros", path.name)

            Q_grid_t = torch.from_numpy(Q_grid)
            sensors = extract_sensors(
                Q_grid_t, geometry, device=torch.device('cpu')
            )  # (512,)

            scenario_scalars = torch.tensor(
                [norm_stats.norm_htc(htc),
                 norm_stats.norm_t_amb(t_amb),
                 tsv_f],
                dtype=torch.float32,
            )  # (3,)

            geom_desc = geom_descs[geom_name]  # (19,)

            branch_input = torch.cat([sensors, scenario_scalars, geom_desc])  # (BRANCH_DIM,)
            assert branch_input.shape[0] == BRANCH_DIM, \
                f"branch_input dim {branch_input.shape[0]} != {BRANCH_DIM}"

            # 3D Q grid and layer-id grid — used by PICNODeepONet branch encoder
            Q_grid_3d      = _build_q_grid_3d(
                data['power'].astype(np.float32),
                data['layer'].astype(np.int32),
                data['coords'].astype(np.float32),
                geometry, norm_stats,
            )
            layer_id_grid = _build_layer_id_grid(geometry)

            self.items.append({
                # ── PIDeepONet (MLP branch) ──────────────────────────────────
                'branch_input': branch_input,
                # ── PICNODeepONet (CNO-FNO branch) ──────────────────────────
                'Q_grid_3d':      torch.from_numpy(Q_grid_3d),      # (nx, ny, nz)
                'layer_id_grid':  torch.from_numpy(layer_id_grid),   # (nx, ny, nz)
                'geom_desc':      geom_desc,                          # (GEOM_DESC_DIM,)
                'htc_norm':       scenario_scalars[0],
                't_amb_norm':     scenario_scalars[1],
                'tsv_frac':       scenario_scalars[2],
                # ── Shared ───────────────────────────────────────────────────
                'coords_norm':    torch.from_numpy(coords_norm),
                'T_norm':         torch.from_numpy(T_norm),
                'power_norm':     torch.from_numpy(Q_norm),
                'geom_name':      geom_name,
                'scenario_name':  meta.get('scenario_name', path.stem),
                'n_layers':       n_layers,
            })
            data.close()

        if not self.items:
            raise ValueError(
                f"MultiGeomDataset: no valid items loaded from {len(npz_files)} files. "
                "Check geometry names match the provided geometries dict."
            )
        _log.info(
            "MultiGeomDataset: %d scenarios across %d geometries",
            len(self.items),
            len({it['geom_name'] for it in self.items}),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def _build_q_grid_3d(
    Q_flat: np.ndarray,
    layer_flat: np.ndarray,
    coords: np.ndarray,
    geometry,
    norm_stats: 'NormStats',
) -> np.ndarray:
    """Reconstruct a (nx, ny, nz) normalised Q grid from flat NPZ arrays."""
    nx, ny, nz = geometry.mesh_resolution
    n_layers = len(geometry.layers)
    N = len(Q_flat)
    Q_norm = norm_stats.norm_power(Q_flat.astype(np.float32))

    if N == nx * ny * nz:
        return Q_norm.reshape(nx, ny, nz)

    if N == n_layers * nx * ny:
        z_total = geometry.get_total_height()
        z_fracs = (np.arange(nz) + 0.5) / nz
        unique_layers = np.unique(layer_flat.astype(np.int32))
        z_coords = coords[:, 2].astype(np.float32)

        layer_z_mids = {
            int(lid): z_coords[layer_flat == lid].mean() / z_total
            for lid in unique_layers
        }
        z_bin_to_layer = np.array([
            min(unique_layers, key=lambda lid: abs(z_fracs[iz] - layer_z_mids[int(lid)]))
            for iz in range(nz)
        ], dtype=np.int32)

        Q_grid = np.zeros((nx, ny, nz), dtype=np.float32)
        for lid in unique_layers:
            mask = layer_flat == lid
            Q_slice = Q_norm[mask].reshape(nx, ny)
            for iz in np.where(z_bin_to_layer == lid)[0]:
                Q_grid[:, :, iz] = Q_slice
        return Q_grid

    # Fallback: can't reconstruct
    return np.zeros((nx, ny, nz), dtype=np.float32)


def _build_layer_id_grid(geometry) -> np.ndarray:
    """Build (nx, ny, nz) float32 layer_id_norm grid from geometry layer z-ranges."""
    nx, ny, nz = geometry.mesh_resolution
    n_layers = len(geometry.layers)
    z_total = geometry.get_total_height()
    z_fracs = (np.arange(nz) + 0.5) / nz
    grid = np.zeros((nx, ny, nz), dtype=np.float32)
    for i, layer in enumerate(geometry.layers):
        z_lo = layer.z_bottom / z_total
        z_hi = layer.z_top   / z_total
        iz_mask = (z_fracs >= z_lo) & (z_fracs < z_hi)
        grid[:, :, iz_mask] = float(i) / max(n_layers - 1, 1)
    return grid


def _infer_geom(stem: str) -> str:
    """Infer geometry name from NPZ filename stem."""
    for g in ['geometry6', 'geometry5', 'geometry4', 'geometry2a',
              'geometry3', 'geometry1']:
        if stem.startswith(g):
            return g
    return ''
