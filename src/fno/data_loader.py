"""
FNO data loader: reshapes flat .npz arrays into 3D grid tensors.

The .npz files store coords/temp/power as flat (N,) or (N,3) arrays produced
by the mesh generator with indexing='ij' and ravel() in C order. Reshaping to
(nx, ny, nz) is therefore safe as long as the mesh resolution matches what's
stored in the file metadata — which it always does if the file was produced by
the standard export pipeline.

Each dataset item is a dict so the trainer can access fields by name without
positional index bugs. No custom collation required; torch's default collate
handles dicts of same-shape tensors correctly.

Multi-geometry training
-----------------------
The eight benchmark geometries produce five incompatible grid shapes:

    (100, 100,  6)  geometry1, geometry3
    ( 80,  80, 10)  geometry2a/b/c
    (100,  56,  6)  geometry4
    (100,  56, 11)  geometry5
    ( 56, 168, 11)  geometry6

FNO's spectral convolution takes an FFT over the spatial dims, so a single model
cannot span these natively. Two options: (a) one dataset and training loop per
grid shape, or (b) resample everything onto a common grid. BOTH are supported.

Pass `target_grid` to enable (b). Every file is then built at its own native
resolution and trilinearly resampled onto the shared grid, so one FNO can train
across all eight geometries.

Resampling alone would be lossy in a way that matters: an 8x8 mm die and a
42x14 mm die resampled to the same array are indistinguishable, and the operator
would be asked to learn contradictory mappings. Each item therefore also carries
`geom_extent_norm` — the physical (width, length, height) of the package — so the
model conditions on real spatial scale rather than array indices. That is what
keeps each geometry's specific physics learnable after resampling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..pinn.data_loader import NormStats

_log = logging.getLogger(__name__)


def _resample(grid: torch.Tensor, target: Tuple[int, int, int],
              mode: str) -> torch.Tensor:
    """
    Resample a (nx, ny, nz) grid onto `target`.

    `mode` is 'trilinear' for continuous fields (power, temperature) or 'nearest'
    for categorical ones (layer identity).
    """
    x = grid[None, None]                                   # (1,1,nx,ny,nz)
    kw = {'align_corners': False} if mode != 'nearest' else {}
    return torch.nn.functional.interpolate(x, size=tuple(target), mode=mode, **kw)[0, 0]


class FNODataset(Dataset):
    """
    Dataset of 3D field tensors for FNO training.

    Each item:
        Q_norm:         (nx, ny, nz) float32 — normalised power density
        layer_id_norm:  (nx, ny, nz) float32 — layer_index / (n_layers-1)
        T_norm:         (nx, ny, nz) float32 — normalised temperature (target)
        htc_norm:       scalar float
        t_amb_norm:     scalar float
        tsv_frac:       scalar float
        name:           str scenario identifier

    Data is kept on CPU; move to device in training loop to avoid pinning issues
    on Windows where pin_memory has known reliability problems with large tensors.
    """

    # Normalisation constants for the physical-extent conditioning channel. Fixed
    # rather than data-derived so a model trained on one subset stays comparable to
    # one trained on another. Covers 8x8 mm up to 42x14 mm packages.
    _EXTENT_XY_REF_UM = 50000.0
    _EXTENT_Z_REF_UM = 10000.0

    def __init__(
        self,
        npz_files: List[Path],
        norm_stats: NormStats,
        expected_grid: Tuple[int, int, int],   # (nx, ny, nz)
        target_grid: Tuple[int, int, int] = None,
    ):
        """
        Args:
            expected_grid: Native grid used when `target_grid` is None. All files
                must match it, and non-matching files are skipped.
            target_grid: If given, each file is built at its OWN native resolution
                (read from its metadata) and resampled onto this shared grid,
                enabling one FNO across geometries with different mesh shapes.
        """
        self.norm_stats = norm_stats
        self.expected_grid = expected_grid
        self.target_grid = target_grid
        self.grid_shape = target_grid or expected_grid
        self.items: List[Dict] = []

        for path in npz_files:
            data = np.load(path, allow_pickle=True)
            meta = dict(data['metadata'][0])
            n_points = data['coords'].shape[0]
            n_layers = int(meta.get('num_layers', 1))

            if target_grid is not None:
                # Native resolution for THIS file. 3D-ICE emits one value per stack
                # ELEMENT per (x,y) cell, and sub-layer discretisation makes the
                # element count exceed the geometry's layer count (geometry1: 10
                # elements from 6 layers). Deriving nz from n_layers therefore
                # broke once subdivision landed, so take it from the point count,
                # which is true regardless of how the stack was discretised.
                mesh = meta.get('mesh_resolution', expected_grid)
                nx, ny = int(mesh[0]), int(mesh[1])
                if nx * ny > 0 and n_points % (nx * ny) == 0:
                    nz = n_points // (nx * ny)
                else:
                    nz = int(mesh[2])
            else:
                nx, ny, nz = expected_grid

            n_expected_full = nx * ny * nz
            n_expected_slice = n_layers * nx * ny

            # TSV area fraction per point, in [0,1]-ish range (raw phi / 0.10 cap,
            # matching src/pinn/data_loader.py's _TSV_MAX). Absent in files
            # exported before this field existed -- defaults to all-zero, which
            # is also correct for the majority of geometries that carry no TSVs.
            tsv_raw = (data['tsv_frac'].astype(np.float32) if 'tsv_frac' in data.files
                      else np.zeros(n_points, dtype=np.float32))
            tsv_raw = tsv_raw / 0.10

            if n_points == n_expected_full:
                # Full mesh layout (mock simulator)
                layer_ids = data['layer'].reshape(nx, ny, nz).astype(np.float32)
                Q_norm = norm_stats.norm_power(data['power']).reshape(nx, ny, nz)
                T_norm = norm_stats.norm_temp(data['temp']).reshape(nx, ny, nz)
                TSV_norm = tsv_raw.reshape(nx, ny, nz)

            elif n_points == n_expected_slice:
                # Per-layer-slice layout (3D-ICE real data):
                # data is (n_layers, nx, ny) ordered by layer index.
                # Broadcast each layer's 2D map across its z-range in the full grid.
                layer_flat = data['layer'].astype(np.int32)  # (n_layers*nx*ny,)
                T_flat     = norm_stats.norm_temp(data['temp'].astype(np.float32))
                Q_flat     = norm_stats.norm_power(data['power'].astype(np.float32))
                TSV_flat   = tsv_raw

                # Build per-z-index arrays using the layer index of each z-cell.
                # z-cell i belongs to the layer whose z-range contains z-coords[i].
                # Simplest approach: assign each z-slice to the nearest data layer.
                layer_grid = np.full((nx, ny, nz), 0, dtype=np.float32)
                T_grid     = np.zeros((nx, ny, nz), dtype=np.float32)
                Q_grid     = np.zeros((nx, ny, nz), dtype=np.float32)
                TSV_grid   = np.zeros((nx, ny, nz), dtype=np.float32)

                # Map: for each unique layer_id, find its (nx,ny) slice in the flat data
                unique_layers = np.unique(layer_flat)
                # z-fraction that the centre of each nz bin falls in
                z_fracs = (np.arange(nz) + 0.5) / nz  # (nz,)

                # Get layer z-ranges from coords to assign z-bins
                z_coords = data['coords'][:, 2].astype(np.float32)  # µm
                z_max = z_coords.max() if z_coords.max() > 0 else 1.0

                for lid in unique_layers:
                    mask = layer_flat == lid
                    if not mask.any():
                        continue
                    z_mid = z_coords[mask].mean() / z_max   # fractional centre of layer

                    # Find z-bins closest to this layer's z-centre
                    # Each z-bin in [0,nz) gets assigned to its nearest layer
                    # We use argmin of |z_frac - z_mid| per bin (but here we only
                    # need "which z-bins belong to this layer" — approximate by
                    # splitting evenly at midpoints between consecutive layer centres)
                    # For simplicity: assign a z-bin to layer lid if that layer's
                    # z_mid is the nearest among all unique layers.
                    # Build a full reassignment below after collecting all z_mids.
                    pass

                # Vectorised: assign each z-bin to the layer with nearest z-centre
                layer_z_mids = {}
                for lid in unique_layers:
                    mask = layer_flat == lid
                    layer_z_mids[lid] = z_coords[mask].mean() / z_max

                z_bin_to_layer = np.zeros(nz, dtype=np.int32)
                for iz in range(nz):
                    z_bin_to_layer[iz] = min(
                        unique_layers,
                        key=lambda lid: abs(z_fracs[iz] - layer_z_mids[lid])
                    )

                # Fill the 3D grids by broadcasting each layer's 2D xy map
                for lid in unique_layers:
                    mask_flat = layer_flat == lid
                    # Reshape this layer's slice to (nx, ny) — order matches C ravel
                    T_slice   = T_flat[mask_flat].reshape(nx, ny)
                    Q_slice   = Q_flat[mask_flat].reshape(nx, ny)
                    TSV_slice = TSV_flat[mask_flat].reshape(nx, ny)
                    z_bins  = np.where(z_bin_to_layer == lid)[0]
                    for iz in z_bins:
                        T_grid[:, :, iz] = T_slice
                        Q_grid[:, :, iz] = Q_slice
                        TSV_grid[:, :, iz] = TSV_slice
                        layer_grid[:, :, iz] = float(lid)

                layer_ids = layer_grid
                Q_norm = Q_grid
                T_norm = T_grid
                TSV_norm = TSV_grid

            else:
                _log.warning(
                    "Skipping %s: expected %d (full) or %d (per-layer-slice) pts, got %d",
                    path.name, n_expected_full, n_expected_slice, n_points
                )
                data.close()
                continue

            layer_id_norm = layer_ids / max(n_layers - 1, 1)
            htc = float(meta.get('htc', 5000.0))
            t_amb_c = float(meta.get('t_ambient_celsius', 40.0))

            Q_t   = torch.from_numpy(Q_norm.astype(np.float32))
            L_t   = torch.from_numpy(layer_id_norm.astype(np.float32))
            T_t   = torch.from_numpy(T_norm.astype(np.float32))
            TSV_t = torch.from_numpy(TSV_norm.astype(np.float32))

            if target_grid is not None and (nx, ny, nz) != tuple(target_grid):
                # Trilinear for the continuous fields; nearest for layer identity,
                # which is categorical -- interpolating it would invent materials
                # that do not exist between a silicon die and a copper spreader.
                Q_t   = _resample(Q_t, target_grid, 'trilinear')
                T_t   = _resample(T_t, target_grid, 'trilinear')
                L_t   = _resample(L_t, target_grid, 'nearest')
                TSV_t = _resample(TSV_t, target_grid, 'trilinear')

            self.items.append({
                'Q_norm': Q_t,
                'layer_id_norm': L_t,
                'T_norm': T_t,
                'geom_extent_norm': torch.tensor([
                    float(meta.get('die_width_um', 0.0)) / self._EXTENT_XY_REF_UM,
                    float(meta.get('die_length_um', 0.0)) / self._EXTENT_XY_REF_UM,
                    float(meta.get('total_height_um', 0.0)) / self._EXTENT_Z_REF_UM,
                ], dtype=torch.float32),
                'geometry': meta.get('geometry', 'unknown'),
                'htc_norm': torch.tensor(norm_stats.norm_htc(htc), dtype=torch.float32),
                't_amb_norm': torch.tensor(norm_stats.norm_t_amb(t_amb_c), dtype=torch.float32),
                # A real per-point field (nx,ny,nz), not the scenario-mean scalar this
                # used to be -- the model can now actually condition on where the TSV
                # density is high vs low, not just how much there is on average.
                'tsv_frac': TSV_t,
                'name': meta.get('scenario_name', path.stem),
            })
            data.close()

        if not self.items:
            raise ValueError(
                f"FNODataset: no valid files loaded from {len(npz_files)} candidates "
                f"for grid {self.grid_shape}. Check grid shape matches mesh_resolution, "
                f"or pass target_grid= to resample mixed geometries onto a common grid."
            )

        geoms = sorted({it['geometry'] for it in self.items})
        _log.info("FNODataset: loaded %d scenarios, grid=%s, geometries=%s%s",
                  len(self.items), self.grid_shape, geoms,
                  " (resampled)" if target_grid is not None else "")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]

    def to_device(self, device: torch.device) -> 'FNODataset':
        for item in self.items:
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    item[k] = v.to(device)
        return self


def predict_to_flat(
    model: 'FNO3d',
    item: Dict,
    device: torch.device,
    norm_stats: NormStats,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single item, return flat (N,) arrays in Kelvin.

    Useful for evaluation code that works with flat coordinate arrays.
    """
    model.eval()
    with torch.no_grad():
        T_hat = model(
            item['Q_norm'].unsqueeze(0).to(device),
            item['layer_id_norm'].unsqueeze(0).to(device),
            item['htc_norm'].to(device),
            item['t_amb_norm'].to(device),
            item['tsv_frac'].unsqueeze(0).to(device),
        ).squeeze(0)  # (nx, ny, nz)

    T_range = norm_stats.T_max - norm_stats.T_min
    T_pred_K = T_hat.cpu().numpy().ravel() * T_range + norm_stats.T_min
    T_true_K = item['T_norm'].cpu().numpy().ravel() * T_range + norm_stats.T_min
    return T_pred_K, T_true_K
