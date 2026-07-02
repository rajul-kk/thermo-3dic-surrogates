"""
Data loading and normalization for PINN training.

Handles:
  - Loading .npz files from the 3D-ICE dataset
  - Computing and storing global normalization statistics
  - Batching by scenario (not by individual point)
  - Sampling random collocation points for PDE/BC evaluation

Normalization convention:
  coords:    [0,1] per axis (divided by domain extent)
  temp:      [0,1] using global T_min / T_max across the dataset
  power:     zero-mean unit-variance using dataset mean/std
  htc:       [0,1] using known physical range [500, 10000] W/m²·K
  t_amb:     [0,1] using known physical range [25, 85] °C
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.geometry import Geometry

_log = logging.getLogger(__name__)

# Physical parameter ranges covering the full scenario suite
_HTC_MIN, _HTC_MAX = 500.0, 10000.0      # W/m²·K
_TAMB_MIN, _TAMB_MAX = 25.0, 85.0        # °C
_T_GLOBAL_MIN, _T_GLOBAL_MAX = 298.0, 600.0  # K (conservative upper bound)


@dataclass
class NormStats:
    """
    Global normalization statistics.

    Computed once over the full training set and frozen.
    Stored alongside model checkpoints so inference can denormalize correctly.
    """
    T_min: float = _T_GLOBAL_MIN
    T_max: float = _T_GLOBAL_MAX
    htc_min: float = _HTC_MIN
    htc_max: float = _HTC_MAX
    t_amb_min: float = _TAMB_MIN
    t_amb_max: float = _TAMB_MAX
    power_mean: float = 0.0
    power_std: float = 1.0
    # Per-geometry domain extents (µm) — populated by compute_from_dataset()
    geom_extents: Dict[str, List[float]] = None  # {geom_name: [L_x, L_y, L_z]}

    def __post_init__(self):
        if self.geom_extents is None:
            self.geom_extents = {}

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'NormStats':
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def norm_coords(self, coords: np.ndarray, geom_name: str) -> np.ndarray:
        extents = self.geom_extents[geom_name]
        return coords / np.array(extents)

    def norm_temp(self, T: np.ndarray) -> np.ndarray:
        return (T - self.T_min) / (self.T_max - self.T_min)

    def denorm_temp(self, T_hat: np.ndarray) -> np.ndarray:
        return T_hat * (self.T_max - self.T_min) + self.T_min

    def norm_power(self, Q: np.ndarray) -> np.ndarray:
        return (Q - self.power_mean) / self.power_std

    def norm_htc(self, htc: float) -> float:
        return (htc - self.htc_min) / (self.htc_max - self.htc_min)

    def norm_t_amb(self, t_amb: float) -> float:
        return (t_amb - self.t_amb_min) / (self.t_amb_max - self.t_amb_min)


def compute_norm_stats(
    npz_files: List[Path],
    geometries: Dict[str, Geometry],
) -> NormStats:
    """
    Compute global normalization statistics from the training dataset.

    Scans all .npz files to determine power distribution.
    Temperature bounds are physical constants, not data-driven.
    """
    all_power = []
    geom_extents: Dict[str, List[float]] = {}

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        meta = dict(data['metadata'][0])
        geom_name = meta.get('geometry', '')

        power = data['power']
        all_power.append(power[power > 0])   # only active regions

        if geom_name not in geom_extents and geom_name in geometries:
            g = geometries[geom_name]
            geom_extents[geom_name] = [
                float(g.die_length),
                float(g.die_width),
                float(g.get_total_height()),
            ]
        data.close()

    if all_power:
        all_power_flat = np.concatenate(all_power)
        power_mean = float(all_power_flat.mean())
        power_std = float(all_power_flat.std()) if all_power_flat.std() > 0 else 1.0
    else:
        power_mean, power_std = 0.0, 1.0

    return NormStats(
        power_mean=power_mean,
        power_std=power_std,
        geom_extents=geom_extents,
    )


@dataclass
class ScenarioData:
    """Normalized data for a single simulation scenario."""
    name: str
    geometry_name: str
    coords: torch.Tensor     # (N, 3) normalised [0,1]
    temp: torch.Tensor       # (N,) normalised [0,1]
    power: torch.Tensor      # (N,) standardised
    layer_ids: torch.Tensor  # (N,) int
    htc_norm: float
    t_amb_norm: float
    tsv_frac: float          # normalised to [0,1] (raw / _TSV_MAX)
    htc: float               # dimensional, for BC computation
    t_amb_K: float           # dimensional ambient in Kelvin, for BC computation
    geom_extents: List[float]                    # [L_x, L_y, L_z] in µm
    power_blocks_wcm2: Dict[str, float] = None  # {block_name: W/cm²} for collocation power
    tim_k_norm: float = 0.05                     # TIM conductivity / 80 W/m·K; 0.05=default 4 W/m·K


_TSV_MAX = 0.10  # maximum TSV density in the benchmark suite (geometry2c)

# Importance weights for layer-stratified collocation sampling.
# Applied as: weight = keyword_weight × layer_thickness, then normalised to fractions.
# Die layers get ~9× their proportional z-fraction; heat sink gets ~0.5×.
_LAYER_TYPE_WEIGHTS: Dict[str, float] = {
    'die':      10.0,
    'tsv':       8.0,
    'tim':       5.0,
    'bonding':   5.0,
    'spreader':  1.5,
    'sink':      0.5,
}


def _layer_sampling_weight(layer_name: str) -> float:
    name = layer_name.lower()
    for keyword, w in _LAYER_TYPE_WEIGHTS.items():
        if keyword in name:
            return w
    return 1.0


class ThermalDataset(Dataset):
    """
    Dataset of thermal simulation scenarios.

    Each item is a full scenario (all N points). Batching by scenario
    ensures each gradient step sees a spatially complete temperature field,
    not random crops.

    All data is loaded into RAM at __init__ (total ~800 MB for 80 scenarios).
    """

    def __init__(
        self,
        npz_files: List[Path],
        geometries: Dict[str, Geometry],
        norm_stats: NormStats,
        si_material_names: Tuple[str, ...] = ('silicon',),
    ):
        self.norm_stats = norm_stats
        self.scenarios: List[ScenarioData] = []

        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            meta = dict(data['metadata'][0])
            geom_name = meta.get('geometry', '')

            if geom_name not in geometries:
                _log.warning("Skipping %s: geometry '%s' not in provided dict", npz_path.name, geom_name)
                data.close()
                continue

            coords_norm = norm_stats.norm_coords(data['coords'], geom_name).astype(np.float32)
            temp_norm = norm_stats.norm_temp(data['temp']).astype(np.float32)
            power_norm = norm_stats.norm_power(data['power']).astype(np.float32)
            layer_ids = data['layer'].astype(np.int64)

            htc = float(meta.get('htc', 5000.0))
            t_amb_c = float(meta.get('t_ambient_celsius', 40.0))
            tsv_raw = float(meta.get('tsv_density', 0.0))

            extents = norm_stats.geom_extents.get(geom_name, [1.0, 1.0, 1.0])

            # Read per-block power densities (W/cm²) stored by NPZExporter.
            # Absent in files exported before this field was added — defaults to
            # empty dict, which leaves collocation-point power as zero (old behaviour).
            power_blocks_wcm2 = {
                key[len('block_power_'):]: float(val)
                for key, val in meta.items()
                if key.startswith('block_power_')
            }

            # TIM k for k-sweep scenarios (geometry5/6 Phase 9); default = 4 W/m·K
            tim_k = float(meta.get('tim_top_k', 4.0))
            tim_k_norm = tim_k / 80.0   # normalise: 80 W/m·K = max in sweep → 1.0

            self.scenarios.append(ScenarioData(
                name=meta.get('scenario_name', npz_path.stem),
                geometry_name=geom_name,
                coords=torch.from_numpy(coords_norm),
                temp=torch.from_numpy(temp_norm),
                power=torch.from_numpy(power_norm),
                layer_ids=torch.from_numpy(layer_ids),
                htc_norm=norm_stats.norm_htc(htc),
                t_amb_norm=norm_stats.norm_t_amb(t_amb_c),
                tsv_frac=tsv_raw / _TSV_MAX,  # normalise to [0,1]
                htc=htc,
                t_amb_K=t_amb_c + 273.15,
                geom_extents=extents,
                power_blocks_wcm2=power_blocks_wcm2,
                tim_k_norm=tim_k_norm,
            ))
            data.close()

        _log.info("ThermalDataset: loaded %d scenarios", len(self.scenarios))

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> ScenarioData:
        return self.scenarios[idx]

    def to_device(self, device: torch.device) -> 'ThermalDataset':
        """Move all tensor data to device (call once at training start)."""
        for sc in self.scenarios:
            sc.coords = sc.coords.to(device)
            sc.temp = sc.temp.to(device)
            sc.power = sc.power.to(device)
            sc.layer_ids = sc.layer_ids.to(device)
        return self


def sample_collocation_points(
    n: int,
    geom_extents: List[float],
    geometry: Geometry,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random interior collocation points for PDE loss.

    Returns:
        coords_norm: (n, 3) uniform in [0,1]³ (normalised)
        layer_ids:   (n,) layer index for each point
    """
    L_x, L_y, L_z = geom_extents
    coords_norm = torch.rand(n, 3, device=device)

    coords_phys = coords_norm.cpu().numpy() * np.array([L_x, L_y, L_z])
    layer_ids = np.array([
        geometry.get_layer_index_at_z(float(z)) for z in coords_phys[:, 2]
    ], dtype=np.int64)
    layer_ids = np.clip(layer_ids, 0, len(geometry.layers) - 1)

    return coords_norm, torch.from_numpy(layer_ids).to(device)


def sample_collocation_stratified(
    n: int,
    geom_extents: List[float],
    geometry: Geometry,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample n collocation points with layer-importance-weighted z-distribution.

    Die/TSV layers receive proportionally more points; the heat sink receives fewer.
    x, y remain uniform over the die footprint.
    Layer ids are assigned directly from the per-layer z-interval (no z-lookup loop).

    Returns:
        coords_norm: (n, 3) normalised in [0,1]³
        layer_ids:   (n,) layer index for each point
    """
    L_x, L_y, L_z = geom_extents

    raw_w = np.array([
        _layer_sampling_weight(l.name) * l.thickness
        for l in geometry.layers
    ], dtype=np.float64)
    fractions = raw_w / raw_w.sum()
    n_per = np.maximum(1, np.round(fractions * n).astype(int))
    # Correct rounding error so sum equals exactly n
    n_per[np.argmax(fractions)] += n - int(n_per.sum())

    coords_list: List[torch.Tensor] = []
    ids_list: List[torch.Tensor] = []
    for i, (layer, n_l) in enumerate(zip(geometry.layers, n_per)):
        if n_l <= 0:
            continue
        xy = torch.rand(int(n_l), 2)
        z_lo = layer.z_bottom / L_z
        z_hi = layer.z_top / L_z
        z = torch.rand(int(n_l)) * (z_hi - z_lo) + z_lo
        coords_list.append(torch.cat([xy, z.unsqueeze(1)], dim=1))
        ids_list.append(torch.full((int(n_l),), i, dtype=torch.long))

    coords = torch.cat(coords_list).to(device)
    ids = torch.cat(ids_list).to(device)
    perm = torch.randperm(len(coords), device=device)
    return coords[perm], ids[perm]


def sample_bc_top_points(
    n: int,
    device: torch.device,
    layer_id: int,
    z_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample n points on a z-face for the convective BC.

    Despite the name (kept for backward compatibility), z_value defaults to
    0.0 (bottom face) — 3D-ICE's ground truth applies convective cooling at
    layer index 0 ("bottom heat sink" directive in ice_simulator.py), not at
    z=1. Pass z_value=1.0, layer_id=<top layer index> to recover the old
    (incorrect for this dataset) behaviour if ever needed elsewhere.
    """
    coords = torch.rand(n, 3, device=device)
    coords[:, 2] = z_value
    layer_ids = torch.full((n,), layer_id, dtype=torch.long, device=device)
    return coords, layer_ids


def sample_bc_side_points(
    n_per_face: int,
    device: torch.device,
    geometry: Geometry,
    geom_extents: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points on all 4 side faces + bottom face for adiabatic BC.

    Returns coords_norm (5*n_per_face, 3) and layer_ids.

    .. deprecated::
        Use sample_bc_faces_grouped() instead — this function passes all five
        face groups as a single concatenated tensor, making it impossible to
        call bc_residual_adiabatic with the correct normal_dim per face.
    """
    coords_list = []
    for dim, val in [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0)]:
        c = torch.rand(n_per_face, 3, device=device)
        c[:, dim] = val
        coords_list.append(c)

    coords = torch.cat(coords_list, dim=0)

    L_x, L_y, L_z = geom_extents
    coords_phys = coords.cpu().numpy() * np.array([L_x, L_y, L_z])
    layer_ids = np.array([
        max(0, geometry.get_layer_index_at_z(float(z)))
        for z in coords_phys[:, 2]
    ], dtype=np.int64)

    return coords, torch.from_numpy(layer_ids).to(device)


def power_at_colloc_points(
    col_coords_norm: torch.Tensor,
    col_layer_ids: torch.Tensor,
    geometry: Geometry,
    power_blocks_wcm2: Dict[str, float],
    geom_extents: List[float],
    power_mean: float,
    power_std: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Return normalised volumetric power density at collocation points.

    Uses vectorised numpy block-membership tests (fast: O(N × n_blocks) in C).
    Points outside any active power block get Q=0, which is physically correct
    for non-active layers and for die-layer points in non-powered regions.

    Args:
        col_coords_norm: (N, 3) normalised coordinates in [0,1]
        col_layer_ids:   (N,) integer layer indices
        geometry:        Geometry object (for power block positions and layer info)
        power_blocks_wcm2: {block_name: power_density_W/cm²} for this scenario
        geom_extents:    [L_x, L_y, L_z] in µm
        power_mean, power_std: normalisation constants from NormStats
        device:          target device for returned tensor

    Returns:
        (N,) normalised power density tensor on `device`
    """
    if not power_blocks_wcm2:
        return torch.zeros(col_coords_norm.shape[0], device=device)

    L_x, L_y, _ = geom_extents
    col_np = col_coords_norm.detach().cpu().numpy()
    ids_np = col_layer_ids.cpu().numpy()

    x_phys = col_np[:, 0] * L_x
    y_phys = col_np[:, 1] * L_y

    layer_name_to_idx = {l.name: i for i, l in enumerate(geometry.layers)}
    power_wm3 = np.zeros(len(col_np), dtype=np.float32)

    for block in geometry.power_blocks:
        p_wcm2 = power_blocks_wcm2.get(block.name, 0.0)
        if p_wcm2 == 0.0:
            continue
        layer_idx = layer_name_to_idx.get(block.layer_name, -1)
        if layer_idx < 0 or not geometry.layers[layer_idx].is_active:
            continue
        thickness_m = geometry.layers[layer_idx].thickness * 1e-6  # µm → m
        p_wm3 = p_wcm2 * 1e4 / thickness_m  # W/cm² → W/m³

        in_block = (
            (ids_np == layer_idx)
            & (x_phys >= block.x) & (x_phys < block.x + block.width)
            & (y_phys >= block.y) & (y_phys < block.y + block.height)
        )
        power_wm3[in_block] = p_wm3

    power_norm = (power_wm3 - power_mean) / power_std
    return torch.from_numpy(power_norm).to(device)


def sample_bc_faces_grouped(
    n_per_face: int,
    device: torch.device,
    geometry: Geometry,
    geom_extents: List[float],
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Sample adiabatic BC points grouped by face normal direction.

    Replaces sample_bc_side_points. Returns one entry per normal direction so
    bc_residual_adiabatic is called with the correct normal_dim for each group.

    z=1 (not z=0) carries the adiabatic BC here: 3D-ICE's ground truth only
    specifies a convective condition at z=0 (layer 0, "bottom heat sink" —
    see sample_bc_top_points), so every OTHER exposed face — including the
    top face near the die/TIM2, previously and incorrectly left unconstrained
    — defaults to adiabatic (dT/dn=0), matching an unspecified-BC-is-adiabatic
    solver convention.

    Returns:
        List of (coords_norm, layer_ids, normal_dim) for:
          normal_dim=0: x=0 and x=1 faces  (2 * n_per_face points)
          normal_dim=1: y=0 and y=1 faces  (2 * n_per_face points)
          normal_dim=2: z=1 top face       (    n_per_face points)
    """
    L_x, L_y, L_z = geom_extents
    groups: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

    for normal_dim, face_vals in [(0, [0.0, 1.0]), (1, [0.0, 1.0]), (2, [1.0])]:
        face_list = []
        for val in face_vals:
            c = torch.rand(n_per_face, 3, device=device)
            c[:, normal_dim] = val
            face_list.append(c)
        coords = torch.cat(face_list, dim=0)

        coords_phys = coords.cpu().numpy() * np.array([L_x, L_y, L_z])
        layer_ids = np.array([
            max(0, geometry.get_layer_index_at_z(float(z)))
            for z in coords_phys[:, 2]
        ], dtype=np.int64)

        groups.append((coords, torch.from_numpy(layer_ids).to(device), normal_dim))

    return groups
