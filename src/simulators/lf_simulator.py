"""
Low-fidelity analytical thermal simulator for multi-fidelity ARO training.

Generates temperature fields using a 1D thermal resistance ladder (for the
z-profile) combined with 2D Gaussian spreading (for lateral hotspot variation).
No 3D-ICE installation required — runs in milliseconds per scenario.

Physical model
--------------
T(x, y, z) = T_base(z) + T_spread(x, y, z)

T_base(z):  1D solution with layer-stacked thermal resistances.
            Treats the die as a 1D stack: T_bottom = T_amb + Q_total * R_total
            R_total = sum_i(dz_i / k_i / A_xy) + 1/(htc * A_xy)

T_spread(x, y, z): Superposition of 2D Gaussian "influence functions", one per
            power block.  Each block at (xc, yc) with power Q_block produces a
            Gaussian bump whose sigma is proportional to sqrt(R_th * k_eff) — a
            rough estimate of the lateral spreading length.

Accuracy vs 3D-ICE
------------------
Expect ~20-40% RMSE vs real 3D-ICE on geometry1.  This is deliberate — the LF
data provides structural correlation with the HF data without exact agreement,
which is the right regime for multi-fidelity training (ARO pre-trains on LF,
fine-tunes on HF).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_log = logging.getLogger(__name__)

# Fallback Gaussian spreading half-width in µm when no layer resistance is known
_DEFAULT_SIGMA_UM = 500.0


class LowFidelitySimulator:
    """
    Analytical 1D + 2D-Gaussian thermal simulator.

    Generates NPZ-compatible output dicts (coords, temp, power, layer, metadata)
    using the same coordinate convention as 3D-ICE output.

    Usage::

        sim = LowFidelitySimulator()
        result = sim.simulate(geometry, scenario_params)
        np.savez_compressed('lf_out.npz', **result)
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        geometry,                          # Geometry object
        htc: float,                        # W/m²·K convective coefficient
        t_ambient_celsius: float,          # °C
        power_blocks_wcm2: Dict[str, float],  # {block_name: W/cm²}
        scenario_name: str = 'lf_scenario',
    ) -> dict:
        """
        Run a low-fidelity simulation and return an NPZ-compatible dict.

        Returns keys: coords, temp, power, layer, metadata (same as 3D-ICE path).

        Physical model (correct dimensional analysis)
        ----------------------------------------------
        Per-unit-area 1D heat conduction + Gaussian-smoothed lateral variation:

          T(x,y,z) = T_amb + Q_density(x,y) [W/m²] × R_above(z) [m²K/W]

        where:
          Q_density(x,y) = 2D power flux map (W/m²) with Gaussian smoothing
          R_above(z)      = sum of h_i/k_i for layers above z  +  1/htc  [m²K/W]

        For non-active layers, Q_density = 0 and the local T reflects
        the conducted heat from all active layers above.
        """
        nx, ny, _ = geometry.mesh_resolution
        n_layers   = len(geometry.layers)

        Lx = float(geometry.die_width)   # µm
        Ly = float(geometry.die_length)  # µm
        A_die_m2 = Lx * Ly * 1e-12      # m²

        T_amb_K = t_ambient_celsius + 273.15

        # Grid coordinates (cell centres)
        xs = np.linspace(0.0, Lx, nx, endpoint=False) + Lx / (2 * nx)
        ys = np.linspace(0.0, Ly, ny, endpoint=False) + Ly / (2 * ny)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')  # (nx, ny)

        # ---- Per-unit-area resistance from each layer to ambient [m²K/W] ----
        # Layers are ordered bottom-to-top (index 0 = bottom = heat sink side,
        # index n-1 = top = die side).  Heat flows DOWN from die to ambient.
        # R_to_ambient[i] = R_conv + sum(R_area[j] for j < i)  [towards z=0]
        R_conv_area = 1.0 / htc          # m²K/W
        layer_R_area = np.array(
            [layer.thickness * 1e-6 / layer.k_thermal for layer in geometry.layers],
            dtype=np.float64,
        )  # R_i per unit area [m²K/W]

        R_to_ambient = np.zeros(n_layers, dtype=np.float64)
        for i in range(n_layers):
            R_to_ambient[i] = R_conv_area + layer_R_area[:i].sum()

        # ---- Two-zone lateral homogenization ----
        # A high-k spreader (Cu) distributes heat laterally before reaching
        # the global thermal path.  Only resistance ABOVE the spreader contributes
        # to the local (x,y) temperature variation.  Below the spreader, all
        # points see the area-averaged heat flux Q_avg.
        #
        # Find spreader = layer with max k × thickness (thermal lateral conductance)
        kt_products = np.array(
            [layer.k_thermal * layer.thickness for layer in geometry.layers],
            dtype=np.float64,
        )
        spreader_idx = int(np.argmax(kt_products))  # index of dominant spreader

        # R_local[i] = per-unit-area resistance for layers above the spreader
        # (these see Q(x,y)); R_global[i] = remaining resistance (sees Q_avg)
        R_local = np.zeros(n_layers, dtype=np.float64)
        for i in range(n_layers):
            # Sum only layer resistances between i and spreader_idx (exclusive)
            if i > spreader_idx:
                R_local[i] = layer_R_area[spreader_idx + 1 : i].sum()

        # ---- Total power and area-averaged flux ----
        Q_total_W = self._total_power_W(geometry, power_blocks_wcm2)
        Q_avg = Q_total_W / A_die_m2  # W/m²  (area-averaged power flux)

        # ---- 2D power flux map Q(x,y) [W/m²] ----
        # Sum Gaussian-smoothed contributions from each active power block.
        # sigma = block_size × 0.4  (modest smoothing, preserves hotspot location)
        Q_map = np.zeros((nx, ny), dtype=np.float64)  # W/m²

        for block_name, q_wcm2 in power_blocks_wcm2.items():
            if q_wcm2 <= 0:
                continue
            block = self._find_block(geometry, block_name)
            if block is None:
                continue
            q_m2 = q_wcm2 * 1e4              # W/cm² → W/m²
            # Block footprint: uniform heat flux over block, Gaussian-smoothed
            xc = block.x + block.width  * 0.5   # µm
            yc = block.y + block.height * 0.5
            sigma = max(block.width, block.height) * 0.4   # µm
            r2 = ((XX - xc) ** 2 + (YY - yc) ** 2) / (sigma ** 2)
            Q_map += q_m2 * np.exp(-0.5 * r2)

        # Re-normalise Q_map so that spatial integral = Q_total_W
        Q_map_integral = Q_map.sum() * (Lx / nx) * (Ly / ny) * 1e-12  # Watts
        if Q_map_integral > 0:
            Q_map = Q_map * (Q_total_W / Q_map_integral)

        # ---- Build per-layer temperature and volumetric power slices ----
        all_coords, all_temps, all_powers, all_layer_ids = [], [], [], []

        for layer_idx, layer in enumerate(geometry.layers):
            z_mid = (layer.z_bottom + layer.z_top) * 0.5  # µm

            # Two-zone temperature:
            #   Global: Q_avg × R_to_ambient  (uniform base rise, dominant term)
            #   Local:  (Q(x,y) - Q_avg) × R_local  (hotspot deviation, small)
            T_layer_K = (
                T_amb_K
                + Q_avg * R_to_ambient[layer_idx]
                + (Q_map - Q_avg) * R_local[layer_idx]
            )  # (nx, ny) K

            # Volumetric power density W/m³ = Q_flux/thickness; 0 for non-active
            Q_layer_W = self._layer_power_W(geometry, layer, power_blocks_wcm2)
            if Q_layer_W > 0 and layer.thickness > 0:
                # Local volumetric Q at each (x,y) for this active layer
                Q_vol_xy = Q_map * (Q_layer_W / (Q_total_W + 1e-30)) / (layer.thickness * 1e-6)
            else:
                Q_vol_xy = np.zeros((nx, ny), dtype=np.float64)

            coords_l = np.stack([
                XX.ravel(),
                YY.ravel(),
                np.full(nx * ny, z_mid),
            ], axis=1).astype(np.float32)

            all_coords.append(coords_l)
            all_temps.append(T_layer_K.ravel().astype(np.float32))
            all_powers.append(Q_vol_xy.ravel().astype(np.float32))
            all_layer_ids.append(np.full(nx * ny, layer_idx, dtype=np.int32))

        coords  = np.concatenate(all_coords,    axis=0)
        temps   = np.concatenate(all_temps,     axis=0)
        powers  = np.concatenate(all_powers,    axis=0)
        layer_ids = np.concatenate(all_layer_ids, axis=0)

        meta = {
            'scenario_name':     scenario_name,
            'geometry_name':     geometry.name,
            'htc':               htc,
            't_ambient_celsius': t_ambient_celsius,
            'tsv_density':       getattr(geometry, 'tsv_density', 0.0),
            'simulator':         'lf_analytical',
        }
        meta.update({f'block_power_{k}': v for k, v in power_blocks_wcm2.items()})

        return {
            'coords':   coords,
            'temp':     temps,
            'power':    powers,
            'layer':    layer_ids,
            'metadata': np.array([meta], dtype=object),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _total_power_W(self, geometry, power_blocks_wcm2: Dict[str, float]) -> float:
        """Sum of all active block powers in Watts."""
        total = 0.0
        for block_name, q_wcm2 in power_blocks_wcm2.items():
            block = self._find_block(geometry, block_name)
            if block is not None:
                area_cm2 = block.width * block.height * 1e-8  # µm² → cm²
                total += q_wcm2 * area_cm2
        return max(total, 1e-9)

    def _layer_power_W(self, geometry, layer, power_blocks_wcm2: Dict[str, float]) -> float:
        """Power deposited in one layer (blocks whose layer_name matches, or active layer)."""
        total = 0.0
        for block_name, q_wcm2 in power_blocks_wcm2.items():
            block = self._find_block(geometry, block_name)
            if block is None:
                continue
            block_layer = getattr(block, 'layer_name', None)
            # Match explicitly by layer_name, or fall back to any active layer
            if block_layer == layer.name or (block_layer is None and layer.is_active):
                area_cm2 = block.width * block.height * 1e-8
                total += q_wcm2 * area_cm2
        return total

    @staticmethod
    def _find_block(geometry, block_name: str):
        """Locate a power block by name. Blocks live at geometry.power_blocks."""
        for block in getattr(geometry, 'power_blocks', []):
            if block.name == block_name:
                return block
        return None


def generate_lf_dataset(
    geometries: dict,
    scenarios: list,           # list of scenario param dicts (from generator)
    output_dir: Path,
    simulator: Optional[LowFidelitySimulator] = None,
) -> list:
    """
    Batch-generate low-fidelity NPZ files for all scenarios.

    Each entry in `scenarios` must have keys: geometry_name, htc, t_ambient_celsius,
    power_blocks_wcm2, scenario_name.

    Returns list of output file paths.
    """
    if simulator is None:
        simulator = LowFidelitySimulator()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for sc in scenarios:
        geom_name = sc['geometry_name']
        if geom_name not in geometries:
            _log.warning("Skipping LF scenario %s: geometry '%s' not loaded",
                         sc.get('scenario_name', '?'), geom_name)
            continue

        geometry = geometries[geom_name]
        result = simulator.simulate(
            geometry=geometry,
            htc=float(sc['htc']),
            t_ambient_celsius=float(sc['t_ambient_celsius']),
            power_blocks_wcm2=sc.get('power_blocks_wcm2', {}),
            scenario_name=sc.get('scenario_name', 'lf'),
        )

        fname = f"{sc.get('scenario_name', 'lf')}_lf.npz"
        out_path = output_dir / fname
        np.savez_compressed(str(out_path), **result)
        paths.append(out_path)
        _log.debug("LF: wrote %s", out_path.name)

    _log.info("LF dataset: wrote %d files to %s", len(paths), output_dir)
    return paths
