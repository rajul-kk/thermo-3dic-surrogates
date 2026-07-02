"""
Thermal statistics calculator for benchmark scenarios.

Computes temperature statistics and hotspot locations from simulation data.
Exports results as JSON for validation and analysis.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import json
from ..core.geometry import Geometry


class StatisticsCalculator:
    """
    Compute thermal statistics from simulation results.

    Calculates:
    - Min/max/mean/median temperature
    - Temperature standard deviation
    - Hotspot location and peak temperature
    - Temperature distribution percentiles
    - Per-layer statistics
    """

    def __init__(self):
        """Initialize statistics calculator."""
        self.stats = {}

    def compute_scenario_stats(self,
                              scenario_name: str,
                              coords: np.ndarray,
                              temperatures: np.ndarray,
                              power_density: np.ndarray,
                              layer_indices: np.ndarray,
                              geometry: Geometry) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a single scenario.

        Args:
            scenario_name: Scenario identifier
            coords: (N, 3) array of (x, y, z) coordinates in μm
            temperatures: (N,) array of temperatures in K
            power_density: (N,) array of volumetric power density in W/m³
            layer_indices: (N,) array of layer indices
            geometry: Geometry object

        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {
            'scenario_name': scenario_name,
            'geometry': geometry.name,
            'num_points': len(temperatures),
        }

        # Global temperature statistics
        stats['temperature'] = {
            'min_k': float(np.min(temperatures)),
            'max_k': float(np.max(temperatures)),
            'mean_k': float(np.mean(temperatures)),
            'median_k': float(np.median(temperatures)),
            'std_k': float(np.std(temperatures)),
            'min_c': float(np.min(temperatures) - 273.15),
            'max_c': float(np.max(temperatures) - 273.15),
            'mean_c': float(np.mean(temperatures) - 273.15),
        }

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        stats['temperature']['percentiles'] = {
            f'p{p}': float(np.percentile(temperatures, p))
            for p in percentiles
        }

        # Hotspot analysis
        hotspot_idx = np.argmax(temperatures)
        stats['hotspot'] = {
            'peak_temperature_k': float(temperatures[hotspot_idx]),
            'peak_temperature_c': float(temperatures[hotspot_idx] - 273.15),
            'location_x_um': float(coords[hotspot_idx, 0]),
            'location_y_um': float(coords[hotspot_idx, 1]),
            'location_z_um': float(coords[hotspot_idx, 2]),
            'layer_index': int(layer_indices[hotspot_idx]),
            'layer_name': geometry.layers[int(layer_indices[hotspot_idx])].name if int(layer_indices[hotspot_idx]) < len(geometry.layers) else 'unknown',
            'power_density_wm3': float(power_density[hotspot_idx]),
        }

        # Power statistics
        stats['power'] = {
            'min_wm3': float(np.min(power_density)),
            'max_wm3': float(np.max(power_density)),
            'mean_wm3': float(np.mean(power_density[power_density > 0])) if np.any(power_density > 0) else 0.0,
            'total_power_w': float(np.sum(power_density) * (geometry.die_width / 100)**2 * (geometry.die_length / 100)**2 / 1e12),  # Rough estimate
        }

        # Per-layer statistics
        stats['by_layer'] = {}
        for layer_idx in range(len(geometry.layers)):
            mask = layer_indices == layer_idx
            if np.any(mask):
                layer_temps = temperatures[mask]
                layer_name = geometry.layers[layer_idx].name

                stats['by_layer'][layer_name] = {
                    'layer_index': layer_idx,
                    'num_points': int(np.sum(mask)),
                    'min_temp_k': float(np.min(layer_temps)),
                    'max_temp_k': float(np.max(layer_temps)),
                    'mean_temp_k': float(np.mean(layer_temps)),
                    'gradient_k_per_um': float((np.max(layer_temps) - np.min(layer_temps)) / (geometry.layers[layer_idx].thickness + 1)),
                }

        # Temperature gradient analysis
        stats['gradients'] = self._compute_gradients(coords, temperatures, geometry)

        # Thermal resistance estimation (ΔT/Q)
        if stats['power']['mean_wm3'] > 0:
            delta_t = stats['temperature']['max_k'] - stats['temperature']['min_k']
            total_power = stats['power']['total_power_w']
            if total_power > 0:
                stats['thermal_resistance_k_per_w'] = float(delta_t / total_power)

        # Temperature distribution bins
        stats['distribution'] = self._compute_distribution(temperatures)

        return stats

    def _compute_gradients(self,
                          coords: np.ndarray,
                          temperatures: np.ndarray,
                          geometry: Geometry) -> Dict[str, float]:
        """
        Compute temperature gradients in x, y, z directions.

        Args:
            coords: (N, 3) coordinate array
            temperatures: (N,) temperature array
            geometry: Geometry object

        Returns:
            Dictionary with gradient statistics
        """
        gradients = {}

        # Lateral gradients (x, y)
        die_points_mask = (coords[:, 2] >= geometry.layers[-1].z_bottom) & \
                         (coords[:, 2] <= geometry.layers[-1].z_top)

        if np.any(die_points_mask):
            die_temps = temperatures[die_points_mask]
            die_x = coords[die_points_mask, 0]
            die_y = coords[die_points_mask, 1]

            # Approximate lateral gradient (max - min) / lateral distance
            if len(die_temps) > 1:
                lateral_spread = np.sqrt((np.max(die_x) - np.min(die_x))**2 +
                                        (np.max(die_y) - np.min(die_y))**2)
                if lateral_spread > 0:
                    gradients['lateral_k_per_um'] = float((np.max(die_temps) - np.min(die_temps)) / lateral_spread)

        # Vertical gradients (z)
        if len(temperatures) > 1:
            dz = np.max(coords[:, 2]) - np.min(coords[:, 2])
            if dz > 0:
                gradients['vertical_k_per_um'] = float((np.max(temperatures) - np.min(temperatures)) / dz)

        # Maximum local gradients
        if len(temperatures) > 10:
            temperature_range = np.max(temperatures) - np.min(temperatures)
            gradients['max_local_gradient_k_per_um'] = float(temperature_range / 100)  # Rough estimate

        return gradients

    def _compute_distribution(self, temperatures: np.ndarray) -> Dict[str, int]:
        """
        Compute temperature distribution histogram.

        Args:
            temperatures: (N,) temperature array

        Returns:
            Dictionary with bin counts
        """
        # Create bins from min to max temperature
        t_min, t_max = np.min(temperatures), np.max(temperatures)
        t_range = t_max - t_min

        if t_range < 1:
            return {'uniform': len(temperatures)}

        # 20 bins
        bins = np.linspace(t_min, t_max, 21)
        counts, _ = np.histogram(temperatures, bins=bins)

        distribution = {}
        for i, count in enumerate(counts):
            bin_center = (bins[i] + bins[i+1]) / 2
            distribution[f'{bin_center:.1f}_K'] = int(count)

        return distribution

    def save_stats(self, output_file: Path, stats: Dict[str, Any], pretty: bool = True) -> None:
        """
        Save statistics to JSON file.

        Args:
            output_file: Path to output JSON file
            stats: Statistics dictionary
            pretty: Whether to pretty-print JSON
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            if pretty:
                json.dump(stats, f, indent=2, default=str)
            else:
                json.dump(stats, f, default=str)

    def load_stats(self, input_file: Path) -> Dict[str, Any]:
        """
        Load statistics from JSON file.

        Args:
            input_file: Path to JSON file

        Returns:
            Statistics dictionary
        """
        with open(input_file, 'r') as f:
            return json.load(f)

    def batch_compute(self,
                     scenarios: list,
                     geometries: Dict[str, object],
                     simulation_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                     output_dir: Optional[Path] = None,
                     verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics for multiple scenarios.

        Args:
            scenarios: List of ScenarioParameters objects
            geometries: Dict mapping geometry names to Geometry objects
            simulation_data: Dict mapping scenario names to (coords, temps, power, layer) tuples
            output_dir: Optional directory to save JSON files
            verbose: Print progress messages

        Returns:
            Dictionary mapping scenario names to statistics
        """
        all_stats = {}

        for i, scenario in enumerate(scenarios):
            if verbose:
                print(f"[{i+1}/{len(scenarios)}] Computing stats for {scenario.name}...", end=' ', flush=True)

            # Get geometry
            if scenario.geometry_name not in geometries:
                if verbose:
                    print("SKIPPED (geometry not found)")
                continue

            geometry = geometries[scenario.geometry_name]

            # Get simulation data
            if scenario.name not in simulation_data:
                if verbose:
                    print("SKIPPED (no data)")
                continue

            coords, temps, power, layers = simulation_data[scenario.name]

            # Compute stats
            try:
                stats = self.compute_scenario_stats(
                    scenario.name,
                    coords,
                    temps,
                    power,
                    layers,
                    geometry
                )

                all_stats[scenario.name] = stats

                # Save to JSON if output dir provided
                if output_dir:
                    output_file = Path(output_dir) / f"{scenario.name}_stats.json"
                    self.save_stats(output_file, stats)

                if verbose:
                    print(f"OK (T_max={stats['temperature']['max_c']:.1f}°C)")

            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                raise

        return all_stats

    @staticmethod
    def create_summary_stats(all_stats: Dict[str, Dict[str, Any]],
                            by_geometry: bool = True) -> Dict[str, Any]:
        """
        Create aggregate statistics across multiple scenarios.

        Args:
            all_stats: Dictionary of per-scenario statistics
            by_geometry: Whether to group by geometry

        Returns:
            Dictionary with aggregate statistics
        """
        summary = {
            'total_scenarios': len(all_stats),
            'global_stats': {}
        }

        if not all_stats:
            return summary

        # Collect all temperature maxima
        all_temps_max = [s['temperature']['max_k'] for s in all_stats.values()]
        all_temps_mean = [s['temperature']['mean_k'] for s in all_stats.values()]

        summary['global_stats'] = {
            'hottest_scenario': max(all_stats.items(), key=lambda x: x[1]['temperature']['max_k'])[0],
            'hottest_temperature_k': float(np.max(all_temps_max)),
            'coldest_scenario': min(all_stats.items(), key=lambda x: x[1]['temperature']['min_k'])[0],
            'coldest_temperature_k': float(np.min([s['temperature']['min_k'] for s in all_stats.values()])),
            'avg_max_temp_k': float(np.mean(all_temps_max)),
            'avg_mean_temp_k': float(np.mean(all_temps_mean)),
        }

        # Group by geometry if requested
        if by_geometry:
            geometries = set(s['geometry'] for s in all_stats.values())
            summary['by_geometry'] = {}

            for geom in sorted(geometries):
                geom_stats = {s: d for s, d in all_stats.items() if d['geometry'] == geom}
                geom_temps_max = [s['temperature']['max_k'] for s in geom_stats.values()]

                summary['by_geometry'][geom] = {
                    'num_scenarios': len(geom_stats),
                    'max_temperature_k': float(np.max(geom_temps_max)),
                    'avg_max_temperature_k': float(np.mean(geom_temps_max)),
                }

        return summary
