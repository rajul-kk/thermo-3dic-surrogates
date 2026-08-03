"""
NumPy data exporter for thermal simulation results.

Converts 3D-ICE/HotSpot simulation output to PINN-ready NumPy format.
Exports (coords, temp, power, layer) arrays as compressed .npz files.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
from ..core.geometry import Geometry
from ..core.mesh import generate_coords_and_indices, generate_power_density_field


class NPZExporter:
    """
    Export thermal simulation results to NumPy .npz format.

    Output Format:
    - coords: (N, 3) array of (x, y, z) coordinates in micrometers
    - temp: (N,) array of temperatures in Kelvin
    - power: (N,) array of volumetric power density in W/m³
    - layer: (N,) array of layer indices (0-based)
    """

    def __init__(self, output_dir: Path):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for output .npz files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_scenario(self,
                       scenario_name: str,
                       geometry: Geometry,
                       scenario_params: Dict[str, Any],
                       temperature_field: np.ndarray,
                       coords: Optional[np.ndarray] = None) -> Path:
        """
        Export a single simulation scenario to .npz format.

        Args:
            scenario_name: Scenario identifier (e.g., 'geometry1_train_001')
            geometry: Geometry object defining the physical structure
            scenario_params: Scenario parameters dict with:
                - power_blocks: {block_name: power_density_W/cm2}
                - htc: Heat transfer coefficient (W/m²·K)
                - t_ambient: Ambient temperature (°C)
            temperature_field: (N,) array of temperatures in K from simulator
            coords: Optional (N, 3) coordinate array. If None, generated from geometry.

        Returns:
            Path to saved .npz file
        """
        # Generate coordinate array if not provided
        if coords is None:
            coords, layer_indices = generate_coords_and_indices(geometry, uniform_z=False)
        else:
            # Derive layer indices from the z-coordinates of the provided grid
            layer_indices = np.array(
                [geometry.get_layer_index_at_z(z) for z in coords[:, 2]],
                dtype=np.int32
            )

        # Generate power density field. When the scenario carries per-cell power
        # maps, they take precedence over the block decomposition: the maps are
        # what the simulator actually used, so rebuilding from blocks here would
        # store a coarse approximation of the source that produced these very
        # temperatures -- a silent train/target mismatch.
        power_density = generate_power_density_field(
            coords, geometry, scenario_params['power_blocks'],
            power_map_by_layer=scenario_params.get('power_map_by_layer'))

        # Prepare metadata
        metadata = {
            'scenario_name': scenario_name,
            'geometry': geometry.name,
            'geometry_type': geometry.geometry_type,
            'num_points': len(coords),
            'num_layers': len(geometry.layers),
            'htc': scenario_params['htc'],
            't_ambient_celsius': scenario_params['t_ambient'],
            't_ambient_kelvin': scenario_params['t_ambient'] + 273.15,
            'pattern': scenario_params.get('pattern', 'unknown'),
            'rdl_joule_fraction': float(scenario_params.get('rdl_joule_fraction', 0.05)),
            'layer_k_overrides': str(scenario_params.get('layer_k_overrides', {})),
            'die_width_um': int(geometry.die_width),
            'die_length_um': int(geometry.die_length),
            'total_height_um': int(geometry.get_total_height()),
            'mesh_resolution': geometry.mesh_resolution,
            'tsv_density': geometry.tsv_density,
        }

        # Store per-block power densities so the PINN trainer can assign
        # correct Q values at collocation points (not just at data-grid points).
        for block_name, power_wcm2 in scenario_params['power_blocks'].items():
            metadata[f'block_power_{block_name}'] = float(power_wcm2)

        # Add layer information
        layer_info = {}
        for i, layer in enumerate(geometry.layers):
            layer_info[f'layer_{i}_name'] = layer.name
            layer_info[f'layer_{i}_k'] = layer.k_thermal
            layer_info[f'layer_{i}_thickness_um'] = int(layer.thickness)
            layer_info[f'layer_{i}_z_bottom_um'] = int(layer.z_bottom)
            layer_info[f'layer_{i}_z_top_um'] = int(layer.z_top)

        metadata.update(layer_info)

        # Store per-scenario normalization constants for PINN denormalization at inference
        norm_stats = {
            'norm_coords_min': coords.min(axis=0).tolist(),
            'norm_coords_max': coords.max(axis=0).tolist(),
            'norm_temp_min': float(temperature_field.min()),
            'norm_temp_max': float(temperature_field.max()),
            'norm_power_mean': float(power_density.mean()),
            'norm_power_std': float(power_density.std()) if power_density.std() > 0 else 1.0,
        }
        metadata.update(norm_stats)

        # Create output filename
        output_file = self.output_dir / f"{scenario_name}.npz"

        # Save to compressed NumPy format
        np.savez_compressed(
            output_file,
            coords=coords.astype(np.float32),  # Save as float32 to reduce size
            temp=temperature_field.astype(np.float32),
            power=power_density.astype(np.float32),
            layer=layer_indices.astype(np.int32),
            metadata=np.array([metadata], dtype=object)
        )

        return output_file

    def load_scenario(self, npz_file: Path) -> Dict[str, Any]:
        """
        Load a scenario from .npz file.

        Args:
            npz_file: Path to .npz file

        Returns:
            Dictionary with keys:
                - coords: (N, 3) array
                - temp: (N,) array
                - power: (N,) array
                - layer: (N,) array
                - metadata: Dict with scenario information
        """
        if not npz_file.exists():
            raise FileNotFoundError(f"File not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)

        result = {
            'coords': data['coords'],
            'temp': data['temp'],
            'power': data['power'],
            'layer': data['layer'],
            'metadata': dict(data['metadata'][0]) if data['metadata'].ndim > 0 else {}
        }

        return result

    def get_file_info(self, npz_file: Path) -> Dict[str, Any]:
        """
        Get summary information about a .npz file without loading all data.

        Args:
            npz_file: Path to .npz file

        Returns:
            Dictionary with file info
        """
        data = np.load(npz_file, allow_pickle=True)

        info = {
            'file_path': str(npz_file),
            'file_size_mb': npz_file.stat().st_size / (1024**2),
            'num_points': len(data['coords']),
            'coords_shape': data['coords'].shape,
            'temp_shape': data['temp'].shape,
            'power_shape': data['power'].shape,
            'layer_shape': data['layer'].shape,
            'metadata': dict(data['metadata'][0]) if data['metadata'].ndim > 0 else {}
        }

        data.close()

        return info

    def batch_export(self,
                    scenarios: list,
                    geometries: Dict[str, Geometry],
                    simulator_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    verbose: bool = True) -> list:
        """
        Export multiple scenarios efficiently.

        Args:
            scenarios: List of ScenarioParameters objects
            geometries: Dict mapping geometry names to Geometry objects
            simulator_results: Dict mapping scenario names to (coords, temps) tuples
            verbose: Print progress messages

        Returns:
            List of output file paths
        """
        output_files = []

        for i, scenario in enumerate(scenarios):
            if verbose:
                print(f"[{i+1}/{len(scenarios)}] Exporting {scenario.name}...", end=' ', flush=True)

            # Get geometry
            if scenario.geometry_name not in geometries:
                raise ValueError(f"Geometry {scenario.geometry_name} not found")
            geometry = geometries[scenario.geometry_name]

            # Get simulator results
            if scenario.name not in simulator_results:
                if verbose:
                    print("SKIPPED (no results)")
                continue

            coords, temps = simulator_results[scenario.name]

            # Export
            try:
                output_file = self.export_scenario(
                    scenario.name,
                    geometry,
                    {
                        'power_blocks': scenario.power_blocks,
                        'htc': scenario.htc,
                        't_ambient': scenario.t_ambient,
                        'pattern': scenario.pattern
                    },
                    temps,
                    coords
                )
                output_files.append(output_file)
                if verbose:
                    print("OK")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                raise

        return output_files

    @staticmethod
    def create_dataset_summary(output_dir: Path) -> Dict[str, Any]:
        """
        Create summary of all .npz files in a directory.

        Args:
            output_dir: Directory containing .npz files

        Returns:
            Dictionary with dataset statistics
        """
        output_dir = Path(output_dir)
        npz_files = sorted(output_dir.glob("*.npz"))

        if not npz_files:
            return {'num_files': 0}

        summary = {
            'num_files': len(npz_files),
            'total_size_mb': sum(f.stat().st_size for f in npz_files) / (1024**2),
            'by_geometry': {},
            'by_type': {'train': 0, 'test': 0},
            'files': []
        }

        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            metadata = dict(data['metadata'][0]) if data['metadata'].ndim > 0 else {}
            data.close()

            geom_name = metadata.get('geometry', 'unknown')
            scenario_name = metadata.get('scenario_name', npz_file.stem)
            scenario_type = 'train' if 'train' in scenario_name else 'test'

            # Group by geometry
            if geom_name not in summary['by_geometry']:
                summary['by_geometry'][geom_name] = {'train': 0, 'test': 0}
            summary['by_geometry'][geom_name][scenario_type] += 1

            # Count by type
            summary['by_type'][scenario_type] += 1

            # File details
            summary['files'].append({
                'name': scenario_name,
                'geometry': geom_name,
                'type': scenario_type,
                'num_points': metadata.get('num_points', 0),
                'htc': metadata.get('htc', 0),
                'ambient_c': metadata.get('t_ambient_celsius', 0)
            })

        return summary
