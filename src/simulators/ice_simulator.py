"""
3D-ICE thermal simulator wrapper.

Generates configuration files, runs 3D-ICE simulations, and parses results.
"""

import subprocess
import shlex
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from .base_simulator import ThermalSimulator
from ..core.geometry import Geometry, Layer, PowerBlock


class ICESimulator(ThermalSimulator):
    """
    Wrapper for 3D-ICE thermal simulator.

    Handles:
    - Stack file (.stk) generation with materials and layers
    - Floorplan file (.flp) generation with power blocks
    - 3D-ICE execution
    - Grid output (Tmap) parsing

    Unit system: 3D-ICE uses µm for length, so thermal properties must be
    converted from SI (m-based) to µm-based units on write.
    """

    def __init__(self, config_dir: Path, output_dir: Path, executable: str = "3D-ICE-Emulator"):
        super().__init__(config_dir, output_dir, executable)

    def generate_config_files(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """
        Generate 3D-ICE configuration files.

        Args:
            geometry: Geometry object
            scenario: Scenario parameters with keys:
                - power_blocks: {block_name: power_density_W/cm2}
                - htc: Heat transfer coefficient (W/m²·K)
                - t_ambient: Ambient temperature (°C)
        """
        exe_parts = shlex.split(self.executable) if self.executable else []
        if not exe_parts:
            raise RuntimeError(
                "3D-ICE executable not configured. "
                "Pass --ice-executable \"wsl /home/<user>/3d-ice/bin/3D-ICE-Emulator\""
            )
        self._use_wsl = exe_parts[0] == 'wsl'

        self._geometry = geometry
        nx, ny, _ = geometry.mesh_resolution
        cell_length = geometry.die_length / nx
        cell_width = geometry.die_width / ny
        self._x_centers = np.linspace(cell_length / 2, geometry.die_length - cell_length / 2, nx)
        self._y_centers = np.linspace(cell_width / 2, geometry.die_width - cell_width / 2, ny)
        self._layer_names = [layer.name for layer in geometry.layers]
        self._layer_z_centers = np.array([
            (layer.z_bottom + layer.z_top) / 2 for layer in geometry.layers
        ])
        self._nx = nx
        self._ny = ny

        # inst_N maps to layers[N] (geometry bottom-to-top order)
        self._inst_to_layer_idx = {f"inst_{i}": i for i in range(len(geometry.layers))}

        self._generate_floorplan_files(geometry, scenario)
        self._generate_stack_file(geometry, scenario)

    def _generate_stack_file(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """Generate 3D-ICE stack description file (.stk) with correct µm-unit syntax."""
        stk_file = self.config_dir / "stack.stk"

        htc = scenario.get('htc', 10000.0)      # W/m²·K
        t_ambient_c = scenario.get('t_ambient', 25.0)
        t_ambient_k = t_ambient_c + 273.15

        # 3D-ICE uses µm; convert SI properties:
        #   k:      W/m/K   → W/µm/K    × 1e-6
        #   rho_cp: J/m³/K  → J/µm³/K   × 1e-18
        #   HTC:    W/m²/K  → W/µm²/K   × 1e-12
        htc_ice = htc * 1e-12

        lines = []
        lines.append("// 3D-ICE Stack Description File")
        lines.append(f"// Geometry: {geometry.name}")
        lines.append("")

        # ── Material definitions ──────────────────────────────────────────────
        layer_k_overrides = scenario.get('layer_k_overrides', {})
        for mat_name, props in self._get_unique_materials(geometry, layer_k_overrides).items():
            k_ice = props['k'] * 1e-6
            rho_cp_ice = props['rho_cp'] * 1e-18
            lines.append(f"material {mat_name} :")
            lines.append(f"   thermal conductivity     {k_ice:.4e} ;")
            lines.append(f"   volumetric heat capacity {rho_cp_ice:.4e} ;")
            lines.append("")

        # ── Heat-sink boundary condition ─────────────────────────────────────
        # Die is the topmost layer; convective cooling is on the bottom surface.
        lines.append("bottom heat sink :")
        lines.append(f"   heat transfer coefficient {htc_ice:.4e} ;")
        lines.append(f"   temperature {t_ambient_k:.2f} ;")
        lines.append("")

        # ── Chip dimensions ───────────────────────────────────────────────────
        nx, ny, _ = geometry.mesh_resolution
        cell_length = geometry.die_length / nx
        cell_width  = geometry.die_width  / ny
        lines.append("dimensions :")
        lines.append(f"   chip length {geometry.die_length:.1f} , width  {geometry.die_width:.1f} ;")
        lines.append(f"   cell length {cell_length:.1f} , width  {cell_width:.1f} ;")
        lines.append("")

        # ── Layer / die type definitions ──────────────────────────────────────
        # 3D-ICE grammar requires ALL layer definitions before ALL die definitions.
        for i, layer in enumerate(geometry.layers):
            if not layer.is_active:
                mat_name = self._get_layer_material_name(layer, layer_k_overrides, i)
                lines.append(f"layer type_layer_{i} :")
                lines.append(f"   height {layer.thickness:.1f} ;")
                lines.append(f"   material {mat_name} ;")
                lines.append("")

        for i, layer in enumerate(geometry.layers):
            if layer.is_active:
                mat_name = self._get_layer_material_name(layer, layer_k_overrides, i)
                lines.append(f"die type_die_{i} :")
                lines.append(f"   source {layer.thickness:.1f} {mat_name} ;")
                lines.append("")

        # ── Stack assembly: 3D-ICE lists TOP → BOTTOM; geometry is BOTTOM → TOP
        # Each active layer has its own per-layer floorplan file.
        lines.append("stack :")
        for i, layer in reversed(list(enumerate(geometry.layers))):
            inst = f"inst_{i}"
            if layer.is_active:
                flp_name = f'floorplan_layer{i}.flp'
                flp_abs  = (self.config_dir / flp_name).resolve()
                flp_path = self._to_wsl_path(flp_abs) if getattr(self, '_use_wsl', False) else str(flp_abs)
                lines.append(f"   die   {inst}  type_die_{i}    floorplan \"{flp_path}\" ;")
            else:
                lines.append(f"   layer {inst}  type_layer_{i} ;")
        lines.append("")

        # ── Solver ────────────────────────────────────────────────────────────
        lines.append("solver :")
        lines.append("   steady ;")
        lines.append(f"   initial temperature {t_ambient_k:.2f} ;")
        lines.append("")

        # ── Output: Tmap per stack element ────────────────────────────────────
        # Files are written to output_dir named output_inst_N.txt
        lines.append("output :")
        for i in reversed(range(len(geometry.layers))):
            inst = f"inst_{i}"
            out_abs  = (self.output_dir / f"output_{inst}.txt").resolve()
            out_path = self._to_wsl_path(out_abs) if getattr(self, '_use_wsl', False) else str(out_abs)
            lines.append(f"   Tmap ( {inst}, \"{out_path}\", final ) ;")
        lines.append("")

        with open(stk_file, 'w') as f:
            f.write('\n'.join(lines))

    def _generate_floorplan_files(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """
        Generate one 3D-ICE floorplan file per active layer.

        Each file (floorplan_layer{i}.flp) contains only the power blocks that
        reference layer i by name.  TSV-region blocks are excluded — they are
        passive thermal conductors in 3D-ICE (heat generated by TSVs is zero;
        their effect enters via the layer's effective thermal conductivity).

        This replaces the old single-file approach, fixing geometry2 (two active
        die layers with different block sets) and enabling geometry5 (die_zone_1
        and die_zone_2 have completely separate block sets).
        """
        power_scenario = scenario.get('power_blocks', {})

        for i, layer in enumerate(geometry.layers):
            if not layer.is_active:
                continue

            flp_file = self.config_dir / f'floorplan_layer{i}.flp'

            # Only blocks for this layer; exclude TSV regions (zero heat source)
            layer_blocks = [
                b for b in geometry.power_blocks
                if b.layer_name == layer.name and not b.is_tsv_region
            ]

            lines = []
            lines.append(f"// 3D-ICE Floorplan — {geometry.name} layer {i}: {layer.name}")
            lines.append("")

            for block in layer_blocks:
                power_density_wcm2 = power_scenario.get(block.name, 0.0)
                power_w = block.power_watts(power_density_wcm2)
                # 3D-ICE floorplan: position(X_sw, Y_sw) where X is along chip_length
                # (= die_length = PINN y-axis) and Y is along chip_width (= die_width
                # = PINN x-axis).  Swap block.x/y and width/height accordingly.
                lines.append(f"{block.name} :")
                lines.append(f"   position  {block.y:.1f}, {block.x:.1f} ;")
                lines.append(f"   dimension {block.height:.1f}, {block.width:.1f} ;")
                lines.append(f"   power values  {power_w:.6f} ;")
                lines.append("")

            with open(flp_file, 'w') as f:
                f.write('\n'.join(lines))

    def _generate_power_trace(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """Placeholder for transient power trace (steady-state uses floorplan power values)."""
        pass

    def _get_layer_material_name(self, layer, layer_k_overrides: dict, layer_idx: int) -> str:
        """Return the 3D-ICE material name for a layer, mangling if k is overridden."""
        if layer.name in layer_k_overrides:
            k_int = int(round(layer_k_overrides[layer.name]))
            return f"{layer.material}_k{k_int}"
        return layer.material

    def _get_unique_materials(self, geometry: Geometry,
                              layer_k_overrides: dict = None) -> Dict[str, Dict]:
        """Extract unique materials and their properties from geometry.

        layer_k_overrides: {layer_name: k_override_W_per_mK} — per-scenario TIM
        pump-out or other material k substitutions.  Overridden layers get a
        uniquely named material entry so non-overridden layers are unaffected.
        """
        layer_k_overrides = layer_k_overrides or {}
        materials = {}
        for i, layer in enumerate(geometry.layers):
            mat_name = self._get_layer_material_name(layer, layer_k_overrides, i)
            if mat_name not in materials:
                k = layer_k_overrides.get(layer.name, layer.k_thermal)
                materials[mat_name] = {
                    'k': k,
                    'rho_cp': layer.volumetric_heat_capacity
                }
        return materials

    @staticmethod
    def _to_wsl_path(windows_path) -> str:
        """Convert a Windows absolute path to its WSL /mnt/<drive>/... equivalent."""
        p = str(windows_path).replace('\\', '/')
        if len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            rest  = p[2:]
            return f"/mnt/{drive}{rest}"
        return p

    def run_simulation(self, scenario_name: str) -> Path:
        """
        Run 3D-ICE simulation.

        Returns:
            Path to output directory containing per-layer Tmap files.

        Raises:
            RuntimeError: If simulation fails or no output is produced.
        """
        stk_file = self.config_dir / "stack.stk"

        # Remove stale output files from a previous geometry with more layers
        for stale in self.output_dir.glob("output_inst_*.txt"):
            try:
                stale.unlink()
            except OSError:
                pass

        exe_parts = shlex.split(self.executable)
        use_wsl   = exe_parts[0] == 'wsl'
        stk_arg   = self._to_wsl_path(stk_file.resolve()) if use_wsl else str(stk_file)
        cmd       = exe_parts + [stk_arg]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=None if use_wsl else str(self.config_dir),
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"3D-ICE simulation failed for {scenario_name}:\n"
                    f"Return code: {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # Expect at least one output_inst_*.txt file
            output_files = list(self.output_dir.glob("output_inst_*.txt"))
            if not output_files:
                raise RuntimeError(
                    f"3D-ICE produced no output files in {self.output_dir}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            return self.output_dir

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"3D-ICE simulation timed out for {scenario_name}")
        except FileNotFoundError:
            raise RuntimeError(
                f"3D-ICE executable not found: {exe_parts[0]}\n"
                f"For WSL: use --ice-executable \"wsl /home/<user>/3d-ice/bin/3D-ICE-Emulator\""
            )

    def parse_results(self, result_path: Path) -> Dict[str, np.ndarray]:
        """
        Parse 3D-ICE Tmap output files.

        3D-ICE Tmap format (uniform grid, one file per stack element):
            % Thermal map for layer <inst> (please find axis information ...)
            T_00  T_01  ...  T_0m
            T_10  T_11  ...  T_1m
            ...

        Rows correspond to x-cells (NRows), columns to y-cells (NColumns).

        Args:
            result_path: Path to output directory (or single output file for
                         backward compatibility with the 4-column text format).

        Returns:
            Dictionary with:
                - 'coords': (N, 3) float32 array of (x, y, z) in µm
                - 'temperature': (N,) float32 array of temperatures in K
        """
        result_path = Path(result_path)

        if not hasattr(self, '_geometry'):
            raise RuntimeError(
                "parse_results called before generate_config_files; "
                "geometry metadata is not available."
            )

        # ── Collect per-layer Tmap files ──────────────────────────────────────
        if result_path.is_dir():
            output_files = sorted(result_path.glob("output_inst_*.txt"))
        else:
            # Fallback: single file passed (4-column or legacy format)
            output_files = [result_path] if result_path.exists() else []

        if not output_files:
            raise ValueError(f"No output files found at {result_path}")

        coords_all: List[np.ndarray] = []
        temps_all:  List[np.ndarray] = []

        for out_file in output_files:
            # Derive layer index from filename: output_inst_N.txt → N
            stem = out_file.stem            # e.g. "output_inst_2"
            try:
                layer_idx = int(stem.split('_')[-1])
            except ValueError:
                continue

            if layer_idx >= len(self._layer_z_centers):
                continue

            matched_z = float(self._layer_z_centers[layer_idx])

            with open(out_file, 'r') as f:
                content = f.read()

            temp_rows = []
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                try:
                    row_vals = [float(v) for v in line.split()]
                    if row_vals:
                        temp_rows.append(row_vals)
                except ValueError:
                    continue

            if not temp_rows:
                continue

            # Guard against rare inhomogeneous 3D-ICE output (some rows have
            # different column counts — seen with near-zero power or high HTC).
            row_lengths = [len(r) for r in temp_rows]
            if len(set(row_lengths)) > 1:
                # Pick expected_width = self._ny (or mode if _ny not set).
                expected_width = getattr(self, '_ny', max(set(row_lengths), key=row_lengths.count))
                temp_rows = [r for r in temp_rows if len(r) == expected_width]
                if not temp_rows:
                    continue

            temp_grid = np.array(temp_rows, dtype=np.float64)   # shape (n_rows, n_cols)
            n_rows, n_cols = temp_grid.shape

            x_c = (self._x_centers if n_rows == self._nx
                   else np.linspace(self._x_centers[0], self._x_centers[-1], n_rows))
            y_c = (self._y_centers if n_cols == self._ny
                   else np.linspace(self._y_centers[0], self._y_centers[-1], n_cols))

            xx, yy = np.meshgrid(x_c, y_c, indexing='ij')
            zz = np.full_like(xx, matched_z)

            # 3D-ICE Tmap rows = chip_length direction (die_length = PINN y),
            # columns = chip_width direction (die_width = PINN x).
            # Swap so that coords[:,0]=x (die_width) and coords[:,1]=y (die_length)
            # to match PINN's coordinate convention for non-square geometries.
            coords_all.append(np.stack([yy.ravel(), xx.ravel(), zz.ravel()], axis=1))
            temps_all.append(temp_grid.ravel())

        if not coords_all:
            # Last-resort fallback: try 4-column (x y z T) whitespace format
            if result_path.is_file():
                try:
                    data = np.loadtxt(result_path, comments='#')
                    if data.ndim == 2 and data.shape[1] >= 4:
                        return {
                            'coords':      data[:, 0:3].astype(np.float32),
                            'temperature': data[:, 3].astype(np.float32)
                        }
                except Exception:
                    pass
            raise ValueError(f"No valid temperature grids found in {result_path}")

        coords      = np.concatenate(coords_all, axis=0).astype(np.float32)
        temperature = np.concatenate(temps_all,  axis=0).astype(np.float32)

        return {'coords': coords, 'temperature': temperature}

    def cleanup_temp_files(self, scenario_name: str) -> None:
        """Clean up temporary config files created during simulation."""
        targets = [self.config_dir / "stack.stk"]
        # Remove all per-layer floorplan files
        targets += list(self.config_dir.glob("floorplan_layer*.flp"))
        # Also remove legacy single-file if present (from old runs)
        targets.append(self.config_dir / "floorplan.flp")
        for f in targets:
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass
