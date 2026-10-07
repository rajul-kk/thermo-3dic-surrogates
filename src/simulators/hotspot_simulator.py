"""HotSpot thermal simulator wrapper."""

import subprocess
import shlex
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from .base_simulator import ThermalSimulator
from ..core.geometry import Geometry


class HotSpotSimulator(ThermalSimulator):
    """Wrapper for HotSpot thermal simulator (grid model, steady-state)."""

    GRID_ROWS = 64
    GRID_COLS = 64
    RIM_FACTOR = 1.0005   # spreader = chip x this, sink = spreader x this

    def __init__(self, config_dir: Path, output_dir: Path,
                 executable: str = "hotspot"):
        super().__init__(config_dir, output_dir, executable)

    # ------------------------------------------------------------------
    # ThermalSimulator interface
    # ------------------------------------------------------------------

    def generate_config_files(self, geometry: Geometry,
                              scenario: Dict[str, Any]) -> None:
        if geometry.geometry_type == '3d_stack':
            raise ValueError(
                f"HotSpotSimulator does not support multi-die 3D stacks "
                f"(geometry '{geometry.name}', type '{geometry.geometry_type}'). "
                f"Use ICESimulator for geometries 2a/2b/2c."
            )

        exe_parts = shlex.split(self.executable)
        self._use_wsl = exe_parts[0] == 'wsl'

        self._geometry = geometry
        nx, ny, _ = geometry.mesh_resolution
        self._nx = nx
        self._ny = ny
        self._x_centers = np.linspace(
            geometry.die_length / (2 * nx),
            geometry.die_length - geometry.die_length / (2 * nx),
            nx
        )
        self._y_centers = np.linspace(
            geometry.die_width / (2 * ny),
            geometry.die_width - geometry.die_width / (2 * ny),
            ny
        )

        # Find the active (die) layer and its z-centre
        die_layers = [l for l in geometry.layers if l.is_active]
        if not die_layers:
            raise ValueError(f"No active die layer found in geometry {geometry.name}")
        self._die_layer = die_layers[0]
        self._die_layer_idx = geometry.layers.index(self._die_layer)
        self._die_z_center = (self._die_layer.z_bottom + self._die_layer.z_top) / 2.0

        self._generate_floorplan_file(geometry, scenario)
        self._generate_power_trace_file(geometry, scenario)

    def run_simulation(self, scenario_name: str) -> Path:
        grid_steady_file = self.output_dir / "hotspot_grid.steady"

        # Remove stale output from a previous run
        for f in [self.output_dir / "hotspot.steady", grid_steady_file]:
            try:
                f.unlink()
            except OSError:
                pass

        geometry = self._geometry
        scenario = self._scenario
        htc = scenario.get('htc', 10000.0)
        t_ambient_k = scenario.get('t_ambient', 25.0) + 273.15

        # Effective r_convec: lump the layers BELOW the die + convective BC.  The ICE path
        # puts the heat sink at the bottom of the stack, so layers above the die (e.g.
        # tim2) are dead ends there and carry no heat.
        chip_area_m2 = (geometry.die_length * 1e-6) * (geometry.die_width * 1e-6)
        r_layers = sum(
            (layer.thickness * 1e-6) / layer.k_thermal
            for layer in geometry.layers[:self._die_layer_idx]
            if not layer.is_active
        )
        r_convec = (r_layers + 1.0 / htc) / chip_area_m2

        die = self._die_layer
        t_chip_m = die.thickness * 1e-6
        k_chip = die.k_thermal
        p_chip = die.volumetric_heat_capacity
        chip_side_m = max(geometry.die_length, geometry.die_width) * 1e-6
        # HotSpot divides by the spreader/sink periphery area, so both must be strictly
        # larger than the chip: equal sizes give garbage (all cells ~0 K, independent of
        # r_convec/ambient) or NaN depending on float rounding of the floorplan extent.
        spreader_side_m = chip_side_m * self.RIM_FACTOR
        sink_side_m = spreader_side_m * self.RIM_FACTOR

        exe_parts = shlex.split(self.executable)
        hotspot_bin = exe_parts[-1]

        try:
            if self._use_wsl:
                result = self._run_via_wsl_stage(
                    scenario_name, hotspot_bin,
                    t_chip_m, k_chip, p_chip, (spreader_side_m, sink_side_m),
                    r_convec, t_ambient_k, grid_steady_file
                )
            else:
                flp_arg = str(self.config_dir / "hotspot.flp")
                ptrace_arg = str(self.config_dir / "hotspot.ptrace")
                steady_arg = str(self.output_dir / "hotspot.steady")
                grid_arg = str(grid_steady_file)
                cmd = [hotspot_bin] + self._hotspot_flags(
                    flp_arg, ptrace_arg, t_chip_m, k_chip, p_chip,
                    (spreader_side_m, sink_side_m), r_convec, t_ambient_k,
                    steady_arg, grid_arg
                )
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(
                    f"HotSpot simulation failed for {scenario_name}:\n"
                    f"Return code: {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            if not grid_steady_file.exists():
                raise RuntimeError(
                    f"HotSpot produced no grid output for {scenario_name}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            return self.output_dir

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"HotSpot simulation timed out for {scenario_name}")
        except FileNotFoundError:
            raise RuntimeError(
                f"HotSpot executable not found: {exe_parts[0]}\n"
                f"For WSL use: --hotspot-executable "
                f"\"wsl /home/<user>/HotSpot/hotspot\""
            )

    def parse_results(self, result_path: Path) -> Dict[str, np.ndarray]:
        """Parse HotSpot grid steady-state output for the die layer (Layer 0)."""
        result_path = Path(result_path)
        if not hasattr(self, '_geometry'):
            raise RuntimeError(
                "parse_results called before generate_config_files"
            )

        grid_file = (result_path / "hotspot_grid.steady"
                     if result_path.is_dir()
                     else result_path)

        if not grid_file.exists():
            raise ValueError(f"Grid output not found: {grid_file}")

        # Parse Layer 0 temperatures
        temps_flat = self._parse_layer0(grid_file)
        n_cells = self.GRID_ROWS * self.GRID_COLS
        if len(temps_flat) != n_cells:
            raise ValueError(
                f"Expected {n_cells} cells in Layer 0, got {len(temps_flat)}"
            )

        geom = self._geometry
        # Build (x, y) coordinates for each grid cell (in µm)
        # Cell index = row * GRID_COLS + col
        # x_center = (col + 0.5) / GRID_COLS * die_length  (µm)
        # y_center = (row + 0.5) / GRID_ROWS * die_width   (µm)
        cols = np.arange(self.GRID_COLS)
        rows = np.arange(self.GRID_ROWS)
        x_c = (cols + 0.5) / self.GRID_COLS * geom.die_length  # (GRID_COLS,) µm
        y_c = (rows + 0.5) / self.GRID_ROWS * geom.die_width   # (GRID_ROWS,) µm

        xx, yy = np.meshgrid(x_c, y_c)   # (GRID_ROWS, GRID_COLS)
        zz = np.full_like(xx, self._die_z_center)

        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
        # HotSpot's grid file lists row 0 at the TOP of the die (largest y); our
        # y_c above increases with the row index, so flip the rows to match.
        temperature = np.array(temps_flat, dtype=np.float32).reshape(
            self.GRID_ROWS, self.GRID_COLS)[::-1].ravel()

        return {'coords': coords, 'temperature': temperature}

    def cleanup_temp_files(self, scenario_name: str) -> None:
        for f in [
            self.config_dir / "hotspot.flp",
            self.config_dir / "hotspot.ptrace",
            self.output_dir / "hotspot.steady",
            self.output_dir / "hotspot_grid.steady",
        ]:
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hotspot_flags(self, flp, ptrace, t_chip, k_chip, p_chip,
                       package_sides, r_convec, ambient, steady, grid_steady):
        spreader_side, sink_side = package_sides
        return [
            '-f', flp, '-p', ptrace,
            '-model_type', 'grid',
            '-grid_rows', str(self.GRID_ROWS),
            '-grid_cols', str(self.GRID_COLS),
            '-t_chip', f'{t_chip:.6e}',
            '-k_chip', f'{k_chip:.4f}',
            '-p_chip', f'{p_chip:.2f}',
            '-r_convec', f'{r_convec:.6f}',
            '-ambient', f'{ambient:.2f}',
            '-s_sink', f'{sink_side:.9f}',
            '-t_sink', '1e-9',
            '-k_sink', '1e6',
            '-s_spreader', f'{spreader_side:.9f}',
            '-t_spreader', '1e-9',
            '-k_spreader', '1e6',
            '-steady_file', steady,
            '-grid_steady_file', grid_steady,
        ]

    def _run_via_wsl_stage(self, scenario_name, hotspot_bin,
                            t_chip, k_chip, p_chip, package_sides,
                            r_convec, ambient, grid_steady_dest: Path):
        """
        Run HotSpot via a WSL staging directory to avoid path-with-spaces issues when passing Windows paths through WSL argument parsing.
        """
        # Create a WSL temp dir with no spaces
        r = subprocess.run(['wsl', 'mktemp', '-d'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"wsl mktemp -d failed: {r.stderr}")
        wsl_tmp = r.stdout.strip()

        flp_wsl = self._to_wsl_path(
            (self.config_dir / "hotspot.flp").resolve())
        ptrace_wsl = self._to_wsl_path(
            (self.config_dir / "hotspot.ptrace").resolve())
        grid_wsl = self._to_wsl_path(grid_steady_dest.resolve())

        try:
            # Copy inputs to temp dir using bash single-quoting for spaces
            def sq(p):
                return f"'{p}'"

            copy_cmd = (
                f"cp {sq(flp_wsl)} {wsl_tmp}/hotspot.flp && "
                f"cp {sq(ptrace_wsl)} {wsl_tmp}/hotspot.ptrace"
            )
            r = subprocess.run(['wsl', 'bash', '-c', copy_cmd],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"Failed to copy HotSpot inputs to WSL temp: {r.stderr}")

            # Build hotspot command with no-space WSL paths
            flags = self._hotspot_flags(
                f'{wsl_tmp}/hotspot.flp',
                f'{wsl_tmp}/hotspot.ptrace',
                t_chip, k_chip, p_chip, package_sides, r_convec, ambient,
                f'{wsl_tmp}/hotspot.steady',
                f'{wsl_tmp}/hotspot_grid.steady',
            )
            cmd = ['wsl', hotspot_bin] + flags
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300)

            if result.returncode == 0:
                # Copy grid output back to Windows-accessible path
                copy_back = (
                    f"cp {wsl_tmp}/hotspot_grid.steady {sq(grid_wsl)}"
                )
                r2 = subprocess.run(['wsl', 'bash', '-c', copy_back],
                                    capture_output=True, text=True)
                if r2.returncode != 0:
                    raise RuntimeError(
                        f"Failed to copy HotSpot output back: {r2.stderr}")

            return result

        finally:
            subprocess.run(['wsl', 'rm', '-rf', wsl_tmp],
                           capture_output=True)

    def _die_blocks(self, geometry: Geometry):
        """Power blocks that sit on the simulated die layer (HotSpot models one die layer)."""
        name = self._die_layer.name
        return [b for b in geometry.power_blocks if b.layer_name in (None, name)]

    def _floorplan_elements(self, geometry: Geometry):
        """(name, x, y, w, h) in um for the die blocks plus zero-power fillers.

        HotSpot sizes its grid to the bounding box of the floorplan, so the floorplan must
        cover the whole die or coordinates and field are silently stretched.  The die is cut
        along every block edge; cells not inside a block become 'fill_<i>' elements."""
        blocks = self._die_blocks(geometry)
        if not blocks:
            raise ValueError(f"No power blocks on die layer '{self._die_layer.name}' "
                             f"in geometry {geometry.name}")
        L, W = geometry.die_length, geometry.die_width
        for i, a in enumerate(blocks):
            if a.x < -1e-6 or a.y < -1e-6 or a.x + a.width > L + 1e-6 or a.y + a.height > W + 1e-6:
                raise ValueError(f"Power block {a.name} extends outside the die")
            for b in blocks[i + 1:]:
                ox = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                oy = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
                if ox > 1e-6 and oy > 1e-6:
                    raise ValueError(f"Power blocks {a.name} and {b.name} overlap; "
                                     f"HotSpot floorplans cannot overlap")

        def breaks(vals, hi):
            out = [0.0]
            for v in sorted(vals):
                if 1e-6 < v < hi - 1e-6 and v - out[-1] > 1e-6:
                    out.append(v)
            if hi - out[-1] <= 1e-6 and len(out) > 1:
                out.pop()
            out.append(hi)
            return out

        xs = breaks([v for b in blocks for v in (b.x, b.x + b.width)], L)
        ys = breaks([v for b in blocks for v in (b.y, b.y + b.height)], W)
        elems = [(b.name, b.x, b.y, b.width, b.height) for b in blocks]
        k = 0
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                cx, cy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
                if not any(b.x <= cx < b.x + b.width and b.y <= cy < b.y + b.height
                           for b in blocks):
                    elems.append((f"fill_{k}", xs[i], ys[j],
                                  xs[i + 1] - xs[i], ys[j + 1] - ys[j]))
                    k += 1
        return elems

    def _generate_floorplan_file(self, geometry: Geometry,
                                 scenario: Dict[str, Any]) -> None:
        """Write HotSpot floorplan file (die blocks + zero-power fillers covering the die)."""
        self._scenario = scenario
        flp_file = self.config_dir / "hotspot.flp"
        lines = [
            "# HotSpot Floorplan",
            f"# Geometry: {geometry.name}",
            "# Format: name  width_m  height_m  left_x_m  bottom_y_m",
        ]
        self._flp_names = []
        for name, x, y, w, h in self._floorplan_elements(geometry):
            self._flp_names.append(name)
            lines.append(f"{name}	{w * 1e-6:.9f}	{h * 1e-6:.9f}	{x * 1e-6:.9f}	{y * 1e-6:.9f}")

        with open(flp_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def _generate_power_trace_file(self, geometry: Geometry,
                                   scenario: Dict[str, Any]) -> None:
        """Write HotSpot power trace file (single steady-state time step)."""
        ptrace_file = self.config_dir / "hotspot.ptrace"
        power_scenario = scenario.get('power_blocks', {})
        by_name = {b.name: b for b in self._die_blocks(geometry)}
        names = list(self._flp_names)
        powers = [
            by_name[n].power_watts(power_scenario.get(n, 0.0)) if n in by_name else 0.0
            for n in names
        ]
        with open(ptrace_file, 'w') as f:
            f.write('\t'.join(names) + '\n')
            f.write('\t'.join(f'{p:.6f}' for p in powers) + '\n')

    def _parse_layer0(self, grid_file: Path) -> List[float]:
        """Read Layer 0 cell temperatures from HotSpot grid steady file."""
        temps = []
        in_layer0 = False
        with open(grid_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Layer 0:'):
                    in_layer0 = True
                    continue
                if in_layer0:
                    if line.startswith('Layer '):
                        break  # next layer, stop
                    if not line:
                        continue
                    try:
                        _idx, t = line.split()
                        temps.append(float(t))
                    except ValueError:
                        continue
        return temps

    @staticmethod
    def _to_wsl_path(windows_path) -> str:
        """Convert a Windows absolute path to WSL /mnt/<drive>/... form."""
        p = str(windows_path).replace('\\', '/')
        if len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            rest = p[2:]
            return f"/mnt/{drive}{rest}"
        return p
