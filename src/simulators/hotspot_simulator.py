"""HotSpot thermal simulator wrapper.

The full package stack is modelled, not just the die: the wrapper writes the same
``stack.stk`` that ICESimulator would hand to 3D-ICE (so layers, layouts, heat-sink side and
htc are identical by construction) and translates it with iceforge's HotSpot translation
(tools/iceforge/iceforge/hotspot.py): every stack layer becomes grid layers of a layer
configuration file, so the thick Cu base and spreader spread heat sideways instead of being
lumped as vertical resistances.
"""

import shlex
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .base_simulator import ThermalSimulator
from .ice_simulator import ICESimulator
from ..core.geometry import Geometry

try:
    import iceforge  # noqa: F401
except ImportError:  # fall back to the in-repo copy
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "iceforge"))

from iceforge import hotspot as _ihs          # noqa: E402
from iceforge.model import parse_stk          # noqa: E402


class HotSpotSimulator(ThermalSimulator):
    """Wrapper for HotSpot (grid model, steady-state) over the full 3D-ICE-equivalent stack."""

    GRID_ROWS = 64
    GRID_COLS = 64

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
        if not any(l.is_active for l in geometry.layers):
            raise ValueError(f"No active die layer found in geometry {geometry.name}")

        self._geometry = geometry
        self._scenario = scenario
        # The 3D-ICE-equivalent stack description (never solved with 3D-ICE here).
        stack_dir = self.config_dir / "hotspot_stack"
        ice = ICESimulator(stack_dir, stack_dir / "out", executable="wsl /bin/true")
        ice.generate_config_files(geometry, scenario)
        st = parse_stk(str(stack_dir / "stack.stk"))
        self._case = _ihs.translate(st, r=1, grid=self.GRID_ROWS)
        self._source_idx = self._case.lcf_index(
            next(i for i, L in enumerate(self._case.model.layers) if L.is_source))
        self._die_layer = next(l for l in geometry.layers if l.is_active)
        self._die_z_center = (self._die_layer.z_bottom + self._die_layer.z_top) / 2.0

    def run_simulation(self, scenario_name: str) -> Path:
        grid_steady_file = self.output_dir / "hotspot_grid.steady"
        try:
            grid_steady_file.unlink()
        except OSError:
            pass
        try:
            err, det_err = _ihs.preflight(self.executable)
            if err:
                raise RuntimeError(err)
            if det_err and self._case.heterogeneous:
                raise RuntimeError(det_err)
            text, _ = _ihs.run_hotspot(self._case, self.executable)
        except _ihs.HotspotError as e:
            raise RuntimeError(f"HotSpot simulation failed for {scenario_name}: {e}") from e
        grid_steady_file.write_text(text)
        return self.output_dir

    def parse_results(self, result_path: Path) -> Dict[str, np.ndarray]:
        """Parse HotSpot grid steady-state output for the die (source) layer."""
        result_path = Path(result_path)
        if not hasattr(self, '_case'):
            raise RuntimeError("parse_results called before generate_config_files")
        grid_file = (result_path / "hotspot_grid.steady"
                     if result_path.is_dir() else result_path)
        if not grid_file.exists():
            raise ValueError(f"Grid output not found: {grid_file}")

        layers = _ihs.parse_grid_steady(grid_file.read_text(), self._case)
        if self._source_idx not in layers:
            raise ValueError(f"Die layer {self._source_idx} missing from {grid_file}")
        # layers[...] is (length index, width index); flatten like ICESimulator.parse_results:
        # rows run along the die width (project x = block.x), columns along the length.
        temperature = layers[self._source_idx].T.ravel().astype(np.float32)

        geom = self._geometry
        w_c = (np.arange(self.GRID_ROWS) + 0.5) / self.GRID_ROWS * geom.die_width    # um
        l_c = (np.arange(self.GRID_COLS) + 0.5) / self.GRID_COLS * geom.die_length   # um
        ww, ll = np.meshgrid(w_c, l_c, indexing='ij')
        zz = np.full_like(ww, self._die_z_center)
        coords = np.stack([ww.ravel(), ll.ravel(), zz.ravel()], axis=1).astype(np.float32)
        return {'coords': coords, 'temperature': temperature}

    def cleanup_temp_files(self, scenario_name: str) -> None:
        import shutil
        shutil.rmtree(self.config_dir / "hotspot_stack", ignore_errors=True)
        for f in [self.output_dir / "hotspot_grid.steady"]:
            try:
                f.unlink()
            except OSError:
                pass
