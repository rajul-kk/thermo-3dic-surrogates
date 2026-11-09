"""Regression: HotSpotSimulator.parse_results must return y increasing with row.

HotSpot's grid output has row 0 at the top (max y). A hot block in the y=0 corner
must therefore peak at small y in the parsed field, not at large y.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry import Geometry, Layer, PowerBlock
from src.simulators.hotspot_simulator import HotSpotSimulator

HOTSPOT_WSL = "wsl /home/rajul/HotSpot_slu/hotspot"


def _hotspot_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        r = subprocess.run(["wsl", "test", "-x", "/home/rajul/HotSpot_slu/hotspot"],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _hotspot_available(), reason="HotSpot binary not available")
def test_corner_block_peaks_at_small_y():
    layers = [Layer('base', 'silicon', 500, 150, 1.6e6),
              Layer('die', 'silicon', 100, 150, 1.6e6, is_active=True)]
    # Blocks tile the die so HotSpot's floorplan bounding box is the full die.
    blocks = [PowerBlock('hot', 0, 0, 3000, 2000, layer_name='die'),
              PowerBlock('a', 3000, 0, 7000, 10000, layer_name='die'),
              PowerBlock('b', 0, 2000, 3000, 8000, layer_name='die')]
    g = Geometry('orient', '2d_stack', layers, blocks, die_width=10000, die_length=10000,
                 mesh_resolution=(20, 20, 4))
    d = Path(tempfile.mkdtemp())
    try:
        sim = HotSpotSimulator(d / 'c', d / 'o', HOTSPOT_WSL)
        sim.generate_config_files(g, {'power_blocks': {'hot': 1e5, 'a': 0.0, 'b': 0.0},
                                      'htc': 10000.0, 't_ambient': 25.0})
        out = sim.parse_results(sim.run_simulation('orient'))
        i = int(np.argmax(out['temperature']))
        x, y = out['coords'][i, 0], out['coords'][i, 1]
        assert x < 3000 and y < 2000, f"hotspot at ({x}, {y}), expected inside block (0-3000, 0-2000)"
    finally:
        shutil.rmtree(d, ignore_errors=True)
