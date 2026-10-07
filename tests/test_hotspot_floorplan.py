"""Regression: HotSpotSimulator must give HotSpot floorplans that cover the whole die.

HotSpot sizes its grid to the bounding box of the floorplan, so power blocks that do not
tile the die (geometry1: four 3 mm blocks on a 10 mm die) used to stretch the field and
shift every coordinate (iceforge's translation tiles every layer).  Also guards the package
setup: a sink/spreader exactly the chip size made HotSpot return ~0 K everywhere.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry import Geometry, Layer, PowerBlock
from src.core.geometry_builders import build_geometry1
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


def test_full_stack_translation_geometry1():
    """The wrapper models every stack layer (not a lumped die): the HotSpot layer file has
    several layers, each floorplan tiles the die, and power is conserved."""
    g = build_geometry1()
    d = Path(tempfile.mkdtemp())
    try:
        sim = HotSpotSimulator(d / 'c', d / 'o', "hotspot")
        sim.generate_config_files(g, {'power_blocks': {'block1': 60.0, 'block2': 30.0,
                                                       'block3': 90.0, 'block4': 45.0},
                                      'htc': 10000.0, 't_ambient': 25.0})
        case = sim._case
        assert len(case.layers) >= 2 * len(g.layers) - 1          # half-thickness split per layer
        L = g.die_length * 1e-6
        for name, text in case.files.items():
            if not name.endswith('.flp'):
                continue
            rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith('#')]
            area = sum(float(r[1]) * float(r[2]) for r in rows)
            assert area == pytest.approx(L * (g.die_width * 1e-6), rel=1e-6), name
        vals = [float(v) for v in case.files['power.ptrace'].splitlines()[1].split('	')]
        assert sum(vals) == pytest.approx(5.4 + 2.7 + 8.1 + 4.05, rel=1e-6)
        assert sim._case.sink_side == 'bottom'
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.skipif(not _hotspot_available(), reason="HotSpot binary not available")
def test_non_tiling_block_locates_and_heats_sensibly():
    layers = [Layer('base', 'silicon', 500, 150, 1.6e6),
              Layer('die', 'silicon', 100, 150, 1.6e6, is_active=True)]
    # One 2 x 2 mm block at (1, 1) mm on a 10 x 10 mm die: the floorplan alone would be 2 x 2 mm.
    blocks = [PowerBlock('hot', 1000, 1000, 2000, 2000, layer_name='die')]
    g = Geometry('nontiling', '2d_stack', layers, blocks, die_width=10000, die_length=10000,
                 mesh_resolution=(20, 20, 4))
    d = Path(tempfile.mkdtemp())
    try:
        sim = HotSpotSimulator(d / 'c', d / 'o', HOTSPOT_WSL)
        sim.generate_config_files(g, {'power_blocks': {'hot': 100.0},
                                      'htc': 10000.0, 't_ambient': 25.0})
        out = sim.parse_results(sim.run_simulation('nontiling'))
        c, t = out['coords'], out['temperature']
        i = int(np.argmax(t))
        assert 1000 <= c[i, 0] <= 3000 and 1000 <= c[i, 1] <= 3000, \
            f"peak at ({c[i, 0]}, {c[i, 1]}), expected inside the block"
        assert c[:, 0].max() > 9500 and c[:, 1].max() > 9500      # field spans the whole die
        rise = t - 298.15
        assert t.min() > 298.15 - 0.01, "temperatures below ambient: degenerate package setup"
        assert 1.0 < rise.max() < 200.0, f"implausible peak rise {rise.max():.2f} K"
        assert rise.min() > 0.1, "far cells should still sit above ambient (shared convective film)"
    finally:
        shutil.rmtree(d, ignore_errors=True)
