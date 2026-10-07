"""Regression: HotSpotSimulator must give HotSpot a floorplan that covers the whole die.

HotSpot sizes its grid to the bounding box of the floorplan, so power blocks that do not
tile the die (geometry1: four 3 mm blocks on a 10 mm die) used to stretch the field and
shift every coordinate.  Also guards the package setup: a sink/spreader exactly the chip
size made HotSpot return ~0 K everywhere (independent of r_convec and ambient).
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


def _read_flp(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        n, w, h, x, y = line.split()
        rows.append((n, float(w), float(h), float(x), float(y)))
    return rows


def test_floorplan_tiles_the_die_geometry1():
    g = build_geometry1()
    d = Path(tempfile.mkdtemp())
    try:
        sim = HotSpotSimulator(d / 'c', d / 'o', "hotspot")
        sim.generate_config_files(g, {'power_blocks': {'block1': 60.0, 'block2': 30.0,
                                                       'block3': 90.0, 'block4': 45.0}})
        rows = _read_flp(d / 'c' / 'hotspot.flp')
        L, W = g.die_length * 1e-6, g.die_width * 1e-6
        assert sum(w * h for _, w, h, _, _ in rows) == pytest.approx(L * W, rel=1e-6)
        assert max(x + w for _, w, _, x, _ in rows) == pytest.approx(L, rel=1e-6)
        assert max(y + h for _, _, h, _, y in rows) == pytest.approx(W, rel=1e-6)
        assert min(x for *_, x, _ in rows) == 0.0 and min(y for *_, y in rows) == 0.0
        # no overlaps
        for i, (_, w1, h1, x1, y1) in enumerate(rows):
            for _, w2, h2, x2, y2 in rows[i + 1:]:
                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                assert not (ox > 1e-9 and oy > 1e-9)
        # power is carried by the real blocks only and is conserved
        lines = (d / 'c' / 'hotspot.ptrace').read_text().splitlines()
        names, vals = lines[0].split('\t'), [float(v) for v in lines[1].split('\t')]
        assert len(names) == len(rows)
        assert sum(vals) == pytest.approx(5.4 + 2.7 + 8.1 + 4.05, rel=1e-6)
        assert all(v == 0.0 for n, v in zip(names, vals) if n.startswith('fill_'))
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
