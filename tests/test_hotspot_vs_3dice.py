"""Regression: HotSpotSimulator (full-stack iceforge translation) agrees with 3D-ICE on geometry1.

The old die-only wrapper lumped the 5 mm Cu base and 1 mm spreader as vertical resistances and was
+33 % on the peak rise (corr ~0).  Scenario: 60/30/90/45 W/cm2, htc 1e4, ambient 25 C.  Measured
after the change: peak -0.7 %, mean 0.0 %, RMSE 0.13 K, max diff 0.56 K, corr 0.9998.
Skipped when either solver is missing."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry_builders import build_geometry1
from src.simulators.hotspot_simulator import HotSpotSimulator
from src.simulators.ice_simulator import ICESimulator, DEFAULT_ICE_EXECUTABLE

HOTSPOT_WSL = "wsl /home/rajul/HotSpot_slu/hotspot"
ICE_WSL = DEFAULT_ICE_EXECUTABLE


def _wsl_has(path: str) -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        return subprocess.run(["wsl", "test", "-x", path], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not (_wsl_has("/home/rajul/HotSpot_slu/hotspot")
                         and _wsl_has("/home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator")),
                    reason="HotSpot and/or 3D-ICE binaries not available")
def test_geometry1_matches_3dice():
    g = build_geometry1()
    sc = {'power_blocks': {'block1': 60.0, 'block2': 30.0, 'block3': 90.0, 'block4': 45.0},
          'htc': 10000.0, 't_ambient': 25.0}
    d = Path(tempfile.mkdtemp())
    try:
        ice = ICESimulator(d / 'ic', d / 'io', ICE_WSL)
        ice.generate_config_files(g, sc)
        ri = ice.parse_results(ice.run_simulation('g1'))
        act = next(l for l in g.layers if l.is_active)
        zc = (act.z_bottom + act.z_top) / 2
        T_ice = ri['temperature'][np.isclose(ri['coords'][:, 2], zc)].reshape(ice._ny, ice._nx)

        hs = HotSpotSimulator(d / 'hc', d / 'ho', HOTSPOT_WSL)
        hs.generate_config_files(g, sc)
        rh = hs.parse_results(hs.run_simulation('g1'))
        T64 = rh['temperature'].reshape(64, 64)            # rows = width (as 3D-ICE Tmap), cols = length
        # area-average HotSpot's 64x64 grid onto the 100x100 3D-ICE cells (rows = width, cols = length)
        from iceforge.hotspot import remap
        T_hs = remap(T64.T, ice._nx, ice._ny).T

        amb = 25.0 + 273.15
        a, b = T_ice - amb, T_hs - amb
        assert abs(b.max() / a.max() - 1) < 0.03, (a.max(), b.max())
        assert abs(b.mean() / a.mean() - 1) < 0.01, (a.mean(), b.mean())
        assert np.sqrt(((a - b) ** 2).mean()) < 0.5
        assert np.abs(a - b).max() < 1.5
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.995
    finally:
        shutil.rmtree(d, ignore_errors=True)
