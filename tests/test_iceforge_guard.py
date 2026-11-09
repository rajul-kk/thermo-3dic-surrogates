"""ICESimulator.generate_config_files runs the iceforge lint on the generated stack.stk (no solver)."""
import sys
import tempfile
from pathlib import Path

import pytest

# prefer the in-repo copy of iceforge over any other installed one
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "iceforge"))
pytest.importorskip("iceforge")

from src.core.geometry_builders import build_geometry4
from src.core.placement import grid_steps, place_chiplets
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator


def _generate(geometry):
    scenario = ScenarioGenerator().generate_all_scenarios(geometry)[0]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "out").mkdir()
    sim = ICESimulator(config_dir=tmp, output_dir=tmp / "out", executable="wsl /bin/true")
    sim.generate_config_files(geometry, scenario.to_dict())
    return tmp


def test_guard_raises_on_off_grid_chiplet(monkeypatch):
    monkeypatch.delenv("ICEFORGE_GUARD", raising=False)
    gx, _ = grid_steps(build_geometry4())
    geom = place_chiplets(build_geometry4(), {"chiplet_a": (0.4 * gx, 0.0)})
    with pytest.raises(RuntimeError, match=r"iceforge guard[\s\S]*S00[1345]"):
        _generate(geom)


def test_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ICEFORGE_GUARD", "0")
    gx, _ = grid_steps(build_geometry4())
    geom = place_chiplets(build_geometry4(), {"chiplet_a": (0.4 * gx, 0.0)})
    assert (_generate(geom) / "stack.stk").exists()


def test_guard_passes_grid_aligned(monkeypatch):
    monkeypatch.delenv("ICEFORGE_GUARD", raising=False)
    gx, gy = grid_steps(build_geometry4())
    geom = place_chiplets(build_geometry4(), {"chiplet_a": (2 * gx, gy)})
    assert (_generate(geom) / "stack.stk").exists()
    assert (_generate(build_geometry4()) / "stack.stk").exists()
