"""
Die-footprint layout emission for geometry4/5/6 (chiplet-on-interposer).

3D-ICE previously received a single uniform silicon material for the whole
die_zone layer; the underfill gap between chiplets (k=0.7 vs Si's 148 W/m·K)
was only modelled in the PINN's PDE loss, not in the 3D-ICE ground truth --
a real train/target inconsistency (assumptions.md sect 6.1). This makes the
3D-ICE stack file carry the same heterogeneous material via a 3D-ICE 4.0
layout, closing that gap.

A previous version of this code swapped the axis convention (see
_generate_floorplan_files' documented X<->chip_length / Y<->chip_width swap)
and 3D-ICE rejected the layout with "Layout element is outside of the IC" --
caught only by running against the real executable. These tests pin the
swapped-axis contract without needing 3D-ICE installed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import (build_geometry1, build_geometry4,
                                        build_geometry5, build_geometry6)
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator


def _configure(geometry):
    gen = ScenarioGenerator()
    scenario = gen.generate_all_scenarios(geometry)[0]
    tmp = Path(tempfile.mkdtemp())
    sim = ICESimulator(config_dir=tmp, output_dir=tmp / "out",
                        executable="wsl /bin/true")
    (tmp / "out").mkdir(exist_ok=True)
    sim.generate_config_files(geometry, scenario.to_dict())
    return sim, tmp


@pytest.mark.parametrize("build", [build_geometry4, build_geometry5, build_geometry6])
def test_footprints_grouped_by_die_layer(build):
    geometry = build()
    sim = ICESimulator(config_dir=Path("."), output_dir=Path("."))
    grouped = sim._footprints_by_layer(geometry)
    assert grouped, "chiplet geometry must declare die_footprints"
    covered_layers = {fp.die_layer_name for fp in geometry.die_footprints}
    assert set(grouped.keys()) == covered_layers


@pytest.mark.parametrize("build", [build_geometry4, build_geometry5, build_geometry6])
def test_footprint_rectangles_stay_within_chip_bounds(build):
    """Regression guard: DiePrint (x, y) is along (die_width, die_length) per the
    same axis convention _generate_floorplan_files documents and swaps on write."""
    geometry = build()
    for fp in geometry.die_footprints:
        assert fp.x + fp.width <= geometry.die_width + 1e-6
        assert fp.y + fp.height <= geometry.die_length + 1e-6


def test_layout_file_uses_swapped_axes_and_silicon_material():
    geometry = build_geometry4()
    sim, tmp = _configure(geometry)

    lyt = (tmp / "layout_footprint_die_zone.lyt").read_text()
    assert "silicon :" in lyt

    fp = geometry.die_footprints[0]
    expected = f"rectangle ( {fp.y:.1f}, {fp.x:.1f}, {fp.height:.1f}, {fp.width:.1f} )"
    assert expected in lyt


def test_stack_file_references_source_layer_with_layout():
    geometry = build_geometry4()
    sim, tmp = _configure(geometry)
    stk = (tmp / "stack.stk").read_text()

    assert "material underfill_k0_70" in stk
    # An active die layer with footprints must be declared as a standalone
    # `layer ... layout ...` and referenced by identifier from `source`,
    # never emitted as a bare `source {thickness} {material} ;`.
    assert "_src :" in stk
    assert "source type_layer_" in stk and "_src ;" in stk


def test_no_footprint_layout_when_geometry_has_none():
    geometry = build_geometry1()
    sim, tmp = _configure(geometry)

    assert not list(tmp.glob("layout_footprint_*.lyt"))
    stk = (tmp / "stack.stk").read_text()
    assert "_src :" not in stk
    assert "underfill" not in stk
