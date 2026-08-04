"""
Regression guard for a real 3D-ICE 4.0 crash: a die's source layer emitted as
an IDENTIFIER-referenced layout (the Si/underfill footprint mechanism in
ice_simulator.py) combined with a full-mesh-resolution per-cell floorplan
(~5600 elements on geometry5) heap-corrupts the binary ("corrupted size vs.
prev_size", return code 6). resolution=64 was validated crash-free against
the real executable; full mesh resolution was not, on geometry5.

ScenarioGenerator.attach_power_maps must cap the map resolution to 64 whenever
the geometry carries die_footprints and the caller left resolution at its
"use full mesh" default (0). An explicit non-zero resolution request is left
alone -- the cap only substitutes for an unset default.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import (build_geometry1, build_geometry4,
                                        build_geometry5)
from src.scenario.generator import ScenarioGenerator


def test_default_resolution_is_capped_for_footprint_geometries():
    for build in (build_geometry4, build_geometry5):
        geometry = build()
        gen = ScenarioGenerator()
        scenarios = gen.generate_all_scenarios(geometry)
        gen.attach_power_maps(scenarios, geometry, kind='mixed')

        active_layer = next(l for l in geometry.layers if l.is_active)
        pmap = next(s.power_map_by_layer[active_layer.name]
                    for s in scenarios if s.power_map_by_layer)
        assert pmap.shape[0] <= 64 and pmap.shape[1] <= 64


def test_default_resolution_is_unrestricted_without_footprints():
    geometry = build_geometry1()
    assert not geometry.die_footprints
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geometry)
    gen.attach_power_maps(scenarios, geometry, kind='mixed')

    active_layer = next(l for l in geometry.layers if l.is_active)
    pmap = next(s.power_map_by_layer[active_layer.name]
                for s in scenarios if s.power_map_by_layer)
    nx, ny = geometry.mesh_resolution[:2]
    assert pmap.shape == (ny, nx)


def test_explicit_resolution_request_is_not_capped():
    geometry = build_geometry5()
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geometry)
    gen.attach_power_maps(scenarios, geometry, kind='mixed', resolution=100)

    active_layer = next(l for l in geometry.layers if l.is_active)
    pmap = next(s.power_map_by_layer[active_layer.name]
                for s in scenarios if s.power_map_by_layer)
    assert max(pmap.shape) == 100
