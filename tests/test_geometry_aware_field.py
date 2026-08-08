"""
Distance-to-nearest-power-block field: the right-sized geometry-aware
conditioning signal for this benchmark (all 6 geometries are structured
Cartesian grids, not the point-cloud/graph targets GINO/PI-GANO-style full
SDF encoders are built for). Unlike the TSV field, this is purely geometric
-- constant per geometry, no simulation data needed. See goal.md Track B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_geometry1, build_all_geometries
from src.core.mesh import generate_coords_and_indices, generate_distance_to_power_block_field


def test_points_inside_a_power_block_score_zero():
    g = build_geometry1()
    coords, _ = generate_coords_and_indices(g, uniform_z=False)
    dist = generate_distance_to_power_block_field(coords, g)
    inside_frac = (dist == 0).mean()
    # geometry1: 4 corner blocks, 3x3mm each, on a 10x10mm die = 36% coverage
    assert inside_frac == pytest.approx(0.36, abs=0.02)


def test_distance_is_nonnegative_and_bounded_by_the_diagonal():
    g = build_geometry1()
    coords, _ = generate_coords_and_indices(g, uniform_z=False)
    dist = generate_distance_to_power_block_field(coords, g)
    diag = np.hypot(g.die_width, g.die_length)
    assert np.all(dist >= 0.0)
    assert np.all(dist <= 1.0 + 1e-9)
    assert dist.max() * diag <= diag


def test_field_is_deterministic_and_scenario_independent():
    """Unlike generate_tsv_field, this takes no scenario argument at all --
    calling it twice on the same geometry must be bit-identical."""
    g = build_geometry1()
    coords, _ = generate_coords_and_indices(g, uniform_z=False)
    d1 = generate_distance_to_power_block_field(coords, g)
    d2 = generate_distance_to_power_block_field(coords, g)
    assert np.array_equal(d1, d2)


def test_farther_points_score_higher_than_near_points():
    g = build_geometry1()
    b = g.power_blocks[0]
    center = np.array([[b.x + b.width / 2, b.y + b.height / 2, 100.0]])
    # Die centre (5000, 5000) sits in the gap between geometry1's 4 corner
    # blocks (each 3x3mm at the die's corners) -- genuinely outside all of them.
    far = np.array([[5000.0, 5000.0, 100.0]])
    d_center = generate_distance_to_power_block_field(center, g)[0]
    d_far = generate_distance_to_power_block_field(far, g)[0]
    assert d_center == 0.0
    assert d_far > d_center


def test_geometry_with_no_power_blocks_is_all_zero():
    from src.core.geometry import Geometry, Layer
    from src.core.material import MaterialLibrary
    mat = MaterialLibrary.get('silicon')
    g = Geometry(
        name='no_blocks', geometry_type='2d_stack',
        layers=[Layer(name='die', material='silicon', thickness=100.0,
                      k_thermal=mat.k_thermal,
                      volumetric_heat_capacity=mat.volumetric_heat_capacity,
                      is_active=True)],
        power_blocks=[], die_width=10000.0, die_length=10000.0,
    )
    coords = np.array([[1000.0, 1000.0, 50.0], [9000.0, 9000.0, 50.0]])
    dist = generate_distance_to_power_block_field(coords, g)
    assert np.all(dist == 0.0)


def test_works_on_every_registered_geometry():
    """Structural smoke test -- must not crash on any of the 6 geometries,
    including the chiplet ones with TSV-region blocks to exclude."""
    for g in build_all_geometries():
        coords, _ = generate_coords_and_indices(g, uniform_z=False)
        dist = generate_distance_to_power_block_field(coords, g)
        assert dist.shape == (coords.shape[0],)
        assert np.all(np.isfinite(dist))
