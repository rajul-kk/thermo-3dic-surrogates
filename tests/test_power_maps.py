"""
Per-cell power maps.

Block-scalar power spans exactly ~4 dimensions no matter how many scenarios are
generated, which is why closed-form ridge regression solved the old dataset
(docs/report.md 9.3). A power map makes the source a function instead, so the
solution operator becomes a genuine Green's function rather than a
low-dimensional linear map.

These tests pin the two properties that make the change worth its cost: the maps
really are high-dimensional, and swapping them in does not break the physical
constraints (TDP budget, silicon density ceiling, cooling adequacy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_geometry1, build_all_geometries
from src.core.mesh import generate_coords_and_indices, generate_power_density_field
from src.scenario.generator import ScenarioGenerator
from src.scenario.power_maps import POWER_MAP_KINDS, generate_power_map


def _eff_rank(M: np.ndarray) -> float:
    """Participation-ratio effective rank of the row space, mean removed."""
    M = M - M.mean(0)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-9]
    return float((sv.sum() ** 2) / (sv ** 2).sum())


# ── Map generation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('kind', POWER_MAP_KINDS)
def test_maps_are_normalised_and_bounded(kind):
    m = generate_power_map(48, 48, kind=kind, seed=0)
    assert m.shape == (48, 48)
    assert np.isfinite(m).all()
    assert m.max() == pytest.approx(1.0)
    assert m.min() > 0.0, "idle floor must keep every cell dissipating"


@pytest.mark.parametrize('kind', POWER_MAP_KINDS)
def test_maps_are_reproducible_and_seed_dependent(kind):
    a = generate_power_map(32, 32, kind=kind, seed=7)
    b = generate_power_map(32, 32, kind=kind, seed=7)
    c = generate_power_map(32, 32, kind=kind, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_map_collection_is_high_dimensional():
    """
    The whole point. A collection of N maps must span close to N dimensions --
    every scenario carrying new information -- rather than collapsing onto the
    handful of degrees of freedom that block scalars provide.
    """
    N = 40
    M = np.stack([generate_power_map(48, 48, kind='mixed', seed=s).ravel()
                  for s in range(N)])
    rank = _eff_rank(M)
    assert rank > 0.6 * N, (
        f"power maps span only {rank:.1f} of {N} dimensions; with too long a "
        f"correlation length they degenerate to roughly the 4 dimensions of the "
        f"block-scalar power they are meant to replace")


def test_long_correlation_length_degenerates():
    """
    Guards the tuning that matters: an over-smoothed map carries no more
    information than block scalars. This was the initial default and it failed.
    """
    N = 40
    smooth = np.stack([generate_power_map(48, 48, kind='grf', seed=s,
                                          corr_frac=0.5).ravel() for s in range(N)])
    assert _eff_rank(smooth) < 10, "expected over-smoothed maps to be low-rank"


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown power map kind"):
        generate_power_map(16, 16, kind='not-a-kind')


# ── Integration with the scenario generator ────────────────────────────────────

@pytest.mark.parametrize('geom', build_all_geometries(),
                         ids=[g.name for g in build_all_geometries()])
def test_attaching_maps_preserves_total_power(geom):
    """The map redistributes the TDP budget in space; it must not change it."""
    gen = ScenarioGenerator()
    scen = gen.generate_all_scenarios(geom)
    area = {b.name: (b.width * b.height) / 1e8
            for b in geom.power_blocks if not b.is_tsv_region}
    before = [sum(v * area.get(n, 0.0) for n, v in s.power_blocks.items()) for s in scen]

    gen.attach_power_maps(scen, geom, kind='mixed', resolution=32)

    for s, w0 in zip(scen, before):
        if w0 <= 0:
            continue
        after = sum(float(np.sum(m)) for m in s.power_map_by_layer.values())
        assert after == pytest.approx(w0, rel=0.02), (
            f"{s.name}: total power changed {w0:.2f} W -> {after:.2f} W")


def test_attaching_maps_respects_density_ceiling_and_cooling():
    geom = build_geometry1()
    gen = ScenarioGenerator()
    scen = gen.generate_all_scenarios(geom)
    gen.attach_power_maps(scen, geom, kind='mixed', resolution=32)

    cell_cm2 = (geom.die_length / 32) * (geom.die_width / 32) / 1e8
    for s in scen:
        for m in s.power_map_by_layer.values():
            peak = float(np.max(m)) / cell_cm2
            assert peak <= ScenarioGenerator.MAX_LOGIC_POWER_DENSITY_WCM2 + 1e-6, \
                f"{s.name}: {peak:.1f} W/cm2 exceeds the silicon ceiling"
        assert ScenarioGenerator.MIN_HTC <= s.htc <= ScenarioGenerator.MAX_HTC


def test_power_field_uses_the_map_not_the_blocks():
    """
    The exported power field must come from the map the simulator actually used.
    Rebuilding it from block means would store a coarse approximation of the
    source that produced the temperatures -- a silent train/target mismatch.
    """
    geom = build_geometry1()
    gen = ScenarioGenerator()
    scen = gen.generate_all_scenarios(geom)
    gen.attach_power_maps(scen, geom, kind='mixed', resolution=0)
    d = scen[0].to_dict()

    coords, _ = generate_coords_and_indices(geom, uniform_z=False)
    blocky = generate_power_density_field(coords, geom, d['power_blocks'])
    mapped = generate_power_density_field(
        coords, geom, d['power_blocks'], power_map_by_layer=d['power_map_by_layer'])

    assert len(np.unique(mapped)) > 100 * len(np.unique(blocky)), (
        "map-derived power field should be far richer than the block field")
    assert mapped.sum() > 0
