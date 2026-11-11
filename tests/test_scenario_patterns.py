"""Scenario power patterns: the rules of _apply_pattern, and a fingerprint of everything the generator emits.

The fingerprint pins the training and test scenarios of every geometry (all entry points and the full extra pools),
so a change to the dataset's power settings is a deliberate edit of FINGERPRINT, never a side effect of a refactor.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import get_geometry_by_name
from src.scenario.generator import ScenarioGenerator

GEOMETRIES = ["geometry1", "geometry2a", "geometry3", "geometry4", "geometry5", "geometry6", "geometry7"]
# sha256 of all scenarios, floats rounded to 12 significant digits (recorded 2026-11-11, before and after the
# _apply_pattern refactor: identical).
FINGERPRINT = "e396272794f1699eca783a5ca1171f4b4971c6caea06119298e4501897a3ed3d"


def _rounded(x):
    if isinstance(x, float):
        return float(f"{x:.12g}")
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v) for v in x]
    return x


def all_scenarios():
    out = {}
    for n in GEOMETRIES:
        g = get_geometry_by_name(n)
        gen = ScenarioGenerator()
        pool = gen._EXTRA_POOL_2P5D if g.geometry_type == "2p5d_stack" else gen._EXTRA_POOL
        sc = gen.generate_all_scenarios(g) + gen.generate_extra_training_scenarios(g, len(pool))
        out[n] = [_rounded(dataclasses.asdict(s)) for s in sc]
    return out


def fingerprint():
    return hashlib.sha256(json.dumps(all_scenarios(), sort_keys=True, default=str).encode()).hexdigest()


def test_generator_output_unchanged():
    assert fingerprint() == FINGERPRINT, (
        "scenario generator output changed; if intended, update FINGERPRINT and say why in the commit")


@pytest.fixture()
def gen():
    return ScenarioGenerator()


def test_relative_patterns(gen):
    names = ["a", "b", "c", "d"]
    assert gen._relative_pattern(names, "hotspot", 2.0) == {"a": 2.0, "b": 0.2, "c": 0.2, "d": 0.2}
    assert gen._relative_pattern(names, "checkerboard", 2.0) == {"a": 2.0, "b": 1.0, "c": 2.0, "d": 1.0}
    assert gen._relative_pattern(names, "gradient", 3.0) == pytest.approx({"a": 1.5, "b": 2.0, "c": 2.5, "d": 3.0})
    assert gen._relative_pattern(["chipA_x", "hbm1_d1"], "split_chiplet_b_hot", 1.0) == {"chipA_x": 0.1, "hbm1_d1": 1.0}
    with pytest.raises(ValueError, match="Unknown pattern"):
        gen._relative_pattern(names, "bogus", 1.0)


def test_caps_and_rdl(gen):
    names = [b.name for b in get_geometry_by_name("geometry6").power_blocks if not b.is_tsv_region]
    pm = gen._apply_pattern(gen._budget(get_geometry_by_name("geometry6")), names, "uniform", 20.0,
                            rdl_joule_fraction=0.1)
    hbm = [n for n in names if n.startswith(gen.HBM_DIE_PREFIXES)]
    assert hbm and all(pm[n] <= gen.HBM_POWER_CAP_WCM2 for n in hbm)
    assert max(pm.values()) <= gen.MAX_LOGIC_POWER_DENSITY_WCM2
    logic = max(v for n, v in pm.items() if "rdl" not in n.lower())
    for n in names:
        if "rdl" in n.lower():
            assert pm[n] == pytest.approx(max(0.02, 0.1 * logic))


def test_budget_is_per_geometry_and_normalises_to_tdp():
    gen = ScenarioGenerator()
    g1, g6 = get_geometry_by_name("geometry1"), get_geometry_by_name("geometry6")
    b1, b6 = gen._budget(g1), gen._budget(g6)
    assert b1.tdp_w == gen.TDP_BY_GEOMETRY_W.get("geometry1", gen.TDP_DEFAULT_W)
    assert set(b6.block_area_cm2) == {b.name for b in g6.power_blocks if not b.is_tsv_region}
    # Generating another geometry in between cannot leak into a budget already built.
    gen.generate_training_scenarios(g6)
    names = list(b1.block_area_cm2)
    pm = gen._normalise_to_tdp(b1, {n: 1.0 for n in names}, 0.5)
    total = sum(p * b1.block_area_cm2[n] for n, p in pm.items())
    assert total == pytest.approx(b1.tdp_w * 0.5 * gen.CORE_FRACTION_OF_TDP)
