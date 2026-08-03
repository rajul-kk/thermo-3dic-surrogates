"""
Structural checks on geometry definitions, and correctness checks on the
non-neural baselines and the seeding helper.

The baseline tests matter because `scripts/baselines.py` is the yardstick every
surrogate is measured against -- if ridge is subtly wrong, every comparison built
on it is wrong too.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_all_geometries, get_geometry_by_name
from src.reproducibility import set_seed
from scripts.baselines import fit_predict, metrics, feature_vector, collect_block_keys

ALL_GEOMS = build_all_geometries()
GEOM_NAMES = [g.name for g in ALL_GEOMS]


# ── Geometry ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('geom', ALL_GEOMS, ids=GEOM_NAMES)
def test_geometry_self_validates(geom):
    geom.validate()          # raises on structural inconsistency


@pytest.mark.parametrize('geom', ALL_GEOMS, ids=GEOM_NAMES)
def test_layers_are_contiguous_and_ordered(geom):
    """Layers must tile [0, total_height] with no gaps or overlaps."""
    z = 0.0
    for layer in geom.layers:
        assert layer.z_bottom == pytest.approx(z, abs=1e-6), (
            f"{geom.name}/{layer.name}: starts at {layer.z_bottom}, expected {z}")
        assert layer.z_top > layer.z_bottom, f"{geom.name}/{layer.name}: non-positive thickness"
        z = layer.z_top
    assert z == pytest.approx(geom.get_total_height(), abs=1e-6)


@pytest.mark.parametrize('geom', ALL_GEOMS, ids=GEOM_NAMES)
def test_every_layer_has_positive_conductivity(geom):
    for layer in geom.layers:
        assert layer.k_thermal > 0, f"{geom.name}/{layer.name}: k={layer.k_thermal}"


@pytest.mark.parametrize('geom', ALL_GEOMS, ids=GEOM_NAMES)
def test_at_least_one_active_layer(geom):
    assert any(l.is_active for l in geom.layers), f"{geom.name} has no active die layer"


@pytest.mark.parametrize('geom', ALL_GEOMS, ids=GEOM_NAMES)
def test_power_blocks_lie_inside_the_die_footprint(geom):
    for b in geom.power_blocks:
        assert b.x >= 0 and b.y >= 0, f"{geom.name}/{b.name}: negative origin"
        assert b.x + b.width <= geom.die_width + 1e-6, f"{geom.name}/{b.name}: overflows width"
        assert b.y + b.height <= geom.die_length + 1e-6, f"{geom.name}/{b.name}: overflows length"


@pytest.mark.parametrize('geom', ALL_GEOMS, ids=GEOM_NAMES)
def test_get_layer_at_z_covers_the_stack(geom):
    h = geom.get_total_height()
    for frac in (0.0, 0.25, 0.5, 0.75, 0.999):
        assert geom.get_layer_at_z(frac * h) is not None, f"{geom.name}: no layer at {frac}*h"


def test_registry_round_trips_every_geometry():
    """build_all_geometries() and get_geometry_by_name() must not drift apart."""
    for name in GEOM_NAMES:
        assert get_geometry_by_name(name).name == name


# ── Baselines ──────────────────────────────────────────────────────────────────

def _synthetic_scenarios(n_scen=25, n_pts=200, n_blocks=3, seed=0, noise=0.0):
    """
    Build scenarios whose temperature is an EXACT linear function of the block
    powers and ambient -- the regime steady-state conduction actually lives in.
    Ridge must recover this to machine precision.
    """
    rng = np.random.default_rng(seed)
    coords = rng.random((n_pts, 3)) * 1000.0
    impedance = rng.random((n_blocks, n_pts))        # A_b(x), scenario-independent
    out = []
    for i in range(n_scen):
        powers = rng.random(n_blocks) * 10.0
        amb = 298.15 + rng.random() * 20.0
        temp = amb + powers @ impedance
        if noise:
            temp = temp + rng.normal(0, noise, n_pts)
        meta = {f'block_power_b{j}': float(powers[j]) for j in range(n_blocks)}
        meta.update(htc=5000.0, t_ambient_kelvin=float(amb), tsv_density=0.0)
        out.append({'name': f'sc{i}', 'coords': coords, 'temp': temp, 'meta': meta})
    return out


def test_ridge_recovers_an_exactly_linear_field():
    scen = _synthetic_scenarios()
    train, test = scen[:20], scen[20:]
    keys = collect_block_keys(train + test)
    res = fit_predict(train, test, keys, k=3, ridge_lambda=1e-10)
    for m in res['ridge']:
        assert m['mae_K'] < 1e-6, f"ridge failed on a linear field: MAE={m['mae_K']:.3e}"
        assert m['spatial_r2'] > 0.999999


def test_mean_baseline_is_worse_than_ridge_on_linear_data():
    scen = _synthetic_scenarios()
    res = fit_predict(scen[:20], scen[20:], collect_block_keys(scen), k=3, ridge_lambda=1e-10)
    assert np.mean([m['mae_K'] for m in res['ridge']]) < \
           np.mean([m['mae_K'] for m in res['mean']])


def test_feature_vector_includes_reciprocal_htc():
    """
    Thermal resistance goes as 1/h, so 1/htc must be a feature -- without it the
    ridge baseline is a strawman and would understate the linear regime.
    """
    meta = {'block_power_a': 1.0, 'htc': 4000.0, 't_ambient_kelvin': 300.0, 'tsv_density': 0.0}
    fv = feature_vector(meta, ['block_power_a'])
    assert pytest.approx(1.0 / 4000.0) in [float(v) for v in fv]


def test_detrended_metric_ignores_a_constant_offset():
    """Adding a constant must inflate raw MAE but leave detrended error untouched."""
    rng = np.random.default_rng(1)
    coords, true = rng.random((100, 3)), rng.random(100)
    m0 = metrics(true.copy(), true, coords)
    m1 = metrics(true + 7.0, true, coords)
    assert m1['mae_K'] == pytest.approx(7.0, rel=1e-9)
    assert m1['mae_detrended_K'] == pytest.approx(m0['mae_detrended_K'], abs=1e-12)
    assert m1['mae_detrended_K'] < 1e-12


def test_detrended_metric_detects_a_spatial_error():
    """A genuine spatial distortion must show up in the detrended metric."""
    rng = np.random.default_rng(2)
    coords, true = rng.random((100, 3)), rng.random(100)
    m = metrics(true * 2.0, true, coords)          # scaling changes structure
    assert m['mae_detrended_K'] > 1e-3


# ── Reproducibility ────────────────────────────────────────────────────────────

def test_set_seed_makes_all_three_rngs_reproducible():
    import torch

    def draw():
        return (torch.randn(4).tolist(), np.random.rand(3).tolist(), random.random())

    set_seed(1234); a = draw()
    set_seed(1234); b = draw()
    set_seed(4321); c = draw()
    assert a == b, "same seed must reproduce"
    assert a != c, "different seeds must diverge"
