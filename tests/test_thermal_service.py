"""The floorplanner's solver engine (src/solver/thermal.py): placement rules, preview vs exact, optimiser."""
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import grid_steps
from src.solver import thermal
from src.validation import fv_solver as fv

REPO = Path(__file__).resolve().parent.parent


def scenario(geom, logic=60.0, memory=6.0):
    blocks = {b.name: (memory if 'hbm' in b.name or 'chipB_d' in b.name else logic)
              for b in geom.power_blocks if not b.is_tsv_region}
    return {'power_blocks': blocks, 'htc': 10000.0, 't_ambient': 45.0}


@pytest.mark.parametrize('name', ['geometry1', 'geometry4', 'geometry5', 'geometry6'])
def test_describe_covers_every_power_block_once(name):
    geom = get_geometry_by_name(name)
    fp = thermal.describe(geom)
    owned = [b for m in fp.movables for b in m.blocks]
    assert sorted(owned) == sorted(b.name for b in geom.power_blocks)
    assert len({m.id for m in fp.movables}) == len(fp.movables)


def test_place_snaps_to_the_cell_grid_and_moves_a_stack_together():
    geom = get_geometry_by_name('geometry5')
    gx, gy = grid_steps(geom)
    fp = thermal.describe(geom)
    stack = max(fp.movables, key=lambda m: len(m.names))        # the HBM stack spans several layers
    placed, snapped = thermal.place(geom, {stack.id: (gx * 2 + 37.0, -gy * 1 + 11.0)})
    dx, dy = snapped[stack.id]
    assert dx == pytest.approx(2 * gx) and dy == pytest.approx(-gy)
    prints = {d.name: d for d in placed.die_footprints}
    for n in stack.names:                                       # every layer of the stack moved by the same amount
        assert prints[n].x == pytest.approx(stack.x + dx) and prints[n].y == pytest.approx(stack.y + dy)
        assert prints[n].x % gx == pytest.approx(0, abs=1e-6) and prints[n].y % gy == pytest.approx(0, abs=1e-6)


def test_place_rejects_overlap_out_of_bounds_and_unknown_items():
    geom = get_geometry_by_name('geometry4')
    a, b = thermal.describe(geom).movables[:2]
    with pytest.raises(thermal.PlacementError, match='overlaps'):
        thermal.place(geom, {a.id: (b.x - a.x, b.y - a.y)})    # drop a on top of b
    with pytest.raises(thermal.PlacementError):
        thermal.place(geom, {a.id: (geom.die_width * 2, 0.0)})
    with pytest.raises(thermal.PlacementError, match='unknown'):
        thermal.place(geom, {'no_such_chiplet': (0.0, 0.0)})


def test_preview_is_exact_on_a_laterally_uniform_stack():
    geom = get_geometry_by_name('geometry1')
    scen = scenario(geom)
    prev = thermal.solve(geom, scen, 'preview')
    exact = thermal.solve(geom, scen, 'exact')
    assert exact['iterations'] <= 2
    assert np.abs(prev['theta'] - exact['theta']).max() < 1e-6 * exact['theta'].max()
    assert prev['preview_risk_layer'] is None


def test_exact_matches_the_direct_finite_volume_solve_on_a_chiplet_package():
    geom = get_geometry_by_name('geometry4')
    scen = scenario(geom)
    exact = thermal.solve(geom, scen, 'exact')
    direct = fv.solve(geom, scen)['T'] - (scen['t_ambient'] + 273.15)
    assert exact['residual'] <= 1e-8
    assert np.abs(exact['theta'] - direct).max() < 1e-4
    # the preview is approximate here, but close at the peak (report §9.32: 0.06-0.15 K on real scenarios)
    prev = thermal.solve(geom, scen, 'preview')
    assert abs(prev['theta'].max() - direct.max()) < 0.03 * direct.max()


def test_solution_is_linear_in_power_and_conserves_energy():
    geom = get_geometry_by_name('geometry5')
    scen = scenario(geom)
    one = thermal.solve(geom, scen, 'exact')
    scen2 = dict(scen, power_blocks={k: 2 * v for k, v in scen['power_blocks'].items()})
    two = thermal.solve(geom, scen2, 'exact')
    assert np.allclose(two['theta'], 2 * one['theta'], rtol=1e-6, atol=1e-7)
    g = one['grid']
    k_lat, k_ver = fv.conductivity(geom, scen, g)
    _, g_bot = fv.assemble(g, k_lat, k_ver, scen['htc'])
    assert float((g_bot * one['theta'][:, :, 0]).sum()) == pytest.approx(one['power_w'], rel=1e-6)


def test_geometry7_bridge_layer_is_flagged_as_a_preview_risk():
    geom = get_geometry_by_name('geometry7')
    sol = thermal.solve(geom, scenario(geom), 'preview')
    assert sol['preview_risk_layer'] == 'substrate_organic'


def test_summarise_reports_the_hottest_cell_and_a_layer_map():
    geom = get_geometry_by_name('geometry5')
    scen = scenario(geom)
    sol = thermal.solve(geom, scen, 'preview')
    out = thermal.summarise(geom, sol, scen['t_ambient'])
    n_len, n_wid, _ = geom.mesh_resolution
    assert (out['nx'], out['ny']) == (n_wid, n_len)
    assert out['peak_c'] == pytest.approx(sol['theta'].max() + 45.0)
    assert out['layer'] == out['hotspot']['layer']
    assert out['field_max_c'] == pytest.approx(out['peak_c'], abs=0.01)
    assert 0 <= out['hotspot']['x_um'] <= geom.die_width and 0 <= out['hotspot']['y_um'] <= geom.die_length
    other = thermal.summarise(geom, sol, scen['t_ambient'], 'heat_sink')
    assert other['layer'] == 'heat_sink' and other['field_max_c'] < out['peak_c']


def test_optimise_never_makes_the_peak_worse_and_returns_a_valid_placement():
    geom = get_geometry_by_name('geometry4')
    scen = scenario(geom)
    res = thermal.optimise(geom, scen, n_evals=80, seed=1)
    assert res['rise_after_k'] <= res['rise_before_k'] + 1e-9
    placed, _ = thermal.place(geom, res['offsets'])            # must itself be a legal placement
    assert thermal.solve(placed, scen, 'preview')['theta'].max() == pytest.approx(res['rise_after_k'])


LAYOUT = REPO / 'data' / '3d-ice-layout-geometry5'


@pytest.mark.skipif(not list(LAYOUT.rglob('*.npz')), reason='no layout dataset present')
def test_exact_solve_reproduces_real_3d_ice_samples():
    """End to end against the ground truth: metadata -> placement -> exact solve -> 3D-ICE's own nodes."""
    import ast
    from src.hybrid import layered_backbone as lb
    base = get_geometry_by_name('geometry5')
    ids = thermal.describe(base).movables
    for f in sorted(LAYOUT.rglob('*.npz'))[:3]:
        d = np.load(f, allow_pickle=True)
        m = d['metadata'].item()
        offs = {mv.id: (float(m['placement_dx_' + mv.names[0]]), float(m['placement_dy_' + mv.names[0]])) for mv in ids}
        placed, snapped = thermal.place(base, offs)
        assert snapped == pytest.approx(offs)                   # the corrected data is already grid-aligned
        scen = {'power_blocks': {b.name: float(m.get(f'block_power_{b.name}', 0.0)) for b in placed.power_blocks},
                'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
                'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')}
        sol = thermal.solve(placed, scen, 'exact')
        y = d['temp'].astype(float)
        p = lb.sample(sol['theta'], sol['grid'], d['coords'].astype(float)) + float(m['t_ambient_kelvin'])
        assert abs(p.max() - y.max()) < 0.1, f.name
        assert np.abs(p - y).mean() < 0.02, f.name
