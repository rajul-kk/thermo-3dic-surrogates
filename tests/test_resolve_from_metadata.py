"""scripts/resolve_from_metadata.py without a 3D-ICE solve: the simulator is replaced by a stub that returns the
stored grid and temperature, so what is tested is the scenario the script would hand to 3D-ICE and the arrays it writes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.resolve_from_metadata as rfm
from src.core.geometry_builders import build_geometry1, build_geometry2a
from src.core.mesh import generate_coords_and_indices
from src.export.npz_exporter import NPZExporter
from src.scenario.generator import ScenarioGenerator
from src.simulators import ice_simulator

PINNED = '4953952a1ef6d38807ff307212a6f15e5b2ef935'


@pytest.fixture
def fake_sim(monkeypatch):
    """ICESimulator.simulate -> the grid and temperature of the file being re-solved; records each scenario."""
    seen = []
    state = {}

    def simulate(self, geometry, scenario, name):
        seen.append({'geometry': geometry, 'scenario': scenario, 'name': name})
        ice_simulator._LAST_RUN.update({'ice_commit': PINNED, 'ice_executable': self.executable})
        return {'coords': state['coords'].astype(np.float64), 'temperature': state['temp'].astype(np.float64)}

    monkeypatch.setattr(rfm.ICESimulator, 'simulate', simulate)
    monkeypatch.setattr(ice_simulator, '_LAST_RUN', {})
    seen_state = state
    seen_state['seen'] = seen
    return seen_state


def _coarse_coords(geometry):
    """One node per cell centre in every layer's mid-plane (the full mesh would be slow to build in a test)."""
    nx, ny = int(geometry.mesh_resolution[0]), int(geometry.mesh_resolution[1])
    x = (np.arange(ny) + 0.5) * geometry.die_width / ny
    y = (np.arange(nx) + 0.5) * geometry.die_length / nx
    xx, yy = np.meshgrid(x, y)
    return np.concatenate([np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.5 * (l.z_bottom + l.z_top))])
                           for l in geometry.layers])


def _write_file(tmp_path, geometry, name, params, *, temp_value=330.0, coords=None):
    if coords is None:
        coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    temp = np.full(len(coords), temp_value, dtype=np.float32)
    src = tmp_path / 'src' / geometry.name
    src.mkdir(parents=True, exist_ok=True)
    out = NPZExporter(src).export_scenario(name, geometry, params, temp, coords=coords)
    return out, coords, temp


def _run(fake_sim, f, coords, temp, tmp_path):
    fake_sim['coords'], fake_sim['temp'] = coords, temp
    out = tmp_path / 'out'
    rfm.main([str(tmp_path / 'src'), '--out', str(out)])
    d = np.load(out / f.parent.name / f.name, allow_pickle=True)
    return {k: d[k] for k in d.files}, d['metadata'].item()


def _block_params(geometry):
    return {'power_blocks': {b.name: 5.0 for b in geometry.power_blocks if not b.is_tsv_region},
            'htc': 5000.0, 't_ambient': 40.0, 'pattern': 'uniform'}


def test_output_metadata_records_the_verified_build(fake_sim, tmp_path):
    g = build_geometry1()
    f, coords, temp = _write_file(tmp_path, g, 'geometry1_test_001', _block_params(g))
    arrays, m = _run(fake_sim, f, coords, temp, tmp_path)
    assert m['ice_commit'] == PINNED
    assert m['ice_executable'] == rfm.ICE_EXE
    assert m['resolved_v5_active_split'] is True
    assert np.array_equal(arrays['power'], np.load(f, allow_pickle=True)['power'])      # block-scalar data: unchanged


def test_microchannel_geometry_is_rebuilt_with_its_cooling(fake_sim, tmp_path):
    g = rfm.load_geometry(rfm.MICROCHANNEL_GEOMETRY)
    assert g.name == 'geometry6_microchannel' and g.layers[0].name == 'cp_base'
    f, coords, temp = _write_file(tmp_path, g, 'geometry6_microchannel_train_007', _block_params(g))
    _run(fake_sim, f, coords, temp, tmp_path)
    (call,) = fake_sim['seen']
    assert call['geometry'].coolant_layer_name == 'heat_sink'
    assert call['scenario']['cooling_mode'] == 'microchannel_2rm'
    assert call['scenario']['microchannel']['flow_rate_ml_min'] == 120.0          # index 7 % 6 = 1


def test_other_geometries_get_no_extra_scenario_keys():
    assert rfm.extra_scenario_keys({'geometry': 'geometry1'}, 'geometry1_train_007') == {}


def _mapped_params(geometry, index=3):
    """A geometry's scenario with the per-cell power maps the main generator attaches (kind 'mixed')."""
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geometry)
    gen.attach_power_maps(scenarios, geometry, kind='mixed')
    return scenarios[index].to_dict()


def test_power_maps_are_recovered_and_power_is_unchanged(fake_sim, tmp_path):
    g = build_geometry1()
    params = _mapped_params(g)
    f, coords, temp = _write_file(tmp_path, g, 'geometry1_train_004', params)
    stored = np.load(f, allow_pickle=True)['power']
    assert len(np.unique(stored)) > 1000                      # genuinely sub-block structure
    arrays, m = _run(fake_sim, f, coords, temp, tmp_path)
    (call,) = fake_sim['seen']
    got = call['scenario']['power_map_by_layer']
    assert set(got) == {'die'}
    np.testing.assert_allclose(got['die'], params['power_map_by_layer']['die'], rtol=1e-6)    # float32 storage
    assert np.array_equal(arrays['power'], stored)             # rebuilt power equals the stored power bit for bit
    assert m['resolved_power_map_recovered'] is True


def test_block_scalar_data_gets_no_power_maps(fake_sim, tmp_path):
    g = build_geometry1()
    f, coords, temp = _write_file(tmp_path, g, 'geometry1_test_001', _block_params(g))
    _, m = _run(fake_sim, f, coords, temp, tmp_path)
    assert 'power_map_by_layer' not in fake_sim['seen'][0]['scenario']
    assert 'resolved_power_map_recovered' not in m


def test_stale_block_scalar_power_is_not_mistaken_for_a_map(tmp_path):
    g = build_geometry1()
    f, coords, temp = _write_file(tmp_path, g, 'geometry1_test_001', _block_params(g))
    d = np.load(f, allow_pickle=True)
    arrays = {k: d[k] for k in d.files}
    arrays['power'] = arrays['power'] * 1.5                    # differs from the rebuild, but still piecewise constant
    m = arrays['metadata'].item()
    blocks = {b.name: m[f'block_power_{b.name}'] for b in g.power_blocks}
    assert rfm.stored_power_maps(arrays, g, blocks) == {}


def test_power_maps_inconsistent_with_block_metadata_are_refused(fake_sim, tmp_path):
    g = build_geometry1()
    params = _mapped_params(g)
    params['power_blocks'] = {k: v * 2 for k, v in params['power_blocks'].items()}   # metadata no longer matches the map
    f, coords, temp = _write_file(tmp_path, g, 'geometry1_train_004', params)
    # the exporter samples the maps, so the stored power is still the map; only block_power_* is doubled
    with pytest.raises(ValueError, match='recovered map holds'):
        _run(fake_sim, f, coords, temp, tmp_path)


@pytest.mark.skipif(not Path('data/3d-ice/geometry4').exists(), reason='shipped data not present')
def test_shipped_geometry4_power_is_reproduced():
    for f in sorted(Path('data/3d-ice/geometry4').glob('*.npz'))[::9]:
        d = np.load(f, allow_pickle=True)
        arrays = {k: d[k] for k in d.files}
        m = arrays['metadata'].item()
        geom = rfm.load_geometry(m['geometry'])
        blocks = {b.name: float(m[f'block_power_{b.name}']) for b in geom.power_blocks}
        maps = rfm.stored_power_maps(arrays, geom, blocks)
        rebuilt = rfm.generate_power_density_field(arrays['coords'].astype(np.float64), geom, blocks,
                                                   power_map_by_layer=maps)
        assert np.array_equal(rebuilt.astype(np.float32), arrays['power'])


def _tsv_params(geometry, index=4):
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geometry)
    gen.attach_tsv_maps(scenarios, geometry)
    gen.attach_power_maps(scenarios, geometry, kind='mixed')
    return scenarios[index].to_dict()


def test_tsv_maps_are_carried_and_regenerated_from_the_seed(fake_sim, tmp_path):
    g = build_geometry2a()
    params = _tsv_params(g)                                   # scenario index 4 -> file geometry2a_train_005
    f, coords, temp = _write_file(tmp_path, g, 'geometry2a_train_005', params, coords=_coarse_coords(g))
    stored = np.load(f, allow_pickle=True)
    assert stored['tsv_frac'].max() > 0
    arrays, m = _run(fake_sim, f, coords, temp, tmp_path)
    (call,) = fake_sim['seen']
    got = call['scenario']['tsv_map_by_layer']
    assert set(got) == set(params['tsv_map_by_layer']) == {'die2_tsv', 'die1_tsv'}
    for name, phi in params['tsv_map_by_layer'].items():
        assert np.array_equal(got[name], phi)                 # the generator's own float64 map, not the float32 readback
    assert np.array_equal(arrays['tsv_frac'], stored['tsv_frac'])
    assert np.array_equal(arrays['power'], stored['power'])
    assert m['resolved_tsv_map_recovered'] is True


def test_tsv_maps_fall_back_to_the_stored_array_without_a_seed(fake_sim, tmp_path):
    g = build_geometry2a()
    params = _tsv_params(g)
    f, coords, temp = _write_file(tmp_path, g, 'custom_name', params, coords=_coarse_coords(g))
    arrays, _ = _run(fake_sim, f, coords, temp, tmp_path)
    got = fake_sim['seen'][0]['scenario']['tsv_map_by_layer']
    for name, phi in params['tsv_map_by_layer'].items():
        np.testing.assert_allclose(got[name], phi, rtol=1e-6)
        assert not np.array_equal(got[name], phi)             # float32 readback, not the seeded map
    assert np.array_equal(arrays['tsv_frac'], np.load(f, allow_pickle=True)['tsv_frac'])


def test_tsv_seed_follows_the_generators_numbering():
    from scripts.resolve_maps import tsv_seed
    assert tsv_seed('geometry5_train_001') == (0, 0)
    assert tsv_seed('geometry5_train_015') == (0, 14)
    assert tsv_seed('geometry5_test_001') == (0, 15)
    assert tsv_seed('geometry5_train_016') == (900000, 0)
    assert tsv_seed('geometry5_train_023') == (900000, 7)
    assert tsv_seed('weird') is None


def test_files_without_tsv_maps_are_left_alone(fake_sim, tmp_path):
    g = build_geometry1()
    f, coords, temp = _write_file(tmp_path, g, 'geometry1_test_001', _block_params(g))
    arrays, m = _run(fake_sim, f, coords, temp, tmp_path)
    assert 'tsv_map_by_layer' not in fake_sim['seen'][0]['scenario']
    assert not arrays['tsv_frac'].any() and 'resolved_tsv_map_recovered' not in m
