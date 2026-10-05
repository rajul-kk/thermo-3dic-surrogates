"""Floorplanner API: /floorplan, /solve, /optimise, /export/3dice, /signoff."""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def plan(client):
    return client.get('/floorplan/geometry5').json()


def request_for(plan, **kw):
    powers = {b['name']: (6.0 if b['limit_wcm2'] <= 10 else 60.0) for b in plan['blocks'] if not b['tsv']}
    return {'geometry': plan['name'], 'offsets': {}, 'power_blocks': powers, 'htc': 10000, 't_ambient': 45, **kw}


def test_page_and_static_assets_are_served(client):
    assert client.get('/floorplanner').status_code == 200
    for f in ('floorplanner.js', 'floorplanner.css'):
        assert client.get(f'/static/{f}').status_code == 200


def test_floorplan_describes_package_items_and_layers(client, plan):
    assert plan['width_um'] == 25000 and plan['length_um'] == 14000
    assert plan['grid_um'] == [250.0, 250.0]
    ids = {m['id'] for m in plan['movables']}
    assert all(b['movable'] in ids for b in plan['blocks'])
    assert any(l['active'] for l in plan['layers'])
    assert client.get('/floorplan/nope').status_code == 404


def test_preview_and_exact_agree_and_return_a_full_layer_map(client, plan):
    prev = client.post('/solve', json=request_for(plan)).json()
    exact = client.post('/solve', json=request_for(plan, mode='exact')).json()
    assert prev['mode'] == 'preview' and prev['iterations'] == 0
    assert exact['mode'] == 'exact' and exact['residual'] <= 1e-8
    assert abs(prev['peak_c'] - exact['peak_c']) < 1.0
    assert len(exact['field']) == exact['nx'] == 100 and len(exact['field'][0]) == exact['ny'] == 56
    assert exact['peak_c'] > 45 and exact['power_w'] > 0
    assert {l['name'] for l in exact['layers']} == {l['name'] for l in plan['layers']}


def test_moving_a_chiplet_is_snapped_and_changes_the_field(client, plan):
    base = client.post('/solve', json=request_for(plan)).json()
    m = plan['movables'][0]
    moved = client.post('/solve', json=request_for(plan, offsets={m['id']: [0.0, 310.0]}))
    assert moved.status_code == 200
    d = moved.json()
    assert d['offsets'][m['id']] == [0.0, 250.0]                 # snapped to the 250 µm cell grid
    assert d['field'] != base['field']


def test_invalid_requests_are_rejected_with_a_reason(client, plan):
    a, b = plan['movables'][:2]
    overlap = client.post('/solve', json=request_for(plan, offsets={a['id']: [b['x'] - a['x'], b['y'] - a['y']]}))
    assert overlap.status_code == 422 and 'overlaps' in overlap.json()['detail']
    out = client.post('/solve', json=request_for(plan, offsets={a['id']: [1e6, 0]}))
    assert out.status_code == 422
    bad_block = request_for(plan)
    bad_block['power_blocks']['not_a_block'] = 1.0
    assert client.post('/solve', json=bad_block).status_code == 422
    assert client.post('/solve', json=request_for(plan, layer='not_a_layer')).status_code == 422
    assert client.post('/solve', json=request_for(plan, mode='magic')).status_code == 422


def test_temperature_limit_produces_a_warning(client, plan):
    d = client.post('/solve', json=request_for(plan, t_limit_c=46.0)).json()
    assert any('exceeds' in w for w in d['warnings'])
    assert client.post('/solve', json=request_for(plan, t_limit_c=500.0)).json()['warnings'] == []


def test_optimise_returns_a_legal_placement_that_is_no_hotter(client, plan):
    r = client.post('/optimise', json=request_for(plan, evaluations=60))
    assert r.status_code == 200
    d = r.json()
    assert d['rise_after_k'] <= d['rise_before_k'] + 1e-9
    assert client.post('/solve', json=request_for(plan, offsets=d['offsets'])).status_code == 200


def test_export_gives_3d_ice_input_files(client, plan):
    r = client.post('/export/3dice', json=request_for(plan))
    assert r.status_code == 200 and r.headers['content-type'] == 'application/zip'
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert any(n.endswith('.stk') for n in names) and any(n.endswith('.flp') for n in names)


def test_signoff_queues_a_3d_ice_job_for_the_placed_geometry(client, plan):
    m = plan['movables'][0]
    r = client.post('/signoff', json=request_for(plan, offsets={m['id']: [0.0, 250.0]}))
    assert r.status_code == 200
    d = r.json()
    assert d['geometry'].startswith('geometry5_fp_')
    assert client.get(f"/jobs/{d['job_id']}").status_code == 200
