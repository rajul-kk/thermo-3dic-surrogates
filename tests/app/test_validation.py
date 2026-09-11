"""Input validation for the app/ API."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


VALID_YAML = """\
name: test_custom_geom
geometry_type: 2d_stack
die_length: 10000.0
die_width: 10000.0
mesh_resolution: [50, 50, 20]
layers:
  - name: spreader
    thickness: 1000.0
    material: copper
    is_active: false
  - name: die
    thickness: 100.0
    material: silicon
    is_active: true
power_blocks:
  - name: core
    layer_name: die
    x: 0.0
    y: 0.0
    width: 10000.0
    height: 10000.0
"""


def test_valid_custom_geometry_is_registered(client):
    r = client.post('/geometries', json={'yaml': VALID_YAML})
    assert r.status_code == 201, r.text
    data = r.json()
    assert data['name'] == 'test_custom_geom'
    assert data['layers'] == 2
    assert data['power_blocks'] == 1

    listed = client.get('/geometries').json()
    assert listed['test_custom_geom']['custom'] is True


def test_malformed_yaml_rejected(client):
    r = client.post('/geometries', json={'yaml': 'name: [unterminated'})
    assert r.status_code == 422
    assert 'YAML' in r.json()['detail']


def test_non_mapping_yaml_rejected(client):
    r = client.post('/geometries', json={'yaml': '- just\n- a\n- list\n'})
    assert r.status_code == 422


def test_cannot_shadow_a_builtin_geometry_name(client):
    yaml_body = VALID_YAML.replace('test_custom_geom', 'geometry1')
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422
    assert 'built-in' in r.json()['detail']

    # geometry1 must still be the real one afterward, not shadowed/corrupted
    listed = client.get('/geometries').json()
    assert listed['geometry1']['custom'] is False
    assert listed['geometry1']['layers'] == 6


def test_missing_required_field_rejected(client):
    yaml_body = VALID_YAML.replace('die_length: 10000.0\n', '')
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422


def test_unknown_material_rejected(client):
    yaml_body = VALID_YAML.replace('material: copper', 'material: unobtainium')
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422
    assert 'unobtainium' in r.json()['detail']


def test_no_layers_rejected(client):
    yaml_body = """\
name: empty_geom
geometry_type: 2d_stack
die_length: 10000.0
die_width: 10000.0
layers: []
power_blocks: []
"""
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422
    assert 'at least one layer' in r.json()['detail']


def test_no_active_layer_rejected(client):
    yaml_body = VALID_YAML.replace('is_active: true', 'is_active: false')
    yaml_body = yaml_body.replace('name: test_custom_geom', 'name: no_active_geom')
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422
    assert 'is_active' in r.json()['detail']


def test_bad_mesh_resolution_length_rejected(client):
    yaml_body = VALID_YAML.replace(
        'mesh_resolution: [50, 50, 20]', 'mesh_resolution: [50, 50]')
    yaml_body = yaml_body.replace('test_custom_geom', 'bad_mesh_geom')
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422
    assert 'mesh_resolution' in r.json()['detail']


def test_negative_mesh_resolution_rejected(client):
    yaml_body = VALID_YAML.replace(
        'mesh_resolution: [50, 50, 20]', 'mesh_resolution: [50, -50, 20]')
    yaml_body = yaml_body.replace('test_custom_geom', 'neg_mesh_geom')
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422


def test_power_block_outside_die_bounds_rejected_via_geometry_validate(client):
    """geom.validate() (already existed, just wasn't being called) catches this."""
    yaml_body = VALID_YAML.replace('name: test_custom_geom', 'name: oob_block_geom')
    yaml_body = yaml_body.replace(
        'width: 10000.0\n    height: 10000.0',
        'width: 10000.0\n    height: 10000.0\n    x: 999999.0'
    )
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422


def test_duplicate_layer_names_rejected_via_geometry_validate(client):
    yaml_body = VALID_YAML.replace('name: test_custom_geom', 'name: dup_layer_geom')
    yaml_body = yaml_body.replace('name: spreader', 'name: die')  # now two layers named "die"
    r = client.post('/geometries', json={'yaml': yaml_body})
    assert r.status_code == 422


def test_submit_job_with_unknown_power_block_rejected(client):
    r = client.post('/jobs', json={
        'geometry': 'geometry1',
        'scenario_name': 'bad_block_001',
        'scenario_params': {
            'power_blocks': {'not_a_real_block': 5.0},
            'htc': 5000, 't_ambient': 45.0, 'pattern': 'uniform',
        }
    })
    assert r.status_code == 422
    assert 'not_a_real_block' in r.json()['detail']
    assert 'block1' in r.json()['detail']  # the real block names are listed


def test_submit_job_with_valid_and_unknown_power_block_mix_rejected(client):
    r = client.post('/jobs', json={
        'geometry': 'geometry1',
        'scenario_name': 'bad_block_002',
        'scenario_params': {
            'power_blocks': {'block1': 3.0, 'typo_block': 1.0},
            'htc': 5000, 't_ambient': 45.0, 'pattern': 'uniform',
        }
    })
    assert r.status_code == 422
