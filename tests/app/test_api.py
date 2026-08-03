import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_get_geometries(client):
    r = client.get('/geometries')
    assert r.status_code == 200
    data = r.json()
    # Must match the canonical registry rather than a hard-coded count, so adding
    # a geometry cannot leave the app silently unable to serve it (geometry6 was
    # missing from the app for exactly this reason).
    from src.core.geometry_builders import build_all_geometries
    expected = {g.name for g in build_all_geometries()}
    assert set(data) == expected
    assert data['geometry1']['layers'] == 6


def test_geometry_schema(client):
    r = client.get('/geometries/schema')
    assert r.status_code == 200
    assert 'schema' in r.json()


def test_submit_job(client):
    r = client.post('/jobs', json={
        'geometry': 'geometry1',
        'scenario_name': 'api_test_001',
        'scenario_params': {
            'power_blocks': {'block1': 2.0},
            'htc': 5000,
            't_ambient': 45.0,
            'pattern': 'uniform',
        }
    })
    assert r.status_code == 200
    data = r.json()
    assert 'job_id' in data
    assert data['status'] == 'pending'


def test_get_job(client):
    r = client.post('/jobs', json={
        'geometry': 'geometry1',
        'scenario_name': 'api_test_002',
        'scenario_params': {
            'power_blocks': {},
            'htc': 5000,
            't_ambient': 45.0,
            'pattern': 'uniform',
        }
    })
    job_id = r.json()['job_id']
    r2 = client.get(f'/jobs/{job_id}')
    assert r2.status_code == 200
    assert r2.json()['job_id'] == job_id


def test_list_jobs(client):
    r = client.get('/jobs')
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_cancel_unknown_job(client):
    r = client.delete('/jobs/doesnotexist')
    assert r.status_code == 400


def test_submit_unknown_geometry(client):
    r = client.post('/jobs', json={
        'geometry': 'geometry_fake',
        'scenario_name': 'bad_001',
        'scenario_params': {
            'power_blocks': {},
            'htc': 5000,
            't_ambient': 45.0,
            'pattern': 'uniform',
        }
    })
    assert r.status_code == 404
