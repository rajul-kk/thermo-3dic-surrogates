"""End-to-end smoke test: submit → poll → NPZ download → heatmap PNG."""
import time
import numpy as np
import pytest
from pathlib import Path


def _fake_run_job(self, job_id):
    """Replaces JobQueue._run_job — writes a synthetic NPZ directly."""
    job = self._jobs[job_id]
    if job.status != 'pending':
        return
    job.status = 'running'
    try:
        out = self._output_base / job_id
        out.mkdir(parents=True, exist_ok=True)
        n = 60_000
        coords = (np.random.rand(n, 3) * 10_000).astype(np.float32)
        temp   = (np.random.rand(n) * 50 + 298).astype(np.float32)
        power  = np.zeros(n, dtype=np.float32)
        layer  = np.zeros(n, dtype=np.int32)
        npz_path = out / f"{job.scenario_name}.npz"
        np.savez(npz_path, coords=coords, temp=temp, power=power, layer=layer)
        job.npz_path = str(npz_path)
        job.stats = {
            'hotspot': {
                'peak_temperature_c': 85.3,
                'location_x_um': 1000.0,
                'location_y_um': 2000.0,
            }
        }
        job.logs.append('INFO Processing done (mock)')
        job.status = 'done'
    except Exception as exc:
        job.status = 'failed'
        job.error = str(exc)
    finally:
        job.finished_at = time.time()


def test_full_job_lifecycle(tmp_path):
    import app.main as main_mod
    from app.job_queue import JobQueue
    from fastapi.testclient import TestClient

    # Swap in a mock queue that skips 3D-ICE
    real_queue = main_mod.queue
    mock_queue = JobQueue(output_base=tmp_path, ice_executable='echo mock')
    import types
    mock_queue._run_job = types.MethodType(_fake_run_job, mock_queue)
    main_mod.queue = mock_queue

    try:
        with TestClient(main_mod.app) as client:
            # 1. Submit
            r = client.post('/jobs', json={
                'geometry': 'geometry1',
                'scenario_name': 'smoke_001',
                'scenario_params': {
                    'power_blocks': {'block1': 1.0},
                    'htc': 5000,
                    't_ambient': 45.0,
                    'pattern': 'uniform',
                }
            })
            assert r.status_code == 200, r.text
            job_id = r.json()['job_id']

            # 2. Poll until done (max 10 s)
            for _ in range(20):
                time.sleep(0.5)
                status = client.get(f'/jobs/{job_id}').json()
                if status['status'] in ('done', 'failed'):
                    break
            assert status['status'] == 'done', f"Job ended with: {status.get('error')}"
            assert status['stats']['hotspot']['peak_temperature_c'] == pytest.approx(85.3)

            # 3. Download NPZ
            r2 = client.get(f'/results/{job_id}.npz')
            assert r2.status_code == 200
            assert r2.headers['content-type'] == 'application/octet-stream'
            assert len(r2.content) > 1000

            # 4. Heatmap PNG
            r3 = client.get(f'/results/{job_id}/heatmap.png?layer=0')
            assert r3.status_code == 200
            assert r3.headers['content-type'] == 'image/png'
            assert r3.content[:4] == b'\x89PNG'

            # 5. List jobs includes our job
            all_jobs = client.get('/jobs').json()
            ids = [j['job_id'] for j in all_jobs]
            assert job_id in ids

    finally:
        main_mod.queue = real_queue
