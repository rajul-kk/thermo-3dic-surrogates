"""
JobQueue's simulator was previously hardcoded (ICESimulator constructed directly inside _run_job), so no test could exercise the real job pipeline
"""
import time
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.job_queue import JobQueue
from src.core.geometry_builders import build_geometry1


class _StubSimulator:
    """Matches ThermalSimulator.simulate()'s return shape; no 3D-ICE involved."""
    def __init__(self, config_dir, output_dir, executable):
        self.config_dir = config_dir
        self.output_dir = output_dir
        self.executable = executable

    def simulate(self, geometry, scenario, scenario_name):
        from src.core.mesh import generate_coords_and_indices
        coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
        temps = np.full(coords.shape[0], 350.0, dtype=np.float32)  # 76.85 C, uniform
        return {'coords': coords, 'temperature': temps}


def _wait_for_terminal(queue: JobQueue, job_id: str, timeout_s: float = 120.0):
    # Generous: the stub still runs the real pipeline, which took >20s when the full suite
    # shared the CPU with a training run (both tests timed out; each passes in ~15s alone).
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = queue.get(job_id)
        if job.status in ('done', 'failed'):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")


def test_stub_simulator_runs_the_real_pipeline_end_to_end(tmp_path):
    queue = JobQueue(
        output_base=tmp_path,
        ice_executable='unused-with-stub',
        simulator_factory=lambda cd, od, ex: _StubSimulator(cd, od, ex),
    )
    geom = build_geometry1()
    job_id = queue.submit(
        geometry_name=geom.name,
        scenario_params={'power_blocks': {'block1': 2.0}, 'htc': 5000.0,
                         't_ambient': 45.0, 'pattern': 'uniform'},
        scenario_name='di_test_001',
    )
    job = _wait_for_terminal(queue, job_id)

    assert job.status == 'done', job.error
    assert job.npz_path and Path(job.npz_path).exists()
    assert job.stats is not None
    # The stub returns a uniform 350K field -- confirms StatisticsCalculator
    # (real, not stubbed) actually ran on the stub's output.
    assert job.stats['hotspot']['peak_temperature_c'] == pytest.approx(350.0 - 273.15, abs=0.1)


def test_simulator_factory_receives_the_configured_executable(tmp_path):
    seen = {}

    def factory(config_dir, output_dir, executable):
        seen['executable'] = executable
        return _StubSimulator(config_dir, output_dir, executable)

    queue = JobQueue(output_base=tmp_path, ice_executable='my-custom-exe',
                     simulator_factory=factory)
    job_id = queue.submit(
        geometry_name='geometry1',
        scenario_params={'power_blocks': {}, 'htc': 5000.0, 't_ambient': 45.0,
                         'pattern': 'uniform'},
        scenario_name='di_test_002',
    )
    _wait_for_terminal(queue, job_id)
    assert seen['executable'] == 'my-custom-exe'


def test_default_factory_still_constructs_a_real_ice_simulator(tmp_path):
    """No factory passed -> falls back to the original ICESimulator behaviour."""
    from src.simulators.ice_simulator import ICESimulator
    queue = JobQueue(output_base=tmp_path, ice_executable='wsl /bin/true')
    sim = queue._simulator_factory(tmp_path / 'cfg', tmp_path / 'out', 'wsl /bin/true')
    assert isinstance(sim, ICESimulator)


def test_unknown_geometry_fails_the_job_cleanly(tmp_path):
    queue = JobQueue(
        output_base=tmp_path, ice_executable='unused',
        simulator_factory=lambda cd, od, ex: _StubSimulator(cd, od, ex),
    )
    job_id = queue.submit(
        geometry_name='no_such_geometry',
        scenario_params={'power_blocks': {}, 'htc': 5000.0, 't_ambient': 45.0,
                         'pattern': 'uniform'},
        scenario_name='di_test_003',
    )
    job = _wait_for_terminal(queue, job_id)
    assert job.status == 'failed'
    assert 'no_such_geometry' in job.error
