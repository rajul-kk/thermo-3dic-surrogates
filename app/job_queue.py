from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_builders import build_all_geometries
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.simulators.ice_simulator import ICESimulator
from src.main import process_scenario


@dataclass
class Job:
    id: str
    status: Literal['pending', 'running', 'done', 'failed']
    geometry_name: str
    scenario_params: dict
    scenario_name: str
    npz_path: Optional[str] = None
    stats: Optional[dict] = None
    logs: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None


class _JobLogHandler(logging.Handler):
    """Captures log records emitted during a job run into job.logs."""

    def __init__(self, job: Job):
        super().__init__()
        self._job = job

    def emit(self, record: logging.LogRecord) -> None:
        self._job.logs.append(self.format(record))


class JobQueue:
    def __init__(self, output_base: Path, ice_executable: str):
        self._jobs: Dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._output_base = Path(output_base)
        self._ice_executable = ice_executable
        self._builtin = {geom.name: geom for geom in build_all_geometries()}
        self._custom: dict = {}

    # ── Geometry management ────────────────────────────────────────────────

    def get_geometry(self, name: str):
        return self._builtin.get(name) or self._custom.get(name)

    def register_custom(self, name: str, geom) -> None:
        self._custom[name] = geom

    def list_geometries(self) -> Dict[str, dict]:
        result = {}
        for name, geom in {**self._builtin, **self._custom}.items():
            result[name] = {
                'name': name,
                'stack_type': geom.geometry_type,
                'layers': len(geom.layers),
                'mesh_points': int(np.prod(geom.mesh_resolution)),
                'die_length_um': float(geom.die_length),
                'die_width_um': float(geom.die_width),
                'power_blocks': [b.name for b in geom.power_blocks],
                'custom': name in self._custom,
            }
        return result

    # ── Job management ─────────────────────────────────────────────────────

    def submit(self, geometry_name: str, scenario_params: dict,
               scenario_name: str) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            status='pending',
            geometry_name=geometry_name,
            scenario_params=scenario_params,
            scenario_name=scenario_name,
        )
        self._jobs[job_id] = job
        self._executor.submit(self._run_job, job_id)
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == 'pending':
            job.status = 'failed'
            job.error = 'Cancelled by user'
            job.finished_at = time.time()
            return True
        return False

    # ── Worker ─────────────────────────────────────────────────────────────

    def _run_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        if job.status != 'pending':
            return

        job.status = 'running'
        handler = _JobLogHandler(job)
        handler.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
        root = logging.getLogger()
        root.addHandler(handler)

        tmpdir = tempfile.mkdtemp(prefix=f'ice_job_{job_id}_')
        try:
            geom = self.get_geometry(job.geometry_name)
            if geom is None:
                raise ValueError(f"Unknown geometry: {job.geometry_name}")

            output_dir = self._output_base / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            config_dir = Path(tmpdir) / 'configs'
            sim_out = Path(tmpdir) / 'output'
            config_dir.mkdir()
            sim_out.mkdir()

            simulator = ICESimulator(
                config_dir=config_dir,
                output_dir=sim_out,
                executable=self._ice_executable,
            )
            exporter = NPZExporter(output_dir=output_dir)
            calculator = StatisticsCalculator()

            stats = process_scenario(
                scenario_name=job.scenario_name,
                scenario_params=job.scenario_params,
                geometry=geom,
                simulator=simulator,
                exporter=exporter,
                calculator=calculator,
                output_dir=output_dir,
                generate_plots=False,
            )

            job.npz_path = str(output_dir / f"{job.scenario_name}.npz")
            job.stats = stats
            job.status = 'done'

        except Exception as exc:
            job.status = 'failed'
            job.error = str(exc)
            job.logs.append(f'ERROR {exc}')
        finally:
            job.finished_at = time.time()
            root.removeHandler(handler)
            shutil.rmtree(tmpdir, ignore_errors=True)
