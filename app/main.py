from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.job_queue import Job, JobQueue
from app.models import JobRequest, JobResponse

ICE_EXECUTABLE = os.environ.get(
    'ICE_EXECUTABLE',
    'wsl /home/rajul/3d-ice/bin/3D-ICE-Emulator',
)
OUTPUT_BASE = Path('data/app')

queue = JobQueue(output_base=OUTPUT_BASE, ice_executable=ICE_EXECUTABLE)

app = FastAPI(title='3D-ICE Pipeline')

_static = Path(__file__).parent / 'static'
if _static.exists():
    app.mount('/static', StaticFiles(directory=str(_static)), name='static')


# ── HTML shell ─────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
async def index():
    return FileResponse(_static / 'index.html')


# ── Geometry routes ────────────────────────────────────────────────────────

GEOMETRY_YAML_TEMPLATE = """\
name: my_geometry
geometry_type: 2d_stack          # 2d_stack | 3d_stack | 2p5d_stack
die_length: 10000.0              # µm
die_width:  10000.0              # µm
mesh_resolution: [100, 100, 40]
layers:
  - name: heat_spreader
    thickness: 1000.0
    material: copper
    is_active: false
  - name: die
    thickness: 100.0
    material: silicon
    is_active: true
power_blocks:
  - name: core_block
    layer_name: die
    x: 0.0
    y: 0.0
    width: 10000.0
    height: 10000.0
"""


@app.get('/geometries')
async def list_geometries():
    return queue.list_geometries()


@app.get('/geometries/schema')
async def geometry_schema():
    return {'schema': GEOMETRY_YAML_TEMPLATE}


@app.post('/geometries', status_code=201)
async def register_geometry(body: dict):
    try:
        import yaml
    except ImportError:
        raise HTTPException(500, 'PyYAML not installed')
    try:
        raw = body.get('yaml', '')
        spec = yaml.safe_load(raw)
        if not isinstance(spec, dict):
            raise ValueError('YAML must parse to a mapping (see /geometries/schema)')
        name = str(spec.get('name', '')).strip()
        if not name:
            raise ValueError("'name' field is required")
        if name in queue.builtin_names():
            raise ValueError(
                f"'{name}' is a built-in geometry name and cannot be overridden "
                f"(built-ins: {sorted(queue.builtin_names())})"
            )

        from src.core.geometry import Geometry, Layer, PowerBlock
        from src.core.material import MaterialLibrary
        mat_lib = MaterialLibrary()

        layer_specs = spec.get('layers', [])
        if not layer_specs:
            raise ValueError("'layers' must contain at least one layer")

        layers = []
        for lspec in layer_specs:
            mat_name = lspec['material']
            mat = mat_lib.get(mat_name)
            if mat is None:
                raise ValueError(f"Unknown material '{mat_name}'. "
                                 f"Available: {mat_lib.list_materials()}")
            layers.append(Layer(
                name=lspec['name'],
                thickness=float(lspec['thickness']),
                material=mat_name,
                k_thermal=mat.k_thermal,
                volumetric_heat_capacity=mat.volumetric_heat_capacity,
                is_active=bool(lspec.get('is_active', False)),
            ))
        if not any(l.is_active for l in layers):
            raise ValueError("at least one layer must have is_active: true "
                             "(otherwise there is no power source to simulate)")

        blocks = []
        for bspec in spec.get('power_blocks', []):
            blocks.append(PowerBlock(
                name=bspec['name'],
                layer_name=bspec['layer_name'],
                x=float(bspec['x']),
                y=float(bspec['y']),
                width=float(bspec['width']),
                height=float(bspec['height']),
            ))

        res = spec.get('mesh_resolution', [100, 100, 40])
        if len(res) != 3 or any(int(r) <= 0 for r in res):
            raise ValueError(
                f"mesh_resolution must be 3 positive integers [nx, ny, nz], got {res}")

        geom = Geometry(
            name=name,
            geometry_type=spec.get('geometry_type', '2d_stack'),
            die_length=float(spec['die_length']),
            die_width=float(spec['die_width']),
            mesh_resolution=tuple(int(r) for r in res),
            layers=layers,
            power_blocks=blocks,
        )
        # Geometry() does not self-validate on construction (build_geometryN()
        # functions in geometry_builders.py call this explicitly) -- without it,
        # a geometry with duplicate layer names, power blocks outside the die
        # footprint, or a power block referencing a nonexistent layer would be
        # silently accepted here and only fail confusingly at job-submission time.
        geom.validate()

        queue.register_custom(name, geom)
        return {'name': name, 'layers': len(layers), 'power_blocks': len(blocks)}
    except yaml.YAMLError as exc:
        raise HTTPException(422, detail=f'Invalid YAML: {exc}')
    except (ValueError, KeyError, TypeError) as exc:
        # KeyError: a required field (e.g. 'die_length') missing from the spec.
        # TypeError: Geometry()/Layer()/PowerBlock() called with a wrong type.
        raise HTTPException(422, detail=str(exc))


# ── Job routes ─────────────────────────────────────────────────────────────

def _job_to_dict(job: Job) -> dict:
    return {
        'job_id':        job.id,
        'status':        job.status,
        'geometry':      job.geometry_name,
        'scenario_name': job.scenario_name,
        'created_at':    job.created_at,
        'finished_at':   job.finished_at,
        'stats':         job.stats,
        'error':         job.error,
        'log_count':     len(job.logs),
    }


@app.post('/jobs')
async def submit_job(req: JobRequest):
    geom = queue.get_geometry(req.geometry)
    if geom is None:
        raise HTTPException(404, f"Geometry '{req.geometry}' not found")

    # A misspelled/unknown block name doesn't error -- it silently delivers
    # zero power to every real block instead, which looks like a valid (if
    # oddly cool) result rather than a rejected request. Reject it instead.
    valid_blocks = {b.name for b in geom.power_blocks}
    unknown = set(req.scenario_params.power_blocks) - valid_blocks
    if unknown:
        raise HTTPException(
            422,
            f"Unknown power block(s) {sorted(unknown)} for geometry "
            f"'{req.geometry}'. Valid blocks: {sorted(valid_blocks)}"
        )

    job_id = queue.submit(
        req.geometry,
        req.scenario_params.model_dump(),
        req.scenario_name,
    )
    return JobResponse(job_id=job_id, status='pending')


@app.get('/jobs')
async def list_jobs():
    return [_job_to_dict(j) for j in queue.list()]


@app.get('/jobs/{job_id}')
async def get_job(job_id: str):
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404)
    return _job_to_dict(job)


@app.delete('/jobs/{job_id}')
async def cancel_job(job_id: str):
    if not queue.cancel(job_id):
        raise HTTPException(400, 'Job cannot be cancelled (not pending)')
    return {'status': 'cancelled'}


# ── WebSocket: live logs ───────────────────────────────────────────────────

@app.websocket('/ws/jobs/{job_id}/logs')
async def job_logs_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = queue.get(job_id)
    if not job:
        await websocket.close(1008)
        return
    sent = 0
    try:
        while True:
            new_lines = job.logs[sent:]
            for line in new_lines:
                await websocket.send_text(line)
            sent += len(new_lines)
            if job.status in ('done', 'failed'):
                await websocket.send_text(f'__STATUS__ {job.status}')
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ── Results routes ─────────────────────────────────────────────────────────

@app.get('/results/{job_id}.npz')
async def download_npz(job_id: str):
    job = queue.get(job_id)
    if not job or job.status != 'done' or not job.npz_path:
        raise HTTPException(404)
    return FileResponse(
        job.npz_path,
        media_type='application/octet-stream',
        filename=f'{job.scenario_name}.npz',
    )


@app.get('/results/{job_id}/heatmap.png')
async def heatmap_png(job_id: str, layer: int = 0):
    job = queue.get(job_id)
    if not job or job.status != 'done' or not job.npz_path:
        raise HTTPException(404)
    data = np.load(job.npz_path)
    coords = data['coords']
    temps = data['temp'] - 273.15    # K → °C
    layers = data['layer']
    mask = layers == layer
    if not np.any(mask):
        raise HTTPException(404, f'Layer {layer} not in this file')
    x, y, t = coords[mask, 0], coords[mask, 1], temps[mask]
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    sc = ax.scatter(x, y, c=t, cmap='hot', s=4, vmin=float(t.min()), vmax=float(t.max()))
    plt.colorbar(sc, ax=ax, label='Temperature (°C)')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(f'{job.scenario_name} — Layer {layer}')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')
