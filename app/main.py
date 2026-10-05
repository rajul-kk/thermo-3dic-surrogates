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
from app.models import FloorplanRequest, JobRequest, JobResponse, OptimiseRequest
from src.core.placement import grid_steps
from src.solver import thermal

ICE_EXECUTABLE = os.environ.get(
    'ICE_EXECUTABLE',
    'wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator',     # the build the benchmark data was made with
)
OUTPUT_BASE = Path('data/app')
MAX_MESH_AXIS = 500
MAX_MESH_XY = 250_000

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
        # Cap solve size: one oversized request would otherwise tie up the single worker for
        # hours. Limits sit ~18x above the largest built-in (56x248 in-plane, geometry7).
        if any(int(r) > MAX_MESH_AXIS for r in res) or int(res[0]) * int(res[1]) > MAX_MESH_XY:
            raise ValueError(
                f"mesh_resolution {res} too large: each axis <= {MAX_MESH_AXIS}, "
                f"nx*ny <= {MAX_MESH_XY}")

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


# ── Floorplanner: classical solver, no trained model (src/solver/thermal.py) ─

SILICON_LIMIT_WCM2 = 300.0      # README dataset bounds: logic ceiling
MEMORY_LIMIT_WCM2 = 8.0         # HBM / memory dies


def _floorplan_geometry(name: str):
    geom = queue.get_geometry(name)
    if geom is None:
        raise HTTPException(404, f"Geometry '{name}' not found")
    return geom


def _scenario(req: FloorplanRequest, geom) -> dict:
    valid = {b.name for b in geom.power_blocks}
    unknown = set(req.power_blocks) - valid
    if unknown:
        raise HTTPException(422, f"Unknown power block(s) {sorted(unknown)}. Valid blocks: {sorted(valid)}")
    if any(v < 0 for v in req.power_blocks.values()):
        raise HTTPException(422, 'power densities must be >= 0')
    return {'power_blocks': dict(req.power_blocks), 'htc': req.htc, 't_ambient': req.t_ambient, 'pattern': 'custom'}


def _place(geom, offsets):
    try:
        return thermal.place(geom, {k: tuple(v) for k, v in offsets.items()})
    except thermal.PlacementError as exc:
        raise HTTPException(422, str(exc))


@app.get('/floorplanner', include_in_schema=False)
async def floorplanner_page():
    return FileResponse(_static / 'floorplanner.html')


@app.get('/floorplan/{name}')
def floorplan(name: str):
    """Package outline, grid, draggable items, power blocks and layer stack of a geometry."""
    geom = _floorplan_geometry(name)
    fp = thermal.describe(geom)
    gx, gy = grid_steps(geom)
    owner = {b: m.id for m in fp.movables for b in m.blocks}
    return {
        'name': name, 'width_um': float(geom.die_width), 'length_um': float(geom.die_length),
        'grid_um': [gx, gy],
        'movables': [{'id': m.id, 'x': m.x, 'y': m.y, 'width': m.width, 'height': m.height,
                      'layers': m.layers, 'blocks': m.blocks} for m in fp.movables],
        'blocks': [{'name': b.name, 'x': b.x, 'y': b.y, 'width': b.width, 'height': b.height,
                    'layer': b.layer_name, 'movable': owner.get(b.name), 'area_cm2': b.area_cm2,
                    'tsv': bool(b.is_tsv_region),
                    'limit_wcm2': MEMORY_LIMIT_WCM2 if 'hbm' in b.name.lower() or 'chipb_d' in b.name.lower()
                    else SILICON_LIMIT_WCM2}
                   for b in geom.power_blocks],
        'layers': [{'name': l.name, 'thickness_um': float(l.thickness), 'k': float(l.k_thermal),
                    'active': bool(l.is_active)} for l in geom.layers],
    }


@app.post('/solve')
def solve_floorplan(req: FloorplanRequest):
    """Preview (layered solver, ~10-25 ms) or exact (preconditioned CG on the finite-volume system)."""
    geom = _floorplan_geometry(req.geometry)
    scen = _scenario(req, geom)
    placed, snapped = _place(geom, req.offsets)
    if req.layer and req.layer != 'hottest' and req.layer not in {l.name for l in geom.layers}:
        raise HTTPException(422, f"Unknown layer '{req.layer}'")
    try:
        sol = thermal.solve(placed, scen, req.mode)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    out = thermal.summarise(placed, sol, req.t_ambient, req.layer)
    warnings = []
    if req.mode == 'preview' and sol['preview_risk_layer']:
        warnings.append(f"Preview is unreliable here: layer '{sol['preview_risk_layer']}' has strong in-plane "
                        f"conductivity contrast between the heat sources and the sink. Use the exact solve.")
    if req.t_limit_c is not None and out['peak_c'] > req.t_limit_c:
        warnings.append(f"Peak {out['peak_c']:.1f} °C exceeds the {req.t_limit_c:.0f} °C limit "
                        f"(layer {out['hotspot']['layer']}).")
    out.update(mode=sol['mode'], iterations=sol['iterations'], residual=sol['residual'],
               power_w=sol['power_w'], seconds=sol['seconds'], offsets=snapped, warnings=warnings)
    return out


@app.post('/optimise')
def optimise_floorplan(req: OptimiseRequest):
    """Move chiplets to lower the peak temperature (hill-climb on the preview solver), then report the result."""
    geom = _floorplan_geometry(req.geometry)
    scen = _scenario(req, geom)
    _place(geom, req.offsets)                      # reject an invalid starting point with a clear message
    res = thermal.optimise(geom, scen, {k: tuple(v) for k, v in req.offsets.items()},
                           n_evals=req.evaluations, seed=req.seed)
    return res


@app.post('/export/3dice')
def export_3dice(req: FloorplanRequest):
    """The 3D-ICE input files (.stk, floorplans, layouts) for this placement, as a zip."""
    import tempfile
    import zipfile
    from src.simulators.ice_simulator import ICESimulator
    geom = _floorplan_geometry(req.geometry)
    scen = _scenario(req, geom)
    placed, _ = _place(geom, req.offsets)
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as td:
        cfg, out = Path(td) / 'configs', Path(td) / 'output'
        cfg.mkdir(); out.mkdir()
        ICESimulator(config_dir=cfg, output_dir=out, executable=ICE_EXECUTABLE).generate_config_files(placed, scen)
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(cfg.rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(cfg).as_posix())
    buf.seek(0)
    return StreamingResponse(buf, media_type='application/zip',
                             headers={'Content-Disposition': f'attachment; filename="{req.geometry}_3dice.zip"'})


@app.post('/signoff')
def signoff(req: FloorplanRequest):
    """Queue a real 3D-ICE run of this placement; poll /jobs/{job_id} for the result."""
    import uuid
    geom = _floorplan_geometry(req.geometry)
    scen = _scenario(req, geom)
    placed, snapped = _place(geom, req.offsets)
    tag = uuid.uuid4().hex[:8]
    placed.name = f'{req.geometry}_fp_{tag}'
    queue.register_custom(placed.name, placed)
    job_id = queue.submit(placed.name, scen, f'floorplan_{tag}')
    return {'job_id': job_id, 'geometry': placed.name, 'offsets': snapped}


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
    with np.load(job.npz_path) as data:   # closes the handle; an open npz blocks deletion on Windows
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
