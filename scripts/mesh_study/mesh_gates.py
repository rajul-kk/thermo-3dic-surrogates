"""Mesh-convergence protocol gates G1-G3: reference-solver feasibility, 3D-ICE agreement, 3D-ICE at 50 um cells."""
import dataclasses, json, shutil, sys, threading, time
from pathlib import Path

REPO = Path(r'D:\Work\3D-ICE Thermal-modelling\Thermo')
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools' / 'iceforge'))
OUT = REPO / 'results' / 'mesh_convergence'; OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
try:
    import psutil
    proc = psutil.Process()
except Exception:
    psutil = None

class Peak:
    """Sample this process's RSS in a thread; report the peak in GB."""
    def __enter__(self):
        self.p = 0.0; self.stop = False
        def run():
            while not self.stop:
                if psutil: self.p = max(self.p, proc.memory_info().rss / 2**30)
                time.sleep(0.2)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self
    def __exit__(self, *a): self.stop = True; self.t.join()

from scripts.resolve_from_metadata import ICE_EXE, load_geometry
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator
from iceforge import refsolve as R
from iceforge.model import parse_stk
from iceforge import diff as D

geom = load_geometry('geometry1')
sc = [s for s in ScenarioGenerator().generate_all_scenarios(geom) if s.scenario_type not in ('leakage', 'throttle')][0].to_dict()
res = {'scenario': {k: sc[k] for k in ('htc', 't_ambient', 'pattern')}, 'mesh_native': list(geom.mesh_resolution)}

def make_stk(g, tag):
    """Write the geometry's .stk/.flp to OUT/<tag>, no solve."""
    d = OUT / tag; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    sim = ICESimulator(d, d / 'sim', ICE_EXE)
    sim.generate_config_files(g, sc)
    return d / 'stack.stk'

def solve_ice(g, tag):
    """3D-ICE solve through the project pipeline; return (seconds, peak_rise_proxy_C, err)."""
    tmp = OUT / (tag + '_run'); shutil.rmtree(tmp, ignore_errors=True); (tmp / 'cfg').mkdir(parents=True); (tmp / 'sim').mkdir()
    outd = OUT / (tag + '_npz'); shutil.rmtree(outd, ignore_errors=True); outd.mkdir()
    t0 = time.perf_counter()
    try:
        process_scenario('gate', sc, g, ICESimulator(tmp / 'cfg', tmp / 'sim', ICE_EXE), NPZExporter(output_dir=outd),
                         StatisticsCalculator(), outd, generate_plots=False)
        import numpy as np
        z = np.load(outd / 'gate.npz', allow_pickle=True)
        return time.perf_counter() - t0, float(z['temp'].max()), None
    except Exception as e:
        return time.perf_counter() - t0, None, repr(e)[:300]

# ---- G1: reference solver feasibility on the standard geometry1 stack (nz per layer capped)
stk = make_stk(geom, 'native')
model = R.build_model(parse_stk(str(stk)))
g1 = []
for nz, r in ((2, 1), (2, 2), (4, 2), (4, 4), (4, 8)):
    with Peak() as pk:
        t0 = time.perf_counter()
        try:
            s = R.solve(model, r=r, nz=nz); ok = True; info = s.info
            peak = float(s.T.max())
        except MemoryError as e:
            ok = False; info = repr(e)[:100]; peak = None
        except Exception as e:
            ok = False; info = repr(e)[:200]; peak = None
    row = dict(r=r, nz=nz, ok=ok, seconds=round(time.perf_counter() - t0, 1), peak_rss_gb=round(pk.p, 2), peak_K=peak,
               info=info if not ok else {k: info[k] for k in ('unknowns', 'method')})
    g1.append(row); log('G1', row)
    if not ok: break
res['G1'] = g1

# ---- G3: reference vs 3D-ICE at native resolution (known-good configuration r=2 nz=2)
with Peak() as pk:
    t0 = time.perf_counter()
    try:
        rep = D.run_diff(str(stk), r=2, nz=2)
        lay = [{k: c.get(k) for k in ('instance', 'verdict')} for c in rep['layers']]
        res['G3'] = dict(overall=rep['verdict'], layers=lay, seconds=round(time.perf_counter() - t0, 1), peak_rss_gb=round(pk.p, 2),
                         tol=rep['tol'])
    except Exception as e:
        res['G3'] = dict(error=repr(e)[:300])
log('G3', res['G3'])

# ---- G2: one 3D-ICE solve at native and at 50 um cells (200 x 200)
nat = solve_ice(geom, 'ice_native'); log('G2 native', nat)
g50 = dataclasses.replace(geom, mesh_resolution=(200, 200, geom.mesh_resolution[2]))
fine = solve_ice(g50, 'ice_50um'); log('G2 50um', fine)
res['G2'] = dict(native=dict(seconds=round(nat[0], 1), peak_C=nat[1], err=nat[2]), half_cell=dict(seconds=round(fine[0], 1), peak_C=fine[1], err=fine[2]))
(OUT / 'gates.json').write_text(json.dumps(res, indent=1, default=str)); log('DONE', json.dumps(res['G2']))
