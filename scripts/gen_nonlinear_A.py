"""Regime A (per-block leakage feedback) data generator. Protocol: docs/protocol_nonlinear_A.md (+ amendments 11, 12).
Stages:  unit            3D-ICE unit-response matrices R(HTC) (4 blocks x 4 blocks), results/nonlinear_A/unit_response.json
         plan            draw scenarios (seed 20261201), solve the fixed point on R, assign splits -> results/nonlinear_A/plan.json
         run W N         worker W of N: one real 3D-ICE solve per converged scenario at the fixed-point power (v5 export),
                         self-consistency check on the exported field, optional real 3D-ICE Picard loop (validation subset)
Power values are densities in W/cm^2 (the pipeline's unit). T in C unless stated. Usage: python scripts/gen_nonlinear_A.py unit|plan|run W N
"""
import json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.core.geometry_builders import get_geometry_by_name
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario
from src.scenario.generator import ScenarioGenerator
from src.scenario.leakage_perblock import block_power, law_arrhenius, law_exp, picard
from src.simulators.ice_simulator import DEFAULT_ICE_EXECUTABLE, ICESimulator

OUT = REPO / 'results' / 'nonlinear_A'; OUT.mkdir(parents=True, exist_ok=True)
DATA = REPO / 'data' / '3d-ice-leakage-A' / 'geometry1'
SEED, TREF = 20261201, 85.0
HTCS = [2000.0, 3500.0, 6000.0, 10000.0, 20000.0]       # W/m^2K, discrete so that R is measured once per value
UNIT_D = 20.0                                             # W/cm^2 per block in the unit solves
GEOM = get_geometry_by_name('geometry1')
NAMES = [b.name for b in GEOM.power_blocks]
LAWS = {'exp': law_exp, 'arrhenius': law_arrhenius}

def source_z():
    z = 0.0
    for L in GEOM.layers:
        if L.name == 'die':
            return z + L.thickness / 2.0
        z += L.thickness
    raise RuntimeError('no die layer')

def block_masks(coords):
    zs = np.unique(coords[:, 2]); z0 = zs[np.argmin(np.abs(zs - source_z()))]
    on = np.isclose(coords[:, 2], z0)
    return [on & (coords[:, 0] >= b.x) & (coords[:, 0] < b.x + b.width) & (coords[:, 1] >= b.y) & (coords[:, 1] < b.y + b.height) for b in GEOM.power_blocks]

def block_T_c(coords, T_k):
    return np.array([T_k[m].mean() for m in block_masks(coords)]) - 273.15

def sim_in(tmp):
    (tmp / 'cfg').mkdir(); (tmp / 'sim').mkdir()
    return ICESimulator(tmp / 'cfg', tmp / 'sim', DEFAULT_ICE_EXECUTABLE)

def base_sc(htc, ta, d):
    sc = ScenarioGenerator().generate_all_scenarios(GEOM)[0].to_dict()
    sc.update(htc=float(htc), t_ambient=float(ta), power_blocks={n: float(v) for n, v in zip(NAMES, d)}, pattern='leakage_A')
    return sc

# ------------------------------------------------------------------------------------------------ unit responses
def stage_unit():
    res = {}
    f = OUT / 'unit_response.json'
    if f.exists(): res = json.loads(f.read_text())
    tmp = Path(tempfile.mkdtemp(prefix='nlA_unit_')); sim = sim_in(tmp)
    try:
        for htc in HTCS:
            key = str(int(htc))
            if key in res: continue
            ta = 25.0; cols = []; t0 = time.perf_counter()
            for j in range(len(NAMES)):
                d = np.zeros(len(NAMES)); d[j] = UNIT_D
                p = sim.simulate(GEOM, base_sc(htc, ta, d), f'unit_{key}_{j}')
                cols.append((block_T_c(p['coords'], p['temperature']) - ta) / UNIT_D)
            res[key] = dict(R=np.array(cols).T.tolist(), seconds=round(time.perf_counter() - t0, 1))
            f.write_text(json.dumps(res, indent=1)); print('unit', key, res[key]['seconds'], 's', flush=True)
        # linearity check: a two-block solve vs superposition
        d = np.array([UNIT_D, 0, UNIT_D * 0.5, 0.0]); key = str(int(HTCS[0]))
        p = sim.simulate(GEOM, base_sc(HTCS[0], 25.0, d), 'lincheck')
        got = block_T_c(p['coords'], p['temperature']); pred = 25.0 + np.array(res[key]['R']) @ d
        res['linearity_check_max_abs_K'] = float(np.abs(got - pred).max()); f.write_text(json.dumps(res, indent=1))
        print('linearity check', res['linearity_check_max_abs_K'], flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ plan
def load_R():
    u = json.loads((OUT / 'unit_response.json').read_text())
    return {float(k): np.array(v['R']) for k, v in u.items() if k.isdigit()}

def fixed_point(R, ta, p_dyn, p_leak, kd, law, tol=1e-12, max_iter=20000):
    """Minimal fixed point by monotone iteration from the dynamic power; None if it runs away."""
    d = p_dyn.copy()
    for _ in range(max_iter):
        T = ta + R @ d
        if T.max() > 300.0: return None
        dn = block_power(T, p_dyn, p_leak, TREF, kd, law)
        if np.max(np.abs(dn - d) / dn) < tol: return dn, ta + R @ dn
        d = dn
    return None

def rho(R, T, p_leak, kd, law):
    from src.scenario.leakage_perblock import dlaw_dt
    return float(np.max(np.abs(np.linalg.eigvals(np.diag(p_leak * dlaw_dt(law, T, TREF, kd)) @ R))))

def draw_candidate(rng, Rs):
    htc = float(rng.choice(HTCS)); R = Rs[htc]; ta = float(rng.uniform(20, 60))
    rel = rng.dirichlet(np.full(4, 2.0)) * 4.0                          # block pattern, mean 1
    share = rng.uniform(0.10, 0.35, 4) * rng.uniform(0.8, 1.2, 4); kd = float(rng.uniform(10, 45))
    t_target = float(rng.uniform(70, 130))
    def at(s, law=law_exp):
        nom = s * rel; return nom, fixed_point(R, ta, nom * (1 - share), nom * share, kd, law)
    lo, hi = np.log(0.05), np.log(400.0)
    for _ in range(45):                                                   # bisection on the mean density scale
        mid = 0.5 * (lo + hi); nom, fp = at(np.exp(mid))
        if fp is None or fp[1].max() > t_target: hi = mid
        else: lo = mid
    s = float(np.exp(lo)); nom, fp = at(s)
    if fp is None or abs(fp[1].max() - t_target) > 1.0 or nom.max() > 250.0: return None
    return dict(htc=htc, t_ambient=ta, k_d=kd, share=share.tolist(), rel=rel.tolist(), scale=s, nominal=nom.tolist(), t_target=t_target)

def finish(c, Rs, law_name):
    R = Rs[c['htc']]; law = LAWS[law_name]; nom = np.array(c['nominal']); share = np.array(c['share'])
    p_leak, p_dyn = nom * share, nom * (1 - np.array(share)); fp = fixed_point(R, c['t_ambient'], p_dyn, p_leak, c['k_d'], law)
    if fp is None: return None
    d, T = fp
    pic = picard(lambda P: c['t_ambient'] + R @ P, p_dyn, p_leak, TREF, c['k_d'], law, tol=1e-3, max_iter=200)
    dT_nom = float((R @ nom).max()); dT_fp = float((T - c['t_ambient']).max())
    return dict(c, law=law_name, delivered=d.tolist(), T_fp=T.tolist(), rho=rho(R, T, p_leak, c['k_d'], law), picard_iterations=pic.iterations,
                picard_converged=pic.converged, nonlin_pct=100.0 * (dT_fp - dT_nom) / dT_nom, peak_rise_nominal=dT_nom, peak_rise_fp=dT_fp)

def stage_plan():
    Rs = load_R(); rng = np.random.default_rng(SEED)
    quota = {('train', 0): 80, ('train', 1): 80, ('train', 2): 80, ('iid', 0): 10, ('iid', 1): 10, ('iid', 2): 10}   # rho bins [.1,.3) [.3,.5) [.5,.7]
    ood_gain, ood_law, near, runaway = [], [], [], []
    items, tried, t0 = [], 0, time.perf_counter()
    def rbin(r): return None if r < 0.1 else 0 if r < 0.3 else 1 if r < 0.5 else 2 if r <= 0.7 else None
    while True:
        done = (all(v == 0 for v in quota.values()) and len(ood_gain) >= 30 and len(ood_law) >= 30 and len(near) >= 30 and len(runaway) >= 30)
        if done or tried > 400000: break
        tried += 1; c = draw_candidate(rng, Rs)
        if c is None: continue
        r = finish(c, Rs, 'exp')
        if r is None: continue
        b = rbin(r['rho'])
        if b is not None:
            for split in ('train', 'iid'):
                if quota[(split, b)] > 0:
                    quota[(split, b)] -= 1; items.append(dict(r, split=split)); break
            else:
                if len(ood_law) < 30:                                       # surplus converged scenario in the training rho range: re-solved under the second law
                    r2 = finish(c, Rs, 'arrhenius')
                    if r2 is not None and rbin(r2['rho']) is not None: ood_law.append(dict(r2, split='ood_law')); items.append(ood_law[-1])
        elif 0.7 < r['rho'] <= 0.9 and len(ood_gain) < 30:
            ood_gain.append(dict(r, split='ood_gain')); items.append(ood_gain[-1])
        elif 0.9 < r['rho'] < 0.97 and len(near) < 30:
            near.append(dict(r, split='near_critical')); items.append(near[-1])
        if tried % 2000 == 0: print('tried', tried, quota, len(ood_gain), len(ood_law), len(near), len(runaway), f'{time.perf_counter()-t0:.0f}s', flush=True)
    # runaway set: take an accepted candidate and raise the scale until no fixed point exists
    rr = np.random.default_rng(SEED + 1)
    while len(runaway) < 30:
        c = draw_candidate(rr, Rs)
        if c is None: continue
        nom = np.array(c['nominal']); share = np.array(c['share']); R = Rs[c['htc']]
        for f in (1.2, 1.5, 2.0, 3.0, 5.0):
            n2 = nom * f
            if fixed_point(R, c['t_ambient'], n2 * (1 - share), n2 * share, c['k_d'], law_exp) is None:
                runaway.append(dict(c, nominal=n2.tolist(), law='exp', split='runaway')); items.append(runaway[-1]); break
    for i, it in enumerate(items): it['name'] = f"geometry1_nlA_{it['split']}_{i:03d}"
    (OUT / 'plan.json').write_text(json.dumps(items, indent=1))
    from collections import Counter
    print('planned', Counter(i['split'] for i in items), 'tried', tried, flush=True)
    for sp in ('train', 'iid', 'ood_gain', 'ood_law', 'near_critical'):
        v = [i['nonlin_pct'] for i in items if i['split'] == sp]
        if v: print(sp, 'nonlin% median %.1f range %.1f..%.1f' % (np.median(v), min(v), max(v)), 'rho', '%.2f..%.2f' % (min(i['rho'] for i in items if i['split'] == sp), max(i['rho'] for i in items if i['split'] == sp)), flush=True)

# ------------------------------------------------------------------------------------------------ run
def stage_run(w, n):
    plan = json.loads((OUT / 'plan.json').read_text()); DATA.mkdir(parents=True, exist_ok=True)
    log = OUT / f'run_{w}.jsonl'
    exporter, calc = NPZExporter(output_dir=DATA), StatisticsCalculator()
    mine = [it for k, it in enumerate(plan) if k % n == w and it['split'] != 'runaway']
    for k, it in enumerate(mine):
        f = DATA / f"{it['name']}.npz"
        if f.exists(): continue
        tmp = Path(tempfile.mkdtemp(prefix=f'nlA_{w}_')); sim = sim_in(tmp)
        try:
            t0 = time.perf_counter(); law = LAWS[it['law']]; share = np.array(it['share']); nom = np.array(it['nominal'])
            sc = base_sc(it['htc'], it['t_ambient'], it['delivered']); sc['leakage_nominal_power_blocks'] = {nm: float(v) for nm, v in zip(NAMES, nom)}
            process_scenario(it['name'], sc, GEOM, sim, exporter, calc, DATA, generate_plots=False)
            t_solve = time.perf_counter() - t0
            z = dict(np.load(f, allow_pickle=True)); coords = z['coords'].astype(np.float64)
            T_exp = block_T_c(coords, z['temp'].astype(np.float64))
            P_back = block_power(T_exp, nom * (1 - share), nom * share, TREF, it['k_d'], law)
            selfcons = float(np.max(np.abs(P_back - np.array(it['delivered'])) / np.array(it['delivered'])))
            meta = z['metadata'][0]
            meta.update(regime='leakage_A', split=it['split'], leakage_law=it['law'], leakage_k_double_c=it['k_d'], leakage_t_ref_c=TREF, leakage_share_blocks=str(it['share']),
                        loop_rho=it['rho'], picard_iterations=it['picard_iterations'], picard_converged=it['picard_converged'], nonlin_pct=it['nonlin_pct'],
                        selfconsistency_rel=selfcons, block_T_export_c=str([float(x) for x in T_exp]), block_T_R_c=str(it['T_fp']),
                        nominal_block_power_note='nominal = dynamic + leakage at Tref; block_power_* = delivered at the fixed point')
            z['metadata'] = np.array([meta], dtype=object); np.savez_compressed(f, **z)
            rec = dict(name=it['name'], split=it['split'], selfcons=selfcons, dT_R_vs_export_K=float(np.abs(T_exp - np.array(it['T_fp'])).max()), seconds=round(t_solve, 1))
            # validation subset: the real 3D-ICE Picard loop from the nominal power
            if it['split'] in ('train', 'iid') and k % 30 == 0:
                calls = []
                def solve_fn(P):
                    s2 = base_sc(it['htc'], it['t_ambient'], P); t1 = time.perf_counter()
                    p = sim.simulate(GEOM, s2, 'loop'); calls.append(time.perf_counter() - t1)
                    return block_T_c(p['coords'], p['temperature'])
                tl = time.perf_counter()
                res = picard(solve_fn, nom * (1 - share), nom * share, TREF, it['k_d'], law, tol=1e-3, max_iter=200)
                rec.update(loop_ice_iterations=res.iterations, loop_ice_converged=res.converged, loop_ice_seconds=round(time.perf_counter() - tl, 1),
                           loop_ice_per_solve_s=round(float(np.median(calls)), 1), loop_ice_peak_C=float(res.T.max()),
                           loop_ice_dT_vs_fixedpoint_K=float(np.abs(res.T - np.array(it['T_fp'])).max()), loop_ice_residual=res.residual)
            with open(log, 'a') as fh: fh.write(json.dumps(rec) + '\n')
            print(w, k + 1, '/', len(mine), rec, flush=True)
        except Exception as e:
            print('ERR', it['name'], repr(e)[:300], flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'unit': stage_unit()
    elif st == 'plan': stage_plan()
    elif st == 'run': stage_run(int(sys.argv[2]), int(sys.argv[3]))
