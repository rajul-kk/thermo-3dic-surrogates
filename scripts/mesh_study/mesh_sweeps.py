"""Mesh-convergence protocol sweeps M1 (discretisation), M2 (sub-cell hotspot), M3 (window size) on geometry1 (+ M1 on geometry5).
Protocol: docs/protocol_mesh_convergence.md (gates G1-G4 closed). Results: results/mesh_convergence/sweeps.json, sweeps.log."""
import json, math, re, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

REPO = Path(r'D:\Work\3D-ICE Thermal-modelling\Thermo')
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools' / 'iceforge'))
OUT = REPO / 'results' / 'mesh_convergence'
BASE = OUT / 'native'
_prev = OUT / 'sweeps.json'
RES = json.loads(_prev.read_text()) if _prev.exists() else {}
for _k in ('m1', 'm2', 'm3'): RES.setdefault(_k, {})
RES['m2'] = {}; RES['m3'] = {}          # the first M2/M3 attempt failed before any result (3D-ICE output dir missing); M1 reference results are kept
log = lambda *a: print(*a, flush=True)
def save(): (OUT / 'sweeps.json').write_text(json.dumps(RES, indent=1, default=float))

from iceforge import refsolve as R, diff as D
from iceforge.model import parse_stk
from scripts.resolve_from_metadata import ICE_EXE, load_geometry
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator

T0 = 298.15
def est_unknowns(model, r, nz):
    return int(model.chip_l / model.cell_l * r) * int(model.chip_w / model.cell_w * r) * len(model.layers) * (nz if isinstance(nz, int) else 12)

def make_case(tag, flp_text=None, cell=None, src=BASE, flp_name='floorplan_layer4.flp'):
    d = OUT / tag; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); (d / 'sim').mkdir()   # the stk writes its outputs to <dir>/sim
    stk = (src / 'stack.stk').read_text(encoding='utf8', errors='replace')
    stk = stk.replace(f'mesh_convergence/{src.name}/', f'mesh_convergence/{tag}/')
    if cell:
        stk = re.sub(r'cell length\s+[\d.]+\s*,\s*width\s+[\d.]+', f'cell length {cell} , width  {cell}', stk)
    (d / 'stack.stk').write_text(stk, encoding='utf8')
    for f in src.iterdir():
        if f.is_file() and f.name != 'stack.stk':
            shutil.copy(f, d / f.name)
    if flp_text is not None:
        (d / flp_name).write_text(flp_text, encoding='utf8')
    return d / 'stack.stk'

def ice_peak(stk, tag):
    t0 = time.perf_counter()
    st = parse_stk(str(stk))
    out = D.load_ice(st, None, 'auto', None, str(OUT / (tag + '_ice')), log=lambda *a: None)
    return max(float(d['T'].max()) for d in out.values()), round(time.perf_counter() - t0, 1)

def src_layer(model):
    return [i for i, L in enumerate(model.layers) if L.is_source][0]

def nz_fine(dz_src):
    def f(L):
        h = L.height
        if h <= 60: return max(2, math.ceil(h / dz_src))
        if h <= 150: return max(2, math.ceil(h / (2.5 * dz_src)))
        return 3
    return f

def read_blocks(flp_path):
    t = Path(flp_path).read_text(encoding='utf8', errors='replace')
    out = {}
    for m in re.finditer(r'(\w+)\s*:\s*position\s+([\d.]+)\s*,\s*([\d.]+)\s*;\s*dimension\s+([\d.]+)\s*,\s*([\d.]+)\s*;\s*power values\s+([\d.eE+-]+)\s*;', t):
        out[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 7)) + (float(m.group(6)),)
    return out

def element(name, x, y, w, h, p):
    return f"{name} :\n   position  {x:.4f}, {y:.4f} ;\n   dimension {w:.4f}, {h:.4f} ;\n   power values  {p:.9f} ;\n\n"

def hot_flp(s, m):
    B = read_blocks(BASE / 'floorplan_layer4.flp')
    bx, by, bw, bh, _, pb = B['block1']
    x0 = 2500.0 if s <= 100 else 2500.0 - s / 2
    Ab, Ah = bw * bh, float(s) * s
    p_hot = m * pb * Ah / Ab; p_rest = pb - p_hot
    assert 0 < p_hot < pb
    parts = [('hot', x0, x0, s, s, p_hot)]
    for nm, x, y, w, h in (('sl', bx, by, x0 - bx, bh), ('sr', x0 + s, by, bx + bw - x0 - s, bh),
                           ('sb', x0, by, s, x0 - by), ('st', x0, x0 + s, s, by + bh - x0 - s)):
        parts.append((nm, x, y, w, h, p_rest * (w * h) / (Ab - Ah)))
    assert abs(sum(p[3] * p[4] for p in parts) - Ab) < 1e-6
    txt = "// hotspot variant\n\n" + ''.join(element(*p) for p in parts)
    for k, v in B.items():
        if k != 'block1': txt += element(k, v[0], v[1], v[2], v[3], v[5])
    return txt, x0

def solve_ref(model, r, nz, tag):
    if isinstance(nz, int) and est_unknowns(model, r, nz) > 9.5e6:
        return None, f'skipped: ~{est_unknowns(model, r, nz)/1e6:.1f}M unknowns'
    t0 = time.perf_counter()
    try:
        s = R.solve(model, r=r, nz=nz)
        return s, round(time.perf_counter() - t0, 1)
    except Exception as e:
        return None, 'FAIL ' + repr(e)[:120]

WHOLE = lambda m: (0.0, m.chip_l, 0.0, m.chip_w)

# ------------------------------------------------------------------------------------------ M2 (+ M3 inputs)
def run_m2():
    model = R.build_model(parse_stk(str(BASE / 'stack.stk')))
    base_ice = ice_peak(BASE / 'stack.stk', 'base')
    log('M2 base 3D-ICE peak', base_ice)
    truths = {}
    for s in (400, 200, 100, 50, 25, 10):
        for m in (3, 10, 30):
            tag = f'm2_s{s}_m{m}'; row = {'s_um': s, 'm': m}
            try:
                flp, x0 = hot_flp(s, m)
                stk = make_case(tag, flp)
                mdl = R.build_model(parse_stk(str(stk)))
                pk_ice, t_ice = ice_peak(stk, tag)
                row.update(ice100_peak_K=pk_ice, ice100_rise=pk_ice - T0, ice_s=t_ice)
                c1, _ = solve_ref(mdl, 1, 4, tag)
                row['ref_r1_peak_K'] = None if c1 is None else float(c1.T.max())
                # truth route A: domain-wide r=4, nz=4 (25 um cells, 12.5 um sub-layers): valid for s >= 100
                if s >= 100:
                    tr, info = solve_ref(mdl, 4, 4, tag)
                    if tr is not None:
                        row.update(fineA_peak_K=float(tr.T.max()), fineA_s=info)
                        if m == 10: truths[s] = (mdl, tr, c1)
                    else: row['fineA_err'] = info
                # truth route B: window solver with cells <= s/4 and fine vertical sub-layers near the source
                r_w = max(1, math.ceil(100.0 / (s / 4.0)))                       # lateral refinement so cell <= s/4
                half = 300.0 if s == 10 else 500.0
                win = (2500.0 - half, 2500.0 + half, 2500.0 - half, 2500.0 + half)
                nzf = nz_fine(min(max(s / 4.0, 5.0), 25.0))
                coarse = R.solve_window(mdl, WHOLE(mdl), 1, nz=nzf)
                for bc in ('dirichlet', 'neumann'):
                    t0 = time.perf_counter()
                    w = R.solve_window(mdl, win, r_w, nz=nzf, coarse=coarse, bc=bc)
                    row[f'fineB_{bc}_peak_K'] = float(w.T.max()); row[f'fineB_{bc}_s'] = round(time.perf_counter() - t0, 1)
                    row[f'fineB_{bc}_unknowns'] = w.info['unknowns']
                row['window_um'] = win; row['r_window'] = r_w; row['cell_um'] = 100.0 / r_w
                fine = row.get('fineA_peak_K', row['fineB_dirichlet_peak_K'])
                row['fine_peak_K'] = fine; row['fine_route'] = 'A' if 'fineA_peak_K' in row else 'B(dirichlet)'
                row['underestimate_pct'] = 100.0 * (fine - pk_ice) / (fine - T0)
            except Exception as e:
                row['error'] = repr(e)[:200]; log(tag, 'ERROR', traceback.format_exc()[-600:])
            RES['m2'][tag] = row; save()
            log('M2', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in ('s_um', 'm', 'ice100_rise', 'fineA_peak_K', 'fineB_dirichlet_peak_K', 'fineB_neumann_peak_K', 'underestimate_pct', 'error')})
    return truths

# ------------------------------------------------------------------------------------------ M3 (window size vs truth)
def run_m3(truths):
    for s, (mdl, tr, c1) in truths.items():
        coarse = R.solve(mdl, r=1, nz=4)
        for half in (300.0, 500.0, 1000.0, 2000.0):
            win = (2500.0 - half, 2500.0 + half, 2500.0 - half, 2500.0 + half)
            i0, i1 = int(win[0] / 100) * 4, int(win[1] / 100) * 4
            ref = tr.T[:, i0:i1, i0:i1]
            for bc in ('dirichlet', 'neumann'):
                tag = f'm3_s{s}_h{int(half)}_{bc}'
                try:
                    t0 = time.perf_counter()
                    w = R.solve_window(mdl, win, 4, nz=4, coarse=coarse, bc=bc)
                    d = w.T - ref
                    RES['m3'][tag] = dict(s_um=s, half_um=half, bc=bc, seconds=round(time.perf_counter() - t0, 1),
                                          peak_err_K=float(w.T.max() - ref.max()), peak_err_pct_of_rise=float(100 * (w.T.max() - ref.max()) / (ref.max() - T0)),
                                          max_abs_dT=float(np.abs(d).max()), rmse_K=float(np.sqrt((d ** 2).mean())), unknowns=w.info['unknowns'])
                except Exception as e:
                    RES['m3'][tag] = dict(error=repr(e)[:200])
                save(); log('M3', tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in RES['m3'][tag].items()})

# ------------------------------------------------------------------------------------------ M1
def m1_case(name, stk, with_ice=True, nz=4, r_list=(1, 2, 4), try_r8=False):
    row = {}
    model = R.build_model(parse_stk(str(stk)))
    isrc = src_layer(model)
    sol = {}
    for r in r_list:
        s, info = solve_ref(model, r, nz, name)
        if s is not None:
            sol[r] = s; row[f'r{r}_peak_K'] = float(s.T.max()); row[f'r{r}_seconds'] = info
            F = s.layer_field(isrc); row[f'r{r}_peak_xy_um'] = [float(s.xe[int(np.unravel_index(s.T.max(axis=0).argmax(), F.shape)[0])]), float(s.ye[int(np.unravel_index(s.T.max(axis=0).argmax(), F.shape)[1])])]
        else: row[f'r{r}_note'] = info
    if try_r8:
        s8, info = solve_ref(model, 8, 2, name)
        row['r8_nz2_peak_K'] = None if s8 is None else float(s8.T.max()); row['r8_nz2_note'] = info
    top = max(sol) if sol else None
    if top and top > 1:
        Ft = sol[top].layer_field(isrc)
        for r, s in sol.items():
            if r == top: continue
            Fr = R.block_average(Ft, top // r)
            row[f'r{r}_field_rmse_vs_r{top}_K'] = float(np.sqrt(((s.layer_field(isrc) - Fr) ** 2).mean()))
    pk = [row.get(f'r{r}_peak_K') for r in (1, 2, 4)]
    if all(p is not None for p in pk) and abs(pk[2] - pk[1]) > 1e-9 and (pk[1] - pk[0]) / (pk[2] - pk[1]) > 0:
        p = math.log2((pk[1] - pk[0]) / (pk[2] - pk[1])); row['observed_order'] = p
        row['richardson_peak_K'] = pk[2] + (pk[2] - pk[1]) / (2 ** p - 1) if p > 0 else None
    # vertical sensitivity at r=1
    for n in (2, 4, 8):
        s, info = solve_ref(model, 1, n, name)
        row[f'vert_r1_nz{n}_peak_K'] = None if s is None else float(s.T.max())
    if with_ice:
        row['ice_native_peak_K'], row['ice_native_s'] = ice_peak(stk, name + '_nat')
        half = make_case(name + '_half', None, cell=float(model.cell_l) / 2, src=Path(stk).parent, flp_name='none')
        try:
            row['ice_half_peak_K'], row['ice_half_s'] = ice_peak(half, name + '_half')
        except Exception as e:
            row['ice_half_note'] = repr(e)[:160]
    return row

def run_m1():
    g1 = load_geometry('geometry1')
    scs = [s for s in ScenarioGenerator().generate_all_scenarios(g1) if s.scenario_type not in ('leakage', 'throttle')][:3]
    for i, sc in enumerate(scs):
        name = f'm1_g1_s{i}'
        try:
            d = OUT / name; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
            ICESimulator(d, d / 'sim', ICE_EXE).generate_config_files(g1, sc.to_dict())
            row = m1_case(name, d / 'stack.stk', try_r8=(i == 0))
            row['pattern'] = sc.to_dict().get('pattern')
        except Exception as e:
            row = {'error': repr(e)[:300]}; log(name, traceback.format_exc()[-600:])
        RES['m1'][name] = row; save(); log('M1', name, row)
    try:
        g5 = load_geometry('geometry5')
        sc5 = [s for s in ScenarioGenerator().generate_all_scenarios(g5) if s.scenario_type not in ('leakage', 'throttle')][0]
        d = OUT / 'm1_g5_s0'; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        ICESimulator(d, d / 'sim', ICE_EXE).generate_config_files(g5, sc5.to_dict())
        row = m1_case('m1_g5_s0', d / 'stack.stk', nz=2)
    except Exception as e:
        row = {'error': repr(e)[:300]}; log('g5', traceback.format_exc()[-800:])
    RES['m1']['m1_g5_s0'] = row; save(); log('M1 g5', row)

def run_m1_half():
    """3D-ICE at half the native cell for the already-generated M1 cases (the first attempt lacked the sim/ output dir)."""
    for name in ('m1_g1_s0', 'm1_g1_s1', 'm1_g1_s2', 'm1_g5_s0'):
        row = RES['m1'].get(name, {})
        try:
            stk = OUT / name / 'stack.stk'
            model = R.build_model(parse_stk(str(stk)))
            half = make_case(name + '_half', None, cell=float(model.cell_l) / 2, src=OUT / name, flp_name='none')
            row['ice_half_peak_K'], row['ice_half_s'] = ice_peak(half, name + '_half'); row.pop('ice_half_note', None)
        except Exception as e:
            row['ice_half_note'] = repr(e)[:200]
        RES['m1'][name] = row; save(); log('M1 half', name, row.get('ice_native_peak_K'), row.get('ice_half_peak_K'), row.get('ice_half_note'))

if __name__ == '__main__':
    which = sys.argv[1:] or ['m2', 'm3', 'm1']
    truths = {}
    if 'm2' in which: truths = run_m2()
    if 'm3' in which and truths: run_m3(truths)
    if 'm1' in which: run_m1()
    if 'm1half' in which: run_m1_half()
    log('DONE')
