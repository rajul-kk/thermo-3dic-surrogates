"""Gates G1 and G3 of docs/protocol_nonlinear_A.md: per-block leakage loop on the FV reference solver, then ONE 3D-ICE solve on
the final delivered powers (the 'final linear solve') for 5 scenarios, timed. Checks FV-vs-3D-ICE peak agreement (< 0.5 K) and
the self-consistency of the loop output. Results: results/nonlinear_A/g3_check.json (gitignored).
Usage: python scripts/leakage_g3_check.py [n_scenarios=5]"""
import copy, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools' / 'iceforge')); sys.path.insert(0, str(REPO / 'scripts' / 'mesh_study'))
sys.argv = sys.argv[:1] + sys.argv[1:]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
sys.argv = sys.argv[:1]
import importlib.util
spec = importlib.util.spec_from_file_location('ms', REPO / 'scripts' / 'mesh_study' / 'mesh_sweeps.py')
ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
R, parse_stk, BASE = ms.R, ms.parse_stk, ms.BASE
from iceforge.model import Rect
from src.scenario.leakage_perblock import block_power, law_exp, picard

OUT = REPO / 'results' / 'nonlinear_A'; OUT.mkdir(parents=True, exist_ok=True)
model = R.build_model(parse_stk(str(BASE / 'stack.stk')))
isrc = ms.src_layer(model)
blocks = ms.read_blocks(BASE / 'floorplan_layer4.flp')          # name -> (x, y, w, h, p, p)
names = sorted(blocks)
rects = [Rect(*blocks[n][:4]) for n in names]
TREF = 85.0

def with_powers(P):
    m2 = copy.copy(model); layers = list(model.layers); L = copy.copy(layers[isrc])
    L.power_rects = [(rc, p / (rc.l * rc.w)) for rc, p in zip(rects, P)]
    layers[isrc] = L; m2.layers = layers
    return m2

def fv_block_T(P):
    sol = R.solve(with_powers(P), r=1, nz=None)
    F = sol.layer_field(isrc) - 273.15 if sol.layer_field(isrc).mean() > 200 else sol.layer_field(isrc)
    out = []
    for rc in rects:
        i0, i1 = int(round(rc.x / model.cell_l)), int(round((rc.x + rc.l) / model.cell_l))
        j0, j1 = int(round(rc.y / model.cell_w)), int(round((rc.y + rc.w) / model.cell_w))
        out.append(F[i0:i1, j0:j1].mean())
    return np.array(out), float(F.max())

rng = np.random.default_rng(20261201)
rows = []
for s in range(N):
    p_nom = rng.uniform(2.0, 5.0, len(names)); share = rng.uniform(0.10, 0.35, len(names)) * rng.uniform(0.8, 1.2, len(names))
    p_leak, p_dyn, k_d = p_nom * share, p_nom * (1 - share), float(rng.uniform(10, 25))
    t0 = time.perf_counter()
    res = picard(lambda P: fv_block_T(P)[0], p_dyn, p_leak, TREF, k_d)
    t_fv_loop = time.perf_counter() - t0
    Tfv, peak_fv = fv_block_T(res.P)
    flp = "// final linear solve\n\n" + ''.join(ms.element(n, *blocks[n][:4], float(p)) for n, p in zip(names, res.P))
    stk = ms.make_case(f'g3_s{s}', flp_text=flp)
    peak_ice, t_ice = ms.ice_peak(stk, f'g3_s{s}')
    # 3D-ICE reports Kelvin or Celsius depending on the stack; compare in the FV's unit
    peak_ice_c = peak_ice - 273.15 if peak_ice > 200 else peak_ice
    sc = block_power(res.T, p_dyn, p_leak, TREF, k_d)
    row = dict(scenario=s, k_d=k_d, share_mean=float(share.mean()), iterations=res.iterations, converged=res.converged, runaway=res.runaway,
               loop_residual=res.residual, gain=res.gain, selfcons=float(np.max(np.abs(sc - res.P) / res.P)),
               peak_fv_C=peak_fv, peak_ice_C=peak_ice_c, d_peak_ice_minus_fv_K=peak_ice_c - peak_fv, t_ice_s=t_ice, t_fv_loop_s=round(t_fv_loop, 1),
               P_nominal=float(p_nom.sum()), P_delivered=float(res.P.sum()))
    rows.append(row); print(row, flush=True)
    (OUT / 'g3_check.json').write_text(json.dumps(rows, indent=1, default=float))
ok = all(abs(r['d_peak_ice_minus_fv_K']) < 0.5 for r in rows)
print('G3 FV-vs-3D-ICE', 'PASS' if ok else 'FAIL', '| 3D-ICE solve time s:', [r['t_ice_s'] for r in rows], flush=True)
