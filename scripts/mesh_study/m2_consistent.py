"""M2 re-analysis with an apples-to-apples metric: same reference solver, same vertical discretisation at coarse (100 um) and fine.
Adds the no-hotspot baseline per size so the hotspot EXCESS can be compared, not only the total peak rise."""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, r'D:\Work\3D-ICE Thermal-modelling\Thermo'); sys.path.insert(0, r'D:\Work\3D-ICE Thermal-modelling\Thermo\tools\iceforge')
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location('ms', Path(__file__).with_name('mesh_sweeps.py')); ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
R, parse_stk, OUT, T0 = ms.R, ms.parse_stk, ms.OUT, ms.T0
res = {}
base_model = R.build_model(parse_stk(str(ms.BASE / 'stack.stk')))
for s in (400, 200, 100, 50, 25, 10):
    r_w = max(1, math.ceil(100.0 / (s / 4.0))); half = 300.0 if s == 10 else 500.0
    win = (2500.0 - half, 2500.0 + half, 2500.0 - half, 2500.0 + half)
    nzf = ms.nz_fine(min(max(s / 4.0, 5.0), 25.0))
    def route(mdl):
        out = {}
        coarse = R.solve_window(mdl, ms.WHOLE(mdl), 1, nz=nzf)
        out['coarse'] = float(coarse.T.max())
        for bc in ('dirichlet', 'neumann'):
            try:
                out[bc] = float(R.solve_window(mdl, win, r_w, nz=nzf, coarse=coarse, bc=bc).T.max())
            except Exception as e:
                out[bc] = None; out[bc + '_err'] = repr(e)[:80]
        return out
    base = route(base_model); res[f's{s}_base'] = base; print('base', s, base, flush=True)
    for m in (3, 10, 30):
        try:
            flp, _ = ms.hot_flp(s, m)
            stk = ms.make_case(f'm2c_s{s}_m{m}', flp)
            mdl = R.build_model(parse_stk(str(stk)))
            row = route(mdl)
            for bc in ('dirichlet', 'neumann'):
                if row.get(bc) is None or base.get(bc) is None: continue
                fine_rise = row[bc] - T0; coarse_rise = row['coarse'] - T0
                exc_f = row[bc] - base[bc]; exc_c = row['coarse'] - base['coarse']
                row[f'under_pct_of_rise_{bc}'] = 100 * (row[bc] - row['coarse']) / fine_rise
                row[f'excess_fine_K_{bc}'] = exc_f; row[f'excess_coarse_K'] = exc_c
                row[f'capture_{bc}'] = exc_c / exc_f if exc_f > 1e-9 else None
            res[f's{s}_m{m}'] = row
            print(s, m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        except Exception as e:
            res[f's{s}_m{m}'] = {'error': repr(e)[:200]}; print(s, m, 'ERROR', repr(e)[:200], flush=True)
    (OUT / 'm2_consistent.json').write_text(json.dumps(res, indent=1, default=float))
print('DONE')
