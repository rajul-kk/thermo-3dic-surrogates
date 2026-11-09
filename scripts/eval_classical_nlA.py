"""Evaluate the classical solvers C0, C1, C2 (src/scenario/leakage_solvers.py) on the held-out regime-A sets against the real 3D-ICE
fields, and the stability test (A5) on the converged + runaway sets. Protocol: docs/protocol_nonlinear_A.md sections 5-6, amendment 15.
Output: results/nonlinear_A/classical_eval.json (gitignored) and a printed summary. Single-threaded numpy, plants cached per HTC.
Usage: python scripts/eval_classical_nlA.py [--limit N]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.scenario import leakage_solvers as S

OUT = REPO / 'results' / 'nonlinear_A'
DATA = REPO / 'data' / '3d-ice-leakage-A' / 'geometry1'
ICE_SOLVE_S = 12.2          # quiet-machine 3D-ICE solve (protocol amendment 11, G1); loop time is ESTIMATED as iterations x this
SETS = ('iid', 'ood_gain', 'ood_law', 'near_critical')
PEAK_TOL = 0.3              # A1: peak error vs 3D-ICE, K


def problem(it):
    return S.Problem(it['t_ambient'], np.array(it['nominal']), np.array(it['share']), it['k_d'], it['law'])


def first_time(run, err, tol):
    for t, e in zip(run.times, err):
        if abs(e) <= tol: return t
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); args = ap.parse_args()
    plan = json.loads((OUT / 'plan.json').read_text())
    plants, recs = {}, []
    todo = [it for it in plan if it['split'] in SETS][:args.limit or None]
    t_all = time.perf_counter()
    for n, it in enumerate(todo):
        if it['htc'] not in plants: plants[it['htc']] = S.Plant(it['htc'])
        P = plants[it['htc']]; pb = problem(it)
        z = np.load(DATA / f"{it['name']}.npz", allow_pickle=True)
        T_ice = P.fv.ice_to_grid(z['coords'].astype(np.float64), z['temp'].astype(np.float64), P.g).ravel()
        have = ~np.isnan(T_ice); T_ice = T_ice[have]           # 3D-ICE reports 10 of the 12 FV z-cell layers (one sub-layer of the die)
        assert have.sum() > 0.8 * have.size
        peak_ice, off = float(T_ice.max()), pb.t_amb + 273.15
        ref = S.reference(P, pb); _, th_ref, Tb_ref = ref
        rec = dict(name=it['name'], split=it['split'], htc=it['htc'], rho=it['rho'], nonlin_pct=it['nonlin_pct'], picard_it_plan=it['picard_iterations'],
                   ice_loop_s_est=it['picard_iterations'] * ICE_SOLVE_S, fv_ref_vs_ice_peak_K=float(th_ref[have].max() + off - peak_ice),
                   fv_ref_vs_ice_rmse_K=float(np.sqrt(np.mean((th_ref[have] + off - T_ice) ** 2))), methods={})
        for f in (S.c0, S.c1, S.c2):
            run = f(P, pb)
            pk = [float(s[have].max() + off - peak_ice) for s in run.states]                       # signed peak error vs 3D-ICE
            pk_ref = [float(s[have].max() - th_ref[have].max()) for s in run.states]                      # algorithmic error vs the FV fixed point
            rm = [float(np.sqrt(np.mean((s[have] + off - T_ice) ** 2))) for s in run.states]
            mae = [float(np.mean(np.abs(s[have] + off - T_ice))) for s in run.states]
            bm = [float(np.abs(P.block_T(s, pb.t_amb) - (P.block_T(th_ref, pb.t_amb))).max()) for s in run.states]
            t_103 = next((t for t, r in zip(run.times, run.resid) if r < 1e-3), None) if f is S.c1 else None
            rec['methods'][run.method] = dict(steps=len(run.states), converged=run.converged, runaway=run.runaway, time_s=run.times[-1], inner=run.inner[-1],
                                              peak_err_ice_K=pk[-1], peak_err_fv_K=pk_ref[-1], rmse_ice_K=rm[-1], mae_ice_K=mae[-1], block_err_fv_K=bm[-1],
                                              t_to_peak03_s=first_time(run, pk, PEAK_TOL), t_to_resid1e3_s=t_103,
                                              curve=dict(t=run.times, peak_err_ice=pk, rmse_ice=rm))
        recs.append(rec)
        if n % 10 == 0:
            m2 = rec['methods']['C2']; print(f"{n+1}/{len(todo)} {it['name']} rho {it['rho']:.2f} | C2 {m2['steps']} steps {m2['time_s']*1000:.0f} ms peak err ice {m2['peak_err_ice_K']:+.3f} K | "
                                             f"C1 {rec['methods']['C1']['steps']} steps | FV ref vs 3D-ICE peak {rec['fv_ref_vs_ice_peak_K']:+.3f} K", flush=True)
    # ---- A5: stability test on converged + near-critical + runaway sets
    cls = []
    for it in plan:
        if it['split'] not in SETS + ('runaway',): continue
        if it['htc'] not in plants: plants[it['htc']] = S.Plant(it['htc'])
        t0 = time.perf_counter(); ok, rho, steps = S.classify(plants[it['htc']], problem(it)); dt = time.perf_counter() - t0
        cls.append(dict(name=it['name'], split=it['split'], truth_converges=it['split'] != 'runaway', predicted_converges=bool(ok), rho_exit=rho, steps=steps, seconds=dt))
    (OUT / 'classical_eval.json').write_text(json.dumps(dict(records=recs, classify=cls, ice_solve_s=ICE_SOLVE_S, peak_tol=PEAK_TOL,
                                                             plant_setup_s={str(k): [v.setup_s, v.unit_s] for k, v in plants.items()}), default=float))
    summarise(recs, cls, time.perf_counter() - t_all)


def summarise(recs, cls, total_s):
    print(f'\n=== {len(recs)} scenarios, total {total_s:.0f} s ===')
    sel = [r for r in recs if r['split'] in ('iid', 'ood_gain', 'ood_law')]
    print(f"FV reference vs real 3D-ICE (the solver gap, all sets): peak {np.median([abs(r['fv_ref_vs_ice_peak_K']) for r in recs]):.4f} K median, "
          f"max {max(abs(r['fv_ref_vs_ice_peak_K']) for r in recs):.4f} K; field RMSE max {max(r['fv_ref_vs_ice_rmse_K'] for r in recs):.4f} K")
    for label, rs in (('A1 pool (iid + OOD-gain + OOD-law)', sel), ('near-critical', [r for r in recs if r['split'] == 'near_critical'])) + tuple((s, [r for r in recs if r['split'] == s]) for s in SETS):
        print(f'-- {label}: n={len(rs)}')
        for m in ('C0', 'C1', 'C2'):
            ms = [r['methods'][m] for r in rs]
            ok = np.mean([abs(x['peak_err_ice_K']) <= PEAK_TOL for x in ms]); conv = np.mean([x['converged'] for x in ms])
            t = np.array([x['time_s'] for x in ms]); ratio = np.array([x['time_s'] / r['ice_loop_s_est'] for x, r in zip(ms, rs)])
            print(f"   {m}: |peak err vs 3D-ICE| <= {PEAK_TOL} K on {100*ok:.0f}%  (median {np.median([abs(x['peak_err_ice_K']) for x in ms]):.4f}, max {max(abs(x['peak_err_ice_K']) for x in ms):.4f} K) | "
                  f"vs FV fixed point median {np.median([abs(x['peak_err_fv_K']) for x in ms]):.2e} K | converged {100*conv:.0f}% | time median {1000*np.median(t):.0f} ms, "
                  f"fraction of est. 3D-ICE loop median {np.median(ratio):.4f} max {ratio.max():.4f} | steps median {int(np.median([x['steps'] for x in ms]))} | RMSE vs 3D-ICE median {np.median([x['rmse_ice_K'] for x in ms]):.4f} K")
    ok = [c for c in cls]; acc = np.mean([c['truth_converges'] == c['predicted_converges'] for c in ok])
    print(f"-- A5 stability test: accuracy {100*acc:.1f}% on {len(ok)} (runaway {sum(not c['truth_converges'] for c in ok)}); errors: {[c['name'] for c in ok if c['truth_converges'] != c['predicted_converges']]}; "
          f"median test time {1000*np.median([c['seconds'] for c in ok]):.1f} ms")


if __name__ == '__main__':
    main()
