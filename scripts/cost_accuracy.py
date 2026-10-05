"""Cost vs accuracy of every solver/surrogate on the corrected layout data (report §9.32, B1).

Per geometry, on the first --n scenarios of data/3d-ice-layout-{g}, all scored against the 3D-ICE nodes:
  3D-ICE            wall time of one solve (the ground truth itself; WSL)
  FV direct         our finite-volume solve (scipy direct)
  backbone          layered DCT x tridiagonal solve, no training (§9.25)
  backbone-PCG k    CG on the FV system preconditioned by the backbone, k iterations (§9.30)
  ThermoNO          backbone + learned correction (§9.26): measured inference time; training time from the CPU
                    seed-0 logs (logs/thermono_snap_{g}_seed0.log); accuracy from results/thermono_snap (5-fold CV)
Run with no other jobs on the machine (timings).  Usage: python scripts/cost_accuracy.py [--n 5]
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.backbone_pcg import backbone_op                              # noqa: E402
from scripts.backbone_vs_fv import scenario                               # noqa: E402
from scripts.hotspot_eval import hotspot_metrics                          # noqa: E402
from scripts.layout_cv import per_scenario_stats                          # noqa: E402
from scripts.thermono_train import build, tensors                         # noqa: E402
from src.fno.thermono import ThermoNO                                     # noqa: E402
from src.hybrid import layered_backbone as lb                             # noqa: E402
from src.simulators.ice_simulator import ICESimulator                     # noqa: E402
from src.validation import fv_solver as fv                                # noqa: E402

ICE_EXE = 'wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator'
PCG_ITERS = (10, 20, 30, 50)


def score(rise_grid, g, c, y, t_amb):
    p = lb.sample(rise_grid, g, c) + t_amb
    s = {**per_scenario_stats(y, p), **hotspot_metrics(p, y, c)}
    return {'abs_peak_K': s['abs_peak_temp_err_K'], 'det_mae_K': s['det_mae'], 'loc_um': s['hotspot_loc_err_um'],
            'r2': s['r2']}


def pcg(A, b, M, iters):
    x = np.zeros_like(b); r = b.copy(); z = M(r); p = z.copy(); rz = r @ z
    out, t0 = {}, time.perf_counter()
    for it in range(1, max(iters) + 1):
        Ap = A @ p; a = rz / (p @ Ap); x += a * p; r -= a * Ap
        if it in iters:
            out[it] = (x.copy(), time.perf_counter() - t0)
        z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
    return out


def thermono_train_seconds(g):
    log = Path(f'logs/thermono_snap_{g}_seed0.log')
    s = [float(x) for x in re.findall(r'fold \d: \d+ epochs, (\d+)s', log.read_text())] if log.exists() else []
    return float(np.mean(s)) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometries', nargs='*', default=['geometry4', 'geometry5', 'geometry6'])
    ap.add_argument('--n', type=int, default=5)
    a = ap.parse_args()
    res = {}
    for gname in a.geometries:
        files = sorted(Path(f'data/3d-ice-layout-{gname}').rglob(f'{gname}_*.npz'))[:a.n]
        rows = {}
        D = build(gname, 'data')
        net = ThermoNO(D['q'].shape[1:], ch=16, n_blocks=3, modes=(16, 16)).eval()
        for i, f in enumerate(files):
            geom, scen, c, y = scenario(f, gname)
            t_amb = scen['t_ambient'] + 273.15
            g = fv.make_grid(geom)
            kl, kv = fv.conductivity(geom, scen, g)

            with tempfile.TemporaryDirectory() as td:
                cfg, out = Path(td) / 'c', Path(td) / 'o'
                cfg.mkdir(); out.mkdir()
                t = time.perf_counter()
                ICESimulator(cfg, out, ICE_EXE).simulate(geom, scen, f.stem)
                rows.setdefault('3D-ICE', []).append({'seconds': time.perf_counter() - t})

            t = time.perf_counter()
            T_fv = fv.solve(geom, scen, g)['T'] - t_amb
            rows.setdefault('FV direct', []).append({'seconds': time.perf_counter() - t, **score(T_fv, g, c, y, t_amb)})

            t = time.perf_counter()
            bb = lb.solve(geom, scen, g, kl, kv)['T'] - t_amb
            t_bb = time.perf_counter() - t
            rows.setdefault('backbone', []).append({'seconds': t_bb, **score(bb, g, c, y, t_amb)})

            A, _ = fv.assemble(g, kl, kv, scen['htc'])
            b = fv.power(geom, scen, g).ravel()
            t = time.perf_counter()
            M = backbone_op(g, kl, kv, scen['htc'])
            t_setup = time.perf_counter() - t
            for k, (x, tt) in pcg(A, b, M, PCG_ITERS).items():
                rows.setdefault(f'backbone-PCG {k}', []).append(
                    {'seconds': t_setup + tt, **score(x.reshape(g.shape), g, c, y, t_amb)})

            # ThermoNO inference: backbone + one forward pass (weights do not change the cost)
            lin, geo, _, _ = tensors(D, np.array([i]), 1.0, 1.0)
            with torch.no_grad():
                net(lin, geo)                                   # warm-up
                t = time.perf_counter()
                for _ in range(5):
                    net(lin, geo)
            rows.setdefault('ThermoNO (inference)', []).append({'seconds': t_bb + (time.perf_counter() - t) / 5})
            print(f'{gname} {f.stem}: ' + ', '.join(f'{k} {v[-1]["seconds"]:.3f}s' for k, v in rows.items()), flush=True)

        summ = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
        summ['ThermoNO (inference)']['train_seconds_per_fold_cpu'] = thermono_train_seconds(gname)
        tn = Path(f'results/thermono_snap/thermono_{gname}_seed0.json')
        if tn.exists():
            s = json.loads(tn.read_text())['summary']['thermono']
            summ['ThermoNO (inference)'].update(abs_peak_K=s['abs_peak_err_K'], det_mae_K=s['det_mae_K'],
                                                loc_um=s['loc_err_um_median'], r2=s['r2_mean'], accuracy_from='5-fold CV')
        res[gname] = {'n': len(files), 'cells': int(np.prod(D['q'].shape[1:])), 'methods': summ}
        print(json.dumps(res[gname], indent=1), flush=True)
    out = Path('results/cost_accuracy.json')
    out.write_text(json.dumps(res, indent=1))
    print(f'saved {out}')


if __name__ == '__main__':
    main()
