"""Does an exact layered-stack backbone make layout-varying thermal prediction easier? (report §9.25)
Compares ridge, field-linear, the zero-training backbone, and backbone + learned linear residual.
Usage: python scripts/backbone_eval.py [--geometries geometry4 ...] [--seeds 0 1 2 3]
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import place_chiplets
from src.hybrid import layered_backbone as lb
from scripts.baselines import (load_scenario, collect_block_keys, predict_all,
                               _field_fit, FIELD_PCA_GRID)
from scripts.hotspot_eval import hotspot_metrics, kfold_indices
from scripts.layout_cv import per_scenario_stats

GEOMS = ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5', 'geometry6']
OUT = Path('results/backbone_eval.json')


def backbone_predictions(files, geom_name):
    """Zero-training backbone field for each file, read at that file's nodes; also returns seconds/solve."""
    preds, secs = [], []
    for f in files:
        m = np.load(f, allow_pickle=True)['metadata'].item()
        offs = {k[len('placement_dx_'):]: (float(m[k]), float(m['placement_dy_' + k[len('placement_dx_'):]]))
                for k in m if k.startswith('placement_dx_')}
        offs = {('' if k == 'blocks' else k): v for k, v in offs.items()}
        geom = place_chiplets(get_geometry_by_name(geom_name), offs)
        scen = {'power_blocks': {b.name: float(m.get(f'block_power_{b.name}', 0.0)) for b in geom.power_blocks},
                'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
                'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')}
        t = time.perf_counter()
        res = lb.solve(geom, scen)
        secs.append(time.perf_counter() - t)
        preds.append(lb.sample(res['T'], res['grid'], np.load(f)['coords'].astype(np.float64)))
    return np.stack(preds), float(np.median(secs))


def fit_residual(P_tr, R_tr, B_tr, Y_tr, grid=FIELD_PCA_GRID, folds=4, seed=0):
    """PCA-ridge from the power field to the backbone's residual; rank chosen by inner CV on detrended R² of T."""
    idx = np.random.default_rng(seed).permutation(len(P_tr))
    parts = np.array_split(idx, folds)
    best, best_score = grid[0], -np.inf
    for k in grid:
        r2 = []
        for te in parts:
            tr = np.setdiff1d(idx, te)
            T_hat = B_tr[te] + _field_fit(P_tr[tr], R_tr[tr], k)(P_tr[te])
            r2 += [per_scenario_stats(y, p)['r2'] for y, p in zip(Y_tr[te], T_hat)]
        if np.mean(r2) > best_score:
            best, best_score = k, float(np.mean(r2))
    return _field_fit(P_tr, R_tr, best), best


def evaluate(geom_name, seeds):
    files = sorted(Path(f'data/3d-ice-layout-{geom_name}').rglob(f'{geom_name}_*.npz'))
    scen = [load_scenario(f) for f in files]
    Y = np.stack([s['temp'] for s in scen]); P = np.stack([s['power'] for s in scen])
    coords = scen[0]['coords']
    B, sec = backbone_predictions(files, geom_name)
    bk = collect_block_keys(scen)
    rows = {m: [] for m in ('ridge', 'linear_field', 'backbone', 'backbone+residual')}
    ks = []
    for seed in seeds:
        for fold in kfold_indices(len(scen), 5, seed):
            tr = np.setdiff1d(np.arange(len(scen)), fold)
            base = predict_all([scen[i] for i in tr], [scen[i] for i in fold], bk, 3, 1.0, 0,
                               include_field_linear=True, field_pca_k=None)
            model, k = fit_residual(P[tr], Y[tr] - B[tr], B[tr], Y[tr], seed=seed)
            ks.append(k)
            corr = B[fold] + model(P[fold])
            for j, i in enumerate(fold):
                for name, pred in (('ridge', base[j]['ridge']), ('linear_field', base[j]['linear_field']),
                                   ('backbone', B[i]), ('backbone+residual', corr[j])):
                    rows[name].append({**per_scenario_stats(Y[i], pred), **hotspot_metrics(pred, Y[i], coords)})
    out = {'n': len(scen), 'seeds': seeds, 'backbone_sec_per_solve': sec, 'residual_pca_k': ks, 'models': {}}
    for name, rr in rows.items():
        out['models'][name] = {
            'r2_mean': float(np.mean([r['r2'] for r in rr])), 'r2_median': float(np.median([r['r2'] for r in rr])),
            'det_mae_K': float(np.mean([r['det_mae'] for r in rr])),
            'loc_err_um_median': float(np.median([r['hotspot_loc_err_um'] for r in rr])),
            'recall_mean': float(np.mean([r['top1pct_recall'] for r in rr])),
            'abs_peak_err_K': float(np.mean([r['abs_peak_temp_err_K'] for r in rr])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometries', nargs='*', default=GEOMS)
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2, 3])
    args = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for g in args.geometries:
        res[g] = evaluate(g, args.seeds)
        OUT.write_text(json.dumps(res, indent=1))
        print(f"\n=== {g}-shelf (n={res[g]['n']}, seeds {args.seeds}; backbone "
              f"{1000 * res[g]['backbone_sec_per_solve']:.0f} ms/solve, no training)")
        print(f"  {'model':<19} {'R2 mean':>8} {'R2 med':>7} {'det.MAE K':>9} {'loc um':>8} {'recall':>7} {'|peak| K':>9}")
        for name, m in res[g]['models'].items():
            print(f"  {name:<19} {m['r2_mean']:>8.3f} {m['r2_median']:>7.3f} {m['det_mae_K']:>9.3f} "
                  f"{m['loc_err_um_median']:>8.0f} {m['recall_mean']:>7.3f} {m['abs_peak_err_K']:>9.2f}", flush=True)


if __name__ == '__main__':
    main()
