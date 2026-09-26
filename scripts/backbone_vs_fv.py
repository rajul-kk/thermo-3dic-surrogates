"""How much of the backbone's error vs 3D-ICE is physics, and how much is the solver? (report §9.28)
Per scenario, on the same FV grid: backbone vs full FV solve (the lateral-heterogeneity error the
layered solver cannot see), FV vs 3D-ICE (the solver gap), backbone vs 3D-ICE (what ThermoNO corrects).
All three are scored at 3D-ICE's own nodes with the §9.26 metrics.
Usage: python scripts/backbone_vs_fv.py geometry4 geometry5 geometry6
"""
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
from src.validation import fv_solver as fv
from scripts.hotspot_eval import hotspot_metrics
from scripts.layout_cv import per_scenario_stats

PAIRS = ('backbone_vs_fv', 'fv_vs_3dice', 'backbone_vs_3dice')


def scenario(f, geom_name):
    """Same geometry/scenario reconstruction as scripts/thermono_train.py::build."""
    d = np.load(f, allow_pickle=True)
    m = d['metadata'].item()
    offs = {k[len('placement_dx_'):]: (float(m[k]), float(m['placement_dy_' + k[len('placement_dx_'):]]))
            for k in m if k.startswith('placement_dx_')}
    offs = {('' if k == 'blocks' else k): v for k, v in offs.items()}
    geom = place_chiplets(get_geometry_by_name(geom_name), offs)
    scen = {'power_blocks': {b.name: float(m.get(f'block_power_{b.name}', 0.0)) for b in geom.power_blocks},
            'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
            'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')}
    return geom, scen, d['coords'].astype(np.float64), d['temp'].astype(np.float64)


def summarise(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    return {'r2_mean': m('r2'), 'det_mae_K': m('det_mae'), 'abs_peak_err_K': m('abs_peak_temp_err_K'),
            'peak_err_signed_K': m('peak_temp_err_K'),
            'loc_err_um_median': float(np.median([r['hotspot_loc_err_um'] for r in rows])),
            'recall_mean': m('top1pct_recall'), 'rise_K_mean': m('rise'), 'n': len(rows)}


def main():
    out = Path('results/backbone_vs_fv.json')
    res = json.loads(out.read_text()) if out.exists() else {}
    for geom_name in sys.argv[1:]:
        files = sorted(Path(f'data/3d-ice-layout-{geom_name}').rglob(f'{geom_name}_*.npz'))
        rows = {p: [] for p in PAIRS}
        t0 = time.time()
        for f in files:
            geom, scen, c, y = scenario(f, geom_name)
            g = fv.make_grid(geom)
            kl, kv = fv.conductivity(geom, scen, g)
            full = fv.solve(geom, scen, g)['T']
            bb = lb.solve(geom, scen, g, kl, kv)['T']
            p_bb, p_fv = lb.sample(bb, g, c), lb.sample(full, g, c)
            rise = float(y.max() - (scen['t_ambient'] + 273.15))
            for name, pred, true in (('backbone_vs_fv', p_bb, p_fv), ('fv_vs_3dice', p_fv, y),
                                     ('backbone_vs_3dice', p_bb, y)):
                rows[name].append({**per_scenario_stats(true, pred), **hotspot_metrics(pred, true, c), 'rise': rise})
        res[geom_name] = {p: summarise(rows[p]) for p in PAIRS}
        out.write_text(json.dumps(res, indent=1))
        print(f'{geom_name}: {len(files)} scenarios, {time.time() - t0:.0f}s, mean peak rise '
              f'{res[geom_name][PAIRS[0]]["rise_K_mean"]:.1f} K', flush=True)
        for p in PAIRS:
            s = res[geom_name][p]
            print(f"  {p:<18} R2 {s['r2_mean']:.4f}  det.MAE {s['det_mae_K']:.3f} K  |peak| {s['abs_peak_err_K']:.2f} K "
                  f"(signed {s['peak_err_signed_K']:+.2f})  loc {s['loc_err_um_median']:.0f} um  recall {s['recall_mean']:.3f}",
                  flush=True)


if __name__ == '__main__':
    main()
