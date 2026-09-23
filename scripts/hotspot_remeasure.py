"""Re-run §9.17's hotspot comparison (ridge vs linear-on-field, nested pca_k) over every dataset and 4 seeds.
Writes results/hotspot_remeasured.json; used after the §9.23 data repair.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DATASETS = [(f'{g}-shelf', g, f'data/3d-ice-layout-{g}') for g in
            ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5', 'geometry6']] + \
           [(f'{g}-fixed', g, 'data/3d-ice') for g in ['geometry1', 'geometry2a', 'geometry3', 'geometry4']]
SEEDS = [0, 1, 2, 3]
OUT = Path('results/hotspot_remeasured.json')


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for label, geom, data in DATASETS:
        for seed in SEEDS:
            key = f'{label}/seed{seed}'
            if key in out:
                continue
            with tempfile.TemporaryDirectory() as td:
                res = Path(td) / 'r.json'
                subprocess.run([sys.executable, 'scripts/hotspot_eval.py', '--geometry', geom, '--data', data,
                                '--field-linear', '--seed', str(seed), '--output', str(res)],
                               check=True, capture_output=True)
                r = json.loads(res.read_text())[geom]
            b = r['baselines']
            out[key] = {
                'spread_um': r['flatness']['peak_spread_um_median'],
                'pca_k': r['field_pca_k_chosen'],
                'loc_ratio': b['ridge']['hotspot_loc_err_um_median'] / b['linear_field']['hotspot_loc_err_um_median'],
                'recall_ratio': b['linear_field']['top1pct_recall_mean'] / max(b['ridge']['top1pct_recall_mean'], 1e-9),
                'ridge': {k: b['ridge'][k] for k in ('abs_peak_temp_err_K_mean', 'hotspot_loc_err_um_median', 'top1pct_recall_mean')},
                'field': {k: b['linear_field'][k] for k in ('abs_peak_temp_err_K_mean', 'hotspot_loc_err_um_median', 'top1pct_recall_mean')},
            }
            OUT.write_text(json.dumps(out, indent=1))
            print(f"{key:<24} loc {out[key]['loc_ratio']:.2f}x  recall {out[key]['recall_ratio']:.2f}x  "
                  f"spread {out[key]['spread_um']:.0f} um", flush=True)

    print('\nseed-mean (range):')
    for label, _, _ in DATASETS:
        rows = [out[f'{label}/seed{s}'] for s in SEEDS]
        lr = np.array([r['loc_ratio'] for r in rows]); rr = np.array([r['recall_ratio'] for r in rows])
        print(f'  {label:<18} loc {lr.mean():.2f}x ({lr.min():.2f}-{lr.max():.2f})  recall {rr.mean():.2f}x  '
              f'spread {rows[0]["spread_um"]:.0f} um  |peak| ridge/field '
              f'{rows[0]["ridge"]["abs_peak_temp_err_K_mean"]:.2f}/{rows[0]["field"]["abs_peak_temp_err_K_mean"]:.2f} K')


if __name__ == '__main__':
    main()
