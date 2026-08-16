"""
Evaluate the geometry1_throttled FNO checkpoint with the same detrended metrics
scripts/baselines.py scores ridge with, so the two are directly comparable.

Written 2026-08-16 to close a reproducibility gap found during a documentation
audit: docs/report.md §9.7/§9.8 and goal.md quoted this checkpoint's det.MAE/
spatial R^2/hotspot error, but the eval that produced those numbers had only
been run inline and was never saved as a re-runnable artifact. Output is
written to results/fno_throttled_eval.json.

Usage:
    python scripts/eval_fno_throttled.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.core.geometry_builders import get_geometry_by_name
from src.fno.data_loader import FNODataset, predict_to_flat
from src.fno.model import build_fno
from src.pinn.data_loader import NormStats
from scripts.baselines import metrics as compute_metrics

CHECKPOINT_DIR = Path('checkpoints/fno/geometry1_throttled')
DATA_DIR = Path('data/3d-ice-throttle-pilot/geometry1')
GRID = (100, 100, 10)
MODES = (8, 8, 6)


def main():
    test_files = sorted(DATA_DIR.glob('geometry1_test_*.npz'))
    if not test_files:
        raise SystemExit(f"No test files found in {DATA_DIR} -- run the throttle "
                         "pilot generation first (see goal.md Track A).")

    geometries = {'geometry1': get_geometry_by_name('geometry1')}
    ns = NormStats(**json.load(open(CHECKPOINT_DIR / 'norm_stats.json')))
    ds = FNODataset(test_files, ns, expected_grid=GRID)

    model = build_fno(grid_shape=GRID, modes=MODES, hidden_ch=16, n_blocks=3)
    ckpt = torch.load(CHECKPOINT_DIR / 'geometry1_throttled_best.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    geom = geometries['geometry1']
    nx, ny, nz = GRID
    xs = np.linspace(0, geom.die_width, nx)
    ys = np.linspace(0, geom.die_length, ny)
    zs = np.linspace(0, geom.get_total_height(), nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    all_m = []
    with torch.no_grad():
        for item in ds.items:
            T_pred_K, T_true_K = predict_to_flat(model, item, torch.device('cpu'), ns)
            m = compute_metrics(T_pred_K, T_true_K, coords)
            m['name'] = item['name']
            all_m.append(m)

    agg = {k: float(np.mean([m[k] for m in all_m]))
          for k in ('mae_K', 'mae_detrended_K', 'spatial_r2', 'hotspot_loc_err_um')}

    out_path = Path('results/fno_throttled_eval.json')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint': str(CHECKPOINT_DIR / 'geometry1_throttled_best.pt'),
            'data': f'{DATA_DIR} (test split, {len(test_files)} scenarios)',
            'per_scenario': all_m,
            'aggregate': agg,
        }, f, indent=2)

    print(json.dumps(agg, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == '__main__':
    main()
