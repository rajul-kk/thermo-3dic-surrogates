"""Re-score the Kaggle FNO-notebook checkpoints (report §9.27) with hotspot metrics and train-fold error.
Reproduces the notebook's folds, hold-out and normalisation exactly; the test R2 must match its JSON.
Usage: python scripts/fno_rescore.py <checkpoints/layout_repr_test dir> [--variants fno lt-fno cno-fno-attn]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.mesh import points_to_grid
from src.fno.data_loader import FNODataset, predict_to_flat
from src.fno.ltfno import build_lt_fno
from src.fno.model import build_cno_fno, build_fno
from src.pinn.data_loader import compute_norm_stats
from scripts.hotspot_eval import hotspot_metrics, kfold_indices
from scripts.layout_cv import per_scenario_stats

CH, BLOCKS, MODES, N_CNO, HEADS = 32, 4, (16, 16, 8), 2, 4       # notebook cell 7
FOLDS, SEED = 5, 0


def build(variant, grid):
    if variant == 'fno':
        return build_fno(grid, modes=MODES, hidden_ch=CH, n_blocks=BLOCKS)
    if variant == 'lt-fno':
        return build_lt_fno(grid, modes=MODES, hidden_ch=CH, n_blocks=BLOCKS)
    return build_cno_fno(grid, ch=CH, n_fno_blocks=BLOCKS, n_cno_layers=N_CNO, use_attention=True, n_heads=HEADS)


def score(model, ds, files, norm):
    rows = []
    for j, f in enumerate(files):
        pred, true = predict_to_flat(model, ds.items[j], torch.device('cpu'), norm)
        c = np.load(f)['coords'].astype(np.float64)
        cg = np.stack(points_to_grid(c, c[:, 0], c[:, 1], c[:, 2]), -1).reshape(-1, 3)
        rows.append({**per_scenario_stats(true, pred), **hotspot_metrics(pred, true, cg)})
    return rows


def summary(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    return {'r2_mean': m('r2'), 'r2_median': float(np.median([r['r2'] for r in rows])), 'det_mae_K': m('det_mae'),
            'loc_err_um_median': float(np.median([r['hotspot_loc_err_um'] for r in rows])),
            'recall_mean': m('top1pct_recall'), 'abs_peak_err_K': m('abs_peak_temp_err_K'),
            'peak_err_signed_K': m('peak_temp_err_K'),
            'abs_hotspot_temp_err_K': float(np.mean([abs(r['hotspot_temp_err_K']) for r in rows])), 'n': len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt_dir', type=Path)
    ap.add_argument('--variants', nargs='*', default=['fno', 'lt-fno', 'cno-fno-attn'])
    ap.add_argument('--geometries', nargs='*', default=['geometry4', 'geometry6'])
    ap.add_argument('--out', type=Path, default=Path('results/fno_kaggle/fno_rescore.json'))
    args = ap.parse_args()
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    for geom in args.geometries:
        files = sorted(Path(f'data/3d-ice-layout-{geom}').rglob(f'{geom}_*.npz'))
        assert len(files) == 45, len(files)
        c0 = np.load(files[0])['coords']
        grid = tuple(len(np.unique(c0[:, i])) for i in (1, 0, 2))
        for variant in args.variants:
            test, train = [], []
            for fi, fold in enumerate(kfold_indices(len(files), FOLDS, SEED), 1):
                te = set(fold.tolist())
                tr_files = [files[i] for i in range(len(files)) if i not in te]
                rng = np.random.default_rng(SEED + fi)
                val_idx = set(rng.choice(len(tr_files), size=max(3, len(tr_files) // 9), replace=False).tolist())
                fit_files = [f for k, f in enumerate(tr_files) if k not in val_idx]
                te_files = [files[i] for i in fold]
                norm = compute_norm_stats(fit_files, {geom: get_geometry_by_name(geom)})
                model = build(variant, grid)
                name = f'{geom}_{variant}_fold{fi}'
                state = torch.load(args.ckpt_dir / name / f'{name}_best.pt', map_location='cpu')
                model.load_state_dict(state['model_state'])
                test += score(model, FNODataset(te_files, norm, grid), te_files, norm)
                train += score(model, FNODataset(fit_files, norm, grid), fit_files, norm)
                print(f'  {name}: test R2 {np.mean([r["r2"] for r in test[-len(te_files):]]):.4f}', flush=True)
            res.setdefault(variant, {})[geom] = {'test': summary(test), 'train': summary(train)}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, indent=1))
            t, r = res[variant][geom]['test'], res[variant][geom]['train']
            print(f"{geom} {variant}: test R2 {t['r2_mean']:.4f} (train {r['r2_mean']:.4f})  "
                  f"|peak| {t['abs_peak_err_K']:.2f} K  hotspot-T {t['abs_hotspot_temp_err_K']:.2f} K  "
                  f"loc {t['loc_err_um_median']:.0f} um  recall {t['recall_mean']:.3f}", flush=True)


if __name__ == '__main__':
    main()
