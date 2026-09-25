"""FNO vs LT-FNO (report §9.27): does replacing FNO's spectral-in-z with per-mode layer coupling help?
5-fold CV on a v5 layout dataset, raw temperature target, identical budget and inputs for both.
Usage: python scripts/ltfno_compare.py geometry4 [--variants fno lt-fno] [--ch 16] [--epochs 300]
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.mesh import points_to_grid
from src.fno.data_loader import FNODataset, predict_to_flat
from src.fno.ltfno import build_lt_fno
from src.fno.model import build_fno
from src.fno.trainer import FNOTrainer
from src.pinn.data_loader import compute_norm_stats
from scripts.hotspot_eval import hotspot_metrics, kfold_indices
from scripts.layout_cv import per_scenario_stats


def build(variant, grid, args):
    kw = dict(modes=(args.modes, args.modes, args.modes_z), hidden_ch=args.ch, n_blocks=args.blocks)
    return build_fno(grid, **kw) if variant == 'fno' else build_lt_fno(grid, **kw)


def run(geom, variant, args):
    files = sorted(Path(f'data/3d-ice-layout-{geom}').rglob(f'{geom}_*.npz'))
    c0 = np.load(files[0])['coords']
    grid = tuple(len(np.unique(c0[:, i])) for i in (1, 0, 2))           # (length, width, z)
    rows, secs, n_params = [], [], None
    for fi, fold in enumerate(kfold_indices(len(files), 5, args.seed)):
        tr = [i for i in range(len(files)) if i not in set(fold.tolist())]
        # early stopping on scenarios held out of the TRAINING fold, never on the test fold
        rng = np.random.default_rng(args.seed + fi)
        val = set(rng.choice(tr, size=max(3, len(tr) // 9), replace=False).tolist())
        fit_files = [files[i] for i in tr if i not in val]
        val_files = [files[i] for i in sorted(val)]
        te_files = [files[i] for i in fold]
        norm = compute_norm_stats(fit_files, {geom: get_geometry_by_name(geom)})
        fit_ds, val_ds, te_ds = (FNODataset(f, norm, grid) for f in (fit_files, val_files, te_files))
        torch.manual_seed(args.seed)
        model = build(variant, grid, args)
        n_params = sum(p.numel() for p in model.parameters())
        with tempfile.TemporaryDirectory() as td:
            trainer = FNOTrainer(model, norm, fit_ds, val_ds, output_dir=Path(td), batch_size=args.batch,
                                 epochs=args.epochs, lr=args.lr, device=torch.device('cpu'), use_amp=False,
                                 log_interval=args.log_interval, patience=args.patience,
                                 geometry_name=f'{geom}_{variant}_fold{fi}')
            t = time.time()
            ckpt = trainer.train()
            secs.append(time.time() - t)
            model.load_state_dict(torch.load(ckpt, map_location='cpu')['model_state'])
        for j, i in enumerate(fold):
            pred, true = predict_to_flat(model, te_ds.items[j], torch.device('cpu'), norm)
            c = np.load(files[i])['coords'].astype(np.float64)
            cg = np.stack(points_to_grid(c, c[:, 0], c[:, 1], c[:, 2]), -1).reshape(-1, 3)
            rows.append({**per_scenario_stats(true, pred), **hotspot_metrics(pred, true, cg)})
        print(f'  {variant} fold {fi}: {secs[-1] / 60:.1f} min, fold R2 '
              f'{np.mean([r["r2"] for r in rows[-len(fold):]]):.4f}', flush=True)
    return {'n_params': n_params, 'minutes_total': float(np.sum(secs) / 60),
            'r2_mean': float(np.mean([r['r2'] for r in rows])),
            'r2_median': float(np.median([r['r2'] for r in rows])),
            'det_mae_K': float(np.mean([r['det_mae'] for r in rows])),
            'loc_err_um_median': float(np.median([r['hotspot_loc_err_um'] for r in rows])),
            'recall_mean': float(np.mean([r['top1pct_recall'] for r in rows])),
            'abs_peak_err_K': float(np.mean([r['abs_peak_temp_err_K'] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('geometry')
    ap.add_argument('--variants', nargs='*', default=['fno', 'lt-fno'])
    ap.add_argument('--ch', type=int, default=16)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--modes', type=int, default=12)
    ap.add_argument('--modes-z', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--log-interval', type=int, default=5)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out = Path(f'results/ltfno_{args.geometry}_seed{args.seed}.json')
    res = json.loads(out.read_text()) if out.exists() else {}
    for v in args.variants:
        print(f'=== {v} on {args.geometry}-shelf', flush=True)
        res[v] = run(args.geometry, v, args)
        res[v]['args'] = vars(args)
        out.write_text(json.dumps(res, indent=1))
        s = res[v]
        print(f"{v:<7} params {s['n_params']:,}  R2 {s['r2_mean']:.4f} (med {s['r2_median']:.4f})  "
              f"det.MAE {s['det_mae_K']:.3f} K  loc {s['loc_err_um_median']:.0f} um  "
              f"recall {s['recall_mean']:.3f}  |peak| {s['abs_peak_err_K']:.2f} K  ({s['minutes_total']:.0f} min)",
              flush=True)


if __name__ == '__main__':
    main()
