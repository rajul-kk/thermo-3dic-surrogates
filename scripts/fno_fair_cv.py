"""A fair training protocol for the raw-temperature FNO family (report §9.27 follow-up).
§9.27 found the notebook's FNOs underfit: the loss (relative L2 on absolute temperature min-max scaled over ~300 K)
was dominated by the per-scenario offset, and early stopping ended training before the one-cycle schedule annealed.
Here the target is the temperature RISE over ambient (ambient is known), the loss adds a detrended term so spatial
structure carries weight, and early stopping is held off until --min-epochs. Same folds and held-out-from-training
early-stopping split as the notebook; scored with the §9.26 metrics.
Usage: python scripts/fno_fair_cv.py geometry4 [--variants fno lt-fno cno-fno-attn] [--seeds 0 1 2] [--device cuda]
       python scripts/fno_fair_cv.py geometry7 --data-dir "data/3d-ice-{g}-pilot"
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.mesh import points_to_grid
from src.fno.data_loader import FNODataset
from src.fno.ltfno import build_lt_fno
from src.fno.model import build_cno_fno, build_fno
from src.pinn.data_loader import compute_norm_stats
from scripts.hotspot_eval import hotspot_metrics, kfold_indices
from scripts.layout_cv import per_scenario_stats


def build(variant, grid, a, dev):
    if variant == 'fno':
        return build_fno(grid, modes=tuple(a.modes), hidden_ch=a.ch, n_blocks=a.blocks, device=dev)
    if variant == 'lt-fno':
        return build_lt_fno(grid, modes=tuple(a.modes), hidden_ch=a.ch, n_blocks=a.blocks, device=dev)
    return build_cno_fno(grid, ch=a.ch, n_fno_blocks=a.blocks, n_cno_layers=2, use_attention=True, n_heads=4, device=dev)


def stack(ds, files, norm, dev):
    """Inputs as batched tensors, and the rise target (K) on the model grid."""
    it = ds.items
    t_rng = norm.T_max - norm.T_min
    amb = np.array([float(np.load(f, allow_pickle=True)['metadata'].item().get('t_ambient_kelvin',
                    float(np.load(f, allow_pickle=True)['metadata'].item()['t_ambient_celsius']) + 273.15))
                    for f in files], dtype=np.float32)
    T = torch.stack([x['T_norm'] for x in it]) * t_rng + norm.T_min
    rise = T - torch.as_tensor(amb)[:, None, None, None]
    X = [torch.stack([x[k] for x in it]).to(dev) for k in ('Q_norm', 'layer_id_norm', 'htc_norm', 't_amb_norm', 'tsv_frac')]
    return X, rise.to(dev), torch.as_tensor(amb, device=dev)


def loss_fn(out, tgt):
    """MSE on the rise plus MSE on the detrended rise, so spatial structure is not swamped by the offset."""
    d = lambda z: z - z.mean(dim=(1, 2, 3), keepdim=True)
    return ((out - tgt) ** 2).mean() + ((d(out) - d(tgt)) ** 2).mean()


def run_fold(files, fold, fi, seed, variant, grid, geom, a, dev):
    te = set(fold.tolist())
    tr_files = [files[i] for i in range(len(files)) if i not in te]
    rng = np.random.default_rng(seed + fi)
    val_idx = set(rng.choice(len(tr_files), size=max(3, len(tr_files) // 9), replace=False).tolist())
    fit_f = [f for k, f in enumerate(tr_files) if k not in val_idx]
    val_f = [f for k, f in enumerate(tr_files) if k in val_idx]
    te_f = [files[i] for i in fold]
    norm = compute_norm_stats(fit_f, {geom: get_geometry_by_name(geom)})
    Xf, Rf, _ = stack(FNODataset(fit_f, norm, grid), fit_f, norm, dev)
    Xv, Rv, _ = stack(FNODataset(val_f, norm, grid), val_f, norm, dev)
    Xt, Rt, At = stack(FNODataset(te_f, norm, grid), te_f, norm, dev)
    scale = float(Rf.std())
    torch.manual_seed(seed)
    model = build(variant, grid, a, dev)
    steps = a.epochs * int(np.ceil(len(fit_f) / a.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    best, best_state, stale, t0 = np.inf, None, 0, time.time()
    fwd = lambda X, idx: model(*[x[idx] for x in X])
    for ep in range(a.epochs):
        model.train()
        for b in torch.randperm(len(fit_f), device=dev).split(a.batch):
            opt.zero_grad()
            loss_fn(fwd(Xf, b), Rf[b] / scale).backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss_fn(fwd(Xv, torch.tensor([i], device=dev)), Rv[i:i + 1] / scale).item()
                               for i in range(len(val_f))]))
        if v < best - 1e-7:
            best, best_state, stale = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= a.patience and ep + 1 >= a.min_epochs:
            break
    model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for j, f in enumerate(te_f):
            out = fwd(Xt, torch.tensor([j], device=dev))[0] * scale
            pred = (out + At[j]).cpu().numpy().ravel().astype(np.float64)
            true = (Rt[j] + At[j]).cpu().numpy().ravel().astype(np.float64)
            c = np.load(f)['coords'].astype(np.float64)
            cg = np.stack(points_to_grid(c, c[:, 0], c[:, 1], c[:, 2]), -1).reshape(-1, 3)
            rows.append({**per_scenario_stats(true, pred), **hotspot_metrics(pred, true, cg)})
    return rows, ep + 1, time.time() - t0, sum(p.numel() for p in model.parameters())


def summary(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    return {'r2_mean': m('r2'), 'r2_median': float(np.median([r['r2'] for r in rows])), 'det_mae_K': m('det_mae'),
            'loc_err_um_median': float(np.median([r['hotspot_loc_err_um'] for r in rows])),
            'recall_mean': m('top1pct_recall'), 'abs_peak_err_K': m('abs_peak_temp_err_K'),
            'n_negative_r2': int(sum(r['r2'] < 0 for r in rows)), 'n': len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('geometry')
    ap.add_argument('--data-dir', default='data/3d-ice-layout-{g}')
    ap.add_argument('--variants', nargs='*', default=['fno', 'lt-fno', 'cno-fno-attn'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--ch', type=int, default=32)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--modes', nargs=3, type=int, default=[16, 16, 8])
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--min-epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    files = sorted(Path(a.data_dir.format(g=a.geometry)).rglob(f'{a.geometry}_*.npz'))
    c0 = np.load(files[0])['coords']
    grid = tuple(len(np.unique(c0[:, i])) for i in (1, 0, 2))
    out = a.out or Path(f'results/fno_fair/{a.geometry}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    print(f'{a.geometry}: {len(files)} scenarios, grid {grid}, device {dev}', flush=True)
    for variant in a.variants:
        for seed in a.seeds:
            key = f'{variant}_seed{seed}'
            if key in res:
                print(key, 'done'); continue
            rows, eps, secs, n_params = [], [], 0.0, 0
            for fi, fold in enumerate(kfold_indices(len(files), 5, seed), 1):
                r, e, s, n_params = run_fold(files, fold, fi, seed, variant, grid, a.geometry, a, dev)
                rows += r; eps.append(e); secs += s
                print(f'  {key} fold {fi}: {e} epochs, R2 {np.mean([x["r2"] for x in r]):.3f} ({s / 60:.1f} min)', flush=True)
            res[key] = {**summary(rows), 'epochs': eps, 'minutes': secs / 60, 'n_params': n_params}
            out.write_text(json.dumps(res, indent=1))
            s = res[key]
            print(f"{key}: R2 {s['r2_mean']:.3f} (med {s['r2_median']:.3f}) det.MAE {s['det_mae_K']:.3f} K "
                  f"loc {s['loc_err_um_median']:.0f} um recall {s['recall_mean']:.3f} |peak| {s['abs_peak_err_K']:.2f} K",
                  flush=True)


if __name__ == '__main__':
    main()
