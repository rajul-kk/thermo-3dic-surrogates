"""Train ThermoNO on the layered backbone's residual, 5-fold CV on a v5 layout dataset (report §9.26).
Usage: python scripts/thermono_train.py geometry4 [--epochs 300] [--ch 16] [--seed 0] [--folds 0 1 2 3 4]
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import place_chiplets
from src.fno.thermono import ThermoNO, peak_weighted_mse
from src.hybrid import layered_backbone as lb
from src.validation import fv_solver as fv
from scripts.hotspot_eval import hotspot_metrics, kfold_indices
from scripts.layout_cv import per_scenario_stats

CACHE = Path('results/thermono_cache')


def build(geom_name):
    """Per-scenario tensors on the FV grid (x = width, y = length, z), cached per geometry."""
    path = CACHE / f'{geom_name}.npz'
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    files = sorted(Path(f'data/3d-ice-layout-{geom_name}').rglob(f'{geom_name}_*.npz'))
    Q, BB, TH, MASK, KL, KV, HTC, TAMB, COORDS, TEMP = [], [], [], [], [], [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        m = d['metadata'].item()
        offs = {k[len('placement_dx_'):]: (float(m[k]), float(m['placement_dy_' + k[len('placement_dx_'):]]))
                for k in m if k.startswith('placement_dx_')}
        offs = {('' if k == 'blocks' else k): v for k, v in offs.items()}
        geom = place_chiplets(get_geometry_by_name(geom_name), offs)
        scen = {'power_blocks': {b.name: float(m.get(f'block_power_{b.name}', 0.0)) for b in geom.power_blocks},
                'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
                'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')}
        g = fv.make_grid(geom)
        kl, kv = fv.conductivity(geom, scen, g)
        t_amb = scen['t_ambient'] + 273.15
        bb = lb.solve(geom, scen, g, kl, kv)['T'] - t_amb
        truth = fv.ice_to_grid(d['coords'].astype(np.float64), d['temp'].astype(np.float64) - t_amb, g)
        Q.append(fv.power(geom, scen, g)); BB.append(bb); TH.append(np.nan_to_num(truth))
        MASK.append(~np.isnan(truth)); KL.append(kl); KV.append(kv)
        HTC.append(scen['htc']); TAMB.append(t_amb)
        COORDS.append(d['coords'].astype(np.float64)); TEMP.append(d['temp'].astype(np.float64))
    dz = np.diff(g.ze)
    out = dict(q=np.stack(Q), bb=np.stack(BB), theta=np.stack(TH), mask=np.stack(MASK),
               kl=np.stack(KL), kv=np.stack(KV), htc=np.array(HTC), tamb=np.array(TAMB),
               xe=g.xe, ye=g.ye, ze=g.ze, dz=dz, coords=np.stack(COORDS), temp=np.stack(TEMP))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def tensors(D, idx, q_scale, th_scale):
    kv = D['kv'][idx]
    # thermal-resistance z-coordinate: cumulative dz/k from the sink, per column (the "warp" lead)
    r = np.cumsum(D['dz'][None, None, None, :] / kv, axis=-1)
    r = r / r[..., -1:]
    htc = np.broadcast_to(np.log10(D['htc'][idx])[:, None, None, None], kv.shape)
    geo = np.stack([np.log10(D['kl'][idx]) / 2.0, np.log10(kv) / 2.0, r, htc - 4.0], 1)
    lin = np.stack([D['q'][idx] / q_scale, D['bb'][idx] / th_scale], 1)
    target = (D['theta'][idx] - D['bb'][idx]) / th_scale
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
    return f(lin), f(geo), f(target), f(D['mask'][idx].astype(np.float32))


def run_fold(D, tr, te, args):
    torch.manual_seed(args.seed)
    q_scale = float(D['q'][tr].std()) or 1.0
    th_scale = float(D['bb'][tr].std()) or 1.0
    rng = np.random.default_rng(args.seed)
    val = rng.choice(tr, size=max(3, len(tr) // 9), replace=False)      # early-stopping holdout
    fit = np.setdiff1d(tr, val)
    lin, geo, tgt, msk = tensors(D, fit, q_scale, th_scale)
    vlin, vgeo, vtgt, vmsk = tensors(D, val, q_scale, th_scale)
    grid = D['q'].shape[1:]
    model = ThermoNO(grid, ch=args.ch, n_blocks=args.blocks, modes=(args.modes, args.modes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state, stale = np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        for b in torch.randperm(len(fit)).split(args.batch):
            opt.zero_grad()
            loss = peak_weighted_mse(model(lin[b], geo[b]), tgt[b], msk[b], w_top=args.w_top)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 4:
            model.eval()
            with torch.no_grad():
                v = peak_weighted_mse(model(vlin, vgeo), vtgt, vmsk, w_top=args.w_top).item()
            if v < best - 1e-6:
                best, best_state, stale = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    tlin, tgeo, _, _ = tensors(D, te, q_scale, th_scale)
    with torch.no_grad():
        corr = model(tlin, tgeo).numpy() * th_scale
    g = fv.FVGrid(D['xe'], D['ye'], D['ze'], None)
    rows = {'backbone': [], 'thermono': []}
    for j, i in enumerate(te):
        c, y, t_amb = D['coords'][i], D['temp'][i], float(D['tamb'][i])
        for name, field in (('backbone', D['bb'][i]), ('thermono', D['bb'][i] + corr[j])):
            p = lb.sample(field, g, c) + t_amb
            rows[name].append({**per_scenario_stats(y, p), **hotspot_metrics(p, y, c)})
    return rows, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('geometry')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--ch', type=int, default=16)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--modes', type=int, default=16)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--w-top', type=float, default=4.0)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--folds', nargs='*', type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    D = build(args.geometry)
    n = D['q'].shape[0]
    all_rows = {'backbone': [], 'thermono': []}
    for fi, fold in enumerate(kfold_indices(n, 5, args.seed)):
        if fi not in args.folds:
            continue
        tr = np.setdiff1d(np.arange(n), fold)
        t = time.time()
        rows, eps = run_fold(D, tr, fold, args)
        for k in rows:
            all_rows[k] += rows[k]
        r = {k: np.mean([x['r2'] for x in v]) for k, v in rows.items()}
        print(f'fold {fi}: {eps} epochs, {time.time() - t:.0f}s | R2 backbone {r["backbone"]:.4f} '
              f'thermono {r["thermono"]:.4f}', flush=True)
    summary = {}
    for name, rr in all_rows.items():
        summary[name] = {'r2_mean': float(np.mean([r['r2'] for r in rr])),
                         'r2_median': float(np.median([r['r2'] for r in rr])),
                         'det_mae_K': float(np.mean([r['det_mae'] for r in rr])),
                         'loc_err_um_median': float(np.median([r['hotspot_loc_err_um'] for r in rr])),
                         'recall_mean': float(np.mean([r['top1pct_recall'] for r in rr])),
                         'abs_peak_err_K': float(np.mean([r['abs_peak_temp_err_K'] for r in rr]))}
        s = summary[name]
        print(f"  {name:<10} R2 {s['r2_mean']:.4f} (med {s['r2_median']:.4f}) det.MAE {s['det_mae_K']:.3f} K "
              f"loc {s['loc_err_um_median']:.0f} um recall {s['recall_mean']:.3f} |peak| {s['abs_peak_err_K']:.2f} K")
    out = Path(f'results/thermono_{args.geometry}_seed{args.seed}.json')
    out.write_text(json.dumps({'args': vars(args), 'n_test': len(all_rows['thermono']), 'summary': summary}, indent=1))


if __name__ == '__main__':
    main()
