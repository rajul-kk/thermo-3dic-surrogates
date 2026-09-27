"""Learned-conductance solver on IC-ThermBench (report §9.31): learn the physics, then solve it exactly.
Same protocol as scripts/icb_thermono.py: their split, their metric code, checkpoint selection on their val split, S5
scored zero-shot with the frozen S4 model. --train-n caps the training set (random subset, fixed seed) to measure
how fast the physics is learned; val and test are always full.
Usage: python scripts/icb_physsolve.py --scope level4 --transfer level5 [--train-n 1080] [--epochs 60] [--device cuda]
       python scripts/icb_physsolve.py --model thermono --scope level4 --transfer level5 --train-n 1080   (same budget)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.ic_thermbench_baselines import REPORT_KEYS, score
from scripts.ic_thermbench_data import load_scope
from scripts.icb_thermono import PUBLISHED, RIDGE_PER_GEOM, Model as ThermoNOModel, batches, predict, to_dev
from src.hybrid.learned_conductance import LearnedConductanceSolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['physsolve', 'thermono'], default='physsolve')
    ap.add_argument('--scope', default='level4')
    ap.add_argument('--transfer', default=None)
    ap.add_argument('--data-root', type=Path, default=Path('data/ic-thermbench/datasets'))
    ap.add_argument('--train-n', type=int, default=None, help='cap on training samples (random subset)')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--ch', type=int, default=32)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None, help='smoke test: first N samples of every split')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    s = load_scope(args.data_root, args.scope)
    print(s.summary(), flush=True)
    amb0 = 298.15
    xtr, ytr = s.x_train, s.y_train
    if args.train_n and args.train_n < len(xtr):
        pick = np.sort(np.random.default_rng(args.seed).choice(len(xtr), args.train_n, replace=False))
        xtr, ytr = xtr[pick], ytr[pick]
    tr = to_dev(xtr, ytr, s.channels, amb0, dev, args.limit)
    va = to_dev(s.x_val, s.y_val, s.channels, amb0, dev, args.limit)
    te = to_dev(s.x_test, s.y_test, s.channels, amb0, dev, args.limit)
    n_geo = tr[1].shape[1]
    if args.model == 'physsolve':
        model = LearnedConductanceSolver(n_geo, ch=args.ch, tol=args.tol).to(dev)
    else:
        from types import SimpleNamespace
        model = ThermoNOModel(n_geo, SimpleNamespace(ch=32, blocks=4, modes=32, geo_arch='mlp', geo_attn=False,
                                                     th_scale=20.0, local_k=1, multiscale=0)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{args.model}: {n_params:,} params, {len(tr[0])} training samples, device {dev}', flush=True)

    steps = args.epochs * len(batches(len(tr[0]), args.batch, False, dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    best, best_state, stale, t0, curve = np.inf, None, 0, time.time(), []
    for ep in range(args.epochs):
        model.train()
        for b in batches(len(tr[0]), args.batch, True, dev):
            opt.zero_grad()
            loss = ((model(tr[0][b], tr[1][b], tr[2][b]) - tr[3][b]) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(model, *va[:3], 64, dev)
        rmse = score(pv.reshape(len(pv), -1), va[3].cpu().numpy().reshape(len(pv), -1), 'v')['rmse']
        curve.append(rmse)
        if rmse < best - 1e-5:
            best, best_state, stale = rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        print(f'  epoch {ep:3d}  val RMSE {rmse:.4f}  best {best:.4f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        if stale >= args.patience and ep + 1 >= args.min_epochs:
            break
    model.load_state_dict(best_state)

    res = {'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, 'n_params': n_params,
           'n_train': len(tr[0]), 'epochs_run': ep + 1, 'val_rmse_best': best, 'val_curve': curve,
           'minutes': (time.time() - t0) / 60}
    pt = predict(model, *te[:3], 64, dev)
    res[args.scope] = score(pt.reshape(len(pt), -1), te[3].cpu().numpy().reshape(len(pt), -1), 't')
    if args.transfer:
        t = load_scope(args.data_root, args.transfer, split_data=False)
        assert t.channels == s.channels, (t.channels, s.channels)
        tt = to_dev(t.x_test, t.y_test, t.channels, amb0, dev, args.limit)
        p5 = predict(model, *tt[:3], 64, dev)
        res[args.transfer] = score(p5.reshape(len(p5), -1), tt[3].cpu().numpy().reshape(len(p5), -1), 'z')
    for sc in [args.scope] + ([args.transfer] if args.transfer else []):
        b, bv, r, rv = PUBLISHED[sc]
        print(f'\n=== {args.model} {sc} test | published {b} {bv:.4f}, {r} {rv:.4f}'
              + (f' | ridge-per-geom {RIDGE_PER_GEOM[sc]:.3f}' if sc in RIDGE_PER_GEOM else ''))
        print('  ' + '  '.join(f'{k} {res[sc][k]:.4f}' for k in REPORT_KEYS), flush=True)
    out = args.out or Path(f'results/icb_{args.model}_{args.scope}_n{len(tr[0])}_seed{args.seed}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print('saved', out)


if __name__ == '__main__':
    main()
