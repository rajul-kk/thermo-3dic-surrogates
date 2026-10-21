"""Leave-cases-out (LOCO) test on IC-ThermBench S4 (report section 9.31, 'LOCO pre-registration').
Cases are identified by domain size (max grid_x, max grid_y per sample, fixed orientation); this equals sample index mod 10.
Two cases are held out. Both arms are tested on the SAME samples: the held-out cases' samples in the benchmark test split (indices >= 12000, 300 per case).
  --arm loco   : train/val drawn only from the 8 other cases
  --arm random : train/val drawn from all 10 cases (so the held-out cases are seen)
Same train size (--train-n) and val size (--val-n) in both arms; subsets come from the benchmark train (idx < 10800) / val (10800-11999) splits.
Models: unet (black-box), thermono, physsolve, ridge (pooled), rule (power-scaling). Metric: mean over samples of per-sample RMSE (K).
Usage: python scripts/icb_loco.py --model unet --arm loco --seed 0 --out results/icb_loco/unet_loco_s0.json
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.ic_thermbench_data import load_scope
from scripts.icb_thermono import Model as ThermoNOModel, batches, predict, to_dev
from src.hybrid.learned_conductance import LearnedConductanceSolver, _Block

HELD_OUT = [(37, 37), (55, 52)]       # fixed in the pre-registration, chosen from domain sizes only
N_TRAIN_END, N_VAL_END, N_TOTAL = 10800, 12000, 15000


class UNetBB(nn.Module):
    """Black-box baseline: U-Net regressing T - T_amb directly from [power, geometry]. Not linear in power by construction."""

    def __init__(self, n_geo, ch=32, levels=3, th_scale=20.0):
        super().__init__()
        self.inc = _Block(1 + n_geo, ch)
        self.down = nn.ModuleList(_Block(ch * 2 ** i, ch * 2 ** (i + 1)) for i in range(levels))
        self.up = nn.ModuleList(_Block(ch * 2 ** (i + 1) + ch * 2 ** i, ch * 2 ** i) for i in reversed(range(levels)))
        self.head = nn.Conv2d(ch, 1, 1)
        self.b = nn.Parameter(torch.zeros(()))
        self.scale = th_scale

    def forward(self, q, geo, amb):
        x = self.inc(torch.cat([q[..., 0], geo[..., 0]], 1))
        skips = [x]
        for d in self.down:
            x = d(F.avg_pool2d(x, 2, ceil_mode=True))
            skips.append(x)
        skips.pop()
        for u in self.up:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False), s], 1))
        return (amb[:, None, None] + self.b + self.scale * self.head(x)[:, 0])[..., None]


def per_sample_rmse(p, y):
    p, y = p.reshape(len(p), -1).astype(np.float64), y.reshape(len(y), -1).astype(np.float64)
    return np.sqrt(((p - y) ** 2).mean(1))


def case_keys(x):
    return [(int(a), int(b)) for a, b in zip(x[..., 1].reshape(len(x), -1).max(1), x[..., 2].reshape(len(x), -1).max(1))]


def ridge_features(x, ch):
    """Pooled ridge inputs: power map, log k map, grid_x, grid_y maps (flattened) + log h, log r_conv, T_amb scalars."""
    g = lambda n: x[..., ch.index(n)].reshape(len(x), -1).astype(np.float32)
    s = lambda n: x[..., ch.index(n)].reshape(len(x), -1)[:, :1].astype(np.float32)
    return np.concatenate([g('chiplet_power') / 100.0, np.log10(g('local_thermal_k')), g('grid_x') / 32.0, g('grid_y') / 32.0,
                           np.log10(s('h_w_m2k')), np.log10(s('r_convec_k_per_w')), s('ambient_K') / 300.0], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['unet', 'thermono', 'physsolve', 'ridge', 'rule'], required=True)
    ap.add_argument('--arm', choices=['loco', 'random'], required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-root', type=Path, default=Path('data/ic-thermbench/datasets'))
    ap.add_argument('--train-n', type=int, default=2000)
    ap.add_argument('--val-n', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--min-epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=None, help='smoke test only')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    s = load_scope(args.data_root, 'level4', split_data=False, fix_orientation=True)
    x, y, ch = s.x_test, s.y_test, s.channels
    assert len(x) == N_TOTAL
    keys = case_keys(x)
    cnt = Counter(keys)
    assert len(cnt) == 10 and all(v == 1500 for v in cnt.values())
    assert all(keys[i] == keys[i % 10] for i in range(N_TOTAL))          # case == index mod 10
    held = np.array([k in HELD_OUT for k in keys])
    idx = np.arange(N_TOTAL)
    pools = {}
    for name, lo, hi in (('train', 0, N_TRAIN_END), ('val', N_TRAIN_END, N_VAL_END)):
        m = (idx >= lo) & (idx < hi)
        pools[name] = idx[m & ~held] if args.arm == 'loco' else idx[m]
    rng = np.random.default_rng(args.seed)
    tr_i = np.sort(rng.choice(pools['train'], args.train_n, replace=False))
    va_i = np.sort(rng.choice(pools['val'], args.val_n, replace=False))
    te_i = idx[(idx >= N_VAL_END) & held]                                  # same test samples in both arms
    assert not (set(tr_i) & set(te_i)) and not (set(va_i) & set(te_i))
    if args.arm == 'loco':
        assert not held[tr_i].any() and not held[va_i].any()
    te_case = np.array([HELD_OUT.index(keys[i]) for i in te_i])
    print(args.model, args.arm, 'seed', args.seed, 'train', len(tr_i), 'val', len(va_i), 'test', len(te_i),
          'held-out train samples', int(held[tr_i].sum()), flush=True)

    res = {'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           'held_out': HELD_OUT, 'n_train': len(tr_i), 'n_val': len(va_i), 'n_test': len(te_i),
           'train_held_out_samples': int(held[tr_i].sum())}
    t0 = time.time()
    amb0 = 298.15
    ia, ip = ch.index('ambient_K'), ch.index('chiplet_power')

    if args.model in ('rule', 'ridge'):
        A = lambda a: a.reshape(len(a), -1)[:, :].astype(np.float64)
        if args.model == 'rule':
            P = lambda xx: xx[..., ip].reshape(len(xx), -1).sum(1).astype(np.float64)
            Am = lambda xx: xx[..., ia].reshape(len(xx), -1).mean(1).astype(np.float64)
            rise = (y[tr_i].astype(np.float64) - Am(x[tr_i])[:, None, None, None]).mean(0)
            pred = Am(x[te_i])[:, None, None, None] + rise[None] * (P(x[te_i]) / P(x[tr_i]).mean())[:, None, None, None]
        else:
            ftr, fva, fte = ridge_features(x[tr_i], ch), ridge_features(x[va_i], ch), ridge_features(x[te_i], ch)
            mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
            ftr, fva, fte = ((f - mu) / sd for f in (ftr, fva, fte))
            tgt = lambda i: (y[i].astype(np.float64).reshape(len(i), -1) - x[i][..., ia].reshape(len(i), -1)[:, :1])
            Ytr, ym = tgt(tr_i), tgt(tr_i).mean(0)
            K = ftr.astype(np.float64) @ ftr.astype(np.float64).T
            best, bl = np.inf, None
            for lam in 10.0 ** np.arange(-2, 7):
                alpha = np.linalg.solve(K + lam * np.eye(len(K)), Ytr - ym)
                pv = fva @ (ftr.T @ alpha) + ym + x[va_i][..., ia].reshape(len(va_i), -1)[:, :1]
                r = per_sample_rmse(pv, y[va_i]).mean()
                if r < best:
                    best, bl = r, lam
            alpha = np.linalg.solve(K + bl * np.eye(len(K)), Ytr - ym)
            pred = (fte @ (ftr.T @ alpha) + ym + x[te_i][..., ia].reshape(len(te_i), -1)[:, :1])
            res.update(ridge_lambda=bl, val_rmse_best=best)
    else:
        tr, va, te = (to_dev(x[i], y[i], ch, amb0, dev, args.limit) for i in (tr_i, va_i, te_i))
        n_geo = tr[1].shape[1]
        if args.model == 'physsolve':
            model = LearnedConductanceSolver(n_geo, ch=32, tol=1e-6).to(dev)
        elif args.model == 'unet':
            model = UNetBB(n_geo).to(dev)
        else:
            model = ThermoNOModel(n_geo, SimpleNamespace(ch=32, blocks=4, modes=32, geo_arch='mlp', geo_attn=False,
                                                         th_scale=20.0, local_k=1, multiscale=0)).to(dev)
        res['n_params'] = sum(p.numel() for p in model.parameters())
        steps = args.epochs * len(batches(len(tr[0]), args.batch, False, dev))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
        best, best_state, stale, curve = np.inf, None, 0, []
        for ep in range(args.epochs):
            model.train()
            for b in batches(len(tr[0]), args.batch, True, dev):
                opt.zero_grad()
                loss = ((model(tr[0][b], tr[1][b], tr[2][b]) - tr[3][b]) ** 2).mean()
                loss.backward()
                opt.step()
                sched.step()
            pv = predict(model, *va[:3], 64, dev)
            r = per_sample_rmse(pv, va[3].cpu().numpy()).mean()
            curve.append(float(r))
            if r < best - 1e-5:
                best, best_state, stale = r, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            print(f'  epoch {ep:3d} val {r:.4f} best {best:.4f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
            if stale >= args.patience and ep + 1 >= args.min_epochs:
                break
        model.load_state_dict(best_state)
        pred = predict(model, *te[:3], 64, dev)
        res.update(epochs_run=ep + 1, val_rmse_best=float(best), val_curve=curve, hit_cap=bool(ep + 1 >= args.epochs))

    ps = per_sample_rmse(pred, y[te_i][: len(pred)])
    cs = te_case[: len(ps)]
    res['rmse_per_case'] = {f'{HELD_OUT[c][0]}x{HELD_OUT[c][1]}': float(ps[cs == c].mean()) for c in range(2)}
    res['rmse_mean_cases'] = float(np.mean(list(res['rmse_per_case'].values())))
    res['minutes'] = (time.time() - t0) / 60
    print(res['rmse_per_case'], res['rmse_mean_cases'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
