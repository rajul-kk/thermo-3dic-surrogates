"""ThermoNO on IC-ThermBench S2-S4 (report §9.29): their split, their metric code, their val split for selection.
IC-ThermBench is a single-layer 64x64 grid with no layered stack, so there is no classical backbone here: the
model predicts T - T_amb directly, exactly linear in the power map and gated by layout / conductivity / cooling.
Usage: python scripts/icb_thermono.py --scope level2 [--epochs 100] [--ch 32] [--device cuda] [--limit 512]
       python scripts/icb_thermono.py --scope level4 --transfer level5     (their S5 zero-shot protocol)
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
from src.fno.thermono import ThermoNO, peak_weighted_mse

PUBLISHED = {'level2': ('Therm-FM', 0.4427, 'SAU-FNO', 0.7028), 'level3': ('Therm-FM', 0.7161, 'U-FNO', 0.8016),
             'level4': ('Therm-FM', 0.9334, 'SAU-FNO', 1.2158), 'level5': ('Therm-FM', 15.51, 'U-Net', 19.10)}
RIDGE_PER_GEOM = {'level2': 1.943, 'level3': 2.488, 'level4': 3.203}   # report §9.13


def features(x, channels):
    """(B, X, Y, Z, P) -> power (B, 1, X, Y, Z), power-free geometry (B, G, X, Y, Z), ambient (B,) or None."""
    ch = {c: x[..., i] for i, c in enumerate(channels)}
    q = ch['chiplet_power'][:, None] / 100.0
    geo = [ch['grid_x'] / 32.0 - 1.0, ch['grid_y'] / 32.0 - 1.0]
    if 'local_thermal_k' in ch:
        geo.append(np.log10(ch['local_thermal_k']) / 2.0)
    if 'h_w_m2k' in ch:
        geo.append(np.log10(ch['h_w_m2k']) - 3.5)
    if 'r_convec_k_per_w' in ch:
        geo.append(np.log10(ch['r_convec_k_per_w']) + 2.0)
    amb = ch['ambient_K'].reshape(len(x), -1)[:, 0] if 'ambient_K' in ch else None
    return q.astype(np.float32), np.stack(geo, 1).astype(np.float32), amb


class Model(torch.nn.Module):
    """T = T_amb + b + scale * ThermoNO(power | geometry). b is one learned offset (the fixed ambient of S2/S3)."""

    def __init__(self, n_geo, args):
        super().__init__()
        self.net = ThermoNO((64, 64, 1), ch=args.ch, n_blocks=args.blocks, modes=(args.modes, args.modes),
                            lin_in=1, geo_in=n_geo, geo_arch=args.geo_arch, geo_attn=args.geo_attn,
                            local_k=args.local_k, multiscale=args.multiscale)
        self.b = torch.nn.Parameter(torch.zeros(()))
        self.scale = args.th_scale

    def forward(self, q, geo, amb):
        return amb[:, None, None, None] + self.b + self.scale * self.net(q, geo)


def batches(n, bs, shuffle, dev):
    idx = torch.randperm(n, device=dev) if shuffle else torch.arange(n, device=dev)
    return idx.split(bs)


def predict(model, q, geo, amb, bs, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for b in batches(len(q), bs, False, dev):
            out.append(model(q[b], geo[b], amb[b]).cpu())
    return torch.cat(out).numpy()


def to_dev(split_x, split_y, channels, default_amb, dev, limit=None):
    x, y = (split_x, split_y) if limit is None else (split_x[:limit], split_y[:limit])
    q, geo, amb = features(x, channels)
    amb = np.full(len(x), default_amb, np.float32) if amb is None else amb.astype(np.float32)
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)
    return f(q), f(geo), f(amb), f(y.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', default='level2')
    ap.add_argument('--transfer', default=None, help='also score zero-shot on this scope (their S5 protocol)')
    ap.add_argument('--data-root', type=Path, default=Path('data/ic-thermbench/datasets'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--ch', type=int, default=32)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--modes', type=int, default=32)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--w-top', type=float, default=0.0, help='peak weighting; 0 = plain MSE, matching their RMSE')
    ap.add_argument('--patience', type=int, default=15, help='epochs without val improvement')
    ap.add_argument('--th-scale', type=float, default=20.0, help='K per unit of network output')
    ap.add_argument('--geo-arch', choices=['mlp', 'cno'], default='mlp')
    ap.add_argument('--geo-attn', action='store_true')
    ap.add_argument('--local-k', type=int, default=1, help='lateral kernel of the linear local map (1 = original)')
    ap.add_argument('--multiscale', type=int, default=0, help='levels of the linear multiscale branch (0 = off)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None, help='smoke test: first N samples of each split')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    s = load_scope(args.data_root, args.scope)
    print(s.summary(), flush=True)
    amb0 = 298.15      # S2/S3 carry no ambient channel; the learned offset b absorbs any difference
    tr = to_dev(s.x_train, s.y_train, s.channels, amb0, dev, args.limit)
    va = to_dev(s.x_val, s.y_val, s.channels, amb0, dev, args.limit)
    te = to_dev(s.x_test, s.y_test, s.channels, amb0, dev, args.limit)
    model = Model(tr[1].shape[1], args).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ThermoNO: {n_params:,} params, geo channels {tr[1].shape[1]}, device {dev}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(batches(len(tr[0]), args.batch, False, dev)))
    ones = torch.ones_like(tr[3][:1])
    best, best_state, stale, t0 = np.inf, None, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for b in batches(len(tr[0]), args.batch, True, dev):
            opt.zero_grad()
            pred = model(tr[0][b], tr[1][b], tr[2][b])
            loss = peak_weighted_mse(pred, tr[3][b], ones.expand_as(pred), w_top=args.w_top)
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(model, *va[:3], 256, dev)
        rmse = score(pv.reshape(len(pv), -1), va[3].cpu().numpy().reshape(len(pv), -1), 'v')['rmse']
        if rmse < best - 1e-5:
            best, best_state, stale = rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if ep % 5 == 0 or stale == 0:
            print(f'  epoch {ep:3d}  val RMSE {rmse:.4f}  best {best:.4f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)

    res = {'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           'n_params': n_params, 'epochs_run': ep + 1, 'val_rmse_best': best, 'minutes': (time.time() - t0) / 60}
    pt = predict(model, *te[:3], 256, dev)
    res[args.scope] = score(pt.reshape(len(pt), -1), te[3].cpu().numpy().reshape(len(pt), -1), 't')
    if args.transfer:
        t = load_scope(args.data_root, args.transfer, split_data=False)
        assert t.channels == s.channels, (t.channels, s.channels)
        tt = to_dev(t.x_test, t.y_test, t.channels, amb0, dev, args.limit)
        p5 = predict(model, *tt[:3], 256, dev)
        res[args.transfer] = score(p5.reshape(len(p5), -1), tt[3].cpu().numpy().reshape(len(p5), -1), 'z')

    for sc in [args.scope] + ([args.transfer] if args.transfer else []):
        b, bv, r, rv = PUBLISHED[sc]
        print(f'\n=== {sc} test (their metrics) | published {b} {bv:.4f}, {r} {rv:.4f} RMSE'
              + (f' | our ridge-per-geom {RIDGE_PER_GEOM[sc]:.3f}' if sc in RIDGE_PER_GEOM else ''))
        print('  ' + '  '.join(f'{k} {res[sc][k]:.4f}' for k in REPORT_KEYS), flush=True)
    out = args.out or Path(f'results/icb_thermono_{args.scope}_seed{args.seed}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print('saved', out)


if __name__ == '__main__':
    main()
