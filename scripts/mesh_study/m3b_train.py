"""M3b training + evaluation (docs/protocol_mesh_convergence.md section 10). Run after m3b_gen.py has produced all 440 samples."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

OUT = Path(r'D:\Work\3D-ICE Thermal-modelling\Thermo\results\mesh_convergence\m3b')
SD = OUT / 'samples'
FAMS = ('T0', 'T1', 'T2', 'T3')

def load(fam, idxs):
    X, Y, YC, CU, TC = [], [], [], [], []
    for i in idxs:
        z = np.load(SD / f'{fam}_{i:03d}.npz')
        X.append(np.stack([z['x_coarse_up'] - z['x_coarse_up'].mean(), (z['x_pow'].astype(np.float64) - 1.0) / 10.0]))
        Y.append(z['y_truth']); YC.append(z['y_classical']); CU.append(z['coarse_up_center']); TC.append(float(z['t_classical']))
    return dict(x=np.array(X, np.float32), y=np.array(Y), yc=np.array(YC), cu=np.array(CU), tcl=np.array(TC))

class Net(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        L, cin = [], 2
        for d in (1, 2, 4, 2, 1):
            L += [nn.Conv2d(cin, c, 3, padding=d, dilation=d), nn.ReLU()]; cin = c
        self.body = nn.Sequential(*L); self.out = nn.Conv2d(c, 1, 1)
    def forward(self, x): return self.out(self.body(x))[:, 0, 8:88, 8:88]

def d4(x, y, k):                      # one of the 8 flips/rotations, applied to a (B,C,96,96) input and (B,80,80) target
    if k >= 4: x, y = x.flip(-1), y.flip(-1)
    r = k % 4
    return torch.rot90(x, r, (-2, -1)), torch.rot90(y, r, (-2, -1))

def stats(pred_abs, truth):
    pk = pred_abs.reshape(len(pred_abs), -1).max(1) - truth.reshape(len(truth), -1).max(1)
    rm = np.sqrt(((pred_abs - truth) ** 2).reshape(len(truth), -1).mean(1))
    return np.abs(pk), rm

def run_seed(seed, tr, va, te, xs, ys):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n_par = sum(p.numel() for p in net.parameters())
    X = torch.tensor(tr['x'] / xs); Y = torch.tensor((tr['y'] - tr['cu']) / ys, dtype=torch.float32)
    Xv = torch.tensor(va['x'] / xs); Yv = (va['y'] - va['cu'])
    best, bad, best_state, ep_run = 1e9, 0, None, 0
    for ep in range(200):
        ep_run = ep + 1; net.train(); perm = torch.randperm(len(X))
        for b in range(0, len(X), 16):
            ib = perm[b:b + 16]; xb, yb = d4(X[ib], Y[ib], int(np.random.randint(8)))
            loss = ((net(xb) - yb) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = float(np.sqrt(((net(Xv).numpy() * ys - Yv) ** 2).mean()))
        if v < best - 1e-7: best, bad, best_state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 20: break
    net.load_state_dict(best_state); net.eval()
    res = dict(seed=seed, params=n_par, epochs=ep_run, best_val_rmse_K=best, fams={})
    for fam, d in te.items():
        with torch.no_grad(): pred = net(torch.tensor(d['x'] / xs)).numpy() * ys + d['cu']
        pk, rm = stats(pred, d['y'])
        res['fams'][fam] = dict(med_abs_peak_K=float(np.median(pk)), med_rmse_K=float(np.median(rm)))
    # latency: batch 1, one thread
    torch.set_num_threads(1); x1 = torch.tensor(te['T0']['x'][:1] / xs)
    with torch.no_grad():
        for _ in range(5): net(x1)
        t = []
        for _ in range(100):
            t0 = time.perf_counter(); net(x1); t.append(time.perf_counter() - t0)
    res['latency_s'] = float(np.median(t))
    return res

def main():
    tr = load('F0', range(0, 240)); va = load('F0', range(240, 280)); te = {f: load(f, range(40)) for f in FAMS}
    xs = np.array([tr['x'][:, 0].std(), tr['x'][:, 1].std()], np.float32)[None, :, None, None]
    ys = float((tr['y'] - tr['cu']).std())
    out = dict(n_train=240, n_val=40, input_scale=xs.ravel().tolist(), target_scale_K=ys, seeds=[], reference={})
    for fam, d in te.items():
        cpk, crm = stats(d['yc'], d['y']); bpk, brm = stats(d['cu'], d['y'])
        out['reference'][fam] = dict(classical=dict(med_abs_peak_K=float(np.median(cpk)), med_rmse_K=float(np.median(crm))),
                                     upsampled_coarse=dict(med_abs_peak_K=float(np.median(bpk)), med_rmse_K=float(np.median(brm))),
                                     classical_latency_s=float(np.median(d['tcl'])))
    for seed in (0, 1, 2):
        t0 = time.perf_counter(); r = run_seed(seed, tr, va, te, xs, ys); r['train_seconds'] = round(time.perf_counter() - t0, 1)
        out['seeds'].append(r); print('seed', seed, {k: v for k, v in r.items() if k != 'fams'}, flush=True)
        (OUT / 'results.json').write_text(json.dumps(out, indent=1))
    # decision rule (protocol section 10): all families, all seeds
    cl_lat = float(np.median([out['reference'][f]['classical_latency_s'] for f in FAMS]))
    checks = {}
    for f in FAMS:
        ref = out['reference'][f]['classical']
        for s in out['seeds']:
            m = s['fams'][f]
            checks[f'{f}_seed{s["seed"]}'] = dict(peak_ok=m['med_abs_peak_K'] <= 2 * ref['med_abs_peak_K'], rmse_ok=m['med_rmse_K'] <= 2 * ref['med_rmse_K'])
    out['latency_ok'] = max(s['latency_s'] for s in out['seeds']) <= 0.1 * cl_lat
    out['classical_latency_median_s'] = cl_lat
    out['accuracy_ok_all'] = all(c['peak_ok'] and c['rmse_ok'] for c in checks.values())
    out['checks'] = checks; out['pays'] = bool(out['latency_ok'] and out['accuracy_ok_all'])
    (OUT / 'results.json').write_text(json.dumps(out, indent=1))
    print('latency_ok', out['latency_ok'], 'accuracy_ok_all', out['accuracy_ok_all'], 'PAYS', out['pays'], flush=True)
    for f in FAMS:
        print(f, 'classical', out['reference'][f]['classical'], 'upsampled', out['reference'][f]['upsampled_coarse'],
              'learned seeds', [(round(s['fams'][f]['med_abs_peak_K'], 4), round(s['fams'][f]['med_rmse_K'], 4)) for s in out['seeds']], flush=True)

if __name__ == '__main__':
    main()
