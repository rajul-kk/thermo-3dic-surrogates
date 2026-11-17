"""Label-swap test on an outside benchmark: does a 1D FNO learn PDEBench Burgers' solver error? (protocol:
tools/fieldlint/bench/protocol_labelswap_burgers.md, frozen before any model was trained)

Old labels: PDEBench 1D_Burgers_Sols_Nu0.001 as stored, u(x, t_out) from u(x, 0). New labels: the same trajectories
re-solved from the stored u(x, 0) with a dealiased pseudo-spectral integrating-factor RK4 solver
(u_t + (u^2/2)_x = (nu/pi) u_xx, periodic on [0, 1)), sampled at the stored points. The same FNO is trained on each,
5-fold CV x 3 seeds, and fieldlint's label-swap analysis decides.

Stages (each resumable; outputs in results/labelswap_burgers/):
  labels   build the corrected labels and the label-only checks (no model)
  train    train and predict all folds, seeds and arms, saving after every unit (--seeds: a subset, for one session
           per seed on separate machines)
  merge    join per-session prediction files into predictions.partial.npz (refuses conflicting units)
  analyse  run the frozen decision rule and write verdict.json
Usage: python scripts/labelswap_burgers.py labels|train|merge|analyse [--threads 6] [--seeds 0 1 2] [FILES]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'fieldlint' / 'src'))
from fieldlint import labelswap as ls

OUT = ROOT / 'results' / 'labelswap_burgers'
DATA = Path(os.environ.get('PDEBENCH_BURGERS', ROOT / 'data' / 'pde-bench' / 'burgers_nu0.001_n400_times.npz'))

# frozen settings (protocol section 3); change only by a dated amendment
NU = 0.001
T_INDEX = 20                  # stored time index of the output, t = 0.2
N_SAMPLES = 400
REF_M, REF_DT = 16384, 2e-5    # reference grid and step (protocol section 3.2 convergence check)
REF_CHECK_N = 16               # samples re-solved at 2M points and dt/2 for the refinement check
SEEDS = (0, 1, 2)
FOLDS, FOLD_SEED = 5, 0
FNO = {'width': 32, 'modes': 16, 'layers': 4, 'epochs': 300, 'batch': 20, 'lr': 1e-3, 'weight_decay': 1e-4}
SUPPORT = 0.01                # |D| threshold of the support fields, in units of u


def resolve(u0, t_end, nu_eff, M=REF_M, dt=REF_DT):
    """Point values at the stored N points of the pseudo-spectral solution from point values u0 (..., N)."""
    N = u0.shape[-1]
    U = np.fft.rfft(u0, axis=-1)
    V = np.zeros(u0.shape[:-1] + (M // 2 + 1,), complex)
    V[..., :N // 2] = U[..., :N // 2] * (M / N)
    k = 2 * np.pi * np.fft.rfftfreq(M, 1.0 / M)
    keep = np.fft.rfftfreq(M, 1.0 / M) < M / 3          # 2/3 dealiasing of the quadratic term
    E = np.exp(-nu_eff * k ** 2 * dt / 2)
    E2 = E * E

    def nl(W):
        u = np.fft.irfft(W, n=M, axis=-1)
        return -0.5j * k * np.fft.rfft(u * u, axis=-1) * keep

    for _ in range(round(t_end / dt)):
        a = nl(V)
        b = nl(E * (V + dt / 2 * a))
        c = nl(E * V + dt / 2 * b)
        d = nl(E2 * V + dt * E * c)
        V = E2 * V + dt / 6 * (E2 * a + 2 * E * (b + c) + d)
    return np.fft.irfft(V, n=M, axis=-1)[..., ::M // N]


def _save_npz(path, **arrays):
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(ls.json_safe(obj), indent=1, allow_nan=False))
    os.replace(tmp, path)


def stage_labels(data):
    d = np.load(data)
    pos = int(np.flatnonzero(d['t_index'] == T_INDEX)[0])
    u0 = d['u'][:N_SAMPLES, 0].astype(np.float64)
    old = d['u'][:N_SAMPLES, pos].astype(np.float64)
    t_end = float(d['t'][T_INDEX])
    new = np.concatenate([resolve(u0[i:i + 25], t_end, NU / np.pi) for i in range(0, N_SAMPLES, 25)])
    fine = resolve(u0[:REF_CHECK_N], t_end, NU / np.pi, M=2 * REF_M, dt=REF_DT / 2)
    D = old - new
    spread = np.linalg.norm(old - old.mean(1, keepdims=True), axis=1)
    checks = {'t_out': t_end, 'x0': float(d['x'][0]), 'dx': float(d['x'][1] - d['x'][0]),
              'rel_D_median': float(np.median(np.linalg.norm(D, axis=1) / spread)),
              'max_abs_D': float(np.abs(D).max()),
              'ref_refinement_max_change': float(np.abs(fine - new[:REF_CHECK_N]).max()),
              'max_abs_D_same_samples': float(np.abs(D[:REF_CHECK_N]).max()), 'support_nodes_median': float(np.median((np.abs(D) > SUPPORT).sum(1)))}
    _save_npz(OUT / 'labels.npz', u0=u0, old=old, new=new)
    _save_json(OUT / 'label_checks.json', checks)
    print(json.dumps(checks, indent=1))


def _fno(width, modes, layers):
    import torch
    from torch import nn

    class Spectral(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / (width * width))

        def forward(self, x):
            X = torch.fft.rfft(x)
            out = torch.zeros(x.shape[0], width, X.shape[-1], dtype=torch.cfloat)
            out[..., :modes] = torch.einsum('bim,iom->bom', X[..., :modes], self.w)
            return torch.fft.irfft(out, n=x.shape[-1])

    class FNO1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Conv1d(2, width, 1)
            self.spec = nn.ModuleList(Spectral() for _ in range(layers))
            self.skip = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
            self.proj = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, 1, 1))

        def forward(self, u0):
            x = torch.linspace(0, 1, u0.shape[-1] + 1)[:-1].expand(u0.shape[0], 1, -1)
            h = self.lift(torch.cat([u0[:, None], x], 1))
            for i, (s, w) in enumerate(zip(self.spec, self.skip)):
                h = s(h) + w(h)
                if i < layers - 1:
                    h = nn.functional.gelu(h)
            return self.proj(h)[:, 0]

    return FNO1d()


def fit_predict(u0, y, tr, te, seed, hp=FNO):
    """Train one FNO on (u0[tr], y[tr]) with a fixed schedule; return predictions on u0[te]."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _fno(hp['width'], hp['modes'], hp['layers'])
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    steps = hp['epochs'] * int(np.ceil(len(tr) / hp['batch']))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp['lr'], total_steps=steps)
    X, Y = torch.tensor(u0, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    for _ in range(hp['epochs']):
        for b in np.array_split(rng.permutation(tr), int(np.ceil(len(tr) / hp['batch']))):
            opt.zero_grad()
            loss = torch.mean((model(X[b]) - Y[b]) ** 2)
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return model(X[te]).double().numpy()


def stage_train(seeds=SEEDS):
    lab = np.load(OUT / 'labels.npz')
    u0, labels = lab['u0'], {'old': lab['old'], 'new': lab['new']}
    part = OUT / 'predictions.partial.npz'
    pred = dict(np.load(part)) if part.exists() else {}
    folds = cv_folds(len(u0))
    for fi, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(u0)), te)
        for seed in seeds:
            for arm in ('old', 'new'):
                key = f'{arm}_s{seed}_f{fi}'
                if key in pred:
                    continue
                t = time.time()
                pred[key] = fit_predict(u0, labels[arm], tr, te, seed)
                _save_npz(part, **pred)
                print(f'{key}: {time.time() - t:.0f}s', flush=True)


def stage_merge(files):
    """Union of the units in ``files``; a unit present in two files must be identical."""
    merged = {}
    for f in files:
        for k, v in np.load(f).items():
            if k in merged and not np.array_equal(merged[k], v):
                raise SystemExit(f'{k} differs between files: refusing to merge')
            merged[k] = v
    want = {f'{arm}_s{s}_f{fi}' for arm in ('old', 'new') for s in SEEDS for fi in range(FOLDS)}
    missing = sorted(want - set(merged))
    _save_npz(OUT / 'predictions.partial.npz', **merged)
    print(f'{len(merged)} units merged; missing: {missing or "none"}')


def cv_folds(n):
    idx = np.arange(n)
    np.random.default_rng(FOLD_SEED).shuffle(idx)
    return np.array_split(idx, FOLDS)


def stage_analyse():
    lab = np.load(OUT / 'labels.npz')
    pred = np.load(OUT / 'predictions.partial.npz')
    folds = cv_folds(len(lab['u0']))
    order = np.concatenate(folds)
    P = {arm: np.stack([np.concatenate([pred[f'{arm}_s{s}_f{fi}'] for fi in range(FOLDS)]) for s in SEEDS])
         for arm in ('old', 'new')}
    recs = ls.records_from_predictions(lab['old'][order], lab['new'][order], P['old'], P['new'], SUPPORT)
    analysis = {f: ls.analyse(recs, f) for f in ls.FIELDS}
    NN = ls.per_seed_matrix(recs, 'null_nn', 'beta')
    mdb = [ls.min_detectable_beta(NN[:, j]) for j in range(NN.shape[1])]
    err = {arm: float(np.sqrt(np.mean((P[arm].mean(0) - lab[arm][order]) ** 2))) for arm in ('old', 'new')}
    _save_json(OUT / 'verdict.json', {'decision_rule': ls.RULE, 'support': SUPPORT, 'n': len(order),
                                       'min_detectable_beta': mdb, 'rmse_vs_own_labels': err, 'analysis': analysis})
    print('\n'.join(ls.report_lines({'fno|burgers_nu0.001': analysis})))
    print('MDB per null pair', [round(m, 3) for m in mdb], 'RMSE vs own labels', err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['labels', 'train', 'merge', 'analyse'])
    ap.add_argument('files', nargs='*', help='merge: the per-session prediction files')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'labels':
        stage_labels(DATA)
    elif a.stage == 'train':
        import torch
        torch.set_num_threads(a.threads)
        stage_train(tuple(a.seeds))
    elif a.stage == 'merge':
        stage_merge(a.files)
    else:
        stage_analyse()


if __name__ == '__main__':
    main()
