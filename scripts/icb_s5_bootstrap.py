"""Bootstrap 95% CIs on IC-ThermBench S5 zero-shot, from per-sample sufficient statistics.
Metric = their RMSE (mean over samples of per-sample RMSE) and pooled R2. Sample bootstrap resamples the 5000
samples; package-cluster bootstrap resamples the 5 packages (Case16-20) with replacement (very coarse, n=5 clusters),
and a two-stage variant resamples packages then samples inside them. Usage: python -m scripts.icb_s5_bootstrap"""
import json
from pathlib import Path
import numpy as np

B, rng = 10000, np.random.default_rng(12345)
man = json.load(open('data/ic-thermbench/datasets/level5_steady/manifest.json'))
case = np.array([m['case'] for m in man]); pk = sorted(set(case)); idx = {p: np.where(case == p)[0] for p in pk}

def load(p):
    d = np.load(p); return d['sse'], d['sy'], d['syy'], int(d['n_cells'])

def stats(sse, sy, syy, n, ix):
    r = np.sqrt(sse[ix] / n).mean()
    N = n * len(ix); ss_tot = syy[ix].sum() - sy[ix].sum() ** 2 / N
    return r, 1 - sse[ix].sum() / ss_tot

def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def analyse(sse, sy, syy, n):
    all_ix = np.arange(len(sse)); out = {}
    r, r2 = stats(sse, sy, syy, n, all_ix); out['rmse'], out['r2'] = float(r), float(r2)
    out['pooled_rmse'] = float(np.sqrt(sse.sum() / (n * len(sse))))
    out['per_package'] = {p: float(np.sqrt(sse[idx[p]] / n).mean()) for p in pk}
    for name in ('sample', 'cluster', 'two_stage'):
        v = np.empty((B, 2))
        for b in range(B):
            if name == 'sample': ix = rng.integers(0, len(sse), len(sse))
            else:
                ps = rng.choice(len(pk), len(pk))
                ix = np.concatenate([idx[pk[i]] if name == 'cluster' else rng.choice(idx[pk[i]], len(idx[pk[i]])) for i in ps])
            v[b] = stats(sse, sy, syy, n, ix)
        out[name] = {'rmse_ci': ci(v[:, 0]), 'r2_ci': ci(v[:, 1])}
    return out

res = {}
res['power_scaling_rule'] = analyse(*load('results/icb_ci/persample_s5_powerscale.npz'))
seeds = sorted(Path('results/icb_physsolve_seeds').glob('persample_s5_seed*.npz'))
for f in seeds: res[f.stem.replace('persample_s5_', 'learned_conductance_')] = analyse(*load(f))
if seeds:  # seed-mean rmse with sample bootstrap on the per-sample seed-averaged RMSE, and paired diff vs the rule
    L = [load(f) for f in seeds]; ps = [np.sqrt(l[0] / l[3]) for l in L]; rule = np.sqrt(load('results/icb_ci/persample_s5_powerscale.npz')[0] / 4096)
    m = np.mean(ps, 0); d = rule - m
    v = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    res['seed_mean_vs_rule'] = {'mean_of_seed_rmse': float(np.mean([p.mean() for p in ps])), 'sd_across_seeds': float(np.std([p.mean() for p in ps], ddof=1)),
                                'rule_minus_model_rmse': float(d.mean()), 'sample_ci': ci(v)}
Path('results/icb_ci/s5_bootstrap.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
