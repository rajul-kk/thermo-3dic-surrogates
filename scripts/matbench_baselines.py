"""Weak-baseline audit on matbench_expt_gap: mean/kNN/ridge on raw elemental fractions vs the leaderboard (§9.18).
Data: data/matbench/matbench_expt_gap.json.gz from ml.materialsproject.org."""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.hotspot_eval import kfold_indices

LEADERBOARD = {
    'Dummy (mean)': 1.1435,
    'RF-SCM/Magpie (leaderboard simple-features baseline)': 0.4461,
    'AMMExpress v2020': 0.4161,
    'CrabNet': 0.3463,
    'MODNet (v0.1.12)': 0.3327,
    'Darwin (best, LLM-based)': 0.2865,
}

FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)')


def parse_composition(formula: str) -> dict:
    """Elemental fraction dict from a formula string, e.g. 'Ag0.5Ge1Pb1.75S4'."""
    counts = {}
    for el, amt in FORMULA_RE.findall(formula):
        if not el:
            continue
        counts[el] = counts.get(el, 0.0) + (float(amt) if amt else 1.0)
    total = sum(counts.values())
    return {el: c / total for el, c in counts.items()} if total > 0 else counts


def build_matrix(compositions):
    parsed = [parse_composition(c) for c in compositions]
    elements = sorted({el for p in parsed for el in p})
    idx = {el: i for i, el in enumerate(elements)}
    X = np.zeros((len(parsed), len(elements)))
    for i, p in enumerate(parsed):
        for el, frac in p.items():
            X[i, idx[el]] = frac
    return X, elements


def ridge_fit_predict(X_tr, y_tr, X_te, lam=1.0):
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd < 1e-12] = 1.0
    Ztr = np.hstack([(X_tr - mu) / sd, np.ones((len(X_tr), 1))])
    Zte = np.hstack([(X_te - mu) / sd, np.ones((len(X_te), 1))])
    reg = lam * np.eye(Ztr.shape[1])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(Ztr.T @ Ztr + reg, Ztr.T @ y_tr)
    return Zte @ W


def knn_predict(X_tr, y_tr, X_te, k=5):
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd < 1e-12] = 1.0
    Ztr, Zte = (X_tr - mu) / sd, (X_te - mu) / sd
    preds = np.zeros(len(X_te))
    for i in range(len(X_te)):
        d = np.linalg.norm(Ztr - Zte[i], axis=1)
        idx = np.argsort(d)[:k]
        w = 1.0 / np.maximum(d[idx], 1e-9)
        preds[i] = (w * y_tr[idx]).sum() / w.sum()
    return preds


def main():
    raw = json.loads(gzip.decompress(Path('data/matbench/matbench_expt_gap.json.gz').read_bytes()))
    comps = [r[0] for r in raw['data']]
    y = np.array([r[1] for r in raw['data']])
    X, elements = build_matrix(comps)
    print(f'n={len(y)}  elemental-fraction features={len(elements)}  '
          f'target mean={y.mean():.3f}  std={y.std():.3f}  metals(gap=0) frac={np.mean(y==0):.3f}')

    seed = 0
    folds = kfold_indices(len(y), 5, seed)
    results = {'mean': [], 'knn': [], 'ridge': []}
    for fold in folds:
        keep = set(fold.tolist())
        te = fold
        tr = np.array([i for i in range(len(y)) if i not in keep])
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        results['mean'].append(np.abs(yte - ytr.mean()))
        results['knn'].append(np.abs(yte - knn_predict(Xtr, ytr, Xte, k=5)))
        results['ridge'].append(np.abs(yte - ridge_fit_predict(Xtr, ytr, Xte, lam=10.0)))

    print(f'\n5-fold CV (seed={seed}), MAE (eV):')
    for name, errs in results.items():
        mae = np.concatenate(errs).mean()
        print(f'  {name:<8} {mae:.4f}')

    print('\n=== vs. published leaderboard (matbench_expt_gap, official) ===')
    for name, mae in LEADERBOARD.items():
        print(f'  {name:<45} {mae:.4f}')

    our_ridge = np.concatenate(results['ridge']).mean()
    dummy, best, rf_magpie = LEADERBOARD['Dummy (mean)'], LEADERBOARD['Darwin (best, LLM-based)'], \
        LEADERBOARD['RF-SCM/Magpie (leaderboard simple-features baseline)']
    frac_gap_closed = (dummy - our_ridge) / (dummy - best)
    print(f'\nOur elemental-fraction ridge MAE {our_ridge:.4f} closes '
          f'{100*frac_gap_closed:.1f}% of the dummy-to-best gap '
          f'(leaderboard\'s own Magpie+RF baseline closes '
          f'{100*(dummy-rf_magpie)/(dummy-best):.1f}%)')


if __name__ == '__main__':
    main()
