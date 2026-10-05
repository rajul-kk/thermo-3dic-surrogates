"""Projection test for the 3D-ICE die-edge artefact (pure numpy, shared by the Kaggle kernel, the report script and the tests).

Setting: a model is trained on OLD labels (artefact present) and on NEW labels (artefact removed) with identical inputs,
hyper-parameters and seed.  For a held-out scenario the true artefact field is D = T_old - T_new and the learned
difference is L = O(x) - N(x).  The projection slope beta = <L, D> / <D, D> is 1 when the artefact is reproduced exactly
and 0 when the learned difference is orthogonal to it; it stays meaningful for a poor model because an inaccurate but
label-independent error cancels in O - N.  Null: the same statistic between two same-label models (seed noise).
"""
from itertools import combinations

import numpy as np
from scipy import stats

SUPPORT_K = 0.1
FIELDS = ('beta', 'corr', 'beta_s', 'corr_s')


def score(L, D, support_k=SUPPORT_K):
    """beta and corr of learned difference L against true artefact D, on all nodes and on the support |D| > support_k."""
    L, D = np.asarray(L, np.float64).ravel(), np.asarray(D, np.float64).ravel()
    out = {}
    for tag, m in (('', np.ones(D.shape, bool)), ('_s', np.abs(D) > support_k)):
        l, d = L[m], D[m]
        dd = float(d @ d)
        if m.sum() < 3 or dd <= 0:
            out['beta' + tag] = out['corr' + tag] = float('nan')
            continue
        out['beta' + tag] = float(l @ d / dd)
        ll = float(l @ l)
        out['corr' + tag] = float(l @ d / np.sqrt(ll * dd)) if ll > 0 else float('nan')
    out['support_n'] = int((np.abs(D) > support_k).sum())
    return out


def scenario_record(pred_old, pred_new, D, support_k=SUPPORT_K):
    """pred_old / pred_new: dict seed -> prediction (same node order as D) of the models trained on old / new labels.

    Returns scores for: 'mean' (mean_s O - mean_s N), 'single' (O_s - N_s per seed), 'null_nn' and 'null_oo' (same-label
    seed pairs).  Each entry is a list of score dicts (a single one for 'mean')."""
    seeds = sorted(pred_old)
    assert sorted(pred_new) == seeds
    pairs = list(combinations(seeds, 2))
    mean_o = np.mean([pred_old[s] for s in seeds], 0)
    mean_n = np.mean([pred_new[s] for s in seeds], 0)
    return {'seeds': seeds, 'pairs': [list(p) for p in pairs],
            'mean': score(mean_o - mean_n, D, support_k),
            'single': [score(pred_old[s] - pred_new[s], D, support_k) for s in seeds],
            'null_nn': [score(pred_new[a] - pred_new[b], D, support_k) for a, b in pairs],
            'null_oo': [score(pred_old[a] - pred_old[b], D, support_k) for a, b in pairs]}


def _col(rec, key, field):
    v = rec[key]
    return np.nanmean([x[field] for x in v]) if isinstance(v, list) else v[field]


def per_scenario_arrays(records, field='beta'):
    """Per-scenario scalars: mean-of-seeds beta, seed-averaged single-seed beta and the two seed-pair nulls."""
    with np.errstate(all='ignore'):
        return {k: np.array([_safe(rec, k, field) for rec in records]) for k in ('mean', 'single', 'null_nn', 'null_oo')}


def _safe(rec, key, field):
    v = rec[key]
    vals = [x[field] for x in v] if isinstance(v, list) else [v[field]]
    vals = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(vals)) if vals else float('nan')


def boot_ci(x, n_boot=4000, alpha=0.05, seed=0):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def decide(ci, null_ci, learned_lo=0.2, not_learned_hi=0.1):
    """ci / null_ci = (mean, lo, hi).  'learned' if the beta CI lies above the null CI and lo > 0.2;
    'not learned' if hi < 0.1 and the CI overlaps the null CI; otherwise 'inconclusive'."""
    _, lo, hi = ci
    _, nlo, nhi = null_ci
    if np.isnan(lo) or np.isnan(nhi):
        return 'inconclusive'
    if lo > nhi and lo > learned_lo:
        return 'learned'
    if hi < not_learned_hi and hi >= nlo and lo <= nhi:
        return 'not learned'
    return 'inconclusive'


def wilcoxon_p(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.all(d == 0):
        return float('nan')
    return float(stats.wilcoxon(d).pvalue)


def _per_seed_matrix(records, key, field):
    """(n_scenarios, n_entries) array of scores in a list-valued record entry."""
    with np.errstate(all='ignore'):
        return np.array([[x[field] for x in rec[key]] for rec in records], np.float64)


def analyse(records, field='beta', n_boot=4000):
    """Inference over held-out scenarios for one (architecture, geometry).

    * 'mean': beta of (mean_s O - mean_s N).  By linearity of beta in L this equals the seed-average of the per-seed betas.
      Its noise is below that of the single-seed same-label null, so comparing it with the null is conservative against 'learned'.
    * 'single_seed': for each seed k, beta(O_k - N_k) over scenarios against the matched same-label pair null beta(N_a - N_b)
      (both are single-seed differences, so the noise is matched); per-seed decisions.  Headline decision = all seeds agree
      ('learned' / 'not learned'), else 'inconclusive'.
    """
    a = per_scenario_arrays(records, field)
    out = {'n_scenarios': int(np.sum(~np.isnan(a['mean']))), 'field': field}
    null_ci = boot_ci(a['null_nn'], n_boot)
    out['null_nn'] = dict(zip(('mean', 'lo', 'hi'), null_ci))
    out['null_oo'] = dict(zip(('mean', 'lo', 'hi'), boot_ci(a['null_oo'], n_boot)))
    ci = boot_ci(a['mean'], n_boot)
    out['mean'] = {'mean': ci[0], 'lo': ci[1], 'hi': ci[2], 'wilcoxon_p_vs_null_nn': wilcoxon_p(a['mean'], a['null_nn']),
                   'wilcoxon_p_vs_null_oo': wilcoxon_p(a['mean'], a['null_oo']), 'decision': decide(ci, null_ci)}
    S = _per_seed_matrix(records, 'single', field)
    NN = _per_seed_matrix(records, 'null_nn', field)
    OO = _per_seed_matrix(records, 'null_oo', field)
    per, decs = [], []
    for k in range(S.shape[1]):
        nn, oo = NN[:, k % NN.shape[1]], OO[:, k % OO.shape[1]]
        c, nc = boot_ci(S[:, k], n_boot), boot_ci(nn, n_boot)
        d = decide(c, nc)
        decs.append(d)
        per.append({'seed_index': k, 'mean': c[0], 'lo': c[1], 'hi': c[2], 'null_nn_mean': nc[0], 'null_nn_lo': nc[1],
                    'null_nn_hi': nc[2], 'wilcoxon_p_vs_null_nn': wilcoxon_p(S[:, k], nn),
                    'wilcoxon_p_vs_null_oo': wilcoxon_p(S[:, k], oo), 'decision': d})
    out['single_seed'] = per
    out['single_seed_decision'] = decs[0] if len(set(decs)) == 1 else 'inconclusive'
    out['decision'] = out['single_seed_decision']
    return out


RULE = ("learned: mean beta CI lies above the same-label null CI and its lower bound > 0.2; "
        "not learned: CI upper bound < 0.1 and CI overlaps the null CI; otherwise inconclusive "
        "(95% bootstrap CI over held-out scenarios; headline = per-seed single-seed comparison, all seeds must agree).")
