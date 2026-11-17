"""Label-swap test: did a model learn an artefact that is in its training labels?

Setting: the same architecture is trained twice with identical inputs, hyper-parameters, folds and seeds, once on the
labels as shipped (OLD, artefact present) and once on corrected labels (NEW). For a held-out sample the true artefact
is D = T_old - T_new and the learned difference is L = O(x) - N(x). The projection slope beta = <L, D> / <D, D> is 1
when the artefact is reproduced exactly and 0 when the learned difference is orthogonal to it. It stays meaningful for
an inaccurate model, because a label-independent error cancels in O - N. Null: the same statistic between two
same-label models (seed noise), N_a - N_b and O_a - O_b.

Decision (pre-registered in the thermal benchmark's report, section 9.32, and unchanged here): 'learned' if the 95%
bootstrap CI of beta lies above the same-label null CI with lower bound > 0.2; 'not learned' if the CI upper bound is
< 0.1 and it overlaps the null CI; otherwise 'inconclusive'. The headline compares each seed singly against a matched
same-label pair and needs every seed to agree. Spec, power and worked examples: docs/labelswap.md.

Core is numpy only; the Wilcoxon p-values need scipy.

Usage:
  fieldlint-labelswap predictions.npz [--json out.json]     # T_old, T_new (n, ...) and pred_old, pred_new (seeds, n, ...)
  fieldlint-labelswap --summary projection_summary.json     # re-analyse stored per-sample scores
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

SUPPORT_K = 0.1
FIELDS = ('beta', 'corr', 'beta_s', 'corr_s')
SCORE_KEYS = ('mean', 'single', 'null_nn', 'null_oo')
LEARNED_LO, NOT_LEARNED_HI = 0.2, 0.1
RULE = ("learned: mean beta CI lies above the same-label null CI and its lower bound > 0.2; "
        "not learned: CI upper bound < 0.1 and CI overlaps the null CI; otherwise inconclusive "
        "(95% bootstrap CI over held-out scenarios; headline = per-seed single-seed comparison, all seeds must agree).")


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


def records_from_predictions(T_old, T_new, pred_old, pred_new, support_k=SUPPORT_K):
    """Records for every held-out sample from arrays: T_old, T_new (n, ...); pred_old, pred_new (n_seeds, n, ...)."""
    T_old, T_new, pred_old, pred_new = (np.asarray(x) for x in (T_old, T_new, pred_old, pred_new))  # score() casts
    if T_old.shape != T_new.shape or pred_old.shape != pred_new.shape or pred_old.shape[1:] != T_old.shape:
        raise ValueError(f'shapes differ: T_old {T_old.shape}, T_new {T_new.shape}, pred_old {pred_old.shape}, '
                         f'pred_new {pred_new.shape}')
    if pred_old.shape[0] < 2:
        raise ValueError('the same-label null needs at least 2 seeds per label set')
    seeds = range(pred_old.shape[0])
    return [scenario_record({s: pred_old[s, i] for s in seeds}, {s: pred_new[s, i] for s in seeds},
                            T_old[i] - T_new[i], support_k) for i in range(T_old.shape[0])]


def records_from_summary(per_scenario):
    """Records from the stored per-sample scores of a summary file ({name: {key: {field: value or [per seed]}}}),
    in name order, the order the stored analysis used."""
    out = []
    for nm in sorted(per_scenario):
        ps = per_scenario[nm]
        rec = {'mean': ps['mean']}
        for k in SCORE_KEYS[1:]:
            rec[k] = [{f: ps[k][f][i] for f in FIELDS} for i in range(len(ps[k]['beta']))]
        out.append(rec)
    return out


def per_scenario_arrays(records, field='beta'):
    """Per-scenario scalars: mean-of-seeds beta, seed-averaged single-seed beta and the two seed-pair nulls."""
    with np.errstate(all='ignore'):
        return {k: np.array([_safe(rec, k, field) for rec in records]) for k in SCORE_KEYS}


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


def decide(ci, null_ci, learned_lo=LEARNED_LO, not_learned_hi=NOT_LEARNED_HI):
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
    from scipy import stats
    return float(stats.wilcoxon(d).pvalue)


def per_seed_matrix(records, key, field):
    """(n_scenarios, n_entries) array of scores in a list-valued record entry."""
    with np.errstate(all='ignore'):
        return np.array([[x[field] for x in rec[key]] for rec in records], np.float64)


def analyse(records, field='beta', n_boot=4000):
    """Inference over held-out scenarios for one (architecture, dataset).

    * 'mean': beta of (mean_s O - mean_s N).  By linearity of beta in L this equals the seed-average of the per-seed betas.
      Its noise is below that of the single-seed same-label null, so comparing it with the null is conservative against 'learned'.
    * 'single_seed': for each seed k, beta(O_k - N_k) over scenarios against the matched same-label pair null beta(N_a - N_b)
      (both are single-seed differences, so the noise is matched); per-seed decisions.  Headline decision = all seeds agree
      ('learned' / 'not learned'), else 'inconclusive'.  Seed k is matched with pair k mod n_pairs: with 2 seeds both are
      compared with the one pair, with 3 seeds seed k with the k-th of the 3 pairs.
    """
    a = per_scenario_arrays(records, field)
    out = {'n_scenarios': int(np.sum(~np.isnan(a['mean']))), 'field': field}
    null_ci = boot_ci(a['null_nn'], n_boot)
    out['null_nn'] = dict(zip(('mean', 'lo', 'hi'), null_ci))
    out['null_oo'] = dict(zip(('mean', 'lo', 'hi'), boot_ci(a['null_oo'], n_boot)))
    ci = boot_ci(a['mean'], n_boot)
    out['mean'] = {'mean': ci[0], 'lo': ci[1], 'hi': ci[2], 'wilcoxon_p_vs_null_nn': wilcoxon_p(a['mean'], a['null_nn']),
                   'wilcoxon_p_vs_null_oo': wilcoxon_p(a['mean'], a['null_oo']), 'decision': decide(ci, null_ci)}
    S = per_seed_matrix(records, 'single', field)
    NN = per_seed_matrix(records, 'null_nn', field)
    OO = per_seed_matrix(records, 'null_oo', field)
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


def min_detectable_beta(null_scores, n=None, z=1.96, learned_lo=LEARNED_LO):
    """Smallest true beta that a single seed would call 'learned' at about 50% power, from same-label null scores.

    The lower CI bound of a single-seed beta must clear both the null CI's upper bound and `learned_lo`. Assuming the
    per-sample spread of beta under an effect equals its spread under the null (seed noise), that bound sits about
    z * sd / sqrt(n) below the mean, so MDB = max(learned_lo, null upper bound) + z * sd / sqrt(n). For about 80% power
    add a further 0.84 * sd / sqrt(n). `n` defaults to the number of finite null scores (the design's sample count)."""
    x = np.asarray(null_scores, np.float64)
    x = x[~np.isnan(x)]
    n = len(x) if n is None else n
    if len(x) < 2 or n < 1:
        return float('nan')
    half = z * float(x.std(ddof=1)) / np.sqrt(n)
    return max(learned_lo, float(x.mean()) + half) + half


def json_safe(x):
    """NaN and inf become null, so the file is strict JSON."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def report_lines(analysis):
    """One line per field of an analyse() result per (unit) key: seed-mean beta [CI], null, per-seed decisions."""
    def fmt(c):
        return f"{c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]"
    out = []
    for key, by_field in analysis.items():
        a = by_field['beta']
        seeds = ','.join(s['decision'][:3] + format(s['mean'], '+.2f') for s in a['single_seed'])
        out.append(f"{key:<24} n={a['n_scenarios']:<4} beta {fmt(a['mean']):<24} null N-N {fmt(a['null_nn']):<24} "
                   f"seeds {seeds:<30} {a['decision']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='fieldlint-labelswap', description=__doc__.split('\n\n')[0])
    ap.add_argument('path', nargs='?', help='.npz with T_old, T_new (n, ...) and pred_old, pred_new (seeds, n, ...)')
    ap.add_argument('--summary', help='summary JSON with stored per_scenario scores per unit')
    ap.add_argument('--support-k', type=float, help=f'|D| threshold of the support fields (default {SUPPORT_K}; '
                    'PATH only: stored scores keep the support they were computed with)')
    ap.add_argument('--json', help='write the analysis of all four fields to this file (strict JSON, NaN as null)')
    a = ap.parse_args(argv)
    if bool(a.path) == bool(a.summary):
        ap.error('give exactly one of PATH or --summary')
    if a.summary and a.support_k is not None:
        ap.error("--support-k applies to PATH only; stored scores keep the summary's support_K")
    if a.summary:
        with open(a.summary, encoding='utf-8') as fh:
            results = json.load(fh)['results']
        units = {k: records_from_summary(e['per_scenario']) for k, e in sorted(results.items())}
    else:
        with np.load(a.path) as d:
            support = SUPPORT_K if a.support_k is None else a.support_k
            units = {a.path: records_from_predictions(d['T_old'], d['T_new'], d['pred_old'], d['pred_new'], support)}
    fields = FIELDS if a.json else ('beta',)          # the printed report shows the primary field only
    analysis = {k: {f: analyse(r, f) for f in fields} for k, r in units.items()}
    print('\n'.join(report_lines(analysis)))
    if a.json:
        tmp = a.json + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(json_safe({'decision_rule': RULE, 'analysis': analysis}), fh, indent=1, allow_nan=False)
        os.replace(tmp, a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
