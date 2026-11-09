import numpy as np

from src.validation import artefact_projection as ap


def _D(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    D = np.zeros(n)
    D[rng.choice(n, 100, replace=False)] = rng.uniform(1, 3, 100)
    return D


def test_score_exact_scaled_and_orthogonal():
    D = _D()
    assert ap.score(D, D)['beta'] == 1.0
    s = ap.score(0.5 * D, D)
    assert abs(s['beta'] - 0.5) < 1e-12 and abs(s['corr'] - 1) < 1e-12
    O = np.zeros_like(D)
    O[D == 0] = 1.0                       # supported only where D == 0
    s = ap.score(O, D)
    assert abs(s['beta']) < 1e-12 and abs(s['corr']) < 1e-12
    assert np.isnan(ap.score(D, np.zeros_like(D))['beta'])


def test_support_restriction():
    D = _D()
    L = D.copy()
    L[D == 0] = 5.0                      # junk off the support
    s = ap.score(L, D)
    assert abs(s['beta_s'] - 1) < 1e-12 and abs(s['corr_s'] - 1) < 1e-12
    assert s['support_n'] == 100


def _records(strength, n_scen=30, noise=0.3, shared_error=3.0, seeds=(0, 1, 2)):
    rng = np.random.default_rng(1)
    recs = []
    for i in range(n_scen):
        D = _D(seed=i)
        base = shared_error * rng.normal(size=D.shape)          # label-independent error: must cancel in O - N
        N = {s: base + noise * rng.normal(size=D.shape) for s in seeds}
        O = {s: base + strength * D + noise * rng.normal(size=D.shape) for s in seeds}
        recs.append(ap.scenario_record(O, N, D))
    return recs


def test_poor_model_that_learned_artefact_is_learned():
    r = ap.analyse(_records(0.8))
    assert r['decision'] == 'learned' and abs(r['mean']['mean'] - 0.8) < 0.1
    assert r['mean']['decision'] == 'learned'
    assert abs(r['null_nn']['mean']) < 0.05 and r['single_seed'][0]['wilcoxon_p_vs_null_nn'] < 1e-3


def test_poor_model_that_did_not_is_not_learned():
    r = ap.analyse(_records(0.0))
    assert r['decision'] == 'not learned', r
    assert abs(r['mean']['mean']) < 0.05


def test_partial_learning_is_inconclusive():
    assert ap.analyse(_records(0.15, noise=0.3))['decision'] == 'inconclusive'


def test_two_seed_pairs_and_nan_scenarios():
    recs = _records(0.5, n_scen=10, seeds=(0, 1))
    assert recs[0]['pairs'] == [[0, 1]]
    rng = np.random.default_rng(0)
    z = np.zeros(50)
    recs.append(ap.scenario_record({0: z, 1: z}, {0: z, 1: z}, z))   # D == 0 -> NaN scores, must be ignored
    r = ap.analyse(recs)
    assert r['n_scenarios'] == 10 and not np.isnan(r['mean']['mean']) and len(r['single_seed']) == 2


def test_decide_rule():
    null = (0.0, -0.05, 0.05)
    assert ap.decide((0.6, 0.5, 0.7), null) == 'learned'
    assert ap.decide((0.6, 0.1, 0.9), null) == 'inconclusive'          # lower bound not above 0.2
    assert ap.decide((0.01, -0.03, 0.06), null) == 'not learned'
    assert ap.decide((0.3, 0.12, 0.45), null) == 'inconclusive'
    assert ap.decide((0.3, float('nan'), 0.45), null) == 'inconclusive'
