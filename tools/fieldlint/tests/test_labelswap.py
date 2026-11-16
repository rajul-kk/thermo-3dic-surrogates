import numpy as np
import pytest
from fieldlint import labelswap as ap


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


def test_records_from_predictions_matches_scenario_record():
    rng = np.random.default_rng(3)
    T_new = rng.normal(size=(4, 6, 5))
    T_old = T_new + (rng.random((4, 6, 5)) < 0.2) * 2.0
    P_new = T_new + 0.1 * rng.normal(size=(3, 4, 6, 5))
    P_old = P_new + 0.7 * (T_old - T_new)
    recs = ap.records_from_predictions(T_old, T_new, P_old, P_new)
    assert len(recs) == 4
    ref = ap.scenario_record({s: P_old[s, 1] for s in range(3)}, {s: P_new[s, 1] for s in range(3)}, T_old[1] - T_new[1])
    assert recs[1] == ref
    assert all(abs(s['beta'] - 0.7) < 1e-9 for s in recs[1]['single'])


def test_records_from_predictions_rejects_bad_shapes():
    T = np.zeros((3, 4))
    with pytest.raises(ValueError, match='shapes differ'):
        ap.records_from_predictions(T, T, np.zeros((2, 3, 5)), np.zeros((2, 3, 5)))
    with pytest.raises(ValueError, match='at least 2 seeds'):
        ap.records_from_predictions(T, T, np.zeros((1, 3, 4)), np.zeros((1, 3, 4)))


def test_records_from_summary_round_trips_analysis():
    recs = _records(0.6, n_scen=12)
    per = {f's{i:02d}': {k: ({f: r[k][f] for f in ap.FIELDS} if k == 'mean' else {f: [x[f] for x in r[k]] for f in ap.FIELDS})
                         for k in ap.SCORE_KEYS} for i, r in enumerate(recs)}
    assert ap.analyse(ap.records_from_summary(per)) == ap.analyse(recs)


def test_min_detectable_beta():
    null = np.random.default_rng(0).normal(0.0, 0.3, 45)
    half = 1.96 * null.std(ddof=1) / np.sqrt(45)
    assert abs(ap.min_detectable_beta(null) - (max(0.2, null.mean() + half) + half)) < 1e-12
    assert ap.min_detectable_beta(np.zeros(45)) == 0.2                       # no seed noise: the 0.2 floor binds
    assert ap.min_detectable_beta(null, n=180) < ap.min_detectable_beta(null)  # more samples, smaller effect detectable
    assert np.isnan(ap.min_detectable_beta([0.1]))


def test_cli_on_prediction_arrays(tmp_path, capsys):
    rng = np.random.default_rng(5)
    T_new = rng.normal(size=(30, 40))
    T_old = T_new + (rng.random((30, 40)) < 0.1) * 3.0
    P_new = T_new + 0.05 * rng.normal(size=(3, 30, 40))
    P_old = P_new + 0.9 * (T_old - T_new) + 0.05 * rng.normal(size=(3, 30, 40))
    f = tmp_path / 'p.npz'
    np.savez(f, T_old=T_old, T_new=T_new, pred_old=P_old, pred_new=P_new)
    out = tmp_path / 'a.json'
    assert ap.main([str(f), '--json', str(out)]) == 0
    assert 'learned' in capsys.readouterr().out
    import json
    a = json.loads(out.read_text())['analysis'][str(f)]['beta']
    assert a['decision'] == 'learned' and abs(a['mean']['mean'] - 0.9) < 0.05


def test_cli_needs_exactly_one_input():
    with pytest.raises(SystemExit):
        ap.main([])


def test_cli_json_is_strict_and_support_k_rejected_with_summary(tmp_path):
    import json
    T = np.zeros((6, 8))
    T_old = T.copy()
    T_old[:, 0] = 1.0                                         # support of 1 node: beta_s and corr_s are NaN
    P = np.random.default_rng(0).normal(size=(2, 6, 8))
    f, out = tmp_path / 'p.npz', tmp_path / 'a.json'
    np.savez(f, T_old=T_old, T_new=T, pred_old=P, pred_new=P[::-1])
    assert ap.main([str(f), '--json', str(out)]) == 0
    a = json.loads(out.read_text(), parse_constant=lambda c: pytest.fail(f'non-strict JSON constant {c}'))
    assert a['analysis'][str(f)]['beta_s']['mean']['mean'] is None
    with pytest.raises(SystemExit):
        ap.main(['--summary', str(out), '--support-k', '0.5'])
