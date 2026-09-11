"""Tests for the molprop audit. Focus on the protocol controls, since those are the contribution."""
import numpy as np
import pytest

from molprop import data as D
from molprop import models as M
from molprop.features import FEATURISERS, morgan
from molprop.protocol import Result, better, score, separable

SMILES = ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'CCN(CC)CC', 'c1ccc2ccccc2c1',
          'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'OCC1OC(O)C(O)C(O)C1O']


class TestFeatures:
    @pytest.mark.parametrize('name', sorted(FEATURISERS))
    def test_shape_and_determinism(self, name):
        f = FEATURISERS[name]
        a, b = f(SMILES), f(SMILES)
        assert a.shape[0] == len(SMILES)
        assert np.array_equal(a, b), f'{name} is not deterministic'
        assert np.isfinite(a).all(), f'{name} produced non-finite values'

    def test_morgan_is_binary_and_sparse(self):
        X = morgan(SMILES)
        assert set(np.unique(X)).issubset({0.0, 1.0})
        assert X.sum() > 0
        assert (X.mean() < 0.1), 'ECFP should be sparse'

    def test_unparseable_smiles_yields_zero_row_not_a_crash(self):
        X = morgan(['CCO', 'not_a_molecule'])
        assert X.shape == (2, 2048)
        assert X[1].sum() == 0


class TestSplits:
    def test_random_split_is_a_partition(self):
        tr, va, te = D.random_split(100, (0.8, 0.1, 0.1), seed=0)
        allidx = np.concatenate([tr, va, te])
        assert len(allidx) == 100
        assert len(np.unique(allidx)) == 100

    def test_scaffold_split_is_a_partition(self):
        tr, va, te = D.scaffold_split(SMILES * 4, (0.8, 0.1, 0.1), seed=0)
        allidx = np.concatenate([tr, va, te])
        assert len(np.unique(allidx)) == len(SMILES) * 4

    def test_scaffold_groups_never_straddle_splits(self):
        """The whole point of a scaffold split: no scaffold appears on both sides."""
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        smis = SMILES * 5
        tr, va, te = D.scaffold_split(smis, (0.8, 0.1, 0.1), seed=1)

        def scafs(idx):
            out = set()
            for i in idx:
                out.add(MurckoScaffold.MurckoScaffoldSmiles(
                    mol=Chem.MolFromSmiles(smis[i]), includeChirality=False))
            return out

        assert not (scafs(tr) & scafs(te)), 'scaffold leaked between train and test'

    def test_random_split_differs_from_scaffold_split(self):
        smis = SMILES * 5
        r = D.random_split(len(smis), (0.8, 0.1, 0.1), seed=0)[2]
        s = D.scaffold_split(smis, (0.8, 0.1, 0.1), seed=0)[2]
        assert set(r.tolist()) != set(s.tolist())


class TestModels:
    @pytest.mark.parametrize('name', sorted(M.MODELS))
    def test_every_model_fits_and_predicts_both_tasks(self, name):
        rng = np.random.default_rng(0)
        X = (rng.random((40, 12)) > 0.5).astype(float)
        for task, y in (('classification', (rng.random(40) > 0.5).astype(int)),
                        ('regression', rng.random(40))):
            params = M.MODELS[name][0](rng)
            mdl = M.build(name, params, task)
            mdl.fit(X, y)
            assert len(mdl.predict(X)) == 40


class TestProtocolControls:
    def test_regression_score_is_rmse_and_lower_is_better(self):
        y = np.array([1.0, 2.0, 3.0])
        assert score(y, y, 'regression') == pytest.approx(0.0)
        assert better(0.1, 0.5, 'regression')
        assert not better(0.5, 0.1, 'regression')

    def test_classification_score_is_auc_and_higher_is_better(self):
        y = np.array([0, 0, 1, 1])
        assert score(y, np.array([0.1, 0.2, 0.8, 0.9]), 'classification') == pytest.approx(1.0)
        assert better(0.9, 0.7, 'classification')

    def test_single_class_target_returns_nan_rather_than_crashing(self):
        assert np.isnan(score(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), 'classification'))

    def test_separability_requires_the_gap_to_exceed_combined_spread(self):
        """The control that stops a within-noise difference being reported as a win."""
        # Gap 0.01, combined spread ~0.08: well inside the noise.
        a = Result('d', 's', 'a', 'classification', scores=[0.75, 0.85, 0.80])
        b = Result('d', 's', 'b', 'classification', scores=[0.74, 0.84, 0.79])
        assert not separable(a, b), 'overlapping spreads must not count as separable'

        c = Result('d', 's', 'c', 'classification', scores=[0.95, 0.95, 0.95])
        d = Result('d', 's', 'd', 'classification', scores=[0.50, 0.50, 0.50])
        assert separable(c, d)

    def test_nan_results_are_never_separable(self):
        a = Result('d', 's', 'a', 'regression', scores=[float('nan')])
        b = Result('d', 's', 'b', 'regression', scores=[1.0])
        assert not separable(a, b)


def test_featurisers_are_finite_and_float32():
    """RDKit's Ipc descriptor is ~1e60 and overflows to +inf on the cast to float32.

    A single inf makes sklearn reject every trial that sampled a descriptor featuriser,
    so RF/kNN/linear silently got half the tuning budget of XGBoost/LightGBM -- which
    breaks the matched-budget control the whole audit rests on.
    """
    from molprop.features import FEATURISERS
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O',
              'C1CCCCC1', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C']
    for name, fn in FEATURISERS.items():
        X = fn(smiles)
        assert X.dtype == np.float32, name
        assert np.isfinite(X).all(), f'{name} produced non-finite values'


def test_budget_counts_successful_trials_not_attempts():
    """An invalid config must be resampled, not silently skipped.

    kNN with weights='distance' genuinely fails when duplicate fingerprints put all k
    neighbours at distance zero. Skipping those trials hands kNN a smaller search than
    its competitors, which is the confound the matched-budget control exists to remove.
    """
    import molprop.models as M
    from molprop.protocol import run_cell
    from molprop.data import Dataset

    calls = {'n': 0}
    real_build = M.build

    def flaky_build(name, params, task):
        calls['n'] += 1
        # Every other config is "invalid" during the search; the final refit (which also
        # calls build) must not be sabotaged, so stop being flaky once the budget is full.
        if calls["n"] % 2 == 0 and calls["n"] < 16:
            raise ValueError('synthetic invalid configuration')
        return real_build(name, params, task)

    rng = np.random.default_rng(0)
    n = 120
    ds = Dataset(name='synth', task='classification',
                 smiles=['CCO', 'c1ccccc1', 'CCN', 'CCC', 'CCCl', 'CCBr'] * (n // 6),
                 y=rng.integers(0, 2, n).astype(float))

    M.build = flaky_build
    try:
        res = run_cell(ds, 'random', 'linear', seeds=[0], budget=8,
                       featurisers=['maccs'])
    finally:
        M.build = real_build

    # Half the attempts fail, so filling a budget of 8 needs ~16 attempts.
    assert calls['n'] > 8, f'budget was not refilled after failures ({calls["n"]} attempts)'
    assert not res.budget_unmatched, 'budget should have been fillable within the cap'
    assert res.failed_trials > 0, 'failures should still be counted and reported'


def test_deterministic_scaffold_split_is_harder_and_seedless():
    """The published MoleculeNet 'scaffold split' is one fixed ordering, not a family.

    Randomising the tie-break between equal-sized scaffold groups makes BBBP ~19 AUC points
    easier for the same model, which is larger than most reported architecture gaps. Both
    splits are legitimately scaffold-disjoint, so this is a protocol difference, not a bug.
    """
    from molprop.data import load, scaffold_split, scaffold_split_deterministic
    ds = load('bbbp')
    a = scaffold_split_deterministic(ds.smiles, (0.8, 0.1, 0.1))
    b = scaffold_split_deterministic(ds.smiles, (0.8, 0.1, 0.1))
    assert (a[2] == b[2]).all(), 'deterministic split must not vary between calls'

    rand_te = set(scaffold_split(ds.smiles, (0.8, 0.1, 0.1), 0)[2].tolist())
    det_te = set(a[2].tolist())
    overlap = len(rand_te & det_te) / len(rand_te | det_te)
    assert overlap < 0.5, f'the two splits should differ substantially (jaccard {overlap:.2f})'


def test_published_dataset_keys_match_the_loader():
    """A published row keyed 'lipophilicity' when the loader says 'lipo' is silently dead.

    It would never match a cell, so the audit would print 'none recorded' and nobody would
    notice the reference data existed.
    """
    from molprop.published import P
    from molprop.data import DATASETS
    unmatched = {p.dataset for p in P} - set(DATASETS)
    assert not unmatched, f'published.py keys not in the loader: {unmatched}'
