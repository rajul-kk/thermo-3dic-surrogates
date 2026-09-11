"""The matched-budget evaluation protocol. Every control here exists because its absence could explain the published disagreement."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from molprop import models as M
from molprop.data import Dataset, make_split
from molprop.features import FEATURISERS

log = logging.getLogger(__name__)


def score(y_true: np.ndarray, pred: np.ndarray, task: str) -> float:
    """ROC-AUC for classification (higher better), RMSE for regression (lower better)."""
    if task == 'classification':
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return float('nan')
        return float(roc_auc_score(y_true, pred))
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def better(a: float, b: float, task: str) -> bool:
    """Is `a` a better score than `b`?"""
    if not np.isfinite(a):
        return False
    if not np.isfinite(b):
        return True
    return a > b if task == 'classification' else a < b


def _predict(model, X: np.ndarray, task: str) -> np.ndarray:
    if task == 'classification':
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        return model.decision_function(X)
    return model.predict(X)


@dataclass
class Result:
    """One (dataset, split, model) cell: test score per seed plus the chosen configuration."""
    dataset: str
    split: str
    model: str
    task: str
    scores: List[float] = field(default_factory=list)
    featuriser: Optional[str] = None
    seconds: float = 0.0
    failed_trials: int = 0

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.scores)) if self.scores else float('nan')

    @property
    def std(self) -> float:
        return float(np.nanstd(self.scores)) if self.scores else float('nan')


def run_cell(ds: Dataset, split_kind: str, model_name: str, *, seeds: List[int],
             budget: int, featurisers: List[str], task_index: int = 0) -> Result:
    """
    Evaluate one model on one dataset/split under the shared budget.

    The featuriser is searched jointly with the hyperparameters and counted against the SAME
    budget for every model, so no model gets more effective search than another. Selection is
    on validation only; test is scored once per seed with the selected configuration.
    """
    t0 = time.time()
    y_all = ds.y if ds.y.ndim == 1 else ds.y[:, task_index]
    res = Result(ds.name, split_kind, model_name, ds.task)

    # Featurise once per featuriser, reused across seeds and trials.
    feats = {f: FEATURISERS[f](ds.smiles) for f in featurisers}
    sampler, _ = M.MODELS[model_name]

    for seed in seeds:
        tr, va, te = make_split(ds, split_kind, seed)
        mask = np.isfinite(y_all)
        tr, va, te = tr[mask[tr]], va[mask[va]], te[mask[te]]
        if len(tr) < 10 or len(te) < 5:
            res.scores.append(float('nan'))
            continue

        rng = np.random.default_rng(seed)
        best = (float('-inf') if ds.task == 'classification' else float('inf'), None, None)
        trials = 1 if model_name == 'trivial' else budget
        failures = 0
        for _ in range(trials):
            fname = featurisers[int(rng.integers(len(featurisers)))]
            X = feats[fname]
            params = sampler(rng)
            try:
                mdl = M.build(model_name, params, ds.task)
                mdl.fit(X[tr], y_all[tr])
                s = score(y_all[va], _predict(mdl, X[va], ds.task), ds.task)
            except Exception as exc:
                # Count failures: a silently-skipped trial means this model got a smaller
                # effective budget than its competitors, which would invalidate the
                # comparison. One such bug (numpy coercing a mixed hyperparameter list to
                # strings, so every RF trial sampling max_features=0.3 raised) was caught
                # this way rather than quietly biasing the result.
                failures += 1
                log.debug('%s trial failed: %s', model_name, exc)
                continue
            if better(s, best[0], ds.task):
                best = (s, (fname, params), mdl)

        if failures:
            log.warning('%s/%s/%s seed %d: %d/%d trials FAILED -- effective budget was '
                        'smaller than other models, comparison may be unfair',
                        ds.name, split_kind, model_name, seed, failures, trials)
        res.failed_trials += failures
        if best[1] is None:
            res.scores.append(float('nan'))
            continue
        fname, params = best[1]
        # Refit on train+val with the selected configuration, then touch test once.
        X = feats[fname]
        trva = np.concatenate([tr, va])
        mdl = M.build(model_name, params, ds.task)
        mdl.fit(X[trva], y_all[trva])
        res.scores.append(score(y_all[te], _predict(mdl, X[te], ds.task), ds.task))
        res.featuriser = fname

    res.seconds = time.time() - t0
    log.info('%-9s %-8s %-9s %s = %.4f +/- %.4f  (feat=%s, %.0fs)',
             ds.name, split_kind, model_name,
             'AUC' if ds.task == 'classification' else 'RMSE',
             res.mean, res.std, res.featuriser, res.seconds)
    return res


def separable(a: Result, b: Result) -> bool:
    """
    Are two models separable given seed variance?

    Deliberately conservative: the gap must exceed the sum of the two standard deviations.
    Differences smaller than this are reported as not separable rather than as a win, which
    is the control most missing from the papers this audit is adjudicating.
    """
    if not (np.isfinite(a.mean) and np.isfinite(b.mean)):
        return False
    return abs(a.mean - b.mean) > (a.std + b.std)
