"""The non-neural model zoo and their hyperparameter spaces, all sampled under one shared budget."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def _space_rf(rng) -> Dict[str, Any]:
    return {'n_estimators': int(rng.choice([200, 400, 800])),
            'max_depth': [None, 8, 16, 32][int(rng.integers(4))],
            'min_samples_leaf': int(rng.choice([1, 2, 4, 8])),
            # Indexed rather than rng.choice on a mixed list: numpy coerces a mixed
            # list to strings, turning 0.3 into np.str_('0.3'), which sklearn rejects.
            'max_features': ['sqrt', 'log2', 0.3][int(rng.integers(3))],
            'n_jobs': -1}


def _space_gbt(rng) -> Dict[str, Any]:
    return {'n_estimators': int(rng.choice([200, 400, 800])),
            'learning_rate': float(rng.choice([0.02, 0.05, 0.1, 0.2])),
            'max_depth': int(rng.choice([3, 4, 6, 8])),
            'subsample': float(rng.choice([0.6, 0.8, 1.0])),
            'colsample_bytree': float(rng.choice([0.3, 0.6, 1.0])),
            'reg_lambda': float(rng.choice([0.0, 1.0, 10.0]))}


def _space_linear(rng) -> Dict[str, Any]:
    return {'C_or_alpha': float(10 ** rng.uniform(-4, 3))}


def _space_knn(rng) -> Dict[str, Any]:
    return {'n_neighbors': int(rng.choice([1, 3, 5, 10, 20])),
            'weights': str(rng.choice(['uniform', 'distance'])),
            'metric': str(rng.choice(['jaccard', 'euclidean']))}


def build(name: str, params: Dict[str, Any], task: str):
    """Instantiate one model. `task` is 'classification' or 'regression'."""
    clf = task == 'classification'
    if name == 'trivial':
        from sklearn.dummy import DummyClassifier, DummyRegressor
        return DummyClassifier(strategy='prior') if clf else DummyRegressor(strategy='mean')
    if name == 'linear':
        from sklearn.linear_model import LogisticRegression, Ridge
        a = params['C_or_alpha']
        return (LogisticRegression(C=a, max_iter=2000)
                if clf else Ridge(alpha=a))
    if name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        p = {k: v for k, v in params.items()}
        # Jaccard is only meaningful on binary features; guard it.
        kw = dict(n_neighbors=p['n_neighbors'], weights=p['weights'], metric=p['metric'])
        return KNeighborsClassifier(**kw) if clf else KNeighborsRegressor(**kw)
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return (RandomForestClassifier(**params) if clf
                else RandomForestRegressor(**params))
    if name == 'xgboost':
        import xgboost as xgb
        kw = dict(params)
        kw.update(tree_method='hist', n_jobs=-1, verbosity=0)
        return xgb.XGBClassifier(eval_metric='logloss', **kw) if clf else xgb.XGBRegressor(**kw)
    if name == 'lightgbm':
        import lightgbm as lgb
        kw = dict(params)
        kw['verbose'] = -1
        kw['n_jobs'] = -1
        return lgb.LGBMClassifier(**kw) if clf else lgb.LGBMRegressor(**kw)
    raise ValueError(f'unknown model {name!r}')


# name -> (hyperparameter sampler, needs_binary_features)
MODELS: Dict[str, Tuple[Callable, bool]] = {
    'trivial':  (lambda rng: {}, False),
    'linear':   (_space_linear, False),
    'knn':      (_space_knn, False),
    'rf':       (_space_rf, False),
    'xgboost':  (_space_gbt, False),
    'lightgbm': (_space_gbt, False),
}
