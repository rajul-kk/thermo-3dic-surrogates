"""Real-data checks, read-only. Skipped when the datasets are not on disk.

Set FIELDLINT_DATA_ROOT to the repo's data/ directory (default: the author's checkout).
"""
import os
from pathlib import Path

import pytest

from fieldlint import load_config, open_dataset, run

pytest.importorskip('h5py')
ROOT = Path(os.environ.get('FIELDLINT_DATA_ROOT', r'D:\Work\3D-ICE Thermal-modelling\Thermo\data'))
ICTHERM = ROOT / '_external' / 'ictherm' / 'datasets'
THERMFM = ROOT / '_external' / 'thermfm' / 'thermal_steady'
N = 400

pytestmark = pytest.mark.realdata


def _lint(path, cfg, rules, head=None, nb=1):
    from fieldlint import Options
    ds = open_dataset(path, cfg)
    if head:
        ds = ds.head(head)
    rep = run(ds, rules, Options(max_samples=None, neighbourhood=nb))
    ds.close()
    return {r.code: r for r in rep.results}


@pytest.mark.parametrize('scope,stored_range,fixed_range', [
    ('level2', (0.35, 0.45), (0.72, 0.82)),
    ('level3', (0.28, 0.40), (0.70, 0.80)),
    ('level5', (0.36, 0.48), (0.64, 0.76)),
])
def test_ictherm_transposition_reproduced(scope, stored_range, fixed_range):
    """Measured on the first 400 samples: corr(smoothed power, T) is ~0.33-0.42 as stored and ~0.70-0.78 transposed,
    and 14-37% of samples violate the maximum principle as stored (depending on the neighbourhood) versus none transposed."""
    path = ICTHERM / f'{scope}_steady'
    if not path.exists():
        pytest.skip(f'{path} not present')
    cfg = load_config('ictherm')
    stored = _lint(path, cfg, ['F001', 'F002', 'F003'], N)
    assert stored['F001'].status == 'ok'
    c = stored['F003'].metrics['mean_corr']
    assert stored_range[0] <= c['stored'] <= stored_range[1], c
    assert fixed_range[0] <= c['transpose'] <= fixed_range[1], c
    assert stored['F003'].status == 'flagged' and stored['F003'].findings[0].severity == 'error'
    assert stored['F003'].metrics['fraction_alternative_better'] > 0.75
    f2 = stored['F002'].metrics
    assert 0.10 < f2['fraction'] < 0.45                                         # 101-154 of 400 with a strict zero-cell neighbourhood
    strict = _lint(path, cfg, ['F002'], N, nb=0)['F002'].metrics
    assert 80 <= strict['violating'] <= 160, strict
    cfg['source']['transpose'] = True
    if cfg['k'].get('channel') is not None:
        cfg['k']['transpose'] = True
    fixed = _lint(path, cfg, ['F002', 'F003'], N)
    assert fixed['F002'].metrics['violating'] == 0 and fixed['F002'].status == 'ok'
    assert fixed['F003'].status == 'ok' and fixed['F003'].metrics['mean_corr']['stored'] > 0.6


@pytest.mark.parametrize('case', ['HS_OC_refine1', 'HS_SC_refine1', 'IND_8C'])
def test_thermfm_public_sets_are_clean(case):
    path = THERMFM / case
    if not path.exists():
        pytest.skip(f'{path} not present')
    res = _lint(path, load_config('thermfm'), ['F001', 'F002', 'F003', 'F004', 'F005'], 200)
    for code, r in res.items():
        assert r.status in ('ok', 'skipped') and not [f for f in r.findings if f.severity != 'info'], (case, code, r.summary)
    assert res['F002'].metrics['violating'] == 0
    assert res['F003'].metrics['fraction_alternative_better'] == 0.0
    if case != 'HS_SC_refine1':        # HS_SC: wide low-contrast power regions make the plain correlation negative (info note only)
        assert res['F003'].metrics['mean_corr']['stored'] > res['F003'].metrics['mean_corr']['transpose']
        assert not res['F003'].findings


@pytest.mark.parametrize('case', ['3d-ice-layout-geometry5/geometry5', '3d-ice-layout-geometry1', '3d-ice-layout-geometry6'])
def test_3dice_npz_clean(case):
    path = ROOT / case
    if not path.exists():
        pytest.skip(f'{path} not present')
    res = _lint(path, load_config('3dice'), ['F001', 'F002', 'F003', 'F004', 'F005'], 20)
    for code, r in res.items():
        assert r.status == 'ok' and not r.findings, (case, code, r.summary)
    assert res['F002'].metrics['checked'] > 0
