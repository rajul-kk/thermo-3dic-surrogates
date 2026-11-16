"""Report section 9.32: the label-swap analysis recomputed from the stored per-scenario scores reproduces the stored
analysis exactly, and with it the section's betas and decisions."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.validation import artefact_projection as ap

SUMMARY = Path(__file__).resolve().parent.parent / 'results' / 'artefact_projection' / 'projection_summary.json'

# report section 9.32, paired label-swap projection test table: seed-mean beta and decision
REPORTED = {'thermono|geometry4': (0.95, 'learned'), 'thermono|geometry5': (0.77, 'learned'),
            'thermono|geometry6': (0.80, 'learned'), 'fno|geometry4': (0.35, 'inconclusive'),
            'fno|geometry5': (0.06, 'inconclusive'), 'fno|geometry6': (0.26, 'inconclusive'),
            'cno-fno-attn|geometry5': (0.06, 'inconclusive'), 'cno-fno-attn|geometry6': (0.34, 'inconclusive')}


@pytest.fixture(scope='module')
def summary():
    return json.loads(SUMMARY.read_text())


def _same(a, b):
    if isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in b)
    if isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(b, float) and math.isnan(b):
        return isinstance(a, float) and math.isnan(a)
    return a == b


def test_recomputed_analysis_equals_stored(summary):
    for key, e in summary['results'].items():
        recs = ap.records_from_summary(e['per_scenario'])
        for f in ap.FIELDS:
            assert _same(ap.analyse(recs, f), e['analysis'][f]), (key, f)


def test_reported_betas_and_decisions(summary):
    assert set(REPORTED) == set(summary['results'])
    for key, (beta, decision) in REPORTED.items():
        a = ap.analyse(ap.records_from_summary(summary['results'][key]['per_scenario']))
        assert round(a['mean']['mean'], 2) == beta, key
        assert a['decision'] == decision, key


def test_shim_scores_through_the_src_path():
    """scripts/artefact_projection_run.py scores through this module: an exact and a half-reproduced artefact."""
    D = np.zeros(200)
    D[::10] = 2.0
    assert ap.score(D, D)['beta'] == 1.0
    rec = ap.scenario_record({0: 0.5 * D, 1: 0.5 * D}, {0: 0 * D, 1: 0 * D}, D)
    assert [s['beta'] for s in rec['single']] == [0.5, 0.5] and rec['null_nn'][0]['beta'] == 0.0
