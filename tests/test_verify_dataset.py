"""scripts/verify_dataset.py without a solver: node matching, the tolerance, and fingerprints."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.verify_dataset as vd


def _grid():
    x, y, z = np.meshgrid([50.0, 150.0], [50.0, 150.0], [10.0, 20.0], indexing='ij')
    return np.stack([x.ravel(), y.ravel(), z.ravel()], 1)


def test_compare_matches_nodes_in_any_order():
    c = _grid()
    t = np.arange(len(c), dtype=np.float64) + 300.0
    perm = np.random.default_rng(0).permutation(len(c))
    r = vd.compare({'coords': c, 'temp': t}, {'coords': c[perm], 'temp': t[perm]})
    assert r['same_grid'] and r['max_abs_K'] == 0.0 and r['within_tol']


def test_compare_reports_differences_and_tolerance():
    c = _grid()
    t = np.full(len(c), 300.0)
    t2 = t.copy()
    t2[3] += 0.5
    r = vd.compare({'coords': c, 'temp': t}, {'coords': c, 'temp': t2})
    assert abs(r['max_abs_K'] - 0.5) < 1e-12 and abs(r['peak_diff_K'] - 0.5) < 1e-12 and not r['within_tol']


def test_compare_flags_a_different_grid():
    c = _grid()
    t = np.full(len(c), 300.0)
    assert vd.compare({'coords': c, 'temp': t}, {'coords': c[:-1], 'temp': t[:-1]})['same_grid'] is False
    c2 = c.copy()
    c2[0, 0] += 1.0
    assert vd.compare({'coords': c, 'temp': t}, {'coords': c2, 'temp': t})['same_grid'] is False


def test_fingerprint_changes_with_one_value(tmp_path):
    g = tmp_path / 'geometry1'
    g.mkdir()
    c = _grid().astype(np.float32)
    t = np.full(len(c), 300.0, np.float32)
    meta = np.array([{'geometry': 'geometry1'}], dtype=object)
    np.savez(g / 'geometry1_train_001.npz', coords=c, temp=t, metadata=meta)
    a = vd.fingerprint(tmp_path, ['geometry1'])
    t[0] = 300.5
    np.savez(g / 'geometry1_train_001.npz', coords=c, temp=t, metadata=meta)
    b = vd.fingerprint(tmp_path, ['geometry1'])
    assert a['geometry1']['n_files'] == 1
    assert a['geometry1']['files']['geometry1_train_001.npz']['coords'] == b['geometry1']['files']['geometry1_train_001.npz']['coords']
    assert a['geometry1']['digest'] != b['geometry1']['digest']


def test_pick_is_deterministic_and_bounded():
    files = [f'f{i}' for i in range(10)]
    assert vd.pick(files, 3, 0) == vd.pick(files, 3, 0)
    assert len(vd.pick(files, 3, 0)) == 3 and vd.pick(files[:2], 3, 0) == files[:2]
