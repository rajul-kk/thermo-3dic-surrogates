"""Tests for scripts/lint_external.py on tiny synthetic MATLAB-v7.3 (HDF5) pairs in both supported layouts."""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from lint_external import lint_pair, sample_excess  # noqa: E402


def _write(folder, power, temp, k=None):
    """power/temp (N, L, H, W); channel 0 = power, optional channel 3 = conductivity."""
    p = np.zeros((power.shape[0], 4) + power.shape[1:], np.float32)
    p[:, 0] = power
    if k is not None:
        p[:, 3] = k
    with h5py.File(folder / 'input.mat', 'w') as f:
        f['data'] = p
    with h5py.File(folder / 'output.mat', 'w') as f:
        f['data'] = temp.astype(np.float32)


def _clean(n=3, layers=1, size=8):
    """One powered 2x2 block; temperature peaks on it (maximum principle holds)."""
    power = np.zeros((n, layers, size, size))
    power[:, :, 2:4, 5:7] = 1.0
    yy, xx = np.mgrid[:size, :size]
    temp = 300 + 20 * np.exp(-((yy - 2.5) ** 2 + (xx - 5.5) ** 2) / 4.0)
    return power, np.broadcast_to(temp, power.shape).copy()


def test_sample_excess_flags_hot_unpowered_cell():
    p, t = _clean(1)
    p, t = p[0], t[0]
    assert sample_excess(p, t)[0] == 0.0
    t[0, 6, 1] = 330.0
    ex, f = sample_excess(p, t)
    assert f and f[0].code == 'L005' and abs(ex - (330.0 - t[p > 0].max())) < 1e-6


def test_thermfm_layout_clean_and_violating(tmp_path):
    p, t = _clean(4, layers=2)
    _write(tmp_path, p, t)
    r = lint_pair(tmp_path, 'thermfm')
    assert r['samples_checked'] == 4 and r['samples_with_L005'] == 0
    t[1, 1, 6, 1] += 40.0                                       # sample 1 gets a hot unpowered node in layer 1
    _write(tmp_path, p, t)
    r = lint_pair(tmp_path, 'thermfm')
    assert r['samples_with_L005'] == 1 and r['first_violating_indices'] == [1]
    assert r['excess_K']['max'] > 15


def test_ictherm_layout_transposed_power_is_detected_and_fixed(tmp_path):
    p, t = _clean(3)
    k = np.where(p > 0, 100.0, 0.5)
    _write(tmp_path, p.swapaxes(-1, -2), t, k.swapaxes(-1, -2))   # power stored transposed relative to temperature
    r = lint_pair(tmp_path, 'ictherm')
    assert r['samples_with_L005'] == 3
    assert r['violations_hot_node_at_k_min_fraction'] == 1.0     # the hot node is in gap material
    r = lint_pair(tmp_path, 'ictherm', transpose_power=True)
    assert r['samples_with_L005'] == 0


def test_fully_powered_field_is_vacuous(tmp_path):
    p, t = _clean(2)
    _write(tmp_path, np.ones_like(p), t)
    r = lint_pair(tmp_path, 'thermfm')
    assert r['samples_with_L005'] == 0 and r['samples_with_no_unpowered_cell'] == 2
