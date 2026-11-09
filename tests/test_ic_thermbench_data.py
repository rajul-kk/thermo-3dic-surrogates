"""IC-ThermBench loader: default reproduces upstream layout; fix_orientation swaps spatial in-plane axes only."""
import h5py
import numpy as np
import pytest

from scripts.ic_thermbench_data import SCOPES, load_mat_pair, load_scope

B, Z, N = 10, 1, 4


def _write(folder, scope):
    ch = SCOPES[scope]
    rng = np.random.default_rng(0)
    x = rng.random((B, len(ch), Z, N, N)).astype(np.float32)   # raw (B,P,Z,Y,X)
    for i, c in enumerate(ch):
        if c in ('ambient_K', 'h_w_m2k', 'r_convec_k_per_w'):
            x[:, i] = rng.random((B, 1, 1, 1))                # constant per sample
    y = rng.random((B, Z, N, N)).astype(np.float32)
    folder.mkdir(parents=True)
    for name, arr in (('input', x), ('output', y)):
        with h5py.File(folder / f'{name}.mat', 'w') as f:
            f['data'] = arr
    return x, y


def test_default_matches_plain_transpose(tmp_path):
    x, y = _write(tmp_path / 'level4_steady', 'level4')
    xl, yl = load_mat_pair(tmp_path / 'level4_steady')
    assert np.array_equal(xl, np.transpose(x, (0, 4, 3, 2, 1)))
    assert np.array_equal(yl, np.transpose(y, (0, 3, 2, 1)))


def test_fix_orientation_swaps_spatial_only(tmp_path):
    ch = SCOPES['level4']
    x, y = _write(tmp_path / 'level4_steady', 'level4')
    x0, y0 = load_mat_pair(tmp_path / 'level4_steady')
    x1, y1 = load_mat_pair(tmp_path / 'level4_steady', fix_orientation=True, channels=ch)
    assert np.array_equal(y0, y1)
    for i, c in enumerate(ch):
        if c in ('ambient_K', 'h_w_m2k', 'r_convec_k_per_w'):
            assert np.array_equal(x1[..., i], x0[..., i])
        else:
            assert np.array_equal(x1[..., i], np.swapaxes(x0[..., i], 1, 2))
            assert not np.array_equal(x1[..., i], x0[..., i])


def test_load_scope_flag_and_split(tmp_path):
    _write(tmp_path / 'level3_steady', 'level3')
    a = load_scope(tmp_path, 'level3', split_data=False)
    b = load_scope(tmp_path, 'level3', split_data=False, fix_orientation=True)
    assert np.array_equal(b.x_test, np.swapaxes(a.x_test, 1, 2))
    assert np.array_equal(a.y_test, b.y_test)
    c = load_scope(tmp_path, 'level3', fix_orientation=True)
    assert len(c.x_train) + len(c.x_val) + len(c.x_test) == B


def test_fix_orientation_requires_channels_and_square(tmp_path):
    _write(tmp_path / 'level3_steady', 'level3')
    with pytest.raises(ValueError):
        load_mat_pair(tmp_path / 'level3_steady', fix_orientation=True)
