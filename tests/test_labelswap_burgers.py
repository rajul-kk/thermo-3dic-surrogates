"""The Burgers label-swap run: the reference solver and the FNO used in the protocol."""
import numpy as np
import pytest

from scripts import labelswap_burgers as lb


def test_small_amplitude_mode_decays_at_the_viscous_rate():
    """At tiny amplitude Burgers is the heat equation: a sine mode decays as exp(-nu k^2 t)."""
    x = (np.arange(256) + 0.5) / 256
    u0 = 1e-6 * np.sin(2 * np.pi * 3 * x)
    u = lb.resolve(u0[None], 0.1, 0.01, M=512, dt=1e-3)[0]
    expect = u0 * np.exp(-0.01 * (2 * np.pi * 3) ** 2 * 0.1)
    assert np.abs(u - expect).max() < 1e-3 * np.abs(u0).max()


def test_reference_conserves_the_mean_and_converges():
    x = (np.arange(256) + 0.5) / 256
    u0 = 0.5 + np.sin(2 * np.pi * x) + 0.3 * np.cos(2 * np.pi * 2 * x)
    a = lb.resolve(u0[None], 0.05, 0.01 / np.pi, M=1024, dt=1e-4)[0]
    b = lb.resolve(u0[None], 0.05, 0.01 / np.pi, M=2048, dt=5e-5)[0]
    assert abs(a.mean() - u0.mean()) < 1e-10
    assert np.abs(a - b).max() < 1e-6


def test_fno_maps_a_batch_of_fields_to_fields():
    torch = pytest.importorskip('torch')
    model = lb._fno(width=8, modes=4, layers=2)
    out = model(torch.zeros(3, 64))
    assert out.shape == (3, 64)


def test_fit_predict_is_seeded():
    pytest.importorskip('torch')
    rng = np.random.default_rng(0)
    u0, y = rng.normal(size=(12, 32)), rng.normal(size=(12, 32))
    hp = {**lb.FNO, 'width': 4, 'modes': 2, 'layers': 1, 'epochs': 2, 'batch': 4}
    a = lb.fit_predict(u0, y, np.arange(8), np.arange(8, 12), seed=1, hp=hp)
    b = lb.fit_predict(u0, y, np.arange(8), np.arange(8, 12), seed=1, hp=hp)
    assert a.shape == (4, 32) and np.array_equal(a, b)


def test_folds_hold_every_sample_out_once():
    folds = lb.cv_folds(400)
    assert len(folds) == lb.FOLDS and np.array_equal(np.sort(np.concatenate(folds)), np.arange(400))


def test_merge_joins_sessions_and_refuses_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, 'OUT', tmp_path)
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    np.savez(a, old_s0_f0=np.ones(3), new_s0_f0=np.zeros(3))
    np.savez(b, old_s1_f0=np.ones(3), new_s0_f0=np.zeros(3))
    lb.stage_merge([a, b])
    assert set(np.load(tmp_path / 'predictions.partial.npz')) == {'old_s0_f0', 'new_s0_f0', 'old_s1_f0'}
    np.savez(b, new_s0_f0=np.ones(3))
    with pytest.raises(SystemExit, match='differs'):
        lb.stage_merge([a, b])
