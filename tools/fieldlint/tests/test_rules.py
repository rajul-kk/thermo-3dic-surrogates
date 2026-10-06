import numpy as np

from fieldlint import Options, dataset_from_arrays
from fieldlint.model import Dataset, Sample
from fieldlint.rules import (dilate, f001_integrity, f002_max_principle, f003_orientation, f004_duplicates, f005_range,
                             f006_linearity, f007_energy, gaussian_smooth)

OPT = Options()


def sev(res):
    return [f.severity for f in res.findings]


# ── helpers ──────────────────────────────────────────────────────────────────────
def test_dilate_and_smooth():
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    d = dilate(m, 1)
    assert d.sum() == 9 and d[2:5, 2:5].all()
    m3 = np.zeros((3, 7, 7), bool)
    m3[1, 3, 3] = True
    assert dilate(m3, 1, 0).sum() == 9 and dilate(m3, 1, 1).sum() == 27
    a = np.zeros((21, 21))
    a[10, 10] = 1
    s = gaussian_smooth(a, 1.5)
    assert abs(s.sum() - 1) < 1e-9 and s.argmax() == 10 * 21 + 10


# ── F001 ─────────────────────────────────────────────────────────────────────────
def test_f001_clean(valid):
    src, u = valid
    r = f001_integrity(dataset_from_arrays(u, src), OPT)
    assert r.status == 'ok'


def test_f001_shape_nan_and_loader_failure(valid):
    src, u = valid
    r = f001_integrity(dataset_from_arrays(u, src[:, :20]), OPT)
    assert 'error' in sev(r) and 'share a grid' in r.findings[0].message
    bad = u.copy()
    bad[3, 4, 4] = np.nan
    r = f001_integrity(dataset_from_arrays(bad, src), OPT)
    assert any('non-finite' in f.message for f in r.findings)

    def boom(i):
        if i == 2:
            raise RuntimeError('broken sample')
        return Sample(u=u[i], source=src[i], id=str(i))
    r = f001_integrity(Dataset(5, boom), OPT)
    assert any('could not be loaded' in f.message and 'broken sample' in f.message for f in r.findings)


# ── F002 ─────────────────────────────────────────────────────────────────────────
def test_f002_clean_passes(valid):
    src, u = valid
    r = f002_max_principle(dataset_from_arrays(u, src, ambient=300.0), OPT)
    assert r.status == 'ok' and r.metrics['violating'] == 0 and r.metrics['checked'] == len(u)


def test_f002_transposed_source_fires(valid):
    src, u = valid
    r = f002_max_principle(dataset_from_arrays(u, src.swapaxes(-1, -2)), OPT)
    assert r.status == 'flagged' and r.metrics['fraction'] > 0.5 and sev(r) == ['error']


def test_f002_injected_off_source_maximum_reports_fraction(valid):
    src, u = valid
    u = u.copy()
    bad = list(range(0, len(u), 8))                                   # every 8th sample (12.5%)
    for i in bad:
        free = np.argwhere(~dilate(src[i] > 0, 3))                     # a cell far from every source
        y, x = free[0]
        u[i, y, x] = u[i].max() + 5.0                                  # sink-free interior maximum off the source
    r = f002_max_principle(dataset_from_arrays(u, src), OPT)
    assert r.metrics['violating'] == len(bad)
    assert abs(r.metrics['fraction'] - len(bad) / len(u)) < 1e-12
    assert sev(r) == ['warning']                                       # below the 20% error threshold
    assert r.metrics['excess']['median'] > 4.0


def test_f002_error_threshold(valid):
    src, u = valid
    u = u.copy()
    for i in range(0, len(u), 2):
        y, x = np.argwhere(~dilate(src[i] > 0, 3))[0]
        u[i, y, x] = u[i].max() + 5.0
    assert sev(f002_max_principle(dataset_from_arrays(u, src), OPT)) == ['error']


def test_f002_discretisation_neighbourhood(valid):
    src, u = valid
    u = u.copy()
    i = 0
    y, x = np.argwhere(src[i] > 0)[0]
    u[i, y - 1, x - 1] = u[i].max() + 1.0                              # one cell outside the source: allowed
    r = f002_max_principle(dataset_from_arrays(u[:1], src[:1]), OPT)
    assert r.metrics['violating'] == 0
    r = f002_max_principle(dataset_from_arrays(u[:1], src[:1]), Options(neighbourhood=0))
    assert r.metrics['violating'] == 1


def test_f002_skips_sinks_vacuous_and_dirichlet(valid):
    src, u = valid
    sink = src.copy()
    sink[:, 0, 0] = -1.0
    r = f002_max_principle(dataset_from_arrays(u, sink), OPT)
    assert r.status == 'skipped' and r.metrics['skipped_negative_source'] == len(u)
    r = f002_max_principle(dataset_from_arrays(u, np.ones_like(src)), OPT)
    assert r.status == 'skipped' and r.metrics['vacuous_or_no_heating'] == len(u)
    # a Dirichlet-held hot boundary is a legitimate maximum
    u2 = u.copy()
    u2[:, 0, :] = u.max() + 10.0
    mask = np.zeros(u.shape[1:], bool)
    mask[0, :] = True
    assert f002_max_principle(dataset_from_arrays(u2, src), OPT).status == 'flagged'
    assert f002_max_principle(dataset_from_arrays(u2, src, dirichlet_mask=mask), OPT).status == 'ok'


def test_f002_no_heating_above_ambient_is_vacuous(valid):
    src, u = valid
    r = f002_max_principle(dataset_from_arrays(300.0 - 1e-3 * np.arange(u.size).reshape(u.shape) / u.size, src,
                                               ambient=300.0), OPT)
    assert r.status == 'skipped'


def test_f002_3d_layers(valid):
    src, u = valid
    s3 = np.zeros((6, 3, 24, 24))
    u3 = np.zeros_like(s3)
    s3[:, 1], u3[:, 1] = src[:6], u[:6]
    for z in (0, 2):
        u3[:, z] = 300 + 0.5 * (u[:6] - 300)                           # cooler neighbouring layers
    assert f002_max_principle(dataset_from_arrays(u3, s3), OPT).metrics['violating'] == 0


# ── F003 ─────────────────────────────────────────────────────────────────────────
def test_f003_clean_ok_and_corr_high(valid):
    src, u = valid
    r = f003_orientation(dataset_from_arrays(u, src), OPT)
    assert r.status == 'ok' and r.metrics['mean_corr']['stored'] > 0.7
    assert r.metrics['mean_corr']['stored'] > r.metrics['mean_corr']['transpose']


def test_f003_transposed_inputs_fire(valid):
    src, u = valid
    r = f003_orientation(dataset_from_arrays(u, src.swapaxes(-1, -2)), OPT)
    assert sev(r) == ['error'] and r.metrics['better_by_variant'].get('transpose', 0) > 0.9 * len(u)
    assert r.metrics['mean_corr']['transpose'] > r.metrics['mean_corr']['stored'] + 0.3


def test_f003_flips_fire(valid):
    src, u = valid
    for name, s in (('flip_x', src[..., ::-1]), ('flip_y', src[:, ::-1, :]), ('flip_xy', src[:, ::-1, ::-1])):
        r = f003_orientation(dataset_from_arrays(u, np.ascontiguousarray(s)), OPT)
        assert sev(r) == ['error'] and max(r.metrics['better_by_variant'], key=r.metrics['better_by_variant'].get) == name


def test_f003_mixed_orientation_is_warning(valid):
    src, u = valid
    s = src.copy()
    s[::5] = s[::5].swapaxes(-1, -2)                                   # about 20% transposed
    r = f003_orientation(dataset_from_arrays(u, s), OPT)
    assert sev(r) == ['warning']


def test_f003_nonsquare_only_flips():
    rng = np.random.default_rng(1)
    src = rng.random((6, 10, 16)) ** 6
    u = 300 + gaussian_smooth(src, 2.0)
    r = f003_orientation(dataset_from_arrays(u, src), OPT)
    assert 'transpose' not in r.metrics['mean_corr'] and r.status == 'ok'


def test_f003_symmetric_source_quiet():
    src = np.zeros((6, 21, 21))
    src[:, 10, 10] = 1.0                                               # centred spot: invariant under every variant
    r = f003_orientation(dataset_from_arrays(300 + gaussian_smooth(src, 3.0), src), OPT)
    assert r.status == 'ok'


# ── F004 ─────────────────────────────────────────────────────────────────────────
def test_f004_clean(valid):
    src, u = valid
    splits = ['train'] * 30 + ['test'] * 18
    r = f004_duplicates(dataset_from_arrays(u, src, splits=splits), OPT)
    assert r.status == 'ok' and r.metrics['duplicate_samples'] == 0


def test_f004_cross_split_leak_including_float32_roundtrip(valid):
    src, u = valid
    u2, s2 = u.copy(), src.copy()
    u2[40] = u2[2]                                                     # test sample 40 copies train sample 2
    u2[41] = u2[3].astype(np.float32).astype(np.float64)               # copy through float32
    s2[40], s2[41] = s2[2], s2[3]
    splits = ['train'] * 30 + ['test'] * 18
    r = f004_duplicates(dataset_from_arrays(u2, s2, splits=splits), OPT)
    assert r.metrics['cross_split'] == 2 and sev(r)[0] == 'error'


def test_f004_within_split_and_unlabelled(valid):
    src, u = valid
    u2 = u.copy()
    u2[10] = u2[5]
    r = f004_duplicates(dataset_from_arrays(u2, src, splits=['train'] * 48), OPT)
    assert r.metrics['within_split'] == 1 and sev(r) == ['warning']
    r = f004_duplicates(dataset_from_arrays(u2, src), OPT)
    assert r.metrics['within_split'] == 1 and not r.metrics['splits_labelled']


def test_f004_distinct_fields_not_flagged(valid):
    src, u = valid
    u2 = u.copy()
    u2[10] = u2[5] + 1e-2                                              # differs by far more than 1e-5 of the range
    assert f004_duplicates(dataset_from_arrays(u2, src), OPT).status == 'ok'


def test_f004_repeated_inputs_across_splits_info(valid):
    src, u = valid
    s2, u2 = src.copy(), u.copy()
    s2[40], u2[40] = s2[2], u2[2] + 1.0
    r = f004_duplicates(dataset_from_arrays(u2, s2, splits=['train'] * 30 + ['test'] * 18), OPT)
    assert sev(r) == ['info']


# ── F005 ─────────────────────────────────────────────────────────────────────────
def test_f005_clean_and_constant(valid):
    src, u = valid
    assert f005_range(dataset_from_arrays(u, src, units='K'), OPT).status == 'ok'
    c = u.copy()
    c[:5] = 300.0
    r = f005_range(dataset_from_arrays(c, src, units='K'), OPT)
    assert r.status == 'flagged' and r.metrics['constant_fields'] == 5


def test_f005_units(valid):
    src, u = valid
    r = f005_range(dataset_from_arrays(u - 273.15, src, units='K'), OPT)            # Celsius labelled K
    assert any('Celsius' in f.message for f in r.findings)
    r = f005_range(dataset_from_arrays(u - 600, src, units='K'), OPT)
    assert 'error' in sev(r)
    assert 'warning' in sev(f005_range(dataset_from_arrays(u + 1500, src, units='K'), OPT))
    assert f005_range(dataset_from_arrays(u - 273.15, src, units='C'), OPT).status == 'ok'
    assert 'error' in sev(f005_range(dataset_from_arrays(u - 600, src, units='C'), OPT))


def test_f005_k_source_and_ambient(valid):
    src, u = valid
    k = np.ones_like(u)
    k[3] = 0.0
    r = f005_range(dataset_from_arrays(u, src, k, units='K'), OPT)
    assert any('k <= 0' in f.message for f in r.findings)
    r = f005_range(dataset_from_arrays(u - 5.0, src, ambient=300.0, units='K'), OPT)
    assert any('below the ambient' in f.message for f in r.findings)
    r = f005_range(dataset_from_arrays(u, np.zeros_like(src), units='K'), OPT)
    assert any('all-zero source' in f.message for f in r.findings)


# ── F006 ─────────────────────────────────────────────────────────────────────────
def test_f006_linear_benchmark_detected(valid):
    from conftest import random_blocks, solve
    rng = np.random.default_rng(5)
    src = random_blocks(rng, 200, 24)
    r = f006_linearity(dataset_from_arrays(solve(src), src), OPT)
    assert r.metrics['r2_heldout'] > 0.95 and sev(r) == ['info'] and 'near-linear' in r.findings[0].message


def test_f006_nonlinear_scores_lower():
    from conftest import random_blocks, solve
    rng = np.random.default_rng(6)
    src = random_blocks(rng, 200, 24)
    lin = solve(src)
    nonlin = 300 + (lin - 300) ** 2 / 20                              # a nonlinear map of the linear response
    r_lin = f006_linearity(dataset_from_arrays(lin, src), OPT).metrics['r2_heldout']
    r_non = f006_linearity(dataset_from_arrays(nonlin, src), OPT).metrics['r2_heldout']
    assert r_non < r_lin - 0.02


def test_f006_skips_small_and_without_source(valid):
    src, u = valid
    assert f006_linearity(dataset_from_arrays(u[:10], src[:10]), OPT).status == 'skipped'
    assert f006_linearity(dataset_from_arrays(u), OPT).status == 'skipped'


# ── F007 ─────────────────────────────────────────────────────────────────────────
def test_f007(valid):
    src, u = valid
    p_in = src.sum(axis=(1, 2)) * 0.5 ** 2
    r = f007_energy(dataset_from_arrays(u, src, spacing=(0.5, 0.5), flux_out=p_in), OPT)
    assert r.status == 'ok' and abs(r.metrics['ratio_median'] - 1) < 1e-12
    bad = p_in.copy()
    bad[:4] *= 0.9
    r = f007_energy(dataset_from_arrays(u, src, spacing=(0.5, 0.5), flux_out=bad), OPT)
    assert r.metrics['violating'] == 4 and sev(r) == ['error']
    assert f007_energy(dataset_from_arrays(u, src), OPT).status == 'skipped'


# ── F008 / F009 ──────────────────────────────────────────────────────────────────
from fieldlint.rules import f008_operator_residual, f009_pairing, lateral_operator, operator_r2  # noqa: E402


def test_lateral_operator_matches_laplacian():
    rng = np.random.default_rng(3)
    u = rng.normal(size=(9, 9))
    L = lateral_operator(u)
    ref = 4 * u[1:-1, 1:-1] - u[2:, 1:-1] - u[:-2, 1:-1] - u[1:-1, 2:] - u[1:-1, :-2]
    assert np.allclose(L, ref)
    assert np.allclose(lateral_operator(u, np.full_like(u, 3.0)), 3 * ref)
    assert np.allclose(lateral_operator(u, None, 2.0, 2.0), ref / 4)


def test_f008_clean_correct_orientation_wins(valid):
    src, u = valid
    assert operator_r2(src[0], None, u[0] - 300.0) > 0.999
    r = f008_operator_residual(dataset_from_arrays(u, src), OPT)
    assert r.status == 'ok' and r.metrics['median_r2_stored'] > 0.999
    assert r.metrics['median_r2']['stored'] >= max(v for k, v in r.metrics['median_r2'].items())


def test_f008_detects_transpose_flips_and_rolls(valid):
    src, u = valid
    cases = {'transpose': np.swapaxes(src, -1, -2), 'flip_x': src[:, :, ::-1], 'flip_y': src[:, ::-1],
             'flip_xy': src[:, ::-1, ::-1], 'roll_x+1': np.roll(src, -1, axis=2)}
    for want, bad in cases.items():
        r = f008_operator_residual(dataset_from_arrays(u, np.ascontiguousarray(bad)), OPT)
        assert r.status == 'flagged' and r.findings[0].severity == 'error', want
        assert max(r.metrics['better_by_variant'], key=r.metrics['better_by_variant'].get).startswith(want[:6]), (want, r.metrics['better_by_variant'])


def test_f008_partial_contamination_is_a_warning(valid):
    src, u = valid
    bad = src.copy()
    bad[:6] = np.swapaxes(src[:6], -1, -2)               # 6 of 48 = 12.5%
    r = f008_operator_residual(dataset_from_arrays(u, bad), OPT)
    assert r.status == 'flagged' and r.findings[0].severity == 'warning'


def test_f008_k_only_uniform_source(darcy_like):
    k, u = darcy_like
    src = np.ones_like(u)
    r = f008_operator_residual(dataset_from_arrays(u, src, k), OPT)
    assert r.status == 'ok' and r.metrics['modes']['uni'] == len(u) and r.metrics['median_r2_stored'] > 0.999
    for bad in (np.swapaxes(k, -1, -2), k[:, :, ::-1], k[:, ::-1]):
        r = f008_operator_residual(dataset_from_arrays(u, src, np.ascontiguousarray(bad)), OPT)
        assert r.status == 'flagged' and r.findings[0].severity == 'error'


def test_f008_skips_when_nothing_varies_or_out_of_scope(valid):
    src, u = valid
    r = f008_operator_residual(dataset_from_arrays(u, np.ones_like(u)), OPT)
    assert r.status == 'skipped' and 'nothing to fit' in r.summary
    r = f008_operator_residual(dataset_from_arrays(u), OPT)
    assert r.status == 'skipped'
    noise = np.random.default_rng(5).normal(size=u.shape)                  # not a diffusion field at all
    r = f008_operator_residual(dataset_from_arrays(noise, src), OPT)
    assert r.status == 'skipped' and 'not applicable' in r.summary
    r = f008_operator_residual(dataset_from_arrays(u[:, :1], src[:, :1]), OPT)      # 1 x 24 cells
    assert r.status == 'skipped'


def test_f008_3d_layers_with_lumped_loss(valid):
    src, u = valid
    u3 = np.stack([u, u + 0.0], 1)                                         # two identical layers
    s3 = np.stack([src, src], 1)
    r = f008_operator_residual(dataset_from_arrays(u3, s3), OPT)
    assert r.status == 'ok' and r.metrics['median_r2_stored'] > 0.999
    r = f008_operator_residual(dataset_from_arrays(u3, np.swapaxes(s3, -1, -2)), OPT)
    assert r.status == 'flagged'


def test_f009_clean_and_shifted_and_shuffled(valid):
    src, u = valid
    r = f009_pairing(dataset_from_arrays(u, src), OPT)
    assert r.status == 'ok' and r.metrics['wrong_rows'] == 0
    r = f009_pairing(dataset_from_arrays(np.roll(u, 1, axis=0), src), OPT)
    assert r.status == 'flagged' and r.findings[0].severity == 'error'
    r = f009_pairing(dataset_from_arrays(np.roll(u[:24], 1, axis=0), src[:24]), OPT)       # all 24 scored: neighbours
    assert r.metrics['most_common_offset_of_best_partner'] == 1
    perm = np.random.default_rng(2).permutation(len(u))
    r = f009_pairing(dataset_from_arrays(u[perm], src), OPT)
    assert r.status == 'flagged'


def test_f009_k_only_and_skips(darcy_like, valid):
    k, u = darcy_like
    src = np.ones_like(u)
    assert f009_pairing(dataset_from_arrays(u, src, k), OPT).status == 'ok'
    r = f009_pairing(dataset_from_arrays(np.roll(u, 1, axis=0), src, k), OPT)
    assert r.status != 'ok'                     # flagged, or skipped when every pair fits badly (documented limitation)
    s, uu = valid
    assert f009_pairing(dataset_from_arrays(uu, np.ones_like(uu)), OPT).status == 'skipped'


def test_f008_face_mean_convention_option(darcy_like):
    k, u = darcy_like
    from fieldlint.rules import lateral_operator as lo
    a = lo(u[0], k[0], face='arithmetic')
    h = lo(u[0], k[0])
    assert a.shape == h.shape and not np.allclose(a, h)                      # two contrasts of k -> two conventions differ
    assert np.allclose(lo(u[0], np.full_like(u[0], 2.0), face='arithmetic'), lo(u[0], np.full_like(u[0], 2.0)))
    r = f008_operator_residual(dataset_from_arrays(u, np.ones_like(u), k), Options(op_face_mean='arithmetic'))
    assert r.status in ('ok', 'skipped')


# round 2: fitted face mean and scope gate
import pytest  # noqa: E402
from conftest import solve_k  # noqa: E402
from fieldlint.rules import choose_face_mean, f002_max_principle as _f002, f003_orientation as _f003  # noqa: E402


@pytest.mark.parametrize('face', ['harmonic', 'arithmetic'])
def test_face_mean_is_chosen_and_orientation_still_detected(face):
    rng = np.random.default_rng(7)
    n = 20
    k = np.ones((16, n, n))
    for m in range(16):
        for _ in range(3):
            h, w = rng.integers(3, 8), rng.integers(3, 11)
            y, x = rng.integers(0, n - h), rng.integers(0, n - w)
            k[m, y:y + h, x:x + w] = 0.1
    u = solve_k(k, 1.0, face=face)
    src = np.ones_like(u)
    ds = dataset_from_arrays(u, src, k)
    chosen, scores = choose_face_mean(ds, OPT)
    assert chosen == face and scores[face] > 0.999
    r = f008_operator_residual(ds, OPT)
    assert r.status == 'ok' and r.metrics['face_mean'] == face
    bad = f008_operator_residual(dataset_from_arrays(u, src, np.ascontiguousarray(np.swapaxes(k, -1, -2))), OPT)
    assert bad.status == 'flagged' and bad.metrics['face_mean'] == face      # choice survives a mis-oriented k


def test_scope_gate_1d_and_time_dependent(valid):
    src, u = valid
    line = dataset_from_arrays(u[:, :1, :], src[:, :1, :], ambient=300.0)       # 1 x 24
    for rule in (_f002, _f003, f008_operator_residual, f009_pairing):
        r = rule(line, OPT)
        assert r.status == 'skipped' and 'out of scope' in r.summary and '1D' in r.summary, rule.__name__
    strip = dataset_from_arrays(u[:, :5, :], src[:, :5, :], ambient=300.0)       # 5 x 24: still under 8 cells
    assert f008_operator_residual(strip, OPT).status == 'skipped'
    ds = dataset_from_arrays(u, src, ambient=300.0)
    ds.time_dependent = True
    for rule in (_f002, _f003, f008_operator_residual, f009_pairing):
        r = rule(ds, OPT)
        assert r.status == 'skipped' and 'time-dependent' in r.summary, rule.__name__
    ds.time_dependent = False
    assert _f003(ds, OPT).status == 'ok'
