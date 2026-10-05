"""Dataset linter (src/validation/lint.py): each rule on small hand-built inputs, then the real datasets on disk,
whose right answers are known (clean benchmark; archived pre-fix data with the die-edge artefact; a transposed copy)."""
from pathlib import Path

import numpy as np
import pytest

from src.core.geometry_builders import get_geometry_by_name
from src.validation import lint

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / 'data'


def toy(nx=6, ny=4, nz=3, W=6000.0, H=4000.0):
    """A small valid sample: a powered block in the middle z-plane, hottest inside it, ambient 300 K."""
    xc = (np.arange(nx) + 0.5) * W / nx
    yc = (np.arange(ny) + 0.5) * H / ny
    zc = np.array([100.0, 250.0, 320.0])[:nz]
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    power = np.where((X > 1000) & (X < 4000) & (Y > 1000) & (Y < 3000) & (Z == 250.0), 1e9, 0.0).ravel()
    temp = (300.0 + 5.0 + 20.0 * np.exp(-((X - 2500) ** 2 + (Y - 2000) ** 2) / 4e6) * (1 + (Z == 250.0))).ravel()
    meta = {'geometry': 'toy', 'htc': 1e4, 't_ambient_kelvin': 300.0, 'die_width_um': W, 'die_length_um': H,
            'mesh_resolution': (ny, nx, nz), 'num_points': coords.shape[0]}
    return coords, temp, power, meta


def codes(findings, severity=None):
    return sorted({f.code for f in findings if severity is None or f.severity == severity})


def test_valid_sample_has_no_findings():
    assert lint.lint_arrays(*toy()) == []


def test_non_finite_and_mismatched_arrays():
    c, t, p, m = toy()
    t2 = t.copy(); t2[5] = np.nan
    assert 'L001' in codes(lint.lint_arrays(c, t2, p, m), 'error')
    assert codes(lint.lint_arrays(c, t[:-1], p, m)) == ['L001']
    p2 = p.copy(); p2[0] = -1.0
    assert 'L001' in codes(lint.lint_arrays(c, t, p2, m), 'error')


def test_missing_node_breaks_the_tensor_grid():
    c, t, p, m = toy()
    m = dict(m, num_points=len(t) - 1)
    assert 'L002' in codes(lint.lint_arrays(c[:-1], t[:-1], p[:-1], m), 'error')


def test_transposed_lateral_grid_is_named_as_such():
    c, t, p, m = toy()
    sw = c.copy()
    sw[:, 0] = c[:, 1] * 6000.0 / 4000.0        # 4 nodes stretched along the 6 mm width
    sw[:, 1] = c[:, 0] * 4000.0 / 6000.0        # 6 nodes squeezed along the 4 mm length
    found = [f for f in lint.lint_arrays(sw, t, p, m) if f.code == 'L003']
    assert found and found[0].severity == 'error' and 'transposed' in found[0].message


def test_grid_that_matches_neither_orientation_and_wrong_spacing():
    c, t, p, m = toy()
    assert 'L003' in codes(lint.lint_arrays(c, t, p, dict(m, mesh_resolution=(5, 7, 3))), 'error')
    stretched = c.copy(); stretched[:, 0] *= 1.3
    assert 'L003' in codes(lint.lint_arrays(stretched, t, p, m), 'error')


def test_unphysical_temperatures():
    c, t, p, m = toy()
    cold = t.copy(); cold[0] = 290.0
    assert 'L004' in codes(lint.lint_arrays(c, cold, p, m), 'error')
    hot = t.copy(); hot[np.argmax(t)] = 3000.0
    assert 'L004' in codes(lint.lint_arrays(c, hot, p, m), 'error')
    # a runaway the generator flagged is reported, but as a warning with the reason
    flagged = lint.lint_arrays(c, hot, p, dict(m, leakage_runaway=True))
    assert 'L004' in codes(flagged, 'warning') and 'L004' not in codes(flagged, 'error')


def test_hottest_node_in_an_unpowered_cell_violates_the_maximum_principle():
    c, t, p, m = toy()
    bad = t.copy()
    i = int(np.flatnonzero(p == 0)[0])
    bad[i] = t.max() + 2.0                      # the die-edge artefact: an unpowered cell above every powered one
    found = [f for f in lint.lint_arrays(c, bad, p, m) if f.code == 'L005']
    assert found and found[0].severity == 'error' and '2.00 K' in found[0].message
    # within float32 noise of the powered maximum is not a violation
    close = t.copy(); close[i] = t[p > 0].max() + 0.001
    assert 'L005' not in codes(lint.lint_arrays(c, close, p, m))
    # no sources at all: nothing to check
    assert 'L005' not in codes(lint.lint_arrays(c, bad, np.zeros_like(p), m))


def test_metadata_rules():
    c, t, p, m = toy()
    assert 'L007' in codes(lint.lint_arrays(c, t, p, {k: v for k, v in m.items() if k != 'htc'}), 'error')
    assert 'L007' in codes(lint.lint_arrays(c, t, p, dict(m, num_points=7)), 'error')
    assert 'L007' in codes(lint.lint_arrays(c, t, p, dict(m, placement_dx_blocks=10.0)), 'error')      # no dy
    over = lint.lint_arrays(c, t, p, dict(m, layer_k_overrides="{'tim_top': 5.0}"))
    assert codes(over) == ['L009'] and over[0].severity == 'info'


def test_placement_keys_must_name_real_dies():
    """The bug that broke seven scripts: 'placement_dx_orig_*' was read as the offset of a die called 'orig_*'."""
    geom = get_geometry_by_name('geometry5')
    die = geom.die_footprints[0].name
    ok = {f'placement_dx_{die}': 250.0, f'placement_dy_{die}': 0.0}
    assert lint.check_metadata(ok, 0, geom) == [] or codes(lint.check_metadata(ok, 0, geom)) == ['L007']  # REQUIRED keys
    bad = dict(ok, **{f'placement_dx_orig_{die}': 105.0, f'placement_dy_orig_{die}': 0.0})
    msgs = [f.message for f in lint.check_metadata(bad, 0, geom) if 'names no die' in f.message]
    assert len(msgs) == 1 and f'placement_dx_orig_{die}' in msgs[0]
    renamed = dict(ok, **{f'orig_placement_dx_{die}': 105.0})
    assert not [f for f in lint.check_metadata(renamed, 0, geom) if 'names no die' in f.message]


def test_off_grid_footprints_are_flagged_and_grid_aligned_ones_are_not():
    geom = get_geometry_by_name('geometry5')
    names = [d.name for d in geom.die_footprints]
    aligned = {f'placement_d{a}_{n}': v for n in names for a, v in (('x', 250.0), ('y', -250.0))}
    assert lint.check_off_grid(aligned, geom) == []
    shifted = {f'placement_d{a}_{n}': v for n in names for a, v in (('x', 105.0), ('y', 0.0))}
    found = lint.check_off_grid(shifted, geom)
    assert codes(found) == ['L006'] and '0.42 cell' in found[0].message
    assert lint.check_off_grid({'placement_dx_blocks': 105.0, 'placement_dy_blocks': 0.0},
                               get_geometry_by_name('geometry1')) == []       # no footprint layout, no gap material


def test_dataset_report_counts_duplicates_and_archive_paths(tmp_path):
    c, t, p, m = toy()
    arch = tmp_path / '_archive_old' / 'toy'
    arch.mkdir(parents=True)
    for name, temp in (('a', t), ('b', t), ('c', t + 1.0)):
        np.savez(arch / f'{name}.npz', coords=c, temp=temp, power=p, metadata=np.array([m], dtype=object))
    rep = lint.lint_dataset([tmp_path], energy=False)
    assert rep['files'] == 3 and rep['counts']['error'] == 0
    assert rep['by_code']['L010']['files'] == 1 and rep['by_code']['L011']['files'] == 3
    (arch / 'broken.npz').write_bytes(b'not a zip')
    assert lint.lint_dataset([arch / 'broken.npz'])['counts']['error'] == 1


def test_cli_exit_status(tmp_path, capsys):
    from scripts.lint_dataset import main
    c, t, p, m = toy()
    np.savez(tmp_path / 'good.npz', coords=c, temp=t, power=p, metadata=np.array([m], dtype=object))
    assert main([str(tmp_path), '--no-energy']) == 0
    bad = t.copy(); bad[int(np.flatnonzero(p == 0)[0])] = t.max() + 2.0
    np.savez(tmp_path / 'bad.npz', coords=c, temp=bad, power=p, metadata=np.array([m], dtype=object))
    assert main([str(tmp_path), '--no-energy', '--json', str(tmp_path / 'r.json')]) == 1
    assert 'L005' in capsys.readouterr().out and (tmp_path / 'r.json').exists()


# ── the real datasets: known right answers ───────────────────────────────────────
def have(rel):
    return bool(list((DATA / rel).rglob('*.npz'))) if (DATA / rel).is_dir() else False


@pytest.mark.skipif(not have('3d-ice'), reason='no generated dataset present')
@pytest.mark.parametrize('geometry', ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5', 'geometry6'])
def test_fixed_placement_benchmark_is_clean(geometry):
    rep = lint.lint_dataset([DATA / '3d-ice' / geometry], max_files=12)
    assert rep['files'] > 0
    assert rep['counts']['error'] == 0 and rep['counts']['warning'] == 0, rep['findings'][:3]


@pytest.mark.parametrize('geometry', ['geometry4', 'geometry5', 'geometry6'])
def test_corrected_layout_data_is_clean(geometry):
    if not have(f'3d-ice-layout-{geometry}'):
        pytest.skip('no layout dataset present')
    rep = lint.lint_dataset([DATA / f'3d-ice-layout-{geometry}'])
    assert rep['files'] == 45
    assert rep['counts']['error'] == 0 and rep['counts']['warning'] == 0, rep['findings'][:3]


# report §9.32 counts 28 / 36 / 35 with no tolerance; one geometry6 file exceeds by 0.001 K, inside the 0.01 K rule
@pytest.mark.parametrize('geometry,violations', [('geometry4', 28), ('geometry5', 36), ('geometry6', 34)])
def test_archived_pre_fix_layout_data_shows_the_die_edge_artefact(geometry, violations):
    d = DATA / '_archive_pre_snap_20260930' / f'3d-ice-layout-{geometry}'
    if not d.is_dir():
        pytest.skip('archive not present')
    rep = lint.lint_dataset([d], energy=False)
    assert rep['by_code']['L006']['files'] == 45                 # every placement was off-grid
    assert rep['by_code']['L005']['files'] == violations         # the counts measured in report §9.32
    assert rep['by_code']['L011']['files'] == 45


def test_unrepaired_geometry7_copy_is_reported_as_transposed():
    d = DATA / '3d-ice-layout-geometry7'
    if not d.is_dir():
        pytest.skip('dataset not present')
    rep = lint.lint_dataset([d], max_files=5)
    assert rep['by_code']['L003']['files'] == 5
    assert all('transposed' in f['message'] for f in rep['findings'] if f['code'] == 'L003')
