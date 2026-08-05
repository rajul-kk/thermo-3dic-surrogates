"""
Dataset integrity checks.

These are the tests that would have caught the three data incidents this project
has already had:

  1. 155 files of synthetic-fallback garbage produced when `--ice-executable`
     was omitted and the error was swallowed.
  2. 13 files cross-contaminated by parallel jobs sharing a temp directory.
  3. Four files with 1600-5000 C temperatures left inside `data/3d-ice/`, where a
     recursive glob would silently ingest them.

They are deliberately cheap (metadata + array statistics only) so they can run on
every commit, and they skip cleanly when no dataset is present.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / 'data' / '3d-ice'

LIVE_GEOMETRIES = ['geometry1', 'geometry2a',
                   'geometry3', 'geometry4', 'geometry5', 'geometry6']

# Physical bounds. Silicon melts at 1414 C; anything approaching that is a bug,
# not a hot chip. The floor is generous -- below any ambient we sweep.
T_MIN_C, T_MAX_C = -50.0, 400.0


def live_npz_files():
    files = []
    for g in LIVE_GEOMETRIES:
        files.extend(sorted((DATA_ROOT / g).glob(f'{g}_*.npz')))
    return files


pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir() or not live_npz_files(),
    reason="no generated dataset present",
)


@pytest.fixture(scope='module')
def files():
    return live_npz_files()


def test_archive_dirs_are_not_inside_live_data_root():
    """
    `*_old_*` archives must live outside data/3d-ice/.

    They contain known-corrupted files. Any script globbing data/3d-ice/**/*.npz
    would train on them silently.
    """
    strays = [p.name for p in DATA_ROOT.glob('*_old_*') if p.is_dir()]
    assert not strays, (
        f"stale archive dirs inside the live data root: {strays}. "
        "Move them to data/_archive/ so recursive globs cannot pick them up."
    )


def test_all_files_have_required_keys(files):
    required = {'coords', 'temp', 'power', 'layer', 'metadata'}
    for f in files:
        keys = set(np.load(f, allow_pickle=True).keys())
        assert required <= keys, f"{f.name} missing {required - keys}"


def test_temperatures_are_physically_plausible(files):
    bad = []
    for f in files:
        T = np.load(f, allow_pickle=True)['temp']
        t_c = T - 273.15
        if not np.isfinite(T).all() or t_c.min() < T_MIN_C or t_c.max() > T_MAX_C:
            bad.append(f"{f.name}: {t_c.min():.1f}..{t_c.max():.1f} C")
    assert not bad, "physically implausible temperatures:\n  " + "\n  ".join(bad)


def test_temperature_never_below_ambient(files):
    """
    A passive conduction solve with only heating sources cannot produce a point
    colder than ambient. Violations indicate a unit error or a corrupted solve.
    """
    bad = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        T, meta = d['temp'], d['metadata'].item()
        amb = float(meta.get('t_ambient_kelvin', 298.15))
        if T.min() < amb - 1.0:          # 1 K tolerance for solver noise
            bad.append(f"{f.name}: min {T.min():.2f} K < ambient {amb:.2f} K")
    assert not bad, "sub-ambient temperatures:\n  " + "\n  ".join(bad)


def test_arrays_are_length_consistent(files):
    for f in files:
        d = np.load(f, allow_pickle=True)
        n = d['coords'].shape[0]
        assert d['coords'].ndim == 2 and d['coords'].shape[1] == 3, f"{f.name} coords shape"
        for k in ('temp', 'power', 'layer'):
            assert d[k].shape[0] == n, f"{f.name}: {k} has {d[k].shape[0]} rows, coords has {n}"


def test_no_constant_temperature_fields(files):
    """
    A perfectly flat field means the solve failed or fell back to a constant.
    Even a near-isothermal scenario has some numerical structure.
    """
    flat = []
    for f in files:
        T = np.load(f, allow_pickle=True)['temp']
        if float(T.std()) < 1e-6:
            flat.append(f.name)
    assert not flat, f"constant temperature fields (failed solve?): {flat}"


def test_power_is_non_negative_and_finite(files):
    for f in files:
        P = np.load(f, allow_pickle=True)['power']
        assert np.isfinite(P).all(), f"{f.name}: non-finite power"
        assert (P >= 0).all(), f"{f.name}: negative volumetric power"


def test_layer_indices_are_in_range(files):
    for f in files:
        d = np.load(f, allow_pickle=True)
        layer, meta = d['layer'], d['metadata'].item()
        n_layers = int(meta['num_layers'])
        assert layer.min() >= 0, f"{f.name}: negative layer index"
        assert layer.max() < n_layers, (
            f"{f.name}: layer index {layer.max()} >= num_layers {n_layers}")


def test_metadata_declares_matching_point_count(files):
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = d['metadata'].item()
        if 'num_points' in meta:
            assert int(meta['num_points']) == d['coords'].shape[0], (
                f"{f.name}: metadata num_points disagrees with coords")
