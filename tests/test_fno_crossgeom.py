"""
Cross-geometry FNO loading.

The point of this benchmark is several geometries with distinct physics, learned
by neural operators. FNO takes an FFT over the spatial dims, so it needs one grid
shape -- but the eight geometries produce five:

    (100, 100,  6)  geometry1, geometry3
    ( 80,  80, 10)  geometry2a/b/c
    (100,  56,  6)  geometry4
    (100,  56, 11)  geometry5
    ( 56, 168, 11)  geometry6

FNODataset(target_grid=...) resamples them onto a shared grid. These tests pin
the two properties that make that safe: the shapes really do unify, and the
geometries remain distinguishable afterwards.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_all_geometries, get_geometry_by_name
from src.fno.data_loader import FNODataset, _resample
from src.pinn.data_loader import compute_norm_stats

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / 'data' / '3d-ice'
GEOMS = ['geometry1', 'geometry2a', 'geometry4', 'geometry6']
TARGET = (32, 32, 16)


def _files(n_per_geom=2):
    out = []
    for g in GEOMS:
        out += [Path(p) for p in sorted(glob.glob(str(DATA / g / f'{g}_train_*.npz')))[:n_per_geom]]
    return out


pytestmark = pytest.mark.skipif(not DATA.is_dir() or not _files(),
                                reason="no generated dataset present")


@pytest.fixture(scope='module')
def dataset():
    files = _files()
    ns = compute_norm_stats(files, {g: get_geometry_by_name(g) for g in GEOMS})
    return FNODataset(files, ns, expected_grid=(1, 1, 1), target_grid=TARGET)


def test_geometries_really_do_have_incompatible_native_grids():
    """If this ever fails, resampling has become unnecessary -- delete it."""
    shapes = {g.mesh_resolution[:2] for g in build_all_geometries()}
    assert len(shapes) > 1, f"expected several native grid shapes, got {shapes}"


def test_all_geometries_unify_to_one_grid(dataset):
    shapes = {tuple(it['T_norm'].shape) for it in dataset.items}
    assert shapes == {TARGET}, f"resampling did not unify shapes: {shapes}"
    assert len({it['geometry'] for it in dataset.items}) == len(GEOMS)


def test_every_field_shares_the_target_shape(dataset):
    for it in dataset.items:
        for key in ('Q_norm', 'layer_id_norm', 'T_norm'):
            assert tuple(it[key].shape) == TARGET, f"{it['geometry']}/{key}"


def test_physical_extents_distinguish_geometries(dataset):
    """
    Without extent conditioning an 8x8 mm die and a 42x14 mm die resample to
    identical arrays, and the operator would be asked to learn contradictory
    mappings from the same input.
    """
    by_geom = {it['geometry']: tuple(it['geom_extent_norm'].tolist())
               for it in dataset.items}
    assert len(set(by_geom.values())) == len(by_geom), (
        f"extent conditioning does not separate geometries: {by_geom}")
    for g, ext in by_geom.items():
        assert all(v > 0 for v in ext), f"{g}: non-positive extent {ext}"


def test_resampling_preserves_the_field(dataset):
    """
    Round-tripping back to native resolution must recover the field. Measured
    error is well under 0.1 K against a signal of several K.
    """
    files = _files(1)
    ns = compute_norm_stats(files, {g: get_geometry_by_name(g) for g in GEOMS})
    for f in files:
        import numpy as np
        d = np.load(f, allow_pickle=True)
        meta = dict(d['metadata'][0])
        mesh = meta['mesh_resolution']
        nx, ny = int(mesh[0]), int(mesh[1])
        # Derive z-depth from the point count, not the layer count: sub-layer
        # discretisation makes stack elements outnumber geometry layers.
        native = (nx, ny, d['coords'].shape[0] // (nx * ny))

        nat = FNODataset([f], ns, expected_grid=native)
        res = FNODataset([f], ns, expected_grid=(1, 1, 1), target_grid=TARGET)
        back = _resample(res.items[0]['T_norm'], native, 'trilinear')

        rmse_K = float(((back - nat.items[0]['T_norm']) ** 2).mean().sqrt()) * (ns.T_max - ns.T_min)
        assert rmse_K < 1.0, f"{f.name}: resample round-trip RMSE {rmse_K:.3f} K"


def test_layer_identity_uses_nearest_not_interpolation(dataset):
    """
    Layer id is categorical. Interpolating it would invent materials that exist
    nowhere in the stack -- a value between a silicon die and a copper spreader.
    """
    for it in dataset.items:
        vals = torch.unique(it['layer_id_norm'])
        assert len(vals) <= 16, (
            f"{it['geometry']}: {len(vals)} distinct layer ids suggests the "
            f"categorical field was interpolated")
