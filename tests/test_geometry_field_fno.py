"""
End-to-end wiring of the Track B geometry-aware field (distance to nearest
power block) through FNODataset -> FNO3d -> FNOTrainer, on real data. Both
opt-in (use_geometry_field=False by default, so existing checkpoints and
callers are unaffected) and functional (the field is nonzero and varies).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import get_geometry_by_name
from src.fno.data_loader import FNODataset
from src.fno.model import build_fno
from src.pinn.data_loader import compute_norm_stats

DATA = Path(__file__).resolve().parent.parent / 'data' / '3d-ice' / 'geometry1'


def _skip_if_no_data():
    if not DATA.exists() or not list(DATA.glob('*.npz')):
        pytest.skip("data/3d-ice/geometry1 not present in this environment")


def test_dataset_without_geometries_zero_fills_the_field():
    _skip_if_no_data()
    files = sorted(DATA.glob('*.npz'))[:3]
    geoms = {'geometry1': get_geometry_by_name('geometry1')}
    ns = compute_norm_stats(files, geoms)
    ds = FNODataset(files, ns, expected_grid=(1, 1, 1), target_grid=(50, 50, 10))
    assert torch.all(ds.items[0]['dist_to_block_norm'] == 0.0)


def test_dataset_with_geometries_computes_a_real_field():
    _skip_if_no_data()
    files = sorted(DATA.glob('*.npz'))[:3]
    geoms = {'geometry1': get_geometry_by_name('geometry1')}
    ns = compute_norm_stats(files, geoms)
    ds = FNODataset(files, ns, expected_grid=(1, 1, 1), target_grid=(50, 50, 10),
                    geometries=geoms)
    field = ds.items[0]['dist_to_block_norm']
    assert field.shape == (50, 50, 10)
    assert field.min() == pytest.approx(0.0, abs=1e-3)
    assert field.max() > 0.0
    # z-invariant: every z-slice should be (near-)identical since the field
    # only depends on (x, y) -- small tolerance for trilinear resampling.
    assert torch.allclose(field[:, :, 0], field[:, :, -1], atol=0.05)


def test_model_without_geometry_field_ignores_extra_kwarg_absence():
    model = build_fno(grid_shape=(8, 8, 6), modes=(2, 2, 2), hidden_ch=4, n_blocks=1)
    assert model.use_geometry_field is False
    Q = torch.randn(2, 8, 8, 6)
    L = torch.rand(2, 8, 8, 6)
    out = model(Q, L, torch.rand(2), torch.rand(2), torch.rand(2, 8, 8, 6))
    assert out.shape == (2, 8, 8, 6)


def test_model_with_geometry_field_requires_the_kwarg():
    model = build_fno(grid_shape=(8, 8, 6), modes=(2, 2, 2), hidden_ch=4, n_blocks=1,
                      use_geometry_field=True)
    Q = torch.randn(2, 8, 8, 6)
    L = torch.rand(2, 8, 8, 6)
    with pytest.raises(ValueError, match='dist_to_block'):
        model(Q, L, torch.rand(2), torch.rand(2), torch.rand(2, 8, 8, 6))


def test_model_with_geometry_field_forward_backward():
    model = build_fno(grid_shape=(8, 8, 6), modes=(2, 2, 2), hidden_ch=4, n_blocks=1,
                      use_geometry_field=True)
    Q = torch.randn(2, 8, 8, 6)
    L = torch.rand(2, 8, 8, 6)
    dist = torch.rand(2, 8, 8, 6)
    out = model(Q, L, torch.rand(2), torch.rand(2), torch.rand(2, 8, 8, 6),
               dist_to_block=dist)
    assert out.shape == (2, 8, 8, 6)
    out.sum().backward()
    assert model.lift.weight.grad is not None
    # 6-channel lift layer, not 5 -- confirms the extra channel is really wired in
    assert model.lift.weight.shape[1] == 6
