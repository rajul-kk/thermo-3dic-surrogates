"""
The spatial TSV-density field affected the 3D-ICE ground truth (since
2026-08-05) but was never exported to .npz or exposed to any model as an
input -- only a per-geometry constant scalar (`metadata['tsv_density']`)
reached PINN/FNO/DeepONet/ARO. That made the field a hidden confounder: it
added real variance to the temperature target with no input a model could
condition on. This tests the fix: a real per-point `tsv_frac` array is now
exported, and `src/fno/model.py` accepts it as a genuine per-cell channel
(with a scalar-broadcast fallback for files/callers predating this field).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_geometry2a, build_geometry1
from src.core.mesh import generate_coords_and_indices, generate_tsv_field
from src.export.npz_exporter import NPZExporter
from src.scenario.generator import ScenarioGenerator
from src.fno.model import _as_field, _as_scalar, build_fno


def test_generate_tsv_field_is_zero_without_a_map():
    geometry = build_geometry1()
    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    field = generate_tsv_field(coords, geometry, tsv_map_by_layer=None)
    assert field.shape == (coords.shape[0],)
    assert np.all(field == 0.0)


def test_generate_tsv_field_matches_the_map_inside_the_tsv_layer():
    geometry = build_geometry2a()
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geometry)
    gen.attach_tsv_maps(scenarios, geometry, resolution=16)
    sc = scenarios[0]
    assert sc.tsv_map_by_layer, "geometry2a must carry a TSV map after attach_tsv_maps"

    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    field = generate_tsv_field(coords, geometry, tsv_map_by_layer=sc.tsv_map_by_layer)

    tsv_layer_names = set(sc.tsv_map_by_layer.keys())
    in_tsv = np.zeros(coords.shape[0], dtype=bool)
    for layer in geometry.layers:
        if layer.name in tsv_layer_names:
            in_tsv |= (coords[:, 2] >= layer.z_bottom) & (coords[:, 2] < layer.z_top)

    assert field[in_tsv].mean() == pytest.approx(geometry.tsv_density, rel=0.5)
    assert np.all(field[~in_tsv] == 0.0)


def test_npz_export_round_trips_tsv_frac():
    geometry = build_geometry2a()
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geometry)
    gen.attach_tsv_maps(scenarios, geometry, resolution=16)
    sc = scenarios[0]

    coords, layer_ids = generate_coords_and_indices(geometry, uniform_z=False)
    fake_temp = np.full(coords.shape[0], 350.0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        exporter = NPZExporter(Path(tmp))
        out = exporter.export_scenario(sc.name, geometry, sc.to_dict(), fake_temp, coords=coords)
        loaded = exporter.load_scenario(out)

    assert 'tsv_frac' in loaded
    assert loaded['tsv_frac'].shape == (coords.shape[0],)
    assert loaded['tsv_frac'].max() > 0.0, "geometry2a export must carry a nonzero TSV field"


def test_load_scenario_falls_back_to_zeros_for_files_without_tsv_frac():
    """Old exports predate this field; loading them must not raise."""
    geometry = build_geometry1()
    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    fake_temp = np.full(coords.shape[0], 320.0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.npz"
        np.savez_compressed(
            path,
            coords=coords.astype(np.float32),
            temp=fake_temp,
            power=np.zeros_like(fake_temp),
            layer=np.zeros(coords.shape[0], dtype=np.int32),
            metadata=np.array([{'scenario_name': 'legacy'}], dtype=object),
        )
        loaded = NPZExporter(Path(tmp)).load_scenario(path)

    assert np.all(loaded['tsv_frac'] == 0.0)


def test_fno_as_field_passes_through_a_real_grid():
    B, nx, ny, nz = 3, 4, 4, 4
    field = torch.rand(B, nx, ny, nz)
    out = _as_field(field, B, nx, ny, nz)
    assert out.shape == (B, 1, nx, ny, nz)
    assert torch.allclose(out.squeeze(1), field)


def test_fno_as_field_broadcasts_a_scalar():
    B, nx, ny, nz = 3, 4, 4, 4
    scalar = torch.tensor([0.03, 0.03, 0.03])
    out = _as_field(scalar, B, nx, ny, nz)
    assert out.shape == (B, 1, nx, ny, nz)
    assert torch.allclose(out, torch.full((B, 1, nx, ny, nz), 0.03))


def test_fno_as_scalar_mean_pools_a_field():
    B, nx, ny, nz = 2, 4, 4, 4
    field = torch.zeros(B, nx, ny, nz)
    field[0] = 0.1
    field[1] = 0.2
    out = _as_scalar(field, B)
    assert out.shape == (B,)
    assert torch.allclose(out, torch.tensor([0.1, 0.2]), atol=1e-6)


def test_fno_forward_accepts_a_real_tsv_field_end_to_end():
    grid = (8, 8, 6)
    B = 2
    model = build_fno(grid_shape=grid, modes=(2, 2, 2), hidden_ch=4, n_blocks=1)
    Q = torch.randn(B, *grid)
    L = torch.rand(B, *grid)
    H = torch.rand(B)
    A = torch.rand(B)
    TSV = torch.rand(B, *grid)  # real per-cell field, not a scalar

    out = model(Q, L, H, A, TSV)
    assert out.shape == (B, *grid)
    loss = out.pow(2).mean()
    loss.backward()
    assert model.lift.weight.grad is not None
