"""
Nominal (pre-throttle) per-cell power export.

Track A found that `block_power_*` npz metadata holds the DELIVERED power --
already derated for throttled scenarios -- so ridge was being fed the closed
loop's resolved answer, not the request (goal.md Track A2). That was fixed
for the ridge/metadata path via `nominal_block_power_*`
(scripts/baselines.py::collect_block_keys). FNO has the same problem one
level down: its per-cell `power` array is also the delivered field, so a
model trained on it solves an easier "resolved power -> temperature" task
instead of ridge's "requested power -> temperature" task. This tests the
fix: a `power_nominal` array recomputed from `throttle_nominal_power_blocks`
when present (identical to `power` otherwise), consumed by FNODataset in
place of `power`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_geometry1
from src.core.mesh import generate_coords_and_indices
from src.export.npz_exporter import NPZExporter
from src.fno.data_loader import FNODataset
from src.pinn.data_loader import compute_norm_stats


def _scenario_params(power_blocks, **overrides):
    params = {
        'power_blocks': power_blocks,
        'htc': 5000.0,
        't_ambient': 40.0,
    }
    params.update(overrides)
    return params


def test_power_nominal_equals_power_when_not_throttled():
    geometry = build_geometry1()
    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    fake_temp = np.full(coords.shape[0], 340.0, dtype=np.float32)
    blocks = {b.name: 5.0 for b in geometry.power_blocks if not b.is_tsv_region}

    with tempfile.TemporaryDirectory() as tmp:
        exporter = NPZExporter(Path(tmp))
        out = exporter.export_scenario(
            'geometry1_test_001', geometry, _scenario_params(blocks), fake_temp, coords=coords)
        loaded = exporter.load_scenario(out)

    assert np.array_equal(loaded['power'], loaded['power_nominal'])


def test_power_nominal_differs_from_delivered_power_when_throttled():
    geometry = build_geometry1()
    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    fake_temp = np.full(coords.shape[0], 340.0, dtype=np.float32)
    real_blocks = [b.name for b in geometry.power_blocks if not b.is_tsv_region]

    nominal_blocks = {name: 10.0 for name in real_blocks}
    delivered_blocks = {name: 3.0 for name in real_blocks}  # throttled down to 30%

    params = _scenario_params(
        delivered_blocks,
        throttle_enabled=True,
        throttle_triggered=True,
        throttle_derate_factor=0.3,
        throttle_nominal_power_blocks=nominal_blocks,
    )

    with tempfile.TemporaryDirectory() as tmp:
        exporter = NPZExporter(Path(tmp))
        out = exporter.export_scenario(
            'geometry1_test_002', geometry, params, fake_temp, coords=coords)
        loaded = exporter.load_scenario(out)

    active = loaded['power'] > 0
    assert active.any(), "expected some active-power cells in a die layer"
    assert not np.allclose(loaded['power'], loaded['power_nominal'])
    # Nominal (10 W/cm^2 requested) must be larger than delivered (3 W/cm^2 derated)
    assert loaded['power_nominal'][active].mean() > loaded['power'][active].mean()


def test_load_scenario_falls_back_to_power_for_files_without_power_nominal():
    """Old exports predate this field; loading them must not raise."""
    geometry = build_geometry1()
    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    fake_temp = np.full(coords.shape[0], 320.0, dtype=np.float32)
    fake_power = np.full(coords.shape[0], 1.5e9, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.npz"
        np.savez_compressed(
            path,
            coords=coords.astype(np.float32),
            temp=fake_temp,
            power=fake_power,
            layer=np.zeros(coords.shape[0], dtype=np.int32),
            metadata=np.array([{'scenario_name': 'legacy'}], dtype=object),
        )
        loaded = NPZExporter(Path(tmp)).load_scenario(path)

    assert np.array_equal(loaded['power_nominal'], loaded['power'])


def test_fno_dataset_uses_nominal_power_when_throttled():
    """
    End-to-end: FNODataset's Q_norm must be built from the nominal (higher)
    power request, not the delivered (derated) field, for a throttled
    scenario -- otherwise FNO is solving an easier problem than ridge.
    """
    geometry = build_geometry1()
    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    fake_temp = np.full(coords.shape[0], 340.0, dtype=np.float32)
    real_blocks = [b.name for b in geometry.power_blocks if not b.is_tsv_region]

    nominal_blocks = {name: 10.0 for name in real_blocks}
    delivered_blocks = {name: 3.0 for name in real_blocks}

    params_throttled = _scenario_params(
        delivered_blocks,
        throttle_enabled=True,
        throttle_triggered=True,
        throttle_derate_factor=0.3,
        throttle_nominal_power_blocks=nominal_blocks,
    )
    params_plain = _scenario_params({name: 10.0 for name in real_blocks})

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        exporter = NPZExporter(tmp_path)
        f_throttled = exporter.export_scenario(
            'geometry1_test_003', geometry, params_throttled, fake_temp, coords=coords)
        f_plain = exporter.export_scenario(
            'geometry1_test_004', geometry, params_plain, fake_temp, coords=coords)

        geoms = {'geometry1': geometry}
        ns = compute_norm_stats([f_throttled, f_plain], geoms)
        ds = FNODataset([f_throttled, f_plain], ns, expected_grid=(1, 1, 1),
                        target_grid=tuple(geometry.mesh_resolution))

    throttled_item = next(it for it in ds.items if 'test_003' in it['name'])
    plain_item = next(it for it in ds.items if 'test_004' in it['name'])

    # Same nominal request (10 W/cm^2) in both scenarios -> same Q_norm field,
    # even though the throttled scenario's DELIVERED power was only 3 W/cm^2.
    assert torch.allclose(throttled_item['Q_norm'], plain_item['Q_norm'], atol=1e-4)
