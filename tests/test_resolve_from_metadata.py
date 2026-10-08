"""scripts/resolve_from_metadata.py without a 3D-ICE solve: the simulator is replaced by a stub that returns the
stored grid and temperature, so what is tested is the scenario the script would hand to 3D-ICE and the arrays it writes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.resolve_from_metadata as rfm
from src.core.geometry_builders import build_geometry1
from src.core.mesh import generate_coords_and_indices
from src.export.npz_exporter import NPZExporter
from src.simulators import ice_simulator

PINNED = '4953952a1ef6d38807ff307212a6f15e5b2ef935'


@pytest.fixture
def fake_sim(monkeypatch):
    """ICESimulator.simulate -> the grid and temperature of the file being re-solved; records each scenario."""
    seen = []
    state = {}

    def simulate(self, geometry, scenario, name):
        seen.append({'geometry': geometry, 'scenario': scenario, 'name': name})
        ice_simulator._LAST_RUN.update({'ice_commit': PINNED, 'ice_executable': self.executable})
        return {'coords': state['coords'].astype(np.float64), 'temperature': state['temp'].astype(np.float64)}

    monkeypatch.setattr(rfm.ICESimulator, 'simulate', simulate)
    monkeypatch.setattr(ice_simulator, '_LAST_RUN', {})
    seen_state = state
    seen_state['seen'] = seen
    return seen_state


def _write_file(tmp_path, geometry, name, params, *, temp_value=330.0):
    coords, _ = generate_coords_and_indices(geometry, uniform_z=False)
    temp = np.full(len(coords), temp_value, dtype=np.float32)
    src = tmp_path / 'src' / geometry.name
    src.mkdir(parents=True, exist_ok=True)
    out = NPZExporter(src).export_scenario(name, geometry, params, temp, coords=coords)
    return out, coords, temp


def _run(fake_sim, f, coords, temp, tmp_path):
    fake_sim['coords'], fake_sim['temp'] = coords, temp
    out = tmp_path / 'out'
    rfm.main([str(tmp_path / 'src'), '--out', str(out)])
    d = np.load(out / f.parent.name / f.name, allow_pickle=True)
    return {k: d[k] for k in d.files}, d['metadata'].item()


def _block_params(geometry):
    return {'power_blocks': {b.name: 5.0 for b in geometry.power_blocks if not b.is_tsv_region},
            'htc': 5000.0, 't_ambient': 40.0, 'pattern': 'uniform'}


def test_output_metadata_records_the_verified_build(fake_sim, tmp_path):
    g = build_geometry1()
    f, coords, temp = _write_file(tmp_path, g, 'geometry1_test_001', _block_params(g))
    arrays, m = _run(fake_sim, f, coords, temp, tmp_path)
    assert m['ice_commit'] == PINNED
    assert m['ice_executable'] == rfm.ICE_EXE
    assert m['resolved_v5_active_split'] is True
    assert np.array_equal(arrays['power'], np.load(f, allow_pickle=True)['power'])      # block-scalar data: unchanged
