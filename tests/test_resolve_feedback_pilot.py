"""scripts/resolve_feedback_pilot.py with a stub simulator in place of 3D-ICE.
A stored file is made by the generators' own process_scenario() path; the script must rebuild the nominal scenario from the
metadata alone and reproduce it, and its generator cross-check must notice a metadata/generator disagreement."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.resolve_feedback_pilot as rfp
from scripts.gen_leakage_pilot import LEAKAGE_SETTINGS
from src.core.geometry_builders import build_geometry1
from src.core.mesh import generate_coords_and_indices
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario
from src.scenario.generator import ScenarioGenerator

_GRID = {}


class StubSim:
    """Temperature = ambient + 0.3 K per watt of delivered power: enough for the feedback loops to iterate and settle."""

    def __init__(self, config_dir=None, output_dir=None, executable=None):
        self.executable = executable

    def simulate(self, geometry, scenario, name):
        if 'coords' not in _GRID:
            _GRID['coords'], _ = generate_coords_and_indices(geometry, uniform_z=False)
        watts = sum(b.power_watts(scenario['power_blocks'].get(b.name, 0.0)) for b in geometry.power_blocks)
        temp = np.full(len(_GRID['coords']), 273.15 + scenario['t_ambient'] + 0.3 * watts)
        return {'coords': _GRID['coords'].astype(np.float64), 'temperature': temp}


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    monkeypatch.setattr(rfp, 'ICESimulator', StubSim)


def _store(tmp_path, kind, n=6):
    """Write `geometry1_train_<n>` the way gen_leakage_pilot.py / src.main --throttle do; returns the source dir."""
    g = build_geometry1()
    gen = ScenarioGenerator()
    sc = gen.generate_all_scenarios(g)
    if kind == 'throttle':
        gen.attach_throttling(sc)
    d = sc[n - 1].to_dict()
    if kind == 'leakage':
        frac, k = LEAKAGE_SETTINGS[(n - 1) % len(LEAKAGE_SETTINGS)]
        d.update(leakage_enabled=True, leakage_fraction=frac, leakage_k_double_c=k, leakage_t_ref_c=85.0)
    src = tmp_path / 'src'
    src.mkdir()
    process_scenario(f'geometry1_train_{n:03d}', d, g, StubSim(), NPZExporter(output_dir=src), StatisticsCalculator(),
                     src, generate_plots=False)
    return src


@pytest.mark.parametrize('kind', ['leakage', 'throttle'])
def test_feedback_file_is_rebuilt_from_its_metadata_and_reproduced(tmp_path, kind, capsys):
    src = _store(tmp_path, kind)
    stored = np.load(src / 'geometry1_train_006.npz', allow_pickle=True)
    assert stored['metadata'].item()[f'{kind}_enabled']
    rc = rfp.main([str(src), 'geometry1_train_006', '--out', str(tmp_path / 'out')])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert 'generator cross-check OK' in text and 'MATCH' in text and 'max|dT|=0 K' in text
    new = np.load(tmp_path / 'out' / 'geometry1_train_006.npz', allow_pickle=True)
    assert np.array_equal(new['temp'], stored['temp']) and np.array_equal(new['power'], stored['power'])
    assert np.array_equal(new['power_nominal'], stored['power_nominal'])


def test_dry_run_does_not_solve(tmp_path, monkeypatch):
    src = _store(tmp_path, 'leakage')
    monkeypatch.setattr(rfp, 'process_scenario', lambda *a, **k: pytest.fail('dry run must not solve'))
    assert rfp.main([str(src), 'geometry1_train_006', '--dry-run', '--out', str(tmp_path / 'out')]) == 0


def test_generator_cross_check_flags_a_changed_metadata(tmp_path, capsys):
    src = _store(tmp_path, 'leakage')
    f = src / 'geometry1_train_006.npz'
    d = np.load(f, allow_pickle=True)
    arrays = {k: d[k] for k in d.files}
    m = arrays['metadata'].item()
    m['nominal_block_power_block1'] *= 1.5
    arrays['metadata'] = np.array([m], dtype=object)
    np.savez_compressed(f, **arrays)
    assert rfp.main([str(src), 'geometry1_train_006', '--dry-run', '--out', str(tmp_path / 'out')]) == 1
    text = capsys.readouterr().out
    assert 'MISMATCH' in text and 'nominal power block1' in text


def test_non_feedback_file_is_refused(tmp_path):
    g = build_geometry1()
    d = ScenarioGenerator().generate_all_scenarios(g)[0].to_dict()
    src = tmp_path / 'src'
    src.mkdir()
    process_scenario('geometry1_train_001', d, g, StubSim(), NPZExporter(output_dir=src), StatisticsCalculator(), src,
                     generate_plots=False)
    with pytest.raises(ValueError, match='not a feedback scenario'):
        rfp.main([str(src), 'geometry1_train_001', '--dry-run', '--out', str(tmp_path / 'out')])
