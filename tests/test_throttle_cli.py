"""
--throttle CLI wiring: ScenarioGenerator.attach_throttling itself is already
covered by tests/test_throttling.py's convergence-math tests; this checks the
flag actually reaches scenario_params and survives export, using --simulator
mock so it runs fast without a real 3D-ICE install. Mock mode never calls
apply_throttling (that only runs on the real-simulator path), so this proves
CLI plumbing, not the derate loop itself.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / 'src' / 'main.py'


def run_main(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        capture_output=True, text=True, cwd=str(cwd or REPO), timeout=300,
    )


def test_throttle_flag_reaches_exported_metadata(tmp_path):
    result = run_main(
        '--simulator', 'mock', '--geometry', 'geometry1',
        '--throttle', '--throttle-temp', '88.0',
        '--skip-test', '--output', str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    files = sorted((tmp_path / 'geometry1').glob('*.npz'))
    assert files, "no npz files produced"

    d = np.load(files[0], allow_pickle=True)
    meta = dict(d['metadata'][0])
    assert meta['throttle_enabled'] is True or meta['throttle_enabled'] == True  # noqa: E712
    # Mock mode doesn't run apply_throttling, so it never fired -- but the
    # scenario-level intent (enabled=True) must still be recorded.
    assert meta['throttle_triggered'] in (False, np.False_)


def test_without_throttle_flag_metadata_is_disabled(tmp_path):
    result = run_main(
        '--simulator', 'mock', '--geometry', 'geometry1',
        '--skip-test', '--output', str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    files = sorted((tmp_path / 'geometry1').glob('*.npz'))
    assert files
    d = np.load(files[0], allow_pickle=True)
    meta = dict(d['metadata'][0])
    assert meta['throttle_enabled'] in (False, np.False_)
