"""
The orchestrator must not report success over empty or partial output.

Per-scenario failures were logged and then swallowed: a run in which every
scenario raised still printed "PROCESSING COMPLETE" and returned 0. A batch
script driving this saw success while producing no files, which is how a broken
run can be mistaken for a finished dataset.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / 'src' / 'main.py'


def run_main(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        capture_output=True, text=True, cwd=str(cwd or REPO), timeout=300,
    )


def test_unreachable_simulator_exits_nonzero(tmp_path):
    """
    A bogus 3D-ICE executable makes every scenario fail. With the synthetic
    fallback off (the default), that must surface as a non-zero exit rather than
    a cheerful summary over an empty directory.
    """
    r = run_main('--simulator', '3d-ice', '--geometry', 'geometry1',
                 '--skip-train', '--ice-executable', 'definitely-not-a-real-binary',
                 '--output', str(tmp_path / 'out'))

    assert r.returncode != 0, (
        "orchestrator reported success despite every scenario failing:\n"
        + r.stdout[-1500:] + r.stderr[-1500:]
    )
    produced = list((tmp_path / 'out').rglob('*.npz'))
    assert not produced, f"failed run still wrote data: {produced}"


def test_failure_output_names_the_failed_scenarios(tmp_path):
    """A failed run must say WHICH scenarios failed, not just that some did."""
    r = run_main('--simulator', '3d-ice', '--geometry', 'geometry1',
                 '--skip-train', '--ice-executable', 'definitely-not-a-real-binary',
                 '--output', str(tmp_path / 'out'))
    combined = r.stdout + r.stderr
    assert 'FAILED' in combined
    assert 'geometry1_test_' in combined, "failed scenario names not reported"


def test_synthetic_fallback_is_opt_in(tmp_path):
    """
    With --allow-synthetic-fallback the same broken run should complete, since
    the caller explicitly accepted approximate data.
    """
    r = run_main('--simulator', '3d-ice', '--geometry', 'geometry1',
                 '--skip-train', '--ice-executable', 'definitely-not-a-real-binary',
                 '--allow-synthetic-fallback',
                 '--output', str(tmp_path / 'out'))
    assert r.returncode == 0, (
        "explicit --allow-synthetic-fallback should still succeed:\n"
        + r.stdout[-1500:] + r.stderr[-1500:]
    )
    assert list((tmp_path / 'out').rglob('*.npz')), "fallback produced no data"
