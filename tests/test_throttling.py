"""
Package-level thermal throttling (DVFS): the first mechanism in this
benchmark where power is a function of the temperature field being solved
for, rather than a fixed scenario input. Real chips do this (Tjmax-triggered
power derating); every other change so far (per-cell power, TSV fields,
underfill layouts, microchannel cooling at fixed flow) keeps the source
fixed per scenario.

These tests use a stub simulator (no real 3D-ICE needed) whose peak
temperature is a deterministic function of total requested power, so the
derate/convergence math can be checked exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenario.throttling import apply_throttling


class _LinearStubSimulator:
    """
    peak_T_C = base_c + slope_c_per_w * sum(power_blocks) -- deterministic,
    so the exact derate/convergence trajectory is checkable by hand.
    """
    def __init__(self, base_c: float, slope_c_per_w: float):
        self.base_c = base_c
        self.slope = slope_c_per_w
        self.calls = 0

    def simulate(self, geometry, scenario_params, scenario_name):
        self.calls += 1
        total_w = sum(scenario_params['power_blocks'].values())
        peak_k = self.base_c + self.slope * total_w + 273.15
        return {
            'coords': np.zeros((1, 3)),
            'temperature': np.array([peak_k]),
        }


def _scenario(power_w: float, **throttle_overrides):
    params = {
        'power_blocks': {'core': power_w},
        'htc': 5000.0, 't_ambient': 25.0,
        'throttle_temp_c': 95.0, 'throttle_gain': 2.0, 'throttle_power_floor': 0.3,
    }
    params.update(throttle_overrides)
    return params


def test_no_throttling_when_peak_under_threshold():
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)  # 25 + 0.5*40 = 45C
    params = _scenario(40.0)
    parsed, info = apply_throttling(sim, geometry=None, scenario_params=params,
                                    scenario_name='s')
    assert info['throttle_triggered'] is False
    assert info['throttle_derate_factor'] == pytest.approx(1.0)
    assert info['throttle_iterations'] == 1
    assert sim.calls == 1
    assert params['power_blocks']['core'] == pytest.approx(40.0)


def test_derates_and_converges_when_over_threshold():
    # 25 + 0.5*160 = 105C at full power -- over the 95C threshold.
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)
    params = _scenario(160.0)
    parsed, info = apply_throttling(sim, geometry=None, scenario_params=params,
                                    scenario_name='s', tol_c=0.3)
    assert info['throttle_triggered'] is True
    assert info['throttle_derate_factor'] < 1.0
    assert info['throttle_peak_temp_c'] <= 95.0 + 0.3 + 1e-6
    assert info['throttle_converged'] is True
    # power_blocks mutated to the FINAL derated value, not the nominal request
    assert params['power_blocks']['core'] < 160.0
    assert params['power_blocks']['core'] == pytest.approx(160.0 * info['throttle_derate_factor'])


def test_pins_at_power_floor_when_cooling_cannot_hold_the_limit():
    # Even at the power floor (30%), this config stays over threshold:
    # 25 + 0.5*(1000*0.3) = 175C -- no achievable derate satisfies the limit.
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)
    params = _scenario(1000.0, throttle_power_floor=0.3)
    parsed, info = apply_throttling(sim, geometry=None, scenario_params=params,
                                    scenario_name='s', max_iterations=8)
    assert info['throttle_triggered'] is True
    assert info['throttle_derate_factor'] == pytest.approx(0.3, abs=1e-3)
    assert info['throttle_converged'] is False  # pinned at floor, still over limit
    assert params['power_blocks']['core'] == pytest.approx(1000.0 * 0.3, rel=1e-2)


def test_respects_max_iterations_as_a_hard_cap():
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)
    params = _scenario(160.0)
    parsed, info = apply_throttling(sim, geometry=None, scenario_params=params,
                                    scenario_name='s', max_iterations=2)
    assert info['throttle_iterations'] <= 2
    assert sim.calls <= 2


def test_returned_parsed_result_matches_the_final_iteration():
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)
    params = _scenario(160.0)
    parsed, info = apply_throttling(sim, geometry=None, scenario_params=params,
                                    scenario_name='s')
    final_peak_k = float(parsed['temperature'].max())
    assert final_peak_k - 273.15 == pytest.approx(info['throttle_peak_temp_c'], abs=1e-6)
