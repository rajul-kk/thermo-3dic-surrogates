"""
Leakage-power / temperature positive feedback.

Unlike throttling (negative feedback, self-limiting), leakage feedback is
self-amplifying and above a critical loop gain has no steady state at all.
These tests use a stub simulator whose peak temperature is a deterministic
linear function of total power, so the fixed point -- and the gain at which it
ceases to exist -- can be computed by hand and checked exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenario.leakage import (RUNAWAY_PEAK_C, apply_leakage_feedback,
                                  leakage_multiplier)


class _LinearStubSimulator:
    """peak_T_C = base_c + slope_c_per_w * sum(power_blocks)."""

    def __init__(self, base_c: float, slope_c_per_w: float):
        self.base_c = base_c
        self.slope = slope_c_per_w
        self.calls = 0

    def simulate(self, geometry, scenario_params, scenario_name):
        self.calls += 1
        total_w = sum(scenario_params['power_blocks'].values())
        peak_k = self.base_c + self.slope * total_w + 273.15
        return {'coords': np.zeros((1, 3)), 'temperature': np.array([peak_k])}


def _scenario(power_w: float, **overrides):
    params = {
        'power_blocks': {'core': power_w},
        'htc': 5000.0, 't_ambient': 25.0,
        'leakage_fraction': 0.3,
        'leakage_k_double_c': 15.0,
        'leakage_t_ref_c': 25.0,
    }
    params.update(overrides)
    return params


# --- the multiplier function itself -------------------------------------------------

def test_multiplier_is_unity_at_reference_temperature():
    # At T_ref leakage is exactly its reference value, so nothing is rescaled.
    assert leakage_multiplier(25.0, 25.0, 15.0, 0.3) == pytest.approx(1.0)


def test_multiplier_doubles_the_leakage_share_after_one_doubling_interval():
    # 70% dynamic + 30% leakage; one doubling interval doubles only the leakage part.
    assert leakage_multiplier(40.0, 25.0, 15.0, 0.3) == pytest.approx(0.7 + 0.6)


def test_multiplier_with_zero_leakage_fraction_is_temperature_independent():
    for t in (25.0, 80.0, 150.0):
        assert leakage_multiplier(t, 25.0, 15.0, 0.0) == pytest.approx(1.0)


def test_multiplier_rejects_invalid_parameters():
    with pytest.raises(ValueError, match='leakage_fraction'):
        leakage_multiplier(50.0, 25.0, 15.0, 1.5)
    with pytest.raises(ValueError, match='k_double_c'):
        leakage_multiplier(50.0, 25.0, 0.0, 0.3)


# --- the closed loop ----------------------------------------------------------------

def test_cool_scenario_converges_and_barely_amplifies():
    # 25 + 0.05*40 = 27 C, only 2 C above reference -> negligible extra leakage.
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.05)
    params = _scenario(40.0)
    parsed, info = apply_leakage_feedback(sim, geometry=None, scenario_params=params,
                                          scenario_name='cool')
    assert info['leakage_converged'] is True
    assert info['leakage_runaway'] is False
    assert info['leakage_multiplier'] == pytest.approx(1.0, abs=0.05)


def test_hot_scenario_converges_to_an_amplified_fixed_point():
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)
    params = _scenario(80.0)
    parsed, info = apply_leakage_feedback(sim, geometry=None, scenario_params=params,
                                          scenario_name='hot')
    if info['leakage_converged']:
        # Self-consistency: the converged multiplier must reproduce itself when
        # re-evaluated at the converged peak temperature.
        expected = leakage_multiplier(info['leakage_peak_temp_c'], 25.0, 15.0, 0.3)
        assert info['leakage_multiplier'] == pytest.approx(expected, rel=0.05)
        # Positive feedback must amplify, never reduce, the delivered power.
        assert info['leakage_multiplier'] > 1.0
    else:
        assert info['leakage_runaway'] is True


def test_high_gain_scenario_runs_away_rather_than_reporting_a_fake_steady_state():
    # Very steep thermal response + fast doubling -> loop gain > 1, no steady state.
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=3.0)
    params = _scenario(120.0, leakage_fraction=0.5, leakage_k_double_c=8.0)
    parsed, info = apply_leakage_feedback(sim, geometry=None, scenario_params=params,
                                          scenario_name='runaway')
    assert info['leakage_runaway'] is True
    assert info['leakage_converged'] is False


def test_power_blocks_are_mutated_to_delivered_power_and_nominal_is_preserved():
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)
    params = _scenario(80.0)
    _, info = apply_leakage_feedback(sim, geometry=None, scenario_params=params,
                                     scenario_name='s')
    # The nominal request survives for export as nominal_block_power_*, so a
    # baseline is asked the question rather than handed the answer.
    assert info['leakage_nominal_power_blocks'] == {'core': 80.0}
    delivered = params['power_blocks']['core']
    assert delivered == pytest.approx(80.0 * info['leakage_multiplier'])
    assert delivered != 80.0


def test_zero_leakage_fraction_reduces_to_a_single_fixed_source_solve():
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=0.5)
    params = _scenario(80.0, leakage_fraction=0.0)
    _, info = apply_leakage_feedback(sim, geometry=None, scenario_params=params,
                                     scenario_name='nofeedback')
    assert info['leakage_multiplier'] == pytest.approx(1.0)
    assert info['leakage_runaway'] is False
    # Degenerates to the ordinary fixed-source problem, i.e. opt-in physics.
    assert params['power_blocks']['core'] == pytest.approx(80.0)


def test_iteration_cap_is_respected():
    sim = _LinearStubSimulator(base_c=25.0, slope_c_per_w=3.0)
    params = _scenario(120.0, leakage_fraction=0.5, leakage_k_double_c=8.0)
    _, info = apply_leakage_feedback(sim, geometry=None, scenario_params=params,
                                     scenario_name='capped', max_iterations=4)
    assert info['leakage_iterations'] <= 4
    assert sim.calls <= 4
