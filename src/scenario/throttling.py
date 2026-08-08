"""
Package-level thermal throttling (DVFS): iterative power derating when peak
temperature exceeds a junction-temperature limit.

Real chips (mobile SoCs, laptop CPUs) reduce power when the hottest point on
the package exceeds Tjmax (commonly ~95-105 C). This makes power a function
of the very temperature field being solved for -- a real, closed-loop
nonlinearity that a scenario-fixed power source cannot express. Every other
mechanism in this benchmark (per-cell power, TSV fields, underfill layouts,
microchannel cooling at a fixed flow rate) still maps a fixed source to a
temperature field; throttling is the first one that doesn't.

3D-ICE itself only solves a single fixed-source steady state, so the loop
lives here: solve, check peak T, derate, re-solve, repeat until the peak
settles within tolerance of the threshold or the power floor is hit. This
means throttled scenarios cost several 3D-ICE solves instead of one -- see
goal.md for the measured cost.
"""

from typing import Any, Dict, Tuple

import numpy as np


def apply_throttling(
    simulator,
    geometry,
    scenario_params: Dict[str, Any],
    scenario_name: str,
    max_iterations: int = 6,
    tol_c: float = 0.3,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Iteratively solve and derate power until peak temperature converges.

    Mutates `scenario_params['power_blocks']` in place to the FINAL derated
    values, so any code that reads scenario_params after this call (npz
    export, statistics) sees the power that was actually delivered, not the
    nominal un-throttled request.

    Args:
        simulator: A ThermalSimulator (must implement .simulate()).
        geometry: Geometry object.
        scenario_params: Scenario dict; must carry 'throttle_temp_c',
            'throttle_gain', 'throttle_power_floor' (see ScenarioParameters).
        scenario_name: Passed through to simulator.simulate().
        max_iterations: Hard cap so a pathological config can't loop forever.
        tol_c: Converged once peak temperature is within this many degrees
            of throttle_temp_c (only reachable while power is above the floor).

    Returns:
        (parsed, throttle_info) where `parsed` matches
        ThermalSimulator.simulate()'s return shape ({'coords', 'temperature'})
        from the FINAL iteration, and throttle_info is a dict of
        {throttle_iterations, throttle_derate_factor, throttle_triggered,
        throttle_peak_temp_c, throttle_converged} for logging/export.
    """
    threshold_c = float(scenario_params.get('throttle_temp_c', 95.0))
    gain = float(scenario_params.get('throttle_gain', 2.0))
    power_floor = float(scenario_params.get('throttle_power_floor', 0.3))

    original_power_blocks = dict(scenario_params['power_blocks'])
    derate = 1.0
    parsed = None
    peak_c = float('nan')
    converged = False

    for iteration in range(1, max_iterations + 1):
        scenario_params['power_blocks'] = {
            name: value * derate for name, value in original_power_blocks.items()
        }
        parsed = simulator.simulate(geometry, scenario_params, scenario_name)
        peak_c = float(parsed['temperature'].max()) - 273.15

        if peak_c <= threshold_c + tol_c:
            converged = True
            break

        overshoot_frac = (peak_c - threshold_c) / threshold_c
        new_derate = max(power_floor, 1.0 - gain * overshoot_frac)
        if abs(new_derate - derate) < 1e-3:
            # Fixed point reached (commonly: pinned at the power floor and
            # still over threshold -- the geometry/cooling genuinely can't
            # hold this workload under this throttle limit).
            converged = (new_derate > power_floor)
            derate = new_derate
            break
        derate = new_derate

    throttle_info = {
        'throttle_iterations': iteration,
        'throttle_derate_factor': derate,
        'throttle_triggered': derate < 1.0,
        'throttle_peak_temp_c': peak_c,
        'throttle_converged': converged,
        # The NOMINAL (pre-throttle) request, distinct from
        # scenario_params['power_blocks'] which this function has mutated to
        # the final delivered power. Without this, a baseline/model trained
        # on the exported metadata sees only the already-resolved delivered
        # power as its input -- the answer, not the question -- which makes
        # the closed-loop nonlinearity invisible and the fit look trivially
        # linear regardless of whether throttling fired. See goal.md.
        'throttle_nominal_power_blocks': original_power_blocks,
    }
    return parsed, throttle_info
