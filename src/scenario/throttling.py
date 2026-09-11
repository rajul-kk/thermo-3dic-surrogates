"""
Package-level thermal throttling (DVFS): iterative power derating when peak temperature exceeds a junction-temperature limit.
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
    """Iteratively solve and derate power until peak temperature converges."""
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
