"""
Leakage-power / temperature positive feedback (self-consistent electrothermal solve).

Subthreshold leakage rises roughly exponentially with temperature, so total power
is a function of the temperature field being solved for:

    P_total(T) = P_dynamic + P_leak_ref * 2 ** ((T - T_ref) / k_double)

This is the classic electrothermal loop that leakage-aware thermal simulators solve
self-consistently (established since at least Chen et al., ICCAD 2006, "Leakage power
dependent temperature estimation to predict thermal runaway"; more recently ATSim3D,
arXiv:2601.11050). 3D-ICE solves a single fixed-source steady state, so the loop lives
here: solve, recompute leakage at the resulting peak temperature, re-solve, repeat.

**Why this is a strictly harder test than throttling** (`src/scenario/throttling.py`),
and the reason it exists in this repo:

  - Throttling is *negative* feedback: hotter -> less power -> cooler. It is
    self-limiting, converges in ~2 solves, and the map from requested to delivered
    power stays monotone and gently compressive. Measured effect on the linear
    baseline was real but modest (spatial R^2 0.970 -> 0.890/0.919, goal.md Track A).
  - Leakage is *positive* feedback: hotter -> more power -> hotter still. It is
    self-amplifying, and above a critical gain there is no steady state at all
    (thermal runaway). Near that threshold the response to a small change in
    requested power becomes arbitrarily large -- which is exactly the regime a
    per-point linear fit cannot represent, because the underlying map is no longer
    close to affine.

So this is the sharpest available test of this benchmark's central claim. If ridge
still solves the dataset with leakage feedback active, that is strong evidence the
linearity finding is robust rather than an artifact of a benign operating regime.

Convergence note: fixed-point iteration T_{n+1} = f(P(T_n)) converges only while the
loop gain |df/dT| < 1. Divergence here is not a numerical failure -- it is the
physical runaway condition, and is reported as such (`leakage_runaway`) rather than
silently clamped, so runaway scenarios can be filtered or studied deliberately.
"""

from typing import Any, Dict, Tuple

import numpy as np

# Above this peak temperature we declare runaway and stop. Silicon does not
# survive anywhere near here; the point is to terminate the loop with a clear
# physical label rather than iterate toward a meaningless number.
RUNAWAY_PEAK_C = 400.0


def leakage_multiplier(peak_c: float, t_ref_c: float, k_double_c: float,
                       leakage_fraction: float) -> float:
    """
    Total-power multiplier relative to the nominal (dynamic + reference-leakage)
    request, given a peak temperature.

    At T = t_ref_c the multiplier is exactly 1.0, so a scenario that never heats
    above its reference temperature is unchanged -- leakage feedback is opt-in
    physics, not a global rescaling of the dataset.
    """
    if not 0.0 <= leakage_fraction <= 1.0:
        raise ValueError(f"leakage_fraction must be in [0,1], got {leakage_fraction}")
    if k_double_c <= 0:
        raise ValueError(f"k_double_c must be > 0, got {k_double_c}")
    dynamic_frac = 1.0 - leakage_fraction
    return dynamic_frac + leakage_fraction * 2.0 ** ((peak_c - t_ref_c) / k_double_c)


def apply_leakage_feedback(
    simulator,
    geometry,
    scenario_params: Dict[str, Any],
    scenario_name: str,
    max_iterations: int = 12,
    tol_c: float = 0.2,
    damping: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Self-consistently solve for temperature with temperature-dependent leakage.

    Mutates `scenario_params['power_blocks']` in place to the FINAL converged
    power, so npz export and statistics see the power actually dissipated. The
    nominal (pre-feedback) request is preserved in the returned info dict under
    'leakage_nominal_power_blocks' -- exported as `nominal_block_power_*`, the
    same convention throttling uses, so a baseline is asked to predict the
    resolved temperature from the *requested* power rather than being handed the
    already-resolved answer (see goal.md Track A for the bug that motivated this).

    Args:
        simulator: A ThermalSimulator (must implement .simulate()).
        geometry: Geometry object.
        scenario_params: Must carry 'leakage_fraction' (share of nominal power
            that is leakage at t_ref), 'leakage_k_double_c' (degrees per doubling)
            and 'leakage_t_ref_c'.
        max_iterations: Cap on solves.
        tol_c: Converged when peak temperature moves less than this between
            iterations.
        damping: Under-relaxation on the multiplier update. Undamped fixed-point
            iteration on positive feedback oscillates or overshoots into false
            runaway near the critical gain; damping trades a few extra solves for
            a trustworthy convergence/runaway distinction.

    Returns:
        (parsed, leakage_info). `parsed` is the final iteration's
        {'coords', 'temperature'}.
    """
    leakage_fraction = float(scenario_params.get('leakage_fraction', 0.3))
    k_double_c = float(scenario_params.get('leakage_k_double_c', 15.0))
    # Default 85 C, NOT ambient: a chip's quoted TDP and its leakage share are
    # specified at a nominal junction temperature, not at room temperature.
    # Referencing to ambient would mean every scenario running at a realistic
    # junction temperature gets a large spurious amplification (at 100 C with a
    # 25 C reference and 15 C doubling, leakage would be 2^5 = 32x its nominal
    # value), which is a modelling error, not physics.
    t_ref_c = float(scenario_params.get('leakage_t_ref_c', 85.0))

    nominal_power_blocks = dict(scenario_params['power_blocks'])
    multiplier = 1.0
    parsed = None
    peak_c = float('nan')
    prev_peak_c = float('nan')
    converged = False
    runaway = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        scenario_params['power_blocks'] = {
            name: value * multiplier for name, value in nominal_power_blocks.items()
        }
        parsed = simulator.simulate(geometry, scenario_params, scenario_name)
        peak_c = float(parsed['temperature'].max()) - 273.15

        if peak_c >= RUNAWAY_PEAK_C or not np.isfinite(peak_c):
            runaway = True
            break

        if np.isfinite(prev_peak_c) and abs(peak_c - prev_peak_c) <= tol_c:
            converged = True
            break

        target = leakage_multiplier(peak_c, t_ref_c, k_double_c, leakage_fraction)
        new_multiplier = (1.0 - damping) * multiplier + damping * target
        prev_peak_c = peak_c
        multiplier = new_multiplier
    else:
        # Ran out of iterations without settling. Distinguish "still climbing"
        # (incipient runaway) from "wandering" rather than calling both converged.
        runaway = bool(np.isfinite(prev_peak_c) and peak_c > prev_peak_c + tol_c)

    leakage_info = {
        'leakage_enabled': True,
        'leakage_iterations': iteration,
        'leakage_multiplier': multiplier,
        'leakage_peak_temp_c': peak_c,
        'leakage_converged': converged,
        'leakage_runaway': runaway,
        'leakage_fraction': leakage_fraction,
        'leakage_k_double_c': k_double_c,
        'leakage_t_ref_c': t_ref_c,
        # Same role as throttle_nominal_power_blocks: the question, not the answer.
        'leakage_nominal_power_blocks': nominal_power_blocks,
    }
    return parsed, leakage_info
