"""Per-block leakage-temperature feedback (docs/protocol_nonlinear_A.md, section 2-3).

P_b(T_b) = P_dyn,b + P_leak,b(Tref) * L(T_b), with T_b the mean temperature of block b on its source layer.
Solver-agnostic: ``solve_fn(P)`` takes the block powers (W, shape (B,)) and returns the block-mean temperatures
(C, shape (B,)); 3D-ICE, the finite-volume reference or a test double can be plugged in. Unlike
``src/scenario/leakage.py`` (one global multiplier driven by the peak) the coupling is local, and the stopping rule is the
fixed-point residual, not the change in the peak.
"""
from dataclasses import dataclass, field

import numpy as np


def law_exp(t, tref, k_d):
    """L(T) = 2^((T - Tref)/k_d); the protocol's primary law."""
    return np.exp2(np.minimum((np.asarray(t, float) - tref) / k_d, 1000.0))   # clip: runaway must not overflow


def law_arrhenius(t, tref, k_d, tk0=273.15):
    """L(T) ~ T^2 exp(-Ea/kT) normalised to 1 at Tref, with Ea chosen so the slope at Tref equals the exponential law's
    (the protocol's OOD law)."""
    tk, tr = np.asarray(t, float) + tk0, tref + tk0
    # d ln L/dT at Tref: 2/tr + Ea/(k tr^2) = ln2/k_d  ->  Ea/k = (ln2/k_d - 2/tr) * tr^2
    ea_k = (np.log(2.0) / k_d - 2.0 / tr) * tr ** 2
    return (tk / tr) ** 2 * np.exp(-ea_k * (1.0 / tk - 1.0 / tr))


def dlaw_dt(law, t, tref, k_d, h=1e-4):
    return (law(np.asarray(t, float) + h, tref, k_d) - law(np.asarray(t, float) - h, tref, k_d)) / (2 * h)


@dataclass
class LoopResult:
    P: np.ndarray                  # delivered block powers at exit
    T: np.ndarray                  # block-mean temperatures at exit
    converged: bool
    runaway: bool
    iterations: int
    residual: float                # max_b |P_b(T_b) - P_b| / P_b at exit
    gain: float                    # estimated loop gain: last ratio of successive residuals
    history: list = field(default_factory=list)   # (residual, max T) per iteration


def block_power(t, p_dyn, p_leak_ref, tref, k_d, law=law_exp):
    return p_dyn + p_leak_ref * law(t, tref, k_d)


def picard(solve_fn, p_dyn, p_leak_ref, tref, k_d, law=law_exp, w=0.5, tol=1e-3, max_iter=20, t_runaway=300.0):
    """Damped Picard iteration. Start from the power at Tref (the nominal power). ``w`` drops to 1.0 once the estimated loop
    gain is below 0.5. Runaway: any block above ``t_runaway`` C, or the residual grows three iterations in a row."""
    p_dyn, p_leak_ref = np.asarray(p_dyn, float), np.asarray(p_leak_ref, float)
    P = p_dyn + p_leak_ref
    hist, prev, grow, gain = [], None, 0, 0.0
    for it in range(1, max_iter + 1):
        T = np.asarray(solve_fn(P), float)
        Pn = block_power(T, p_dyn, p_leak_ref, tref, k_d, law)
        res = float(np.max(np.abs(Pn - P) / P))
        hist.append((res, float(T.max())))
        if T.max() > t_runaway:
            return LoopResult(P, T, False, True, it, res, gain, hist)
        if res < tol:
            return LoopResult(P, T, True, False, it, res, gain, hist)
        if prev is not None:
            gain = res / prev if prev > 0 else 0.0
            grow = grow + 1 if res > prev else 0
            if grow >= 3:
                return LoopResult(P, T, False, True, it, res, gain, hist)
            if gain < 0.5:
                w = 1.0
        prev = res
        P = (1.0 - w) * P + w * Pn
    return LoopResult(P, T, False, False, max_iter, res, gain, hist)
