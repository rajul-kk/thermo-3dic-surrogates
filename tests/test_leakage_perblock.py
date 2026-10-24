"""Gate G3 (docs/protocol_nonlinear_A.md): implementation tests for the per-block leakage loop."""
import numpy as np
import pytest
from scipy.optimize import brentq, fsolve

from src.scenario.leakage_perblock import (block_power, dlaw_dt, law_arrhenius, law_exp, picard)

TREF, KD = 85.0, 15.0


def linear_plant(Rm, t_amb):
    Rm = np.asarray(Rm, float)
    return lambda P: t_amb + Rm @ P


def test_zero_leakage_share_is_a_single_linear_solve():
    Rm = np.array([[3.0, 0.5], [0.5, 2.0]])
    p_dyn = np.array([4.0, 6.0])
    calls = []
    plant = linear_plant(Rm, 45.0)
    res = picard(lambda P: (calls.append(P.copy()), plant(P))[1], p_dyn, np.zeros(2), TREF, KD)
    assert res.converged and not res.runaway and res.iterations == 1 and len(calls) == 1
    assert np.allclose(res.T, 45.0 + Rm @ p_dyn, atol=0, rtol=0)
    assert np.allclose(res.P, p_dyn)


def test_one_block_matches_analytic_fixed_point():
    r_th, t_amb, p_dyn, p_leak = 2.0, 50.0, 10.0, 3.0
    f = lambda T: T - (t_amb + r_th * (p_dyn + p_leak * 2.0 ** ((T - TREF) / KD)))
    t_star = brentq(f, t_amb, 100.0)
    res = picard(lambda P: np.array([t_amb + r_th * P[0]]), [p_dyn], [p_leak], TREF, KD, tol=1e-10, max_iter=300)
    assert res.converged and abs(res.T[0] - t_star) < 1e-6
    assert abs(res.P[0] - (p_dyn + p_leak * 2.0 ** ((t_star - TREF) / KD))) < 1e-8


def test_multi_block_matches_root_find_and_residual_rule():
    rng = np.random.default_rng(0)
    B = 4
    Rm = 0.4 * np.ones((B, B)) + np.diag(rng.uniform(1.0, 2.0, B))
    p_dyn, p_leak = rng.uniform(3, 8, B), rng.uniform(0.5, 2.0, B)
    plant = linear_plant(Rm, 40.0)
    res = picard(plant, p_dyn, p_leak, TREF, KD, tol=1e-9, max_iter=300)
    T_star = fsolve(lambda T: T - plant(block_power(T, p_dyn, p_leak, TREF, KD)), np.full(B, 80.0), xtol=1e-12)
    assert res.converged and np.abs(res.T - T_star).max() < 1e-5
    # self-consistency (protocol 3(i)): P(T_final) equals the delivered power to the stopping tolerance
    assert np.max(np.abs(block_power(res.T, p_dyn, p_leak, TREF, KD) - res.P) / res.P) < 1e-8
    assert res.history[-1][0] == pytest.approx(res.residual)


def test_damping_does_not_change_the_fixed_point():
    Rm = np.array([[2.0, 0.6], [0.6, 2.5]])
    args = ([6.0, 5.0], [1.5, 2.0], TREF, KD)
    a = picard(linear_plant(Rm, 45.0), *args, w=0.5, tol=1e-10, max_iter=300)
    b = picard(linear_plant(Rm, 45.0), *args, w=1.0, tol=1e-10, max_iter=300)
    assert a.converged and b.converged and np.abs(a.T - b.T).max() < 1e-6


def test_runaway_is_detected_and_not_dropped():
    # loop gain > 1: R * dP/dT = 2.5 * 8 * ln2/15 ~ 0.9 at the start grows past 1 as T rises
    res = picard(lambda P: np.array([60.0 + 12.0 * P[0]]), [10.0], [8.0], TREF, KD)
    assert res.runaway and not res.converged


def test_arrhenius_law_matches_exponential_value_and_slope_at_tref():
    assert law_arrhenius(TREF, TREF, KD) == pytest.approx(1.0)
    assert dlaw_dt(law_arrhenius, TREF, TREF, KD) == pytest.approx(dlaw_dt(law_exp, TREF, TREF, KD), rel=1e-6)
    # and it differs away from Tref (otherwise it is not an OOD law)
    assert abs(law_arrhenius(TREF + 40, TREF, KD) / law_exp(TREF + 40, TREF, KD) - 1.0) > 0.01
