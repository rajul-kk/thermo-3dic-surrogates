"""Gate G3 for the classical solvers C0-C2 and the stability test (docs/protocol_nonlinear_A.md amendment 15)."""
import numpy as np
import pytest

from src.scenario import leakage_solvers as S


@pytest.fixture(scope="module")
def plant():
    return S.Plant(6000.0)


def problem(share=0.2, k_d=15.0, scale=40.0, law='exp'):
    return S.Problem(35.0, scale * np.array([1.0, 1.4, 0.7, 1.1]), np.full(4, share), k_d, law)


def test_newton_jacobian_is_symmetric_by_construction(plant):
    assert plant.symmetry_defect() < 1e-12


def test_unit_responses_solve_the_linear_system(plant):
    for j in range(4):
        r = plant.A.dot(plant.X[:, j]) - plant.E[:, j]
        assert np.linalg.norm(r) / np.linalg.norm(plant.E[:, j]) < 1e-9


def test_zero_leakage_share_is_one_linear_solve_for_every_method(plant):
    pb = problem(share=0.0)
    lin = plant.X @ pb.nominal
    for f in (S.c0, S.c1, S.c2):
        run = f(plant, pb)
        assert run.converged or f is S.c0
        assert np.abs(run.states[-1] - lin).max() < 1e-6


def test_c0_is_one_picard_step(plant):
    pb = problem()
    th0 = plant.X @ pb.nominal
    d1 = pb.power(plant.block_T(th0, pb.t_amb))
    run = S.c0(plant, pb)
    assert len(run.states) == 2
    assert np.abs(run.states[1] - plant.X @ d1).max() < 1e-5


@pytest.mark.parametrize("law", ["exp", "arrhenius"])
def test_c1_and_c2_reach_the_reference_fixed_point(plant, law):
    pb = problem(share=0.25, k_d=14.0, scale=45.0, law=law)
    ref = S.reference(plant, pb)
    assert ref is not None
    _, th_ref, Tb_ref = ref
    for f, tol in ((S.c1, 1e-3), (S.c2, 1e-6)):
        run = f(plant, pb)
        assert run.converged and not run.runaway
        assert np.abs(run.states[-1] - th_ref).max() < tol
    # reference is self-consistent
    assert np.max(np.abs(Tb_ref - (pb.t_amb + plant.R @ pb.power(Tb_ref)))) < 1e-9


def test_newton_needs_far_fewer_steps_than_picard_near_critical(plant):
    lo, hi = 10.0, 400.0                                   # bisect the power scale up to the edge of existence
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if S.reference(plant, problem(share=0.35, k_d=11.0, scale=mid)) is not None else (lo, mid)
    pb = problem(share=0.35, k_d=11.0, scale=lo * 0.999)
    ref = S.reference(plant, pb)
    assert ref is not None and S.loop_gain(plant, pb, ref[2]) > 0.9
    n2, n1 = len(S.c2(plant, pb).states), len(S.c1(plant, pb).states)
    assert n2 * 5 < n1


def test_runaway_is_flagged_by_all_methods_and_by_the_stability_test(plant):
    pb = problem(share=0.35, k_d=10.0, scale=400.0)
    assert S.reference(plant, pb) is None
    assert S.c1(plant, pb).runaway and S.c2(plant, pb).runaway
    ok, rho, _ = S.classify(plant, pb)
    assert not ok and rho >= 1.0


def test_stability_test_accepts_a_stable_case(plant):
    pb = problem(share=0.2, k_d=20.0, scale=30.0)
    ok, rho, _ = S.classify(plant, pb)
    ref = S.reference(plant, pb)
    assert ok and ref is not None and rho < 1.0
    assert S.loop_gain(plant, pb, ref[2]) == pytest.approx(rho, rel=1e-6)
