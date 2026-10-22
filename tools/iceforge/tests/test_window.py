"""Gate G4 (docs/protocol_mesh_convergence.md, amendment M3): the local window solver.

* a window covering the whole die reproduces the domain-wide solve;
* with EXACT interface temperatures taken from the domain-wide solve, any interior window reproduces it inside the window;
* with boundary data from the coarse (r=1) solution the error is finite, small, and shrinks as the window grows.
"""
import os

import numpy as np
import pytest

from iceforge import refsolve as R
from iceforge.model import parse_stk

pytest.importorskip("scipy")

STK = os.path.join(os.path.dirname(__file__), "fixtures", "repro", "aligned", "s.stk")
NZ, RW = 2, 2                       # sub-layers per layer, window refinement


@pytest.fixture(scope="module")
def model():
    return R.build_model(parse_stk(STK))


@pytest.fixture(scope="module")
def full(model):
    return R.solve(model, r=RW, nz=NZ, tol=1e-12)


@pytest.fixture(scope="module")
def coarse(model):
    return R.solve(model, r=1, nz=NZ, tol=1e-12)


def crop(full, model, window, r):
    cl, cw = model.cell_l, model.cell_w
    i0, i1 = int(round(window[0] / cl)) * r, int(round(window[1] / cl)) * r
    j0, j1 = int(round(window[2] / cw)) * r, int(round(window[3] / cw)) * r
    return full.T[:, i0:i1, j0:j1]


def test_whole_die_window_equals_domain_wide_solve(model, full):
    win = R.solve_window(model, (0.0, model.chip_l, 0.0, model.chip_w), RW, nz=NZ, tol=1e-12)
    assert win.T.shape == full.T.shape
    assert np.abs(win.T - full.T).max() < 1e-6
    assert win.info['interior_sides'] == []


@pytest.mark.parametrize("window", [
    (2500.0, 7500.0, 2500.0, 7500.0),    # all four sides interior
    (0.0, 5000.0, 2500.0, 7500.0),       # left side on the chip edge (adiabatic)
    (5000.0, 10000.0, 0.0, 5000.0),      # right and bottom sides on the chip edge
    (3000.0, 4000.0, 3000.0, 4000.0),    # small window, 4 x 4 coarse cells
])
def test_exact_interface_data_reproduces_domain_wide_solution(model, full, window):
    ft = R.face_T_from_solution(model, full, window, nz=NZ)
    win = R.solve_window(model, window, RW, nz=NZ, face_T=ft, tol=1e-12)
    assert np.abs(win.T - crop(full, model, window, RW)).max() < 1e-6


def test_coarse_driven_window_error_is_small_and_shrinks_with_window_size(model, full, coarse):
    cx, cy = 0.5 * model.chip_l, 0.5 * model.chip_w
    errs = {}
    for bc in ("dirichlet", "neumann"):
        errs[bc] = []
        for half in (1000.0, 2500.0, 4000.0):
            w = (cx - half, cx + half, cy - half, cy + half)
            win = R.solve_window(model, w, RW, nz=NZ, coarse=coarse, bc=bc, tol=1e-12)
            errs[bc].append(float(np.abs(win.T - crop(full, model, w, RW)).max()))
    for bc, e in errs.items():
        assert all(np.isfinite(e)), (bc, e)
        assert e[-1] <= e[0] + 1e-9, (bc, e)           # the largest window is no worse than the smallest
        assert max(e) < 0.5, (bc, e)                   # regression bound: the rise on this fixture is ~11 K
    print("coarse-driven max |dT| (K) vs half-width 1000/2500/4000 um:", errs)


def test_window_validation(model, coarse):
    with pytest.raises(ValueError):
        R.solve_window(model, (100.0, 5000.0, 0.0, 5000.0), RW, nz=NZ, coarse=coarse)     # not on coarse cell edges
    with pytest.raises(ValueError):
        R.solve_window(model, (0.0, 20000.0, 0.0, 5000.0), RW, nz=NZ, coarse=coarse)      # outside the chip
    with pytest.raises(ValueError):
        R.solve_window(model, (2500.0, 5000.0, 2500.0, 5000.0), RW, nz=NZ)                # interior window, no coarse data
    with pytest.raises(ValueError):
        R.solve_window(model, (2500.0, 5000.0, 2500.0, 5000.0), RW, nz=NZ + 2, coarse=coarse)   # vertical mismatch
    with pytest.raises(ValueError):
        R.solve_window(model, (2500.0, 5000.0, 2500.0, 5000.0), RW, nz=NZ, coarse=coarse, bc="robin")
