"""Analytic checks on the PINN physics kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pinn.physics import pde_residual, thermal_conductivity


T_MIN, T_MAX = 300.0, 400.0
T_RANGE = T_MAX - T_MIN
GEOM = (1000.0, 1000.0, 500.0)          # L_x, L_y, L_z in µm


class AnalyticField(nn.Module):
    """Model stub returning a prescribed closed-form normalised temperature."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._p = nn.Parameter(torch.zeros(1))   # keeps autograd machinery happy

    def forward(self, coords, layer_ids, power, htc_norm, t_amb_norm, tsv_frac,
                region_ids=None, tim_k_norm=None):
        return self.fn(coords) + 0.0 * self._p


def run_residual(fn, k_value=100.0, power_norm=0.0, power_scale=1.0, n=64):
    """Evaluate pde_residual for a manufactured field in a single homogeneous layer."""
    torch.manual_seed(0)
    coords = torch.rand(n, 3, dtype=torch.float64)
    layer_ids = torch.zeros(n, dtype=torch.long)
    power = torch.full((n,), power_norm, dtype=torch.float64)
    scal = torch.tensor(0.5, dtype=torch.float64)

    # Adding 0 * sum(coords) keeps the field mathematically identical but ensures
    # every coordinate participates in the autograd graph. Without it, a field that
    # ignores an axis (a constant, or a purely 1D profile) makes torch.autograd.grad
    # raise "differentiated Tensors appear not to have been used".
    connected = lambda c: fn(c) + 0.0 * c.sum(dim=1)

    model = AnalyticField(connected).double()
    return pde_residual(
        model, coords, layer_ids, power,
        htc_norm=scal, t_amb_norm=scal, tsv_frac=scal,
        layer_k=torch.tensor([k_value], dtype=torch.float64),
        si_layer_mask=torch.tensor([False]),      # constant k, no k(T) coupling
        T_min=T_MIN, T_max=T_MAX,
        geom_scale=GEOM, power_scale=power_scale,
    ), coords


# ── Zero-conduction cases ──────────────────────────────────────────────────────
#
# A constant or purely-linear field cannot be used here: its first derivative is
# constant, so k*grad(T) detaches from the autograd graph and the second-derivative
# call has nothing to differentiate. Instead we use a HARMONIC field,
#     T_hat = c * (x_hat^2 - y_hat^2),
# whose Laplacian cancels exactly when L_x == L_y (true for GEOM). It exercises the
# full second-derivative path while having an analytically zero conduction term.

def harmonic(c_coef=0.3):
    return lambda c: c_coef * (c[:, 0] ** 2 - c[:, 1] ** 2)


def test_harmonic_field_has_zero_conduction_term():
    """div(k grad T) = 0 for a harmonic field; any residual is a chain-rule error."""
    assert GEOM[0] == GEOM[1], "harmonic cancellation requires L_x == L_y"
    r, _ = run_residual(harmonic())
    assert r.abs().max().item() < 1e-9, \
        f"harmonic field must give zero conduction, got max |r|={r.abs().max():.3e}"


# ── Source term ────────────────────────────────────────────────────────────────

def test_residual_equals_source_when_conduction_vanishes():
    """With the conduction term zero, the residual must equal Q exactly."""
    # power_scale = 1e18 cancels the W/m^3 -> W/um^3 factor, so Q_phys == power_norm.
    r, _ = run_residual(harmonic(), power_norm=2.0, power_scale=1e18)
    assert torch.allclose(r, torch.full_like(r, 2.0), atol=1e-9), \
        f"expected residual == Q == 2.0, got {r.mean().item():.6f}"


def test_source_term_is_linear_in_power():
    r1, _ = run_residual(harmonic(), power_norm=1.0, power_scale=1e18)
    r2, _ = run_residual(harmonic(), power_norm=2.0, power_scale=1e18)
    assert torch.allclose(r2, 2.0 * r1, atol=1e-9)


def test_source_term_sign_is_positive():
    """
    Residual is div(k grad T) + Q, so a heat source must enter positively. A sign
    flip here would make the PDE loss drive heat out of powered regions.
    """
    r, _ = run_residual(harmonic(), power_norm=1.0, power_scale=1e18)
    assert r.mean().item() > 0


# ── The quantitative check: does div(k grad T) carry the right units? ──────────

def test_quadratic_field_matches_analytic_divergence():
    """Manufactured solution: T_hat = c * z_hat^2, constant k, no source."""
    c, k = 0.25, 100.0
    r, _ = run_residual(lambda cc: c * cc[:, 2] ** 2, k_value=k)
    expected = k * 2.0 * c * T_RANGE / (GEOM[2] ** 2)
    got = r.mean().item()
    assert r.std().item() < 1e-9, "residual should be spatially constant here"
    assert got == pytest.approx(expected, rel=1e-6), (
        f"div(k grad T) unit error: got {got:.6e}, analytic {expected:.6e}, "
        f"ratio {got / expected:.6f} (a ratio equal to T_range={T_RANGE} means "
        f"T_range is applied twice)"
    )


# ── k(T) ───────────────────────────────────────────────────────────────────────

def test_silicon_conductivity_follows_power_law():
    """k_Si(T) = k_300 * (300/T)^1.3, and equals k_300 at exactly 300 K."""
    T_at_300 = torch.tensor([(300.0 - T_MIN) / T_RANGE], dtype=torch.float64)
    k = thermal_conductivity(
        T_at_300, torch.zeros(1, dtype=torch.long),
        torch.tensor([148.0], dtype=torch.float64), torch.tensor([True]),
        T_MIN, T_MAX,
    )
    assert k.item() == pytest.approx(148.0, rel=1e-6)


def test_silicon_conductivity_decreases_with_temperature():
    T = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    k = thermal_conductivity(
        T, torch.zeros(3, dtype=torch.long),
        torch.tensor([148.0], dtype=torch.float64), torch.tensor([True]),
        T_MIN, T_MAX,
    )
    assert k[0] > k[1] > k[2], f"k(T) must fall with temperature, got {k.tolist()}"


def test_non_silicon_conductivity_is_constant():
    T = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    k = thermal_conductivity(
        T, torch.zeros(3, dtype=torch.long),
        torch.tensor([400.0], dtype=torch.float64), torch.tensor([False]),
        T_MIN, T_MAX,
    )
    assert torch.allclose(k, torch.full_like(k, 400.0))
