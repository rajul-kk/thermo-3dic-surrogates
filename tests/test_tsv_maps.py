"""
Spatially varying TSV density and anisotropic effective conductivity.

TSV density was one scalar with four discrete values, so it could not support an
interpolation claim. As a field it becomes a per-scenario spatial coefficient --
the part of the solution operator that is genuinely nonlinear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenario.tsv_maps import (K_CU, K_SI, generate_tsv_density_map,
                                   quantise_materials, tsv_effective_k)


def _eff_rank(M):
    M = M - M.mean(0)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-12]
    return float((sv.sum() ** 2) / (sv ** 2).sum())


def test_mean_density_is_preserved():
    for target in (0.03, 0.05, 0.10):
        m = generate_tsv_density_map(48, 48, mean_density=target, seed=3)
        assert m.mean() == pytest.approx(target, rel=0.02)


def test_zero_contrast_reproduces_the_old_scalar():
    """contrast=0 must give a uniform field, so the old behaviour is recoverable."""
    m = generate_tsv_density_map(32, 32, mean_density=0.05, seed=1, contrast=0.0)
    assert m.std() < 1e-12
    assert m.mean() == pytest.approx(0.05)


def test_density_stays_physical():
    m = generate_tsv_density_map(48, 48, mean_density=0.10, seed=5, contrast=1.0)
    assert m.min() >= 0.0 and m.max() <= 0.35 + 1e-9


def test_zero_density_is_all_silicon():
    m = generate_tsv_density_map(16, 16, mean_density=0.0)
    assert np.all(m == 0.0)
    kl, kv = tsv_effective_k(m)
    assert np.allclose(kl, K_SI) and np.allclose(kv, K_SI)


def test_vertical_conductivity_exceeds_lateral():
    """
    A TSV is a copper cylinder: heat runs ALONG it (parallel, arithmetic mean) but
    must CROSS phases laterally (series, harmonic mean). Vertical must therefore
    dominate, and both must sit between the bulk values.
    """
    for phi in (0.03, 0.05, 0.10, 0.20):
        kl, kv = tsv_effective_k(phi)
        assert kv > kl, f"phi={phi}: vertical {kv} should exceed lateral {kl}"
        assert K_SI <= kl <= K_CU and K_SI <= kv <= K_CU


def test_known_values_at_ten_percent():
    """Pins the arithmetic; docs/references.md previously had both figures wrong."""
    kl, kv = tsv_effective_k(0.10)
    assert kv == pytest.approx(0.9 * K_SI + 0.1 * K_CU)      # 173.2
    assert kl == pytest.approx(1.0 / (0.9 / K_SI + 0.1 / K_CU))  # 158.0


def test_quantisation_preserves_structure():
    m = generate_tsv_density_map(64, 64, mean_density=0.05, seed=7)
    idx, centres = quantise_materials(m, n_levels=12)
    assert len(centres) <= 12
    assert idx.shape == m.shape
    approx = centres[idx]
    assert np.abs(approx - m).max() < (m.max() - m.min()) / 12 + 1e-9


def test_maps_are_high_dimensional_across_scenarios():
    """The point of the change: the coefficient field must carry real information."""
    N = 40
    M = np.stack([generate_tsv_density_map(48, 48, mean_density=0.05, seed=s).ravel()
                  for s in range(N)])
    assert _eff_rank(M) > 0.5 * N, "TSV maps collapsed to near the old scalar"


def test_rejects_out_of_range_mean():
    with pytest.raises(ValueError):
        generate_tsv_density_map(16, 16, mean_density=0.9)
