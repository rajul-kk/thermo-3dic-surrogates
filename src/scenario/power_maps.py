"""Per-cell power map generation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Map family names accepted by `generate_power_map` and stored in scenario metadata.
POWER_MAP_KINDS = ('grf', 'floorplan', 'mixed')


def _smooth(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian blur via separable FFT convolution."""
    if sigma_cells <= 0:
        return field
    nx, ny = field.shape
    fx = np.fft.fftfreq(nx)[:, None]
    fy = np.fft.fftfreq(ny)[None, :]
    kernel = np.exp(-2.0 * (np.pi * sigma_cells) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * kernel))


def _normalise(field: np.ndarray, floor: float) -> np.ndarray:
    """Shift/scale to [floor, 1]. A floor > 0 keeps idle regions dissipating."""
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-12:
        return np.full_like(field, 1.0)
    unit = (field - lo) / (hi - lo)
    return floor + (1.0 - floor) * unit


def _grf(nx: int, ny: int, corr_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian random field with correlation length `corr_frac` of the die width."""
    sigma_cells = max(corr_frac * max(nx, ny), 0.75)
    return _smooth(rng.standard_normal((nx, ny)), sigma_cells)


def _floorplan(nx: int, ny: int, n_tiles: int, rng: np.random.Generator) -> np.ndarray:
    """Rectangular tiles at independent power levels."""
    def cuts(n, k):
        interior = np.sort(rng.choice(np.arange(1, n), size=min(k - 1, n - 1),
                                      replace=False)) if k > 1 else np.array([], int)
        return np.concatenate([[0], interior, [n]])

    xs, ys = cuts(nx, n_tiles), cuts(ny, n_tiles)
    field = np.zeros((nx, ny), dtype=np.float64)
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            # Heavy-tailed levels: most tiles quiet, a few hot, like real activity.
            field[xs[i]:xs[i + 1], ys[j]:ys[j + 1]] = rng.random() ** 2.5
    return field


def generate_power_map(
    nx: int,
    ny: int,
    kind: str = 'mixed',
    seed: int = 0,
    corr_frac: float = 0.02,
    n_tiles: int = 16,
    idle_floor: float = 0.05,
) -> np.ndarray:
    """Build a normalised (nx, ny) power map with values in [idle_floor, 1]."""
    if kind not in POWER_MAP_KINDS:
        raise ValueError(f"unknown power map kind {kind!r}; expected one of {POWER_MAP_KINDS}")

    rng = np.random.default_rng(seed)

    if kind == 'grf':
        field = _grf(nx, ny, corr_frac, rng)
    elif kind == 'floorplan':
        field = _floorplan(nx, ny, n_tiles, rng)
    else:  # mixed
        tiles = _normalise(_floorplan(nx, ny, n_tiles, rng), 0.0)
        # Blur tile edges slightly: real power does not step discontinuously at a
        # block boundary, it is smeared by interconnect and local spreading.
        tiles = _smooth(tiles, 0.5)
        field = tiles + 0.35 * _normalise(_grf(nx, ny, corr_frac, rng), 0.0)

    return _normalise(field, idle_floor)


def map_to_rectangles(power_map: np.ndarray,
                      die_length_um: float,
                      die_width_um: float) -> Tuple[np.ndarray, float, float]:
    """Convert a power map into the rectangle grid a 3D-ICE floorplan needs."""
    nx, ny = power_map.shape
    return power_map, die_length_um / nx, die_width_um / ny
