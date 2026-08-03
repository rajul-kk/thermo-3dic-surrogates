"""
Per-cell power map generation.

Why this exists
---------------
Until now a scenario's power was described by 4-13 block scalars. That makes the
whole dataset lie on a low-dimensional, nearly-linear manifold: steady-state
conduction is linear in its sources, so with ~8 scalars in, closed-form ridge
regression reconstructs the temperature field at spatial R2 0.987 and
extrapolates at R2 > 0.94 (docs/report.md 9.3). No choice of parameter RANGES
fixes that -- the input dimensionality is the cause.

A per-cell power map changes the input from a handful of numbers into a
function. The physics stays linear, but the solution operator becomes a genuine
Green's function: for geometry1's ~100k nodes that is a ~1e10-entry object,
which a per-point linear regression cannot represent, while an FNO compresses it
to ~1e7 via spectral truncation. This is the regime neural operators exist for
(Neural Green's Operators, arXiv:2406.01857), and it is reachable without
abandoning one-FNO-per-geometry, because the geometry stays fixed.

Cost: essentially nothing. 3D-ICE solve time is set by the mesh, not the
floorplan -- measured 13.3 s for 4 blocks and 13.5 s for 10,000 on geometry1.

Map families
------------
grf         Gaussian random field, smoothed white noise. Correlation length is
            expressed as a fraction of die width so it is geometry-independent.
            Models diffuse activity with no floorplan structure.
floorplan   Rectangular tiles at random power levels, like a real core/cache/IO
            layout. Produces the sharp lateral gradients that make interface
            behaviour interesting.
mixed       floorplan tiles plus a GRF perturbation. Structured layout with
            within-block activity variation, closest to a real workload.

All maps are returned normalised to peak 1.0; absolute scale is applied later
from the TDP budget, so the existing power model is untouched.

Choosing the defaults
---------------------
The parameter that matters is not how complex ONE map looks, but how many
dimensions a COLLECTION of maps spans -- that is what a linear model has to fit.
Measured effective rank of N stacked maps (64x64, mean removed):

    family / corr / tiles        N=40    N=120   N=200
    grf   0.02                   37.4    107.0   169.1
    mixed 0.02 / 16 tiles        37.2    104.9   163.8
    floorplan / 16 tiles         37.8    109.1   174.4
    -- block scalars (old)        4.0      4.0     4.0

The maps span very nearly N-1 dimensions: every scenario adds a genuinely new
direction, so a linear fit gains no leverage from extra samples. Block scalars
span exactly 4 no matter how many scenarios are generated, which is why ridge
solved the old dataset.

An earlier default of corr_frac=0.15 with 4 tiles gave an effective rank of only
2-7 -- no better than the block scalars it replaced. Correlation length has to be
short relative to the die for the map to carry real spatial information.

The defaults below are also physically sensible: corr_frac 0.02 is ~200 um on a
10 mm die, the scale of a real hotspot, and 16 tiles is ~625 um, the scale of a
functional block.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Map family names accepted by `generate_power_map` and stored in scenario metadata.
POWER_MAP_KINDS = ('grf', 'floorplan', 'mixed')


def _smooth(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """
    Gaussian blur via separable FFT convolution.

    Uses FFT rather than scipy.ndimage so the module has no dependency beyond
    numpy, matching the rest of the pipeline.
    """
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
    """
    Rectangular tiles at independent power levels.

    Tile edges are placed at random cut positions rather than a uniform grid, so
    the model cannot key on a fixed block pitch.
    """
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
    """
    Build a normalised (nx, ny) power map with values in [idle_floor, 1].

    Args:
        nx, ny:      Map resolution, normally the die layer's lateral mesh.
        kind:        One of POWER_MAP_KINDS.
        seed:        Per-scenario seed; identical seeds reproduce identical maps.
        corr_frac:   GRF correlation length as a fraction of the larger die edge.
        n_tiles:     Tiles per axis for the floorplan family.
        idle_floor:  Minimum fraction of peak that quiet regions dissipate.
                     Real silicon leaks even when idle, and a hard zero would
                     make the source discontinuous at tile edges.

    Returns:
        (nx, ny) float64 array, max == 1.0.
    """
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
    """
    Convert a power map into the rectangle grid a 3D-ICE floorplan needs.

    Returns (levels, tile_length_um, tile_width_um) where levels[i, j] is the
    normalised power of the tile whose lower corner is (i*tile_length,
    j*tile_width). Index order matches 3D-ICE floorplan convention: the first
    axis runs along die LENGTH, the second along die WIDTH.
    """
    nx, ny = power_map.shape
    return power_map, die_length_um / nx, die_width_um / ny
