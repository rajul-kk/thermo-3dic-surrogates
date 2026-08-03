"""
Mesh generation utilities for thermal simulations.

Provides functions to generate structured meshes for thermal geometries:
- 3D Cartesian grids
- Coordinate arrays for exporting
- Layer index mapping
"""

import logging
import numpy as np
from typing import Tuple, Optional
from .geometry import Geometry

_log = logging.getLogger(__name__)

# Minimum z-points per layer, keyed by substring found in the layer name (lowercase).
# Ensures thermally critical thin layers (die, TIM) get enough resolution even when a
# thick layer (heat sink) would otherwise absorb almost all points proportionally.
_LAYER_MIN_POINTS = {
    'die':      8,   # steepest gradient; hotspot accuracy depends on this
    'tsv':      6,   # conductivity discontinuity at TSV boundary
    'tim':      4,   # thin, high-gradient interface layer
    'bonding':  3,
    'spreader': 3,
    'sink':     2,   # large isothermal bulk; 2 pts sufficient
}


def _min_points_for_layer(layer_name: str) -> int:
    """Return the minimum z-point count for a layer based on its name."""
    name = layer_name.lower()
    for keyword, pts in _LAYER_MIN_POINTS.items():
        if keyword in name:
            return pts
    return 2


def generate_cartesian_mesh(
    geometry: Geometry,
    uniform_z: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 3D Cartesian mesh for the geometry.

    Args:
        geometry: Geometry object
        uniform_z: If True, use uniform z-spacing. If False, adapt to layer boundaries.

    Returns:
        Tuple of (X, Y, Z) meshgrid arrays with shape (nx, ny, nz)
    """
    nx, ny, nz = geometry.mesh_resolution

    # Generate X and Y coordinates (uniform spacing)
    x = np.linspace(0, geometry.die_width, nx)
    y = np.linspace(0, geometry.die_length, ny)

    # Generate Z coordinates
    if uniform_z:
        z = np.linspace(0, geometry.get_total_height(), nz)
    else:
        # Adapt z-spacing to capture layer interfaces
        z = generate_adaptive_z_coords(geometry, nz)

    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    return X, Y, Z


def generate_adaptive_z_coords(geometry: Geometry, nz: int) -> np.ndarray:
    """
    Generate z-coordinates that align with layer boundaries.

    Distributes more points in thin layers (like TIM) and fewer in thick layers
    (like heat sink) while ensuring layer interfaces are captured.

    Args:
        geometry: Geometry object
        nz: Total number of z-points desired

    Returns:
        1D array of z-coordinates
    """
    total_height = geometry.get_total_height()

    if not geometry.layers:
        # Fallback to uniform spacing
        return np.linspace(0, total_height, nz)

    # Allocate points proportional to layer thickness, with per-layer minimums.
    # Die and TIM layers get a higher floor so critical thermal gradients are resolved
    # even when a thick heat sink would dominate a pure proportional allocation.
    layer_thicknesses = np.array([layer.thickness for layer in geometry.layers])
    layer_proportions = layer_thicknesses / total_height

    min_per_layer = np.array([_min_points_for_layer(l.name) for l in geometry.layers], dtype=int)
    available_points = max(nz - min_per_layer.sum(), 0)

    # Distribute remaining points proportionally
    points_per_layer = min_per_layer.copy()
    if available_points > 0:
        additional_points = (available_points * layer_proportions).astype(int)
        points_per_layer += additional_points

    # Adjust to exactly match nz
    while points_per_layer.sum() < nz:
        idx = np.argmax(layer_thicknesses)
        points_per_layer[idx] += 1

    while points_per_layer.sum() > nz:
        # Remove from layers that are above their minimum, preferring the thinnest first
        candidates = np.where(points_per_layer > min_per_layer)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmin(layer_thicknesses[candidates])]
        points_per_layer[idx] -= 1

    # Generate z-coordinates layer by layer
    z_coords = []
    for layer, n_points in zip(geometry.layers, points_per_layer):
        layer_z = np.linspace(layer.z_bottom, layer.z_top, n_points, endpoint=False)
        z_coords.extend(layer_z)

    z_array = np.array(z_coords)

    z_unique = np.unique(z_array)
    if len(z_unique) < len(z_array):
        _log.warning(
            "generate_adaptive_z_coords: np.unique removed %d duplicate z-coordinates "
            "(geometry '%s'). Mesh will have fewer z-levels than requested.",
            len(z_array) - len(z_unique), geometry.name
        )
    return z_unique


def meshgrid_to_coords(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray
) -> np.ndarray:
    """
    Convert meshgrid arrays to coordinate array.

    Args:
        X, Y, Z: Meshgrid arrays of shape (nx, ny, nz)

    Returns:
        Coordinate array of shape (N, 3) where N = nx*ny*nz
    """
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return coords


def generate_coords_and_indices(
    geometry: Geometry,
    uniform_z: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate coordinate array and corresponding layer indices.

    Args:
        geometry: Geometry object
        uniform_z: Whether to use uniform z-spacing

    Returns:
        Tuple of:
        - coords: (N, 3) array of (x, y, z) coordinates in μm
        - layer_indices: (N,) array of layer indices (0-based)
    """
    X, Y, Z = generate_cartesian_mesh(geometry, uniform_z=uniform_z)
    coords = meshgrid_to_coords(X, Y, Z)

    # Compute layer index for each point
    layer_indices = np.zeros(coords.shape[0], dtype=int)
    oob_count = 0
    for i, (x, y, z) in enumerate(coords):
        layer_idx = geometry.get_layer_index_at_z(z)
        if layer_idx < 0:
            oob_count += 1
            layer_idx = 0  # clamp floating-point noise at boundaries
        layer_indices[i] = layer_idx

    if oob_count > 10:
        raise ValueError(
            f"generate_coords_and_indices: {oob_count} points have z outside the layer "
            f"stack for geometry '{geometry.name}'. Expected ≤10 for floating-point noise."
        )
    if oob_count > 0:
        _log.warning(
            "generate_coords_and_indices: %d boundary point(s) clamped to layer 0 "
            "(floating-point noise at z-boundaries, geometry '%s').",
            oob_count, geometry.name
        )

    return coords, layer_indices


def _power_field_from_maps(
    coords: np.ndarray,
    geometry: Geometry,
    power_map_by_layer: dict,
) -> np.ndarray:
    """
    Sample per-cell power maps onto arbitrary coordinates.

    Maps are indexed (along-length, along-width) to match the 3D-ICE floorplan
    convention, while coords are (x=width, y=length) in the PINN convention --
    hence the axis swap when looking up a cell.
    """
    power_field = np.zeros(coords.shape[0])

    for layer in geometry.layers:
        if not layer.is_active:
            continue
        pmap = power_map_by_layer.get(layer.name)
        if pmap is None:
            continue
        pmap = np.asarray(pmap, dtype=np.float64)
        n_l, n_w = pmap.shape

        in_layer = (coords[:, 2] >= layer.z_bottom) & (coords[:, 2] < layer.z_top)
        if not in_layer.any():
            continue

        # Cell volume in m^3: W per cell -> W/m^3
        cell_vol_m3 = ((geometry.die_length / n_l) * (geometry.die_width / n_w)
                       * layer.thickness) * 1e-18

        ia = np.clip((coords[in_layer, 1] / geometry.die_length * n_l).astype(int), 0, n_l - 1)
        ib = np.clip((coords[in_layer, 0] / geometry.die_width * n_w).astype(int), 0, n_w - 1)
        power_field[in_layer] = pmap[ia, ib] / cell_vol_m3

    return power_field


def generate_power_density_field(
    coords: np.ndarray,
    geometry: Geometry,
    power_scenario: dict,
    power_map_by_layer: dict = None,
) -> np.ndarray:
    """
    Generate volumetric power density field for given coordinates.

    Args:
        coords: (N, 3) array of (x, y, z) coordinates in μm
        geometry: Geometry object
        power_scenario: Dictionary mapping block names to power densities (W/cm²)
        power_map_by_layer: Optional {layer_name: (n_l, n_w) array of WATTS per
            cell}. When given it REPLACES the block decomposition, because it is
            what the simulator actually used. Falling back to blocks here would
            train a model on a coarse approximation of the source that produced
            its own targets.

    Returns:
        (N,) array of volumetric power densities (W/m³)
    """
    if power_map_by_layer:
        return _power_field_from_maps(coords, geometry, power_map_by_layer)

    power_field = np.zeros(coords.shape[0])

    # Get die layers (layers with power dissipation)
    die_layers = geometry.get_die_layers()

    if not die_layers:
        return power_field  # No active layers

    for i, (x, y, z) in enumerate(coords):
        # Check if point is in a die layer
        layer = geometry.get_layer_at_z(z)
        if layer is None or not layer.is_active:
            continue

        # Find which power block this point belongs to.
        # Skip TSV-region blocks — they are passive conductors in 3D-ICE (zero
        # heat source); their thermal effect enters via the layer's effective k.
        block = geometry.get_power_block_at_xy(x, y)
        if block is None or block.is_tsv_region:
            continue

        # Get power density from scenario (W/cm²)
        power_density_wcm2 = power_scenario.get(block.name, 0.0)

        # Convert to volumetric power density (W/m³)
        # Power is dissipated throughout the layer thickness
        thickness_m = layer.thickness / 1e6  # μm to m
        power_density_wm3 = (power_density_wcm2 * 1e4) / thickness_m  # W/cm² to W/m³

        power_field[i] = power_density_wm3

    return power_field


def compute_cell_volumes(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray
) -> np.ndarray:
    """
    Compute cell volumes for finite volume discretization.

    Args:
        X, Y, Z: Meshgrid arrays of shape (nx, ny, nz)

    Returns:
        Array of cell volumes in μm³ with shape (nx-1, ny-1, nz-1)
    """
    # Compute cell dimensions
    dx = np.diff(X, axis=0)[:, :-1, :-1]
    dy = np.diff(Y, axis=1)[:-1, :, :-1]
    dz = np.diff(Z, axis=2)[:-1, :-1, :]

    # Cell volume
    volumes = dx * dy * dz

    return volumes


def interpolate_field_to_coords(
    field: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    target_coords: np.ndarray
) -> np.ndarray:
    """
    Interpolate a field defined on meshgrid to arbitrary coordinates.

    Args:
        field: Field values on meshgrid, shape (nx, ny, nz)
        X, Y, Z: Meshgrid coordinate arrays
        target_coords: Target coordinates, shape (N, 3)

    Returns:
        Interpolated field values at target_coords, shape (N,)
    """
    from scipy.interpolate import RegularGridInterpolator

    # Get unique coordinate values for each dimension
    x_unique = X[:, 0, 0]
    y_unique = Y[0, :, 0]
    z_unique = Z[0, 0, :]

    # Create interpolator
    interpolator = RegularGridInterpolator(
        (x_unique, y_unique, z_unique),
        field,
        bounds_error=False,
        fill_value=np.nan
    )

    # Interpolate
    interp_values = interpolator(target_coords)

    return interp_values


def create_uniform_grid_summary(geometry: Geometry) -> str:
    """
    Create a summary of the mesh for a geometry.

    Args:
        geometry: Geometry object

    Returns:
        Formatted string summary
    """
    nx, ny, nz = geometry.mesh_resolution
    total_points = nx * ny * nz

    dx = geometry.die_width / nx
    dy = geometry.die_length / ny
    dz_avg = geometry.get_total_height() / nz

    lines = [
        f"Mesh Summary for {geometry.name}:",
        f"  Resolution: {nx} × {ny} × {nz} = {total_points:,} points",
        f"  Cell size (avg): {dx:.2f} × {dy:.2f} × {dz_avg:.2f} μm",
        f"  Domain: [{0:.0f}, {geometry.die_width:.0f}] × "
        f"[{0:.0f}, {geometry.die_length:.0f}] × "
        f"[{0:.0f}, {geometry.get_total_height():.0f}] μm",
    ]

    return "\n".join(lines)
