"""
Visualization utilities for thermal simulation data.

Provides functions to create publication-quality plots of:
- Temperature fields (2D slices, 3D volumes)
- Power distribution maps
- Layer cross-sections
- Scenario comparisons
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from ..core.geometry import Geometry


def plot_temperature_field_2d(coords: np.ndarray,
                              temperatures: np.ndarray,
                              layer_indices: np.ndarray,
                              geometry: Geometry,
                              plane: str = 'z_mid',
                              title: str = '',
                              output_file: Optional[Path] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create 2D temperature field visualization (cross-section).

    Args:
        coords: (N, 3) coordinate array in μm
        temperatures: (N,) temperature array in K
        layer_indices: (N,) layer index array
        geometry: Geometry object
        plane: Cross-section plane ('z_mid', 'z_top', 'y_mid', 'x_mid')
        title: Plot title
        output_file: Optional file path to save plot
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Select plane
    if plane == 'z_mid':
        z_target = geometry.get_total_height() / 2
        mask = np.abs(coords[:, 2] - z_target) < 100  # Within 100 μm of mid-plane
        x, y, T = coords[mask, 0], coords[mask, 1], temperatures[mask]
        xlabel, ylabel = 'X (μm)', 'Y (μm)'
    elif plane == 'z_top':
        z_target = geometry.get_total_height()
        mask = np.abs(coords[:, 2] - z_target) < 50
        x, y, T = coords[mask, 0], coords[mask, 1], temperatures[mask]
        xlabel, ylabel = 'X (μm)', 'Y (μm)'
    elif plane == 'y_mid':
        y_target = geometry.die_length / 2
        mask = np.abs(coords[:, 1] - y_target) < 100
        x, z, T = coords[mask, 0], coords[mask, 2], temperatures[mask]
        xlabel, ylabel = 'X (μm)', 'Z (μm)'
    elif plane == 'x_mid':
        x_target = geometry.die_width / 2
        mask = np.abs(coords[:, 0] - x_target) < 100
        y, z, T = coords[mask, 1], coords[mask, 2], temperatures[mask]
        xlabel, ylabel = 'Y (μm)', 'Z (μm)'
    else:
        raise ValueError(f"Unknown plane: {plane}")

    if len(T) == 0:
        print(f"Warning: No points found in plane {plane}")
        return fig

    # Create scatter plot
    scatter = ax.scatter(x, y if plane in ['z_mid', 'z_top'] else z,
                        c=T, cmap='RdYlBu_r', s=50, alpha=0.6, edgecolors='none')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
    T_min_c = np.min(T) - 273.15
    T_max_c = np.max(T) - 273.15
    cbar.set_label(f'Temperature (°C: {T_min_c:.1f} to {T_max_c:.1f})')

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not title:
        title = f'Temperature Distribution ({plane})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Set aspect ratio
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save if requested
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def plot_power_map_2d(coords: np.ndarray,
                      power_density: np.ndarray,
                      geometry: Geometry,
                      output_file: Optional[Path] = None,
                      figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create 2D power distribution map (die-level view).

    Args:
        coords: (N, 3) coordinate array in μm
        power_density: (N,) power density array in W/m³
        geometry: Geometry object
        output_file: Optional file path to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get die layer (top active layer)
    die_layers = geometry.get_die_layers()
    if not die_layers:
        print("No die layers found")
        return fig

    die_layer = die_layers[-1]  # Get last (topmost) die layer

    # Filter points in die layer
    mask = (coords[:, 2] >= die_layer.z_bottom) & (coords[:, 2] <= die_layer.z_top)
    if not np.any(mask):
        print("No points in die layer")
        return fig

    x = coords[mask, 0]
    y = coords[mask, 1]
    p = power_density[mask]

    # Create plot with logarithmic scale for power
    p_nonzero = p[p > 0]
    if len(p_nonzero) == 0:
        print("No non-zero power")
        return fig

    scatter = ax.scatter(x, y, c=p, cmap='hot', s=50, alpha=0.7,
                        norm=LogNorm(vmin=p_nonzero.min(), vmax=p_nonzero.max()),
                        edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax, label='Power Density (W/m³)')

    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(f'Power Distribution - {die_layer.name}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw power block boundaries
    for block in geometry.power_blocks:
        if not block.is_tsv_region:  # Draw only core blocks
            rect = patches.Rectangle((block.x, block.y), block.width, block.height,
                                    linewidth=1.5, edgecolor='blue', facecolor='none',
                                    linestyle='--', label=block.name if block == geometry.power_blocks[0] else '')
            ax.add_patch(rect)

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def plot_temperature_profile(coords: np.ndarray,
                            temperatures: np.ndarray,
                            geometry: Geometry,
                            direction: str = 'z',
                            output_file: Optional[Path] = None,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create 1D temperature profile along a direction.

    Args:
        coords: (N, 3) coordinate array in μm
        temperatures: (N,) temperature array in K
        geometry: Geometry object
        direction: Direction for profile ('x', 'y', or 'z')
        output_file: Optional file path to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if direction == 'z':
        # Profile along vertical direction
        # Take average temperature at each z-coordinate
        z_unique = np.unique(coords[:, 2])
        T_avg = []

        for z in z_unique:
            mask = np.abs(coords[:, 2] - z) < 50  # Within 50 μm
            if np.any(mask):
                T_avg.append(np.mean(temperatures[mask]))
            else:
                T_avg.append(np.nan)

        ax.plot(z_unique / 1000, np.array(T_avg) - 273.15, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Depth Z (mm)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Vertical Temperature Profile')

        # Add layer labels
        for i, layer in enumerate(geometry.layers):
            z_mid = (layer.z_bottom + layer.z_top) / 2 / 1000
            ax.axvline(layer.z_top / 1000, color='gray', linestyle='--', alpha=0.5)
            ax.text(z_mid, ax.get_ylim()[0], layer.name, rotation=90, ha='right', fontsize=9)

    else:
        ax.text(0.5, 0.5, f"Direction '{direction}' not yet implemented",
                ha='center', va='center', transform=ax.transAxes)
        if output_file:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
        return fig

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def plot_scenario_comparison(scenarios_data: dict,
                            output_file: Optional[Path] = None,
                            figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create comparison plot of multiple scenarios.

    Args:
        scenarios_data: Dict mapping scenario names to temperature arrays
        output_file: Optional file path to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax_idx, (scenario_name, temps) in enumerate(list(scenarios_data.items())[:4]):
        ax = axes[ax_idx]

        # Create histogram
        ax.hist(temps - 273.15, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(temps) - 273.15, color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(np.max(temps) - 273.15, color='orange', linestyle='--', linewidth=2, label='Max')

        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Frequency')
        ax.set_title(scenario_name, fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def create_visualization_summary(coords: np.ndarray,
                                temperatures: np.ndarray,
                                power_density: np.ndarray,
                                geometry: Geometry,
                                scenario_name: str,
                                output_dir: Path) -> None:
    """
    Create a comprehensive set of visualization plots.

    Args:
        coords: (N, 3) coordinate array
        temperatures: (N,) temperature array
        power_density: (N,) power density array
        geometry: Geometry object
        scenario_name: Scenario identifier
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temperature field (z-mid)
    plot_temperature_field_2d(coords, temperatures, np.zeros_like(temperatures, dtype=int),
                             geometry, plane='z_mid',
                             title=f'{scenario_name} - Temperature (Z-mid)',
                             output_file=output_dir / f"{scenario_name}_temp_z_mid.png")

    # Power map
    plot_power_map_2d(coords, power_density, geometry,
                     output_file=output_dir / f"{scenario_name}_power_map.png")

    # Vertical profile
    plot_temperature_profile(coords, temperatures, geometry, direction='z',
                            output_file=output_dir / f"{scenario_name}_profile_z.png")

    print(f"Visualizations saved to {output_dir}")
