"""
Visualization utilities for thermal simulations.

Provides functions to create plots of:
- Temperature fields (2D cross-sections, profiles)
- Power distribution maps
- Scenario comparisons
"""

from .visualization import (
    plot_temperature_field_2d,
    plot_power_map_2d,
    plot_temperature_profile,
    plot_scenario_comparison,
    create_visualization_summary
)

__all__ = [
    'plot_temperature_field_2d',
    'plot_power_map_2d',
    'plot_temperature_profile',
    'plot_scenario_comparison',
    'create_visualization_summary'
]
