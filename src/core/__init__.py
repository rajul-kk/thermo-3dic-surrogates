"""
Core data structures for thermal simulation geometries.
"""

from .geometry import Layer, PowerBlock, Geometry
from .material import MaterialLibrary, Material
from .geometry_builders import (
    build_geometry1,
    build_geometry2a,
    build_all_geometries,
    get_geometry_by_name
)

__all__ = [
    'Layer',
    'PowerBlock',
    'Geometry',
    'MaterialLibrary',
    'Material',
    'build_geometry1',
    'build_geometry2a',
    'build_all_geometries',
    'get_geometry_by_name'
]
