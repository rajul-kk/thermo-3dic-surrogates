"""
Geometry data structures and builders for thermal simulations.

Defines the core classes for representing thermal stack geometries:
- Layer: Single material layer in the thermal stack
- PowerBlock: Power dissipation region
- Geometry: Complete geometry specification with layers and power map
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Layer:
    """
    Represents a single layer in the thermal stack.

    Attributes:
        name: Layer identifier (e.g., 'die', 'tim_top', 'spreader')
        material: Material name (references MaterialLibrary)
        thickness: Layer thickness in micrometers (μm)
        k_thermal: Thermal conductivity in W/m·K
        volumetric_heat_capacity: Volumetric heat capacity in J/m³·K
        z_bottom: Bottom z-coordinate in μm (set during stack assembly)
        z_top: Top z-coordinate in μm (set during stack assembly)
        is_active: Whether this layer has power dissipation
    """
    name: str
    material: str
    thickness: float  # μm
    k_thermal: float  # W/m·K
    volumetric_heat_capacity: float  # J/m³·K
    z_bottom: float = 0.0  # Set during stack assembly
    z_top: float = 0.0  # Set during stack assembly
    is_active: bool = False  # True for die layers with power dissipation

    def __post_init__(self):
        """Validate layer parameters."""
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.k_thermal <= 0:
            raise ValueError(f"Thermal conductivity must be positive, got {self.k_thermal}")
        if self.volumetric_heat_capacity <= 0:
            raise ValueError(f"Heat capacity must be positive, got {self.volumetric_heat_capacity}")


@dataclass
class PowerBlock:
    """
    Represents a power dissipation region in a die layer.

    Attributes:
        name: Block identifier
        x: Bottom-left x-coordinate in μm
        y: Bottom-left y-coordinate in μm
        width: Block width in μm
        height: Block height in μm
        power_density: Power density in W/cm² (set by scenario)
        is_tsv_region: True if this block represents a TSV array region
        die_index: Die index for multi-die stacks (0-based)
        layer_name: Name of the associated die layer
    """
    name: str
    x: float  # μm
    y: float  # μm
    width: float  # μm
    height: float  # μm
    power_density: float = 0.0  # W/cm² (updated per scenario)
    is_tsv_region: bool = False  # Special handling for TSV regions
    die_index: int = 0  # For multi-die stacks
    layer_name: Optional[str] = None  # Associated die layer

    def __post_init__(self):
        """Validate block parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Block dimensions must be positive: {self.width}x{self.height}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Block position must be non-negative: ({self.x}, {self.y})")

    @property
    def area_cm2(self) -> float:
        """Return block area in cm²."""
        return (self.width * self.height) / 1e8  # μm² to cm²

    @property
    def area_m2(self) -> float:
        """Return block area in m²."""
        return (self.width * self.height) / 1e12  # μm² to m²

    def power_watts(self, power_density_wcm2: Optional[float] = None) -> float:
        """
        Calculate total power in Watts.

        Args:
            power_density_wcm2: Override default power density (W/cm²)

        Returns:
            Total power in Watts
        """
        pd = power_density_wcm2 if power_density_wcm2 is not None else self.power_density
        return pd * self.area_cm2

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside this block."""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)


@dataclass
class DiePrint:
    """
    Footprint of a single chiplet on a shared interposer (2p5d_stack geometries).

    Defines which (x, y) region of the die_layer_name layer contains Si die
    material vs underfill/gap material.
    """
    name: str
    x: float           # µm from left edge of interposer
    y: float           # µm from bottom edge of interposer
    width: float       # µm
    height: float      # µm
    die_layer_name: str  # layer whose lateral material depends on this footprint

    def contains_point(self, x: float, y: float) -> bool:
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)


@dataclass
class Geometry:
    """
    Complete geometry specification for thermal simulation.

    Attributes:
        name: Geometry identifier (e.g., 'geometry1', 'geometry2a')
        geometry_type: Type identifier ('2d_stack' or '3d_stack')
        layers: List of Layer objects from bottom to top
        power_blocks: List of PowerBlock objects
        die_width: Die width in μm
        die_length: Die length in μm
        mesh_resolution: Grid resolution (nx, ny, nz)
        tsv_density: TSV area fraction (0.0-1.0), 0 for no TSVs
    """
    name: str
    geometry_type: str  # "2d_stack" or "3d_stack"
    layers: List[Layer] = field(default_factory=list)
    power_blocks: List[PowerBlock] = field(default_factory=list)
    die_width: float = 10000.0  # μm
    die_length: float = 10000.0  # μm
    mesh_resolution: Tuple[int, int, int] = (100, 100, 50)
    tsv_density: float = 0.0  # TSV area fraction (0.0 for no TSVs)
    die_footprints: List[DiePrint] = field(default_factory=list)
    underfill_k: float = 0.7  # W/m·K — underfill/gap conductivity for 2p5d_stack

    def __post_init__(self):
        """Validate geometry and assemble stack."""
        if self.geometry_type not in ['2d_stack', '3d_stack', '2p5d_stack']:
            raise ValueError(f"Invalid geometry_type: {self.geometry_type}")
        if self.tsv_density < 0.0 or self.tsv_density > 1.0:
            raise ValueError(f"TSV density must be between 0 and 1, got {self.tsv_density}")

        # Assemble stack if layers are provided
        if self.layers:
            self.assemble_stack()

    def assemble_stack(self) -> None:
        """
        Calculate z-coordinates for each layer in the stack.
        Layers are stacked from bottom (z=0) to top.
        """
        z_current = 0.0
        for layer in self.layers:
            layer.z_bottom = z_current
            layer.z_top = z_current + layer.thickness
            z_current = layer.z_top

    def get_total_height(self) -> float:
        """Return total stack height in μm."""
        if not self.layers:
            return 0.0
        return self.layers[-1].z_top

    def get_layer_at_z(self, z: float) -> Optional[Layer]:
        """
        Find the layer containing the given z-coordinate.

        Uses half-open intervals [z_bottom, z_top) for interior layers.
        The top surface of the last layer (z == total height) is included
        so that convective BC points are correctly assigned.

        Args:
            z: Z-coordinate in μm

        Returns:
            Layer object or None if not found
        """
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i < n - 1:
                if layer.z_bottom <= z < layer.z_top:
                    return layer
            else:
                if layer.z_bottom <= z <= layer.z_top:
                    return layer
        return None

    def get_layer_index_at_z(self, z: float) -> int:
        """
        Get the index of the layer containing z-coordinate.

        Uses half-open intervals [z_bottom, z_top) for interior layers.
        The top surface of the last layer (z == total height) is included.

        Args:
            z: Z-coordinate in μm

        Returns:
            Layer index (0-based) or -1 if not found
        """
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i < n - 1:
                if layer.z_bottom <= z < layer.z_top:
                    return i
            else:
                if layer.z_bottom <= z <= layer.z_top:
                    return i
        return -1

    def get_power_block_at_xy(self, x: float, y: float) -> Optional[PowerBlock]:
        """
        Find the power block containing point (x, y).

        Args:
            x, y: Coordinates in μm

        Returns:
            PowerBlock object or None if point is not in any block
        """
        for block in self.power_blocks:
            if block.contains_point(x, y):
                return block
        return None

    def validate(self) -> None:
        """
        Raise ValueError if the geometry has any structural inconsistency.

        Checks (fatal):
          1. Layer z-ranges are contiguous (no gaps or overlaps)
          2. No duplicate layer names
          3. PowerBlock.layer_name references exist in the layer stack
          4. Power blocks lie within the die footprint

        Checks (warning only):
          5. Power block pairs that overlap by more than 5% of the smaller block's area
        """
        import warnings

        errors = []

        # 1 — z-continuity
        for i in range(1, len(self.layers)):
            prev, curr = self.layers[i - 1], self.layers[i]
            if abs(curr.z_bottom - prev.z_top) > 1e-6:
                errors.append(
                    f"Layer gap/overlap between '{prev.name}' (z_top={prev.z_top:.3f}) "
                    f"and '{curr.name}' (z_bottom={curr.z_bottom:.3f})"
                )

        # 2 — duplicate layer names
        layer_names = [l.name for l in self.layers]
        dupes = sorted({n for n in layer_names if layer_names.count(n) > 1})
        if dupes:
            errors.append(f"Duplicate layer names: {dupes}")

        # 3 — power block layer references
        valid_names = set(layer_names)
        for block in self.power_blocks:
            if block.layer_name and block.layer_name not in valid_names:
                errors.append(
                    f"PowerBlock '{block.name}' references unknown layer '{block.layer_name}'"
                )

        # 4 — power blocks within die footprint
        for block in self.power_blocks:
            if block.x + block.width > self.die_width + 1e-6:
                errors.append(
                    f"PowerBlock '{block.name}' x-extent {block.x + block.width:.0f} µm "
                    f"exceeds die_width {self.die_width:.0f} µm"
                )
            if block.y + block.height > self.die_length + 1e-6:
                errors.append(
                    f"PowerBlock '{block.name}' y-extent {block.y + block.height:.0f} µm "
                    f"exceeds die_length {self.die_length:.0f} µm"
                )

        if errors:
            raise ValueError(
                f"Geometry '{self.name}' validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        # 5 — overlap warnings within the same layer (non-fatal)
        # Blocks on different layers (die_zone_1 vs die_zone_2, die1_active vs die2_active)
        # legitimately share the same x/y footprint (3D stacks, 2.5D multi-die).
        # Only warn when two blocks reference the exact same layer_name.
        blocks = self.power_blocks
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                a, b = blocks[i], blocks[j]
                if a.layer_name != b.layer_name:
                    continue
                ox = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
                oy = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
                overlap_area = ox * oy
                if overlap_area > 0:
                    min_block_area = min(a.width * a.height, b.width * b.height)
                    frac = overlap_area / min_block_area
                    if frac > 0.05:
                        warnings.warn(
                            f"Geometry '{self.name}': PowerBlocks '{a.name}' and "
                            f"'{b.name}' (both die_index={a.die_index}) overlap by "
                            f"{overlap_area / 1e6:.3f} mm² "
                            f"({frac * 100:.1f}% of smaller block)",
                            stacklevel=3,
                        )

    def get_die_layers(self) -> List[Layer]:
        """Return list of active die layers (layers with power dissipation)."""
        return [layer for layer in self.layers if layer.is_active]

    def get_num_dies(self) -> int:
        """Return number of dies in the stack."""
        return len(self.get_die_layers())

    def get_tsv_layers(self) -> List[Layer]:
        """Return list of TSV layers (enhanced conductivity regions)."""
        return [layer for layer in self.layers if 'tsv' in layer.name.lower()]

    def summary(self) -> str:
        """Return human-readable geometry summary."""
        lines = [
            f"Geometry: {self.name}",
            f"Type: {self.geometry_type}",
            f"Die size: {self.die_width/1000:.1f}mm x {self.die_length/1000:.1f}mm",
            f"Total height: {self.get_total_height()/1000:.1f}mm",
            f"Mesh resolution: {self.mesh_resolution[0]}x{self.mesh_resolution[1]}x{self.mesh_resolution[2]}",
            f"Number of layers: {len(self.layers)}",
            f"Number of dies: {self.get_num_dies()}",
            f"TSV density: {self.tsv_density*100:.1f}%" if self.tsv_density > 0 else "No TSVs",
            f"Power blocks: {len(self.power_blocks)}",
            "",
            "Layer stack (bottom to top):"
        ]

        for i, layer in enumerate(self.layers):
            lines.append(
                f"  {i}: {layer.name:20s} {layer.thickness:8.1f}um  "
                f"k={layer.k_thermal:6.1f}W/m*K  "
                f"z=[{layer.z_bottom:8.1f}, {layer.z_top:8.1f}]um"
            )

        if self.power_blocks:
            lines.append("\nPower blocks:")
            for block in self.power_blocks:
                lines.append(
                    f"  {block.name:20s} {block.width/1000:.1f}x{block.height/1000:.1f}mm  "
                    f"at ({block.x/1000:.1f}, {block.y/1000:.1f})mm"
                )

        return "\n".join(lines)
