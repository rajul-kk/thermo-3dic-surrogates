"""
Scenario generator for benchmark thermal simulations.

Generates training and test scenarios with parameter sweeps:
- Power density variations
- Heat transfer coefficient (HTC) variations
- Ambient temperature variations
- Power distribution patterns
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
import yaml
from pathlib import Path
import numpy as np
from ..core.geometry import Geometry


@dataclass
class ScenarioParameters:
    """
    Parameters for a single thermal scenario.

    Attributes:
        name: Scenario identifier
        geometry_name: Associated geometry name
        scenario_type: 'train' or 'test'
        power_blocks: Dictionary mapping block names to power densities (W/cm²)
        htc: Heat transfer coefficient (W/m²·K)
        t_ambient: Ambient temperature (°C)
        pattern: Power distribution pattern name
        description: Human-readable description
    """
    name: str
    geometry_name: str
    scenario_type: str  # 'train' or 'test'
    power_blocks: Dict[str, float] = field(default_factory=dict)
    htc: float = 10000.0
    t_ambient: float = 25.0
    pattern: str = "uniform"
    description: str = ""
    rdl_joule_fraction: float = 0.05
    layer_k_overrides: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML export."""
        return {
            'name': self.name,
            'geometry': self.geometry_name,
            'type': self.scenario_type,
            'power_blocks': self.power_blocks,
            'htc': self.htc,
            't_ambient': self.t_ambient,
            'pattern': self.pattern,
            'description': self.description,
            'rdl_joule_fraction': self.rdl_joule_fraction,
            'layer_k_overrides': self.layer_k_overrides,
        }


class ScenarioGenerator:
    """
    Generate training and test scenarios for thermal benchmarks.

    Creates parameter sweeps across power density, HTC, ambient and spatial pattern.

    Operating regime (revised 2026-07-31)
    -------------------------------------
    The original ranges (0.1-20 W/cm2, HTC 1000-10000, ambient 25-65 C) produced a
    SPATIALLY DEGENERATE dataset: median within-scenario spatial dT was 1.10 K against
    a 68 K between-scenario range, and 88% of temperature variance was explained by the
    ambient input alone. Closed-form ridge regression reconstructed the field at
    spatial R2 = 0.999, so the benchmark could not discriminate between architectures.
    See docs/report.md 9.2.

    Three coupled changes fix the regime:

    1. POWER up ~15x. Self-heating scales with total power. The old peak of 20 W/cm2
       was far below the 100-300 W/cm2 of real CPU hotspots (references.md already
       flagged this as conservative); the consequence was that self-heating (0.1-36 K)
       never dominated the ambient sweep.

    2. HTC up. The fraction of the temperature drop that appears as SPATIAL structure
       (rather than a uniform offset) is R_cond / (R_cond + R_conv), and R_conv = 1/h.
       For the geometry1 stack R_cond ~ 7.9e-5 m2K/W, so h=1000 puts only ~7% of the
       drop inside the stack while h=50000 puts ~80% there. Low HTC was actively
       flattening the field.

    3. AMBIENT narrowed to 25-45 C. A 60 K ambient sweep swamped a 0.1-36 K
       self-heating signal; real datacentre inlet air is 15-45 C anyway.

    Together these target a junction rise of roughly 40-90 K with spatial gradients of
    tens of K, instead of ~1 K.
    """

    # Thermal design power per package class, in watts.
    #
    # Power is set from a TDP BUDGET rather than a per-geometry multiplier tuned to
    # hit a target temperature. The tuned-multiplier approach was circular -- it used
    # the output (peak junction temperature) to choose the input (power) -- and it
    # contradicted the measured physics: geometry3 has ~3x LOWER thermal resistance
    # than geometry2a (0.051 vs 0.164 K per summed W/cm2, measured from generated
    # fields), so it can absorb far more power, yet the tuned scales gave it the same
    # multiplier as geometry1.
    #
    # A TDP budget is a real design input, so each geometry's power is citable against
    # shipping hardware, and the temperature differences between package types EMERGE
    # from the geometry instead of being normalised away. That matters here because
    # PI-DeepONet trains across all geometries at once and should be learning exactly
    # that: stacking costs you thermally.
    #
    # Consequence, accepted deliberately: geometries no longer all peak near the same
    # temperature. A dense stacked part runs hotter than a large server die at the
    # same workload fraction, which is true of real hardware.
    TDP_BY_GEOMETRY_W = {
        'geometry1':  125.0,   # desktop-class single die, 10x10 mm
        'geometry2a':  30.0,   # mobile-class 3D stack, 8x8 mm
        'geometry2b':  30.0,
        'geometry2c':  30.0,
        'geometry3':  250.0,   # server CPU die, 25x25 mm
        'geometry4':  200.0,   # 2.5D chiplet assembly on interposer
        'geometry5':  400.0,   # CoWoS accelerator: compute chiplet + HBM stack
        'geometry6':  700.0,   # CoWoS accelerator with 6 HBM stacks
    }
    TDP_DEFAULT_W = 150.0

    # A pattern's `base_power` argument now selects a WORKLOAD FRACTION of TDP rather
    # than an absolute W/cm2: base_power = 10 means 100% of TDP, 20 means a 120%
    # boost excursion, and low values represent idle/light load. The pattern itself
    # still sets the SPATIAL distribution; only the total is normalised.
    WORKLOAD_PER_BASE_POWER = 0.1        # base_power 10 -> 1.0 x TDP
    MIN_WORKLOAD_FRACTION = 0.10         # ~idle
    MAX_WORKLOAD_FRACTION = 1.20         # short boost above TDP

    # Absolute ceiling on logic power density [W/cm2]. Silicon cannot dissipate
    # arbitrarily much per unit area regardless of the package budget; measured CPU
    # hotspots peak around 100-300 W/cm2 (references.md).
    #
    # This is what stops a concentrated pattern from becoming unphysical. Normalising
    # to TDP alone would let extreme_hotspot pour an entire 125 W budget into one
    # 0.09 cm2 block -- 1618 W/cm2, which is not silicon. With the cap, that scenario
    # instead represents one core at its density limit while the rest idle, and the
    # chip's TOTAL falls well below TDP. That is the correct physics: a part running
    # a single core flat-out does not draw full TDP.
    MAX_LOGIC_POWER_DENSITY_WCM2 = 300.0

    # Floor applied to any HTC drawn from the extra-scenario pools. Those pools
    # predate the regime change and contain values down to 500 W/m2K, which pair
    # with the scaled power to produce implausible temperatures.
    MIN_HTC = 2000.0
    MAX_HTC = 50000.0
    MAX_AMBIENT_C = 45.0

    # Minimum cooling capability per unit of peak power density [W/m2K per W/cm2].
    #
    # Power density and cooling are not independent in real hardware: nobody
    # air-cools a 300 W/cm2 hotspot, because it would throttle instantly. Sweeping
    # them as free variables manufactures design points that cannot exist, and that
    # is what produced 220-259 C junctions on geometry1/geometry4 -- the extra
    # scenario pools pair their highest power entries (150 W/cm2) with mid-range
    # HTC around 3500 W/m2K.
    #
    # Calibrated from the validated geometry1 operating point: 300 W/cm2 peak at
    # h = 50000 W/m2K yields a 112 C junction. 300 * 165 = 49500, just inside
    # MAX_HTC, so the hottest legitimate scenario remains reachable.
    #
    # This deliberately correlates HTC with power. It costs some independence in
    # the HTC sweep, but the excluded region is unphysical, so the benchmark loses
    # nothing real -- and HTC still varies independently wherever power leaves headroom.
    HTC_PER_WCM2 = 165.0

    # HTC levels and the real cooling hardware each one stands for. 3D-ICE's
    # `bottom heat sink` accepts only a scalar coefficient, so a cooling solution
    # can be represented only by its area-averaged effective HTC at the sink base:
    #
    #     2000  W/m2K  active air, tower cooler + fan
    #     5000  W/m2K  entry liquid / AIO cold plate
    #    10000  W/m2K  good liquid cold plate
    #    20000  W/m2K  high-performance cold plate
    #    35000  W/m2K  microchannel liquid
    #    50000  W/m2K  aggressive microchannel / two-phase
    #
    # KNOWN BIAS: this range is liquid-weighted. Passive and low-profile air
    # cooling (~500-1500 W/m2K) is deliberately excluded because at those levels
    # the convective film dominates the stack resistance and flattens the field
    # (see item 2 above) -- that choice serves benchmark discriminability, not
    # fidelity, and should be disclosed rather than presented as realism.
    # docs/assumptions.md 1.3 tracks the uniform-HTC limitation.

    # Training parameter ranges (pre-scale; effective power is value * POWER_SCALE)
    TRAIN_POWER_DENSITIES = [0.5, 1.0, 2.0, 4.0, 8.0]  # -> 7.5-120 W/cm²
    TRAIN_HTCS = [5000.0, 20000.0, 50000.0]  # W/m²·K
    TRAIN_AMBIENTS = [25.0, 35.0, 45.0]  # °C

    # Test parameter ranges (interpolation values)
    TEST_POWER_DENSITIES = [0.75, 3.0, 6.0]  # -> 11.25-90 W/cm²
    TEST_HTCS = [10000.0, 35000.0]  # W/m²·K
    TEST_AMBIENTS = [30.0, 40.0]  # °C

    def __init__(self):
        """Initialize scenario generator."""
        self.scenarios: List[ScenarioParameters] = []
        # TDP budget and block areas of the geometry currently being generated. Set by
        # every public generate_* entry point via _set_active_geometry, read by
        # _apply_pattern to normalise a relative power map onto the budget.
        self._active_tdp_w: float = self.TDP_DEFAULT_W
        self._active_block_area_cm2: Dict[str, float] = {}

    def _set_active_geometry(self, geometry: Geometry) -> None:
        """
        Select the TDP budget and block areas for `geometry`.

        Must precede any _apply_pattern call: the pattern needs block areas to
        convert a total-watt budget into per-block W/cm2.
        """
        self._active_tdp_w = self.TDP_BY_GEOMETRY_W.get(
            geometry.name, self.TDP_DEFAULT_W)
        # Block areas in cm2 (geometry dimensions are in µm; 1 cm2 = 1e8 µm2).
        self._active_block_area_cm2 = {
            b.name: (b.width * b.height) / 1e8
            for b in geometry.power_blocks if not b.is_tsv_region
        }

    def _enforce_cooling_adequacy(
            self, scenarios: List[ScenarioParameters]) -> List[ScenarioParameters]:
        """
        Raise each scenario's HTC to match its peak power density.

        Applied last, after power maps exist, so it sees the true peak including
        the per-geometry power scale. Mutates and returns the same list.
        """
        for s in scenarios:
            if not s.power_blocks:
                continue
            required = self.HTC_PER_WCM2 * max(s.power_blocks.values())
            if required > s.htc:
                s.htc = min(required, self.MAX_HTC)
        return scenarios

    def _clamp_bc(self, htc: float, ambient_c: float) -> tuple:
        """
        Bring a pool entry's boundary conditions into the current regime.

        The extra-scenario pools are literal tables written for the old power
        levels; unclamped they pair scaled power with HTCs as low as 500 W/m2K
        and ambients up to 75 C, which produced a 273 C junction on geometry2a.
        """
        return (min(max(htc, self.MIN_HTC), self.MAX_HTC),
                min(ambient_c, self.MAX_AMBIENT_C))

    def generate_all_scenarios(self, geometry: Geometry) -> List[ScenarioParameters]:
        """
        Generate all scenarios (train + test) for a geometry.

        Args:
            geometry: Geometry object

        Returns:
            List of ScenarioParameters (15 training + 5 test = 20 scenarios)
        """
        self._set_active_geometry(geometry)
        scenarios = []

        # Generate training scenarios
        train_scenarios = self.generate_training_scenarios(geometry)
        scenarios.extend(train_scenarios)

        # Generate test scenarios
        test_scenarios = self.generate_test_scenarios(geometry)
        scenarios.extend(test_scenarios)

        # Power density and cooling are physically coupled; enforce that last,
        # once power maps (and the per-geometry scale) are final.
        self._enforce_cooling_adequacy(scenarios)

        # Add to accumulated scenarios (don't replace)
        self.scenarios.extend(scenarios)
        return scenarios

    def generate_training_scenarios(self, geometry: Geometry) -> List[ScenarioParameters]:
        """
        Generate 15 training scenarios with diverse parameter combinations.

        Strategy:
        - Cover all power patterns (6 patterns including extreme_hotspot)
        - Vary power levels, HTC, and ambient temperature
        - Include 5 extreme scenarios for PINN robustness (Tier 1 improvement)
        - Ensure good coverage of parameter space
        """
        self._set_active_geometry(geometry)
        scenarios = []
        block_names = [b.name for b in geometry.power_blocks if not b.is_tsv_region]

        # Scenario 1: Uniform low power, medium HTC, room temp
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_001",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'uniform', 1.0),
            htc=20000.0,
            t_ambient=25.0,
            pattern='uniform',
            description='Uniform 1.0 W/cm² across all blocks'
        ))

        # Scenario 2: Hotspot corner, high HTC, room temp
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_002",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'hotspot', 5.0),
            htc=50000.0,
            t_ambient=25.0,
            pattern='hotspot',
            description='Single hotspot at 5.0 W/cm², others at 0.1 W/cm²'
        ))

        # Scenario 3: Checkerboard pattern, medium HTC, room temp
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_003",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'checkerboard', 2.0),
            htc=20000.0,
            t_ambient=25.0,
            pattern='checkerboard',
            description='Checkerboard: alternating 2.0 and 0.5 W/cm²'
        ))

        # Scenario 4: Gradient pattern, low HTC, elevated temp
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_004",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'gradient', 2.0),
            htc=5000.0,
            t_ambient=35.0,
            pattern='gradient',
            description='Linear gradient from 0.5 to 2.0 W/cm²'
        ))

        # Scenario 5: Dual hotspot, high HTC, high temp
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_005",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'dual_hotspot', 5.0),
            htc=50000.0,
            t_ambient=42.0,
            pattern='dual_hotspot',
            description='Two hotspots at 5.0 W/cm², others at 0.2 W/cm²'
        ))

        # Scenario 6: Very low power, low HTC, room temp
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_006",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'uniform', 0.1),
            htc=5000.0,
            t_ambient=25.0,
            pattern='uniform',
            description='Uniform low power 0.1 W/cm²'
        ))

        # Scenario 7: Hotspot with elevated ambient
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_007",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'hotspot', 2.0),
            htc=20000.0,
            t_ambient=35.0,
            pattern='hotspot',
            description='Single hotspot 2.0 W/cm² at elevated ambient'
        ))

        # Scenario 8: High uniform power, high HTC, high temp
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_008",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'uniform', 5.0),
            htc=50000.0,
            t_ambient=42.0,
            pattern='uniform',
            description='High uniform power 5.0 W/cm² at high temperature'
        ))

        # Scenario 9: Gradient with low HTC
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_009",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'gradient', 1.0),
            htc=5000.0,
            t_ambient=25.0,
            pattern='gradient',
            description='Gradient pattern with poor cooling (low HTC)'
        ))

        # Scenario 10: Medium power, medium params (baseline)
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_010",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'uniform', 0.5),
            htc=20000.0,
            t_ambient=35.0,
            pattern='uniform',
            description='Medium power baseline scenario'
        ))

        # ===== TIER 1 EXTREME SCENARIOS (5 additional) =====

        # Scenario 11: Very high power (stress test)
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_011",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'uniform', 10.0),
            htc=50000.0,
            t_ambient=25.0,
            pattern='uniform',
            description='Extreme power 10.0 W/cm² - stress test scenario'
        ))

        # Scenario 12: Very low HTC (poor cooling)
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_012",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'uniform', 2.0),
            htc=2000.0,
            t_ambient=25.0,
            pattern='uniform',
            description='Poor cooling (HTC=500 W/m²·K) - natural convection'
        ))

        # Scenario 13: High ambient + high power
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_013",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'uniform', 5.0),
            htc=20000.0,
            t_ambient=45.0,
            pattern='uniform',
            description='Elevated ambient 85°C - automotive/industrial conditions'
        ))

        # Scenario 14: Extreme hotspot
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_014",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'extreme_hotspot', 20.0),
            htc=50000.0,
            t_ambient=25.0,
            pattern='extreme_hotspot',
            description='Extreme hotspot 20.0 W/cm² in one block - maximum gradient'
        ))

        # Scenario 15: Mixed extreme conditions
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_train_015",
            geometry_name=geometry.name,
            scenario_type='train',
            power_blocks=self._apply_pattern(block_names, 'gradient', 3.0),
            htc=35000.0,
            t_ambient=38.0,
            pattern='gradient',
            description='Mixed extremes - gradient pattern with intermediate parameters'
        ))

        # 2.5D geometries: replace scenarios 13-15 with split-chiplet scenarios that
        # exercise lateral thermal coupling (the defining physical feature of 2p5d stacks).
        if geometry.geometry_type == '2p5d_stack':
            scenarios = scenarios[:12]   # keep scenarios 1-12

            scenarios.append(ScenarioParameters(
                name=f"{geometry.name}_train_013",
                geometry_name=geometry.name,
                scenario_type='train',
                power_blocks=self._apply_pattern(block_names, 'split_chiplet_a_hot', 5.0),
                htc=50000.0,
                t_ambient=25.0,
                pattern='split_chiplet_a_hot',
                description='Chiplet A at 5 W/cm2, chiplet B at 10% - hot-neighbour coupling'
            ))
            scenarios.append(ScenarioParameters(
                name=f"{geometry.name}_train_014",
                geometry_name=geometry.name,
                scenario_type='train',
                power_blocks=self._apply_pattern(block_names, 'split_chiplet_b_hot', 5.0),
                htc=20000.0,
                t_ambient=35.0,
                pattern='split_chiplet_b_hot',
                description='Chiplet B at 5 W/cm2, chiplet A at 10% - reverse hot-neighbour'
            ))
            scenarios.append(ScenarioParameters(
                name=f"{geometry.name}_train_015",
                geometry_name=geometry.name,
                scenario_type='train',
                power_blocks=self._apply_pattern(block_names, 'split_chiplet_a_hot', 2.0),
                htc=5000.0,
                t_ambient=42.0,
                pattern='split_chiplet_a_hot',
                description='Hot chiplet A with poor cooling - maximum thermal stress'
            ))

        return scenarios

    def generate_test_scenarios(self, geometry: Geometry) -> List[ScenarioParameters]:
        """
        Generate 5 test scenarios with interpolation values.

        Use parameter values NOT in training set to test PINN generalization.
        """
        self._set_active_geometry(geometry)
        scenarios = []
        block_names = [b.name for b in geometry.power_blocks if not b.is_tsv_region]

        # Test 1: Interpolated uniform power
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_test_001",
            geometry_name=geometry.name,
            scenario_type='test',
            power_blocks=self._apply_pattern(block_names, 'uniform', 1.5),
            htc=35000.0,
            t_ambient=30.0,
            pattern='uniform',
            description='Test interpolation: 1.5 W/cm², 7500 W/m²·K HTC, 35°C'
        ))

        # Test 2: Hotspot with interpolated values
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_test_002",
            geometry_name=geometry.name,
            scenario_type='test',
            power_blocks=self._apply_pattern(block_names, 'hotspot', 3.0),
            htc=10000.0,
            t_ambient=38.0,
            pattern='hotspot',
            description='Test hotspot: 3.0 W/cm² peak, 3000 W/m²·K HTC, 55°C'
        ))

        # Test 3: Checkerboard with interpolated power
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_test_003",
            geometry_name=geometry.name,
            scenario_type='test',
            power_blocks=self._apply_pattern(block_names, 'checkerboard', 1.5),
            htc=35000.0,
            t_ambient=30.0,
            pattern='checkerboard',
            description='Test checkerboard with interpolated parameters'
        ))

        # Test 4: Low power with interpolated ambient
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_test_004",
            geometry_name=geometry.name,
            scenario_type='test',
            power_blocks=self._apply_pattern(block_names, 'uniform', 0.3),
            htc=10000.0,
            t_ambient=30.0,
            pattern='uniform',
            description='Test low power: 0.3 W/cm² at 35°C'
        ))

        # Test 5: Gradient with interpolated HTC
        scenarios.append(ScenarioParameters(
            name=f"{geometry.name}_test_005",
            geometry_name=geometry.name,
            scenario_type='test',
            power_blocks=self._apply_pattern(block_names, 'gradient', 3.0),
            htc=35000.0,
            t_ambient=38.0,
            pattern='gradient',
            description='Test gradient with high HTC and elevated temperature'
        ))

        return scenarios

    def _apply_pattern(self, block_names: List[str], pattern: str,
                      base_power: float,
                      rdl_joule_fraction: float = 0.05) -> Dict[str, float]:
        """
        Apply power distribution pattern to blocks.

        Args:
            block_names: List of power block names
            pattern: Pattern type ('uniform', 'hotspot', 'checkerboard', 'gradient',
                                   'dual_hotspot', 'extreme_hotspot')
            base_power: Base power level to scale the pattern. Multiplied by
                POWER_SCALE here, so callers keep using the original relative
                numbers and every pattern shifts regime together.

        Returns:
            Dictionary mapping block names to power densities in W/cm²
        """
        n_blocks = len(block_names)
        power_map = {}

        # base_power now selects a workload fraction of TDP; the absolute W/cm2 is
        # fixed later by _normalise_to_tdp. Keeping the raw value here preserves each
        # pattern's internal ratios (hotspot vs background), which is all the code
        # below depends on.
        workload = min(max(base_power * self.WORKLOAD_PER_BASE_POWER,
                           self.MIN_WORKLOAD_FRACTION),
                       self.MAX_WORKLOAD_FRACTION)

        if pattern == 'uniform':
            # All blocks at same power
            for name in block_names:
                power_map[name] = base_power

        elif pattern == 'hotspot':
            # First block at base_power, others at 0.1 × base_power
            for i, name in enumerate(block_names):
                if i == 0:
                    power_map[name] = base_power
                else:
                    power_map[name] = base_power * 0.1

        elif pattern == 'checkerboard':
            # Alternating high (base_power) and low (0.5 × base_power)
            for i, name in enumerate(block_names):
                if i % 2 == 0:
                    power_map[name] = base_power
                else:
                    power_map[name] = base_power * 0.5

        elif pattern == 'gradient':
            # Linear gradient from 0.5 × base_power to base_power
            for i, name in enumerate(block_names):
                fraction = 0.5 + 0.5 * (i / max(1, n_blocks - 1))
                power_map[name] = base_power * fraction

        elif pattern == 'dual_hotspot':
            # First two blocks at base_power, others at 0.2 × base_power
            for i, name in enumerate(block_names):
                if i < 2:
                    power_map[name] = base_power
                else:
                    power_map[name] = base_power * 0.2

        elif pattern == 'extreme_hotspot':
            # Single extreme hotspot (for Tier 1 extreme scenarios)
            # First block at base_power, others at 0.01 × base_power (very low background)
            for i, name in enumerate(block_names):
                if i == 0:
                    power_map[name] = base_power
                else:
                    power_map[name] = base_power * 0.01

        elif pattern == 'split_chiplet_a_hot':
            # 2.5D geometries: chiplet A hot, all other chiplets (B or hbm1-N) at 10%
            for name in block_names:
                if name.startswith('chipA'):
                    power_map[name] = base_power
                else:
                    power_map[name] = base_power * 0.1

        elif pattern == 'split_chiplet_b_hot':
            # 2.5D geometries: chiplet B / HBM stacks hot, chiplet A at 10%
            # Works for both geometry4/5 (chipB_*) and geometry6 (hbm1-4_*)
            for name in block_names:
                if name.startswith('chipB') or name.startswith('hbm'):
                    power_map[name] = base_power
                else:
                    power_map[name] = base_power * 0.1

        elif pattern == 'random_smooth':
            # Spatially correlated random powers — each block gets an independent
            # random multiplier in [0.3, 2.0] × base_power.  Seed is derived from
            # block count so it's deterministic for a given geometry.
            rng = np.random.default_rng(seed=n_blocks * 7 + int(base_power * 100))
            multipliers = rng.uniform(0.3, 2.0, size=n_blocks)
            for i, name in enumerate(block_names):
                power_map[name] = max(0.05, base_power * multipliers[i])

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        # Memory-stack (HBM) dies are capped below compute-logic power density,
        # regardless of pattern. Real HBM3 stacks dissipate roughly 0.1-0.5
        # W/cm² under typical-to-heavy load (engineering estimate from published
        # per-stack power figures over the ~48mm² footprint used here, not a
        # specific cited source); compute logic in this dataset legitimately
        # sweeps up to 5-20 W/cm² (including Tier-1 extreme scenarios). Without
        # this cap, patterns like 'uniform' or 'split_chiplet_b_hot' assign
        # HBM dies the SAME power range as compute logic, producing scenarios
        # where HBM3 stacks exceed their ~95-105°C junction-temperature spec
        # (geometry6 recorded T up to 134°C) -- unrealistic for the "6x HBM3,
        # MI300X-like" claim in that geometry's docstring. 2.0 W/cm² ceiling
        # gives ~4-6x headroom above typical HBM load for legitimate worst-case
        # coverage without reaching logic-die-like power density.
        # Applies to: geometry6 hbm{n}_d1/d2, geometry5 chipB_d1_c*/chipB_d2_c*
        # (chipB is documented as "HBM-style" in that geometry too).
        # Raised from 2.0 alongside the POWER_SCALE regime change (2026-07-31).
        # HBM3E dies genuinely run at ~1-5 W/cm², far below logic, so the cap must
        # stay an absolute physical ceiling rather than scale with compute power --
        # but 2.0 against a 300 W/cm² logic hotspot was an unrealistic 150x ratio.
        # 8.0 keeps HBM well under its ~95-105°C junction spec while allowing
        # legitimate worst-case coverage.
        # Convert the relative map into absolute W/cm2 against the TDP budget. Done
        # BEFORE the HBM cap and RDL terms so those remain absolute physical limits
        # rather than fractions of a budget.
        power_map = self._normalise_to_tdp(power_map, workload)

        # Absolute density ceilings. Applied after TDP normalisation, so a
        # concentrated pattern ends up below TDP rather than at an impossible density.
        for name in list(power_map):
            power_map[name] = min(power_map[name], self.MAX_LOGIC_POWER_DENSITY_WCM2)

        _HBM_POWER_CAP_WCM2 = 8.0
        for name in block_names:
            is_hbm_die = name.startswith('hbm') or name.startswith('chipB_d1') or name.startswith('chipB_d2')
            if is_hbm_die and name in power_map:
                power_map[name] = min(power_map[name], _HBM_POWER_CAP_WCM2)

        # RDL routing layers self-heat at rdl_joule_fraction of the local logic power.
        # Varies 1–10% across scenarios to teach the PINN that RDL heating is a
        # free variable (pump-out degradation, routing density, current load).
        logic = [v for n, v in power_map.items() if 'rdl' not in n.lower()]
        ref = max(logic) if logic else 1.0
        for name in block_names:
            if 'rdl' in name.lower():
                power_map[name] = max(0.02, ref * rdl_joule_fraction)

        return power_map

    def _normalise_to_tdp(self, power_map: Dict[str, float],
                          workload: float) -> Dict[str, float]:
        """
        Rescale a relative power map so total dissipation equals workload x TDP.

        The pattern fixes the spatial distribution; this fixes the total. Blocks with
        no known area (or a degenerate map) are left untouched rather than silently
        divided by zero.
        """
        target_w = self._active_tdp_w * workload
        current_w = sum(p * self._active_block_area_cm2.get(n, 0.0)
                        for n, p in power_map.items())
        if current_w <= 0.0:
            return power_map
        factor = target_w / current_w
        return {n: p * factor for n, p in power_map.items()}

    # ---------------------------------------------------------------------------
    # Extra training scenarios (extending beyond the fixed 15)
    # ---------------------------------------------------------------------------

    # Ordered pool of (pattern, power_wcm2, htc, ambient_c) combos that fill
    # the gaps left by the original 15.  Take the first n_extra entries.
    _EXTRA_POOL = [
        ('random_smooth',    1.5,  2000,  30),
        ('random_smooth',    4.0,  8000,  50),
        ('uniform',          0.3,  2000,  30),
        ('uniform',          7.0,  8000,  25),
        ('hotspot',          3.0,  2000,  60),
        ('hotspot',          1.5,  7500,  40),
        ('checkerboard',     4.0,  3000,  35),
        ('checkerboard',     0.7,  6000,  50),
        ('gradient',         2.5,  4000,  45),
        ('gradient',         7.0,  1500,  35),
        ('dual_hotspot',     2.5,  4000,  55),
        ('dual_hotspot',     0.5, 15000,  25),
        ('random_smooth',    5.0,  3500,  70),
        ('random_smooth',    0.5,  6000,  35),
        ('uniform',          4.0,  6000,  55),
        ('hotspot',          7.0, 15000,  45),
        ('checkerboard',     1.5,  1500,  65),
        ('gradient',         3.5,  8000,  30),
        ('random_smooth',    2.0,  1500,  40),
        ('extreme_hotspot', 10.0, 15000,  25),
        ('random_smooth',    4.0,   500,  25),
        ('uniform',          1.5,   3500,  75),
        ('hotspot',          0.3,   4000,  25),
        ('dual_hotspot',     3.5,   2000,  65),
        ('checkerboard',    10.0,   3500,  55),
        # Tier 0 liquid-cooling extension
        ('uniform',          5.0,  50000,  25),
        ('hotspot',         10.0, 200000,  25),
        ('gradient',         7.0,  50000,  35),
    ]

    # For 2.5D geometries, replace some pool entries with split-chiplet patterns
    # that exercise lateral thermal coupling (the defining feature of these stacks).
    _EXTRA_POOL_2P5D = [
        # ── original 15 (indices 0–14) ────────────────────────────────────────
        ('split_chiplet_a_hot', 3.0, 3000,  30),
        ('split_chiplet_b_hot', 3.0, 8000,  50),
        ('split_chiplet_a_hot', 1.5, 2000,  60),
        ('split_chiplet_b_hot', 5.0,  500,  25),
        ('random_smooth',       1.5, 4000,  45),
        ('split_chiplet_a_hot', 7.0, 8000,  25),
        ('split_chiplet_b_hot', 2.0, 6000,  55),
        ('random_smooth',       4.0, 1500,  70),
        ('split_chiplet_a_hot', 0.5,15000,  25),
        ('split_chiplet_b_hot', 4.0, 2000,  65),
        ('uniform',             4.0, 3500,  35),
        ('checkerboard',        2.5, 6000,  50),
        ('gradient',            3.5, 4000,  40),
        ('split_chiplet_a_hot',10.0, 8000,  45),
        ('split_chiplet_b_hot', 0.3, 4000,  25),
        # ── extension (indices 15–34): TIM pump-out k-sweep + RDL fraction +
        #    unique parameter combos — all within the standard n_extra=35 window ─
        # TIM1 pump-out k-sweep: k=80→40→10→5 W/m·K (progressive degradation)
        ('uniform',             5.0,  10000, 25, 0.05, {'tim_top': 40.0}),
        ('split_chiplet_a_hot', 5.0,  10000, 25, 0.05, {'tim_top': 40.0}),
        ('hotspot',             5.0,   7500, 45, 0.05, {'tim_top': 10.0}),
        ('split_chiplet_b_hot', 3.0,   5000, 35, 0.05, {'tim_top': 10.0}),
        ('uniform',             7.0,   5000, 55, 0.05, {'tim_top':  5.0}),
        ('split_chiplet_a_hot', 7.0,   3000, 65, 0.05, {'tim_top':  5.0}),
        ('gradient',            3.0,   8000, 40, 0.05, {'tim_top': 40.0}),
        ('checkerboard',        4.0,   6000, 35, 0.05, {'tim_top': 10.0}),
        # RDL Joule fraction sweep (1%–10%)
        ('uniform',             3.0,   5000, 35, 0.01, {}),
        ('split_chiplet_b_hot', 4.0,   8000, 45, 0.01, {}),
        ('hotspot',             5.0,  10000, 25, 0.10, {}),
        ('split_chiplet_a_hot', 3.0,   5000, 55, 0.10, {}),
        ('gradient',            2.0,   7000, 30, 0.03, {}),
        ('uniform',             6.0,   4000, 50, 0.08, {}),
        # additional unique combos (fill slots 29–34)
        ('split_chiplet_a_hot', 2.0, 5000,  45),
        ('split_chiplet_b_hot', 1.0, 3000,  55),
        ('random_smooth',       3.0, 7000,  25),
        ('split_chiplet_b_hot', 7.0,10000,  30),
        ('random_smooth',       5.0, 8000,  55),
        ('split_chiplet_a_hot', 4.0,10000,  30),
        # ── overflow / liquid-cooling (indices 35+, used if n_extra > 35) ────
        ('uniform',             5.0,  50000, 25),
        ('split_chiplet_a_hot', 8.0, 100000, 25),
        ('split_chiplet_b_hot', 5.0,  50000, 35),
        ('hotspot',            10.0, 200000, 25),
        ('split_chiplet_a_hot', 3.0, 200000, 45),
        ('uniform',             1.5, 1000,  35),
        ('split_chiplet_a_hot', 0.3, 2000,  75),
        ('uniform',             3.0, 1500,  45),
        ('split_chiplet_a_hot', 5.0, 3000,  65),
        ('split_chiplet_b_hot', 2.5, 7500,  25),
        ('split_chiplet_a_hot', 1.0, 6000,  35),
        ('split_chiplet_b_hot', 3.5, 4000,  70),
        ('uniform',             0.5,10000,  30),
        ('split_chiplet_a_hot', 6.0,  2000,  55),
        ('split_chiplet_b_hot', 1.5,  1000,  45),
        ('random_smooth',       2.5, 12000,  40),
    ]

    def generate_extra_training_scenarios(
        self,
        geometry: Geometry,
        n_extra: int,
        start_index: int = 16,
        pool_start_idx: int = 0,
    ) -> List[ScenarioParameters]:
        """
        Generate n_extra additional training scenarios starting at start_index.

        Uses a fixed parameter pool that fills gaps in the original 15 scenarios:
        denser HTC coverage (500–15000), more power levels, random_smooth pattern,
        and extra split-chiplet scenarios for 2.5D geometries.

        Args:
            geometry:      Target geometry.
            n_extra:       Number of extra scenarios to generate.
            start_index:   File index for the first scenario (e.g. 16 if 15 exist).
            pool_start_idx: Index into the pool to start drawing from (default 0).
                           Use this to avoid duplicating scenarios from a prior call.

        Returns:
            List of n_extra ScenarioParameters.
        """
        self._set_active_geometry(geometry)
        block_names = [b.name for b in geometry.power_blocks if not b.is_tsv_region]
        pool = (self._EXTRA_POOL_2P5D
                if geometry.geometry_type == '2p5d_stack'
                else self._EXTRA_POOL)

        pool_slice = pool[pool_start_idx: pool_start_idx + n_extra]
        if len(pool_slice) < n_extra:
            raise ValueError(
                f"Requested {n_extra} extra scenarios starting at pool index "
                f"{pool_start_idx}, but only {len(pool_slice)} entries available "
                f"(pool size {len(pool)}). Add more entries to "
                "_EXTRA_POOL/_EXTRA_POOL_2P5D."
            )

        scenarios = []
        for idx, entry in enumerate(pool_slice):
            pattern, power, htc, ambient = entry[:4]
            htc, ambient = self._clamp_bc(htc, ambient)
            rdl_frac   = entry[4] if len(entry) > 4 else 0.05
            k_overrides = entry[5] if len(entry) > 5 else {}
            scenario_idx = start_index + idx
            name = f"{geometry.name}_train_{scenario_idx:03d}"
            scenarios.append(ScenarioParameters(
                name=name,
                geometry_name=geometry.name,
                scenario_type='train',
                power_blocks=self._apply_pattern(block_names, pattern, power, rdl_frac),
                htc=float(htc),
                t_ambient=float(ambient),
                pattern=pattern,
                rdl_joule_fraction=rdl_frac,
                layer_k_overrides=dict(k_overrides),
                description=(
                    f"Extra scenario {scenario_idx}: {pattern} "
                    f"{power} W/cm2, HTC={htc}, T_amb={ambient}C "
                    f"(pool[{pool_start_idx + idx}])"
                ),
            ))

        self._enforce_cooling_adequacy(scenarios)
        self.scenarios.extend(scenarios)
        return scenarios

    def save_to_yaml(self, output_dir: Path, geometry_name: str) -> None:
        """
        Save scenarios to YAML file.

        Args:
            output_dir: Output directory for YAML files
            geometry_name: Geometry identifier
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter scenarios for this geometry
        geom_scenarios = [s for s in self.scenarios if s.geometry_name == geometry_name]

        # Convert to dictionary format
        data = {
            'geometry': geometry_name,
            'num_scenarios': len(geom_scenarios),
            'num_train': len([s for s in geom_scenarios if s.scenario_type == 'train']),
            'num_test': len([s for s in geom_scenarios if s.scenario_type == 'test']),
            'scenarios': [s.to_dict() for s in geom_scenarios]
        }

        # Write to YAML
        output_file = output_dir / f"{geometry_name}_scenarios.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Saved {len(geom_scenarios)} scenarios to {output_file}")

    def load_from_yaml(self, yaml_file: Path) -> List[ScenarioParameters]:
        """
        Load scenarios from YAML file.

        Args:
            yaml_file: Path to YAML file

        Returns:
            List of ScenarioParameters
        """
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        scenarios = []
        for s_data in data['scenarios']:
            scenario = ScenarioParameters(
                name=s_data['name'],
                geometry_name=s_data['geometry'],
                scenario_type=s_data['type'],
                power_blocks=s_data['power_blocks'],
                htc=s_data['htc'],
                t_ambient=s_data['t_ambient'],
                pattern=s_data['pattern'],
                description=s_data.get('description', '')
            )
            scenarios.append(scenario)

        return scenarios

    def summary(self) -> str:
        """Generate summary of all scenarios."""
        if not self.scenarios:
            return "No scenarios generated yet."

        lines = ["Scenario Summary:", "=" * 80]

        # Group by geometry
        geometries = set(s.geometry_name for s in self.scenarios)

        for geom in sorted(geometries):
            geom_scenarios = [s for s in self.scenarios if s.geometry_name == geom]
            train = [s for s in geom_scenarios if s.scenario_type == 'train']
            test = [s for s in geom_scenarios if s.scenario_type == 'test']

            lines.append(f"\n{geom}:")
            lines.append(f"  Training scenarios: {len(train)}")
            lines.append(f"  Test scenarios: {len(test)}")
            lines.append(f"  Total: {len(geom_scenarios)}")

        lines.append(f"\nOverall total: {len(self.scenarios)} scenarios")

        return "\n".join(lines)
