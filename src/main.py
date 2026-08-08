"""
Main orchestration script for 3D-IC Thermal PINN Benchmark System.

This script:
1. Loads geometry definitions (Geometry 1, Geometry 2a/2b/2c with varying TSV density)
2. Loads scenario parameters from YAML configs
3. Runs thermal simulations (3D-ICE or synthetic data)
4. Exports results to NumPy .npz format for PINN training
5. Computes comprehensive thermal statistics
6. Generates validation visualizations

Usage:
    python main.py --simulator 3d-ice --geometry geometry1 --output data/geometry1/train
    python main.py --simulator mock --all-geometries --all-scenarios
    python main.py --generator-only --output data/all_scenarios
"""

import sys
import argparse
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_builders import (
    build_geometry1, build_geometry2a,
    build_geometry3, build_geometry4, build_geometry5, build_geometry6,
)
from src.core.mesh import (
    generate_coords_and_indices,
    generate_power_density_field,
)
from src.scenario.generator import ScenarioGenerator
from src.scenario.throttling import apply_throttling
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.visualization.visualization import create_visualization_summary
from src.simulators.ice_simulator import ICESimulator
from src.simulators.hotspot_simulator import HotSpotSimulator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_geometries() -> Dict[str, object]:
    """Load all geometry definitions."""
    logger.info("Loading geometry definitions...")
    geometries = {
        'geometry1':  build_geometry1(),
        'geometry2a': build_geometry2a(),
        'geometry3':  build_geometry3(),
        'geometry4':  build_geometry4(),
        'geometry5':  build_geometry5(),
        'geometry6':  build_geometry6(),
    }
    for name, geom in geometries.items():
        logger.info(f"  {name}: {geom.name} "
                   f"({np.prod(geom.mesh_resolution):,} mesh points, "
                   f"{len(geom.layers)} layers)")
    return geometries


def generate_synthetic_temperature(coords: np.ndarray,
                                   geometry: object,
                                   ambient_temp: float = 25.0,
                                   power_field: np.ndarray = None) -> np.ndarray:
    """
    Generate synthetic temperature field using simplified thermal model.

    This enables testing without 3D-ICE when simulator unavailable.

    Args:
        coords: (N, 3) coordinate array [um]
        geometry: Geometry object
        ambient_temp: Ambient temperature [degC]
        power_field: (N,) volumetric power density [W/m³]

    Returns:
        (N,) temperature array in Kelvin
    """
    # Base temperature = ambient
    temps = np.ones_like(coords[:, 0]) * (ambient_temp + 273.15)

    # Vertical gradient (hotter at top/die, cooler toward sink)
    z_max = geometry.get_total_height()
    temps += (coords[:, 2] / z_max) * 50  # 0-50K gradient

    # Power-driven heating (simplified resistance model)
    # Assume effective thermal resistance ~1e-5 K/W per point
    if power_field is not None:
        temps += (power_field * 1e-5)
    else:
        # Default: add hotspot at die center
        die_layers = geometry.get_die_layers()
        if die_layers:
            die_layer = die_layers[-1]
            die_z = (die_layer.z_bottom + die_layer.z_top) / 2
            center_x = geometry.die_width / 2
            center_y = geometry.die_length / 2
            dist = np.sqrt((coords[:, 0] - center_x)**2 +
                          (coords[:, 1] - center_y)**2 +
                          ((coords[:, 2] - die_z) / 10)**2)
            hotspot = 80 * np.exp(-(dist / 1500)**2)
            temps += hotspot

    return temps


def process_scenario(scenario_name: str,
                     scenario_params: Dict,
                     geometry: object,
                     simulator: object,
                     exporter: NPZExporter,
                     calculator: StatisticsCalculator,
                     output_dir: Path,
                     use_synthetic: bool = False,
                     generate_plots: bool = True,
                     allow_synthetic_fallback: bool = False) -> Dict:
    logger.info(f"Processing {scenario_name}...")

    # Step 1: Generate coordinates
    logger.debug(f"  Generating coordinates...")
    coords, layer_indices = generate_coords_and_indices(geometry, uniform_z=False)
    logger.debug(f"    {len(coords):,} points, {len(np.unique(layer_indices))} layers")

    # Step 2: Run simulation
    logger.debug(f"  Running thermal simulation...")
    # Always generate power and layer indices from geometry (simulator only returns coords+temp)
    power = generate_power_density_field(
        coords, geometry, scenario_params['power_blocks'],
        power_map_by_layer=scenario_params.get('power_map_by_layer'))

    if use_synthetic:
        # Synthetic data for testing (no 3D-ICE required)
        temps = generate_synthetic_temperature(coords, geometry,
                                              scenario_params.get('t_ambient', 25.0),
                                              power)
    else:
        # Real simulator (3D-ICE or HotSpot)
        try:
            if scenario_params.get('throttle_enabled'):
                # Several 3D-ICE solves instead of one: iteratively derate
                # power until peak T settles under the throttle threshold.
                # Mutates scenario_params['power_blocks'] to the delivered
                # (derated) power, so everything below sees the real thing.
                parsed, throttle_info = apply_throttling(
                    simulator, geometry, scenario_params, scenario_name)
                scenario_params.update(throttle_info)
                if throttle_info['throttle_triggered']:
                    logger.info(
                        f"  Throttled: {throttle_info['throttle_iterations']} iters, "
                        f"derate={throttle_info['throttle_derate_factor']:.2f}, "
                        f"peak={throttle_info['throttle_peak_temp_c']:.1f}C"
                        + ("" if throttle_info['throttle_converged'] else " (pinned at power floor)")
                    )
            else:
                parsed = simulator.simulate(geometry, scenario_params, scenario_name)
            # parse_results returns 'temperature' key (not 'temp')
            coords = parsed['coords']
            temps = parsed['temperature']
            # Regenerate power and layer_indices on the simulator's output grid
            # (scenario_params['power_blocks'] already reflects any throttling)
            power = generate_power_density_field(
        coords, geometry, scenario_params['power_blocks'],
        power_map_by_layer=scenario_params.get('power_map_by_layer'))
            layer_indices = np.array([
                geometry.get_layer_index_at_z(z) for z in coords[:, 2]
            ], dtype=np.int32)
        except Exception as e:
            # A failed simulation must NOT silently become fabricated data. This
            # fallback previously ran unconditionally and produced 155 files of
            # synthetic garbage that were indistinguishable from real 3D-ICE output
            # until checked by hand. Failing loudly is the only safe default when
            # the caller asked for a real simulator.
            if not allow_synthetic_fallback:
                raise RuntimeError(
                    f"Simulator failed for {scenario_name}: {e}\n"
                    f"Refusing to substitute synthetic data. Re-run with "
                    f"--allow-synthetic-fallback if approximate data is acceptable."
                ) from e
            logger.warning(f"  Simulator failed, falling back to synthetic: {e}")
            temps = generate_synthetic_temperature(coords, geometry,
                                                  scenario_params.get('t_ambient', 25.0),
                                                  power)

    temp_c_min = np.min(temps) - 273.15
    temp_c_max = np.max(temps) - 273.15
    logger.debug(f"    Temperature range: {temp_c_min:.1f} to {temp_c_max:.1f} degC")

    # Step 3: Export to .npz
    logger.debug(f"  Exporting to .npz...")
    output_file = exporter.export_scenario(
        scenario_name,
        geometry,
        scenario_params,
        temps,
        coords
    )
    file_size_mb = output_file.stat().st_size / (1024**2)
    logger.debug(f"    {output_file.name} ({file_size_mb:.2f} MB)")

    # Step 4: Compute statistics
    logger.debug(f"  Computing statistics...")
    stats = calculator.compute_scenario_stats(
        scenario_name,
        coords,
        temps,
        power,
        layer_indices,
        geometry
    )

    # Export statistics
    stats_file = output_dir / f"{scenario_name}_stats.json"
    calculator.save_stats(stats_file, stats)
    logger.debug(f"    {stats_file.name}")

    # Step 5: Generate visualizations (optional)
    if generate_plots:
        logger.debug(f"  Generating plots...")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        try:
            create_visualization_summary(
                scenario_name,
                coords,
                temps,
                power,
                geometry,
                output_dir=plot_dir
            )
            logger.debug(f"    Plots saved to {plot_dir.name}/")
        except Exception as e:
            logger.warning(f"    Plotting failed: {e}")

    # Log summary
    logger.info(f"  [OK] {scenario_name}")
    logger.info(f"       Peak: {stats['hotspot']['peak_temperature_c']:.1f} degC "
               f"at ({stats['hotspot']['location_x_um']:.0f}, "
               f"{stats['hotspot']['location_y_um']:.0f}) um")

    return stats


def process_geometry(geometry_name: str,
                     geometry: object,
                     scenarios_per_type: Dict[str, List[Dict]],
                     simulator: object,
                     output_base: Path,
                     use_synthetic: bool = False,
                     skip_train: bool = False,
                     skip_test: bool = False,
                     train_start_index: int = 1,
                     allow_synthetic_fallback: bool = False) -> Dict:
    """
    Process all scenarios for a single geometry.

    Args:
        geometry_name: Name of geometry
        geometry: Geometry object
        scenarios_per_type: Dict with 'train' and 'test' lists of scenario params
        simulator: ThermalSimulator instance
        output_base: Base output directory
        use_synthetic: Use synthetic thermal data
        skip_train: Skip training scenarios
        skip_test: Skip test scenarios

    Returns:
        Dict with aggregated statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {geometry_name}")
    logger.info(f"{'='*80}")

    exporter = NPZExporter(output_base)
    calculator = StatisticsCalculator()

    all_stats = {'train': {}, 'test': {}, 'failed': []}
    scenario_count = 0

    # Process training scenarios
    if not skip_train:
        logger.info(f"\nTraining Scenarios ({len(scenarios_per_type['train'])} total):")
        train_dir = output_base / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        for i, scenario_params in enumerate(scenarios_per_type['train'], train_start_index):
            scenario_name = f"{geometry_name}_train_{i:03d}"
            if (output_base / f"{scenario_name}.npz").exists():
                logger.info(f"  [SKIP] {scenario_name} (already exists)")
                scenario_count += 1
                continue
            try:
                stats = process_scenario(
                    scenario_name,
                    scenario_params,
                    geometry,
                    simulator,
                    exporter,
                    calculator,
                    train_dir,
                    use_synthetic=use_synthetic,
                    generate_plots=(i == 1),  # Plot only first for speed
                    allow_synthetic_fallback=allow_synthetic_fallback
                )
                all_stats['train'][scenario_name] = stats
                scenario_count += 1
            except Exception as e:
                logger.error(f"  [FAILED] {scenario_name}: {e}")
                all_stats['failed'].append(scenario_name)

    # Process test scenarios
    if not skip_test:
        logger.info(f"\nTest Scenarios ({len(scenarios_per_type['test'])} total):")
        test_dir = output_base / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        for i, scenario_params in enumerate(scenarios_per_type['test'], 1):
            scenario_name = f"{geometry_name}_test_{i:03d}"
            if (output_base / f"{scenario_name}.npz").exists():
                logger.info(f"  [SKIP] {scenario_name} (already exists)")
                scenario_count += 1
                continue
            try:
                stats = process_scenario(
                    scenario_name,
                    scenario_params,
                    geometry,
                    simulator,
                    exporter,
                    calculator,
                    test_dir,
                    use_synthetic=use_synthetic,
                    generate_plots=(i == 1),  # Plot only first for speed
                    allow_synthetic_fallback=allow_synthetic_fallback
                )
                all_stats['test'][scenario_name] = stats
                scenario_count += 1
            except Exception as e:
                logger.error(f"  [FAILED] {scenario_name}: {e}")
                all_stats['failed'].append(scenario_name)

    logger.info(f"\n{geometry_name}: {scenario_count} scenarios processed")
    return all_stats


def main():
    """Main orchestration entry point."""
    parser = argparse.ArgumentParser(
        description='3D-IC Thermal PINN Benchmark Data Generator'
    )
    parser.add_argument(
        '--simulator',
        choices=['3d-ice', 'hotspot', 'mock'],
        default='mock',
        help='Thermal simulator backend (default: mock for synthetic data)'
    )
    parser.add_argument(
        '--ice-executable',
        type=str,
        default='3D-ICE-Emulator',
        help='Path to 3D-ICE-Emulator binary (default: "3D-ICE-Emulator" on PATH). '
             'For WSL use e.g. "wsl /home/user/3d-ice/bin/3D-ICE-Emulator"'
    )
    parser.add_argument(
        '--hotspot-executable',
        type=str,
        default='hotspot',
        help='Path to HotSpot binary (default: "hotspot" on PATH). '
             'For WSL use e.g. "wsl /home/user/HotSpot/hotspot"'
    )
    parser.add_argument(
        '--geometry',
        choices=['geometry1', 'geometry2a', 'geometry3',
                 'geometry4', 'geometry5', 'geometry6', 'all'],
        default='geometry1',
        help='Geometry to process (default: geometry1)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data'),
        help='Output directory for .npz and stats files (default: data/)'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training scenarios'
    )
    parser.add_argument(
        '--power-map', choices=['grf', 'floorplan', 'mixed'], default=None,
        help='Use per-cell power maps instead of block scalars. Block power spans '
             'only ~4 dimensions no matter how many scenarios are generated, which '
             'is why ridge regression solves the block-scalar dataset; power maps '
             'span ~N-1. TDP budget and density limits are preserved.')
    parser.add_argument(
        '--power-map-resolution', type=int, default=0,
        help='Power map cells per axis (0 = use the lateral mesh resolution).')
    parser.add_argument(
        '--throttle', action='store_true',
        help='Enable package-level thermal throttling (DVFS): power is iteratively '
             'derated when peak temperature exceeds --throttle-temp, then re-solved. '
             'Costs ~2x a normal solve (see src/scenario/throttling.py). Off by '
             'default: it multiplies solve time and should not silently slow down '
             'every regeneration.')
    parser.add_argument(
        '--throttle-temp', type=float, default=95.0,
        help='Throttle threshold in Celsius (default: 95.0).')
    parser.add_argument(
        '--throttle-gain', type=float, default=2.0,
        help='Derate aggressiveness per degree of overshoot (default: 2.0).')
    parser.add_argument(
        '--throttle-floor', type=float, default=0.3,
        help='Minimum power fraction throttling will derate to (default: 0.3).')
    parser.add_argument(
        '--allow-synthetic-fallback', action='store_true',
        help='Substitute an analytical approximation when the simulator fails. OFF by default: a silent fallback once produced 155 files of synthetic data indistinguishable from real 3D-ICE output.')
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip test scenarios'
    )
    parser.add_argument(
        '--all-geometries',
        action='store_true',
        help='Process all 4 geometries'
    )
    parser.add_argument(
        '--all-scenarios',
        action='store_true',
        help='Generate 15 scenarios per geometry (train + test)'
    )
    parser.add_argument(
        '--extra-train',
        type=int,
        default=0,
        metavar='N',
        help='Generate N additional training scenarios beyond the default 15 '
             '(uses _EXTRA_POOL in generator.py; max 25 for standard geometries, '
             '15 for 2p5d). Combine with --skip-test to only run the new scenarios.'
    )
    parser.add_argument(
        '--train-start',
        type=int,
        default=None,
        metavar='M',
        help='File index for the first extra training scenario '
             '(default: auto-detect from existing files, or 16 if none found).'
    )
    parser.add_argument(
        '--pool-start',
        type=int,
        default=0,
        metavar='P',
        help='Index into _EXTRA_POOL/_EXTRA_POOL_2P5D to start drawing scenarios from '
             '(default 0). Use this to avoid duplicating scenarios from a prior --extra-train run.'
    )
    parser.add_argument(
        '--generator-only',
        action='store_true',
        help='Only generate scenarios, do not run simulations'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 80)
    logger.info("3D-IC THERMAL PINN BENCHMARK SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Simulator: {args.simulator}")
    logger.info(f"Output: {args.output}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load geometries
    geometries = load_geometries()

    # Determine which geometries to process
    geometry_list = ['geometry1', 'geometry2a',
                     'geometry3', 'geometry4', 'geometry5', 'geometry6'] \
        if args.all_geometries else [args.geometry]

    # Initialize simulator — use a per-process temp dir so concurrent main.py
    # invocations (one per geometry) don't share 3D-ICE config/output files.
    simulator = None
    _sim_tmpdir = None
    if args.simulator == '3d-ice':
        _sim_tmpdir = tempfile.mkdtemp(prefix=f'ice_pid{os.getpid()}_')
        config_dir = Path(_sim_tmpdir) / 'configs'
        sim_output_dir = Path(_sim_tmpdir) / 'output'
        config_dir.mkdir()
        sim_output_dir.mkdir()
        simulator = ICESimulator(
            config_dir=config_dir,
            output_dir=sim_output_dir,
            executable=args.ice_executable
        )
        logger.info(f"3D-ICE simulator: {args.ice_executable}")
    elif args.simulator == 'hotspot':
        _sim_tmpdir = tempfile.mkdtemp(prefix=f'hotspot_pid{os.getpid()}_')
        config_dir = Path(_sim_tmpdir) / 'configs'
        sim_output_dir = Path(_sim_tmpdir) / 'output'
        config_dir.mkdir()
        sim_output_dir.mkdir()
        simulator = HotSpotSimulator(
            config_dir=config_dir,
            output_dir=sim_output_dir,
            executable=args.hotspot_executable
        )
        logger.info(f"HotSpot simulator: {args.hotspot_executable}")
        logger.info("Note: HotSpot supports geometry1 (single-die) only")

    # Generate scenarios for all requested geometries
    logger.info("\nGenerating scenario parameters...")
    scenario_generator = ScenarioGenerator()
    scenarios_by_geometry = {}
    train_start_by_geometry = {}

    for geom_name in geometry_list:
        geom = geometries[geom_name]

        # Always generate base 15 train + 5 test first
        geom_scenarios = scenario_generator.generate_all_scenarios(geom)
        # Spatially varying TSV density field, replacing the geometry's single
        # tsv_density scalar. No-op for geometries without TSV layers -- always
        # on, since it strictly increases fidelity at no measured solve-time
        # cost (see assumptions.md TSV section).
        scenario_generator.attach_tsv_maps(geom_scenarios, geom)
        if args.power_map:
            # Replace block scalars with per-cell power fields. Keeps the TDP
            # budget, density ceiling and cooling rule; only redistributes power
            # in space. This is what makes the input a function rather than ~8
            # numbers -- see src/scenario/power_maps.py.
            scenario_generator.attach_power_maps(
                geom_scenarios, geom, kind=args.power_map,
                resolution=args.power_map_resolution)
        if args.throttle:
            # Package-level DVFS: power becomes a function of the temperature
            # being solved for, a real closed feedback loop no fixed-source
            # scenario can express -- see src/scenario/throttling.py. Costs
            # ~2x solve time per scenario (measured convergence in 2 solves).
            scenario_generator.attach_throttling(
                geom_scenarios, throttle_temp_c=args.throttle_temp,
                gain=args.throttle_gain, power_floor=args.throttle_floor)
        scenario_dicts = [s.to_dict() for s in geom_scenarios]
        train = [s for s in scenario_dicts if s['type'] == 'train']
        test  = [s for s in scenario_dicts if s['type'] == 'test']

        if args.extra_train > 0:
            # Append extra training scenarios starting after any already on disk.
            if args.train_start is not None:
                start = args.train_start
            else:
                existing = sorted(
                    (args.output / geom_name).glob(f"{geom_name}_train_*.npz")
                )
                start = (int(existing[-1].stem.split('_')[-1]) + 1
                         if existing else 16)
            extra = scenario_generator.generate_extra_training_scenarios(
                geom, args.extra_train, start_index=start,
                pool_start_idx=args.pool_start,
            )
            scenario_generator.attach_tsv_maps(extra, geom, seed_base=900_000)
            if args.power_map:
                scenario_generator.attach_power_maps(
                    extra, geom, kind=args.power_map,
                    resolution=args.power_map_resolution, seed_base=500_000)
            if args.throttle:
                scenario_generator.attach_throttling(
                    extra, throttle_temp_c=args.throttle_temp,
                    gain=args.throttle_gain, power_floor=args.throttle_floor)
            train = train + [s.to_dict() for s in extra]
            logger.info(
                "  %s: 15 base + %d extra train + 5 test  (extra indices %d-%d)",
                geom_name, args.extra_train, start, start + args.extra_train - 1,
            )
        else:
            logger.info(f"  {geom_name}: {len(train)} train + {len(test)} test scenarios")

        scenarios_by_geometry[geom_name] = {'train': train, 'test': test}
        train_start_by_geometry[geom_name] = 1

    # If generator-only mode, save scenarios and exit
    if args.generator_only:
        logger.info("\nSaving scenario configurations...")
        for geom_name, scenario_dict in scenarios_by_geometry.items():
            geom_output = args.output / geom_name
            geom_output.mkdir(parents=True, exist_ok=True)
            config_file = geom_output / "scenarios.json"

            # Convert scenario dicts to serializable format
            def _serialize_scenario(s):
                d = {
                    'name': s['name'],
                    'geometry': s['geometry'],
                    'type': s['type'],
                    'power_blocks': s['power_blocks'],
                    'htc': float(s['htc']),
                    't_ambient': float(s['t_ambient']),
                    'pattern': s['pattern'],
                    'description': s['description'],
                    'rdl_joule_fraction': float(s.get('rdl_joule_fraction', 0.05)),
                    'layer_k_overrides': s.get('layer_k_overrides', {}),
                }
                return d
            train_serializable = [_serialize_scenario(s) for s in scenario_dict['train']]
            test_serializable  = [_serialize_scenario(s) for s in scenario_dict['test']]

            with open(config_file, 'w') as f:
                json.dump({
                    'train': train_serializable,
                    'test': test_serializable
                }, f, indent=2)
            logger.info(f"  {config_file}")
        logger.info("\n[OK] Scenario generation complete")
        return 0

    # Process geometries
    all_results = {}
    for geom_name in geometry_list:
        geom = geometries[geom_name]
        output_geom = args.output / geom_name

        results = process_geometry(
            geom_name,
            geom,
            scenarios_by_geometry[geom_name],
            simulator,
            output_geom,
            use_synthetic=(args.simulator == 'mock'),
            skip_train=args.skip_train,
            skip_test=(args.skip_test or args.extra_train > 0),
            train_start_index=train_start_by_geometry[geom_name],
            allow_synthetic_fallback=args.allow_synthetic_fallback,
        )
        all_results[geom_name] = results

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*80}")

    total_train = sum(len(all_results[g]['train']) for g in all_results)
    total_test = sum(len(all_results[g]['test']) for g in all_results)
    total_scenarios = total_train + total_test
    failed = {g: all_results[g].get('failed', []) for g in all_results}
    n_failed = sum(len(v) for v in failed.values())
    total_size_mb = sum(
        (args.output / g / "train").stat().st_size / (1024**2)
        for g in all_results if (args.output / g / "train").exists()
    )

    logger.info(f"Geometries processed: {len(all_results)}")
    logger.info(f"Training scenarios: {total_train}")
    logger.info(f"Test scenarios: {total_test}")
    logger.info(f"Total scenarios: {total_scenarios}")
    logger.info(f"Data directory: {args.output}")
    logger.info(f"Estimated size: {total_size_mb:.1f} MB")

    logger.info("\nDataset structure:")
    for geom_name in all_results:
        logger.info(f"  {geom_name}/")
        logger.info(f"    train/  ({len(all_results[geom_name]['train'])} scenarios)")
        logger.info(f"    test/   ({len(all_results[geom_name]['test'])} scenarios)")
        logger.info(f"    plots/  (optional visualizations)")

    if _sim_tmpdir and Path(_sim_tmpdir).exists():
        shutil.rmtree(_sim_tmpdir, ignore_errors=True)

    # Exit non-zero on partial or empty output. Per-scenario failures used to be
    # logged and then swallowed: a run in which EVERY scenario raised still
    # printed "PROCESSING COMPLETE" and returned 0, so a batch script saw success
    # while producing no files at all. Anything driving this in a loop needs the
    # exit code to mean something.
    if n_failed:
        logger.error("%d scenario(s) FAILED:", n_failed)
        for geom_name, names in failed.items():
            if names:
                logger.error("  %s: %s", geom_name, ', '.join(names))

    if total_scenarios == 0:
        logger.error(
            "No scenarios were produced. Nothing was written to %s.", args.output)
        return 1

    if n_failed:
        logger.error("Completed with failures -- dataset is INCOMPLETE.")
        return 1

    logger.info("\nNext steps:")
    logger.info("1. Load .npz files for PINN training")
    logger.info("2. Compute additional statistics across dataset")
    logger.info("3. Validate physical consistency of thermal results")
    logger.info("4. Train PINNs on benchmark dataset")

    return 0


if __name__ == '__main__':
    sys.exit(main())
