"""
Generate low-fidelity analytical thermal data for all (or selected) geometries.

Runs the 1D-resistance + 2D-Gaussian simulator for every train/test scenario
produced by the standard ScenarioGenerator.  Output is written to data/lf/
in the same NPZ format as the 3D-ICE HF files — drop-in compatible with the
ARO dataset loader.

Usage
-----
# All geometries:
python scripts/generate_lf_data.py --output data/lf

# Specific geometries:
python scripts/generate_lf_data.py --output data/lf \
    --geometries geometry1 geometry4 geometry5

# Quiet run (no progress bar):
python scripts/generate_lf_data.py --output data/lf --quiet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_builders import get_geometry_by_name, build_all_geometries
from src.scenario.generator import ScenarioGenerator
from src.simulators.lf_simulator import LowFidelitySimulator

_log = logging.getLogger(__name__)

ALL_GEOMS = [
    'geometry1',
    'geometry2a',
    'geometry3',
    'geometry4',
    'geometry5',
    'geometry6',
]


def generate_for_geometry(
    geom_name: str,
    output_dir: Path,
    sim: LowFidelitySimulator,
    quiet: bool = False,
) -> tuple:
    """Generate all LF scenarios for one geometry. Returns (n_ok, n_fail)."""
    geometry = get_geometry_by_name(geom_name)
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geometry)

    n_ok = 0
    n_fail = 0

    for sc in scenarios:
        out_path = output_dir / f"{sc.name}_lf.npz"
        if out_path.exists():
            if not quiet:
                _log.debug("Skip (exists): %s", out_path.name)
            n_ok += 1
            continue

        try:
            result = sim.simulate(
                geometry=geometry,
                htc=float(sc.htc),
                t_ambient_celsius=float(sc.t_ambient),
                power_blocks_wcm2=sc.power_blocks,
                scenario_name=sc.name,
            )
            # Inject extra metadata fields that the ARO / PINN loaders expect
            meta = dict(result['metadata'][0])
            meta['scenario_name']      = sc.name
            meta['geometry_name']      = geom_name
            meta['htc']                = sc.htc
            meta['t_ambient_celsius']  = sc.t_ambient
            meta['tsv_density']        = getattr(geometry, 'tsv_density', 0.0)
            meta['scenario_type']      = sc.scenario_type
            meta['pattern']            = sc.pattern
            meta['tim_top_k']          = 4.0  # default TIM k; no k-sweep for LF
            result['metadata'] = np.array([meta], dtype=object)

            np.savez_compressed(str(out_path), **result)
            n_ok += 1
            if not quiet:
                _log.info("  wrote %s", out_path.name)
        except Exception as exc:
            _log.error("  FAILED %s: %s", sc.name, exc)
            n_fail += 1

    return n_ok, n_fail


def main():
    parser = argparse.ArgumentParser(description="Generate LF thermal data per geometry")
    parser.add_argument('--output', default='data/lf', type=Path,
                        help="Output directory for LF NPZ files (default: data/lf)")
    parser.add_argument('--geometries', nargs='+', default=ALL_GEOMS,
                        choices=ALL_GEOMS, metavar='GEOM',
                        help="Geometries to generate (default: all 8)")
    parser.add_argument('--quiet',  action='store_true', help="Suppress per-file log lines")
    parser.add_argument('--seed',   type=int, default=0)
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Output directory: %s", output_dir.resolve())

    sim = LowFidelitySimulator(seed=args.seed)

    total_ok = 0
    total_fail = 0
    t0 = time.time()

    for geom_name in args.geometries:
        t_geom = time.time()
        print(f"\n[{geom_name}] generating LF scenarios ...", flush=True)
        n_ok, n_fail = generate_for_geometry(geom_name, output_dir, sim, quiet=args.quiet)
        elapsed = time.time() - t_geom
        status = "OK" if n_fail == 0 else f"{n_fail} FAILED"
        print(f"  -> {n_ok} files  {status}  ({elapsed:.1f}s)", flush=True)
        total_ok   += n_ok
        total_fail += n_fail

    elapsed_total = time.time() - t0
    print(f"\nDone: {total_ok} LF files generated, {total_fail} failures  "
          f"(total {elapsed_total:.1f}s)", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)

    if total_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
