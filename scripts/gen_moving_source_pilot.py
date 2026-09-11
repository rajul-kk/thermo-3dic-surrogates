"""
Moving-source pilot: the fix for the defect in docs/report.md §9.14, and a controlled test of the mechanism claimed there.
"""
import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import (lateral_chiplets, place_chiplets, placement_summary,
                                random_block_placement, random_placement)
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('moving_pilot')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', default='geometry4')
    ap.add_argument('--n', type=int, default=45, help='total scenarios (last 5 become test)')
    ap.add_argument('--max-shift-um', type=float, default=1500.0)
    ap.add_argument('--margin-um', type=float, default=250.0)
    ap.add_argument('--independent-blocks', action='store_true',
                    help='place every power block independently (2 DOF per block) rather '
                         'than translating chiplets rigidly -- see docs/report.md 9.15')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--ice-executable',
                    default='wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator')
    args = ap.parse_args()

    out_root = args.output or Path(f'data/3d-ice-moving-{args.geometry}')
    geom = get_geometry_by_name(args.geometry)
    n_chiplets = len(lateral_chiplets(geom))
    log.info('%s: %d lateral chiplets, max shift %.0f um, margin %.0f um',
             args.geometry, n_chiplets, args.max_shift_um, args.margin_um)
    if n_chiplets == 0:
        log.info('no die footprints: the power blocks translate as a rigid group '
                 '(homogeneous-medium control case)')

    gen = ScenarioGenerator()
    base = gen.generate_all_scenarios(geom)
    if args.n > len(base):
        base = base + gen.generate_extra_training_scenarios(
            geom, args.n - len(base), start_index=len(base) + 1, pool_start_idx=0)
    base = base[:args.n]

    output_dir = out_root / args.geometry
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = NPZExporter(output_dir=output_dir)
    calculator = StatisticsCalculator()
    rng = np.random.default_rng(args.seed)

    # Last 5 are the test split, matching the naming the rest of the project uses.
    n_test = 5
    ok, failed, moved_any = 0, [], 0
    for i, sc in enumerate(base):
        split = 'test' if i >= len(base) - n_test else 'train'
        name = f'{args.geometry}_{split}_{i + 1:03d}'

        if args.independent_blocks:
            offsets = random_block_placement(geom, rng, margin_um=args.margin_um)
        else:
            offsets = random_placement(geom, rng, max_shift_um=args.max_shift_um,
                                       margin_um=args.margin_um)
        placed = place_chiplets(geom, offsets)
        if any(dx or dy for dx, dy in offsets.values()):
            moved_any += 1

        params = sc.to_dict()
        # Recorded so the npz metadata carries the placement actually simulated.
        params['chiplet_offsets'] = {k: list(v) for k, v in offsets.items()}
        params['placement_seed'] = args.seed

        tmp = Path(tempfile.mkdtemp(prefix='moving_'))
        cfg, simout = tmp / 'cfg', tmp / 'simout'
        cfg.mkdir(); simout.mkdir()
        sim = ICESimulator(config_dir=cfg, output_dir=simout,
                           executable=args.ice_executable)
        try:
            stats = process_scenario(
                scenario_name=name, scenario_params=params, geometry=placed,
                simulator=sim, exporter=exporter, calculator=calculator,
                output_dir=output_dir, generate_plots=False)
            log.info('[OK] %s  %s  peak=%.1fC', name, placement_summary(offsets)[:60],
                     stats['hotspot']['peak_temperature_c'])
            ok += 1
        except Exception as exc:
            log.error('[FAILED] %s: %s', name, exc)
            failed.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    log.info('%s moving-source pilot: %d ok, %d failed, %d/%d scenarios actually moved',
             args.geometry, ok, len(failed), moved_any, len(base))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
