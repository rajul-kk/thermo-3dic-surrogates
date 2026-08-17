"""
Generate a 40-scenario pilot dataset for geometry7 (CoWoS-L reticle-stitched
package, see src/core/geometry_builders.py build_geometry7 docstring).

geometry7 is NOT part of the standard 6-geometry benchmark dataset -- it is a
pilot testing whether this benchmark's central "ridge already solves this"
finding survives a structurally new mechanism: sparse high-k (silicon LSI
bridge) islands in an otherwise low-k (organic substrate) passive spreading
layer, replacing every other 2.5D geometry's uniform-material interposer.

40 scenarios = the same 15 base-train + 5 test scenarios every geometry gets
(ScenarioGenerator.generate_all_scenarios), plus 20 extra-train scenarios
(ScenarioGenerator.generate_extra_training_scenarios) drawing denser HTC/power
coverage from the 2.5D-geometry pool -- the same recipe used to grow
geometry1..6 from their base 20 to the full per-geometry counts recorded in
docs/geometry_reference.md.

Usage:
    python scripts/gen_geometry7_pilot.py --output data/3d-ice-geometry7-pilot
"""
import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_geometry7
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('data/3d-ice-geometry7-pilot'))
    ap.add_argument('--ice-executable',
                    default='wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator')
    ap.add_argument('--n-extra', type=int, default=20,
                    help='Extra training scenarios beyond the base 15 train + 5 test '
                         '(default 20, giving 40 total)')
    ap.add_argument('--start-index', type=int, default=0,
                    help='Skip the first N scenarios (0-based) -- for resuming a '
                         'partially-generated run without re-solving finished ones.')
    args = ap.parse_args()

    geom = build_geometry7()
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geom)
    if args.n_extra > 0:
        scenarios = scenarios + gen.generate_extra_training_scenarios(
            geom, args.n_extra, start_index=16, pool_start_idx=0)
    total = len(scenarios)
    scenarios = scenarios[args.start_index:]
    log.info("geometry7 pilot: %d scenarios total, starting at index %d (%d to run)",
             total, args.start_index, len(scenarios))

    output_dir = args.output / 'geometry7'
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = NPZExporter(output_dir=output_dir)
    calculator = StatisticsCalculator()

    ok, failed = 0, []
    t_start = time.time()
    for offset, sc in enumerate(scenarios):
        i = args.start_index + offset
        scenario_name = f"geometry7_{sc.scenario_type}_{i+1:03d}"
        tmp = Path(tempfile.mkdtemp(prefix='g7pilot_'))
        config_dir = tmp / 'cfg'
        sim_out = tmp / 'simout'
        config_dir.mkdir()
        sim_out.mkdir()
        simulator = ICESimulator(config_dir=config_dir, output_dir=sim_out,
                                 executable=args.ice_executable)
        t0 = time.time()
        try:
            stats = process_scenario(
                scenario_name=scenario_name,
                scenario_params=sc.to_dict(),
                geometry=geom,
                simulator=simulator,
                exporter=exporter,
                calculator=calculator,
                output_dir=output_dir,
                generate_plots=False,
            )
            dt = time.time() - t0
            log.info("[OK] %s (%d/%d, %.1fs)  pattern=%s htc=%.0f t_amb=%.1fC  peak=%.1fC",
                     scenario_name, i + 1, total, dt, sc.pattern, sc.htc, sc.t_ambient,
                     stats['hotspot']['peak_temperature_c'])
            ok += 1
        except Exception as exc:
            log.error("[FAILED] %s: %s", scenario_name, exc)
            failed.append(scenario_name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    total_time = time.time() - t_start
    log.info("geometry7 pilot complete: %d ok, %d failed, %.1f min total (%.1fs/scenario avg)",
             ok, len(failed), total_time / 60.0, total_time / max(1, ok))
    if failed:
        log.error("Failed: %s", failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
