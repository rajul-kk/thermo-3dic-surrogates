"""
Generate a leakage-feedback pilot batch: the sharpest available test of this
benchmark's central linearity claim.

Throttling (goal.md Track A) is *negative* feedback -- self-limiting, converges in
~2 solves, and degraded the linear baseline only modestly (spatial R^2 0.970 ->
0.890/0.919). Leakage is *positive* feedback: hotter -> more leakage -> hotter, with
no steady state at all above a critical loop gain. If ridge still solves the dataset
under positive electrothermal feedback, the linearity finding is robust rather than
an artifact of a benign regime. If it does not, that is the first mechanism in this
benchmark that genuinely requires a learned operator.

Sweeps leakage aggressiveness across the batch so the dataset spans benign to
near-runaway rather than sitting at one operating point -- the interesting behaviour
is concentrated near the critical gain, and a single setting would miss it.

Usage:
    python scripts/gen_leakage_pilot.py --geometry geometry1 --n 20
"""
import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import get_geometry_by_name
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# (leakage_fraction, k_double_c). Lower k_double = leakage doubles faster with
# temperature = higher loop gain. Spans benign (0.15/25) to aggressive (0.45/8).
LEAKAGE_SETTINGS = [
    (0.15, 25.0),
    (0.25, 20.0),
    (0.30, 15.0),
    (0.35, 12.0),
    (0.45, 8.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', default='geometry1')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--start-index', type=int, default=0,
                    help='Skip the first N scenarios (0-based) -- for extending an '
                         'existing pilot without re-running already-generated ones.')
    ap.add_argument('--extra-train', type=int, default=0,
                    help='Append this many extra training scenarios (denser HTC/power '
                         'coverage, same pool the full dataset draws from) after the '
                         'base 15 train + 5 test, so the pilot spans a wider and more '
                         'realistic range of base operating points instead of just the '
                         '20-scenario base set.')
    ap.add_argument('--output', type=Path, default=Path('data/3d-ice-leakage-pilot'))
    ap.add_argument('--ice-executable',
                    default='wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator')
    ap.add_argument('--t-ref', type=float, default=85.0,
                    help='Junction temperature at which leakage equals its '
                         'nominal share (multiplier is exactly 1.0 here). '
                         'Defaults to 85 C because TDP/leakage splits are quoted '
                         'at a nominal junction temperature, not at ambient; '
                         'referencing to ambient would spuriously amplify every '
                         'realistically-hot scenario.')
    args = ap.parse_args()

    geom = get_geometry_by_name(args.geometry)
    gen = ScenarioGenerator()
    all_scenarios = gen.generate_all_scenarios(geom)
    if args.extra_train > 0:
        all_scenarios = all_scenarios + gen.generate_extra_training_scenarios(
            geom, args.extra_train, start_index=16, pool_start_idx=0)
    scenarios = all_scenarios[args.start_index:args.n]

    output_dir = args.output / args.geometry
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = NPZExporter(output_dir=output_dir)
    calculator = StatisticsCalculator()

    ok, failed, runaway, converged = 0, [], 0, 0
    for offset, sc in enumerate(scenarios):
        i = args.start_index + offset
        frac, k_double = LEAKAGE_SETTINGS[i % len(LEAKAGE_SETTINGS)]
        sc_dict = sc.to_dict()
        sc_dict['leakage_enabled'] = True
        sc_dict['leakage_fraction'] = frac
        sc_dict['leakage_k_double_c'] = k_double
        sc_dict['leakage_t_ref_c'] = args.t_ref

        scenario_name = f"{args.geometry}_{sc.scenario_type}_{i+1:03d}"

        tmp = Path(tempfile.mkdtemp(prefix='leakpilot_'))
        config_dir = tmp / 'cfg'
        sim_out = tmp / 'simout'
        config_dir.mkdir()
        sim_out.mkdir()
        simulator = ICESimulator(config_dir=config_dir, output_dir=sim_out,
                                 executable=args.ice_executable)
        try:
            stats = process_scenario(
                scenario_name=scenario_name,
                scenario_params=sc_dict,
                geometry=geom,
                simulator=simulator,
                exporter=exporter,
                calculator=calculator,
                output_dir=output_dir,
                generate_plots=False,
            )
            if sc_dict.get('leakage_runaway'):
                runaway += 1
            if sc_dict.get('leakage_converged'):
                converged += 1
            log.info("[OK] %s  frac=%.2f k=%.0f  x%.3f  peak=%.1fC%s",
                     scenario_name, frac, k_double,
                     sc_dict.get('leakage_multiplier', 1.0),
                     stats['hotspot']['peak_temperature_c'],
                     ' RUNAWAY' if sc_dict.get('leakage_runaway') else '')
            ok += 1
        except Exception as exc:
            log.error("[FAILED] %s: %s", scenario_name, exc)
            failed.append(scenario_name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    log.info("Leakage pilot complete: %d ok (%d converged, %d runaway), %d failed",
             ok, converged, runaway, len(failed))
    if failed:
        log.error("Failed: %s", failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
