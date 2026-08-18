"""
Interface-uncertainty-style sweep for geometry7's two invented material constants:
organic_substrate (the bare CoWoS-L field, k=0.5 nominal, literature range 0.3-0.8
W/m*K per material.py) and lsi_bridge_via (the bridge/via composite under each
compute die, k=60 nominal, "matching hybrid_bonding's existing precedent, not
independently derived" per material.py -- the most genuinely uncertain constant in
this geometry).

Same methodology as scripts/gen_interface_uncertainty_pilot.py (Track C): hold
power pattern, HTC and ambient FIXED at one real operating point and vary exactly
one material constant at a time, so the resulting spread isolates that constant's
effect rather than conflating it with anything else.

Fixed operating point: the actual hottest scenario from the (post-fix) 40-scenario
geometry7 pilot (geometry7_train_040: uniform pattern, htc=5000, t_amb=45C, peak
181.2C) -- a real, already-validated operating point, not an invented one.

Usage:
    python scripts/gen_geometry7_material_sweep.py --output data/3d-ice-geometry7-material-sweep
"""
import argparse
import logging
import shutil
import sys
import tempfile
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

# organic_substrate: literature range 0.3-0.8 W/m*K (material.py). lsi_bridge_via:
# genuinely uncertain engineering estimate -- sweep brackets the chosen 60 W/m*K
# from sparse via density (20) to dense (150, approaching but not reaching bulk Si).
SWEEPS = {
    'substrate_organic__gap': [0.3, 0.4, 0.5, 0.65, 0.8],   # organic_substrate field
    'substrate_organic':      [20.0, 40.0, 60.0, 100.0, 150.0],  # lsi_bridge_via islands
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('data/3d-ice-geometry7-material-sweep'))
    ap.add_argument('--ice-executable',
                    default='wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator')
    args = ap.parse_args()

    geom = build_geometry7()
    gen = ScenarioGenerator()
    all_scenarios = gen.generate_all_scenarios(geom) + gen.generate_extra_training_scenarios(
        geom, 20, start_index=16, pool_start_idx=0)
    # geometry7_train_040 is the 40th scenario (index 39) -- the real hottest
    # scenario from the validated pilot (uniform, htc=5000, t_amb=45C, 181.2C).
    base = all_scenarios[39]
    log.info("Fixed operating point: %s pattern=%s htc=%.0f t_amb=%.1fC",
             base.name, base.pattern, base.htc, base.t_ambient)

    output_dir = args.output / 'geometry7'
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = NPZExporter(output_dir=output_dir)
    calculator = StatisticsCalculator()

    runs = []
    for override_key, values in SWEEPS.items():
        tag = 'gap' if override_key.endswith('__gap') else 'bridge'
        for v in values:
            runs.append((f"{tag}_{v:g}", {override_key: v}))
    log.info("Running %d sweep points", len(runs))

    ok, failed = 0, []
    for name_frag, overrides in runs:
        scenario_name = f"geometry7_matsweep_{name_frag}"
        sc_dict = base.to_dict()
        sc_dict['layer_k_overrides'] = overrides

        tmp = Path(tempfile.mkdtemp(prefix='g7matsweep_'))
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
            log.info("[OK] %s  overrides=%s  peak=%.2fC",
                     scenario_name, overrides, stats['hotspot']['peak_temperature_c'])
            ok += 1
        except Exception as exc:
            log.error("[FAILED] %s: %s", scenario_name, exc)
            failed.append(scenario_name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    log.info("geometry7 material sweep complete: %d ok, %d failed", ok, len(failed))
    if failed:
        log.error("Failed: %s", failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
