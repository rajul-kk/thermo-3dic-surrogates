"""
Generate a pilot batch of microchannel-cooled geometry6 scenarios.

The process_scenario -> NPZExporter pipeline was already confirmed correct for
microchannel-cooled geometries (2026-08-09, see goal.md) -- what was actually
missing was a way to select a microchannel-cooled geometry at all, since none
of the 6 registered geometries have coolant_layer_name set. This builds one
ad hoc (geometry6 + a thin copper base plate below the channel, matching the
pattern already validated against the real 3D-ICE 4.0 binary) and runs a
small scenario sweep through the real pipeline, varying coolant flow rate --
the axis that actually introduces advection, i.e. the one thing in this
benchmark that isn't confined to the linear-conduction regime.

Usage:
    python scripts/gen_microchannel_pilot.py --n 12 --output data/3d-ice-microchannel-pilot
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry import Geometry, Layer
from src.core.geometry_builders import build_geometry6
from src.core.material import MaterialLibrary
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

GEOM_NAME = 'geometry6_microchannel'

# Flow rate is the axis that actually introduces advection; sweep it across a
# realistic range for a package this size rather than holding it fixed.
FLOW_RATES_ML_MIN = [80.0, 120.0, 160.0, 200.0, 240.0, 280.0]


def build_microchannel_geometry6() -> Geometry:
    g0 = build_geometry6()
    mat_cu = MaterialLibrary.get('copper')
    # 3D-ICE rejects a channel as the bottom-most stack element; a thin base
    # plate below it matches how real cold plates are built anyway.
    cp_base = Layer(name='cp_base', material='copper', thickness=500.0,
                    k_thermal=mat_cu.k_thermal,
                    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity)
    return Geometry(
        name=GEOM_NAME, geometry_type=g0.geometry_type,
        layers=[cp_base] + g0.layers, power_blocks=g0.power_blocks,
        die_width=g0.die_width, die_length=g0.die_length,
        mesh_resolution=g0.mesh_resolution, tsv_density=g0.tsv_density,
        die_footprints=g0.die_footprints, underfill_k=g0.underfill_k,
        coolant_layer_name='heat_sink',
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12, help='Number of pilot scenarios')
    ap.add_argument('--output', type=Path, default=Path('data/3d-ice-microchannel-pilot'))
    ap.add_argument('--ice-executable', default='wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator')
    args = ap.parse_args()

    geom = build_microchannel_geometry6()
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geom)[:args.n]

    output_dir = args.output / GEOM_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = NPZExporter(output_dir=output_dir)
    calculator = StatisticsCalculator()

    ok, failed = 0, []
    for i, sc in enumerate(scenarios):
        sc_dict = sc.to_dict()
        sc_dict['cooling_mode'] = 'microchannel_2rm'
        sc_dict['microchannel'] = {
            'channel_height_um': 200.0,
            'channel_length_um': 100.0,
            'wall_length_um': 100.0,
            'wall_material': 'copper',
            'flow_rate_ml_min': FLOW_RATES_ML_MIN[i % len(FLOW_RATES_ML_MIN)],
            'coolant_htc_top_wm2k': 25000.0,
            'coolant_htc_bottom_wm2k': 15000.0,
            'coolant_vhc_jm3k': 4.18e6,
        }
        scenario_name = f"{GEOM_NAME}_{sc.scenario_type}_{i:03d}"

        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix=f'mcpilot_{i}_'))
        config_dir = tmp / 'cfg'; sim_out = tmp / 'simout'
        config_dir.mkdir(); sim_out.mkdir()
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
            log.info("[OK] %s  flow=%.0f mL/min  peak=%.1fC", scenario_name,
                     sc_dict['microchannel']['flow_rate_ml_min'],
                     stats['hotspot']['peak_temperature_c'])
            ok += 1
        except Exception as exc:
            log.error("[FAILED] %s: %s", scenario_name, exc)
            failed.append(scenario_name)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    log.info("Pilot batch complete: %d ok, %d failed", ok, len(failed))
    if failed:
        log.error("Failed scenarios: %s", failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
