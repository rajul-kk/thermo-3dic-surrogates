"""Controlled interface-property (TBR-proxy) uncertainty sweep."""
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

# One interface property varied at a time; everything else held at nominal.
# Not every geometry has every layer -- layers absent from a geometry are
# skipped rather than silently ignored, so a missing sweep is visible in the log.
SWEEPS = {
    'tim_top':        [80.0, 40.0, 20.0, 10.0, 5.0],    # TIM1 indium lifecycle
    'tim_sink':       [4.0, 1.0, 2.0, 6.0, 8.0],        # thermal grease spread
    'tim_bottom':     [4.0, 1.0, 2.0, 6.0, 8.0],        # grease, g1/g3 naming
    'tim_die':        [4.0, 1.0, 2.0, 6.0, 8.0],        # grease, g4 naming
    'tim_die2':       [4.0, 1.0, 2.0, 6.0, 8.0],        # grease, g2a naming
    'tim2':           [4.0, 1.0, 2.0, 6.0, 8.0],        # top-side grease
    'hybrid_bonding': [60.0, 100.0, 200.0, 300.0, 400.0],  # modelled -> real hardware
}

# Pairwise interaction probe: the single-parameter sweeps cannot see whether two
# interface uncertainties compound super- or sub-additively, which was the main
# caveat on the first pass. Run a small 3x3 grid on the two that actually moved
# the answer, rather than a full factorial over all pairs.
INTERACTION_PAIRS = [
    (('tim_sink', [1.0, 4.0, 8.0]), ('tim_top', [5.0, 20.0, 80.0])),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('data/3d-ice-interface-pilot'))
    ap.add_argument('--ice-executable',
                    default='wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator')
    ap.add_argument('--pattern', default='uniform',
                    help='Power pattern held fixed across the whole sweep')
    ap.add_argument('--power', choices=['lowest', 'highest', 'median'], default='highest',
                    help='Which scenario of that pattern to hold fixed. Interface '
                         'resistance only matters in proportion to the heat flux '
                         'crossing it, so a low-power scenario understates '
                         'sensitivity by construction -- run both to show the '
                         'effect scales with power rather than quoting one number. '
                         '"median" picks the middle-ranked-by-power scenario of that '
                         'pattern, for a "typical operating point" comparison against '
                         'the worst-case "highest" number.')
    ap.add_argument('--tag', default='', help='Suffix for scenario names (avoids '
                                              'collisions between sweeps)')
    ap.add_argument('--geometry', default='geometry5',
                    help='Which benchmark geometry to sweep')
    ap.add_argument('--interactions', action='store_true',
                    help='Also run the pairwise interaction grid (see '
                         'INTERACTION_PAIRS) to test whether two interface '
                         'uncertainties compound super- or sub-additively.')
    args = ap.parse_args()

    GEOM_NAME = args.geometry
    geom = get_geometry_by_name(GEOM_NAME)
    layer_names = {l.name for l in geom.layers}
    gen = ScenarioGenerator()

    # Take ONE scenario and reuse it for every run, so power/HTC/ambient are
    # identical by construction -- the whole point of this sweep.
    matching = [sc for sc in gen.generate_all_scenarios(geom)
                if sc.pattern == args.pattern]
    if not matching:
        raise SystemExit(f"No scenario with pattern={args.pattern!r} found")
    key = lambda sc: sum(sc.power_blocks.values())
    ranked = sorted(matching, key=key)
    if args.power == 'highest':
        base = ranked[-1]
    elif args.power == 'lowest':
        base = ranked[0]
    else:  # median
        base = ranked[len(ranked) // 2]
    log.info("Selected %s-power %r scenario: total block power %.1f W/cm^2 "
             "(of %d candidates, range %.1f-%.1f)", args.power, args.pattern, key(base),
             len(matching), key(ranked[0]), key(ranked[-1]))

    base_dict = base.to_dict()
    log.info("Fixed operating point: pattern=%s htc=%.0f t_amb=%.1fC",
             base_dict.get('pattern'), base_dict['htc'], base_dict['t_ambient'])

    output_dir = args.output / GEOM_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = NPZExporter(output_dir=output_dir)
    calculator = StatisticsCalculator()

    # Build the full run list first, so the log states up front what will run
    # and -- importantly -- which sweeps were skipped for lacking that layer.
    runs = []  # (name_fragment, {layer: k})
    for layer_name, values in SWEEPS.items():
        if layer_name not in layer_names:
            log.info("skip sweep %-16s (no such layer in %s)", layer_name, GEOM_NAME)
            continue
        for value in values:
            runs.append((f"{layer_name}_{value:g}", {layer_name: value}))

    if args.interactions:
        for (la, va), (lb, vb) in INTERACTION_PAIRS:
            if la not in layer_names or lb not in layer_names:
                log.info("skip interaction %s x %s (layer absent in %s)",
                         la, lb, GEOM_NAME)
                continue
            for a in va:
                for b in vb:
                    runs.append((f"x_{la}_{a:g}_{lb}_{b:g}", {la: a, lb: b}))

    log.info("%d runs queued for %s", len(runs), GEOM_NAME)

    ok, failed = 0, []
    for frag, overrides in runs:
        sc_dict = dict(base_dict)
        sc_dict['layer_k_overrides'] = overrides
        suffix = f"_{args.tag}" if args.tag else ""
        scenario_name = f"{GEOM_NAME}_iface_{frag}{suffix}"

        tmp = Path(tempfile.mkdtemp(prefix='ifacepilot_'))
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
            log.info("[OK] %-40s peak=%.2fC", scenario_name,
                     stats['hotspot']['peak_temperature_c'])
            ok += 1
        except Exception as exc:
            log.error("[FAILED] %s: %s", scenario_name, exc)
            failed.append(scenario_name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    log.info("Interface sweep complete: %d ok, %d failed", ok, len(failed))
    if failed:
        log.error("Failed: %s", failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
