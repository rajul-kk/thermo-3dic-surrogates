"""Re-run the feedback loop (leakage or throttling) of named pilot files under the pinned 3D-ICE and compare.
resolve_from_metadata.py refuses these files because one file is several solves, not one. The loop itself is
deterministic given the NOMINAL scenario, which the metadata stores (nominal_block_power_*, htc, ambient, leakage_fraction,
leakage_k_double_c / throttle_derate state), so each file is rebuilt by feeding that nominal scenario through the same
process_scenario() path the generators use. Parameters the metadata does not store take the generators' defaults
(leakage T_ref 85 C; throttle 95 C, gain 2.0, floor 0.3) and can be overridden.
--dry-run builds the scenarios and cross-checks them against the generator (no solve).
Usage: python scripts/resolve_feedback_pilot.py data/3d-ice-leakage-pilot/geometry1 geometry1_test_017 geometry1_train_003
       [--out <scratch dir>] [--dry-run] [--no-check-generator]
Output: <out>/<stem>.npz (default out = a new directory next to the source), one comparison line per file.
"""
import argparse
import ast
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_leakage_pilot import LEAKAGE_SETTINGS
from scripts.resolve_from_metadata import ICE_EXE, load_geometry
from src.export.npz_exporter import NPZExporter
from src.export.statistics import StatisticsCalculator
from src.main import process_scenario
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator

LEAKAGE_EXTRA_TRAIN = 25        # scripts/regen_pilots_v5.sh: --n 45 --extra-train 25 (15 train + 5 test + 25 extra)


def feedback_kind(m):
    if m.get('leakage_enabled'):
        return 'leakage'
    if m.get('throttle_enabled'):
        return 'throttle'
    raise ValueError(f"{m.get('scenario_name')}: not a feedback scenario (use resolve_from_metadata.py)")


def scenario_from_metadata(m, geom, args):
    """The NOMINAL scenario handed to apply_leakage_feedback / apply_throttling, rebuilt from the stored metadata."""
    kind = feedback_kind(m)
    try:
        nominal = {b.name: float(m[f'nominal_block_power_{b.name}']) for b in geom.power_blocks}
    except KeyError as e:
        raise ValueError(f"{m['scenario_name']}: no {e.args[0]} in the metadata; cannot recover the nominal power") from e
    sc = {'power_blocks': nominal, 'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
          'pattern': m.get('pattern', 'unknown'), 'rdl_joule_fraction': float(m.get('rdl_joule_fraction', 0.05)),
          'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')}
    if kind == 'leakage':
        sc.update(leakage_enabled=True, leakage_fraction=float(m['leakage_fraction']),
                  leakage_k_double_c=float(m['leakage_k_double_c']), leakage_t_ref_c=args.t_ref)
    else:
        sc.update(throttle_enabled=True, throttle_temp_c=args.throttle_temp, throttle_gain=args.throttle_gain,
                  throttle_power_floor=args.throttle_floor)
    return sc


def generator_scenario(stem, m, geom):
    """The scenario the generator makes for file `stem`, or None when the name is not <geometry>_<train|test>_<NNN>."""
    mt = re.fullmatch(r'.*_(train|test)_(\d+)', stem)
    if not mt:
        return None
    kind, n = mt.group(1), int(mt.group(2))
    gen = ScenarioGenerator()
    scenarios = gen.generate_all_scenarios(geom)
    if feedback_kind(m) == 'leakage':
        # gen_leakage_pilot.py numbers train and test together over [base set + extra training]
        scenarios = scenarios + gen.generate_extra_training_scenarios(
            geom, LEAKAGE_EXTRA_TRAIN, start_index=16, pool_start_idx=0)
        pick = scenarios[n - 1] if 0 < n <= len(scenarios) else None
        if pick is not None:
            frac, k_double = LEAKAGE_SETTINGS[(n - 1) % len(LEAKAGE_SETTINGS)]
            pick = (pick, {'leakage_fraction': frac, 'leakage_k_double_c': k_double})
    else:
        # src/main.py numbers train and test separately; the shipped throttle pilot is the base 15 + 5
        same = [s for s in scenarios if s.scenario_type == kind]
        pick = (same[n - 1], {}) if 0 < n <= len(same) else None
    if pick is None or pick[0].scenario_type != kind:
        return None
    return pick


def generator_mismatches(stem, m, geom, sc):
    """Differences between the metadata-built nominal scenario and the generator's scenario for the same file."""
    got = generator_scenario(stem, m, geom)
    if got is None:
        return ['file name does not map to a generator scenario']
    gs, extra = got
    gd = gs.to_dict()
    bad = []
    for k in ('htc', 't_ambient', 'pattern'):
        if (gd[k] != sc[k]) if isinstance(gd[k], str) else (abs(gd[k] - sc[k]) > 1e-9 * max(1.0, abs(gd[k]))):
            bad.append(f'{k}: generator {gd[k]!r}, metadata {sc[k]!r}')
    for name, v in gd['power_blocks'].items():
        if abs(v - sc['power_blocks'].get(name, np.nan)) > 1e-9 * max(1.0, abs(v)):
            bad.append(f'nominal power {name}: generator {v!r}, metadata {sc["power_blocks"].get(name)!r}')
    for k, v in extra.items():
        if abs(v - sc[k]) > 1e-12:
            bad.append(f'{k}: generator {v!r}, metadata {sc[k]!r}')
    return bad


def compare(old, new):
    """One-line summary of stored vs re-run feedback result."""
    om, nm = old['metadata'].item(), new['metadata'].item()
    kind = feedback_kind(om)
    keys = (['leakage_iterations', 'leakage_multiplier', 'leakage_converged', 'leakage_runaway'] if kind == 'leakage'
            else ['throttle_iterations', 'throttle_derate_factor', 'throttle_triggered'])
    meta = ' '.join(f'{k.split("_", 1)[1]}={om[k]:.6g}/{nm[k]:.6g}' if isinstance(om[k], float)
                    else f'{k.split("_", 1)[1]}={om[k]}/{nm[k]}' for k in keys)
    if old['temp'].shape != new['temp'].shape:
        return f'GRID DIFFERS {old["temp"].shape} vs {new["temp"].shape}; {meta}', None
    dt = float(np.abs(old['temp'].astype(np.float64) - new['temp'].astype(np.float64)).max())
    return f'max|dT|={dt:.3g} K; stored/new {meta}', dt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path, help='dataset directory, e.g. data/3d-ice-leakage-pilot/geometry1')
    ap.add_argument('stems', nargs='+', help='file names without .npz, e.g. geometry1_test_017')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--dry-run', action='store_true', help='build and cross-check the scenarios only; no 3D-ICE solve')
    ap.add_argument('--no-check-generator', action='store_true')
    ap.add_argument('--t-ref', type=float, default=85.0, help='leakage reference temperature, C (generator default)')
    ap.add_argument('--throttle-temp', type=float, default=95.0)
    ap.add_argument('--throttle-gain', type=float, default=2.0)
    ap.add_argument('--throttle-floor', type=float, default=0.3)
    ap.add_argument('--tol', type=float, default=0.0, help='max |dT| in K counted as a match (default: bit-identical)')
    ap.add_argument('--ice-executable', default=ICE_EXE)
    args = ap.parse_args(argv)
    out = args.out or args.src.with_name(args.src.name + '-feedback-resolve')
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for stem in args.stems:
        old = np.load(args.src / f'{stem}.npz', allow_pickle=True)
        m = old['metadata'].item()
        geom = load_geometry(m['geometry'])
        sc = scenario_from_metadata(m, geom, args)
        if not args.no_check_generator:
            bad = generator_mismatches(stem, m, geom, sc)
            print(f'{stem}: generator cross-check ' + ('OK' if not bad else 'MISMATCH ' + '; '.join(bad)), flush=True)
            failed += bool(bad)
        if args.dry_run:
            continue
        tmp = Path(tempfile.mkdtemp(prefix='fbresolve_'))
        (tmp / 'cfg').mkdir(); (tmp / 'sim').mkdir()
        try:
            process_scenario(stem, sc, geom, ICESimulator(tmp / 'cfg', tmp / 'sim', args.ice_executable),
                             NPZExporter(output_dir=out), StatisticsCalculator(), out, generate_plots=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        new = np.load(out / f'{stem}.npz', allow_pickle=True)
        line, dt = compare(old, new)
        ok = dt is not None and dt <= args.tol
        failed += not ok
        print(f'{stem}: {"MATCH" if ok else "DIFFERS"}  {line}  ice_commit={new["metadata"].item().get("ice_commit")}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
