"""
Quantify how much interface-property uncertainty moves the answer, and compare
that spread against the model-vs-model differences surrogate papers optimise.

The question this answers: neural thermal surrogates compete over sub-Kelvin
field-error improvements. If plausible uncertainty in the interface properties
fed to the simulator moves peak junction temperature (or hotspot location) by
MORE than those differences, the accuracy race is being run inside the noise
floor of its own inputs.

Reads the sweep produced by scripts/gen_interface_uncertainty_pilot.py, in which
power/HTC/ambient are held fixed and exactly one interface conductivity varies
per scenario. Writes results/interface_uncertainty.json.

Usage:
    python scripts/analyze_interface_uncertainty.py
"""
import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

DEFAULT_DATA_DIR = Path('data/3d-ice-interface-pilot/geometry5')

# Nominal (as-modelled, not swept) conductivity per interface layer -- must match
# the first/nominal entry of each list in gen_interface_uncertainty_pilot.py's
# SWEEPS dict. Used to find the true "everything at nominal" baseline point in an
# interaction grid, since no two swept values are ever equal to each other (an
# earlier version of this script used "do the two override values match" as a
# baseline heuristic, which is wrong whenever the two layers' sweep ranges don't
# overlap -- as here -- and silently picked an arbitrary corner instead).
NOMINAL_K = {
    'tim_top': 80.0, 'tim_sink': 4.0, 'tim_bottom': 4.0, 'tim_die': 4.0,
    'tim_die2': 4.0, 'tim2': 4.0, 'hybrid_bonding': 60.0,
}

# Sub-Kelvin field-error differences are what surrogate papers compete over; this
# repo's own measured ridge-vs-FNO detrended-MAE gap on throttled data is ~1.1 K
# (report.md 9.7). Used only as a reference line in the printed comparison.
REFERENCE_SURROGATE_GAP_K = 1.088


def load(data_dir):
    """
    Returns (single_rows, interaction_rows). A file whose layer_k_overrides has
    TWO keys (from --interactions) is an interaction run, not a single-parameter
    sweep point -- folding it into a single layer's group by taking only its
    first key (an earlier bug in this function) silently corrupts that group's
    peak-T spread with data from a different, two-variable experiment.
    """
    single_rows, interaction_rows = [], []
    for f in sorted(data_dir.glob('*.npz')):
        d = np.load(f, allow_pickle=True)
        meta = dict(d['metadata'][0])
        ko = meta.get('layer_k_overrides', '{}')
        try:
            ko = ast.literal_eval(ko) if isinstance(ko, str) else ko
        except (ValueError, SyntaxError):
            ko = {}
        if not ko:
            d.close()
            continue
        T = d['temp'].astype(np.float64)
        coords = d['coords'].astype(np.float64)
        peak_idx = int(np.argmax(T))
        common = {
            'file': f.name,
            'peak_C': float(T.max() - 273.15),
            'mean_C': float(T.mean() - 273.15),
            'hotspot_xy_um': [float(coords[peak_idx][0]), float(coords[peak_idx][1])],
            'htc': float(meta.get('htc', 0.0)),
            'pattern': str(meta.get('pattern', '')),
        }
        if len(ko) == 1:
            layer, value = next(iter(ko.items()))
            single_rows.append({**common, 'layer': layer, 'k': float(value)})
        else:
            interaction_rows.append({**common, 'overrides': {k: float(v) for k, v in ko.items()}})
        d.close()
    return single_rows, interaction_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--out', type=Path,
                    default=Path('results/interface_uncertainty.json'))
    args = ap.parse_args()

    rows, interaction_rows = load(args.data)
    if not rows and not interaction_rows:
        raise SystemExit(f"No sweep data in {args.data} -- run "
                         "scripts/gen_interface_uncertainty_pilot.py first.")

    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r['layer']].append(r)

    out = {'n_scenarios': len(rows), 'n_interaction_scenarios': len(interaction_rows),
          'by_layer': {}}
    print(f"{'interface':<18} {'k range (W/m.K)':<20} {'peak T range (C)':<22} "
          f"{'spread (K)':<12} {'hotspot moved':<14}")
    print('-' * 92)

    for layer, rs in sorted(by_layer.items()):
        rs = sorted(rs, key=lambda r: r['k'])
        peaks = [r['peak_C'] for r in rs]
        spread = max(peaks) - min(peaks)
        xys = [tuple(r['hotspot_xy_um']) for r in rs]
        moved = len(set(xys)) > 1
        max_shift = 0.0
        if moved:
            ref = np.array(xys[0])
            max_shift = float(max(np.hypot(*(np.array(xy) - ref)) for xy in xys))

        out['by_layer'][layer] = {
            'k_values': [r['k'] for r in rs],
            'peak_C': peaks,
            'peak_spread_K': spread,
            'hotspot_moved': moved,
            'max_hotspot_shift_um': max_shift,
            'scenarios': rs,
        }
        print(f"{layer:<18} {min(r['k'] for r in rs):>7.1f} - {max(r['k'] for r in rs):<10.1f} "
              f"{min(peaks):>9.2f} - {max(peaks):<10.2f} {spread:>10.2f}   "
              f"{('yes, %.0f um' % max_shift) if moved else 'no':<14}")

    if rows:
        total = max(r['peak_C'] for r in rows) - min(r['peak_C'] for r in rows)
        out['total_peak_spread_K'] = total
    out['reference_surrogate_gap_K'] = REFERENCE_SURROGATE_GAP_K

    if interaction_rows:
        # Additive-effect check: does varying two interfaces together move peak
        # T by more or less than the sum of moving each alone (from the
        # single-parameter sweep above)? Requires both layers to also have a
        # single-parameter sweep in this same dataset to compute the additive
        # prediction; otherwise reports raw interaction-grid spread only.
        print(f"\nInteraction grid ({len(interaction_rows)} scenarios):")
        peaks = [r['peak_C'] for r in interaction_rows]
        print(f"  peak T range: {min(peaks):.2f} - {max(peaks):.2f} C "
              f"(spread {max(peaks) - min(peaks):.2f} K)")
        layers_in_grid = sorted({l for r in interaction_rows for l in r['overrides']})
        print(f"  layers varied jointly: {layers_in_grid}")

        def _is_all_nominal(r):
            return all(abs(v - NOMINAL_K.get(layer, float('nan'))) < 1e-6
                      for layer, v in r['overrides'].items())

        baseline_candidates = [r for r in interaction_rows if _is_all_nominal(r)]
        if not baseline_candidates:
            print("  WARNING: no all-nominal point found in the interaction grid "
                 "(NOMINAL_K may be out of sync with SWEEPS) -- additive check skipped.")
            baseline_candidates = None
        additive_check = []
        if baseline_candidates:
            nominal_peak = baseline_candidates[0]['peak_C']
            print(f"  double-nominal baseline: {baseline_candidates[0]['overrides']} "
                 f"-> peak={nominal_peak:.2f}C")
            for r in interaction_rows:
                single_deltas = 0.0
                ok = True
                for layer, k in r['overrides'].items():
                    pts = by_layer.get(layer, [])
                    if not pts:
                        ok = False
                        break
                    nearest = min(pts, key=lambda p: abs(p['k'] - k))
                    single_deltas += nearest['peak_C'] - nominal_peak
                if ok:
                    joint_delta = r['peak_C'] - nominal_peak
                    additive_check.append({
                        'overrides': r['overrides'],
                        'joint_delta_K': joint_delta,
                        'sum_of_individual_deltas_K': single_deltas,
                        'super_additive': joint_delta > single_deltas + 0.1,
                    })
        out['interaction'] = {
            'scenarios': interaction_rows,
            'peak_spread_K': max(peaks) - min(peaks),
            'additive_check': additive_check,
        }
        if additive_check:
            n_super = sum(1 for a in additive_check if a['super_additive'])
            print(f"  super-additive points (joint > sum of individual effects): "
                 f"{n_super}/{len(additive_check)}")
            for a in additive_check:
                print(f"    {a['overrides']}: joint={a['joint_delta_K']:+.2f}K  "
                     f"sum_of_individual={a['sum_of_individual_deltas_K']:+.2f}K  "
                     f"{'SUPER-ADDITIVE' if a['super_additive'] else 'sub-additive/linear'}")

    print('-' * 92)
    print(f"Largest single-interface peak-T spread : "
          f"{max(v['peak_spread_K'] for v in out['by_layer'].values()):.2f} K")
    print(f"Reference ridge-vs-FNO det.MAE gap     : {REFERENCE_SURROGATE_GAP_K:.2f} K "
          f"(report.md 9.7, throttled pilot)")
    print(f"Operating point: pattern={rows[0]['pattern']!r} htc={rows[0]['htc']:.0f}")

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == '__main__':
    main()
