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

# Sub-Kelvin field-error differences are what surrogate papers compete over; this
# repo's own measured ridge-vs-FNO detrended-MAE gap on throttled data is ~1.1 K
# (report.md 9.7). Used only as a reference line in the printed comparison.
REFERENCE_SURROGATE_GAP_K = 1.088


def load(data_dir):
    rows = []
    for f in sorted(data_dir.glob('*.npz')):
        d = np.load(f, allow_pickle=True)
        meta = dict(d['metadata'][0])
        ko = meta.get('layer_k_overrides', '{}')
        try:
            ko = ast.literal_eval(ko) if isinstance(ko, str) else ko
        except (ValueError, SyntaxError):
            ko = {}
        if not ko:
            continue
        layer, value = next(iter(ko.items()))
        T = d['temp'].astype(np.float64)
        coords = d['coords'].astype(np.float64)
        peak_idx = int(np.argmax(T))
        rows.append({
            'file': f.name,
            'layer': layer,
            'k': float(value),
            'peak_C': float(T.max() - 273.15),
            'mean_C': float(T.mean() - 273.15),
            'hotspot_xy_um': [float(coords[peak_idx][0]), float(coords[peak_idx][1])],
            'htc': float(meta.get('htc', 0.0)),
            'pattern': str(meta.get('pattern', '')),
        })
        d.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--out', type=Path,
                    default=Path('results/interface_uncertainty.json'))
    args = ap.parse_args()

    rows = load(args.data)
    if not rows:
        raise SystemExit(f"No sweep data in {args.data} -- run "
                         "scripts/gen_interface_uncertainty_pilot.py first.")

    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r['layer']].append(r)

    out = {'n_scenarios': len(rows), 'by_layer': {}}
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

    total = max(r['peak_C'] for r in rows) - min(r['peak_C'] for r in rows)
    out['total_peak_spread_K'] = total
    out['reference_surrogate_gap_K'] = REFERENCE_SURROGATE_GAP_K

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
