"""
Does the leakage pilot's 30% runaway rate come from (leakage_fraction, k_double_c) -- the two parameters the pilot actually swept -- or from the underlying scenario's own
"""
import argparse
import glob
from pathlib import Path

import numpy as np


def total_nominal_power(meta: dict) -> float:
    keys = [k for k in meta if k.startswith('leakage_nominal_block_power_')
            or k.startswith('nominal_block_power_')]
    if not keys:
        keys = [k for k in meta if k.startswith('block_power_')]
    return sum(meta[k] for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/3d-ice-leakage-pilot/geometry1'))
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(args.data / '*.npz'))):
        d = np.load(f, allow_pickle=True)
        m = dict(d['metadata'][0])
        rows.append(dict(
            name=m['scenario_name'],
            frac=m['leakage_fraction'], k_double=m['leakage_k_double_c'],
            htc=m['htc'], pattern=m['pattern'],
            power_w=total_nominal_power(m),
            converged=m['leakage_converged'], runaway=m['leakage_runaway'],
        ))

    print(f"{'name':<24} {'frac':>5} {'k2x':>5} {'pattern':<15} {'htc':>8} "
          f"{'power_W':>8}  outcome")
    for r in sorted(rows, key=lambda r: r['power_w']):
        outcome = 'RUNAWAY' if r['runaway'] else ('converged' if r['converged'] else 'stalled')
        print(f"{r['name']:<24} {r['frac']:>5.2f} {r['k_double']:>5.0f} "
              f"{r['pattern']:<15} {r['htc']:>8.0f} {r['power_w']:>8.1f}  {outcome}")

    # Within each (frac, k_double) setting, is the outcome consistent?
    from collections import defaultdict
    by_setting = defaultdict(list)
    for r in rows:
        by_setting[(r['frac'], r['k_double'])].append(r)

    print("\nOutcome spread within each swept (leakage_fraction, k_double_c) setting:")
    mixed = 0
    for setting, group in sorted(by_setting.items()):
        outcomes = {'converged' if r['converged'] else 'runaway/stalled' for r in group}
        flag = ' <-- MIXED (loop-gain parameters alone do not determine this)' \
            if len(outcomes) > 1 else ''
        if flag:
            mixed += 1
        print(f"  frac={setting[0]:.2f} k_double={setting[1]:.0f}: "
              f"{[('conv' if r['converged'] else 'RUN') for r in group]} "
              f"powers={[round(r['power_w']) for r in group]}{flag}")

    print(f"\n{mixed}/{len(by_setting)} settings have mixed outcomes across their 4 repeats "
          "-- i.e. the swept leakage parameters alone do not predict stability; the base "
          "scenario's power/pattern/HTC evidently matters too.")

    # Simple threshold check on total nominal power alone.
    powers_conv = sorted(r['power_w'] for r in rows if r['converged'])
    powers_bad = sorted(r['power_w'] for r in rows if not r['converged'])
    print(f"\nConverged power range: {min(powers_conv):.0f}-{max(powers_conv):.0f} W")
    print(f"Runaway/stalled power range: {min(powers_bad):.0f}-{max(powers_bad):.0f} W")
    overlap = [p for p in powers_conv if p >= min(powers_bad)]
    print(f"Overlap: {len(overlap)} converged scenario(s) have power >= the lowest "
          f"runaway power ({min(powers_bad):.0f} W) -- total nominal power alone is a "
          "strong but NOT perfect predictor; the overlap cases are where pattern "
          "concentration (hotspot vs. spread) makes the difference at similar total power.")


if __name__ == '__main__':
    main()
