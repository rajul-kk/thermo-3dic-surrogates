"""Ridge-vs-leakage-feedback test: the key question this pilot exists to answer."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.baselines import load_scenario, collect_block_keys, fit_predict, aggregate

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--ridge-lambda', type=float, default=1.0)
    ap.add_argument('--min-test', type=int, default=2,
                    help='Abort if fewer than this many converged test scenarios')
    args = ap.parse_args()

    root = args.data / args.geometry
    all_files = sorted(root.glob(f'{args.geometry}_*.npz'))
    if not all_files:
        raise SystemExit(f"No files found in {root}")

    all_scenarios = [load_scenario(f) for f in all_files]

    runaway = [sc for sc in all_scenarios if sc['meta'].get('leakage_runaway')]
    not_conv = [sc for sc in all_scenarios
               if not sc['meta'].get('leakage_runaway')
               and not sc['meta'].get('leakage_converged', True)]
    converged = [sc for sc in all_scenarios
                if sc['meta'].get('leakage_converged') and not sc['meta'].get('leakage_runaway')]

    log.info("%d total: %d converged, %d runaway (excluded), %d neither (excluded)",
             len(all_scenarios), len(converged), len(runaway), len(not_conv))
    for sc in runaway:
        log.info("  excluded (runaway): %s peak=%.0fC mult=%.1fx",
                 sc['meta']['scenario_name'], sc['meta'].get('leakage_peak_temp_c', float('nan')),
                 sc['meta'].get('leakage_multiplier', float('nan')))

    train = [sc for sc in converged if '_train_' in sc['meta']['scenario_name']]
    test = [sc for sc in converged if '_test_' in sc['meta']['scenario_name']]
    log.info("converged split: %d train, %d test", len(train), len(test))

    if len(test) < args.min_test or len(train) < 3:
        raise SystemExit(
            f"Too few converged scenarios for a meaningful fit "
            f"(train={len(train)}, test={len(test)}, need >=3 train / "
            f"{args.min_test} test). Not attempting a fit on this little data -- "
            f"generate a larger or gentler pilot instead of trusting a noisy number."
        )

    block_keys = collect_block_keys(train + test)
    log.info("block_keys: %s", block_keys)

    results = fit_predict(train, test, block_keys, k=args.k, ridge_lambda=args.ridge_lambda)

    print(f"\n{args.geometry} leakage pilot -- CONVERGED scenarios only "
         f"(train={len(train)}, test={len(test)}, {len(runaway)} runaway excluded)")
    print(f"{'baseline':<10} {'MAE':>8} {'RMSE':>8} {'worst':>8} {'det.MAE':>9} "
         f"{'spat.R2':>9} {'hot_dx':>9}")
    print('-' * 68)
    for name, per_scenario in results.items():
        agg = aggregate(per_scenario)
        print(f"{name:<10} {agg['mae_K']:>8.3f} {agg['rmse_K']:>8.3f} "
             f"{agg['max_abs_err_K']:>8.3f} {agg['mae_detrended_K']:>9.3f} "
             f"{agg['spatial_r2']:>9.3f} {agg['hotspot_loc_err_um']:>9.1f}")


if __name__ == '__main__':
    main()
