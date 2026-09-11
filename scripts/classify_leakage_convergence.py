"""
Turn the eyeballed "total nominal power >~271W predicts leakage runaway" observation (scripts/analyze_leakage_convergence.py) into an actual fitted, cross-validated
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PATTERN_CONCENTRATION = {
    # Rough, hand-assigned concentration score (higher = more localised power),
    # standing in for "peak local flux at fixed total power" until a proper
    # per-cell power-field feature is wired in. Ordinal, not measured.
    'uniform': 0, 'random_smooth': 1, 'gradient': 2, 'checkerboard': 3,
    'hotspot': 4, 'dual_hotspot': 5, 'extreme_hotspot': 6,
}


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
            power_w=total_nominal_power(m), htc=m['htc'],
            t_amb=m.get('t_ambient_celsius', m.get('t_ambient', 25.0)),
            frac=m['leakage_fraction'], k_double=m['leakage_k_double_c'],
            concentration=PATTERN_CONCENTRATION.get(m['pattern'], 2),
            stable=bool(m['leakage_converged']),  # target: reached a steady state at all
        ))

    n = len(rows)
    X = np.array([[r['power_w'], r['htc'], r['t_amb'], r['frac'], r['k_double'],
                    r['concentration']] for r in rows])
    y = np.array([r['stable'] for r in rows], dtype=int)
    feature_names = ['power_w', 'htc', 't_amb', 'leakage_fraction', 'k_double_c',
                      'pattern_concentration']

    print(f"n={n}, {y.sum()} stable / {n - y.sum()} unstable "
          f"({100*(n-y.sum())/n:.0f}% runaway/stalled)")

    # Leave-one-out CV -- the only honest evaluation at this sample size.
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
    loo_pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
    loo_acc = (loo_pred == y).mean()
    print(f"\nFull model (all 6 features), leave-one-out CV accuracy: {loo_acc:.3f} "
          f"({(loo_pred == y).sum()}/{n})")

    # Power alone -- the single feature that looked strongest by eye.
    X_power = X[:, [0]]
    loo_pred_power = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')),
        X_power, y, cv=LeaveOneOut())
    loo_acc_power = (loo_pred_power == y).mean()
    print(f"Power-only model, leave-one-out CV accuracy:            {loo_acc_power:.3f} "
          f"({(loo_pred_power == y).sum()}/{n})")

    # Power + concentration -- power alone missed the pattern-driven boundary case.
    X_pc = X[:, [0, 5]]
    loo_pred_pc = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')),
        X_pc, y, cv=LeaveOneOut())
    loo_acc_pc = (loo_pred_pc == y).mean()
    print(f"Power + pattern-concentration model, LOO CV accuracy:   {loo_acc_pc:.3f} "
          f"({(loo_pred_pc == y).sum()}/{n})")

    # Fit on all data (not CV) to inspect which features the full model actually weights.
    clf.fit(X, y)
    coefs = clf.named_steps['logisticregression'].coef_[0]
    print("\nStandardised coefficients (full model, fit on all data -- for inspection, "
          "not for the accuracy numbers above):")
    for name, c in sorted(zip(feature_names, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:<24} {c:+.3f}")

    print(f"\nCaveat: n={n}. Leave-one-out accuracy at this sample size has wide "
          "uncertainty -- treat directional signal (which feature set wins) with caution "
          "too: at n=21 the power+concentration model looked like a clean 21/21 win over "
          "the full model, and that ranking reversed once n grew to 45 (see "
          "docs/report.md Sec 9.10). Re-run whenever --data grows.")


if __name__ == '__main__':
    main()
