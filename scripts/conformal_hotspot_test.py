"""Split conformal intervals for peak temperature, nested inside 5-fold CV (§9.20)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.baselines import load_scenario, collect_block_keys, predict_all
from scripts.hotspot_eval import kfold_indices

TARGETS = [0.80, 0.90]
SEED = 0
FOLDS = 5
CALIB_FRAC = 0.35


def conformal_half_width(residuals: np.ndarray, target_coverage: float) -> float:
    """Finite-sample-corrected quantile for a TARGET COVERAGE (not a miscoverage rate --
    TARGETS below already IS the desired coverage, e.g. 0.90 for 90% coverage)."""
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
    return float(np.quantile(residuals, q_level, method='higher'))


def run(geom: str, root: Path, model_name: str, include_field_linear: bool):
    files = sorted(root.rglob(f'{geom}_*.npz'))
    scen = [load_scenario(f) for f in files]
    bk = collect_block_keys(scen)
    n = len(scen)

    coverage = {a: [] for a in TARGETS}
    widths = {a: [] for a in TARGETS}
    rng = np.random.default_rng(SEED)

    for fold in kfold_indices(n, FOLDS, SEED):
        keep = set(fold.tolist())
        test_idx = fold
        train_idx = np.array([i for i in range(n) if i not in keep])

        # Split TRAIN into proper-train + calibration; the point predictor never sees calib.
        perm = rng.permutation(len(train_idx))
        n_cal = max(3, int(round(CALIB_FRAC * len(train_idx))))
        cal_idx = train_idx[perm[:n_cal]]
        fit_idx = train_idx[perm[n_cal:]]

        fit_set = [scen[i] for i in fit_idx]
        cal_set = [scen[i] for i in cal_idx]
        test_set = [scen[i] for i in test_idx]

        cal_preds = predict_all(fit_set, cal_set, bk, 3, 1.0, 0, include_field_linear=include_field_linear)
        cal_resid = np.array([abs(cal_preds[j][model_name].max() - cal_set[j]['temp'].max())
                              for j in range(len(cal_set))])

        test_preds = predict_all(fit_set, test_set, bk, 3, 1.0, 0, include_field_linear=include_field_linear)
        test_pred_peak = np.array([test_preds[j][model_name].max() for j in range(len(test_set))])
        test_true_peak = np.array([s['temp'].max() for s in test_set])

        for alpha in TARGETS:
            w = conformal_half_width(cal_resid, alpha)
            hit = np.abs(test_pred_peak - test_true_peak) <= w
            coverage[alpha].extend(hit.tolist())
            widths[alpha].append(w)

    print(f'\n=== {geom} / {model_name} (n={n}) ===')
    print(f'{"nominal":>9} {"empirical":>10} {"mean half-width (K)":>20} {"n test":>8}')
    for alpha in TARGETS:
        emp = np.mean(coverage[alpha])
        print(f'{alpha:>9.0%} {emp:>10.1%} {np.mean(widths[alpha]):>20.2f} {len(coverage[alpha]):>8}')
    return coverage


def main():
    for geom, root in [('geometry1', Path('data/3d-ice')), ('geometry6', Path('data/3d-ice'))]:
        run(geom, root, 'ridge', include_field_linear=False)
    print('\n' + '=' * 60)
    print('SHARP shelf geometry, with the field-linear model too:')
    run('geometry1', Path('data/3d-ice-layout-geometry1'), 'ridge', include_field_linear=True)
    run('geometry1', Path('data/3d-ice-layout-geometry1'), 'linear_field', include_field_linear=True)


if __name__ == '__main__':
    main()
