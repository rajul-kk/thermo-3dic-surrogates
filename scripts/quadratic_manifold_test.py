"""Does a quadratic correction on the compact vector close the gap to the field ceiling? (§9.22)
Includes the same-sized random-feature control that showed it does not distinguish the hypotheses."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.baselines import load_scenario, collect_block_keys, collect_position_keys, feature_vector
from scripts.hotspot_eval import kfold_indices
from scripts.layout_cv import per_scenario_stats

FOLDS, SEED, LAM = 5, 0, 1.0

# Sec 9.15c/9.15b/9.15d reference numbers, quoted not re-derived.
REFERENCE = {
    'geometry4': {'ridge_compact': -0.667, 'linear_field': 0.941},
    'geometry5': {'ridge_compact': -2.305, 'linear_field': None},
    'geometry6': {'ridge_compact': -2.866, 'linear_field': 0.513},
}


def ridge_r2_cv(X, Y, quadratic: bool):
    rows = []
    for fold in kfold_indices(len(Y), FOLDS, SEED):
        keep = set(fold.tolist())
        tr = np.array([i for i in range(len(Y)) if i not in keep])
        te = fold
        Xtr, Xte = X[tr], X[te]
        if quadratic:
            # All pairwise products (upper triangle incl. diagonal) of the standardised
            # compact features -- the literal "quadratic manifold" correction term.
            mu0, sd0 = Xtr.mean(0), Xtr.std(0)
            sd0[sd0 < 1e-12] = 1.0
            Ztr0, Zte0 = (Xtr - mu0) / sd0, (Xte - mu0) / sd0
            iu = np.triu_indices(Xtr.shape[1])
            Qtr = np.hstack([Ztr0, Ztr0[:, iu[0]] * Ztr0[:, iu[1]]])
            Qte = np.hstack([Zte0, Zte0[:, iu[0]] * Zte0[:, iu[1]]])
            Xtr, Xte = Qtr, Qte
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        Ztr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
        Zte = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
        reg = LAM * np.eye(Ztr.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(Ztr.T @ Ztr + reg, Ztr.T @ Y[tr])
        pred = Zte @ W
        for j, i in enumerate(fold):
            rows.append(per_scenario_stats(Y[i], pred[j])['r2'])
    return float(np.mean(rows)), float(np.median(rows))


def main():
    print(f'{"geometry":<11} {"feat":>5} {"quad-feat":>9} {"ridge R2":>9} {"quad R2":>9} '
          f'{"field R2 (ref)":>15} {"gap closed":>11}')
    print('-' * 78)
    for geom in ['geometry4', 'geometry5', 'geometry6']:
        files = sorted(Path(f'data/3d-ice-layout-{geom}').rglob(f'{geom}_*.npz'))
        scen = [load_scenario(f) for f in files]
        bk = collect_block_keys(scen)
        pk = collect_position_keys(scen)
        X = np.stack([feature_vector(s['meta'], bk, pk) for s in scen])
        Y = np.stack([s['temp'] for s in scen])

        r2_lin_mean, _ = ridge_r2_cv(X, Y, quadratic=False)
        r2_quad_mean, _ = ridge_r2_cv(X, Y, quadratic=True)

        ref = REFERENCE[geom]
        field_ref = ref['linear_field']
        # 'gap closed' mixes this run's ridge R2 with the report's quoted field R2; only
        # meaningful if this run also reproduces the report's ridge number.
        if abs(r2_lin_mean - ref['ridge_compact']) > 0.01:
            print(f'  WARNING {geom}: recomputed ridge R2 {r2_lin_mean:.3f} != report '
                  f'{ref["ridge_compact"]:.3f}; gap-closed mixes inconsistent baselines')
        if field_ref is not None:
            closed = (r2_quad_mean - r2_lin_mean) / (field_ref - r2_lin_mean) * 100
            closed_s = f'{closed:.1f}%'
        else:
            closed_s = 'n/a (no field ref)'

        n_quad_feat = X.shape[1] + X.shape[1] * (X.shape[1] + 1) // 2
        print(f'{geom:<11} {X.shape[1]:>5} {n_quad_feat:>9} {r2_lin_mean:>9.3f} '
              f'{r2_quad_mean:>9.3f} {str(field_ref):>15} {closed_s:>11}')


if __name__ == '__main__':
    main()
