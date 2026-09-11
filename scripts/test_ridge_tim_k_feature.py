"""
Follow-up to analyze_interface_linearity.py's finding that peak-T is linear in 1/k_tim, not k. That result predicts a specific, testable claim: a ridge model given 1/k_tim as an
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.baselines import metrics  # reuse the exact same detrended-MAE / spatial-R2 code

DATA_DIRS = {
    'high':   Path('data/3d-ice-interface-multi/geometry4'),
    'median': Path('data/3d-ice-interface-pilot-median/geometry4'),
}
LAYER = 'tim_die'
NAME_RE = re.compile(r'^geometry4_iface_%s_(?P<k>[\d.]+)' % LAYER)


def load_scenarios():
    rows = []
    for tag, d in DATA_DIRS.items():
        for f in sorted(d.glob(f'geometry4_iface_{LAYER}_*.npz')):
            stem = re.sub(r'__\w+$', '', f.stem)
            m = NAME_RE.match(stem)
            if not m:
                continue
            data = np.load(f, allow_pickle=True)
            meta = dict(data['metadata'][0])
            rows.append(dict(
                tag=tag, k=float(m.group('k')), meta=meta,
                temp=data['temp'].astype(np.float64),
                coords=data['coords'].astype(np.float64),
            ))
    return rows


def base_features(meta: dict, block_keys) -> list:
    htc = float(meta.get('htc', 0.0))
    feats = [float(meta.get(bk, 0.0)) for bk in block_keys]
    feats.append(htc)
    feats.append(1.0 / htc if htc > 0 else 0.0)
    feats.append(float(meta.get('t_ambient_kelvin', 298.15)))
    feats.append(float(meta.get('tsv_density', 0.0)))
    return feats


def build_X(rows, block_keys, variant):
    X = []
    for r in rows:
        feats = base_features(r['meta'], block_keys)
        if variant == '+k':
            feats.append(r['k'])
        elif variant == '+1/k':
            feats.append(1.0 / r['k'])
        X.append(feats)
    return np.asarray(X, dtype=np.float64)


def ridge_loo(rows, block_keys, variant, ridge_lambda=1.0):
    """Leave-one-scenario-out ridge, same math as scripts/baselines.py fit_predict."""
    X_all = build_X(rows, block_keys, variant)
    Y_all = np.stack([r['temp'] for r in rows])
    n = len(rows)
    per_scenario = []
    for held in range(n):
        tr = [i for i in range(n) if i != held]
        X_tr, Y_tr = X_all[tr], Y_all[tr]
        mu, sigma = X_tr.mean(0), X_tr.std(0)
        sigma[sigma < 1e-12] = 1.0
        Z_tr = (X_tr - mu) / sigma
        A = np.hstack([Z_tr, np.ones((len(tr), 1))])
        reg = ridge_lambda * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(A.T @ A + reg, A.T @ Y_tr)

        x = (X_all[held] - mu) / sigma
        pred = np.append(x, 1.0) @ W
        m = metrics(pred, rows[held]['temp'], rows[held]['coords'])
        m['k'] = rows[held]['k']
        m['tag'] = rows[held]['tag']
        per_scenario.append(m)
    return per_scenario


def main():
    rows = load_scenarios()
    if len(rows) < 4:
        raise SystemExit(f"Only found {len(rows)} scenarios for geometry4/{LAYER} -- "
                          "expected 10 (5 k-values x 2 power levels).")
    block_keys = sorted({k for r in rows for k in r['meta']
                          if k.startswith('block_power_')})
    print(f"{len(rows)} scenarios, {len(block_keys)} block-power features, "
          f"k values: {sorted(set(r['k'] for r in rows))}\n")

    for variant in ('blind', '+k', '+1/k'):
        per = ridge_loo(rows, block_keys, variant)
        mae = np.mean([m['mae_detrended_K'] for m in per])
        r2 = np.mean([m['spatial_r2'] for m in per])
        worst = max(per, key=lambda m: m['mae_detrended_K'])
        print(f"ridge [{variant:<5}]  mean det.MAE={mae:6.3f} K   mean spatial R2={r2:6.3f}"
              f"   worst: k={worst['k']:.0f} ({worst['tag']}) det.MAE={worst['mae_detrended_K']:.3f} K")
        for m in sorted(per, key=lambda m: (m['tag'], m['k'])):
            print(f"    k={m['k']:>4.0f} ({m['tag']:<6})  det.MAE={m['mae_detrended_K']:6.3f} K"
                  f"   spatial_R2={m['spatial_r2']:6.3f}   hotspot_err_K={m['hotspot_temp_err_K']:+6.2f}")
        print()

    print("If '+1/k' driving det.MAE and spatial R2 close to the mean/blind baseline's "
          "IN-distribution numbers (compare against Sec 9.1's ~0.2-1 K det.MAE, R2>0.9 on "
          "the standard dataset) while 'blind' stays large, that confirms giving ridge the "
          "physically-motivated feature recovers the interface-uncertainty axis -- and that "
          "'+k' underperforms '+1/k' confirms the reciprocal, not the raw value, is what "
          "actually enters linearly.")


if __name__ == '__main__':
    main()
