"""
Track C found that peak temperature is highly sensitive to TIM/interface k -- up to 75.98K on geometry4. But *sensitive* isn't the same question this
"""
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIRS = [
    Path('data/3d-ice-interface-multi'),
    Path('data/3d-ice-interface-pilot-hi'),
    Path('data/3d-ice-interface-pilot'),
    Path('data/3d-ice-interface-pilot-median'),
]

# geometry -> layer -> [(k, peak_T_C, source_tag)]
# Trailing "__<tag>" suffix (e.g. gen_interface_uncertainty_pilot.py's --tag) is
# stripped before matching, not required to appear before the layer name.
NAME_RE = re.compile(r'^(?P<geom>geometry\d[a-z]?)_iface_(?P<layer>tim_top|tim_sink|'
                      r'tim_bottom|tim_die2|tim_die|tim2|hybrid_bonding)_(?P<k>[\d.]+)$')


def r2_of_linear_fit(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0:
        return float('nan')
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def main():
    points = defaultdict(list)  # (geom, layer, tag) -> [(k, peakT)]

    for d in DATA_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob('*/*.npz')):
            stem = re.sub(r'__\w+$', '', f.stem)  # strip trailing __<tag>, if any
            m = NAME_RE.match(stem)
            if not m:
                continue
            data = np.load(f, allow_pickle=True)
            temp = data['temp'] if 'temp' in data else data['temperature']
            peak_c = float(temp.max()) - 273.15
            k = float(m.group('k'))
            # Tag by source directory, NOT just filename -- 'pilot' (low-power) and
            # 'pilot-hi' (high-power) both produce identically-named files for the
            # same geometry/layer/k, and silently merging two different fixed
            # operating points into one regression corrupts the linear fit (each
            # operating point has its own additive offset).
            tag = d.name
            points[(m.group('geom'), m.group('layer'), tag)].append((k, peak_c))

    print(f"{'geometry':<12} {'layer':<14} {'tag':<8} {'n':>3}  "
          f"{'R2(T vs k)':>11}  {'R2(T vs 1/k)':>13}  {'spread_K':>9}  winner")
    print('-' * 95)
    for (geom, layer, tag), pts in sorted(points.items()):
        pts = sorted(set(pts))
        if len(pts) < 3:
            continue
        ks = np.array([p[0] for p in pts])
        ts = np.array([p[1] for p in pts])
        r2_k = r2_of_linear_fit(ks, ts)
        r2_inv_k = r2_of_linear_fit(1.0 / ks, ts)
        spread = ts.max() - ts.min()
        winner = '1/k' if (r2_inv_k == r2_inv_k and r2_inv_k > r2_k) else 'k'
        print(f"{geom:<12} {layer:<14} {tag:<8} {len(pts):>3}  "
              f"{r2_k:>11.4f}  {r2_inv_k:>13.4f}  {spread:>9.2f}  {winner}")

    print("\nInterpretation: if R2(T vs 1/k) is consistently >= R2(T vs k) and close to "
          "1.0, the interface-uncertainty axis is linear in the SAME sense the rest of "
          "this benchmark is -- just linear in 1/k rather than in k directly. A ridge "
          "baseline extended with a 1/k_tim feature (mirroring the existing 1/htc feature "
          "in scripts/baselines.py) should then fit it near-exactly, the same way it "
          "already fits 1/htc.")


if __name__ == '__main__':
    main()
