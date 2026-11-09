"""Diagnostics for scripts/icb_hotspot_repro.py (CALIBRATION, not a result).

D1 shape: per-sample best affine map pred -> truth (rmse, corr): is the spatial pattern right?
D2 global affine (a, b) fitted on 5 of the 10 samples, scored on the other 5 (and swapped).
D3 one package knob (spreader conductivity) chosen on 5 samples, scored on the other 5 (and swapped).
Folds: even / odd position in SAMPLES.  Uses the identity orientation (transposed-input fix) and
variant 'at'.  The fitted quantities use released labels, so this is calibration against the
labels, not a reproduction.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icb_hotspot_repro as m  # noqa: E402
from iceforge import hotspot as ihs  # noqa: E402

KS = [400.0, 1000.0, 2000.0, 4000.0, 10000.0, 40000.0]


def predict(Q, x, k_sp):
    c, _ = m.build_case(Q, float(x[1].max()), float(x[2].max()), "at", 298.15, 0.01)
    if k_sp != 400.0:
        c.files["hotspot.config"] = re.sub(r"-k_spreader\t[^\n]*", f"-k_spreader\t{k_sp}", c.files["hotspot.config"])
    return m.parse_layer(ihs.run_hotspot(c, m.HOTSPOT)[0], 4)


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def main():
    err, _ = ihs.preflight(m.HOTSPOT)
    if err:
        sys.exit(err)
    T, X, P = {}, {}, {}
    for i in m.SAMPLES:
        x, t = m.load(i)
        T[i], X[i] = t, x
        for k in KS:
            P[(i, k)] = predict(x[0].T, x, k)
        print(i, "done", flush=True)
    out = {"d1_shape": [], "d2_affine": [], "d3_kspreader": [], "k_grid": KS}
    for i in m.SAMPLES:
        p, t = P[(i, 400.0)], T[i]
        A = np.c_[p.ravel(), np.ones(p.size)]
        co = np.linalg.lstsq(A, t.ravel(), rcond=None)[0]
        out["d1_shape"].append({"index": i, "case": i % 10, "slope": float(co[0]), "offset": float(co[1]),
                                "rmse_affine": rmse(A @ co, t.ravel()), "corr": float(np.corrcoef(p.ravel(), t.ravel())[0, 1])})
    folds = [(m.SAMPLES[0::2], m.SAMPLES[1::2]), (m.SAMPLES[1::2], m.SAMPLES[0::2])]
    for fit, test in folds:
        A = np.concatenate([np.c_[P[(i, 400.0)].ravel(), np.ones(4096)] for i in fit])
        y = np.concatenate([T[i].ravel() for i in fit])
        co = np.linalg.lstsq(A, y, rcond=None)[0]
        out["d2_affine"].append({"fit": fit, "test": test, "slope": float(co[0]), "offset": float(co[1]),
                                 "test_rmse": {i: rmse(co[0] * P[(i, 400.0)] + co[1], T[i]) for i in test}})
        pooled = {k: float(np.sqrt(np.mean([rmse(P[(i, k)], T[i]) ** 2 for i in fit]))) for k in KS}
        kbest = min(pooled, key=pooled.get)
        out["d3_kspreader"].append({"fit": fit, "test": test, "fit_rmse_by_k": pooled, "k_best": kbest,
                                    "test_rmse": {i: rmse(P[(i, kbest)], T[i]) for i in test},
                                    "test_metrics": {i: m.metrics(P[(i, kbest)], T[i]) for i in test}})
    allk = {k: {i: rmse(P[(i, k)], T[i]) for i in m.SAMPLES} for k in KS}
    out["rmse_all_samples_by_k"] = allk
    p = os.path.join(m.ROOT, "results", "icb_hotspot_repro", "diag.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("d2_affine", "d3_kspreader")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
