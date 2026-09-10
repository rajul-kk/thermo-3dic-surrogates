"""
Why is our benchmark linear-solvable and IC-ThermBench's not? (docs/report.md 9.14)

Two diagnostics, both structural properties of the data rather than of any model:

1. Error decomposition -- does the linear model's error concentrate at hotspots, and does it
   concentrate MORE on IC-ThermBench than on ours? (Answer: no. 2.1x vs 2.4x. The
   "their benchmark scores hotspots, which is ridge's weakness" hypothesis is refuted.)

2. Source-support and domain-scale variation -- see analyze_source_variation() below and
   docs/report.md 9.14 for the numbers that actually explain the difference: our heat sources
   never move (one support pattern per geometry, IoU 1.000) while theirs move per sample
   (IoU 0.449), and their physical cell pitch varies ~2x across samples while ours is fixed.

Run: python scripts/analyze_operator_variation.py
"""

import sys, glob, numpy as np
from pathlib import Path
sys.path.insert(0, r'D:/Work/3D-ICE Thermal-modelling/Thermo')

TOP = 0.01


def decomp(pred, true, label):
    """Split squared error and signal variance into top-1% hottest true cells vs rest."""
    err2_hot = err2_cold = var_hot = var_cold = 0.0
    n_hot = 0
    for p, t in zip(pred, true):
        k = max(1, int(round(TOP * t.size)))
        idx = np.argpartition(t, -k)[-k:]
        mask = np.zeros(t.size, bool); mask[idx] = True
        e2 = (p - t) ** 2
        d2 = (t - t.mean()) ** 2
        err2_hot += e2[mask].sum();  err2_cold += e2[~mask].sum()
        var_hot += d2[mask].sum();   var_cold += d2[~mask].sum()
        n_hot = k
    tot_e, tot_v = err2_hot + err2_cold, var_hot + var_cold
    frac_cells = n_hot / true.shape[1]
    print(f'{label:<34} top-1% cells hold {100*var_hot/tot_v:5.1f}% of signal variance '
          f'and absorb {100*err2_hot/tot_e:5.1f}% of squared error  '
          f'(concentration {(err2_hot/tot_e)/frac_cells:5.1f}x)')


# ---- IC-ThermBench S2, ridge (PCA variant, cheap) ----------------------------------
from scripts.ic_thermbench_data import load_scope, spatial_channel_indices
from scripts.ic_thermbench_baselines import (flatten_fields, fit_pca, standardise,
                                             with_intercept, ridge_gram, ridge_solve_gram)
s = load_scope(Path(r'D:/Work/3D-ICE Thermal-modelling/Thermo/data/ic-thermbench/datasets'),
               'level2')
sp = spatial_channel_indices(s.channels)
def feats(x, fits=None):
    cols, out = [], {}
    for n in sp:
        f = flatten_fields(x, sp[n])
        if fits is None:
            mu, b = fit_pca(f, 256); out[n] = (mu, b)
        else:
            mu, b = fits[n]
        cols.append((f - mu) @ b.T)
    return np.hstack(cols), out
F_tr, fits = feats(s.x_train)
F_te, _ = feats(s.x_test, fits)
y_tr = s.y_train.reshape(len(s.y_train), -1).astype(np.float64)
y_te = s.y_test.reshape(len(s.y_test), -1).astype(np.float64)
mu, sd = standardise(F_tr)
A_tr = with_intercept((F_tr - mu) / sd); A_te = with_intercept((F_te - mu) / sd)
W = ridge_solve_gram(*ridge_gram(A_tr, y_tr), 1e-4)
decomp(A_te @ W, y_te, 'IC-ThermBench S2 / ridge')
decomp(np.repeat(y_tr.mean(0)[None], len(y_te), 0), y_te, 'IC-ThermBench S2 / mean field')

# ---- this project's geometry1, ridge ----------------------------------------------
from scripts.baselines import load_scenario, collect_block_keys, predict_all
tr = [Path(p) for p in sorted(glob.glob(r'D:/Work/3D-ICE Thermal-modelling/Thermo/data/3d-ice/**/geometry1_train_*.npz', recursive=True))]
te = [Path(p) for p in sorted(glob.glob(r'D:/Work/3D-ICE Thermal-modelling/Thermo/data/3d-ice/**/geometry1_test_*.npz', recursive=True))]
train = [load_scenario(f) for f in tr]; test = [load_scenario(f) for f in te]
bk = collect_block_keys(train + test)
preds = predict_all(train, test, bk, 3, 1.0, 0)
P = np.stack([p['ridge'] for p in preds]); T = np.stack([sc['temp'] for sc in test])
decomp(P, T, 'this project geometry1 / ridge')
decomp(np.stack([p['mean'] for p in preds]), T, 'this project geometry1 / mean field')


# ---- source-support and domain-scale variation -------------------------------------
def analyze_source_variation():
    """Do the heat sources move, and is the physical grid the same across samples?"""
    import hashlib
    print()
    print('--- heat-source support: does it move between samples? ---')
    for geom in ['geometry1', 'geometry4', 'geometry6', 'geometry7']:
        pat = ('data/3d-ice-geometry7-pilot/geometry7/*.npz' if geom == 'geometry7'
               else f'data/3d-ice/**/{geom}_*.npz')
        fs = sorted(glob.glob(str(Path(r'D:/Work/3D-ICE Thermal-modelling/Thermo') / pat),
                              recursive=True))
        if not fs:
            continue
        m = np.array([np.load(f, allow_pickle=True)['power'].ravel() > 0 for f in fs])
        hashes = {hashlib.md5(r.tobytes()).hexdigest() for r in m}
        n = min(60, len(m))
        inter = (m[:n, None, :] & m[None, :n, :]).sum(-1)
        union = (m[:n, None, :] | m[None, :n, :]).sum(-1)
        iou = inter / np.maximum(union, 1)
        print(f'  {geom:<11} {len(fs):3d} scenarios -> {len(hashes):3d} distinct supports, '
              f'mean pairwise IoU {iou[~np.eye(n, dtype=bool)].mean():.3f}')

    sp2 = spatial_channel_indices(s.channels)
    supp = [hashlib.md5((s.x_train[i, ..., sp2['chiplet_power']] > 0).tobytes()).hexdigest()
            for i in range(300)]
    print(f'  IC-ThermBench S2  300 samples -> {len(set(supp))} distinct supports')

    print()
    print('--- physical grid: is cell (i,j) the same physical place in every sample? ---')
    gx = s.x_train[..., sp2['grid_x']][:, :, :, 0]
    for i in (0, 1, 7):
        pitch = np.diff(np.unique(np.round(gx[i], 4)))
        print(f'  sample {i}: uniform pitch {pitch[0]:.3f} mm/cell')
    print('  => the 64x64 array is a NORMALISED grid over physically different-sized '
          'packages;\n     a linear map over array indices compares incommensurable domains.')


analyze_source_variation()
