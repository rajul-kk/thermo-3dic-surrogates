"""Does explicit geometry conditioning (PI-GANO-style: a per-cell signed-distance-to-block
field, computed from the TRUE per-scenario placement) help a linear model on layout-varying
data? Tests problem statement 7 (arXiv:2408.01600, PI-GANO) as a fast linear ablation rather
than training a full neural operator: same linear_field_cv machinery as Sec 9.15d, with an
extra input channel appended.

Critical fix vs. the existing src/fno FNODataset.use_geometry_field option: that path looks
up geom_obj = self.geometries.get(meta['geometry']) -- the NOMINAL geometry, identical for
every scenario. On shelf data this is WRONG (it would condition on where blocks are NOT),
so it is not reused here. Instead, per-scenario offsets are read from
metadata['placement_dx_<footprint>']/['placement_dy_<footprint>'], applied via
place_chiplets() to get the geometry actually simulated, and the SDF is computed from that.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import place_chiplets
from src.core.mesh import generate_distance_to_power_block_field
from scripts.baselines import load_scenario
from scripts.hotspot_eval import kfold_indices
from scripts.layout_cv import per_scenario_stats

FOLDS, SEED, LAM, PCA_K = 5, 0, 1e-2, 8


def scenario_offsets(meta: dict, footprint_names) -> dict:
    return {name: (float(meta.get(f'placement_dx_{name}', 0.0)),
                   float(meta.get(f'placement_dy_{name}', 0.0)))
            for name in footprint_names}


def per_scenario_sdf(geom, files):
    """True per-scenario SDF field, from the placement actually simulated."""
    prints = getattr(geom, 'die_footprints', None) or []
    names = [d.name for d in prints]
    sdfs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = d['metadata'].item()
        offs = scenario_offsets(meta, names)
        placed = place_chiplets(geom, offs)
        sdfs.append(generate_distance_to_power_block_field(d['coords'].astype(np.float64), placed))
    return np.stack(sdfs)


def field_cv(X, Y, pca_k, lam):
    """Economy-SVD PCA-ridge, same recipe as Sec 9.15d's linear_field_cv."""
    rows = []
    for fold in kfold_indices(len(Y), FOLDS, SEED):
        keep = set(fold.tolist())
        tr = np.array([i for i in range(len(Y)) if i not in keep])
        te = fold
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xc = (Xtr - mu) / sd
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(pca_k, Vt.shape[0])
        B = Vt[:k].T
        A = np.hstack([Xc @ B, np.ones((len(tr), 1))])
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y[tr])
        F = np.hstack([((Xte - mu) / sd) @ B, np.ones((len(te), 1))])
        pred = F @ W
        for j, i in enumerate(te):
            rows.append(per_scenario_stats(Y[i], pred[j])['r2'])
    return float(np.mean(rows)), float(np.median(rows))


def main():
    print(f'{"geometry":<11} {"n":>3} {"field-only R2":>14} {"field+SDF R2":>13} {"delta":>8}')
    print('-' * 54)
    for geom_name in ['geometry4', 'geometry5', 'geometry6']:
        geom = get_geometry_by_name(geom_name)
        files = sorted(Path(f'data/3d-ice-layout-{geom_name}').rglob(f'{geom_name}_*.npz'))
        scen = [load_scenario(f) for f in files]
        Y = np.stack([s['temp'] for s in scen])
        Xpower = np.stack([s['power'] for s in scen])

        r2_field, med_field = field_cv(Xpower, Y, PCA_K, LAM)

        sdf = per_scenario_sdf(geom, files)
        Xcombo = np.hstack([Xpower, sdf])
        r2_combo, med_combo = field_cv(Xcombo, Y, PCA_K, LAM)

        print(f'{geom_name:<11} {len(scen):>3} {r2_field:>14.4f} {r2_combo:>13.4f} '
              f'{r2_combo - r2_field:>+8.4f}')
        print(f'{"":<11} {"":>3} {"(median " + f"{med_field:.3f})":>14} '
              f'{"(median " + f"{med_combo:.3f})":>13}')


if __name__ == '__main__':
    main()
