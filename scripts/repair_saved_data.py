"""Repair saved 3D-ICE NPZs for the two pipeline faults in docs/report.md §9.23 (transposed coords, layer-blind power).
Exact, no re-simulation: temperatures are kept; coords are relabelled and the power field rebuilt. Idempotent.
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.mesh import generate_power_density_field
from src.core.placement import place_chiplets
from src.scenario.generator import ScenarioGenerator

GEOMS = ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5', 'geometry6']
# --extra-train counts regen_v4_final.ps1 used for data/3d-ice (verified to reproduce every stored map)
EXTRA = {'geometry1': 25, 'geometry2a': 10, 'geometry3': 25, 'geometry4': 25, 'geometry5': 35, 'geometry6': 35}
FLAG = 'repaired_2026_09_23'


def true_coords(geom, c):
    """The old parser labelled Tmap row r (a WIDTH index) as a length position and column c as a width position."""
    ux, uy = np.unique(c[:, 0]), np.unique(c[:, 1])
    n_len, n_wid = geom.mesh_resolution[:2]
    if (len(ux), len(uy)) != (n_len, n_wid):
        raise ValueError(f'unexpected label grid {len(ux)}x{len(uy)} for mesh {n_len}x{n_wid}')
    col = np.searchsorted(ux, c[:, 0])            # length index
    row = np.searchsorted(uy, c[:, 1])            # width index
    wc = (np.arange(n_wid) + 0.5) * geom.die_width / n_wid
    lc = (np.arange(n_len) + 0.5) * geom.die_length / n_len
    return np.stack([wc[row], lc[col], c[:, 2]], axis=1)


def regenerated_maps(gname):
    """Rebuild data/3d-ice's per-cell power maps exactly as src/main.py generated them."""
    geom, gen = get_geometry_by_name(gname), ScenarioGenerator()
    base = gen.generate_all_scenarios(geom)
    gen.attach_tsv_maps(base, geom)
    gen.attach_power_maps(base, geom, kind='mixed')
    extra = gen.generate_extra_training_scenarios(geom, EXTRA[gname], start_index=16, pool_start_idx=0)
    gen.attach_tsv_maps(extra, geom, seed_base=900_000)
    gen.attach_power_maps(extra, geom, kind='mixed', seed_base=500_000)
    return {s.to_dict()['name']: s.to_dict() for s in base + extra}


def repair(f: Path, maps=None):
    d = np.load(f, allow_pickle=True)
    arrays = {k: d[k] for k in d.files}
    meta = arrays['metadata'].item()
    if meta.get(FLAG):
        return 'skip'
    base = get_geometry_by_name(meta['geometry'])
    old = arrays['coords'].astype(np.float64)
    new = true_coords(base, old)

    if maps is not None:                          # fixed dataset: per-cell maps
        s = maps[f.stem]
        check = generate_power_density_field(old, base, s['power_blocks'], power_map_by_layer=s['power_map_by_layer'])
        if np.abs(check - arrays['power']).max() > 1e-4 * np.abs(arrays['power']).max():
            raise ValueError(f'{f}: regenerated maps do not reproduce the stored power')
        power = generate_power_density_field(new, base, s['power_blocks'], power_map_by_layer=s['power_map_by_layer'])
    else:                                         # layout dataset: block scalars at the placed positions
        offs = {k[len('placement_dx_'):]: (float(meta[k]), float(meta['placement_dy_' + k[len('placement_dx_'):]]))
                for k in meta if k.startswith('placement_dx_')}
        geom = place_chiplets(base, offs)
        blocks = {b.name: float(meta.get(f'block_power_{b.name}', 0.0)) for b in geom.power_blocks}
        power = generate_power_density_field(new, geom, blocks)

    arrays['coords'] = new.astype(np.float32)
    arrays['power'] = power.astype(np.float32)
    if 'power_nominal' in arrays:
        arrays['power_nominal'] = arrays['power']
    meta[FLAG] = True
    meta['norm_coords_min'] = new.min(0).tolist()
    meta['norm_coords_max'] = new.max(0).tolist()
    meta['norm_power_mean'] = float(power.mean())
    meta['norm_power_std'] = float(power.std())
    arrays['metadata'] = np.array([meta], dtype=object)
    np.savez_compressed(f, **arrays)
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometries', nargs='*', default=GEOMS)
    args = ap.parse_args()
    for g in args.geometries:
        maps = regenerated_maps(g)
        n = {'ok': 0, 'skip': 0}
        for f in sorted(Path(f'data/3d-ice/{g}').glob('*.npz')):
            n[repair(f, maps)] += 1
        for f in sorted(Path(f'data/3d-ice-layout-{g}').rglob('*.npz')):
            n[repair(f)] += 1
        print(f'{g:<11} repaired {n["ok"]}, already done {n["skip"]}')


if __name__ == '__main__':
    main()
