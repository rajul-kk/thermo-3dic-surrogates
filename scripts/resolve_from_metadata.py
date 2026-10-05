"""Re-solve a saved dataset under the current ICE wrapper, reusing each file's own scenario.
Geometry, placement, block powers, HTC, ambient and k-overrides come from each file's metadata,
so no random draw has to be reproduced. Only for block-scalar data with no feedback loop
(leakage/throttle) and no TSV or power maps. Writes to a new directory for validation first.
Usage: python scripts/resolve_from_metadata.py data/3d-ice-layout-geometry4 [--out <dir>] [--snap]
--snap moves every chiplet to the nearest 3D-ICE cell boundary first (src/core/placement.py::snap_offsets), which
removes the die-edge heated-insulator artefact of off-grid placements (2026-09-30); block metadata is rewritten to match.
"""
import argparse
import ast
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.mesh import generate_power_density_field
from src.core.placement import place_chiplets, snap_offsets
from src.simulators.ice_simulator import ICESimulator

ICE_EXE = 'wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--snap', action='store_true', help='snap chiplet offsets to the 3D-ICE cell grid first')
    args = ap.parse_args()
    src = args.src
    out = args.out or src.with_name(src.name + '-v5')
    files = sorted(src.rglob('*.npz'))
    for i, f in enumerate(files):
        target = out / f.relative_to(src)
        if target.exists():
            continue
        d = np.load(f, allow_pickle=True)
        arrays = {k: d[k] for k in d.files}
        m = arrays['metadata'].item()
        if m.get('leakage_enabled') or m.get('throttle_enabled'):
            raise ValueError(f'{f}: feedback scenarios need their generator, not a re-solve')
        offs = {k[len('placement_dx_'):]: (float(m[k]), float(m['placement_dy_' + k[len('placement_dx_'):]]))
                for k in m if k.startswith('placement_dx_')}
        offs = {('' if k == 'blocks' else k): v for k, v in offs.items()}   # exporter writes '' as 'blocks'
        base = get_geometry_by_name(m['geometry'])
        if args.snap and offs:
            snapped = snap_offsets(base, offs, margin_um=0.0)
            if snapped is None:
                raise ValueError(f'{f}: no grid-aligned placement within one cell of the saved one')
            for k, (dx, dy) in snapped.items():
                key = k or 'blocks'
                # NOT 'placement_dx_orig_*': consumers parse every 'placement_dx_' key as a die offset
                m[f'orig_placement_dx_{key}'], m[f'orig_placement_dy_{key}'] = m[f'placement_dx_{key}'], m[f'placement_dy_{key}']
                m[f'placement_dx_{key}'], m[f'placement_dy_{key}'] = float(dx), float(dy)
            offs = snapped
            m['placement_snapped_2026_09_30'] = True
        geom = place_chiplets(base, offs)
        for b in geom.power_blocks:                      # positions are model features: keep them in sync
            m[f'block_x_{b.name}'], m[f'block_y_{b.name}'] = float(b.x), float(b.y)
        blocks = {b.name: float(m.get(f'block_power_{b.name}', 0.0)) for b in geom.power_blocks}
        scen = {'power_blocks': blocks, 'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
                'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')}
        with tempfile.TemporaryDirectory() as td:
            cfg, res = Path(td) / 'c', Path(td) / 'o'
            cfg.mkdir(); res.mkdir()
            parsed = ICESimulator(cfg, res, ICE_EXE).simulate(geom, scen, f.stem)
        coords = parsed['coords'].astype(np.float64)
        temp = parsed['temperature'].astype(np.float64)
        if 'tsv_frac' in arrays and np.abs(arrays['tsv_frac']).max() > 0:
            raise ValueError(f'{f}: nonzero tsv_frac cannot be carried to a new grid')
        # the grid may differ from the saved one (geometry4/5 meshes were corrected), so
        # every per-point array is rebuilt on 3D-ICE's own nodes
        power = generate_power_density_field(coords, geom, blocks)
        arrays['coords'] = coords.astype(np.float32)
        arrays['temp'] = temp.astype(np.float32)
        arrays['power'] = power.astype(np.float32)
        arrays['layer'] = np.array([geom.get_layer_index_at_z(z) for z in coords[:, 2]], dtype=np.int32)
        if 'power_nominal' in arrays:
            arrays['power_nominal'] = arrays['power']
        if 'tsv_frac' in arrays:
            arrays['tsv_frac'] = np.zeros(len(coords), np.float32)
        m['num_points'] = len(coords)
        m['mesh_resolution'] = tuple(geom.mesh_resolution)
        m['norm_coords_min'], m['norm_coords_max'] = coords.min(0).tolist(), coords.max(0).tolist()
        m['norm_power_mean'], m['norm_power_std'] = float(power.mean()), float(power.std())
        m['resolved_v5_active_split'] = True
        m['norm_temp_min'], m['norm_temp_max'] = float(temp.min()), float(temp.max())
        arrays['metadata'] = np.array([m], dtype=object)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        print(f'{i + 1}/{len(files)} {f.stem}  peak {temp.max() - 273.15:.1f} C '
              f'(was {float(d["temp"].max()) - 273.15:.1f} C)', flush=True)


if __name__ == '__main__':
    main()
