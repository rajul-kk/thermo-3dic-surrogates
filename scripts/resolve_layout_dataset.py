"""Re-solve a saved layout dataset under the current ICE wrapper, reusing each file's own scenario.
Placement, block powers, HTC, ambient and k-overrides come from the file's metadata, so no random
draw has to be reproduced. Writes to a new directory; the caller swaps it in after validation.
Usage: python scripts/resolve_layout_dataset.py geometry4 [--out data/3d-ice-layout-geometry4-v5]
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
from src.core.placement import place_chiplets
from src.simulators.ice_simulator import ICESimulator

ICE_EXE = 'wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('geometry')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    src = Path(f'data/3d-ice-layout-{args.geometry}')
    out = args.out or Path(f'data/3d-ice-layout-{args.geometry}-v5')
    files = sorted(src.rglob(f'{args.geometry}_*.npz'))
    for i, f in enumerate(files):
        target = out / f.relative_to(src)
        if target.exists():
            continue
        d = np.load(f, allow_pickle=True)
        arrays = {k: d[k] for k in d.files}
        m = arrays['metadata'].item()
        offs = {k[len('placement_dx_'):]: (float(m[k]), float(m['placement_dy_' + k[len('placement_dx_'):]]))
                for k in m if k.startswith('placement_dx_')}
        geom = place_chiplets(get_geometry_by_name(args.geometry), offs)
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
