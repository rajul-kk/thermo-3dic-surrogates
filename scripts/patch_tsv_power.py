"""
One-time patch: zero out power at TSV-region blocks in geometry2/geometry5 NPZ files.

Bug: generate_power_density_field() did not check block.is_tsv_region, so TSV-region
blocks (is_tsv_region=True) were assigned nonzero power in NPZ files even though
3D-ICE assigns them zero heat (they're passive conductors, excluded from floorplans).
This inconsistency corrupted the PINN PDE loss at those points.

Fix applied in mesh.py (generate_power_density_field) and generator.py (block_names).
This script patches existing NPZ files by recomputing the power field with the fix.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import build_geometry2a, build_geometry5
from src.core.mesh import generate_power_density_field


def patch_npz(npz_path: Path, geometry) -> bool:
    data = np.load(npz_path, allow_pickle=True)
    coords    = data['coords'].astype(np.float32)
    temp      = data['temp']
    layer     = data['layer']
    old_power = data['power']
    metadata  = data['metadata']

    meta = dict(metadata[0])
    power_blocks = {
        k[len('block_power_'):]: float(v)
        for k, v in meta.items() if k.startswith('block_power_')
    }

    new_power = generate_power_density_field(coords, geometry, power_blocks)
    changed = not np.allclose(old_power, new_power, atol=1.0)

    if changed:
        np.savez_compressed(
            npz_path,
            coords=coords,
            temp=temp,
            power=new_power.astype(np.float32),
            layer=layer,
            metadata=metadata,
        )

    data.close()
    return changed


def main():
    geom_dirs = [
        ('geometry2a', Path('data/3d-ice/geometry2a'), build_geometry2a()),
        ('geometry5',  Path('data/3d-ice/geometry5'),  build_geometry5()),
    ]

    total_patched = 0
    for gname, data_dir, geom in geom_dirs:
        npz_files = sorted(data_dir.glob('*.npz'))
        n_patched = 0
        for f in npz_files:
            try:
                if patch_npz(f, geom):
                    n_patched += 1
                    print(f'  patched: {f.name}')
            except Exception as e:
                print(f'  ERROR {f.name}: {e}')
        print(f'{gname}: {len(npz_files)} files, {n_patched} patched')
        total_patched += n_patched

    print(f'\nDone. Total patched: {total_patched}')


if __name__ == '__main__':
    main()
