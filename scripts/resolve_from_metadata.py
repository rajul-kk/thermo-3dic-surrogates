"""Re-solve a saved dataset under the current ICE wrapper, reusing each file's own scenario.
Geometry, placement, block powers, HTC, ambient and k-overrides come from each file's metadata,
so no random draw has to be reproduced. Only for data with no feedback loop (leakage/throttle).
Per-cell power maps (fixed-placement geometry1-6) and TSV density maps (geometry2a/5/6)
are read back from the stored power and tsv_frac arrays. The TSV layout is 32x32 tiles that fall mid-cell, which the iceforge
guard refuses (S003); re-solving TSV data therefore needs ICEFORGE_GUARD=0, which reproduces how the files were made.
Writes to a new directory for validation first.
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
from src.core.mesh import generate_power_density_field, generate_tsv_field
from src.core.placement import place_chiplets, snap_offsets
from src.simulators.ice_simulator import ICESimulator, DEFAULT_ICE_EXECUTABLE, last_run_provenance
from scripts.resolve_maps import check_power_maps, recover_power_maps, recover_tsv_maps

ICE_EXE = DEFAULT_ICE_EXECUTABLE
MICROCHANNEL_GEOMETRY = 'geometry6_microchannel'      # built by scripts/gen_microchannel_pilot.py, not a named geometry


def load_geometry(name):
    if name == MICROCHANNEL_GEOMETRY:
        from scripts.gen_microchannel_pilot import build_microchannel_geometry6
        return build_microchannel_geometry6()
    return get_geometry_by_name(name)


def stored_power_maps(arrays, geom, blocks):
    """{} when the saved power is the plain block decomposition; otherwise the per-cell maps it was sampled from.
    The maps are not stored, but the power array is a sampling of them (tile-constant), so they are read back from
    it and checked against the block_power_* metadata."""
    coords, stored = arrays['coords'].astype(np.float64), arrays['power'].astype(np.float64)
    if np.abs(generate_power_density_field(coords, geom, blocks) - stored).max() <= 1e-5 * max(stored.max(), 1e-300):
        return {}
    if len(np.unique(stored[stored > 0])) <= 4 * len(blocks) + 8:        # still piecewise constant per block: no map
        return {}
    maps = recover_power_maps(coords, stored, geom)
    check_power_maps(maps, geom, blocks)
    return maps


def stored_tsv_maps(arrays, geom, stem):
    """{} when the file has no TSV density field; otherwise the per-layer maps behind its tsv_frac array.
    They are 32x32 tile grids, independent of the node grid, so they carry to a new mesh as well as the stored one."""
    frac = arrays.get('tsv_frac')
    if frac is None or not np.abs(frac).max() > 0:
        return {}
    return recover_tsv_maps(arrays['coords'].astype(np.float64), frac, geom, stem)


def extra_scenario_keys(m, stem):
    """Scenario keys the metadata does not carry but the generator set (microchannel cooling and its flow rate)."""
    if m['geometry'] != MICROCHANNEL_GEOMETRY:
        return {}
    from scripts.gen_microchannel_pilot import microchannel_cooling
    return microchannel_cooling(int(stem.rsplit('_', 1)[1]))      # the file index is the generator's loop index


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--snap', action='store_true', help='snap chiplet offsets to the 3D-ICE cell grid first')
    args = ap.parse_args(argv)
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
        base = load_geometry(m['geometry'])
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
                'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}'),
                **extra_scenario_keys(m, f.stem)}
        power_maps = stored_power_maps(arrays, geom, blocks)
        if power_maps and offs:
            raise ValueError(f'{f}: power maps together with a randomised placement are not supported')
        if power_maps:
            scen['power_map_by_layer'] = power_maps
            m['resolved_power_map_recovered'] = True
        tsv_maps = stored_tsv_maps(arrays, geom, f.stem)
        if tsv_maps:
            scen['tsv_map_by_layer'] = tsv_maps
            m['resolved_tsv_map_recovered'] = True
        with tempfile.TemporaryDirectory() as td:
            cfg, res = Path(td) / 'c', Path(td) / 'o'
            cfg.mkdir(); res.mkdir()
            parsed = ICESimulator(cfg, res, ICE_EXE).simulate(geom, scen, f.stem)
        coords = parsed['coords'].astype(np.float64)
        temp = parsed['temperature'].astype(np.float64)
        # the grid may differ from the saved one (geometry4/5 meshes were corrected), so
        # every per-point array is rebuilt on 3D-ICE's own nodes
        power = generate_power_density_field(coords, geom, blocks, power_map_by_layer=power_maps or None)
        arrays['coords'] = coords.astype(np.float32)
        arrays['temp'] = temp.astype(np.float32)
        arrays['power'] = power.astype(np.float32)
        arrays['layer'] = np.array([geom.get_layer_index_at_z(z) for z in coords[:, 2]], dtype=np.int32)
        if 'power_nominal' in arrays:
            arrays['power_nominal'] = arrays['power']
        if 'tsv_frac' in arrays:
            arrays['tsv_frac'] = generate_tsv_field(coords, geom, tsv_maps).astype(np.float32)
        m['num_points'] = len(coords)
        m['mesh_resolution'] = tuple(geom.mesh_resolution)
        m['norm_coords_min'], m['norm_coords_max'] = coords.min(0).tolist(), coords.max(0).tolist()
        m['norm_power_mean'], m['norm_power_std'] = float(power.mean()), float(power.std())
        m['resolved_v5_active_split'] = True
        m.update(last_run_provenance())          # ice_commit / ice_executable, as verified by the solve just run
        m['norm_temp_min'], m['norm_temp_max'] = float(temp.min()), float(temp.max())
        arrays['metadata'] = np.array([m], dtype=object)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        print(f'{i + 1}/{len(files)} {f.stem}  peak {temp.max() - 273.15:.1f} C '
              f'(was {float(d["temp"].max()) - 273.15:.1f} C)', flush=True)


if __name__ == '__main__':
    main()
