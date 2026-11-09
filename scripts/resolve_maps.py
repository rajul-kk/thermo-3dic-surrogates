"""Recover the per-cell power and TSV-density maps of a saved dataset file, so a re-solve uses the same sources.
The npz files store the sampled `power` and `tsv_frac` arrays but not the maps they came from; the maps are
resolution-independent tile grids, so they can be read back from the stored arrays and carried to any new grid.
"""
import re

import numpy as np

TSV_RESOLUTION = 32          # ScenarioGenerator.attach_tsv_maps default tile count per side
BLOCK_RTOL = 1e-3            # recovered watts per block vs the block_power_* metadata


def power_tiles(geom):
    """Tile grid (n_l, n_w) ScenarioGenerator.attach_power_maps uses for `geom` (resolution=0)."""
    n_len, n_wid = int(geom.mesh_resolution[0]), int(geom.mesh_resolution[1])
    f = 1
    while geom.die_footprints and (n_len % f or n_wid % f or (n_len // f) * (n_wid // f) > 4096):
        f += 1
    return n_len // f, n_wid // f


def _tile_index(coords, geom, mask, n_l, n_w):
    ia = np.clip((coords[mask, 1] / geom.die_length * n_l).astype(int), 0, n_l - 1)
    ib = np.clip((coords[mask, 0] / geom.die_width * n_w).astype(int), 0, n_w - 1)
    return ia, ib


def recover_map(coords, values, geom, layer, n_l, n_w, scale=1.0):
    """Invert mesh._power_field_from_maps / generate_tsv_field for one layer: the (n_l, n_w) map whose
    sampling at `coords` gives `values * scale`. Raises if a tile has no node or the layer is not tile-constant."""
    inl = (coords[:, 2] >= layer.z_bottom) & (coords[:, 2] < layer.z_top)
    ia, ib = _tile_index(coords, geom, inl, n_l, n_w)
    v = np.asarray(values, dtype=np.float64)[inl] * scale
    total, count = np.zeros((n_l, n_w)), np.zeros((n_l, n_w))
    np.add.at(total, (ia, ib), v)
    np.add.at(count, (ia, ib), 1)
    if (count == 0).any():
        raise ValueError(f'layer {layer.name}: {(count == 0).sum()} of {n_l * n_w} tiles have no node; '
                         f'the stored grid is too coarse to recover the map')
    pm = total / count
    if np.abs(v - pm[ia, ib]).max() > 1e-6 * max(np.abs(v).max(), 1e-300):
        raise ValueError(f'layer {layer.name}: values are not constant within a tile of a {n_l}x{n_w} map')
    return pm


def recover_power_maps(coords, power, geom):
    """{active layer name: W/tile map} read back from the stored power-density array (float32-exact)."""
    n_l, n_w = power_tiles(geom)
    maps = {}
    for layer in geom.layers:
        if layer.is_active:
            vol_m3 = (geom.die_length / n_l) * (geom.die_width / n_w) * layer.thickness * 1e-18
            maps[layer.name] = recover_map(coords, power, geom, layer, n_l, n_w, vol_m3)
    return maps


def check_power_maps(maps, geom, blocks):
    """Independent check of recovered maps: the watts in each block's footprint equal block_power_* x area."""
    n_l, n_w = power_tiles(geom)
    ae, be = np.arange(n_l + 1) * (geom.die_length / n_l), np.arange(n_w + 1) * (geom.die_width / n_w)
    for b in geom.power_blocks:
        if b.is_tsv_region or b.layer_name not in maps:
            continue
        oa = np.clip(np.minimum(ae[1:], b.y + b.height) - np.maximum(ae[:-1], b.y), 0, None)
        ob = np.clip(np.minimum(be[1:], b.x + b.width) - np.maximum(be[:-1], b.x), 0, None)
        got = float(maps[b.layer_name][np.outer(oa, ob) > 0].sum())
        want = b.power_watts(blocks.get(b.name, 0.0))
        if abs(got - want) > BLOCK_RTOL * max(abs(want), 1e-9):
            raise ValueError(f'block {b.name}: recovered map holds {got:.6g} W, metadata says {want:.6g} W')


def tsv_layers(geom):
    return [l for l in geom.layers if 'tsv' in l.name.lower()]


def tsv_seed(stem):
    """(seed_base, scenario index) ScenarioGenerator.attach_tsv_maps used for file `<geometry>_<train|test>_<NNN>`.
    Base set: train 1-15 then test 1-5 share one list (seed_base 0, index n-1 / 14+n); extra train 16+ use
    seed_base 900000 and index n-16 (src/main.py). None if the name does not follow that pattern."""
    m = re.fullmatch(r'.*_(train|test)_(\d+)', stem)
    if not m:
        return None
    kind, n = m.group(1), int(m.group(2))
    if kind == 'test':
        return (0, 14 + n) if n <= 5 else None
    return (0, n - 1) if n <= 15 else (900000, n - 16)


def recover_tsv_maps(coords, tsv_frac, geom, stem=None, resolution=TSV_RESOLUTION):
    """{tsv layer name: density map} for a file with nonzero tsv_frac. Read back from the stored array, then
    replaced by the generator's own float64 map when regenerating it from the file's seed reproduces the same
    float32 values (so the 12-level quantisation in the stack file matches the original solve)."""
    from src.scenario.tsv_maps import generate_tsv_density_map
    seed = tsv_seed(stem) if stem else None
    maps = {}
    for j, layer in enumerate(tsv_layers(geom)):
        pm = recover_map(coords, tsv_frac, geom, layer, resolution, resolution)
        if seed is not None:
            regen = generate_tsv_density_map(resolution, resolution, mean_density=geom.tsv_density,
                                             seed=seed[0] + 100 * seed[1] + j, contrast=0.8)
            if np.array_equal(regen.astype(np.float32), pm.astype(np.float32)):
                pm = regen
        maps[layer.name] = pm
    return maps
