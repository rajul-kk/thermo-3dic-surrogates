"""Validate the 3D-ICE ground truth: energy balance on every saved solve, an independent FV re-solve, grid convergence.
Usage: python scripts/validate_3dice.py [--check A|B|all] [--geometries geometry1 ...]
"""
import argparse
import ast
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import place_chiplets
from src.scenario.generator import ScenarioGenerator
from src.scenario.tsv_maps import generate_tsv_density_map
from src.simulators.ice_simulator import ICESimulator
from src.validation import fv_solver as fv

ICE_EXE = 'wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator'
GEOMS = ['geometry1', 'geometry2a', 'geometry3', 'geometry4', 'geometry5', 'geometry6']
LEVELS = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 4)]      # (lateral refine, z multiplier)
MAX_CELLS = 2_500_000
OUT = Path('results/validation_3dice.json')


# ── Check A: energy balance of every saved 3D-ICE solve ─────────────────────────
def energy_balance(f: Path):
    """Heat leaving the bottom face vs power injected (from the NPZ power field, i.e. what 3D-ICE was given)."""
    d = np.load(f, allow_pickle=True)
    m = d['metadata'].item()
    if float(m.get('throttle_derate_factor', 1.0)) != 1.0 or float(m.get('leakage_multiplier', 1.0)) != 1.0:
        return None
    geom = get_geometry_by_name(m['geometry'])
    over = ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')
    g = fv.make_grid(geom)
    dz = np.diff(g.ze) * 1e-6
    zc = 0.5 * (g.ze[:-1] + g.ze[1:])
    area = (geom.die_width / g.shape[0]) * (geom.die_length / g.shape[1]) * 1e-12
    c, T = d['coords'], d['temp'].astype(np.float64)
    kz = np.abs(c[:, 2:3] - zc[None]).argmin(1)
    w = d['power'].astype(np.float64) * area * dz[kz]
    p_field = float(w.sum())
    p_meta = sum(b.power_watts(float(m.get(f'block_power_{b.name}', 0.0)))
                 for b in geom.power_blocks if not b.is_tsv_region)
    sink = geom.layers[0]
    k0 = float(over.get(sink.name, sink.k_thermal))
    bottom = kz == 0
    g_cell = area / (dz[0] / (2 * k0) + 1.0 / float(m['htc']))
    p_out = float((g_cell * (T[bottom] - float(m['t_ambient_kelvin']))).sum())
    gap = np.ones(len(c), bool)
    for fp in geom.die_footprints:
        gap &= ~((c[:, 0] >= fp.x) & (c[:, 0] < fp.x + fp.width) & (c[:, 1] >= fp.y) & (c[:, 1] < fp.y + fp.height))
    return {'file': f.name, 'geometry': m['geometry'], 'dataset': f.parts[1],
            'P_field_W': p_field, 'P_out_W': p_out, 'P_meta_W': p_meta,
            'ratio': p_out / p_field, 'meta_over_field': p_meta / p_field,
            # nominal footprints only describe fixed-placement data
            'underfill_power_frac': float(w[gap].sum() / p_field)
            if geom.die_footprints and not m.get('placement_randomized') else 0.0}


def check_a():
    files = sorted(Path('data/3d-ice').rglob('*.npz'))
    for g in GEOMS:
        files += sorted(Path(f'data/3d-ice-layout-{g}').rglob('*.npz'))
    rows = [r for r in (energy_balance(f) for f in files) if r]
    out = {'n': len(rows)}
    print(f'\n[A] energy balance over {len(rows)} saved solves (power from the NPZ power field)')
    for ds in sorted({r['dataset'] for r in rows}):
        rr = [r for r in rows if r['dataset'] == ds]
        ratio = np.array([r['ratio'] for r in rr])
        meta = np.array([r['meta_over_field'] for r in rr])
        uf = np.array([r['underfill_power_frac'] for r in rr])
        out[ds] = {'n': len(rr), 'worst_abs_1_minus_ratio': float(np.abs(1 - ratio).max()),
                   'meta_over_field_median': float(np.median(meta)),
                   'meta_over_field_range': [float(meta.min()), float(meta.max())],
                   'underfill_power_frac_median': float(np.median(uf)), 'underfill_power_frac_max': float(uf.max())}
        print(f"    {ds:<28} n={len(rr):>3}  worst |1-out/in| {out[ds]['worst_abs_1_minus_ratio']:.1e}  "
              f"metadata/actual power median {np.median(meta):.3f}  power in underfill median {100*np.median(uf):.0f}%")
    return out


# ── Checks B/C: paired 3D-ICE vs independent FV, and FV grid convergence ─────────
def tsv_maps_for(geom, seed):
    layers = [l.name for l in geom.layers if 'tsv' in l.name.lower()]
    if not layers or geom.tsv_density <= 0:
        return None
    return {n: generate_tsv_density_map(32, 32, mean_density=geom.tsv_density, seed=seed + j, contrast=0.8)
            for j, n in enumerate(layers)}


def _drop_footprint_tsv(geom, scen):
    """3D-ICE takes one layout per layer, so a layer with chiplet footprints cannot also carry a TSV map."""
    fp_layers = {fp.die_layer_name for fp in geom.die_footprints}
    if scen.get('tsv_map_by_layer'):
        scen['tsv_map_by_layer'] = {k: v for k, v in scen['tsv_map_by_layer'].items() if k not in fp_layers} or None
    return scen


def fixed_cases(name):
    geom = get_geometry_by_name(name)
    gen = ScenarioGenerator()
    scens = gen.generate_all_scenarios(geom)
    gen.attach_tsv_maps(scens, geom)
    gen.attach_power_maps(scens, geom, kind='mixed')     # as regen_v4_final.ps1 generated data/3d-ice
    tests = [_drop_footprint_tsv(geom, s.to_dict()) for s in scens if s.to_dict()['type'] == 'test']
    return [(f'{name}_fixed_{s["pattern"]}', geom, s) for s in tests[:2]]


def layout_case(name):
    f = sorted(Path(f'data/3d-ice-layout-{name}').rglob('*.npz'))[0]
    m = np.load(f, allow_pickle=True)['metadata'].item()
    base = get_geometry_by_name(name)
    offs = {k[len('placement_dx_'):]: (float(m[k]), float(m['placement_dy_' + k[len('placement_dx_'):]]))
            for k in m if k.startswith('placement_dx_')}
    geom = place_chiplets(base, offs)
    scen = {'power_blocks': {b.name: float(m.get(f'block_power_{b.name}', 0.0)) for b in geom.power_blocks},
            'htc': float(m['htc']), 't_ambient': float(m['t_ambient_celsius']),
            'layer_k_overrides': ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}'),
            'tsv_map_by_layer': tsv_maps_for(geom, 7)}
    return (f'{name}_layout_{f.stem}', geom, _drop_footprint_tsv(geom, scen))


def top_active_peak(theta, g, geom):
    """(peak rise K, (x, y) µm of the argmax) on the uppermost active layer."""
    top = max(i for i, l in enumerate(geom.layers) if l.is_active)
    ks = np.nonzero(g.z_layer == top)[0]
    lay = theta[:, :, ks].mean(axis=2)
    i, j = np.unravel_index(np.nanargmax(lay), lay.shape)
    xc = 0.5 * (g.xe[:-1] + g.xe[1:]); yc = 0.5 * (g.ye[:-1] + g.ye[1:])
    return float(lay[i, j]), (float(xc[i]), float(yc[j]))


def run_case(label, geom, scen, workdir: Path):
    t_amb = float(scen.get('t_ambient', 25.0)) + 273.15
    cfg, out = workdir / 'cfg', workdir / 'out'
    cfg.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    sim = ICESimulator(cfg, out, ICE_EXE)
    t = time.time()
    parsed = sim.simulate(geom, scen, label)
    t_ice = time.time() - t

    g0 = fv.make_grid(geom)
    ice = fv.ice_to_grid(parsed['coords'].astype(np.float64), parsed['temperature'].astype(np.float64), g0)
    assert not np.isnan(ice).any(), '3D-ICE output does not map one-to-one onto the native grid'
    th_ice = ice - t_amb

    levels = {}
    for r, mz in LEVELS:
        g = fv.make_grid(geom, lateral_refine=r, mult=mz)
        if np.prod(g.shape) > MAX_CELLS:
            continue
        t = time.time()
        res = fv.solve(geom, scen, g)
        th = fv.coarsen_to(res['T'], g, g0) - t_amb if (r, mz) != (1, 1) else res['T'] - t_amb
        levels[(r, mz)] = {'theta': th, 'cells': int(np.prod(g.shape)), 'sec': time.time() - t,
                           'balance': res['P_out'] / res['P_in']}

    fine_key = max(levels, key=lambda k: levels[k]['cells'])
    th_fine = levels[fine_key]['theta']
    th_fv0 = levels[(1, 1)]['theta']
    peak_ice, loc_ice = top_active_peak(th_ice, g0, geom)
    peak_fv0, loc_fv0 = top_active_peak(th_fv0, g0, geom)
    peak_fine, loc_fine = top_active_peak(th_fine, g0, geom)
    rise = float(th_ice.max())

    def rel_rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))

    row = {
        'case': label, 'ice_sec': round(t_ice, 1), 'max_rise_K': round(rise, 3),
        # B: does 3D-ICE solve the problem we specified? (same discretisation)
        'B_max_abs_diff_K': float(np.abs(th_ice - th_fv0).max()),
        'B_max_diff_pct_of_rise': float(100 * np.abs(th_ice - th_fv0).max() / rise),
        'B_peak_loc_shift_um': float(np.hypot(loc_ice[0] - loc_fv0[0], loc_ice[1] - loc_fv0[1])),
        # C: how far is the dataset's discretisation from converged?
        'C_fine_level': list(fine_key), 'C_fine_cells': levels[fine_key]['cells'],
        'C_peak_err_pct': float(100 * (peak_ice - peak_fine) / peak_fine),
        'C_field_rel_rms_pct': float(100 * rel_rms(th_ice, th_fine)),
        'C_peak_loc_shift_um': float(np.hypot(loc_ice[0] - loc_fine[0], loc_ice[1] - loc_fine[1])),
        'C_levels': {f'{k[0]}x{k[1]}': {'cells': v['cells'], 'sec': round(v['sec'], 1),
                                         'peak_top_K': round(top_active_peak(v['theta'], g0, geom)[0], 4),
                                         'energy_balance': v['balance']}
                     for k, v in levels.items()},
    }
    print(f"  {label:<44} rise {rise:7.2f} K | B max|d| {row['B_max_abs_diff_K']:.3g} K "
          f"({row['B_max_diff_pct_of_rise']:.2f}%) | C peak {row['C_peak_err_pct']:+.2f}% "
          f"rms {row['C_field_rel_rms_pct']:.2f}% loc {row['C_peak_loc_shift_um']:.0f} um")
    print('      peak rise by level (lateral x z):',
          {k: v['peak_top_K'] for k, v in row['C_levels'].items()}, '| 3D-ICE', round(peak_ice, 4))
    return row


def _save_partial(rows):
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev = {r['case']: r for r in out.get('BC_cases', [])}
    prev.update({r['case']: r for r in rows})
    out['BC_cases'] = list(prev.values())
    OUT.write_text(json.dumps(out, indent=1, default=float))


def check_bc(names):
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for name in names:
            cases = fixed_cases(name) + [layout_case(name)]
            for label, geom, scen in cases:
                rows.append(run_case(label, geom, scen, Path(td) / label))
                _save_partial(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', default='all', choices=['A', 'B', 'all'])
    ap.add_argument('--geometries', nargs='*', default=GEOMS)
    args = ap.parse_args()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.check in ('A', 'all'):
        out['A_energy_balance'] = check_a()
    if args.check in ('B', 'all'):
        print('\n[B/C] paired 3D-ICE vs independent FV, and FV grid convergence')
        prev = {r['case']: r for r in out.get('BC_cases', [])}
        for r in check_bc(args.geometries):
            prev[r['case']] = r
        out['BC_cases'] = list(prev.values())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
