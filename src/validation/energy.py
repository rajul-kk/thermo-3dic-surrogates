"""Energy balance of a saved 3D-ICE solve: heat leaving the bottom face vs power injected."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.core.geometry_builders import get_geometry_by_name
from src.validation import fv_solver as fv


def energy_balance(f: Path, include_feedback: bool = False):
    """Heat leaving the bottom face vs power injected (from the NPZ power field, i.e. what 3D-ICE was given)."""
    f = Path(f)
    d = np.load(f, allow_pickle=True)
    m = d['metadata'].item()
    feedback = float(m.get('throttle_derate_factor', 1.0)) != 1.0 or float(m.get('leakage_multiplier', 1.0)) != 1.0
    if feedback and not include_feedback:     # the power field is the delivered power either way
        return None
    geom = get_geometry_by_name(m['geometry'])
    over = ast.literal_eval(m.get('layer_k_overrides', '{}') or '{}')
    g = fv.make_grid(geom)
    dz = np.diff(g.ze) * 1e-6
    zc = 0.5 * (g.ze[:-1] + g.ze[1:])
    area = (geom.die_width / g.shape[0]) * (geom.die_length / g.shape[1]) * 1e-12
    c, T = d['coords'], d['temp'].astype(np.float64)
    kz = np.abs(c[:, 2:3] - zc[None]).argmin(1)
    # the power field is W/m^3 over a whole active layer, whatever its z-split
    thick = np.array([geom.layers[i].thickness * 1e-6 if geom.layers[i].is_active else 0.0 for i in g.z_layer])
    vol_dz = np.where(thick[kz] > 0, thick[kz], dz[kz])
    w = d['power'].astype(np.float64) * area * vol_dz
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
    return {'file': f.name, 'geometry': m['geometry'], 'dataset': f.parts[1] if len(f.parts) > 1 else '',
            'P_field_W': p_field, 'P_out_W': p_out, 'P_meta_W': p_meta,
            'ratio': p_out / p_field, 'meta_over_field': p_meta / p_field,
            # nominal footprints only describe fixed-placement data
            'underfill_power_frac': float(w[gap].sum() / p_field)
            if geom.die_footprints and not m.get('placement_randomized') else 0.0}
