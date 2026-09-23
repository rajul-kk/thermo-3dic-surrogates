"""The validation solver must match exact 1D conduction and conserve energy before it can check 3D-ICE."""
import copy

import numpy as np

from src.core.geometry import PowerBlock
from src.core.geometry_builders import get_geometry_by_name
from src.validation import fv_solver as fv


def _full_die_geometry():
    geom = copy.deepcopy(get_geometry_by_name('geometry1'))
    die = next(l for l in geom.layers if l.is_active)
    geom.power_blocks = [PowerBlock(name='all', x=0.0, y=0.0, width=geom.die_width,
                                    height=geom.die_length, layer_name=die.name)]
    geom.mesh_resolution = (10, 10, geom.mesh_resolution[2])
    return geom


def test_matches_exact_1d_solution_below_the_source():
    geom = _full_die_geometry()
    htc, p_density = 20000.0, 50.0                     # W/m²K, W/cm²
    res = fv.solve(geom, {'power_blocks': {'all': p_density}, 'htc': htc, 't_ambient': 25.0},
                   fv.make_grid(geom, max_sub_um=250.0))
    g = res['grid']
    flux = p_density * 1e4                              # W/m²
    zc = 0.5 * (g.ze[:-1] + g.ze[1:])
    die = next(i for i, l in enumerate(geom.layers) if l.is_active)
    exact = []
    for kz, z in enumerate(zc):
        if g.z_layer[kz] >= die:
            break
        r = 1.0 / htc
        for lay in geom.layers:
            lo, hi = lay.z_bottom, min(lay.z_top, z)
            if hi > lo:
                r += (hi - lo) * 1e-6 / lay.k_thermal
        exact.append(25.0 + 273.15 + flux * r)
    got = res['T'][0, 0, :len(exact)]
    assert np.allclose(got, exact, rtol=0, atol=1e-8)
    assert np.allclose(res['T'][:, :, 0], res['T'][0, 0, 0])   # laterally uniform


def test_energy_balance_holds_with_lateral_structure():
    geom = get_geometry_by_name('geometry4')
    blocks = {b.name: 20.0 for b in geom.power_blocks}
    res = fv.solve(geom, {'power_blocks': blocks, 'htc': 8000.0, 't_ambient': 30.0})
    assert abs(res['P_out'] - res['P_in']) / res['P_in'] < 1e-8
