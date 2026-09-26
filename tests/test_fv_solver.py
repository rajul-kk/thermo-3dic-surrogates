"""The validation solver must match exact 1D conduction and conserve energy before it can check 3D-ICE."""
import copy

import numpy as np
import pytest

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


def test_power_field_keeps_each_block_on_its_own_layer():
    from src.core.mesh import generate_power_density_field
    geom = get_geometry_by_name('geometry5')
    g = fv.make_grid(geom)
    xc = 0.5 * (g.xe[:-1] + g.xe[1:]); yc = 0.5 * (g.ye[:-1] + g.ye[1:]); zc = 0.5 * (g.ze[:-1] + g.ze[1:])
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    blocks = {b.name: 10.0 + i for i, b in enumerate(geom.power_blocks)}
    p = generate_power_density_field(coords, geom, blocks)
    vol = ((geom.die_width / g.shape[0]) * (geom.die_length / g.shape[1]) * 1e-12
           * np.diff(g.ze)[np.abs(coords[:, 2:3] - zc[None]).argmin(1)] * 1e-6)
    for layer in geom.layers:
        if not layer.is_active:
            continue
        sel = (coords[:, 2] >= layer.z_bottom) & (coords[:, 2] < layer.z_top)
        want = sum(b.power_watts(blocks[b.name]) for b in geom.power_blocks
                   if b.layer_name == layer.name and not b.is_tsv_region)
        assert abs((p[sel] * vol[sel]).sum() - want) / want < 0.02, layer.name


def test_points_to_grid_ignores_storage_order():
    from src.core.mesh import points_to_grid
    L, W, Z = np.meshgrid(np.arange(4) + 0.5, np.arange(3) + 0.5, [1.0, 5.0], indexing='ij')
    coords = np.stack([W.ravel(), L.ravel(), Z.ravel()], 1)       # (x = width, y = length, z)
    field = 100 * L.ravel() + 10 * W.ravel() + Z.ravel()
    perm = np.random.default_rng(0).permutation(len(field))
    g = points_to_grid(coords[perm], field[perm])
    assert g.shape == (4, 3, 2)                                   # (length, width, z)
    assert np.allclose(g, 100 * L + 10 * W + Z)


def test_fno_dataset_grid_is_spatially_coherent_on_real_data():
    import glob
    import pytest
    files = sorted(glob.glob('data/3d-ice-layout-geometry4/**/*.npz', recursive=True))[:2]
    if not files:
        pytest.skip('v5 layout data not on disk')
    from pathlib import Path
    from src.fno.data_loader import FNODataset
    from src.pinn.data_loader import compute_norm_stats
    d = np.load(files[0], allow_pickle=True)
    c, t = d['coords'], d['temp']
    from src.core.geometry_builders import get_geometry_by_name
    ns = compute_norm_stats([Path(f) for f in files], {"geometry4": get_geometry_by_name("geometry4")})
    mesh = d['metadata'].item()['mesh_resolution']
    nz = len(t) // (mesh[0] * mesh[1])
    ds = FNODataset([Path(files[0])], ns, (mesh[0], mesh[1], nz), target_grid=(mesh[0], mesh[1], nz))
    T = ds[0]['T_norm'].numpy()
    i, j, k = np.unravel_index(T.argmax(), T.shape)
    lc, wc = np.unique(c[:, 1]), np.unique(c[:, 0])
    peak = c[t.argmax()]
    assert abs(lc[i] - peak[1]) < 1 and abs(wc[j] - peak[0]) < 1   # grid peak = file peak


def test_layered_backbone_is_exact_for_laterally_uniform_stacks():
    """With no lateral heterogeneity the DCT x tridiagonal backbone must equal the full FV solve."""
    from src.hybrid import layered_backbone as lb
    geom = get_geometry_by_name('geometry1')                   # uniform layers, blocks only in power
    blocks = {b.name: 20.0 + 15 * i for i, b in enumerate(geom.power_blocks)}
    scen = {'power_blocks': blocks, 'htc': 15000.0, 't_ambient': 30.0}
    ref = fv.solve(geom, scen)['T']
    got = lb.solve(geom, scen)['T']
    assert np.abs(got - ref).max() < 1e-6 * (ref.max() - 303.15)


def test_thermono_is_exactly_linear_in_power_and_dct_is_orthonormal():
    import torch
    from src.fno.thermono import ThermoNO, dct_matrix
    D = dct_matrix(12)
    assert torch.allclose(D @ D.T, torch.eye(12), atol=1e-5)
    torch.manual_seed(0)
    m = ThermoNO((12, 8, 5), ch=6, n_blocks=2, modes=(6, 4)).double()
    lin = torch.randn(2, 2, 12, 8, 5, dtype=torch.float64)
    geo = torch.randn(2, 4, 12, 8, 5, dtype=torch.float64)
    out = m(lin, geo)
    assert torch.allclose(m(3.0 * lin, geo), 3.0 * out, atol=1e-9)           # homogeneous
    lin2 = torch.randn_like(lin)
    assert torch.allclose(m(lin + lin2, geo), out + m(lin2, geo), atol=1e-9)  # additive
    assert torch.allclose(m(torch.zeros_like(lin), geo), torch.zeros_like(out))


def test_lt_fno_couples_layers_only_through_its_z_matrix():
    """With identity z-coupling each layer is an independent 2D spectral conv; off-diagonal terms couple them."""
    import torch
    from src.fno.ltfno import LayerTransferConv3d, build_lt_fno
    torch.manual_seed(0)
    conv = LayerTransferConv3d(3, 3, (4, 3), nz=6).double()
    with torch.no_grad():
        conv.zcouple.copy_(torch.eye(6).expand(4, 3, 6, 6))
        conv.weight.data = conv.weight.data.to(torch.cdouble)
    x = torch.randn(2, 3, 10, 8, 6, dtype=torch.float64)
    x2 = x.clone(); x2[..., 4] += torch.randn_like(x2[..., 4])        # perturb layer 4 only
    d = (conv(x2) - conv(x)).abs().amax(dim=(0, 1, 2, 3))
    assert torch.all(d[[0, 1, 2, 3, 5]] < 1e-10) and d[4] > 1e-3      # no leakage between layers
    with torch.no_grad():
        conv.zcouple[..., 4, 1] = 0.5                                   # couple layer 4 into layer 1
    d = (conv(x2) - conv(x)).abs().amax(dim=(0, 1, 2, 3))
    assert d[1] > 1e-3
    m = build_lt_fno((12, 10, 7), modes=(4, 4, 3), hidden_ch=4, n_blocks=2)
    out = m(torch.randn(2, 12, 10, 7), torch.randn(2, 12, 10, 7), torch.randn(2), torch.randn(2), torch.randn(2))
    assert out.shape == (2, 12, 10, 7)


@pytest.mark.parametrize('local_k,multiscale', [(3, 0), (1, 3), (3, 3)])
def test_thermono_stays_linear_with_local_and_multiscale_paths(local_k, multiscale):
    import torch
    from src.fno.thermono import ThermoNO
    torch.manual_seed(0)
    m = ThermoNO((13, 9, 3), ch=6, n_blocks=2, modes=(6, 4), local_k=local_k, multiscale=multiscale).double()
    lin = torch.randn(2, 2, 13, 9, 3, dtype=torch.float64)
    geo = torch.randn(2, 4, 13, 9, 3, dtype=torch.float64)
    out = m(lin, geo)
    assert out.shape == (2, 13, 9, 3)
    assert torch.allclose(m(3.0 * lin, geo), 3.0 * out, atol=1e-9)
    lin2 = torch.randn_like(lin)
    assert torch.allclose(m(lin + lin2, geo), out + m(lin2, geo), atol=1e-9)


def test_thermono_defaults_unchanged_by_new_options():
    """local_k=1, multiscale=0 must build the same parameters in the same order as before."""
    import torch
    from src.fno.thermono import ThermoNO
    torch.manual_seed(0)
    a = ThermoNO((12, 8, 5), ch=6, n_blocks=2, modes=(6, 4))
    assert a.ms is None and all(l.kernel_size == (1, 1, 1) for l in a.local)
    assert sum(p.numel() for p in a.parameters()) == sum(
        p.numel() for n, p in a.named_parameters() if not n.startswith('ms.'))


@pytest.mark.parametrize('geo_arch,geo_attn', [('cno', False), ('cno', True)])
def test_thermono_stays_linear_in_power_with_cno_geometry_encoder(geo_arch, geo_attn):
    import torch
    from src.fno.thermono import ThermoNO
    torch.manual_seed(0)
    m = ThermoNO((12, 8, 5), ch=8, n_blocks=2, modes=(6, 4), geo_arch=geo_arch, geo_attn=geo_attn).double()
    lin = torch.randn(2, 2, 12, 8, 5, dtype=torch.float64)
    geo = torch.randn(2, 4, 12, 8, 5, dtype=torch.float64)
    out = m(lin, geo)
    assert torch.allclose(m(3.0 * lin, geo), 3.0 * out, atol=1e-9)
    lin2 = torch.randn_like(lin)
    assert torch.allclose(m(lin + lin2, geo), out + m(lin2, geo), atol=1e-9)
