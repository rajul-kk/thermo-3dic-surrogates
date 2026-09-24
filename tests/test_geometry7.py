"""
geometry7 (CoWoS-L reticle-stitched pilot) structural checks, and the two `ice_simulator.py` fixes it depends on:
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry import Geometry, Layer, PowerBlock, DiePrint
from src.core.geometry_builders import build_geometry7, get_geometry_by_name
from src.core.material import MaterialLibrary
from src.simulators.ice_simulator import ICESimulator


# ── geometry7 structural checks ─────────────────────────────────────────────

def test_geometry7_self_validates():
    build_geometry7().validate()


def test_geometry7_reachable_via_registry():
    assert get_geometry_by_name('geometry7').name == 'geometry7'


def test_geometry7_excluded_from_standard_dataset():
    from src.core.geometry_builders import build_all_geometries
    names = [g.name for g in build_all_geometries()]
    assert 'geometry7' not in names, (
        "geometry7 is a pilot, not part of the standard 6-geometry dataset -- "
        "if this now fails, every doc/script that assumes 6 geometries needs updating"
    )


def test_geometry7_every_geometry_has_an_explicit_tdp():
    from src.scenario.generator import ScenarioGenerator
    assert 'geometry7' in ScenarioGenerator.TDP_BY_GEOMETRY_W


def test_geometry7_power_blocks_lie_inside_the_die_footprint():
    geom = build_geometry7()
    for b in geom.power_blocks:
        assert b.x >= 0 and b.x + b.width <= geom.die_width, b.name
        assert b.y >= 0 and b.y + b.height <= geom.die_length, b.name


def test_geometry7_die_footprints_lie_inside_the_die_footprint():
    geom = build_geometry7()
    for fp in geom.die_footprints:
        assert fp.x >= 0 and fp.x + fp.width <= geom.die_width, fp.name
        assert fp.y >= 0 and fp.y + fp.height <= geom.die_length, fp.name


def test_geometry7_bridge_layer_has_gap_material_set():
    """The one structural feature this geometry exists to test."""
    geom = build_geometry7()
    bridge_layer = next(l for l in geom.layers if l.name == 'substrate_organic')
    assert bridge_layer.gap_material == 'organic_substrate'
    assert bridge_layer.material == 'lsi_bridge_via'  # the bridge/via islands themselves
    # every other geometry's layers must be untouched by adding this field
    other_names = [n for n in ('geometry1', 'geometry2a', 'geometry3',
                                'geometry4', 'geometry5', 'geometry6')]
    for name in other_names:
        for layer in get_geometry_by_name(name).layers:
            assert layer.gap_material is None, f"{name}/{layer.name}"


def test_geometry7_bridge_islands_dont_overlap_hbm_stacks():
    geom = build_geometry7()
    bridges = [fp for fp in geom.die_footprints if fp.name.startswith('bridge_')]
    hbm_footprints = [fp for fp in geom.die_footprints if fp.name.startswith('hbm')]
    assert len(bridges) == 10  # 2 full compute-die coverage + 8 HBM D2D bridges
    for br in bridges:
        for hb in hbm_footprints:
            if hb.name.endswith('_die1') or hb.name.endswith('_tsv') or hb.name.endswith('_die2'):
                x_overlap = br.x < hb.x + hb.width and hb.x < br.x + br.width
                y_overlap = br.y < hb.y + hb.height and hb.y < br.y + br.height
                if x_overlap and y_overlap and br.name != f"bridge_{hb.name.rsplit('_', 1)[0]}":
                    # a bridge is allowed to touch/overlap ITS OWN stack's near edge
                    # (that's the design -- see build_geometry7 docstring), just not
                    # a DIFFERENT stack's footprint
                    stack_n = hb.name.split('_')[0]  # e.g. 'hbm3'
                    assert br.name == f'bridge_{stack_n}', (
                        f"{br.name} unexpectedly overlaps {hb.name}")


# ── ice_simulator: passive-layer footprints + gap_material ─────────────────

def _make_probe_geometry(gap_material=None):
    mat_cu = MaterialLibrary.get('copper')
    mat_si = MaterialLibrary.get('silicon')
    bridge_mat = MaterialLibrary.get(gap_material) if gap_material else mat_si
    return Geometry(
        name='probe', geometry_type='2p5d_stack',
        layers=[
            Layer(name='heat_sink', material='copper', thickness=1000.0,
                  k_thermal=mat_cu.k_thermal,
                  volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
            Layer(name='passive_probe', material='silicon', thickness=200.0,
                  k_thermal=(bridge_mat.k_thermal if gap_material else mat_si.k_thermal),
                  volumetric_heat_capacity=mat_si.volumetric_heat_capacity,
                  gap_material=gap_material),
            Layer(name='die_zone_1', material='silicon', thickness=50.0,
                  k_thermal=mat_si.k_thermal,
                  volumetric_heat_capacity=mat_si.volumetric_heat_capacity,
                  is_active=True),
        ],
        power_blocks=[PowerBlock(name='b1', x=0.0, y=0.0, width=5000.0, height=5000.0,
                                 layer_name='die_zone_1')],
        die_width=10000.0, die_length=10000.0, mesh_resolution=(20, 20, 20),
        die_footprints=[DiePrint('bridge1', x=2000.0, y=2000.0, width=3000.0,
                                 height=3000.0, die_layer_name='passive_probe')],
        underfill_k=0.7,
    )


def _render_stack(geom, power_w=10.0):
    tmp = Path(tempfile.mkdtemp())
    cfg, out = tmp / 'cfg', tmp / 'out'
    cfg.mkdir(); out.mkdir()
    sc = {'htc': 5000.0, 't_ambient': 25.0, 'power_blocks': {'b1': power_w}}
    sim = ICESimulator(config_dir=cfg, output_dir=out, executable='echo')
    sim._sublayers = sim._plan_sublayers(geom)
    sim._generate_layout_files(geom, sc)
    sim._generate_stack_file(geom, sc)
    return (cfg / 'stack.stk').read_text(), cfg


def test_passive_layer_footprint_layout_is_referenced_in_the_stack_file():
    """
    Regression test: before the 2026-08-17 fix, the .lyt file for a passive layer's DiePrint footprints was written but never referenced in stack.stk,
    """
    geom = _make_probe_geometry()
    stk, cfg = _render_stack(geom)
    assert (cfg / 'layout_footprint_passive_probe.lyt').exists()
    assert 'layout_footprint_passive_probe.lyt' in stk


def test_passive_layer_without_gap_material_falls_back_to_underfill():
    geom = _make_probe_geometry(gap_material=None)
    stk, _ = _render_stack(geom)
    lines = stk.splitlines()
    decl = next(l for l in lines if l.strip().startswith('layer type_layer_1 :'))
    idx = lines.index(decl)
    mat_line = next(l for l in lines[idx:idx + 4] if 'material' in l)
    assert 'underfill' in mat_line


def test_passive_layer_gap_material_overrides_underfill():
    geom = _make_probe_geometry(gap_material='organic_substrate')
    stk, _ = _render_stack(geom)
    lines = stk.splitlines()
    decl = next(l for l in lines if l.strip().startswith('layer type_layer_1 :'))
    idx = lines.index(decl)
    mat_line = next(l for l in lines[idx:idx + 4] if 'material' in l)
    assert 'organic_substrate' in mat_line
    assert 'underfill' not in mat_line
    # and the gap material itself must be declared with its real k, not silently
    # missing (which would make 3D-ICE fail on an undeclared material name)
    decl_lines = [l for l in lines if l.strip().startswith('material organic_substrate')]
    assert decl_lines, "organic_substrate must appear in the material declarations"


def test_tsv_map_on_a_footprint_layer_is_clipped_to_the_footprints():
    """A layer holds one `layout`, so a TSV map on a footprint layer (geometry5/6's tsv_zone) is
    clipped to the footprints and the rest of the layer takes the gap material."""
    geom = _make_probe_geometry(gap_material='organic_substrate')
    tmp = Path(tempfile.mkdtemp())
    cfg, out = tmp / 'cfg', tmp / 'out'
    cfg.mkdir(); out.mkdir()
    sc = {
        'htc': 5000.0, 't_ambient': 25.0, 'power_blocks': {'b1': 10.0},
        'tsv_map_by_layer': {'passive_probe': np.linspace(0.01, 0.05, 16).reshape(4, 4)},
    }
    sim = ICESimulator(config_dir=cfg, output_dir=out, executable='echo')
    sim._sublayers = sim._plan_sublayers(geom)
    sim._generate_layout_files(geom, sc)
    sim._generate_stack_file(geom, sc)

    rects = [tuple(float(v) for v in line.split('(')[1].split(')')[0].split(','))
             for line in (cfg / 'layout_passive_probe.lyt').read_text().splitlines() if 'rectangle' in line]
    assert rects
    area = 0.0
    for x, y, dl, dw in rects:              # 3D-ICE axes; the footprint is 2000..5000 on both
        assert 2000.0 - 1e-6 <= x and x + dl <= 5000.0 + 1e-6
        assert 2000.0 - 1e-6 <= y and y + dw <= 5000.0 + 1e-6
        area += dl * dw
    assert abs(area - 3000.0 * 3000.0) < 1.0  # the footprint is fully tiled, nothing outside it
    stk = (cfg / 'stack.stk').read_text()
    assert 'layout_passive_probe.lyt' in stk
    assert 'organic_substrate' in stk


def test_organic_substrate_material_registered():
    mat = MaterialLibrary.get('organic_substrate')
    assert 0.0 < mat.k_thermal < 148.0  # below bulk silicon, above vacuum


# ── ice_simulator: numofcores + footprint-material k-override mangling ─────

def test_solver_block_emits_numofcores():
    """
    The solver block explicitly states numofcores rather than relying on the grammar's implicit default. Default is 1 (deterministic) -- an
    """
    geom = _make_probe_geometry()
    stk, _ = _render_stack(geom)
    assert 'numofcores 1 ;' in stk  # current default -- deterministic


def test_solver_block_num_cores_is_configurable():
    tmp = Path(tempfile.mkdtemp())
    cfg, out = tmp / 'cfg', tmp / 'out'
    cfg.mkdir(); out.mkdir()
    geom = _make_probe_geometry()
    sc = {'htc': 5000.0, 't_ambient': 25.0, 'power_blocks': {'b1': 10.0}, 'num_cores': 2}
    sim = ICESimulator(config_dir=cfg, output_dir=out, executable='echo')
    sim._sublayers = sim._plan_sublayers(geom)
    sim._generate_layout_files(geom, sc)
    sim._generate_stack_file(geom, sc)
    stk = (cfg / 'stack.stk').read_text()
    assert 'numofcores 2 ;' in stk


def test_footprint_material_k_override_is_reflected_in_the_layout_file():
    """
    Regression test: overriding a FOOTPRINT-carrying layer's k (as opposed to a full uniform layer's, e.g. tim_top) mangles the material name in the
    """
    # _make_probe_geometry's 'passive_probe' layer has material='silicon'
    # (the footprint fill) regardless of gap_material -- override its k and
    # confirm BOTH the .lyt rectangle-group header and the .stk material
    # declaration use the same mangled name ('silicon_k40').
    geom = _make_probe_geometry(gap_material='organic_substrate')
    tmp = Path(tempfile.mkdtemp())
    cfg, out = tmp / 'cfg', tmp / 'out'
    cfg.mkdir(); out.mkdir()
    sc = {'htc': 5000.0, 't_ambient': 25.0, 'power_blocks': {'b1': 10.0},
          'layer_k_overrides': {'passive_probe': 40.0}}
    sim = ICESimulator(config_dir=cfg, output_dir=out, executable='echo')
    sim._sublayers = sim._plan_sublayers(geom)
    sim._generate_layout_files(geom, sc)
    lyt = (cfg / 'layout_footprint_passive_probe.lyt').read_text()
    assert 'silicon_k40 :' in lyt
    assert lyt.strip().splitlines()[2] == 'silicon_k40 :'
    # and the stack file must declare that exact mangled material name too
    sim._generate_stack_file(geom, sc)
    stk = (cfg / 'stack.stk').read_text()
    assert 'material silicon_k40 :' in stk
