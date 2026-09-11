"""
Microchannel liquid cooling: an opt-in `cooling_mode='microchannel_2rm'` scenario mode that replaces the idealised `bottom heat sink` boundary condition with an
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry import Geometry, Layer
from src.core.geometry_builders import build_geometry1
from src.core.material import MaterialLibrary
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator


def _microchannel_geometry():
    """geometry1 with a copper base plate + a microchannel-cooled heat_sink."""
    base = build_geometry1()
    mat_cu = MaterialLibrary.get('copper')
    cp_base = Layer(name='cp_base', material='copper', thickness=500.0,
                    k_thermal=mat_cu.k_thermal,
                    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity)
    return Geometry(
        name=base.name, geometry_type=base.geometry_type,
        layers=[cp_base] + base.layers, power_blocks=base.power_blocks,
        die_width=base.die_width, die_length=base.die_length,
        mesh_resolution=base.mesh_resolution, tsv_density=base.tsv_density,
        coolant_layer_name='heat_sink',
    )


def _configure(geometry, cooling_mode=None):
    gen = ScenarioGenerator()
    scenario = gen.generate_all_scenarios(geometry)[0]
    d = scenario.to_dict()
    if cooling_mode:
        d['cooling_mode'] = cooling_mode
        d['microchannel'] = {
            'channel_height_um': 200.0, 'channel_length_um': 100.0,
            'wall_length_um': 100.0, 'wall_material': 'copper',
            'flow_rate_ml_min': 200.0, 'coolant_htc_top_wm2k': 25000.0,
            'coolant_htc_bottom_wm2k': 5000.0, 'coolant_vhc_jm3k': 4.18e6,
        }
    tmp = Path(tempfile.mkdtemp())
    sim = ICESimulator(config_dir=tmp, output_dir=tmp / "out",
                        executable="wsl /bin/true")
    (tmp / "out").mkdir(exist_ok=True)
    sim.generate_config_files(geometry, d)
    return sim, tmp


def test_coolant_layer_forced_to_a_single_sublayer():
    geometry = _microchannel_geometry()
    sim, tmp = _configure(geometry, cooling_mode='microchannel_2rm')
    coolant_subs = [s for s in sim._sublayers if s.get('is_coolant')]
    assert len(coolant_subs) == 1
    # heat_sink is 5000um, well above MAX_SUBLAYER_UM=1200 -- would normally split
    assert coolant_subs[0]['thickness'] == pytest.approx(5000.0)


def test_stack_file_emits_microchannel_block_and_channel_element():
    geometry = _microchannel_geometry()
    sim, tmp = _configure(geometry, cooling_mode='microchannel_2rm')
    stk = (tmp / "stack.stk").read_text()

    assert "microchannel 2rm :" in stk
    assert "bottom heat sink :" not in stk
    assert "channel inst_" in stk
    # No `layer type_layer_N :` declaration for the coolant sublayer -- it has
    # no material/height declaration of its own, only the channel stack entry.
    coolant_idx = next(i for i, s in enumerate(sim._sublayers) if s.get('is_coolant'))
    assert f"layer type_layer_{coolant_idx} :" not in stk


def test_no_tmap_requested_for_the_coolant_element():
    geometry = _microchannel_geometry()
    sim, tmp = _configure(geometry, cooling_mode='microchannel_2rm')
    stk = (tmp / "stack.stk").read_text()
    coolant_idx = next(i for i, s in enumerate(sim._sublayers) if s.get('is_coolant'))
    assert f"Tmap ( inst_{coolant_idx}," not in stk


def test_default_cooling_mode_is_unaffected():
    """Regression guard: geometries/scenarios without cooling_mode see no change."""
    geometry = build_geometry1()
    sim, tmp = _configure(geometry, cooling_mode=None)
    stk = (tmp / "stack.stk").read_text()
    assert "bottom heat sink :" in stk
    assert "microchannel" not in stk
    assert "channel inst_" not in stk
    assert not any(s.get('is_coolant') for s in sim._sublayers)


def test_geometry_without_coolant_layer_name_raises_loudly():
    """coolant_layer_name=None (the default) means no layer can become a channel."""
    geometry = build_geometry1()
    assert geometry.coolant_layer_name is None
    with pytest.raises(ValueError, match="coolant_layer_name"):
        _configure(geometry, cooling_mode='microchannel_2rm')
