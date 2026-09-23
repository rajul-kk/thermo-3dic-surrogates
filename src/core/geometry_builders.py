"""Geometry builders for benchmark geometries."""

from typing import List
from .geometry import Geometry, Layer, PowerBlock, DiePrint
from .material import MaterialLibrary


def build_geometry1() -> Geometry:
    """Build Geometry 1: 2D Cross-Sectional Stack."""
    # Get materials
    mat_cu = MaterialLibrary.get('copper')
    mat_tim = MaterialLibrary.get('tim')
    mat_si = MaterialLibrary.get('silicon')

    # Define layers (bottom to top)
    layers = [
        Layer(
            name='heat_sink',
            material='copper',
            thickness=5000.0,
            k_thermal=mat_cu.k_thermal,
            volumetric_heat_capacity=mat_cu.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='tim_bottom',
            material='tim',
            thickness=100.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='spreader',
            material='copper',
            thickness=1000.0,
            k_thermal=mat_cu.k_thermal,
            volumetric_heat_capacity=mat_cu.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='tim_top',
            material='tim',
            thickness=100.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='die',
            material='silicon',
            thickness=150.0,
            k_thermal=mat_si.k_thermal,
            volumetric_heat_capacity=mat_si.volumetric_heat_capacity,
            is_active=True
        ),
        # TIM2: die-to-cooler interface material. BC is applied at z_top of this layer
        # so the heat equation carries the contact resistance explicitly rather than
        # absorbing it into the nominal htc value.
        Layer(
            name='tim2',
            material='tim',
            thickness=50.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
    ]

    # Define power blocks (3mm × 3mm blocks in corners)
    # Die is 10mm × 10mm, place 3mm blocks with 1mm margin
    power_blocks = [
        PowerBlock(
            name='block1',
            x=1000.0,  # 1mm from left
            y=1000.0,  # 1mm from bottom
            width=3000.0,
            height=3000.0,
            die_index=0,
            layer_name='die'
        ),
        PowerBlock(
            name='block2',
            x=6000.0,  # 10mm - 1mm - 3mm = 6mm from left
            y=1000.0,
            width=3000.0,
            height=3000.0,
            die_index=0,
            layer_name='die'
        ),
        PowerBlock(
            name='block3',
            x=1000.0,
            y=6000.0,
            width=3000.0,
            height=3000.0,
            die_index=0,
            layer_name='die'
        ),
        PowerBlock(
            name='block4',
            x=6000.0,
            y=6000.0,
            width=3000.0,
            height=3000.0,
            die_index=0,
            layer_name='die'
        ),
    ]

    geometry = Geometry(
        name='geometry1',
        geometry_type='2d_stack',
        layers=layers,
        power_blocks=power_blocks,
        die_width=10000.0,
        die_length=10000.0,
        mesh_resolution=(100, 100, 40),
        tsv_density=0.0
    )

    geometry.validate()
    return geometry


def build_geometry2(tsv_density: float, variant_name: str) -> Geometry:
    """Build Geometry 2 variants: 3D Stack with TSV."""
    # Get materials
    mat_cu = MaterialLibrary.get('copper')
    mat_tim = MaterialLibrary.get('tim')
    mat_si = MaterialLibrary.get('silicon')
    mat_hb = MaterialLibrary.get('hybrid_bonding')

    # Create TSV material based on density
    mat_tsv = MaterialLibrary.create_tsv_material(tsv_density)

    # Define layers (bottom to top)
    layers = [
        Layer(
            name='heat_sink',
            material='copper',
            thickness=5000.0,
            k_thermal=mat_cu.k_thermal,
            volumetric_heat_capacity=mat_cu.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='tim_sink',
            material='tim',
            thickness=100.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='spreader',
            material='copper',
            thickness=1000.0,
            k_thermal=mat_cu.k_thermal,
            volumetric_heat_capacity=mat_cu.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='tim_die2',
            material='tim',
            thickness=100.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='die2_active',
            material='silicon',
            thickness=50.0,
            k_thermal=mat_si.k_thermal,
            volumetric_heat_capacity=mat_si.volumetric_heat_capacity,
            is_active=True
        ),
        Layer(
            name='die2_tsv',
            material=mat_tsv.name,
            thickness=100.0,
            k_thermal=mat_tsv.k_thermal,
            volumetric_heat_capacity=mat_tsv.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='hybrid_bonding',
            material='hybrid_bonding',
            thickness=5.0,
            k_thermal=mat_hb.k_thermal,
            volumetric_heat_capacity=mat_hb.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='die1_tsv',
            material=mat_tsv.name,
            thickness=100.0,
            k_thermal=mat_tsv.k_thermal,
            volumetric_heat_capacity=mat_tsv.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='die1_active',
            material='silicon',
            thickness=50.0,
            k_thermal=mat_si.k_thermal,
            volumetric_heat_capacity=mat_si.volumetric_heat_capacity,
            is_active=True
        ),
        Layer(
            name='tim2',
            material='tim',
            thickness=50.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
    ]

    # Define power blocks
    # Die is 8mm × 8mm
    # Each die has: 2 cores (2.5mm × 2.5mm) + 1 TSV region (2mm × 2mm)
    power_blocks = [
        # Die 1 (bottom die) blocks
        PowerBlock(
            name='core1_d1',
            x=1000.0,  # 1mm from left
            y=1000.0,  # 1mm from bottom
            width=2500.0,
            height=2500.0,
            die_index=0,
            layer_name='die1_active',
            is_tsv_region=False
        ),
        PowerBlock(
            name='core2_d1',
            x=4500.0,  # 8mm - 1mm - 2.5mm = 4.5mm from left
            y=1000.0,
            width=2500.0,
            height=2500.0,
            die_index=0,
            layer_name='die1_active',
            is_tsv_region=False
        ),
        PowerBlock(
            name='tsv_array_d1',
            x=3000.0,
            y=3500.0,  # moved from 3000 to avoid overlap with core1_d1/core2_d1 (y∈[1000,3500])
            width=2000.0,
            height=2000.0,
            die_index=0,
            layer_name='die1_active',
            is_tsv_region=True
        ),
        # Die 2 (top die) blocks
        PowerBlock(
            name='core1_d2',
            x=1000.0,
            y=1000.0,  # lower-left, mirrors core1_d1 position
            width=2500.0,
            height=2500.0,
            die_index=1,
            layer_name='die2_active',
            is_tsv_region=False
        ),
        PowerBlock(
            name='core2_d2',
            x=4500.0,
            y=5500.0,  # upper-right; die2 uses diagonal layout (core1 lower-left, core2 upper-right)
            width=2500.0,
            height=2500.0,
            die_index=1,
            layer_name='die2_active',
            is_tsv_region=False
        ),
        PowerBlock(
            name='tsv_array_d2',
            x=3000.0,
            y=3500.0,  # moved from 3000 to avoid overlap with core1_d2 (y∈[1000,3500])
            width=2000.0,
            height=2000.0,
            die_index=1,
            layer_name='die2_active',
            is_tsv_region=True
        ),
    ]

    geometry = Geometry(
        name=variant_name,
        geometry_type='3d_stack',
        layers=layers,
        power_blocks=power_blocks,
        die_width=8000.0,
        die_length=8000.0,
        # nz=72: per-layer minimums sum to 48 (10 layers incl. tim2); 24 extra pts
        # go proportionally to the heat sink. Die layers are guaranteed 8 pts each.
        mesh_resolution=(80, 80, 72),
        tsv_density=tsv_density
    )

    geometry.validate()
    return geometry


def build_geometry3() -> Geometry:
    """Build Geometry 3: Server-class single die (25mm × 25mm)."""
    mat_cu = MaterialLibrary.get('copper')
    mat_tim = MaterialLibrary.get('tim')
    mat_si = MaterialLibrary.get('silicon')

    layers = [
        Layer(
            name='heat_sink',
            material='copper',
            thickness=5000.0,
            k_thermal=mat_cu.k_thermal,
            volumetric_heat_capacity=mat_cu.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='tim_bottom',
            material='tim',
            thickness=100.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='spreader',
            material='copper',
            thickness=2000.0,  # 2× geometry1 to handle higher server TDP
            k_thermal=mat_cu.k_thermal,
            volumetric_heat_capacity=mat_cu.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='tim_top',
            material='tim',
            thickness=100.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
        Layer(
            name='die',
            material='silicon',
            thickness=200.0,  # slightly thicker than geometry1 (150 µm)
            k_thermal=mat_si.k_thermal,
            volumetric_heat_capacity=mat_si.volumetric_heat_capacity,
            is_active=True
        ),
        Layer(
            name='tim2',
            material='tim',
            thickness=50.0,
            k_thermal=mat_tim.k_thermal,
            volumetric_heat_capacity=mat_tim.volumetric_heat_capacity,
            is_active=False
        ),
    ]

    # 8 core-cluster blocks: 4 bottom row + 4 top row
    # Block: 4mm × 4mm; pitch 5.5mm; margins: left/right ~2mm, top/bottom ~5mm
    _block_w = 4000.0
    _block_h = 4000.0
    _x_positions = [2000.0, 7500.0, 13000.0, 18500.0]
    _y_rows = [2000.0, 16000.0]  # gap y∈[6000,16000] = IO area

    power_blocks = []
    for row_idx, y in enumerate(_y_rows):
        for col_idx, x in enumerate(_x_positions):
            power_blocks.append(PowerBlock(
                name=f'core{row_idx * 4 + col_idx + 1}',
                x=x,
                y=y,
                width=_block_w,
                height=_block_h,
                die_index=0,
                layer_name='die',
                is_tsv_region=False
            ))

    geometry = Geometry(
        name='geometry3',
        geometry_type='2d_stack',
        layers=layers,
        power_blocks=power_blocks,
        die_width=25000.0,
        die_length=25000.0,
        # 100×100×40 = 400k points, same as geometry1 — CPU training cost equivalent.
        # Cell resolution is 250 µm/cell in x/y vs geometry1's 100 µm/cell.
        mesh_resolution=(100, 100, 40),
        tsv_density=0.0
    )

    geometry.validate()
    return geometry


def build_geometry2a() -> Geometry:
    """Build Geometry 2a: 3D Stack with 3% TSV density."""
    return build_geometry2(0.03, 'geometry2a')


def build_geometry4() -> Geometry:
    """Geometry 4: 2.5D chiplet assembly — two chiplets on a silicon interposer."""
    mat_cu = MaterialLibrary.get('copper')
    mat_tim = MaterialLibrary.get('tim')
    mat_si = MaterialLibrary.get('silicon')

    layers = [
        Layer(name='heat_sink', material='copper', thickness=5000.0,
              k_thermal=mat_cu.k_thermal,
              volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        Layer(name='tim_sink', material='tim', thickness=100.0,
              k_thermal=mat_tim.k_thermal,
              volumetric_heat_capacity=mat_tim.volumetric_heat_capacity),
        Layer(name='spreader', material='copper', thickness=1000.0,
              k_thermal=mat_cu.k_thermal,
              volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        Layer(name='tim_die', material='tim', thickness=100.0,
              k_thermal=mat_tim.k_thermal,
              volumetric_heat_capacity=mat_tim.volumetric_heat_capacity),
        Layer(name='die_zone', material='silicon', thickness=150.0,
              k_thermal=mat_si.k_thermal,
              volumetric_heat_capacity=mat_si.volumetric_heat_capacity,
              is_active=True),
        Layer(name='interposer', material='silicon', thickness=100.0,
              k_thermal=mat_si.k_thermal,
              volumetric_heat_capacity=mat_si.volumetric_heat_capacity),
    ]

    power_blocks = [
        # Chiplet A (10×12mm at x=2000, y=1000) — 2×2 grid of 5×6mm blocks
        PowerBlock(name='chipA_c1', x=2000.0, y=1000.0,
                   width=5000.0, height=6000.0, layer_name='die_zone'),
        PowerBlock(name='chipA_c2', x=7000.0, y=1000.0,
                   width=5000.0, height=6000.0, layer_name='die_zone'),
        PowerBlock(name='chipA_c3', x=2000.0, y=7000.0,
                   width=5000.0, height=6000.0, layer_name='die_zone'),
        PowerBlock(name='chipA_c4', x=7000.0, y=7000.0,
                   width=5000.0, height=6000.0, layer_name='die_zone'),
        # Chiplet B (8×12mm at x=15000, y=1000) — 2×2 grid of 4×6mm blocks
        PowerBlock(name='chipB_c1', x=15000.0, y=1000.0,
                   width=4000.0, height=6000.0, layer_name='die_zone'),
        PowerBlock(name='chipB_c2', x=19000.0, y=1000.0,
                   width=4000.0, height=6000.0, layer_name='die_zone'),
        PowerBlock(name='chipB_c3', x=15000.0, y=7000.0,
                   width=4000.0, height=6000.0, layer_name='die_zone'),
        PowerBlock(name='chipB_c4', x=19000.0, y=7000.0,
                   width=4000.0, height=6000.0, layer_name='die_zone'),
    ]

    die_footprints = [
        DiePrint(name='chiplet_a', x=2000.0, y=1000.0,
                 width=10000.0, height=12000.0, die_layer_name='die_zone'),
        DiePrint(name='chiplet_b', x=15000.0, y=1000.0,
                 width=8000.0, height=12000.0, die_layer_name='die_zone'),
    ]

    geometry = Geometry(
        name='geometry4',
        geometry_type='2p5d_stack',
        layers=layers,
        power_blocks=power_blocks,
        die_width=25000.0,
        die_length=14000.0,
        mesh_resolution=(56, 100, 40),  # (length, width): 14000/56=250µm, 25000/100=250µm
        die_footprints=die_footprints,
        underfill_k=0.7,
    )
    geometry.validate()
    return geometry


def build_geometry5() -> Geometry:
    """Geometry 5 (Tier 0+1 upgraded): CoWoS-style HBM-stack + compute die on shared interposer."""
    mat_cu    = MaterialLibrary.get('copper')
    mat_tim   = MaterialLibrary.get('tim')
    mat_si    = MaterialLibrary.get('silicon')
    mat_si_lk = MaterialLibrary.get('silicon_low_k')
    mat_tim1  = MaterialLibrary.get('tim_indium')
    mat_c4    = MaterialLibrary.get('c4_bump_array')
    mat_hb    = MaterialLibrary.get('hybrid_bonding')
    mat_tsv   = MaterialLibrary.create_tsv_material(0.03)

    layers = [
        Layer(name='heat_sink',  material='copper',       thickness=5000.0,
              k_thermal=mat_cu.k_thermal,    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        # TIM2 grease: 125µm (Tier 1)
        Layer(name='tim_sink',   material='tim',          thickness=125.0,
              k_thermal=mat_tim.k_thermal,   volumetric_heat_capacity=mat_tim.volumetric_heat_capacity),
        Layer(name='spreader',   material='copper',       thickness=1000.0,
              k_thermal=mat_cu.k_thermal,    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        # TIM1 indium solder: 50µm, k=80 W/m·K (Tier 1)
        Layer(name='tim_top',    material='tim_indium',   thickness=50.0,
              k_thermal=mat_tim1.k_thermal,  volumetric_heat_capacity=mat_tim1.volumetric_heat_capacity),
        # Interposer: 300µm bulk Si (Kou 2022 published dims, Tier 1)
        Layer(name='interposer', material='silicon',      thickness=300.0,
              k_thermal=mat_si.k_thermal,    volumetric_heat_capacity=mat_si.volumetric_heat_capacity),
        # RDL self-heating layer: 5µm, active (Tier 0)
        Layer(name='rdl_layer',  material='silicon_low_k', thickness=5.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
        # C4 bump array: 100µm, k_eff=15 W/m·K (Tier 1)
        Layer(name='c4_bumps',   material='c4_bump_array', thickness=100.0,
              k_thermal=mat_c4.k_thermal,    volumetric_heat_capacity=mat_c4.volumetric_heat_capacity),
        # Active die layers at N5/N3 low-k effective conductivity (Tier 0)
        Layer(name='die_zone_1', material='silicon_low_k', thickness=50.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
        # Cu-Cu hybrid bonding interface between die1 and TSV zone (5µm, k=60)
        Layer(name='hybrid_bonding', material='hybrid_bonding', thickness=5.0,
              k_thermal=mat_hb.k_thermal,    volumetric_heat_capacity=mat_hb.volumetric_heat_capacity),
        Layer(name='tsv_zone',   material=mat_tsv.name,  thickness=100.0,
              k_thermal=mat_tsv.k_thermal,   volumetric_heat_capacity=mat_tsv.volumetric_heat_capacity),
        Layer(name='die_zone_2', material='silicon_low_k', thickness=50.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
    ]

    power_blocks = [
        # RDL self-heating blocks (Tier 0) — always 5% of base_power via pattern post-process
        PowerBlock(name='chipA_rdl', x=2000.0,  y=1000.0, width=10000.0, height=12000.0,
                   layer_name='rdl_layer'),
        PowerBlock(name='chipB_rdl', x=15000.0, y=1000.0, width=8000.0,  height=12000.0,
                   layer_name='rdl_layer'),
        # Chiplet A — die_zone_1 only (2×2 grid, 5×6mm blocks)
        PowerBlock(name='chipA_c1', x=2000.0, y=1000.0, width=5000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c2', x=7000.0, y=1000.0, width=5000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c3', x=2000.0, y=7000.0, width=5000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c4', x=7000.0, y=7000.0, width=5000.0, height=6000.0, layer_name='die_zone_1'),
        # Chiplet B die1 — die_zone_1 (2×2 grid, 4×6mm blocks)
        PowerBlock(name='chipB_d1_c1', x=15000.0, y=1000.0, width=4000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipB_d1_c2', x=19000.0, y=1000.0, width=4000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipB_d1_c3', x=15000.0, y=7000.0, width=4000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipB_d1_c4', x=19000.0, y=7000.0, width=4000.0, height=6000.0, layer_name='die_zone_1'),
        # Chiplet B TSV region — tsv_zone (passive; lateral k override)
        PowerBlock(name='chipB_tsv', x=15000.0, y=1000.0, width=8000.0, height=12000.0,
                   layer_name='tsv_zone', is_tsv_region=True),
        # Chiplet B die2 — die_zone_2 (2×2 grid, 4×6mm blocks)
        PowerBlock(name='chipB_d2_c1', x=15000.0, y=1000.0, width=4000.0, height=6000.0, layer_name='die_zone_2'),
        PowerBlock(name='chipB_d2_c2', x=19000.0, y=1000.0, width=4000.0, height=6000.0, layer_name='die_zone_2'),
        PowerBlock(name='chipB_d2_c3', x=15000.0, y=7000.0, width=4000.0, height=6000.0, layer_name='die_zone_2'),
        PowerBlock(name='chipB_d2_c4', x=19000.0, y=7000.0, width=4000.0, height=6000.0, layer_name='die_zone_2'),
    ]

    die_footprints = [
        DiePrint('chiplet_a_die1', x=2000.0,  y=1000.0, width=10000.0, height=12000.0, die_layer_name='die_zone_1'),
        DiePrint('chiplet_b_die1', x=15000.0, y=1000.0, width=8000.0,  height=12000.0, die_layer_name='die_zone_1'),
        DiePrint('chiplet_b_tsv',  x=15000.0, y=1000.0, width=8000.0,  height=12000.0, die_layer_name='tsv_zone'),
        DiePrint('chiplet_b_die2', x=15000.0, y=1000.0, width=8000.0,  height=12000.0, die_layer_name='die_zone_2'),
    ]

    geometry = Geometry(
        name='geometry5',
        geometry_type='2p5d_stack',
        layers=layers,
        power_blocks=power_blocks,
        die_width=25000.0,
        die_length=14000.0,
        mesh_resolution=(56, 100, 50),  # (length, width): 14000/56=250µm, 25000/100=250µm
        die_footprints=die_footprints,
        underfill_k=0.7,
        tsv_density=0.03,  # matches create_tsv_material(0.03) above; without this
                           # ScenarioGenerator.attach_tsv_maps no-ops (its guard is
                           # `geometry.tsv_density <= 0`) and tsv_zone stays a single
                           # uniform-k material for every scenario.
    )
    geometry.validate()
    return geometry


def build_geometry6() -> Geometry:
    """Geometry 6: geometry5 + 6 HBM stacks — CoWoS-style with full HBM complement."""
    mat_cu    = MaterialLibrary.get('copper')
    mat_tim   = MaterialLibrary.get('tim')
    mat_si    = MaterialLibrary.get('silicon')
    mat_si_lk = MaterialLibrary.get('silicon_low_k')
    mat_tim1  = MaterialLibrary.get('tim_indium')
    mat_c4    = MaterialLibrary.get('c4_bump_array')
    mat_hb    = MaterialLibrary.get('hybrid_bonding')
    mat_tsv   = MaterialLibrary.create_tsv_material(0.03)

    # Layer stack identical to upgraded geometry5 (11 layers)
    layers = [
        Layer(name='heat_sink',  material='copper',        thickness=5000.0,
              k_thermal=mat_cu.k_thermal,    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        Layer(name='tim_sink',   material='tim',           thickness=125.0,
              k_thermal=mat_tim.k_thermal,   volumetric_heat_capacity=mat_tim.volumetric_heat_capacity),
        Layer(name='spreader',   material='copper',        thickness=1000.0,
              k_thermal=mat_cu.k_thermal,    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        Layer(name='tim_top',    material='tim_indium',    thickness=50.0,
              k_thermal=mat_tim1.k_thermal,  volumetric_heat_capacity=mat_tim1.volumetric_heat_capacity),
        Layer(name='interposer', material='silicon',       thickness=300.0,
              k_thermal=mat_si.k_thermal,    volumetric_heat_capacity=mat_si.volumetric_heat_capacity),
        Layer(name='rdl_layer',  material='silicon_low_k', thickness=5.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
        Layer(name='c4_bumps',   material='c4_bump_array', thickness=100.0,
              k_thermal=mat_c4.k_thermal,    volumetric_heat_capacity=mat_c4.volumetric_heat_capacity),
        Layer(name='die_zone_1', material='silicon_low_k', thickness=50.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
        Layer(name='hybrid_bonding', material='hybrid_bonding', thickness=5.0,
              k_thermal=mat_hb.k_thermal,    volumetric_heat_capacity=mat_hb.volumetric_heat_capacity),
        Layer(name='tsv_zone',   material=mat_tsv.name,   thickness=100.0,
              k_thermal=mat_tsv.k_thermal,   volumetric_heat_capacity=mat_tsv.volumetric_heat_capacity),
        Layer(name='die_zone_2', material='silicon_low_k', thickness=50.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
    ]

    # 6 HBM stacks at 5mm pitch (all span y=1000–13000, 4×12mm each)
    _hbm_x = {1: 12000.0, 2: 17000.0, 3: 22000.0, 4: 27000.0, 5: 32000.0, 6: 37000.0}
    _hbm_w, _hbm_h = 4000.0, 12000.0
    _y0 = 1000.0

    power_blocks = [
        PowerBlock(name='chipA_rdl', x=1000.0, y=_y0, width=10000.0, height=12000.0, layer_name='rdl_layer'),
    ]
    for n, hx in _hbm_x.items():
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_rdl', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, layer_name='rdl_layer'))

    # Chiplet A — die_zone_1 (2×2 grid, 5×6mm blocks)
    power_blocks += [
        PowerBlock(name='chipA_c1', x=1000.0, y=_y0,          width=5000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c2', x=6000.0, y=_y0,          width=5000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c3', x=1000.0, y=_y0 + 6000.0, width=5000.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c4', x=6000.0, y=_y0 + 6000.0, width=5000.0, height=6000.0, layer_name='die_zone_1'),
    ]

    # 6 HBM stacks: die_zone_1 bottom die, tsv_zone passive, die_zone_2 top die
    for n, hx in _hbm_x.items():
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_d1', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, layer_name='die_zone_1'))
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_tsv', x=hx, y=_y0, width=_hbm_w, height=_hbm_h,
            layer_name='tsv_zone', is_tsv_region=True))
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_d2', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, layer_name='die_zone_2'))

    die_footprints = [
        DiePrint('chiplet_a_die1', x=1000.0, y=_y0, width=10000.0, height=12000.0, die_layer_name='die_zone_1'),
    ]
    for n, hx in _hbm_x.items():
        die_footprints += [
            DiePrint(f'hbm{n}_die1', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, die_layer_name='die_zone_1'),
            DiePrint(f'hbm{n}_tsv',  x=hx, y=_y0, width=_hbm_w, height=_hbm_h, die_layer_name='tsv_zone'),
            DiePrint(f'hbm{n}_die2', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, die_layer_name='die_zone_2'),
        ]

    geometry = Geometry(
        name='geometry6',
        geometry_type='2p5d_stack',
        layers=layers,
        power_blocks=power_blocks,
        die_width=42000.0,
        die_length=14000.0,
        mesh_resolution=(56, 168, 50),  # 14000/56=250µm, 42000/168=250µm — integer cell sizes
        die_footprints=die_footprints,
        underfill_k=0.7,
        tsv_density=0.03,  # see build_geometry5 -- same missing-kwarg bug applied here
    )
    geometry.validate()
    return geometry


def build_geometry7() -> Geometry:
    """Geometry 7: CoWoS-L-style package (2 compute + 2 I/O dies + 8 HBM4), reduced scale -- not a Rubin model."""
    mat_cu    = MaterialLibrary.get('copper')
    mat_tim   = MaterialLibrary.get('tim')
    mat_si_lk = MaterialLibrary.get('silicon_low_k')
    mat_tim1  = MaterialLibrary.get('tim_indium')
    mat_c4    = MaterialLibrary.get('c4_bump_array')
    mat_hb    = MaterialLibrary.get('hybrid_bonding')
    mat_bridge = MaterialLibrary.get('lsi_bridge_via')
    mat_tsv   = MaterialLibrary.create_tsv_material(0.03)

    layers = [
        Layer(name='heat_sink',  material='copper',        thickness=5000.0,
              k_thermal=mat_cu.k_thermal,    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        Layer(name='tim_sink',   material='tim',           thickness=125.0,
              k_thermal=mat_tim.k_thermal,   volumetric_heat_capacity=mat_tim.volumetric_heat_capacity),
        Layer(name='spreader',   material='copper',        thickness=1000.0,
              k_thermal=mat_cu.k_thermal,    volumetric_heat_capacity=mat_cu.volumetric_heat_capacity),
        Layer(name='tim_top',    material='tim_indium',    thickness=50.0,
              k_thermal=mat_tim1.k_thermal,  volumetric_heat_capacity=mat_tim1.volumetric_heat_capacity),
        # CoWoS-L bridge layer: base/gap material is the organic substrate;
        # `material='lsi_bridge_via'` supplies the bridge/via islands via
        # DiePrint footprints below (see gap_material on Layer -- the footprint
        # fills with `material`, the surrounding field fills with `gap_material`).
        Layer(name='substrate_organic', material='lsi_bridge_via', thickness=300.0,
              k_thermal=mat_bridge.k_thermal,
              volumetric_heat_capacity=mat_bridge.volumetric_heat_capacity,
              gap_material='organic_substrate'),
        Layer(name='rdl_layer',  material='silicon_low_k', thickness=5.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
        Layer(name='c4_bumps',   material='c4_bump_array', thickness=100.0,
              k_thermal=mat_c4.k_thermal,    volumetric_heat_capacity=mat_c4.volumetric_heat_capacity),
        Layer(name='die_zone_1', material='silicon_low_k', thickness=50.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
        Layer(name='hybrid_bonding', material='hybrid_bonding', thickness=5.0,
              k_thermal=mat_hb.k_thermal,    volumetric_heat_capacity=mat_hb.volumetric_heat_capacity),
        Layer(name='tsv_zone',   material=mat_tsv.name,   thickness=100.0,
              k_thermal=mat_tsv.k_thermal,   volumetric_heat_capacity=mat_tsv.volumetric_heat_capacity),
        Layer(name='die_zone_2', material='silicon_low_k', thickness=50.0,
              k_thermal=mat_si_lk.k_thermal, volumetric_heat_capacity=mat_si_lk.volumetric_heat_capacity,
              is_active=True),
    ]

    _y0 = 1000.0
    # 8 HBM4 stacks at 4.5mm pitch (500um gap between 4mm-wide stacks)
    _hbm_x = {n: 24500.0 + (n - 1) * 4500.0 for n in range(1, 9)}
    _hbm_w, _hbm_h = 4000.0, 12000.0

    power_blocks = [
        PowerBlock(name='chipA_rdl', x=1000.0,  y=_y0, width=9000.0, height=12000.0, layer_name='rdl_layer'),
        PowerBlock(name='chipB_rdl', x=10500.0, y=_y0, width=9000.0, height=12000.0, layer_name='rdl_layer'),
        PowerBlock(name='io1_rdl',   x=20000.0, y=_y0,          width=3000.0, height=5500.0, layer_name='rdl_layer'),
        PowerBlock(name='io2_rdl',   x=20000.0, y=_y0 + 6500.0, width=3000.0, height=5500.0, layer_name='rdl_layer'),
    ]
    for n, hx in _hbm_x.items():
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_rdl', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, layer_name='rdl_layer'))

    # Compute dies -- 2x2 sub-block grid each (4.5x6mm blocks), matching the
    # spatial resolution geometry5/6 already use for their single compute chiplet.
    power_blocks += [
        PowerBlock(name='chipA_c1', x=1000.0,  y=_y0,          width=4500.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c2', x=5500.0,  y=_y0,          width=4500.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c3', x=1000.0,  y=_y0 + 6000.0, width=4500.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipA_c4', x=5500.0,  y=_y0 + 6000.0, width=4500.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipB_c1', x=10500.0, y=_y0,          width=4500.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipB_c2', x=15000.0, y=_y0,          width=4500.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipB_c3', x=10500.0, y=_y0 + 6000.0, width=4500.0, height=6000.0, layer_name='die_zone_1'),
        PowerBlock(name='chipB_c4', x=15000.0, y=_y0 + 6000.0, width=4500.0, height=6000.0, layer_name='die_zone_1'),
        # I/O dies -- SerDes/analog, single block each (not compute-grade hotspot density)
        PowerBlock(name='io1', x=20000.0, y=_y0,          width=3000.0, height=5500.0, layer_name='die_zone_1'),
        PowerBlock(name='io2', x=20000.0, y=_y0 + 6500.0, width=3000.0, height=5500.0, layer_name='die_zone_1'),
    ]

    # 8 HBM4 stacks: die_zone_1 bottom die, tsv_zone passive, die_zone_2 top die
    for n, hx in _hbm_x.items():
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_d1', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, layer_name='die_zone_1'))
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_tsv', x=hx, y=_y0, width=_hbm_w, height=_hbm_h,
            layer_name='tsv_zone', is_tsv_region=True))
        power_blocks.append(PowerBlock(
            name=f'hbm{n}_d2', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, layer_name='die_zone_2'))

    die_footprints = [
        DiePrint('chipA_die1', x=1000.0,  y=_y0,          width=9000.0, height=12000.0, die_layer_name='die_zone_1'),
        DiePrint('chipB_die1', x=10500.0, y=_y0,          width=9000.0, height=12000.0, die_layer_name='die_zone_1'),
        DiePrint('io1_die',    x=20000.0, y=_y0,          width=3000.0, height=5500.0,  die_layer_name='die_zone_1'),
        DiePrint('io2_die',    x=20000.0, y=_y0 + 6500.0, width=3000.0, height=5500.0,  die_layer_name='die_zone_1'),
        # LSI bridge/via composite in the organic-substrate field, only where
        # die-to-die routing or power delivery density needs it -- the CoWoS-L
        # structural feature this geometry exists to test. Full footprint under
        # each compute die (dense power/ground via coverage, 2026-08-18 fix);
        # the 0.5mm true reticle-stitch gap between chipA and chipB (x=10-10.5mm)
        # is deliberately left as bare organic substrate.
        DiePrint('bridge_chipA', x=1000.0,  y=_y0, width=9000.0, height=12000.0,
                 die_layer_name='substrate_organic'),
        DiePrint('bridge_chipB', x=10500.0, y=_y0, width=9000.0, height=12000.0,
                 die_layer_name='substrate_organic'),
    ]
    for n, hx in _hbm_x.items():
        die_footprints += [
            DiePrint(f'hbm{n}_die1', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, die_layer_name='die_zone_1'),
            DiePrint(f'hbm{n}_tsv',  x=hx, y=_y0, width=_hbm_w, height=_hbm_h, die_layer_name='tsv_zone'),
            DiePrint(f'hbm{n}_die2', x=hx, y=_y0, width=_hbm_w, height=_hbm_h, die_layer_name='die_zone_2'),
            DiePrint(f'bridge_hbm{n}', x=hx - 500.0, y=_y0, width=1000.0, height=_hbm_h,
                     die_layer_name='substrate_organic'),
        ]

    geometry = Geometry(
        name='geometry7',
        geometry_type='2p5d_stack',
        layers=layers,
        power_blocks=power_blocks,
        die_width=62000.0,
        die_length=14000.0,
        mesh_resolution=(56, 248, 50),  # 14000/56=250µm, 62000/248=250µm — integer cell sizes
        die_footprints=die_footprints,
        underfill_k=0.7,
        tsv_density=0.03,
    )
    geometry.validate()
    return geometry


def build_all_geometries() -> List[Geometry]:
    """Build all benchmark geometries."""
    return [
        build_geometry1(),
        build_geometry2a(),
        build_geometry3(),
        build_geometry4(),
        build_geometry5(),
        build_geometry6(),
    ]


def get_geometry_by_name(name: str) -> Geometry:
    """Get geometry by name."""
    builders = {
        'geometry1':  build_geometry1,
        'geometry2a': build_geometry2a,
        'geometry3':  build_geometry3,
        'geometry4':  build_geometry4,
        'geometry5':  build_geometry5,
        'geometry6':  build_geometry6,
        'geometry7':  build_geometry7,
    }

    if name not in builders:
        raise ValueError(
            f"Invalid geometry name: '{name}'. "
            f"Valid names: {', '.join(builders.keys())}"
        )

    return builders[name]()
