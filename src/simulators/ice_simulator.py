"""3D-ICE thermal simulator wrapper."""

import subprocess
import shlex
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from .base_simulator import ThermalSimulator
from ..core.geometry import Geometry, Layer, PowerBlock


class ICESimulator(ThermalSimulator):
    """Wrapper for 3D-ICE thermal simulator."""

    def __init__(self, config_dir: Path, output_dir: Path, executable: str = "3D-ICE-Emulator"):
        super().__init__(config_dir, output_dir, executable)

    def generate_config_files(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """Generate 3D-ICE configuration files."""
        exe_parts = shlex.split(self.executable) if self.executable else []
        if not exe_parts:
            raise RuntimeError(
                "3D-ICE executable not configured. "
                "Pass --ice-executable \"wsl /home/<user>/3d-ice/bin/3D-ICE-Emulator\""
            )
        self._use_wsl = exe_parts[0] == 'wsl'

        self._geometry = geometry
        nx, ny, _ = geometry.mesh_resolution
        cell_length = geometry.die_length / nx
        cell_width = geometry.die_width / ny
        self._x_centers = np.linspace(cell_length / 2, geometry.die_length - cell_length / 2, nx)
        self._y_centers = np.linspace(cell_width / 2, geometry.die_width - cell_width / 2, ny)
        self._sublayers = self._plan_sublayers(geometry)
        self._layer_names = [layer.name for layer in geometry.layers]
        # One z-centre per EMITTED stack element (sub-layer), because 3D-ICE writes
        # one Tmap per stack element and inst_N indexes that list.
        self._layer_z_centers = np.array([s['z_center'] for s in self._sublayers])
        self._nx = nx
        self._ny = ny

        # inst_N maps to the ORIGINAL geometry layer a sub-layer came from.
        self._inst_to_layer_idx = {
            f"inst_{i}": s['layer_idx'] for i, s in enumerate(self._sublayers)
        }

        self._generate_layout_files(geometry, scenario)
        self._generate_floorplan_files(geometry, scenario)
        self._generate_stack_file(geometry, scenario)

    # Vertical discretisation. 3D-ICE's compact model puts ONE temperature node at
    # the centre of each stack element, so z-resolution equals the number of elements
    # emitted -- not the geometry's declared mesh nz. Emitting one element per
    # geometry layer gave geometry1 six z-nodes spanning 6.4 mm, with spacing ranging
    # 100-2550 um (25x non-uniform); geometry5/6 reached 93x. That is the direction
    # heat actually flows in a layered stack, and it starved every downstream model:
    # an FNO asked for 12 spectral z-modes silently got min(12, 6//2+1) = 4.
    #
    # Splitting a layer into N sub-layers of the same material leaves the physics
    # untouched -- same total thickness, same conductivity -- and multiplies the
    # nodes through it. Only non-active layers are split; active dies carry the
    # `source` term and stay single elements so the floorplan mapping is unchanged.
    # Chosen against the solver's real limits, measured on geometry1:
    #
    #   z-nodes   nodes    solve      status
    #        6     60k      4.0 s     OK   (one element per geometry layer)
    #        8     80k      7.8 s     OK
    #       10    100k     13.1 s     OK
    #       13    130k     35.7 s     OK
    #       28    280k        --      FAILS: SuperLU "Storage for U columns
    #                                 exceeded ... need 96441435", 3D-ICE uses a
    #                                 direct sparse LU whose fill-in blows up.
    #
    # Solve cost grows roughly as nodes^2.6, so resolution is bought steeply:
    # 1200 um gives 10-15 z-nodes across the suite for ~1.5 h of regeneration,
    # while 800 um gives 13-18 for ~2.5 h. 1200 roughly doubles the usable FNO
    # z-modes (4 -> 6-8) at a cost comparable to the existing pipeline.
    MAX_SUBLAYER_UM = 1200.0
    MAX_SUBLAYERS_PER_LAYER = 12  # bounds fill-in on very thick heat sinks

    # A die holds exactly one `source` element (grammar: top layers, source, bottom
    # layers), so an active layer cannot be split like a passive one. With a single node,
    # geometry4's 150 um chiplet layer was 11-40% hot against a converged FV solve
    # (docs/report.md 9.23). Active layers thicker than this are emitted as
    # layer / source / layer thirds: three nodes, power in the middle third, and the
    # Tmap (source) node still sits at the layer's centre, so exports are unchanged.
    ACTIVE_SPLIT_UM = 100.0

    # Number of quantised conductivity levels per TSV layer. 3D-ICE needs a named
    # material per distinct conductivity, so a continuous field would mean one
    # material per cell. ~12 keeps the stack file small while preserving the
    # spatial structure; the quantisation error is far below the modelling error
    # in the rule of mixtures itself.
    TSV_MATERIAL_LEVELS = 12

    def _tsv_levels(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Quantise each layer's TSV density field into material levels."""
        from ..scenario.tsv_maps import quantise_materials

        out = {}
        for layer_name, phi in (scenario.get('tsv_map_by_layer') or {}).items():
            phi = np.asarray(phi, dtype=np.float64)
            idx, centres = quantise_materials(phi, self.TSV_MATERIAL_LEVELS)
            out[layer_name] = (idx, centres)
        return out

    def _tsv_materials(self, scenario: Dict[str, Any]) -> Dict[str, tuple]:
        """{material_name: (k_lateral, k_vertical, rho_cp)} for every TSV level."""
        from ..scenario.tsv_maps import tsv_effective_k

        mats = {}
        for layer_name, (_, centres) in self._tsv_levels(scenario).items():
            safe = self._sanitise(layer_name)
            for li, phi in enumerate(centres):
                k_lat, k_vert = tsv_effective_k(float(phi))
                # Volumetric heat capacity blends the same way (simple mixture).
                rho_cp = (1.0 - phi) * 1.628e6 + phi * 3.45e6
                mats[f"tsvmat_{safe}_{li}"] = (float(k_lat), float(k_vert), rho_cp)
        return mats

    @staticmethod
    def _sanitise(name: str) -> str:
        """3D-ICE identifiers must not contain separators."""
        return ''.join(c if c.isalnum() else '_' for c in name)

    # Chiplet/interposer die layers (geometry4/5/6) carry Si only under each
    # DiePrint footprint; the rest of the layer is underfill (k=0.7 W/m·K vs
    # Si's 148). 3D-ICE's own default-material fallback (any cell not covered
    # by a layout rectangle keeps the layer's declared base material -- see
    # sources/layer.c:get_thermal_conductivity) means the base material can be
    # underfill and only the die footprints need explicit rectangles, which
    # DiePrint already stores exactly. See assumptions.md §6.1 -- this was
    # previously a real train/target inconsistency (data used uniform Si,
    # the PINN physics loss used heterogeneous k), not just a simplification.
    def _footprints_by_layer(self, geometry: Geometry) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for fp in geometry.die_footprints:
            out.setdefault(fp.die_layer_name, []).append(fp)
        return out

    def _underfill_material_name(self, geometry: Geometry) -> str:
        return f"underfill_k{geometry.underfill_k:.2f}".replace('.', '_')

    def _generate_layout_files(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """
        Write one 3D-ICE 4.0 layout file per TSV layer carrying a density field, plus one per die layer with DiePrint footprints (silicon under the
        """
        for layer_name, (idx, centres) in self._tsv_levels(scenario).items():
            n_l, n_w = idx.shape
            tile_l = geometry.die_length / n_l
            tile_w = geometry.die_width / n_w
            safe = self._sanitise(layer_name)

            lines = [f"// TSV density layout — {geometry.name} / {layer_name}", ""]
            for li in range(len(centres)):
                cells = np.argwhere(idx == li)
                if cells.size == 0:
                    continue
                lines.append(f"tsvmat_{safe}_{li} :")
                for a, b in cells:
                    lines.append(f"   rectangle ( {a * tile_l:.1f}, {b * tile_w:.1f}, "
                                 f"{tile_l:.1f}, {tile_w:.1f} ) ;")
                lines.append("")

            path = self.config_dir / f"layout_{safe}.lyt"
            with open(path, 'w') as f:
                f.write('\n'.join(lines))

        layer_k_overrides = scenario.get('layer_k_overrides', {})
        for layer_name, footprints in self._footprints_by_layer(geometry).items():
            layer_idx, layer = next(
                (i, l) for i, l in enumerate(geometry.layers) if l.name == layer_name)
            safe = self._sanitise(layer_name)
            # Must match whatever _generate_stack_file declares for this layer's
            # footprint material -- if layer_k_overrides mangles the name (e.g.
            # 'lsi_bridge_via' -> 'lsi_bridge_via_k40'), the .lyt rectangle group
            # header has to use the SAME mangled name or 3D-ICE rejects it as an
            # undeclared material. Found 2026-08-18: this line still used the
            # bare layer.material unconditionally, so overriding a footprint-
            # carrying layer's k (as opposed to a full uniform layer's, which
            # never goes through this file at all) failed outright rather than
            # silently using the wrong value -- a loud failure, not a silent one,
            # but still a real bug in the override path.
            footprint_mat_name = self._get_layer_material_name(
                layer, layer_k_overrides, layer_idx)
            lines = [f"// Die footprint layout (Si under footprint) — "
                     f"{geometry.name} / {layer_name}", "", f"{footprint_mat_name} :"]
            # Same axis swap as _generate_floorplan_files: 3D-ICE X is along
            # chip_length (= die_length = PINN y-axis).
            for fp in footprints:
                lines.append(f"   rectangle ( {fp.y:.1f}, {fp.x:.1f}, "
                              f"{fp.height:.1f}, {fp.width:.1f} ) ;")
            lines.append("")

            path = self.config_dir / f"layout_footprint_{safe}.lyt"
            with open(path, 'w') as f:
                f.write('\n'.join(lines))

    def _plan_sublayers(self, geometry: Geometry) -> List[Dict[str, Any]]:
        """Expand geometry layers into the stack elements actually emitted."""
        coolant_layer_name = getattr(geometry, 'coolant_layer_name', None)
        plan: List[Dict[str, Any]] = []
        for i, layer in enumerate(geometry.layers):
            is_coolant = coolant_layer_name is not None and layer.name == coolant_layer_name
            if layer.is_active or is_coolant:
                # Source layers stay whole because the floorplan mapping assumes
                # one element per active layer. A microchannel is likewise a
                # single 3D-ICE stack element (`channel` in the stack: list) --
                # it isn't a solid material that sub-layer splitting applies to.
                n_sub = 1
            else:
                n_sub = int(np.ceil(layer.thickness / self.MAX_SUBLAYER_UM))
                n_sub = max(1, min(n_sub, self.MAX_SUBLAYERS_PER_LAYER))

            t_sub = layer.thickness / n_sub
            for j in range(n_sub):
                z_bot = layer.z_bottom + j * t_sub
                plan.append({
                    'layer_idx': i,
                    'thickness': t_sub,
                    'z_center': z_bot + t_sub / 2.0,
                    'is_active': layer.is_active,
                    'is_coolant': is_coolant,
                })
        return plan

    def _generate_stack_file(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """Generate 3D-ICE stack description file (.stk) with correct µm-unit syntax."""
        stk_file = self.config_dir / "stack.stk"

        htc = scenario.get('htc', 10000.0)      # W/m²·K
        t_ambient_c = scenario.get('t_ambient', 25.0)
        t_ambient_k = t_ambient_c + 273.15

        # 3D-ICE uses µm; convert SI properties:
        #   k:      W/m/K   → W/µm/K    × 1e-6
        #   rho_cp: J/m³/K  → J/µm³/K   × 1e-18
        #   HTC:    W/m²/K  → W/µm²/K   × 1e-12
        htc_ice = htc * 1e-12

        lines = []
        lines.append("// 3D-ICE Stack Description File")
        lines.append(f"// Geometry: {geometry.name}")
        lines.append("")

        # ── Material definitions ──────────────────────────────────────────────
        layer_k_overrides = scenario.get('layer_k_overrides', {})
        for mat_name, props in self._get_unique_materials(geometry, layer_k_overrides).items():
            k_ice = props['k'] * 1e-6
            rho_cp_ice = props['rho_cp'] * 1e-18
            lines.append(f"material {mat_name} :")
            lines.append(f"   thermal conductivity     {k_ice:.4e} ;")
            lines.append(f"   volumetric heat capacity {rho_cp_ice:.4e} ;")
            lines.append("")

        # Anisotropic materials for spatially varying TSV density. One material per
        # quantised density level, each with a distinct lateral/vertical conductivity
        # (a TSV is a copper cylinder: heat runs ALONG it but must CROSS phases
        # sideways). Requires 3D-ICE 4.0 -- 3.0.0 took a single isotropic value.
        for mat_name, (k_lat, k_vert, rho_cp) in self._tsv_materials(scenario).items():
            lines.append(f"material {mat_name} :")
            lines.append(f"   thermal conductivity     {k_lat * 1e-6:.4e}, "
                         f"{k_lat * 1e-6:.4e}, {k_vert * 1e-6:.4e} ;")
            lines.append(f"   volumetric heat capacity {rho_cp * 1e-18:.4e} ;")
            lines.append("")

        footprints_by_layer = self._footprints_by_layer(geometry)
        if footprints_by_layer:
            # Epoxy underfill volumetric heat capacity -- engineering estimate
            # (no specific cited source); steady-state solves are insensitive to it.
            underfill_rho_cp = 1.8e6
            lines.append(f"material {self._underfill_material_name(geometry)} :")
            lines.append(f"   thermal conductivity     {geometry.underfill_k * 1e-6:.4e} ;")
            lines.append(f"   volumetric heat capacity {underfill_rho_cp * 1e-18:.4e} ;")
            lines.append("")

        cooling_mode = scenario.get('cooling_mode', 'air')
        if cooling_mode == 'microchannel_2rm' and getattr(geometry, 'coolant_layer_name', None) is None:
            raise ValueError(
                f"scenario requests cooling_mode='microchannel_2rm' but "
                f"{geometry.name} has no coolant_layer_name set -- there would be "
                f"no `channel` stack element and no bottom heat sink either, "
                f"leaving the thermal problem with no heat-rejection boundary."
            )
        if cooling_mode == 'microchannel_2rm':
            # Liquid microchannel cold plate replaces the idealised HTC boundary
            # condition with an explicitly modelled coolant layer -- see
            # bison/stack_description_parser.y "MicroChannel", grammar confirmed
            # against 3D-ICE 4.0's own test/mc2rm/steady example .stk files.
            # Coolant advection along the flow direction is genuinely NOT linear
            # in the boundary data the way a fixed HTC scalar is (unlike every
            # other change in this benchmark, which stays inside the linear
            # conduction regime) -- see goal.md and assumptions.md.
            mc = scenario.get('microchannel', {})
            channel_height_um = float(mc.get('channel_height_um', 100.0))
            channel_length_um = float(mc.get('channel_length_um', 50.0))
            wall_length_um    = float(mc.get('wall_length_um', 50.0))
            wall_material     = mc.get('wall_material', 'silicon')
            flow_rate_ml_min  = float(mc.get('flow_rate_ml_min', 48.0))
            coolant_htc_top_wm2k    = float(mc.get('coolant_htc_top_wm2k', 20000.0))
            coolant_htc_bottom_wm2k = float(mc.get('coolant_htc_bottom_wm2k', 20000.0))
            # Water: rho*cp ~ 4.18e6 J/m^3/K
            coolant_vhc = float(mc.get('coolant_vhc_jm3k', 4.18e6))

            lines.append("microchannel 2rm :")
            lines.append(f"   height {channel_height_um:.1f} ;")
            lines.append(f"   channel length {channel_length_um:.1f} ;")
            lines.append(f"   wall    length {wall_length_um:.1f} ;")
            lines.append(f"   wall material {wall_material} ;")
            lines.append(f"   coolant flow rate {flow_rate_ml_min:.4f} ;")
            lines.append(f"   coolant heat transfer coefficient top    "
                         f"{coolant_htc_top_wm2k * 1e-12:.4e} ,")
            lines.append(f"                                     bottom "
                         f"{coolant_htc_bottom_wm2k * 1e-12:.4e} ;")
            lines.append(f"   coolant volumetric heat capacity {coolant_vhc * 1e-18:.4e} ;")
            lines.append(f"   coolant incoming temperature {t_ambient_k:.2f} ;")
            lines.append("")
        else:
            # ── Heat-sink boundary condition ─────────────────────────────────
            # Die is the topmost layer; convective cooling is on the bottom surface.
            lines.append("bottom heat sink :")
            lines.append(f"   heat transfer coefficient {htc_ice:.4e} ;")
            lines.append(f"   temperature {t_ambient_k:.2f} ;")
            lines.append("")

        # ── Chip dimensions ───────────────────────────────────────────────────
        nx, ny, _ = geometry.mesh_resolution
        cell_length = geometry.die_length / nx
        cell_width  = geometry.die_width  / ny
        lines.append("dimensions :")
        lines.append(f"   chip length {geometry.die_length:.1f} , width  {geometry.die_width:.1f} ;")
        lines.append(f"   cell length {cell_length:.1f} , width  {cell_width:.1f} ;")
        lines.append("")

        # ── Layer / die type definitions ──────────────────────────────────────
        # 3D-ICE grammar requires ALL layer definitions before ALL die definitions.
        # One definition per emitted sub-layer; several may share a geometry layer's
        # material but each needs its own type because heights differ across layers.
        for s_idx, sub in enumerate(self._sublayers):
            if sub['is_active'] or sub.get('is_coolant'):
                continue
            layer = geometry.layers[sub['layer_idx']]
            mat_name = self._get_layer_material_name(layer, layer_k_overrides, sub['layer_idx'])
            has_tsv_map = layer.name in (scenario.get('tsv_map_by_layer') or {})
            # PASSIVE layers can carry DiePrint footprints too (geometry7's
            # CoWoS-L bridge layer: silicon LSI bridge islands embedded in an
            # organic RDL sea). Same convention as the active-layer case below --
            # the layer's declared base material becomes the gap/underfill
            # material and the footprint rectangles supply the real material via
            # the layout. Before 2026-08-17 this branch only honoured tsv maps,
            # so footprints on a passive layer were silently dropped: the .lyt
            # file was written but never referenced, and the layer stayed
            # uniformly its own material. Anything relying on lateral structure
            # in a passive layer would have been quietly simulated without it.
            has_footprints = layer.name in footprints_by_layer
            if has_footprints and has_tsv_map:
                raise ValueError(
                    f"Layer {layer.name!r} has BOTH DiePrint footprints and a TSV "
                    "density map. 3D-ICE takes at most one `layout` per layer "
                    "declaration, so these cannot both be expressed -- split the "
                    "lateral structure across two layers instead."
                )
            if has_footprints:
                mat_name = self._gap_material_name(geometry, layer, layer_k_overrides)
            lines.append(f"layer type_layer_{s_idx} :")
            lines.append(f"   height {sub['thickness']:.1f} ;")
            lines.append(f"   material {mat_name} ;")
            # Spatially varying TSV conductivity, if this layer carries a density
            # field. 3D-ICE 4.0 only: a layer declaration takes an optional layout
            # that overrides the uniform material per rectangle.
            if has_tsv_map:
                lyt = (self.config_dir / f"layout_{self._sanitise(layer.name)}.lyt").resolve()
                lyt_path = self._to_wsl_path(lyt) if getattr(self, '_use_wsl', False) else str(lyt)
                lines.append(f'   layout "{lyt_path}" ;')
            elif has_footprints:
                lyt = (self.config_dir
                       / f"layout_footprint_{self._sanitise(layer.name)}.lyt").resolve()
                lyt_path = self._to_wsl_path(lyt) if getattr(self, '_use_wsl', False) else str(lyt)
                lines.append(f'   layout "{lyt_path}" ;')
            lines.append("")

        # Active (source) sub-layers whose geometry layer carries DiePrint
        # footprints need a standalone `layer` declaration with a layout,
        # because the grammar's `source` clause only accepts a bare
        # thickness+material pair OR a reference to a pre-declared `layer`
        # (which is where a layout can be attached) -- see
        # bison/stack_description_parser.y die_layer_content / layer_copy.
        for s_idx, sub in enumerate(self._sublayers):
            if not sub['is_active']:
                continue
            layer = geometry.layers[sub['layer_idx']]
            if layer.name not in footprints_by_layer:
                continue
            lyt = (self.config_dir / f"layout_footprint_{self._sanitise(layer.name)}.lyt").resolve()
            lyt_path = self._to_wsl_path(lyt) if getattr(self, '_use_wsl', False) else str(lyt)
            lines.append(f"layer type_layer_{s_idx}_src :")
            lines.append(f"   height {self._source_thickness(sub):.1f} ;")
            lines.append(f"   material {self._gap_material_name(geometry, layer, layer_k_overrides)} ;")
            lines.append(f'   layout "{lyt_path}" ;')
            lines.append("")

        for s_idx, sub in enumerate(self._sublayers):
            if not sub['is_active']:
                continue
            layer = geometry.layers[sub['layer_idx']]
            mat_name = self._get_layer_material_name(layer, layer_k_overrides, sub['layer_idx'])
            lines.append(f"die type_die_{s_idx} :")
            t_src = self._source_thickness(sub)
            if layer.name in footprints_by_layer:
                content = f"type_layer_{s_idx}_src"          # same height and layout for all thirds
            else:
                content = f"{t_src:.1f} {mat_name}"
            split = t_src < sub['thickness']
            if split:
                lines.append(f"   layer  {content} ;")
            lines.append(f"   source {content} ;")
            if split:
                lines.append(f"   layer  {content} ;")
            lines.append("")

        # ── Stack assembly: 3D-ICE lists TOP → BOTTOM; geometry is BOTTOM → TOP
        # Each active layer has its own per-layer floorplan file, keyed by the
        # ORIGINAL geometry layer index (floorplans are unaffected by subdivision).
        lines.append("stack :")
        for s_idx, sub in reversed(list(enumerate(self._sublayers))):
            inst = f"inst_{s_idx}"
            if sub.get('is_coolant'):
                lines.append(f"   channel {inst} ;")
            elif sub['is_active']:
                flp_name = f"floorplan_layer{sub['layer_idx']}.flp"
                flp_abs  = (self.config_dir / flp_name).resolve()
                flp_path = self._to_wsl_path(flp_abs) if getattr(self, '_use_wsl', False) else str(flp_abs)
                lines.append(f"   die   {inst}  type_die_{s_idx}    floorplan \"{flp_path}\" ;")
            else:
                lines.append(f"   layer {inst}  type_layer_{s_idx} ;")
        lines.append("")

        # ── Solver ────────────────────────────────────────────────────────────
        # numofcores: 3D-ICE 4.0's SLU_Options.nprocs feeds SuperLU_MT's parallel
        # factorization (pdgstrf). CORRECTION 2026-08-18: an earlier version of
        # this comment claimed 8 cores was a free, exact speedup verified
        # byte-identical to the 1-core baseline -- that was true for exactly one
        # run pair and NOT re-checked against a second independent run at the
        # same core count before being made the default. It was wrong. Running
        # the identical scenario twice at numofcores=8 (nothing else changed)
        # produced peak-adjacent values differing by >1 K between the two runs;
        # numofcores=1 run twice in a row was bit-for-bit identical both times.
        # The non-determinism is real and isolated to the multi-threaded path
        # (thread-scheduling-dependent floating-point reduction order in
        # SuperLU_MT's parallel factorization, not this project's code), not a
        # fluke of one comparison. Defaulting to >1 core would have silently
        # made every future dataset non-reproducible run to run -- a much worse
        # problem than the wall-clock this was meant to save. Reverted to the
        # deterministic default; multi-core remains available for anyone who
        # explicitly wants the tradeoff (e.g. quick exploratory sweeps where
        # exact reproducibility doesn't matter), never as a silent default.
        num_cores = int(scenario.get('num_cores', 1))
        lines.append("solver :")
        lines.append("   steady ;")
        lines.append(f"   initial temperature {t_ambient_k:.2f} ;")
        lines.append(f"   numofcores {num_cores} ;")
        lines.append("")

        # ── Output: Tmap per stack element ────────────────────────────────────
        # Files are written to output_dir named output_inst_N.txt. The coolant
        # element is skipped -- it is fluid, not a solid layer/die, and this
        # first pass does not yet reconcile its node output with the coords/
        # npz-export pipeline built for solid stack elements (see goal.md).
        lines.append("output :")
        for i in reversed(range(len(self._sublayers))):
            if self._sublayers[i].get('is_coolant'):
                continue
            inst = f"inst_{i}"
            out_abs  = (self.output_dir / f"output_{inst}.txt").resolve()
            out_path = self._to_wsl_path(out_abs) if getattr(self, '_use_wsl', False) else str(out_abs)
            lines.append(f"   Tmap ( {inst}, \"{out_path}\", final ) ;")
        lines.append("")

        with open(stk_file, 'w') as f:
            f.write('\n'.join(lines))

    def _source_thickness(self, sub) -> float:
        t = sub['thickness']
        return t / 3.0 if sub['is_active'] and t > self.ACTIVE_SPLIT_UM else t

    def _generate_floorplan_files(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """Generate one 3D-ICE floorplan file per active layer."""
        power_scenario = scenario.get('power_blocks', {})

        for i, layer in enumerate(geometry.layers):
            if not layer.is_active:
                continue

            flp_file = self.config_dir / f'floorplan_layer{i}.flp'

            # Only blocks for this layer; exclude TSV regions (zero heat source)
            layer_blocks = [
                b for b in geometry.power_blocks
                if b.layer_name == layer.name and not b.is_tsv_region
            ]

            lines = []
            lines.append(f"// 3D-ICE Floorplan — {geometry.name} layer {i}: {layer.name}")
            lines.append("")

            # Per-cell power map, when the scenario carries one. The map replaces
            # the block decomposition entirely for this layer: each cell becomes
            # its own floorplan element. Solve cost is unaffected -- 3D-ICE's cost
            # is set by the mesh, not the floorplan (measured 13.3 s for 4 elements
            # and 13.5 s for 10,000 on geometry1) -- but the input stops being a
            # handful of scalars, which is what makes the solution operator a
            # genuine Green's function rather than a low-dimensional linear map.
            pmap = (scenario.get('power_map_by_layer') or {}).get(layer.name)
            if pmap is not None:
                pmap = np.asarray(pmap, dtype=np.float64)
                n_l, n_w = pmap.shape
                tile_l = geometry.die_length / n_l
                tile_w = geometry.die_width / n_w
                for a in range(n_l):
                    for b in range(n_w):
                        p_w = float(pmap[a, b])
                        if p_w <= 0.0:
                            continue
                        lines.append(f"c_{a}_{b} :")
                        lines.append(f"   position  {a * tile_l:.1f}, {b * tile_w:.1f} ;")
                        lines.append(f"   dimension {tile_l:.1f}, {tile_w:.1f} ;")
                        lines.append(f"   power values  {p_w:.6f} ;")
                        lines.append("")
                with open(flp_file, 'w') as f:
                    f.write('\n'.join(lines))
                continue

            for block in layer_blocks:
                power_density_wcm2 = power_scenario.get(block.name, 0.0)
                power_w = block.power_watts(power_density_wcm2)
                # 3D-ICE floorplan: position(X_sw, Y_sw) where X is along chip_length
                # (= die_length = PINN y-axis) and Y is along chip_width (= die_width
                # = PINN x-axis).  Swap block.x/y and width/height accordingly.
                lines.append(f"{block.name} :")
                lines.append(f"   position  {block.y:.1f}, {block.x:.1f} ;")
                lines.append(f"   dimension {block.height:.1f}, {block.width:.1f} ;")
                lines.append(f"   power values  {power_w:.6f} ;")
                lines.append("")

            with open(flp_file, 'w') as f:
                f.write('\n'.join(lines))

    def _generate_power_trace(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """Placeholder for transient power trace (steady-state uses floorplan power values)."""
        pass

    def _get_layer_material_name(self, layer, layer_k_overrides: dict, layer_idx: int) -> str:
        """Return the 3D-ICE material name for a layer, mangling if k is overridden."""
        if layer.name in layer_k_overrides:
            k_int = int(round(layer_k_overrides[layer.name]))
            return f"{layer.material}_k{k_int}"
        return layer.material

    @staticmethod
    def _gap_override_key(layer_name: str) -> str:
        """
        layer_k_overrides key convention for a layer's gap_material (distinct from the layer's own footprint material, which uses the bare layer
        """
        return f"{layer_name}__gap"

    def _get_unique_materials(self, geometry: Geometry,
                              layer_k_overrides: dict = None) -> Dict[str, Dict]:
        """Extract unique materials and their properties from geometry."""
        layer_k_overrides = layer_k_overrides or {}
        materials = {}
        for i, layer in enumerate(geometry.layers):
            mat_name = self._get_layer_material_name(layer, layer_k_overrides, i)
            if mat_name not in materials:
                k = layer_k_overrides.get(layer.name, layer.k_thermal)
                materials[mat_name] = {
                    'k': k,
                    'rho_cp': layer.volumetric_heat_capacity
                }
            # A layer with an explicit gap_material (e.g. geometry7's organic
            # substrate the LSI bridge islands sit in) needs THAT material
            # declared too -- it is not layer.material (which fills the
            # footprints, not the gap) and not the generic underfill material.
            if layer.gap_material:
                from ..core.material import MaterialLibrary
                gap_mat = MaterialLibrary.get(layer.gap_material)
                gap_name = self._gap_material_name(geometry, layer, layer_k_overrides)
                if gap_name not in materials:
                    k = layer_k_overrides.get(self._gap_override_key(layer.name),
                                              gap_mat.k_thermal)
                    materials[gap_name] = {
                        'k': k,
                        'rho_cp': gap_mat.volumetric_heat_capacity,
                    }
        return materials

    def _gap_material_name(self, geometry: Geometry, layer,
                           layer_k_overrides: dict = None) -> str:
        """
        Material name for the region outside a footprint-carrying layer's DiePrint rectangles: the layer's own gap_material if it declares one
        """
        base = layer.gap_material or self._underfill_material_name(geometry)
        layer_k_overrides = layer_k_overrides or {}
        if layer.gap_material:
            override_key = self._gap_override_key(layer.name)
            if override_key in layer_k_overrides:
                k_int = int(round(layer_k_overrides[override_key]))
                return f"{base}_k{k_int}"
        return base

    @staticmethod
    def _to_wsl_path(windows_path) -> str:
        """Convert a Windows absolute path to its WSL /mnt/<drive>/... equivalent."""
        p = str(windows_path).replace('\\', '/')
        if len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            rest  = p[2:]
            return f"/mnt/{drive}{rest}"
        return p

    def run_simulation(self, scenario_name: str) -> Path:
        """Run 3D-ICE simulation."""
        stk_file = self.config_dir / "stack.stk"

        # Remove stale output files from a previous geometry with more layers
        for stale in self.output_dir.glob("output_inst_*.txt"):
            try:
                stale.unlink()
            except OSError:
                pass

        exe_parts = shlex.split(self.executable)
        use_wsl   = exe_parts[0] == 'wsl'
        stk_arg   = self._to_wsl_path(stk_file.resolve()) if use_wsl else str(stk_file)
        cmd       = exe_parts + [stk_arg]

        # Scale the timeout with problem size. A flat 300 s was fine at one stack
        # element per geometry layer, but sub-layer discretisation multiplies the
        # node count ~5x and the solve exceeded it -- which surfaced as a "simulator
        # failed" fallback rather than an obvious timeout.
        n_nodes = self._nx * self._ny * max(len(getattr(self, '_sublayers', [])), 1)
        timeout_s = max(300.0, 300.0 + n_nodes / 400.0)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=None if use_wsl else str(self.config_dir),
                timeout=timeout_s
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"3D-ICE simulation failed for {scenario_name}:\n"
                    f"Return code: {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # Expect at least one output_inst_*.txt file
            output_files = list(self.output_dir.glob("output_inst_*.txt"))
            if not output_files:
                raise RuntimeError(
                    f"3D-ICE produced no output files in {self.output_dir}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            return self.output_dir

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"3D-ICE simulation timed out for {scenario_name}")
        except FileNotFoundError:
            raise RuntimeError(
                f"3D-ICE executable not found: {exe_parts[0]}\n"
                f"For WSL: use --ice-executable \"wsl /home/<user>/3d-ice/bin/3D-ICE-Emulator\""
            )

    def parse_results(self, result_path: Path) -> Dict[str, np.ndarray]:
        """Parse 3D-ICE Tmap output files."""
        result_path = Path(result_path)

        if not hasattr(self, '_geometry'):
            raise RuntimeError(
                "parse_results called before generate_config_files; "
                "geometry metadata is not available."
            )

        # ── Collect per-layer Tmap files ──────────────────────────────────────
        if result_path.is_dir():
            output_files = sorted(result_path.glob("output_inst_*.txt"))
        else:
            # Fallback: single file passed (4-column or legacy format)
            output_files = [result_path] if result_path.exists() else []

        if not output_files:
            raise ValueError(f"No output files found at {result_path}")

        coords_all: List[np.ndarray] = []
        temps_all:  List[np.ndarray] = []

        for out_file in output_files:
            # Derive layer index from filename: output_inst_N.txt → N
            stem = out_file.stem            # e.g. "output_inst_2"
            try:
                layer_idx = int(stem.split('_')[-1])
            except ValueError:
                continue

            if layer_idx >= len(self._layer_z_centers):
                continue

            matched_z = float(self._layer_z_centers[layer_idx])

            with open(out_file, 'r') as f:
                content = f.read()

            temp_rows = []
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                try:
                    row_vals = [float(v) for v in line.split()]
                    if row_vals:
                        temp_rows.append(row_vals)
                except ValueError:
                    continue

            if not temp_rows:
                continue

            # Guard against rare inhomogeneous 3D-ICE output (some rows have
            # different column counts — seen with near-zero power or high HTC).
            row_lengths = [len(r) for r in temp_rows]
            if len(set(row_lengths)) > 1:
                temp_rows = [r for r in temp_rows if len(r) == self._nx]
                if not temp_rows:
                    continue

            temp_grid = np.array(temp_rows, dtype=np.float64)   # shape (n_rows, n_cols)

            # Tmap rows run along chip WIDTH (die_width = PINN x) and columns along
            # chip LENGTH (die_length = PINN y), verified 2026-09-23 against an
            # independent FV solve (docs/report.md §9.23). This parser previously
            # assumed the opposite, which silently transposed every square field
            # and, via a linspace fallback, stretched every non-square one.
            if temp_grid.shape != (self._ny, self._nx):
                raise ValueError(
                    f"{out_file.name}: Tmap is {temp_grid.shape}, expected "
                    f"(width cells, length cells) = ({self._ny}, {self._nx})")

            ww, ll = np.meshgrid(self._y_centers, self._x_centers, indexing='ij')
            zz = np.full_like(ww, matched_z)
            coords_all.append(np.stack([ww.ravel(), ll.ravel(), zz.ravel()], axis=1))
            temps_all.append(temp_grid.ravel())

        if not coords_all:
            # Last-resort fallback: try 4-column (x y z T) whitespace format
            if result_path.is_file():
                try:
                    data = np.loadtxt(result_path, comments='#')
                    if data.ndim == 2 and data.shape[1] >= 4:
                        return {
                            'coords':      data[:, 0:3].astype(np.float32),
                            'temperature': data[:, 3].astype(np.float32)
                        }
                except Exception:
                    pass
            raise ValueError(f"No valid temperature grids found in {result_path}")

        coords      = np.concatenate(coords_all, axis=0).astype(np.float32)
        temperature = np.concatenate(temps_all,  axis=0).astype(np.float32)

        return {'coords': coords, 'temperature': temperature}

    def cleanup_temp_files(self, scenario_name: str) -> None:
        """Clean up temporary config files created during simulation."""
        targets = [self.config_dir / "stack.stk"]
        # Remove all per-layer floorplan files
        targets += list(self.config_dir.glob("floorplan_layer*.flp"))
        # Also remove legacy single-file if present (from old runs)
        targets.append(self.config_dir / "floorplan.flp")
        for f in targets:
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass
