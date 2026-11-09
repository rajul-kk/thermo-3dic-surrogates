"""Build 3D-ICE inputs (.stk/.flp/.lyt) from a small YAML/JSON spec, grid-aligned by construction.

Spec units are SI-flavoured (k in W/m/K, volumetric heat capacity in J/m^3/K, HTC in W/m^2/K,
lengths in um); they are converted to 3D-ICE's units (W/um/K, J/um^3/K, W/um^2/K) on output.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from .snap import _snap_axis_nearest, _on_grid, _fmt, EPS

K_SCALE, RHOCP_SCALE, HTC_SCALE = 1e-6, 1e-18, 1e-12
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXAMPLE_YAML = """\
# iceforge build spec: 6x6 mm, 10 W die on a 10x10 mm chip (the die-edge artefact repro case, aligned).
# Lengths in um; k in W/m/K; rho_cp in J/m^3/K; HTC in W/m^2/K.
chip: {length_um: 10000, width_um: 10000, cell_um: 250}
heat_sink: {side: bottom, htc_W_m2K: 20000, t_ambient_K: 300}
materials:
  SI:  {k_W_mK: 148, rho_cp_J_m3K: 1.628e+6}
  GAP: {k_W_mK: 0.7, rho_cp_J_m3K: 1.628e+6}
layers:                      # bottom -> top
  - {name: SUB, height_um: 300, material: SI}
  - name: SRC
    height_um: 50
    material: GAP            # gap/base material around the dies
    dies:
      - {name: blk, material: SI, x_um: 1500, y_um: 1500, length_um: 6000, width_um: 6000, power_W: 10}
solver: steady
"""


class BuildError(ValueError):
    pass


def load_spec(path: str) -> dict:
    text = open(path, "r", encoding="utf-8").read()
    if path.lower().endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError:
        try:
            return json.loads(text)
        except ValueError:
            raise BuildError("pyyaml is not installed: `pip install iceforge[yaml]`, or give the spec as JSON")
    return yaml.safe_load(text)


def _num(v, where, positive=True):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BuildError(f"{where}: expected a number, got {v!r}")
    if positive and v <= 0:
        raise BuildError(f"{where}: must be > 0, got {v}")
    return float(v)


def _name(v, where):
    if not isinstance(v, str) or not _NAME.match(v):
        raise BuildError(f"{where}: name {v!r} must be an identifier (letters, digits, underscore)")
    return v


def _req(d, key, where):
    if not isinstance(d, dict) or key not in d:
        raise BuildError(f"{where}: missing '{key}'")
    return d[key]


def _e(v):
    return repr(float(v))


def build(spec: dict, outdir: str, snap: bool = False) -> Tuple[str, List[str]]:
    """Write model.stk (+ .flp/.lyt per source layer) into outdir. Returns (stk path, notes)."""
    notes: List[str] = []
    chip = _req(spec, "chip", "spec")
    L = _num(_req(chip, "length_um", "chip"), "chip.length_um")
    W = _num(_req(chip, "width_um", "chip"), "chip.width_um")
    if "cell_um" in chip:
        cl = cw = _num(chip["cell_um"], "chip.cell_um")
    else:
        cl = _num(_req(chip, "cell_length_um", "chip (cell_um, or cell_length_um and cell_width_um)"),
                  "chip.cell_length_um")
        cw = _num(_req(chip, "cell_width_um", "chip"), "chip.cell_width_um")
    if not _on_grid(L, cl) or not _on_grid(W, cw):
        raise BuildError(f"chip {L:g} x {W:g} um is not a whole number of {cl:g} x {cw:g} um cells")

    hs = _req(spec, "heat_sink", "spec")
    side = _req(hs, "side", "heat_sink")
    if side not in ("top", "bottom"):
        raise BuildError("heat_sink.side must be 'top' or 'bottom'")
    htc = _num(_req(hs, "htc_W_m2K", "heat_sink"), "heat_sink.htc_W_m2K") * HTC_SCALE
    t_amb = _num(_req(hs, "t_ambient_K", "heat_sink"), "heat_sink.t_ambient_K")

    mats = _req(spec, "materials", "spec")
    if not isinstance(mats, dict) or not mats:
        raise BuildError("materials: need at least one material")
    for m, v in mats.items():
        _name(m, "materials")
        _num(_req(v, "k_W_mK", f"materials.{m}"), f"materials.{m}.k_W_mK")
        _num(_req(v, "rho_cp_J_m3K", f"materials.{m}"), f"materials.{m}.rho_cp_J_m3K")

    if spec.get("solver", "steady") != "steady":
        raise BuildError("solver: only 'steady' is supported")

    layers = _req(spec, "layers", "spec")
    if not isinstance(layers, list) or not layers:
        raise BuildError("layers: need a non-empty list (bottom to top)")
    seen = set()
    parsed = []   # (name, height, material, dies)
    for i, ly in enumerate(layers):
        w = f"layers[{i}]"
        name = _name(_req(ly, "name", w), w)
        if name in seen:
            raise BuildError(f"{w}: duplicate layer name {name}")
        seen.add(name)
        h = _num(_req(ly, "height_um", w), f"{w}.height_um")
        mat = _req(ly, "material", w)
        if mat not in mats:
            raise BuildError(f"{w}: material {mat!r} is not defined under materials")
        dies = []
        for j, d in enumerate(ly.get("dies") or []):
            dw = f"layers[{i}] ({name}).dies[{j}]" + (f" ({d['name']})" if isinstance(d, dict) and "name" in d else "")
            dn = _name(_req(d, "name", dw), dw)
            dm = _req(d, "material", dw)
            if dm not in mats:
                raise BuildError(f"{dw}: material {dm!r} is not defined under materials")
            dies.append(dict(name=dn, material=dm,
                             x=_num(_req(d, "x_um", dw), dw + ".x_um", False),
                             y=_num(_req(d, "y_um", dw), dw + ".y_um", False),
                             l=_num(_req(d, "length_um", dw), dw + ".length_um"),
                             w=_num(_req(d, "width_um", dw), dw + ".width_um"),
                             p=_num(d.get("power_W", 0.0), dw + ".power_W", False), where=dw))
        if "dies" in ly and not dies:
            raise BuildError(f"{w}: 'dies' is empty")
        parsed.append((name, h, mat, dies))

    # grid alignment, bounds, overlaps
    for name, _, _, dies in parsed:
        placed = []
        for d in dies:
            fix = {}
            for ax, pos, size, c, ext in (("x", d["x"], d["l"], cl, L), ("y", d["y"], d["w"], cw, W)):
                if not (_on_grid(pos, c) and _on_grid(size, c)):
                    cands = _snap_axis_nearest(pos, size, c, ext)
                    if not cands:
                        raise BuildError(f"{d['where']}: cannot be placed on the grid inside the chip")
                    fix[ax] = cands[0]
            if fix:
                nx = fix.get("x", (d["x"], d["l"]))
                ny = fix.get("y", (d["y"], d["w"]))
                msg = (f"{d['where']}: edges are off the {cl:g} x {cw:g} um cell grid; nearest valid values: "
                       f"x_um={nx[0]:g}, y_um={ny[0]:g}, length_um={nx[1]:g}, width_um={ny[1]:g}")
                if not snap:
                    raise BuildError(msg + "  (use --snap to apply them)")
                notes.append(f"snapped {d['where']}: ({d['x']:g},{d['y']:g},{d['l']:g},{d['w']:g}) -> "
                             f"({nx[0]:g},{ny[0]:g},{nx[1]:g},{ny[1]:g})")
                d["x"], d["l"] = nx
                d["y"], d["w"] = ny
            if d["x"] < -EPS or d["y"] < -EPS or d["x"] + d["l"] > L + EPS or d["y"] + d["w"] > W + EPS:
                raise BuildError(f"{d['where']}: lies outside the {L:g} x {W:g} um chip")
            for o in placed:
                if min(d["x"] + d["l"], o["x"] + o["l"]) - max(d["x"], o["x"]) > EPS and \
                   min(d["y"] + d["w"], o["y"] + o["w"]) - max(d["y"], o["y"]) > EPS:
                    raise BuildError(f"{d['where']}: overlaps die {o['name']} in layer {name}")
            placed.append(d)
        names = [d["name"] for d in dies]
        if len(set(names)) != len(names):
            raise BuildError(f"layer {name}: duplicate die names")

    sources = [n for n, _, _, d in reversed(parsed) if d]
    if not sources:
        raise BuildError("no source layer: at least one layer needs 'dies'")

    os.makedirs(outdir, exist_ok=True)
    o = []
    for m, v in mats.items():
        o += [f"material {m} :",
              f"   thermal conductivity     {_e(v['k_W_mK'] * K_SCALE)} ;",
              f"   volumetric heat capacity {_e(v['rho_cp_J_m3K'] * RHOCP_SCALE)} ;"]
    o += ["", f"{side} heat sink :", f"   heat transfer coefficient {_e(htc)} ;", f"   temperature {t_amb:.2f} ;", "",
          "dimensions :", f"   chip length {_fmt(L)} , width  {_fmt(W)} ;",
          f"   cell length {_fmt(cl)} , width  {_fmt(cw)} ;", ""]
    # grammar: all layer definitions first, then die definitions, then the stack (TOP to BOTTOM)
    for name, h, mat, dies in parsed:
        o += [f"layer L_{name} :", f"   height {_fmt(h)} ;", f"   material {mat} ;"]
        if dies:
            o.append(f'   layout "{name}.lyt" ;')
        o.append("")
    for name, h, mat, dies in parsed:
        if dies:
            o += [f"die D_{name} :", f"   source L_{name} ;", ""]
    o.append("stack :")
    for name, h, mat, dies in reversed(parsed):
        if dies:
            o.append(f'   die   {name}  D_{name}  floorplan "{name}.flp" ;')
        else:
            o.append(f"   layer {name}  L_{name} ;")
    o += ["", "solver :", "   steady ;", f"   initial temperature {t_amb:.2f} ;", "   numofcores 1 ;", "", "output :"]
    for n in sources:
        o.append(f'   Tmap ( {n}, "tmap_{n}.txt", final ) ;')
    o.append("")

    for name, h, mat, dies in parsed:
        if not dies:
            continue
        flp = []
        for d in dies:
            flp += [f"{d['name']} :", f"   position  {_fmt(d['x'])}, {_fmt(d['y'])} ;",
                    f"   dimension {_fmt(d['l'])}, {_fmt(d['w'])} ;", f"   power values  {d['p']:g} ;", ""]
        lyt = []
        by_mat: Dict[str, list] = {}
        for d in dies:
            by_mat.setdefault(d["material"], []).append(d)
        for m, ds in by_mat.items():
            lyt.append(f"{m} :")
            lyt += [f"   rectangle ( {_fmt(d['x'])}, {_fmt(d['y'])}, {_fmt(d['l'])}, {_fmt(d['w'])} ) ;" for d in ds]
            lyt.append("")
        for fn, lines in ((f"{name}.flp", flp), (f"{name}.lyt", lyt)):
            with open(os.path.join(outdir, fn), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines))
    stk = os.path.join(outdir, "model.stk")
    with open(stk, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(o))
    return stk, notes
