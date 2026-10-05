"""Grid snapping: write a copy of a model whose floorplan/layout edges lie on cell boundaries.

Input files are never modified.  The copied .stk, .flp and .lyt files are written into
the output directory (flat, de-duplicated names); the .stk's path references are rewritten
to those relative names.  Regenerated .flp/.lyt files are canonical (comments are lost).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .model import Stack, Rect, Floorplan, Layout, Element, parse_stk

EPS = 1e-6


@dataclass
class Move:
    file: str
    owner: str       # element or material name
    old: Tuple[float, float, float, float]
    new: Tuple[float, float, float, float]
    note: str = ""

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        o, n = self.old, self.new
        return (f"{os.path.basename(self.file)}: {self.owner}: "
                f"({o[0]:g},{o[1]:g},{o[2]:g},{o[3]:g}) -> ({n[0]:g},{n[1]:g},{n[2]:g},{n[3]:g})"
                + (f"  [{self.note}]" if self.note else ""))


def _on_grid(v, c):
    r = v / c
    return abs(r - round(r)) <= 1e-6


def _snap_axis_nearest(pos: float, size: float, c: float, extent: float):
    """Candidate (pos, size) pairs on the grid, nearest first. Size is preserved when it is a
    whole number of cells; otherwise both edges go to their nearest boundary (min one cell)."""
    cands = []
    if _on_grid(size, c):
        k = pos / c
        lo, hi = math.floor(k + EPS), math.ceil(k - EPS)
        first, second = (lo, hi) if abs(k - lo) <= abs(k - hi) else (hi, lo)
        for kk in (first, second):
            cands.append((kk * c, size))
    else:
        a = round(pos / c) * c
        b = max(round((pos + size) / c) * c, a + c)
        cands.append((a, b - a))
        a2 = math.floor(pos / c) * c
        b2 = math.ceil((pos + size) / c) * c
        cands.append((a2, b2 - a2))
    return [(p, s) for p, s in cands if p >= -EPS and p + s <= extent + EPS]


def _overlap(a: Rect, b: Rect) -> bool:
    return (min(a.x1, b.x1) - max(a.x, b.x) > EPS) and (min(a.y1, b.y1) - max(a.y, b.y) > EPS)


def snap_floorplan(fp: Floorplan, st: Stack) -> List[Move]:
    d = st.dims
    moves: List[Move] = []
    placed: List[Tuple[str, Rect]] = []
    for e in fp.elements:
        for r in e.rects:
            old = (r.x, r.y, r.l, r.w)
            xs = _snap_axis_nearest(r.x, r.l, d.cell_l, d.chip_l)
            ys = _snap_axis_nearest(r.y, r.w, d.cell_w, d.chip_w)
            chosen = None
            note = ""
            for (x, l) in xs:
                for (y, w) in ys:
                    cand = Rect(x, y, l, w)
                    if not any(n != e.name and _overlap(cand, o) for n, o in placed):
                        chosen = cand
                        break
                if chosen:
                    break
            if chosen is None:
                # keep original, report it
                note = "could not snap without leaving the chip or overlapping another element; left unchanged"
                chosen = Rect(*old)
            elif (chosen.x, chosen.y) != (xs[0][0], ys[0][0]) and old != (chosen.x, chosen.y, chosen.l, chosen.w):
                note = "fell back to the other neighbouring boundary"
            if abs(chosen.l - r.l) > EPS or abs(chosen.w - r.w) > EPS:
                note = (note + "; " if note else "") + "size changed to whole cells"
            new = (chosen.x, chosen.y, chosen.l, chosen.w)
            r.x, r.y, r.l, r.w = new
            placed.append((e.name, chosen))
            if any(abs(a - b) > 1e-9 for a, b in zip(old, new)) or note:
                moves.append(Move(fp.path, e.name, old, new, note))
    return moves


def snap_layout(ly: Layout, st: Stack, mode: str) -> List[Move]:
    d = st.dims
    moves: List[Move] = []
    placed: List[Tuple[str, Rect]] = []
    for mat, rects in ly.shapes.items():
        for r in rects:
            old = (r.x, r.y, r.l, r.w)
            note = ""
            if mode == 'outward':
                x0 = max(math.floor(r.x / d.cell_l + EPS) * d.cell_l, 0.0)
                y0 = max(math.floor(r.y / d.cell_w + EPS) * d.cell_w, 0.0)
                x1 = min(math.ceil(r.x1 / d.cell_l - EPS) * d.cell_l, d.chip_l)
                y1 = min(math.ceil(r.y1 / d.cell_w - EPS) * d.cell_w, d.chip_w)
                new = (x0, y0, x1 - x0, y1 - y0)
                if new != old:
                    note = "grown outward to whole cells"
            else:
                xs = _snap_axis_nearest(r.x, r.l, d.cell_l, d.chip_l)
                ys = _snap_axis_nearest(r.y, r.w, d.cell_w, d.chip_w)
                chosen = None
                for (x, l) in xs:
                    for (y, w) in ys:
                        cand = Rect(x, y, l, w)
                        if not any(n != mat and _overlap(cand, o) for n, o in placed):
                            chosen = cand
                            break
                    if chosen:
                        break
                if chosen is None:
                    chosen = Rect(*old)
                    note = "could not snap without leaving the chip or overlapping another material; unchanged"
                new = (chosen.x, chosen.y, chosen.l, chosen.w)
                if abs(new[2] - old[2]) > EPS or abs(new[3] - old[3]) > EPS:
                    note = "size changed to whole cells"
            r.x, r.y, r.l, r.w = new
            placed.append((mat, Rect(*new)))
            if any(abs(a - b) > 1e-9 for a, b in zip(old, new)):
                moves.append(Move(ly.path, mat, old, new, note))
    return moves


def _fmt(v: float) -> str:
    return f"{v:.6f}".rstrip('0').rstrip('.') if abs(v - round(v)) > 1e-9 else f"{int(round(v))}.0"


def write_flp(fp: Floorplan, path: str) -> None:
    out = []
    for e in fp.elements:
        out.append(f"{e.name} :")
        if len(e.rects) == 1:
            r = e.rects[0]
            out.append(f"   position  {_fmt(r.x)}, {_fmt(r.y)} ;")
            out.append(f"   dimension {_fmt(r.l)}, {_fmt(r.w)} ;")
        else:
            for r in e.rects:
                out.append(f"   rectangle ( {_fmt(r.x)}, {_fmt(r.y)}, {_fmt(r.l)}, {_fmt(r.w)} ) ;")
        if e.discretization:
            out.append(f"   discretization {e.discretization[0]:g}, {e.discretization[1]:g} ;")
        if e.material:
            out.append(f"   material {e.material} ;")
        if e.power:
            out.append("   power values " + ", ".join(f"{p:g}" for p in e.power) + " ;")
        out.append("")
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out))


def write_lyt(ly: Layout, path: str) -> None:
    out = []
    for m in ly.local_materials.values():
        out.append(f"material {m.name} :")
        out.append("   thermal conductivity " + ", ".join(f"{k:g}" for k in m.k) + " ;")
        out.append(f"   volumetric heat capacity {m.vhc:g} ;")
        out.append("")
    for mat, rects in ly.shapes.items():
        out.append(f"{mat} :")
        for r in rects:
            out.append(f"   rectangle ( {_fmt(r.x)}, {_fmt(r.y)}, {_fmt(r.l)}, {_fmt(r.w)} ) ;")
        out.append("")
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out))


def _unique(outdir: str, name: str, used: set) -> str:
    base, ext = os.path.splitext(name)
    cand, i = name, 1
    while cand in used:
        i += 1
        cand = f"{base}_{i}{ext}"
    used.add(cand)
    return cand


def snap(stk_path: str, outdir: str, mode: str = 'nearest') -> Tuple[str, List[Move], List[str]]:
    """Returns (new stk path, moves, notes)."""
    if mode not in ('nearest', 'outward'):
        raise ValueError("mode must be nearest or outward")
    st = parse_stk(stk_path)
    notes: List[str] = []
    if st.dims.non_uniform:
        notes.append("non-uniform grid: nothing to snap; files copied unchanged")
    os.makedirs(outdir, exist_ok=True)
    src_text = open(st.path, 'r', encoding='utf-8', errors='replace').read()
    used = {os.path.basename(st.path)}
    moves: List[Move] = []
    text = src_text

    def rewrite(ref: str, newname: str):
        nonlocal text
        text = text.replace(f'"{ref}"', f'"{newname}"')

    # floorplans
    refs = {}
    for se in st.stack:
        if se.floorplan_path and se.floorplan_ref:
            refs[se.floorplan_path] = se.floorplan_ref
    for p, ref in refs.items():
        if p not in st.floorplans:
            continue
        name = _unique(outdir, os.path.basename(p), used)
        fp = st.floorplans[p]
        if mode == 'nearest' and not st.dims.non_uniform:
            moves += snap_floorplan(fp, st)
        write_flp(fp, os.path.join(outdir, name))
        rewrite(ref, name)
    # layouts
    lrefs = {}
    for ld in st.layers.values():
        if ld.layout_path and ld.layout_ref:
            lrefs[ld.layout_path] = ld.layout_ref
    for dd in st.dies.values():
        for dl in dd.layers:
            if dl.layout_path and dl.layout_ref:
                lrefs[dl.layout_path] = dl.layout_ref
    for p, ref in lrefs.items():
        if p not in st.layouts:
            continue
        name = _unique(outdir, os.path.basename(p), used)
        ly = st.layouts[p]
        if not st.dims.non_uniform:
            moves += snap_layout(ly, st, mode)
        write_lyt(ly, os.path.join(outdir, name))
        rewrite(ref, name)

    out_stk = os.path.join(outdir, os.path.basename(st.path))
    with open(out_stk, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return out_stk, moves, notes
