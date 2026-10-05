"""Tolerant parser for the 3D-ICE 4.0 stack-description subset (.stk, .flp, .lyt).

Units follow 3D-ICE: lengths in micrometres, conductivity in W/um/K, HTC in W/um^2/K.
The parser never raises on unknown statements; it records them in ``warnings``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

_TOKEN = re.compile(
    r'''"[^"]*"                                   # quoted path
      | [-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)? # number
      | [A-Za-z_][\w\-]*                          # word (allows non-uniform)
      | [:;,().]                                  # punctuation
    ''', re.X)


def strip_comments(text: str) -> str:
    """Remove // and /* */ comments, leaving quoted strings intact."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = text.find('"', i + 1)
            j = n - 1 if j < 0 else j
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
            out.append(' ')
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(strip_comments(text))


def _units(tokens: List[str]):
    """Split tokens into (header_tokens_or_None, [statement_tokens, ...]) groups.

    A unit ends with ';' (statement) or ':' (section header).  Statements before the
    first header go under header None.
    """
    groups = [(None, [])]
    cur: List[str] = []
    for t in tokens:
        if t == ';':
            groups[-1][1].append(cur)
            cur = []
        elif t == ':':
            groups.append((cur, []))
            cur = []
        else:
            cur.append(t)
    if cur:
        groups[-1][1].append(cur)
    return groups


def _nums(tokens: List[str]) -> List[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _unq(t: str) -> str:
    return t[1:-1] if len(t) >= 2 and t[0] == '"' and t[-1] == '"' else t


def _kw(tokens: List[str]) -> str:
    return ' '.join(t.lower() for t in tokens if not re.match(r'^[-+\d.]', t) and t not in ',()')


# --------------------------------------------------------------------------- data

@dataclass
class Rect:
    x: float
    y: float
    l: float  # extent along chip length (x)
    w: float  # extent along chip width (y)

    @property
    def x1(self):
        return self.x + self.l

    @property
    def y1(self):
        return self.y + self.w


@dataclass
class Element:
    name: str
    rects: List[Rect] = field(default_factory=list)
    power: List[float] = field(default_factory=list)
    material: Optional[str] = None
    discretization: Optional[Tuple[float, float]] = None


@dataclass
class Floorplan:
    path: str
    elements: List[Element] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Material:
    name: str
    k: List[float] = field(default_factory=list)   # W/um/K (1 or 3 values)
    vhc: Optional[float] = None                    # J/um^3/K

    @property
    def k_min(self):
        return min(self.k) if self.k else None

    @property
    def k_max(self):
        return max(self.k) if self.k else None


@dataclass
class Layout:
    path: str
    local_materials: Dict[str, Material] = field(default_factory=dict)
    shapes: Dict[str, List[Rect]] = field(default_factory=dict)  # material name -> rectangles
    warnings: List[str] = field(default_factory=list)


@dataclass
class HeatSink:
    side: str  # top | bottom
    htc: Optional[float] = None
    temperature: Optional[float] = None
    extra: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class Dimensions:
    chip_l: float = 0.0
    chip_w: float = 0.0
    cell_l: float = 0.0
    cell_w: float = 0.0
    non_uniform: bool = False

    @property
    def n_cols(self):  # cells along length (x)
        return int(self.chip_l / self.cell_l) if self.cell_l else 0

    @property
    def n_rows(self):  # cells along width (y)
        return int(self.chip_w / self.cell_w) if self.cell_w else 0


@dataclass
class LayerDef:
    name: str
    height: Optional[float] = None
    material: Optional[str] = None
    layout_path: Optional[str] = None  # resolved
    layout_ref: Optional[str] = None   # as written


@dataclass
class DieLayer:
    height: Optional[float]
    material: Optional[str]
    ref: Optional[str]          # name of a global layer, if referenced
    is_source: bool = False
    layout_path: Optional[str] = None
    layout_ref: Optional[str] = None


@dataclass
class DieDef:
    name: str
    layers: List[DieLayer] = field(default_factory=list)  # top to bottom


@dataclass
class StackElement:
    kind: str                   # die | layer | channel
    instance: str
    ref: str
    floorplan_path: Optional[str] = None
    floorplan_ref: Optional[str] = None
    discretization: Optional[Tuple[float, float]] = None


@dataclass
class Output:
    kind: str                   # Tmap, T, Tflp, Tflpel, Tcoolant, Pmap, T3d
    instance: Optional[str]
    path: Optional[str]         # resolved
    path_ref: Optional[str]
    when: Optional[str]
    raw: str = ""


@dataclass
class Solver:
    mode: Optional[str] = None  # steady | transient
    initial_temperature: Optional[float] = None
    step: Optional[float] = None
    slot: Optional[float] = None
    cores: Optional[float] = None


@dataclass
class Stack:
    path: str
    base_dir: str
    materials: Dict[str, Material] = field(default_factory=dict)
    heatsinks: List[HeatSink] = field(default_factory=list)
    dims: Dimensions = field(default_factory=Dimensions)
    layers: Dict[str, LayerDef] = field(default_factory=dict)
    dies: Dict[str, DieDef] = field(default_factory=dict)
    stack: List[StackElement] = field(default_factory=list)  # top to bottom
    solver: Solver = field(default_factory=Solver)
    outputs: List[Output] = field(default_factory=list)
    floorplans: Dict[str, Floorplan] = field(default_factory=dict)   # by resolved path
    layouts: Dict[str, Layout] = field(default_factory=dict)         # by resolved path
    missing_files: List[Tuple[str, str]] = field(default_factory=list)  # (kind, path)
    warnings: List[str] = field(default_factory=list)

    @property
    def sink_temperature(self) -> Optional[float]:
        for side in ("top", "bottom"):
            for h in self.heatsinks:
                if h.side == side and h.temperature is not None:
                    return h.temperature
        return self.solver.initial_temperature


# --------------------------------------------------------------------------- parsers

def _resolve(base: str, p: str) -> str:
    m = re.match(r'^/mnt/([A-Za-z])(/.*)?$', p)
    if m and os.name == 'nt':          # a stk written for WSL: /mnt/c/x -> C:/x so files can be read on Windows
        p = f"{m.group(1).upper()}:{m.group(2) or '/'}"
    absolute = os.path.isabs(p) or re.match(r'^[A-Za-z]:[\\/]', p) or p.startswith('/')
    return os.path.normpath(p if absolute else os.path.join(base, p))


def _read(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def _parse_rect_statements(stmts: List[List[str]]):
    """Shared by .flp elements and .lyt shapes: returns (rects, extras, material)."""
    rects: List[Rect] = []
    pos = dim = None
    extras: Dict[str, List[float]] = {}
    mat = None
    for s in stmts:
        k = _kw(s)
        n = _nums(s)
        if k.startswith('rectangle') and len(n) >= 4:
            rects.append(Rect(*n[:4]))
        elif k.startswith('position') and len(n) >= 2:
            pos = n[:2]
        elif k.startswith('dimension') and len(n) >= 2:
            dim = n[:2]
        elif k.startswith('power values'):
            extras['power'] = n
        elif k.startswith('discretization'):
            extras['discretization'] = n
        elif k.startswith('material') and len(s) >= 2:
            mat = s[1]
    if pos is not None and dim is not None:
        rects.insert(0, Rect(pos[0], pos[1], dim[0], dim[1]))
    return rects, extras, mat


def parse_flp(path: str) -> Floorplan:
    fp = Floorplan(path=path)
    for hdr, stmts in _units(tokenize(_read(path))):
        if hdr is None:
            if any(stmts):
                fp.warnings.append("statements before first element ignored")
            continue
        if not hdr:
            continue
        rects, extras, mat = _parse_rect_statements(stmts)
        d = extras.get('discretization')
        fp.elements.append(Element(
            name=hdr[0], rects=rects, power=extras.get('power', []), material=mat,
            discretization=(d[0], d[1]) if d and len(d) >= 2 else None))
    return fp


def parse_lyt(path: str) -> Layout:
    ly = Layout(path=path)
    for hdr, stmts in _units(tokenize(_read(path))):
        if hdr is None or not hdr:
            continue
        if hdr[0].lower() == 'material' and len(hdr) >= 2:
            m = Material(name=hdr[1])
            for s in stmts:
                k = _kw(s)
                if k.startswith('thermal conductivity'):
                    m.k = _nums(s)
                elif k.startswith('volumetric heat capacity'):
                    n = _nums(s)
                    m.vhc = n[0] if n else None
            ly.local_materials[m.name] = m
            continue
        rects, _, _ = _parse_rect_statements(stmts)
        ly.shapes.setdefault(hdr[0], []).extend(rects)
    return ly


def _path_in(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        if t.startswith('"'):
            return _unq(t)
    return None


def parse_stk(path: str, load_children: bool = True) -> Stack:
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    st = Stack(path=path, base_dir=base)

    for hdr, stmts in _units(tokenize(_read(path))):
        if hdr is None:
            if any(stmts):
                st.warnings.append("statements before first section ignored")
            continue
        low_h = [t.lower() for t in hdr]
        h = ' '.join(low_h[:3]) if low_h[1:3] == ['heat', 'sink'] else (low_h[0] if low_h else '')
        if h == 'material':
            m = Material(name=hdr[1] if len(hdr) > 1 else '?')
            for s in stmts:
                k = _kw(s)
                if k.startswith('thermal conductivity'):
                    m.k = _nums(s)
                elif k.startswith('volumetric heat capacity'):
                    n = _nums(s)
                    m.vhc = n[0] if n else None
            st.materials[m.name] = m
        elif h in ('top heat sink', 'bottom heat sink'):
            hs = HeatSink(side=h.split()[0])
            for s in stmts:
                k, n = _kw(s), _nums(s)
                if k.startswith('heat transfer coefficient') and n:
                    hs.htc = n[0]
                elif k.startswith('temperature') and n:
                    hs.temperature = n[0]
                else:
                    hs.extra[k] = n
            st.heatsinks.append(hs)
        elif h == 'dimensions':
            d = st.dims
            for s in stmts:
                k, n = _kw(s), _nums(s)
                if k.startswith('chip') and len(n) >= 2:
                    d.chip_l, d.chip_w = n[0], n[1]
                elif k.startswith('cell') and len(n) >= 2:
                    d.cell_l, d.cell_w = n[0], n[1]
                elif k.startswith('non-uniform'):
                    d.non_uniform = any(t.lower() == 'true' for t in s)
        elif h == 'layer':
            ld = LayerDef(name=hdr[1] if len(hdr) > 1 else '?')
            for s in stmts:
                k, n = _kw(s), _nums(s)
                if k.startswith('height') and n:
                    ld.height = n[0]
                elif k.startswith('material') and len(s) >= 2:
                    ld.material = s[1]
                elif k.startswith('layout'):
                    p = _path_in(s)
                    if p:
                        ld.layout_ref, ld.layout_path = p, _resolve(base, p)
            st.layers[ld.name] = ld
        elif h == 'die':
            dd = DieDef(name=hdr[1] if len(hdr) > 1 else '?')
            for s in stmts:
                if not s or s[0].lower() not in ('layer', 'source'):
                    continue
                is_src = s[0].lower() == 'source'
                rest = s[1:]
                nums = _nums(rest)
                if nums and len(rest) >= 2:
                    dl = DieLayer(height=nums[0], material=rest[1], ref=None, is_source=is_src)
                elif rest:
                    dl = DieLayer(height=None, material=None, ref=rest[0], is_source=is_src)
                else:
                    continue
                p = _path_in(rest)
                if p:
                    dl.layout_ref, dl.layout_path = p, _resolve(base, p)
                dd.layers.append(dl)
            st.dies[dd.name] = dd
        elif h == 'stack':
            for s in stmts:
                if not s:
                    continue
                kind = s[0].lower()
                if kind in ('die', 'layer', 'channel') and len(s) >= 2:
                    se = StackElement(kind=kind, instance=s[1],
                                      ref=s[2] if len(s) > 2 and kind != 'channel' else '')
                    low = [t.lower() for t in s]
                    if 'floorplan' in low:
                        p = _path_in(s)
                        if p:
                            se.floorplan_ref, se.floorplan_path = p, _resolve(base, p)
                    if 'discretization' in low:
                        n = _nums(s[low.index('discretization'):])
                        if len(n) >= 2:
                            se.discretization = (n[0], n[1])
                    st.stack.append(se)
        elif h == 'solver':
            sv = st.solver
            for s in stmts:
                k, n = _kw(s), _nums(s)
                low = [t.lower() for t in s]
                if k.startswith('steady'):
                    sv.mode = 'steady'
                elif k.startswith('transient'):
                    sv.mode = 'transient'
                    if 'step' in low and len(s) > low.index('step') + 1:
                        sv.step = float(s[low.index('step') + 1])
                    if 'slot' in low and len(s) > low.index('slot') + 1:
                        sv.slot = float(s[low.index('slot') + 1])
                elif k.startswith('initial temperature') and n:
                    sv.initial_temperature = n[0]
                elif k.startswith('numofcores') and n:
                    sv.cores = n[0]
        elif h == 'output':
            for s in stmts:
                if not s:
                    continue
                p = _path_in(s)
                inst = s[2] if len(s) > 2 and s[1] == '(' and not s[2].startswith('"') else None
                when = s[-2] if len(s) > 2 and s[-1] == ')' else None
                st.outputs.append(Output(kind=s[0], instance=inst,
                                         path=_resolve(base, p) if p else None,
                                         path_ref=p, when=when, raw=' '.join(s)))
        else:
            st.warnings.append(f"unknown section ignored: {' '.join(hdr)}")

    if st.dims.non_uniform:
        st.warnings.append("non-uniform grid: off-grid checks skipped")
    if load_children:
        _load_children(st)
    return st


def _load_children(st: Stack) -> None:
    flp_paths = {se.floorplan_path for se in st.stack if se.floorplan_path}
    lyt_paths = {ld.layout_path for ld in st.layers.values() if ld.layout_path}
    for dd in st.dies.values():
        for dl in dd.layers:
            if dl.layout_path:
                lyt_paths.add(dl.layout_path)
    for p in sorted(flp_paths):
        if os.path.isfile(p):
            st.floorplans[p] = parse_flp(p)
        else:
            st.missing_files.append(('floorplan', p))
    for p in sorted(lyt_paths):
        if os.path.isfile(p):
            st.layouts[p] = parse_lyt(p)
        else:
            st.missing_files.append(('layout', p))


# --------------------------------------------------------------------------- helpers

def resolved_die_layers(st: Stack, dd: DieDef) -> List[DieLayer]:
    """Die layers with global-layer references expanded (height/material/layout filled in)."""
    out = []
    for dl in dd.layers:
        if dl.ref and dl.ref in st.layers:
            ld = st.layers[dl.ref]
            out.append(DieLayer(height=ld.height, material=ld.material, ref=dl.ref,
                                is_source=dl.is_source, layout_path=ld.layout_path,
                                layout_ref=ld.layout_ref))
        else:
            out.append(dl)
    return out


def to_dict(st: Stack) -> dict:
    d = asdict(st)
    d['dims']['n_cols'] = st.dims.n_cols
    d['dims']['n_rows'] = st.dims.n_rows
    d['sink_temperature'] = st.sink_temperature
    return d


def summary(st: Stack) -> str:
    L: List[str] = []
    dm = st.dims
    L.append(f"stack      {st.path}")
    L.append(f"chip       {dm.chip_l:g} x {dm.chip_w:g} um (length x width)")
    if dm.non_uniform:
        L.append("grid       non-uniform grid: off-grid checks skipped")
    else:
        L.append(f"grid       cell {dm.cell_l:g} x {dm.cell_w:g} um -> "
                 f"{dm.n_cols} columns (length) x {dm.n_rows} rows (width)")
    L.append("materials")
    for m in st.materials.values():
        ks = ', '.join(f"{v:g}" for v in m.k)
        L.append(f"  {m.name:12s} k = {ks} W/um/K   C = {m.vhc if m.vhc is not None else '?'} J/um^3/K")
    for h in st.heatsinks:
        L.append(f"{h.side} heat sink: HTC {h.htc} W/um^2/K, T {h.temperature} K")
    if st.layers:
        L.append("layers")
        for ld in st.layers.values():
            lay = f"  layout {ld.layout_ref}" if ld.layout_ref else ""
            L.append(f"  {ld.name:12s} {ld.height} um  {ld.material}{lay}")
    for dd in st.dies.values():
        L.append(f"die {dd.name} (top to bottom)")
        for dl in dd.layers:
            what = dl.ref if dl.ref else f"{dl.height} um {dl.material}"
            L.append(f"  {'source ' if dl.is_source else 'layer  '}{what}")
    L.append("stack (top to bottom)")
    for se in st.stack:
        fp = f" floorplan {se.floorplan_ref}" if se.floorplan_ref else ""
        L.append(f"  {se.kind:5s} {se.instance} <- {se.ref}{fp}")
    sv = st.solver
    L.append(f"solver     {sv.mode}, initial T {sv.initial_temperature}")
    for o in st.outputs:
        L.append(f"output     {o.raw}")
    for p, fp in st.floorplans.items():
        L.append(f"floorplan {os.path.basename(p)}: {len(fp.elements)} elements")
        for e in fp.elements:
            rs = '; '.join(f"({r.x:g},{r.y:g},{r.l:g},{r.w:g})" for r in e.rects)
            L.append(f"  {e.name:14s} {rs}  power {e.power[:3]}{'...' if len(e.power) > 3 else ''}")
    for p, ly in st.layouts.items():
        L.append(f"layout {os.path.basename(p)}: {sum(len(v) for v in ly.shapes.values())} rectangles")
        for mat, rs in ly.shapes.items():
            for r in rs:
                L.append(f"  {mat:10s} rectangle({r.x:g},{r.y:g},{r.l:g},{r.w:g})")
    for k, p in st.missing_files:
        L.append(f"MISSING {k}: {p}")
    for w in st.warnings:
        L.append(f"note: {w}")
    return '\n'.join(L)
