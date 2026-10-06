"""HotSpot (grid model, steady state) as a third, independently written solver for ``iceforge diff``.

What is translated
  * every 3D-ICE stack layer / die layer -> one HotSpot layer (layer configuration file, .lcf) with
    its thickness, resistivity, and a floorplan file (.flp);
  * ``.lyt`` layouts -> HotSpot's *heterogeneous layer* feature (``-detailed_3D on``): the layer's
    floorplan is a tiling of the chip in which every tile carries its own resistivity (7-column flp
    line).  HotSpot's grid model then blends a grid cell that is shared by several tiles by AREA
    OCCUPANCY (parallel conductances; a tile covering >= 95 % of a cell takes the whole cell).  This
    is not 3D-ICE's cell-centre rule; it is the same arithmetic area-weighted rule as refsolve.py;
  * die floorplan power -> one ptrace column per tile of the source layer, power split by tile area;
  * the 3D-ICE ``top``/``bottom heat sink`` (film coefficient) -> HotSpot's convection resistance on a
    zero-thickness, near-infinite-conductivity spreader and sink of the chip's own side length:
    ``r_convec = 1 / (h * side^2)``.

What is refused (clear message, no approximation)
  microchannel layers, pluggable heat sinks, transient models, non-uniform grids, both a top and a
  bottom heat sink (HotSpot has one primary heat path), no heat sink, non-square chips (HotSpot's
  spreader/sink are square), anisotropic conductivity, overlapping layout rectangles, overlapping
  floorplan elements, floorplan elements given as several rectangles of different material.

Known modelling differences against 3D-ICE, kept and reported rather than hidden
  * vertical: one node per layer, layers joined through half-thickness resistances on both sides (3D-ICE
    uses the full thickness at a stack end without heat sink); the sink-side boundary is a film on the
    outer layer through a numerically negligible interface/spreader/sink stack, not 3D-ICE's
    "half a cell + film" conductance;
  * lateral: grid-model cell = 3D-ICE cell by default (``r = 1``), so partial cells are blended by area
    here and by centre sampling in 3D-ICE.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import refsolve as R
from .model import Material, Rect

EPS_UM = 1e-6
TINY = 1e-9            # m: thickness of the pass-through interface, spreader and sink
BIGK = 10.0            # W/m/K: conductivity of those layers (low enough that they do not spread heat sideways:
#                        sheet conductance k*t = 1e-8 W/K, vertical resistance still < 1e-5 of the film)
DEFAULT_SH = 1.6e6     # J/m^3/K, steady state does not use it but HotSpot wants a number
MAX_TILES = 20000
DET_FILM_FACTOR = 1.5  # -detailed_3D adds an extra film/2 in series per cell (measured); divide r_convec by this
PREFLIGHT_TIMEOUT = 60  # s; a build without SuperLU iterates for minutes on the self-test
RIM_FACTOR = 1.0005   # spreader = chip x this, sink = spreader x this


class HotspotRefused(ValueError):
    """The model uses something HotSpot's grid model cannot represent faithfully."""


class HotspotError(RuntimeError):
    pass


@dataclass
class HotspotLayer:
    name: str
    instance: str
    thickness_m: float
    resistivity: float            # m K / W (layer default; tiles may override)
    sh: float                     # J/m^3/K
    power: bool
    flp: str                      # file name
    n_units: int
    ref_index: int                # index into RefModel.layers (top to bottom)
    role: str = 'mid'             # 'far' | 'mid': the half-layer whose node is the 3D-ICE layer mid-plane is 'mid'


@dataclass
class HotspotCase:
    files: Dict[str, str]                   # name -> text
    layers: List[HotspotLayer]              # in lcf order (farthest from the sink first)
    rows: int                               # grid rows (along width / y)
    cols: int                               # grid cols (along length / x)
    r: int
    n_cols: int                             # 3D-ICE cells along length
    n_rows: int
    side_m: float
    ambient: float
    sink_side: str
    argv: List[str]                         # hotspot arguments (file names relative to the working dir)
    model: R.RefModel = None
    notes: List[str] = field(default_factory=list)
    heterogeneous: bool = False

    def lcf_index(self, ref_index: int) -> int:
        for i, L in enumerate(self.layers):
            if L.ref_index == ref_index and L.role == 'mid':
                return i
        raise KeyError(ref_index)


# --------------------------------------------------------------------------- translation

def _k_iso(m: Material) -> float:
    if not m.k:
        raise HotspotRefused(f"material {m.name} has no thermal conductivity")
    if len(m.k) == 1:
        return m.k[0]
    ks = m.k[:3]
    if max(ks) - min(ks) > 1e-9 * max(ks):
        raise HotspotRefused(f"material {m.name} is anisotropic ({ks}); HotSpot's grid layers are isotropic")
    return ks[0]


def _res(m: Material) -> float:
    """m K / W from k in W/um/K."""
    return 1.0 / (_k_iso(m) * 1e6)


def _sh(m: Material) -> float:
    return m.vhc * 1e18 if m.vhc else DEFAULT_SH


def _overlap(a: Rect, b: Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x, b.x)) * max(0.0, min(a.y1, b.y1) - max(a.y, b.y))


def _breaks(vals: List[float], lo: float, hi: float) -> List[float]:
    """Sorted breakpoints in [lo, hi], values closer than EPS_UM merged; lo and hi are always kept."""
    inner = sorted(v for v in vals if lo + EPS_UM < v < hi - EPS_UM)
    out = [lo]
    for v in inner:
        if v - out[-1] > EPS_UM:
            out.append(v)
    if hi - out[-1] <= EPS_UM and len(out) > 1:
        out.pop()
    out.append(hi)
    return out


def _fmt(v_um: float) -> str:
    return repr(float(v_um) * 1e-6)


def _tile_layer(L: R.RefLayer, chip_l: float, chip_w: float):
    """Tile the chip into rectangles of constant material and constant power density.

    Returns (tiles, heterogeneous) where each tile is (x, y, l, w, Material|None, power_W); the material
    is None for the layer's own base material."""
    patch_rects = [(m, r) for m, rs in L.patches for r in rs]
    pw = list(L.power_rects)
    # overlap checks (HotSpot cannot take two materials in one place; 3D-ICE's rule for it is unspecified)
    for i in range(len(patch_rects)):
        for j in range(i + 1, len(patch_rects)):
            if _overlap(patch_rects[i][1], patch_rects[j][1]) > EPS_UM ** 2 * 1e3:
                raise HotspotRefused(f"layer {L.name}: layout rectangles overlap; the material of the overlap is "
                                     "ambiguous, refusing")
    for i in range(len(pw)):
        for j in range(i + 1, len(pw)):
            if _overlap(pw[i][0], pw[j][0]) > EPS_UM ** 2 * 1e3:
                raise HotspotRefused("floorplan elements overlap; refusing")
    for _, r in patch_rects:
        if r.x < -EPS_UM or r.y < -EPS_UM or r.x1 > chip_l + EPS_UM or r.y1 > chip_w + EPS_UM:
            raise HotspotRefused(f"layer {L.name}: layout rectangle extends outside the chip; refusing")
    for r, _ in pw:
        if r.x < -EPS_UM or r.y < -EPS_UM or r.x1 > chip_l + EPS_UM or r.y1 > chip_w + EPS_UM:
            raise HotspotRefused("floorplan element extends outside the chip; refusing")
    rects = [r for _, r in patch_rects] + [r for r, _ in pw]
    xs = _breaks([v for r in rects for v in (r.x, r.x1)], 0.0, chip_l)
    ys = _breaks([v for r in rects for v in (r.y, r.y1)], 0.0, chip_w)
    if (len(xs) - 1) * (len(ys) - 1) > MAX_TILES:
        raise HotspotRefused(f"layer {L.name}: more than {MAX_TILES} tiles")
    tiles = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
            mat = None
            for m, r in patch_rects:
                if r.x <= cx < r.x1 and r.y <= cy < r.y1:
                    mat = m
                    break
            p = 0.0
            area = (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
            for r, dens in pw:
                if r.x <= cx < r.x1 and r.y <= cy < r.y1:
                    p += dens * area
            tiles.append((xs[i], ys[j], xs[i + 1] - xs[i], ys[j + 1] - ys[j], mat, p))
    return tiles, bool(patch_rects)


def _pow2_at_least(n: int) -> int:
    g = 1
    while g < n:
        g *= 2
    return g


def translate(st, r: int = 1, ambient: Optional[float] = None, grid: Optional[int] = None) -> HotspotCase:
    """Translate a parsed stack to a HotSpot grid-model case (raises HotspotRefused / RefusedModel).

    HotSpot demands that the grid rows and columns are powers of two, so the grid cannot in general be the
    3D-ICE grid: it is the smallest power of two >= r x (3D-ICE cells) per axis (or ``grid`` if given) and the
    result is remapped onto the 3D-ICE cells by area-weighted averaging (``remap``)."""
    if st.dims.non_uniform:
        raise HotspotRefused("non-uniform 3D-ICE grid: HotSpot's grid model is uniform, refusing")
    model = R.build_model(st)          # shared refusals: transient, channels, non-uniform, missing files, no power
    if model.top and model.bottom:
        raise HotspotRefused("both a top and a bottom heat sink: HotSpot's grid model has one primary heat path "
                             "(a secondary path exists but is a different package model); refusing")
    if not model.top and not model.bottom:
        raise HotspotRefused("no heat sink: HotSpot needs a convection path to ambient; refusing")
    if abs(model.chip_l - model.chip_w) > 1e-6:
        raise HotspotRefused(f"chip is {model.chip_l:g} x {model.chip_w:g} um: HotSpot's spreader and sink are "
                             "square, so a non-square chip would get an extra uncontrolled lateral path; refusing")
    if r < 1:
        raise HotspotRefused("hotspot grid refinement r must be >= 1")
    htc, t_amb = model.top if model.top else model.bottom
    sink_side = 'top' if model.top else 'bottom'
    ambient = t_amb if ambient is None else ambient
    # lcf order: layer 0 is the farthest from the sink (HotSpot's package sits after the last lcf layer)
    order = list(range(len(model.layers)))            # model.layers is top to bottom
    if sink_side == 'top':
        order = order[::-1]
    files: Dict[str, str] = {}
    hs_layers: List[HotspotLayer] = []
    lcf = ["# generated by iceforge: layer, lateral flow, power, specific heat J/m3K, resistivity mK/W, thickness m, flp"]
    ptrace_names: List[str] = []
    ptrace_vals: List[float] = []
    hetero = False
    notes: List[str] = []
    n = 0
    for idx in order:
        L = model.layers[idx]
        base = L.base
        tiles, het = _tile_layer(L, model.chip_l, model.chip_w)
        hetero |= het
        # HotSpot places a layer's node at the face FAR from the sink and joins it to the next layer through
        # the layer's full thickness (measured: see the module docstring).  Splitting each 3D-ICE layer into two
        # half-thickness HotSpot layers puts the near-sink half's node at the mid-plane, where 3D-ICE's node is.
        for role, has_power in (('far', False), ('mid', L.is_source)):
            fname = f"L{n}.flp"
            lines = ["# name width_m height_m left_x_m bottom_y_m [specific_heat resistivity]"]
            for k, (x, y, l, w, mat, p) in enumerate(tiles):
                nm = f"L{n}_u{k}"
                if het:
                    m_ = mat or base
                    lines.append(f"{nm}\t{_fmt(l)}\t{_fmt(w)}\t{_fmt(x)}\t{_fmt(y)}\t{_sh(m_):.6g}\t{_res(m_):.9g}")
                else:
                    lines.append(f"{nm}\t{_fmt(l)}\t{_fmt(w)}\t{_fmt(x)}\t{_fmt(y)}")
                if has_power:
                    ptrace_names.append(nm)
                    ptrace_vals.append(p)
            files[fname] = '\n'.join(lines) + '\n'
            lcf += [f"# layer {n}: {L.instance}/{L.name} {role} half ({'source' if has_power else 'passive'})",
                    str(n), 'Y', 'Y' if has_power else 'N', f"{_sh(base):.6g}", f"{_res(base):.9g}",
                    repr(0.5 * L.height * 1e-6), fname, ""]
            hs_layers.append(HotspotLayer(L.name, L.instance, 0.5 * L.height * 1e-6, _res(base), _sh(base),
                                          has_power, fname, len(tiles), idx, role))
            n += 1
    if not ptrace_names:
        raise HotspotRefused("no source layer with power")
    files['stack.lcf'] = '\n'.join(lcf) + '\n'
    files['power.ptrace'] = '\t'.join(ptrace_names) + '\n' + '\t'.join(f"{v:.12g}" for v in ptrace_vals) + '\n'
    side_m = model.chip_l * 1e-6
    htc_si = htc * 1e12
    # HotSpot divides by the spreader/sink periphery area, so both must be strictly larger than the chip
    # (equal sizes give NaN).  A rim of RIM_FACTOR - 1 per side keeps the rim below 0.1 % of the film area.
    spreader_m, sink_m = side_m * RIM_FACTOR, side_m * RIM_FACTOR * RIM_FACTOR
    r_convec = 1.0 / (htc_si * sink_m * sink_m)
    if hetero:
        r_convec /= DET_FILM_FACTOR
    n_cols, n_rows = int(round(model.chip_l / model.cell_l)), int(round(model.chip_w / model.cell_w))
    if grid is not None:
        if grid < 1 or grid & (grid - 1):
            raise HotspotRefused("--hotspot-grid must be a power of two")
        cols = rows = grid
    else:
        cols, rows = _pow2_at_least(n_cols * r), _pow2_at_least(n_rows * r)
    argv = ['-p', 'power.ptrace', '-grid_layer_file', 'stack.lcf', '-model_type', 'grid',
            '-detailed_3D', 'on' if hetero else 'off',
            '-grid_rows', str(rows), '-grid_cols', str(cols),
            '-s_sink', repr(sink_m), '-t_sink', repr(TINY), '-k_sink', repr(BIGK),
            '-s_spreader', repr(spreader_m), '-t_spreader', repr(TINY), '-k_spreader', repr(BIGK),
            '-t_interface', repr(TINY), '-k_interface', repr(BIGK),
            '-r_convec', repr(r_convec), '-c_convec', '0',
            '-ambient', repr(float(ambient)), '-init_temp', repr(float(ambient)),
            '-model_secondary', '0',
            '-steady_file', 'out.steady', '-grid_steady_file', 'out.grid.steady']
    return HotspotCase(files=files, layers=hs_layers, rows=rows, cols=cols, r=r, n_cols=n_cols, n_rows=n_rows,
                       side_m=side_m, ambient=float(ambient), sink_side=sink_side, argv=argv, model=model,
                       notes=notes, heterogeneous=hetero)


# --------------------------------------------------------------------------- running

DEFAULT_WSL_HOTSPOT = "HotSpot/hotspot"          # relative to $HOME inside WSL


def _wsl(*args, input=None, timeout=300):
    return subprocess.run(["wsl", *args], capture_output=True, text=True, errors="replace", input=input,
                          timeout=timeout)


def find_hotspot(exe: Optional[str] = None) -> Tuple[str, str]:
    """Return (kind, path): kind is 'wsl' or 'native'.  ``exe`` or $HOTSPOT_EXECUTABLE may be
    ``wsl /path/hotspot`` or a native path; otherwise WSL ``~/HotSpot/hotspot``, then ``hotspot`` on PATH."""
    exe = exe or os.environ.get("HOTSPOT_EXECUTABLE")
    if exe:
        exe = exe.strip()
        if exe.lower().startswith("wsl "):
            return "wsl", exe[4:].strip()
        if exe.startswith("/") and os.name == "nt":
            return "wsl", exe
        return "native", exe
    if os.name == "nt" or shutil.which("wsl"):
        try:
            home = _wsl("-e", "printenv", "HOME", timeout=30).stdout.strip()
            if home and _wsl("-e", "test", "-x", f"{home}/{DEFAULT_WSL_HOTSPOT}", timeout=30).returncode == 0:
                return "wsl", f"{home}/{DEFAULT_WSL_HOTSPOT}"
        except Exception:
            pass
    p = shutil.which("hotspot")
    if p:
        return "native", p
    raise HotspotError("HotSpot not found: pass --hotspot-exe 'wsl /path/to/hotspot', set HOTSPOT_EXECUTABLE, "
                       "or install it at ~/HotSpot/hotspot in WSL")


def run_hotspot(case: HotspotCase, exe: Optional[str] = None, timeout: float = 600) -> Tuple[str, str]:
    """Run HotSpot on the case; returns (grid steady text, stdout).

    WSL: inputs are streamed into a ``mktemp -d`` directory (no Windows paths are ever passed on a
    command line, so spaces in the project path cannot break it) and the result is read back with ``cat``."""
    kind, path = find_hotspot(exe)
    if kind == "wsl":
        r = _wsl("mktemp", "-d", timeout=60)
        if r.returncode != 0 or not r.stdout.strip():
            raise HotspotError(f"wsl mktemp -d failed: {r.stderr}")
        tmp = r.stdout.strip()
        try:
            for name, text in case.files.items():
                w = _wsl("bash", "-c", f"cat > '{tmp}/{name}'", input=text, timeout=60)
                if w.returncode != 0:
                    raise HotspotError(f"could not stage {name} in WSL: {w.stderr}")
            res = _wsl("bash", "-c", f"cd '{tmp}' && '{path}' " + ' '.join(case.argv), timeout=timeout)
            if res.returncode != 0:
                raise HotspotError(f"HotSpot failed (exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}")
            out = _wsl("cat", f"{tmp}/out.grid.steady", timeout=60)
            if out.returncode != 0 or not out.stdout.strip():
                raise HotspotError(f"HotSpot produced no grid output: {res.stdout} {res.stderr}")
            return out.stdout, res.stdout
        except subprocess.TimeoutExpired:
            raise HotspotError("HotSpot timed out")
        finally:
            _wsl("rm", "-rf", tmp, timeout=60)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="iceforge_hs_") as td:
        for name, text in case.files.items():
            with open(os.path.join(td, name), "w", newline="\n") as f:
                f.write(text)
        try:
            res = subprocess.run([path, *case.argv], cwd=td, capture_output=True, text=True, errors="replace",
                                 timeout=timeout)
        except subprocess.TimeoutExpired:
            raise HotspotError("HotSpot timed out")
        if res.returncode != 0:
            raise HotspotError(f"HotSpot failed (exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}")
        p = os.path.join(td, "out.grid.steady")
        if not os.path.isfile(p):
            raise HotspotError(f"HotSpot produced no grid output: {res.stdout} {res.stderr}")
        with open(p) as f:
            return f.read(), res.stdout


def parse_grid_steady(text: str, case: HotspotCase) -> Dict[int, np.ndarray]:
    """{lcf layer index: T (cols=length index, rows=width index) in K}.

    HotSpot prints ``Layer n:`` then ``idx<TAB>T`` with idx = row * cols + col and row 0 at the top (max y)."""
    layers: Dict[int, List[float]] = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'^Layer\s+(\d+):', line)
        if m:
            cur = int(m.group(1))
            layers[cur] = []
            continue
        if cur is None or not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                layers[cur].append(float(parts[1]))
            except ValueError:
                pass
    out = {}
    n = case.rows * case.cols
    for k, v in layers.items():
        if len(v) != n:
            continue
        a = np.array(v).reshape(case.rows, case.cols)       # [row (y), col (x)]
        # HotSpot's row 0 is the TOP of the die (largest y): verified against the reference solver, where
        # flipping y drops the max difference from 4.7 K to 0.06 K on the misaligned repro
        out[k] = a[::-1].T.copy()                            # -> (length index, width index), y increasing
    return out




def _overlap_matrix(n_out: int, n_in: int) -> np.ndarray:
    """W[i, j] = share of output cell i (of n_out equal cells on [0, 1]) lying in input cell j."""
    eo = np.linspace(0.0, 1.0, n_out + 1)
    ei = np.linspace(0.0, 1.0, n_in + 1)
    lo = np.maximum(eo[:-1, None], ei[None, :-1])
    hi = np.minimum(eo[1:, None], ei[None, 1:])
    return np.clip(hi - lo, 0.0, None) * n_out


def remap(field: np.ndarray, n_cols: int, n_rows: int) -> np.ndarray:
    """Area-weighted average of a (cols, rows) field of equal cells onto an (n_cols, n_rows) grid of the
    same domain.  Exact for any pair of grid sizes (conserves the mean)."""
    return _overlap_matrix(n_cols, field.shape[0]) @ field @ _overlap_matrix(n_rows, field.shape[1]).T


_PREFLIGHT_CACHE: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]] = {}


def _selftest_case(k_pkg: float, detailed: bool = False) -> HotspotCase:
    """Two-layer chip with a quarter-chip hot block (stiff: k = 0.7 over k = 148), bottom-sink film.
    ``k_pkg`` is the conductivity of the pass-through interface/spreader/sink layers; changing it by a factor
    ten moves the true answer by well under a millikelvin."""
    side, h, P = 0.01, 2e4, 10.0
    rim = RIM_FACTOR
    sink = side * rim * rim
    half = side / 2
    flp0 = "".join(f"{n}\t{w!r}\t{hh!r}\t{x!r}\t{y!r}\n" for n, w, hh, x, y in [
        ("A", half, half, 0.0, 0.0), ("B", half, half, half, 0.0), ("C", half, half, 0.0, half),
        ("D", half, half, half, half)])
    files = {
        'L0.flp': flp0, 'L1.flp': f"U\t{side!r}\t{side!r}\t0.0\t0.0\n",
        'stack.lcf': ("0\nY\nY\n1.6e6\n{r0!r}\n5e-5\nL0.flp\n\n1\nY\nN\n1.6e6\n{r1!r}\n3e-4\nL1.flp\n"
                      .format(r0=1.0 / 0.7, r1=1.0 / 148.0)),
        'power.ptrace': f"A\tB\tC\tD\n{P!r}\t0\t0\t0\n",
    }
    argv = ['-p', 'power.ptrace', '-grid_layer_file', 'stack.lcf', '-model_type', 'grid', '-detailed_3D', 'on' if detailed else 'off',
            '-grid_rows', '16', '-grid_cols', '16', '-s_sink', repr(sink), '-t_sink', repr(TINY), '-k_sink', repr(k_pkg),
            '-s_spreader', repr(side * rim), '-t_spreader', repr(TINY), '-k_spreader', repr(k_pkg),
            '-t_interface', repr(TINY), '-k_interface', repr(k_pkg),
            '-r_convec', repr(1.0 / (h * sink * sink) / (DET_FILM_FACTOR if detailed else 1.0)),
            '-c_convec', '0', '-ambient', '300.0', '-init_temp', '300.0', '-model_secondary', '0',
            '-steady_file', 'out.steady', '-grid_steady_file', 'out.grid.steady']
    return HotspotCase(files=files, layers=[HotspotLayer('L0', 'L0', 5e-5, 1 / 0.7, 1.6e6, True, 'L0.flp', 4, 0, 'mid'),
                                            HotspotLayer('L1', 'L1', 3e-4, 1 / 148, 1.6e6, False, 'L1.flp', 1, 1, 'mid')],
                       rows=16, cols=16, r=1, n_cols=16, n_rows=16, side_m=side, ambient=300.0, sink_side='bottom',
                       argv=argv)


def preflight(exe: Optional[str] = None, tol_K: float = 0.02, tol_det: float = 0.015) -> Tuple[Optional[str], Optional[str]]:
    """Self-tests of the HotSpot build: returns (error, error_for_layouts), each None when the test passes.

    1. Convergence.  A HotSpot built without SuperLU (the default ``make``) solves the steady state iteratively
       and returns unconverged fields on stiff stacks without any warning.  A stiff two-layer chip is solved
       twice with pass-through layers whose conductivity differs tenfold (true answers equal to < 1e-3 K); the
       two must agree.
    2. Heterogeneous mode.  ``-detailed_3D on`` (needed for per-tile conductivity, i.e. layouts) is compared with
       the ordinary mode on the same homogeneous chip, with the film compensation of ``DET_FILM_FACTOR``
       applied (chip layers only).  Layout models are refused if they differ by more than ``tol_det`` of the rise."""
    kind, path = find_hotspot(exe)
    key = (kind, path)
    if key in _PREFLIGHT_CACHE:
        return _PREFLIGHT_CACHE[key]
    err = det_err = None
    try:
        fields = []
        for kp in (BIGK, BIGK * 10):
            case = _selftest_case(kp)
            txt, _ = run_hotspot(case, exe, timeout=PREFLIGHT_TIMEOUT)
            fields.append(parse_grid_steady(txt, case))
        d = max(float(np.abs(fields[0][k] - fields[1][k]).max()) for k in fields[0])
        if d > tol_K:
            err = (f"HotSpot self-test failed: the same stiff stack solved with two numerically equivalent "
                   f"package settings differs by {d:.3f} K. This build probably solves the steady state "
                   "iteratively (no SuperLU) and does not converge; rebuild with `make SUPERLU=1` "
                   "(needs libsuperlu-dev) and pass it with --hotspot-exe")
        else:
            case = _selftest_case(BIGK, detailed=True)
            txt, _ = run_hotspot(case, exe, timeout=PREFLIGHT_TIMEOUT)
            fd = parse_grid_steady(txt, case)
            ks = range(len(case.layers))          # chip layers only: the package layers' own temperatures differ
            d2 = max(float(np.abs(fd[k] - fields[0][k]).max()) for k in ks)
            rise = max(float(fields[0][k].max()) for k in ks) - 300.0
            if d2 > tol_det * rise:
                det_err = (f"HotSpot -detailed_3D mode disagrees with the ordinary mode by {d2:.3f} K on a "
                           "homogeneous control even after the film compensation; layouts (per-tile "
                           "conductivity) cannot be represented with this build")
    except HotspotError as e:
        err = (f"HotSpot self-test could not run: {e}" + (" (a build without SuperLU is far too slow on stiff stacks; "
               "rebuild with `make SUPERLU=1`)" if "timed out" in str(e) else ""))
    _PREFLIGHT_CACHE[key] = (err, det_err)
    return err, det_err


def hotspot_fields(case: HotspotCase, exe: Optional[str] = None, check: bool = True) -> Dict[int, np.ndarray]:
    """Solve and return {lcf layer index: T (length index, width index)} on HotSpot's own grid."""
    if check:
        err, det_err = preflight(exe)
        if err:
            raise HotspotError(err)
        if det_err and case.heterogeneous:
            raise HotspotError(det_err)
    text, _ = run_hotspot(case, exe)
    return parse_grid_steady(text, case)
