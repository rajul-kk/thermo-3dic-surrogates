"""Linter for 3D-ICE thermal datasets (.npz files written by src/export/npz_exporter.py).

Every rule is a bug this project actually shipped at least once (docs/report.md §9.23, §9.32); each produced data that
looked valid and trained models without an error. Rules are pure functions of the arrays and metadata, so they can be
unit-tested on small hand-built inputs.

  L001 arrays          coords/temp/power have matching lengths and are finite
  L002 tensor-grid     the nodes form a full x * y * z tensor grid
  L003 orientation     the lateral grid matches mesh_resolution and the package size (catches transposed fields)
  L004 temperature     temperatures are physical: not below ambient, not hotter than T_MAX_C
  L005 max-principle   the hottest node is a powered one. Steady conduction with sources >= 0 attains its maximum at
                       a source, so a strict maximum in an unpowered cell means the temperature, power and material
                       fields disagree (the 3D-ICE die-edge artefact; a transposed or layer-blind power field)
  L006 off-grid        a chiplet edge falls inside a cell: 3D-ICE heats that cell but gives it the gap material
  L007 metadata        required keys present; placement keys name real dies (a `placement_dx_*` key is parsed as a die)
  L008 energy          heat leaving the sink equals the injected power
  L009 k-override      per-sample conductivity overrides exist; the metadata's layer_{i}_k fields show the defaults
  L010 duplicate       two files in a dataset hold the same temperature field
  L011 archive         the file sits in an archive or superseded directory
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

T_MAX_C = 400.0                 # silicon melts at 1414 C; anything near this is a bug, not a hot chip
T_BELOW_AMBIENT_K = 0.5
MAX_PRINCIPLE_TOL_K = 0.01      # float32 temperatures
ENERGY_TOL = 1e-3
REQUIRED = ('geometry', 'htc', 't_ambient_kelvin', 'die_width_um', 'die_length_um', 'mesh_resolution')
SEVERITIES = ('error', 'warning', 'info')


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    file: str = ''


def _f(code, severity, message):
    return Finding(code, severity, message)


# ── rules on the arrays ──────────────────────────────────────────────────────────
def check_arrays(coords, temp, power) -> List[Finding]:
    out = []
    n = len(temp)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) != n or len(power) != n:
        return [_f('L001', 'error', f'array shapes disagree: coords {coords.shape}, temp {temp.shape}, power {power.shape}')]
    for name, a in (('coords', coords), ('temp', temp), ('power', power)):
        bad = int((~np.isfinite(a)).sum())
        if bad:
            out.append(_f('L001', 'error', f'{bad} non-finite value(s) in {name}'))
    if (power < 0).any():
        out.append(_f('L001', 'error', f'{int((power < 0).sum())} negative power densities'))
    return out


def check_tensor_grid(coords) -> List[Finding]:
    ux, uy, uz = (np.unique(coords[:, i]) for i in range(3))
    want = len(ux) * len(uy) * len(uz)
    if want != len(coords):
        return [_f('L002', 'error', f'{len(coords)} nodes are not a full {len(ux)} x {len(uy)} x {len(uz)} '
                                    f'tensor grid ({want} expected)')]
    return []


def check_orientation(coords, meta) -> List[Finding]:
    try:
        n_len, n_wid = int(meta['mesh_resolution'][0]), int(meta['mesh_resolution'][1])
        W, H = float(meta['die_width_um']), float(meta['die_length_um'])
    except (KeyError, TypeError, IndexError, ValueError):
        return []                                           # reported by check_metadata
    ux, uy = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    nx, ny = len(ux), len(uy)
    if (nx, ny) != (n_wid, n_len):
        if (nx, ny) == (n_len, n_wid) and n_len != n_wid:
            return [_f('L003', 'error', f'lateral grid is transposed: {nx} nodes along the {W / 1000:g} mm width and '
                                        f'{ny} along the {H / 1000:g} mm length; mesh_resolution says {n_wid} and {n_len}')]
        return [_f('L003', 'error', f'lateral grid {nx} x {ny} does not match mesh_resolution '
                                    f'({n_wid} along width, {n_len} along length)')]
    out = []
    for axis, u, extent, n in (('x', ux, W, nx), ('y', uy, H, ny)):
        step = extent / n
        if n > 1 and not np.allclose(np.diff(u), step, rtol=1e-3):
            out.append(_f('L003', 'error', f'{axis} spacing is {np.diff(u).mean():.2f} um, not the {step:.2f} um cell '
                                           f'size implied by mesh_resolution'))
        elif abs(u[0] - step / 2) > 1e-3 * step or abs(u[-1] - (extent - step / 2)) > 1e-3 * step:
            out.append(_f('L003', 'warning', f'{axis} nodes are not at cell centres of a {extent:g} um package'))
    return out


def check_temperature(temp, meta) -> List[Finding]:
    out = []
    t_amb = meta.get('t_ambient_kelvin')
    t = temp[np.isfinite(temp)]
    if not len(t):
        return out
    if t.max() - 273.15 > T_MAX_C:
        if meta.get('leakage_runaway'):     # a flagged electrothermal runaway has no steady state: exclude, don't fix
            out.append(_f('L004', 'warning', f'peak {t.max() - 273.15:.3g} C: thermal runaway, flagged in the metadata '
                                             f'(no physical steady state; exclude from training)'))
        else:
            out.append(_f('L004', 'error', f'peak {t.max() - 273.15:.0f} C exceeds {T_MAX_C:.0f} C'))
    if t_amb is not None and t.min() < float(t_amb) - T_BELOW_AMBIENT_K:
        out.append(_f('L004', 'error', f'minimum {t.min():.2f} K is below ambient {float(t_amb):.2f} K'))
    if t.max() - t.min() < 1e-6:
        out.append(_f('L004', 'warning', 'temperature field is constant'))
    return out


def check_max_principle(temp, power) -> List[Finding]:
    powered = power > 0
    if not powered.any() or not np.isfinite(temp).all():
        return []
    i = int(temp.argmax())
    excess = float(temp[i] - temp[powered].max())
    if not powered[i] and excess > MAX_PRINCIPLE_TOL_K:
        return [_f('L005', 'error', f'hottest node is unpowered and {excess:.2f} K above the hottest powered node '
                                    f'(violates the maximum principle)')]
    return []


# ── rules that need the geometry ─────────────────────────────────────────────────
def _offsets(meta) -> Dict[str, tuple]:
    return {k[len('placement_dx_'):]: (float(meta[k]), float(meta.get('placement_dy_' + k[len('placement_dx_'):], 0.0)))
            for k in meta if k.startswith('placement_dx_')}


def check_metadata(meta, n_nodes: int, geometry=None) -> List[Finding]:
    out = []
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        out.append(_f('L007', 'error', f'missing metadata key(s): {missing}'))
    if 'num_points' in meta and int(meta['num_points']) != n_nodes:
        out.append(_f('L007', 'error', f"num_points {meta['num_points']} but the file holds {n_nodes} nodes"))
    for k in meta:
        if k.startswith('placement_dx_') and 'placement_dy_' + k[len('placement_dx_'):] not in meta:
            out.append(_f('L007', 'error', f'{k} has no matching placement_dy_ key'))
    if geometry is not None:
        known = {d.name for d in (getattr(geometry, 'die_footprints', None) or [])} \
            | {b.name for b in geometry.power_blocks} | {'blocks'}
        for name in _offsets(meta):
            if name not in known:
                out.append(_f('L007', 'error', f"placement key 'placement_dx_{name}' names no die or block; every "
                                               f"'placement_dx_*' key is read as a die offset, so keep other "
                                               f"data under a different prefix"))
    over = meta.get('layer_k_overrides')
    if over and over != '{}':
        try:
            d = ast.literal_eval(over) if isinstance(over, str) else dict(over)
        except (ValueError, SyntaxError):
            out.append(_f('L007', 'error', f'layer_k_overrides is not a dict literal: {over!r}'))
        else:
            if d:
                out.append(_f('L009', 'info', f'conductivity overrides {d}: the layer_{{i}}_k metadata fields show '
                                              f'the defaults, not these'))
    return out


def check_off_grid(meta, geometry) -> List[Finding]:
    """Chiplet edges of the placed geometry that fall inside a 3D-ICE cell."""
    from src.core.placement import grid_steps, place_chiplets
    if not (getattr(geometry, 'die_footprints', None) or []):
        return []                                           # no footprint layout, so no gap material to mis-assign
    offs = {('' if k == 'blocks' else k): v for k, v in _offsets(meta).items()}
    try:
        placed = place_chiplets(geometry, offs)
    except ValueError:
        return []                                           # reported by check_metadata
    gx, gy = grid_steps(geometry)
    worst, names = 0.0, []
    for fp in placed.die_footprints:
        d = max(abs(v / s - round(v / s)) for v, s in
                ((fp.x, gx), (fp.x + fp.width, gx), (fp.y, gy), (fp.y + fp.height, gy)))
        if d > 1e-6:
            worst, names = max(worst, d), names + [fp.name]
    if names:
        return [_f('L006', 'warning', f'{len(names)} chiplet footprint(s) are off the {gx:g} x {gy:g} um cell grid '
                                      f'(up to {worst:.2f} cell): 3D-ICE heats the partly covered cells but gives '
                                      f'them the gap material; e.g. {names[0]}')]
    return []


def check_energy(path) -> List[Finding]:
    from src.validation.energy import energy_balance
    try:
        eb = energy_balance(Path(path), include_feedback=True)
    except Exception as exc:                                # a wrong grid or geometry makes the balance undefined
        return [_f('L008', 'warning', f'energy balance could not be computed: {type(exc).__name__}: {exc}')]
    if eb['P_field_W'] <= 0:
        return []
    if abs(eb['ratio'] - 1.0) > ENERGY_TOL:
        return [_f('L008', 'error', f"heat out / power in = {eb['ratio']:.4f} "
                                    f"({eb['P_out_W']:.2f} W out, {eb['P_field_W']:.2f} W in)")]
    return []


# ── files and datasets ───────────────────────────────────────────────────────────
def lint_arrays(coords, temp, power, meta: Dict[str, Any], geometry=None) -> List[Finding]:
    """Every rule that needs only the arrays, the metadata and (optionally) the geometry."""
    coords, temp, power = np.asarray(coords, float), np.asarray(temp, float), np.asarray(power, float)
    out = check_arrays(coords, temp, power)
    if any(f.code == 'L001' and 'shapes' in f.message for f in out):
        return out
    out += check_tensor_grid(coords)
    out += check_orientation(coords, meta)
    out += check_temperature(temp, meta)
    out += check_max_principle(temp, power)
    out += check_metadata(meta, len(temp), geometry)
    if geometry is not None:
        out += check_off_grid(meta, geometry)
    return out


def lint_file(path, energy: bool = True) -> List[Finding]:
    path = Path(path)
    try:
        d = np.load(path, allow_pickle=True)
        coords, temp, power = d['coords'], d['temp'], d['power']
        meta = d['metadata'].item()
    except Exception as exc:
        return [Finding('L001', 'error', f'cannot read file: {type(exc).__name__}: {exc}', str(path))]
    geometry = None
    try:
        from src.core.geometry_builders import get_geometry_by_name
        geometry = get_geometry_by_name(str(meta.get('geometry')))
    except Exception:
        pass                                                # custom geometry: geometry-dependent rules are skipped
    out = lint_arrays(coords, temp, power, meta, geometry)
    # the balance is only meaningful on a sane grid of a known geometry
    if energy and geometry is not None and not any(f.code in ('L001', 'L002', 'L003') for f in out):
        out += check_energy(path)
    if any(p.startswith('_archive') or '_old_' in p for p in path.parts):
        out.append(_f('L011', 'warning', 'file is inside an archive / superseded directory'))
    for f in out:
        f.file = str(path)
    return out


def lint_dataset(paths: Iterable, energy: bool = True, max_files: Optional[int] = None) -> Dict[str, Any]:
    """Lint files and/or directories (searched recursively for *.npz). Returns findings and a summary."""
    files: List[Path] = []
    for p in map(Path, paths):
        files += sorted(p.rglob('*.npz')) if p.is_dir() else [p]
    if max_files:
        files = files[:max_files]
    findings: List[Finding] = []
    seen: Dict[str, Path] = {}
    for f in files:
        findings += lint_file(f, energy)
        try:
            h = hashlib.sha1(np.ascontiguousarray(np.load(f, allow_pickle=True)['temp']).tobytes()).hexdigest()
        except Exception:
            continue
        if h in seen:
            findings.append(Finding('L010', 'warning', f'same temperature field as {seen[h].name}', str(f)))
        else:
            seen[h] = f
    by_code: Dict[str, Dict[str, int]] = {}
    for f in findings:
        by_code.setdefault(f.code, {'severity': f.severity, 'files': 0})
    for code in by_code:
        by_code[code]['files'] = len({f.file for f in findings if f.code == code})
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITIES}
    return {'files': len(files), 'files_with_errors': len({f.file for f in findings if f.severity == 'error'}),
            'counts': counts, 'by_code': dict(sorted(by_code.items())), 'findings': [asdict(f) for f in findings]}
