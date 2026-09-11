"""
Per-scenario chiplet placement — the fix for the defect diagnosed in docs/report.md §9.14.

Measurement showed every geometry in this benchmark has exactly ONE heat-source support
pattern across all of its scenarios (mean pairwise IoU 1.000): the sources never move, only
their amplitudes change. With the medium and mesh also fixed, the solution map is

    T(x) = T_amb + sum_b A_b(x) * Q_b        with A_b fixed

which is exactly, and only, the hypothesis class of a linear model — and the regime classical
power blurring has solved to ~1 K since 2007 (Kemper et al., arXiv:0709.1850). A benchmark
confined to it cannot discriminate between surrogate architectures.

This module moves the sources. It translates a *chiplet* — a `DiePrint` together with the
`PowerBlock`s sitting on it — as a rigid unit, so that in 2.5D geometries the silicon island
moves with its heat, changing the lateral material distribution and therefore the thermal
operator itself. That is the property IC-ThermBench has (chiplets move between samples, mean
pairwise support IoU 0.449) and this benchmark lacked.

The distinction matters and is the point of the experiment in
`scripts/gen_moving_source_pilot.py`:

  * In a laterally HOMOGENEOUS geometry (e.g. geometry1, a uniform silicon die), moving a
    power block changes where heat enters but not the medium. The operator is unchanged, so
    T is still a fixed linear functional of the per-cell power map. A linear model given the
    power map should still do well.
  * In a 2.5D geometry (geometry4/5/6/7), a chiplet IS a material region — silicon in
    underfill. Moving it rearranges the medium, so the operator changes per scenario and no
    single fixed linear map suffices.

Blocks are associated to dies by geometric containment rather than by name, so this works
across every geometry in the project without a naming convention.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

Offset = Tuple[float, float]


def associate_blocks_to_dies(geometry) -> Dict[str, List[str]]:
    """
    Map each DiePrint name -> names of the PowerBlocks lying inside it.

    Containment, not naming: a block belongs to the footprint that contains its centre.
    Blocks in no footprint (and all blocks of non-2.5D geometries, which have no footprints)
    come back under the key ''.
    """
    groups: Dict[str, List[str]] = {}
    prints = list(getattr(geometry, 'die_footprints', None) or [])
    for blk in geometry.power_blocks:
        cx = blk.x + blk.width / 2.0
        cy = blk.y + blk.height / 2.0
        owner = ''
        for dp in prints:
            if dp.contains_point(cx, cy):
                owner = dp.name
                break
        groups.setdefault(owner, []).append(blk.name)
    return groups


def _bounds(items) -> Tuple[float, float, float, float]:
    x0 = min(i.x for i in items)
    y0 = min(i.y for i in items)
    x1 = max(i.x + i.width for i in items)
    y1 = max(i.y + i.height for i in items)
    return x0, y0, x1, y1


def place_chiplets(geometry, offsets: Dict[str, Offset]):
    """
    Return a copy of `geometry` with each named chiplet translated by its (dx, dy) in µm.

    A chiplet is its DiePrint plus the PowerBlocks contained in it; both move together so
    the material region and its heat source stay registered. Use the key '' to translate
    blocks that belong to no footprint (the only option for non-2.5D geometries).

    Raises ValueError if any moved element would leave the package footprint; the caller is
    expected to sample offsets that fit (see `random_placement`).
    """
    geom = copy.deepcopy(geometry)
    groups = associate_blocks_to_dies(geom)
    by_name = {b.name: b for b in geom.power_blocks}
    prints = {d.name: d for d in (getattr(geom, 'die_footprints', None) or [])}

    for key, (dx, dy) in offsets.items():
        if dx == 0.0 and dy == 0.0:
            continue
        # A key may name a die footprint (move the die and its blocks), a single power
        # block (move that block alone -- the independent-block mode, which gives the
        # layout 2 degrees of freedom PER BLOCK instead of 2 for the whole group), or ''
        # (move every block with no footprint, as a rigid group).
        if key in by_name and key not in prints:
            b = by_name[key]
            b.x += dx
            b.y += dy
            continue
        if key and key not in prints:
            raise ValueError(f'unknown die footprint or block {key!r}; '
                             f'have dies {sorted(prints)}')
        if key:
            dp = prints[key]
            dp.x += dx
            dp.y += dy
        for bname in groups.get(key, []):
            b = by_name[bname]
            b.x += dx
            b.y += dy

    # Everything must still lie inside the package footprint.
    W, H = geom.die_width, geom.die_length
    movable = list(geom.power_blocks) + list(prints.values())
    for item in movable:
        if item.x < -1e-6 or item.y < -1e-6 or item.x + item.width > W + 1e-6 \
                or item.y + item.height > H + 1e-6:
            raise ValueError(
                f'{type(item).__name__} {item.name!r} leaves the footprint after placement: '
                f'x=[{item.x:.0f},{item.x + item.width:.0f}] '
                f'y=[{item.y:.0f},{item.y + item.height:.0f}] vs package {W:.0f}x{H:.0f}')

    geom.validate()
    return geom


def _overlaps(a, b, margin: float) -> bool:
    return not (a.x + a.width + margin <= b.x or b.x + b.width + margin <= a.x or
                a.y + a.height + margin <= b.y or b.y + b.height + margin <= a.y)


def lateral_chiplets(geometry) -> List[Tuple[Tuple[float, float, float, float], List[str]]]:
    """
    Group DiePrints that occupy the SAME lateral rectangle on different layers.

    A stacked chiplet (geometry6's `hbm1_die1` / `hbm1_die2`, geometry5's TSV stack) is one
    physical object appearing once per layer it spans. Those footprints must translate
    together: moving them independently would shear a die stack apart, and treating them as
    separate objects makes them look mutually overlapping, which made rejection sampling fail
    outright on geometry6.

    Returns [((x, y, w, h), [dieprint names])], one entry per physical chiplet.
    """
    groups: Dict[Tuple[float, float, float, float], List[str]] = {}
    for dp in (getattr(geometry, 'die_footprints', None) or []):
        groups.setdefault((dp.x, dp.y, dp.width, dp.height), []).append(dp.name)
    return list(groups.items())


def random_block_placement(geometry, rng, margin_um: float = 500.0,
                           max_tries: int = 2000) -> Dict[str, Offset]:
    """
    Place every power block INDEPENDENTLY anywhere in the footprint, not as a rigid group.

    Added 2026-09-11 after the first moving-source pilot failed to make the benchmark
    discriminative (docs/report.md §9.15). Translating a fixed arrangement rigidly gives the
    layout only 2 degrees of freedom per chiplet, and because the temperature field is
    laterally smooth, a linear model handed the displacement as a feature covers that with a
    first-order response — ridge's error relative to the signal barely moved even at a
    source-overlap IoU of 0.472, which matches IC-ThermBench's 0.449.

    IC-ThermBench's layouts come from a placement optimiser: genuinely reconfigurable
    arrangements, not translations of one arrangement. This mode reproduces that property by
    sampling each block's position independently (2 DOF per block, so 8 for geometry1's four
    blocks), which changes the *relative* geometry of the sources and not just their common
    offset.
    """
    blocks = geometry.power_blocks
    W, H = geometry.die_width, geometry.die_length
    for _ in range(max_tries):
        placed, offsets, ok = [], {}, True
        for b in blocks:
            hi_x, hi_y = W - b.width, H - b.height
            if hi_x < 0 or hi_y < 0:
                return {}
            nx = float(rng.uniform(0.0, hi_x))
            ny = float(rng.uniform(0.0, hi_y))
            cand = replace(b, x=nx, y=ny)
            if any(_overlaps(cand, p, margin_um) for p in placed):
                ok = False
                break
            placed.append(cand)
            offsets[b.name] = (nx - b.x, ny - b.y)
        if ok and len(offsets) == len(blocks):
            return offsets
    log.warning('random_block_placement: no valid layout for %s in %d tries; '
                'falling back to nominal', geometry.name, max_tries)
    return {}


def random_placement(geometry, rng, max_shift_um: float = 2000.0,
                     margin_um: float = 250.0, max_tries: int = 200
                     ) -> Dict[str, Offset]:
    """
    Sample per-chiplet offsets that keep everything inside the package and non-overlapping.

    `max_shift_um` bounds the translation per axis. `margin_um` is the minimum clearance kept
    between chiplets (one mesh cell at geometry4's 250 µm pitch), so moved chiplets never
    touch — a benchmark where dies sometimes abut and sometimes do not would confound
    placement with contact.

    Rejection sampling; returns all-zero offsets if no valid configuration is found within `max_tries`,
    so a caller always gets a usable (if unmoved) geometry rather than an exception.
    """
    prints = list(getattr(geometry, 'die_footprints', None) or [])
    W, H = geometry.die_width, geometry.die_length

    if not prints:
        # Laterally homogeneous geometry: move the power blocks themselves, as a rigid group
        # (moving them independently would change their relative layout, which is a different
        # and much larger perturbation than translating the source pattern).
        x0, y0, x1, y1 = _bounds(geometry.power_blocks)
        lo_x, hi_x = -x0, W - x1
        lo_y, hi_y = -y0, H - y1
        dx = float(rng.uniform(max(lo_x, -max_shift_um), min(hi_x, max_shift_um)))
        dy = float(rng.uniform(max(lo_y, -max_shift_um), min(hi_y, max_shift_um)))
        return {'': (dx, dy)}

    # One physical chiplet per lateral rectangle, however many layers it spans.
    chiplets = lateral_chiplets(geometry)
    rect_of = {names[0]: rect for rect, names in chiplets}

    for _ in range(max_tries):
        offsets: Dict[str, Offset] = {}
        placed = []
        ok = True
        # Place the largest chiplets first: they have the least slack, so committing them
        # early makes rejection sampling converge instead of repeatedly dead-ending on the
        # last big die (geometry6 failed 200/200 tries without this).
        order = sorted(chiplets, key=lambda c: -c[0][2] * c[0][3])
        for (x, y, w, h), names in order:
            lo_x, hi_x = -x, W - (x + w)
            lo_y, hi_y = -y, H - (y + h)
            if hi_x < lo_x or hi_y < lo_y:
                ok = False
                break
            dx = float(rng.uniform(max(lo_x, -max_shift_um), min(hi_x, max_shift_um)))
            dy = float(rng.uniform(max(lo_y, -max_shift_um), min(hi_y, max_shift_um)))
            moved = replace(prints[0], name='probe', x=x + dx, y=y + dy, width=w, height=h)
            if any(_overlaps(moved, p, margin_um) for p in placed):
                ok = False
                break
            placed.append(moved)
            for n in names:                       # every layer of this stack moves together
                offsets[n] = (dx, dy)
        if ok and len(offsets) == len(prints):
            return offsets

    log.warning('random_placement: no valid configuration for %s in %d tries; '
                'falling back to the nominal placement', geometry.name, max_tries)
    return {dp.name: (0.0, 0.0) for dp in prints}


def placement_summary(offsets: Optional[Dict[str, Offset]]) -> str:
    if not offsets:
        return 'nominal'
    return ', '.join(f'{k or "blocks"}=({dx:+.0f},{dy:+.0f})µm'
                     for k, (dx, dy) in sorted(offsets.items()))
