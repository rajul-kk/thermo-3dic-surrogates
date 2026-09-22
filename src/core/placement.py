"""Per-scenario chiplet placement — the fix for the defect diagnosed in docs/report.md §9.14."""
from __future__ import annotations

import copy
import logging
import numpy as np
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

Offset = Tuple[float, float]


def associate_blocks_to_dies(geometry) -> Dict[str, List[str]]:
    """Map each DiePrint name -> names of the PowerBlocks lying inside it."""
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
    """Return a copy of `geometry` with each named chiplet translated by its (dx, dy) in µm."""
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
    """Group DiePrints that occupy the SAME lateral rectangle on different layers."""
    groups: Dict[Tuple[float, float, float, float], List[str]] = {}
    for dp in (getattr(geometry, 'die_footprints', None) or []):
        groups.setdefault((dp.x, dp.y, dp.width, dp.height), []).append(dp.name)
    return list(groups.items())


def random_block_placement(geometry, rng, margin_um: float = 500.0,
                           max_tries: int = 2000, grid_um: float = 100.0
                           ) -> Dict[str, Offset]:
    """Place every power block INDEPENDENTLY anywhere in the footprint, not as a rigid group.

    Largest-first ordering alone (the fix that saved random_placement's chiplet packing on
    geometry6) does NOT transfer here -- tested and found to still fail 0/45 at geometry2a's
    51.6% block occupancy (docs/compute.md, 2026-09-22). The reason is structural, not just
    ordering: pure i.i.d. rejection sampling has no memory across attempts, so ANY single
    block failing discards all others placed so far and restarts from zero. Fixed with a
    randomised first-fit search instead: each block gets a shuffled grid of candidate anchor
    points (not one random draw) and takes the first non-overlapping one, so a hard block
    gets thousands of tries within a single overall attempt rather than one.
    """
    blocks = sorted(geometry.power_blocks, key=lambda b: -b.width * b.height)
    W, H = geometry.die_width, geometry.die_length
    for _ in range(max_tries):
        placed, offsets, ok = [], {}, True
        for b in blocks:
            hi_x, hi_y = W - b.width, H - b.height
            if hi_x < 0 or hi_y < 0:
                return {}
            xs = np.arange(0.0, hi_x + grid_um, grid_um)
            ys = np.arange(0.0, hi_y + grid_um, grid_um)
            candidates = [(x, y) for x in xs for y in ys]
            rng.shuffle(candidates)
            found = None
            for nx, ny in candidates:
                cand = replace(b, x=float(nx), y=float(ny))
                if not any(_overlaps(cand, p, margin_um) for p in placed):
                    found = cand
                    break
            if found is None:
                ok = False
                break
            placed.append(found)
            offsets[b.name] = (found.x - b.x, found.y - b.y)
        if ok and len(offsets) == len(blocks):
            return offsets
    log.warning('random_block_placement: no valid layout for %s in %d tries; '
                'falling back to nominal', geometry.name, max_tries)
    return {}


def free_chiplet_placement(geometry, rng, margin_um: float = 250.0,
                           max_tries: int = 4000) -> Dict[str, Offset]:
    """Place each chiplet anywhere it fits, not within a bounded shift -- the 2.5D analogue of random_block_placement."""
    # The 2.5D counterpart of random_block_placement. random_placement bounds each chiplet
    # to +/- max_shift_um of its nominal spot, which keeps the arrangement recognisable and
    # leaves the layout a low-dimensional family -- docs/report.md 9.15 showed that is not
    # enough to stop a linear model. Here each chiplet is placed uniformly at random
    # anywhere it fits, so the relative arrangement genuinely changes.
    chiplets = lateral_chiplets(geometry)
    if not chiplets:
        return {}
    W, H = geometry.die_width, geometry.die_length
    from dataclasses import replace as _replace
    proto = (getattr(geometry, 'die_footprints', None) or [None])[0]

    for _ in range(max_tries):
        placed, offsets, ok = [], {}, True
        # Largest first: they have the least slack, so committing them early keeps
        # rejection sampling from dead-ending on the last big die.
        for (x, y, w, h), names in sorted(chiplets, key=lambda c: -c[0][2] * c[0][3]):
            hi_x, hi_y = W - w, H - h
            if hi_x < 0 or hi_y < 0:
                return {}
            nx = float(rng.uniform(0.0, hi_x))
            ny = float(rng.uniform(0.0, hi_y))
            cand = _replace(proto, name='probe', x=nx, y=ny, width=w, height=h)
            if any(_overlaps(cand, p, margin_um) for p in placed):
                ok = False
                break
            placed.append(cand)
            for n in names:
                offsets[n] = (nx - x, ny - y)
        if ok and len(offsets) == len(getattr(geometry, 'die_footprints', []) or []):
            return offsets
    log.warning('free_chiplet_placement: no valid layout for %s in %d tries; '
                'falling back to nominal', geometry.name, max_tries)
    return {}


def shelf_chiplet_placement(geometry, rng, margin_um: float = 250.0) -> Dict[str, Offset]:
    """Permute chiplets along x and redistribute the slack -- layout variation for densely packed packages."""
    # free_chiplet_placement cannot serve geometry6 (7 chiplets filling ~69% of the package)
    # or geometry7: uniform random placement essentially always overlaps, so rejection
    # sampling fails outright. Dense 2.5D packages are effectively a 1D packing problem, and
    # what a floorplanner actually varies there is the ORDER of the dies and the gaps between
    # them. That is still genuinely high-dimensional -- n! orderings plus n continuous gaps
    # plus n vertical offsets -- and unlike a bounded translation it changes which die
    # neighbours which, so the thermal coupling structure changes rather than shifting.
    chiplets = lateral_chiplets(geometry)
    if not chiplets:
        return {}
    W, H = geometry.die_width, geometry.die_length

    order = list(rng.permutation(len(chiplets)))
    widths = [chiplets[i][0][2] for i in order]
    total_w = sum(widths)
    slack = W - total_w - margin_um * (len(order) - 1)
    if slack < 0:
        return {}
    # Dirichlet split of the leftover width across the n+1 gaps (before, between, after).
    cuts = np.sort(rng.uniform(0.0, 1.0, size=len(order)))
    frac = np.diff(np.concatenate([[0.0], cuts, [1.0]]))
    gaps = frac * slack

    offsets: Dict[str, Offset] = {}
    cursor = gaps[0]
    for gi, idx in enumerate(order):
        (x, y, w, h), names = chiplets[idx]
        hi_y = H - h
        ny = float(rng.uniform(0.0, hi_y)) if hi_y > 0 else 0.0
        for n in names:
            offsets[n] = (cursor - x, ny - y)
        cursor += w + margin_um + gaps[gi + 1]
    return offsets


def random_placement(geometry, rng, max_shift_um: float = 2000.0,
                     margin_um: float = 250.0, max_tries: int = 200
                     ) -> Dict[str, Offset]:
    """Sample per-chiplet offsets that keep everything inside the package and non-overlapping."""
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
