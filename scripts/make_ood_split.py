"""
Build out-of-distribution (OOD) evaluation splits from scenario metadata.

Why this exists
---------------
The shipped `*_test_*.npz` files are interpolation points inside the same
parameter sweep as training: same power patterns, HTC and ambient values drawn
from the middle of the training ranges. They measure interpolation, not
generalisation, and every model looks good on them.

These splits hold out a whole *axis* instead, and write a manifest rather than
copying NPZ files (the data is large; the split is just a list of names).

Axes
----
pattern    Train on the smooth patterns (uniform, checkerboard, gradient);
           test on the peaked ones (hotspot, dual_hotspot, extreme_hotspot).
           Tests whether a model extrapolates to unseen spatial structure.

power      Train on the low-power half, test on the high-power half. This is
           the axis that can actually break a linear model: steady-state
           conduction is linear in power only while k(T) is constant, so
           nonlinearity appears exactly where self-heating is largest.

htc        Train on mid-range cooling, test on the extremes. Temperature depends
           on 1/h, so this probes extrapolation in a reciprocal coordinate.

ambient    Train on low ambient, test on high ambient. Expected to be easy --
           ambient enters as a near-exact additive offset -- and included as a
           control: a split that everything passes.

Usage
-----
    python scripts/make_ood_split.py --geometry geometry1 --axis power
    python scripts/baselines.py --geometry geometry1 \\
        --split-manifest results/ood_geometry1_power.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
_log = logging.getLogger('ood_split')

PEAKED_PATTERNS = {'hotspot', 'dual_hotspot', 'extreme_hotspot',
                   'split_chiplet_a_hot', 'split_chiplet_b_hot'}


def scenario_meta(path: Path) -> dict:
    return np.load(path, allow_pickle=True)['metadata'].item()


def total_power(meta: dict) -> float:
    return sum(v for k, v in meta.items() if k.startswith('block_power_'))


def split_by_axis(files: List[Path], metas: List[dict], axis: str
                  ) -> Tuple[List[Path], List[Path], str]:
    """Return (train_files, test_files, human-readable rule)."""
    if axis == 'pattern':
        held = [f for f, m in zip(files, metas) if m.get('pattern') in PEAKED_PATTERNS]
        kept = [f for f, m in zip(files, metas) if m.get('pattern') not in PEAKED_PATTERNS]
        return kept, held, f"held out patterns {sorted(PEAKED_PATTERNS)}"

    if axis == 'power':
        vals = np.array([total_power(m) for m in metas])
        thr = float(np.median(vals))
        kept = [f for f, v in zip(files, vals) if v <= thr]
        held = [f for f, v in zip(files, vals) if v > thr]
        return kept, held, f"held out total block power > {thr:.2f} W/cm2 (median split)"

    if axis == 'htc':
        vals = np.array([float(m.get('htc', 0.0)) for m in metas])
        lo, hi = np.percentile(vals, [25, 75])
        kept = [f for f, v in zip(files, vals) if lo <= v <= hi]
        held = [f for f, v in zip(files, vals) if v < lo or v > hi]
        return kept, held, f"held out HTC outside [{lo:.0f}, {hi:.0f}] W/m2K"

    if axis == 'ambient':
        vals = np.array([float(m.get('t_ambient_kelvin', 298.15)) for m in metas])
        thr = float(np.median(vals))
        kept = [f for f, v in zip(files, vals) if v <= thr]
        held = [f for f, v in zip(files, vals) if v > thr]
        return kept, held, f"held out ambient > {thr - 273.15:.1f} C (median split)"

    raise ValueError(f"Unknown axis: {axis}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--geometry', required=True)
    p.add_argument('--data', type=Path, default=Path('data/3d-ice'))
    p.add_argument('--axis', required=True,
                   choices=['pattern', 'power', 'htc', 'ambient'])
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()

    geom_dir = args.data / args.geometry
    root = geom_dir if geom_dir.is_dir() else args.data

    # Only *_train_* files: the shipped test files are interpolation points and
    # would contaminate an extrapolation split.
    files = sorted(root.glob(f'{args.geometry}_train_*.npz'))
    if not files:
        _log.error("No training files for %s under %s", args.geometry, root)
        sys.exit(1)

    metas = [scenario_meta(f) for f in files]
    train, test, rule = split_by_axis(files, metas, args.axis)

    if not train or not test:
        _log.error("Degenerate split on axis '%s' (train=%d, test=%d)",
                   args.axis, len(train), len(test))
        sys.exit(1)

    out = args.output or Path('results') / f'ood_{args.geometry}_{args.axis}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'geometry': args.geometry,
        'axis': args.axis,
        'rule': rule,
        'n_train': len(train),
        'n_test': len(test),
        'train': [f.name for f in train],
        'test': [f.name for f in test],
    }, indent=2), encoding='utf-8')

    _log.info("%s / %s: %d train, %d test -- %s",
              args.geometry, args.axis, len(train), len(test), rule)
    _log.info("Wrote %s", out)


if __name__ == '__main__':
    main()
