"""
Validate a generated NPZ dataset for shape, key, and physical correctness.

Usage:
    python scripts/validate_dataset.py data/mock
    python scripts/validate_dataset.py data/3d-ice --strict
"""

import sys
import argparse
import numpy as np
from pathlib import Path


EXPECTED_KEYS = {'coords', 'temp', 'power', 'layer', 'metadata'}

# Physical bounds (Kelvin) -- mock data may exceed the strict range
TEMP_MIN_K = 270.0      # below ambient = problem
TEMP_MAX_K_STRICT = 600.0   # physically unlikely for steady-state chips
TEMP_MAX_K_LOOSE  = 10000.0 # catches Inf/extreme outliers only


def validate_file(npz_path: Path, strict: bool = False) -> list[str]:
    """Return list of error strings (empty = OK)."""
    errors = []

    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return [f"LOAD FAILED: {e}"]

    # Key presence
    missing = EXPECTED_KEYS - set(d.files)
    if missing:
        errors.append(f"missing keys: {missing}")

    if 'coords' not in d or 'temp' not in d:
        return errors  # cannot continue without these

    coords = d['coords']
    temp   = d['temp']

    # Shape consistency
    if coords.ndim != 2 or coords.shape[1] != 3:
        errors.append(f"coords shape {coords.shape} (expected (N,3))")
    if temp.ndim != 1:
        errors.append(f"temp shape {temp.shape} (expected (N,))")
    if coords.shape[0] != temp.shape[0]:
        errors.append(f"coords/temp length mismatch: {coords.shape[0]} vs {temp.shape[0]}")

    if 'power' in d and d['power'].shape != temp.shape:
        errors.append(f"power/temp length mismatch")
    if 'layer' in d and d['layer'].shape != temp.shape:
        errors.append(f"layer/temp length mismatch")

    # NaN / Inf
    if np.any(np.isnan(temp)):
        errors.append(f"NaN values in temp ({np.sum(np.isnan(temp))} points)")
    if np.any(np.isinf(temp)):
        errors.append(f"Inf values in temp ({np.sum(np.isinf(temp))} points)")

    # Physical range
    t_min, t_max = float(np.nanmin(temp)), float(np.nanmax(temp))
    if t_min < TEMP_MIN_K:
        errors.append(f"temp min {t_min:.1f} K below physical floor {TEMP_MIN_K} K")
    limit = TEMP_MAX_K_STRICT if strict else TEMP_MAX_K_LOOSE
    if t_max > limit:
        level = "ERROR" if strict else "WARN"
        errors.append(f"[{level}] temp max {t_max:.1f} K exceeds {'strict' if strict else 'loose'} limit {limit:.0f} K")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate NPZ thermal dataset")
    parser.add_argument('data_dir', type=Path, help="Directory to scan for .npz files")
    parser.add_argument('--strict', action='store_true',
                        help="Apply strict temperature range check (600 K max)")
    args = parser.parse_args()

    npz_files = sorted(args.data_dir.rglob('*.npz'))
    if not npz_files:
        print(f"No .npz files found under {args.data_dir}")
        sys.exit(1)

    print(f"Validating {len(npz_files)} files in {args.data_dir}")
    print(f"Mode: {'strict' if args.strict else 'loose'} temperature bounds")
    print()

    total_errors = 0
    total_warnings = 0
    files_with_errors = []

    for f in npz_files:
        issues = validate_file(f, strict=args.strict)
        errors = [i for i in issues if not i.startswith('[WARN')]
        warnings = [i for i in issues if i.startswith('[WARN')]

        if errors:
            print(f"  FAIL  {f.relative_to(args.data_dir)}")
            for e in errors:
                print(f"         {e}")
            total_errors += len(errors)
            files_with_errors.append(f)
        elif warnings:
            print(f"  WARN  {f.relative_to(args.data_dir)}")
            for w in warnings:
                print(f"         {w}")
            total_warnings += len(warnings)
        else:
            d = np.load(f, allow_pickle=True)
            t = d['temp']
            print(f"  OK    {f.relative_to(args.data_dir)}  "
                  f"N={t.shape[0]:>7,}  T={np.min(t)-273.15:.0f}..{np.max(t)-273.15:.0f} C")

    print()
    print("=" * 60)
    if total_errors == 0 and total_warnings == 0:
        print(f"ALL {len(npz_files)} FILES PASS")
    else:
        print(f"Summary: {len(npz_files)} files  |  "
              f"{total_errors} errors  |  {total_warnings} warnings")
    if files_with_errors:
        print(f"Files with errors ({len(files_with_errors)}):")
        for f in files_with_errors:
            print(f"  {f}")
    print("=" * 60)

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == '__main__':
    main()
