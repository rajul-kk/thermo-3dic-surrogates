"""Compare two iceforge_realdata output directories metric by metric.
Usage: python scripts/compare_realdata_runs.py OLD_DIR NEW_DIR
Compares every cases/*.json (numbers: abs diff; everything else: equality, verdicts included),
the *_fields.npz arrays (max abs diff) and summary.json rows. Keys ignored: seconds, ice_commit, ice_executable.
"""
import json
import sys
from pathlib import Path

import numpy as np

IGNORE = {'seconds', 'ice_commit', 'ice_executable'}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append((path + '/' + k, 'missing', None))
            else:
                walk(a[k], b[k], path + '/' + k, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, 'length', None))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}[{i}]', out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        out.append((path, 'num', abs(float(a) - float(b))))
    else:
        out.append((path, 'eq' if a == b else 'DIFF', None))


def main():
    old, new = Path(sys.argv[1]), Path(sys.argv[2])
    names = lambda d: {p.name for p in (d / 'cases').glob('*.json')}
    only = names(old) ^ names(new)
    if only:
        print('files present in only one dir:', sorted(only))
    max_num, max_where, nonnum, verdict_chg, nnum = 0.0, '', [], [], 0
    for n in sorted(names(old) & names(new)):
        res = []
        walk(json.loads((old / 'cases' / n).read_text()), json.loads((new / 'cases' / n).read_text()), n, res)
        for path, kind, d in res:
            if kind == 'num':
                nnum += 1
                if d > max_num:
                    max_num, max_where = d, path
            elif kind != 'eq':
                nonnum.append((path, kind))
                if 'verdict' in path:
                    verdict_chg.append(path)
    if (old / 'summary.json').exists() and (new / 'summary.json').exists():
        res = []
        walk(json.loads((old / 'summary.json').read_text()), json.loads((new / 'summary.json').read_text()), 'summary.json', res)
        for path, kind, d in res:
            if kind == 'num':
                nnum += 1
                if d > max_num:
                    max_num, max_where = d, path
            elif kind != 'eq':
                nonnum.append((path, kind))
                if 'verdict' in path:
                    verdict_chg.append(path)
    fmax, fwhere, nf, nfshape = 0.0, '', 0, []
    fn = lambda d: {p.name for p in (d / 'cases').glob('*_fields.npz')}
    for n in sorted(fn(old) & fn(new)):
        A, B = np.load(old / 'cases' / n), np.load(new / 'cases' / n)
        for k in sorted(set(A.files) | set(B.files)):
            if k not in A.files or k not in B.files or A[k].shape != B[k].shape:
                nfshape.append((n, k))
                continue
            nf += 1
            d = float(np.max(np.abs(A[k].astype(float) - B[k].astype(float)))) if A[k].size else 0.0
            if d > fmax:
                fmax, fwhere = d, f'{n}:{k}'
    print(f'numeric values compared (json): {nnum}; max abs diff {max_num:.3e} at {max_where or "-"}')
    print(f'field arrays compared: {nf}; max abs diff {fmax:.3e} at {fwhere or "-"}; shape/key mismatches: {len(nfshape)}')
    print(f'non-numeric differences (verdicts, flags, missing keys): {len(nonnum)}; verdict changes: {len(verdict_chg)}')
    for x in nonnum[:30]:
        print('  ', x)
    ok = not only and not nonnum and not nfshape and max_num == 0 and fmax == 0
    print('IDENTICAL' if ok else 'DIFFERENCES FOUND')


if __name__ == '__main__':
    main()
