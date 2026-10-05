"""Lint 3D-ICE thermal datasets for the silent bugs listed in src/validation/lint.py.
Usage: python scripts/lint_dataset.py data/3d-ice-layout-geometry5 [more dirs or .npz files]
                                      [--json report.json] [--no-energy] [--max-files N] [--show N] [--strict]
Exit status: 1 if any error was found (with --strict, also on warnings), else 0.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.validation.lint import lint_dataset  # noqa: E402

TITLES = {'L001': 'arrays', 'L002': 'tensor grid', 'L003': 'orientation', 'L004': 'temperature',
          'L005': 'maximum principle', 'L006': 'off-grid footprint', 'L007': 'metadata', 'L008': 'energy balance',
          'L009': 'k override', 'L010': 'duplicate field', 'L011': 'archive directory'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--no-energy', action='store_true', help='skip the energy balance (faster)')
    ap.add_argument('--max-files', type=int)
    ap.add_argument('--show', type=int, default=5, help='example findings to print per rule')
    ap.add_argument('--strict', action='store_true', help='warnings also fail')
    a = ap.parse_args(argv)
    missing = [p for p in a.paths if not Path(p).exists()]
    if missing:
        ap.error(f'no such path: {missing}')
    rep = lint_dataset(a.paths, energy=not a.no_energy, max_files=a.max_files)
    c = rep['counts']
    print(f"{rep['files']} file(s): {c['error']} error(s) in {rep['files_with_errors']} file(s), "
          f"{c['warning']} warning(s), {c['info']} note(s)")
    for code, info in rep['by_code'].items():
        print(f"\n{code} {TITLES.get(code, '')} [{info['severity']}] in {info['files']} file(s)")
        shown = [f for f in rep['findings'] if f['code'] == code][:a.show]
        for f in shown:
            print(f"  {Path(f['file']).name}: {f['message']}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    failed = c['error'] > 0 or (a.strict and c['warning'] > 0)
    print('\nFAILED' if failed else '\nOK')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
