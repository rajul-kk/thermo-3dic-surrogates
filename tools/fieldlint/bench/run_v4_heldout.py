"""Round 4 (v4) held-out measurement of fieldlint: see protocol_v4.md.

This is the ONLY code path that may produce v4 held-out results. It re-uses the v2 machinery of run_bench.py unchanged (fault injection,
summarise(), the hash check) with the FROZEN V4 file (bench/results/frozen_thresholds_v4.json, commit bf27d68), and adds only (a) the v4
dataset list, (b) a run-once guard, (c) provenance fields in every result file.

  python tools/fieldlint/bench/run_v4_heldout.py --list                      # print the registered runs, run nothing
  python tools/fieldlint/bench/run_v4_heldout.py --run <name>                # one registered run; refuses if its result file exists
  python tools/fieldlint/bench/run_v4_heldout.py --run-all inscope|diagnostic|control

Result file per run: bench/results/v4_heldout/<name>.json. A run whose result file exists is NEVER repeated (protocol_v4.md section 8);
a crashed run that wrote no file may be re-launched, and the re-launch is logged in protocol_v4.md section 12.

Mapping configs are the staged presets in bench/presets_v4_staging/ (NOT in src/, so that the v4 hash is untouched); they are passed as
file paths, as fno_darcy_native.json was in v2. No config sets `operator:`; detection is automatic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import run_bench as rb                                           # same directory; the v2 machinery, unchanged

HERE = Path(__file__).resolve().parent
OUT = rb.RESULTS / 'v4_heldout'
EXT = rb.DATA / '_external_v4'
PRE = HERE / 'presets_v4_staging'
FROZEN = 'frozen_thresholds_v4.json'


def P(name: str) -> str:
    return str(PRE / f'{name}.json')


CD = EXT / 'condiff/converted'
DA = EXT / 'pdebench_darcy/converted'
# (name, role, path, preset path, extras).  `extras` as in run_bench (units, cfg, head, time_dependent).
INSCOPE = [
    ('condiff_poisson', 'inscope', CD / 'poisson_grid128_test.npz', P('v4_condiff_poisson'), {'units': ''}),
    ('condiff_gaussian0.1', 'inscope', CD / 'gaussian0.1_grid128_test.npz', P('v4_condiff'), {'units': ''}),
    ('condiff_gaussian1.0', 'inscope', CD / 'gaussian1.0_grid128_test.npz', P('v4_condiff'), {'units': ''}),
    ('condiff_cubic0.4', 'inscope', CD / 'cubic0.4_grid128_test.npz', P('v4_condiff'), {'units': ''}),
    ('condiff_exponential2.0', 'inscope', CD / 'exponential2.0_grid128_test.npz', P('v4_condiff'), {'units': ''}),
    ('elado_b1_test', 'inscope', EXT / 'elado/converted/B1_test.h5', P('v4_elado_poisson'), {'units': ''}),
    ('pdebench_darcy_beta0.1', 'inscope', DA / 'darcy_beta0.1_n1000.npz', P('v4_pdebench_darcy'), {'units': ''}),
    ('pdebench_darcy_beta10.0', 'inscope', DA / 'darcy_beta10.0_n1000.npz', P('v4_pdebench_darcy'), {'units': ''}),
    ('pdebench_darcy_beta100.0', 'inscope', DA / 'darcy_beta100.0_n1000.npz', P('v4_pdebench_darcy'), {'units': ''}),
]
DIAGNOSTIC = [
    ('condiff_gaussian0.1_nok', 'diagnostic', CD / 'gaussian0.1_grid128_test.npz', P('v4_condiff_nok'), {'units': ''}),
]
CONTROL = [
    ('pdegym_helmholtz', 'control', EXT / 'helmholtz_pdegym/converted/helmholtz_first1000.npz', P('v4_pdegym_helmholtz'), {'units': ''}),
    ('elado_helmholtz_c1', 'control', EXT / 'elado/converted/C1_test.h5', P('v4_elado_helmholtz'), {'units': ''}),
    ('pdegym_wave_layer', 'control', EXT / 'wave_layer/converted/wave_layer_first256.npz', P('v4_pdegym_wave_layer'),
     {'units': '', 'time_dependent': True}),
]
ALL = INSCOPE + DIAGNOSTIC + CONTROL


def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.run(['git', *a], capture_output=True, text=True, cwd=HERE).stdout.strip()
        except OSError:
            return ''
    fr = rb.RESULTS / FROZEN
    try:
        import h5py
        h5v = h5py.__version__
    except ImportError:
        h5v = ''
    return {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'git_head': git('rev-parse', 'HEAD'),
            'git_dirty_tracked': git('status', '--porcelain', '--untracked-files=no'),
            'frozen_file': fr.name, 'frozen_file_sha256': hashlib.sha256(fr.read_bytes()).hexdigest(),
            'code_sha256_now': rb.code_hash(),
            'staged_presets_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(PRE.glob('*.json'))},
            'python': platform.python_version(), 'numpy': np.__version__, 'h5py': h5v, 'argv': sys.argv}


def run_one(spec, log):
    name, role = spec[0], spec[1]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.json'
    if path.exists():
        sys.exit(f'refusing: {path} exists. v4 held-out runs are run once (protocol_v4.md section 8).')
    opt, _fr = rb.load_frozen('_v4')                              # exits if any frozen file changed
    prov = provenance()
    if role in ('inscope', 'diagnostic'):
        recs = [rb.run_dataset(spec, opt, rb.FAULTS, log)]
    else:                                                          # control: clean only, with and without the scope declaration
        recs = []
        variants = [spec]
        if spec[4].get('time_dependent'):
            variants = [(spec[0] + '+time_dependent_flag', *spec[1:4], {**spec[4], 'cfg': {**spec[4].get('cfg', {}), 'time_dependent': True}}), spec]
        for s in variants:
            base = rb.with_splits(rb.build_clean(s))
            recs.append({'name': s[0], 'group': 'control', 'n': base.n, 'units': base.units, 'faults': {}, 'clean': rb.summarise(base, opt)})
            base.close()
            log(f'  {s[0]} done')
    path.write_text(json.dumps({'provenance': prov, 'role': role, 'records': recs}, indent=1, default=float))
    log(f'wrote {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--list', action='store_true')
    g.add_argument('--run')
    g.add_argument('--run-all', choices=['inscope', 'diagnostic', 'control'])
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)                           # noqa: E731
    if a.list:
        for s in ALL:
            print(f'{s[1]:11s} {s[0]:28s} {Path(s[3]).stem:24s} {s[2]}  {"[exists]" if s[2].exists() else "[MISSING]"}  '
                  f'{"[result exists]" if (OUT / (s[0] + ".json")).exists() else ""}')
        return 0
    specs = [s for s in ALL if (s[0] == a.run if a.run else s[1] == a.run_all)]
    if not specs:
        sys.exit('no such registered run')
    for s in specs:
        run_one(s, log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
