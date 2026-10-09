"""Round 3 (v3) held-out measurement of fieldlint. DRAFT, not yet frozen: see protocol_v3.md.

This is the ONLY code path that may produce v3 results. It re-uses the v2 machinery of run_bench.py unchanged (fault injection,
summarise(), the hash check against bench/results/frozen_thresholds_v2.json) and adds only (a) the v3 dataset list, (b) a run-once
guard, (c) provenance fields in every result file.

  python tools/fieldlint/bench/run_v3.py --list                       # print the registered runs, run nothing
  python tools/fieldlint/bench/run_v3.py --run <name>                 # one registered run; refuses if its result file exists
  python tools/fieldlint/bench/run_v3.py --run-all <role>             # role: inscope | diagnostic | control

Result file per run: bench/results/v3/<name>.json. A run whose result file exists is NEVER repeated (protocol_v3.md section 8);
a crashed run that wrote no file may be re-launched, and the re-launch is logged in the file as `relaunch_of_crash`.

Roles
  inscope     counted in the pass/fail criteria. Clean run + the full v2 fault set.
  diagnostic  as-released variants whose documented transformation (standardisation) violates a rule premise by construction;
              clean run + the full fault set, reported, NOT counted in the criteria.
  control     out-of-scope data: clean run only, once with the scope declaration the README prescribes (time_dependent for
              time-dependent data) and once without it.
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
OUT = rb.RESULTS / 'v3'
EXT = rb.DATA / '_external'

# (name, role, path, preset, extras).  `extras` as in run_bench (units, cfg, head).  Preset files live in src/fieldlint/presets/v3_*.json.
C = EXT / 'cno_rpb/converted'
INSCOPE = [
    ('pdegym_poisson_gauss', 'inscope', EXT / 'poisson_gauss/converted/poisson_gauss_first1000.npz', 'v3_poisson_gauss', {'units': ''}),
    ('cno_poisson', 'inscope', C / 'PoissonData_64x64_IN_first1000.npz', 'v3_cno_poisson', {'units': ''}),
    ('cno_darcy', 'inscope', C / 'Darcy_64x64_IN_first512.npz', 'v3_cno_darcy', {'units': ''}),
    ('elado_a1_test', 'inscope', EXT / 'elado/converted/A1_test.h5', 'v3_elado_poisson', {'units': ''}),
    ('dpde_poisson_denorm', 'inscope', EXT / 'diffusionpde_norm/converted/poisson_hf_first1000.npz', 'v3_dpde_poisson_denorm', {'units': ''}),
    ('dpde_darcy_denorm', 'inscope', EXT / 'diffusionpde_norm/converted/darcy_hf_first1000.npz', 'v3_dpde_darcy_denorm', {'units': ''}),
    ('ucsd_poisson', 'inscope', EXT / 'ucsd_pde/converted/Poisson4.npz', 'v3_ucsd_poisson', {'units': '', 'head': 1000}),
]
DIAGNOSTIC = [
    ('dpde_poisson_asreleased', 'diagnostic', EXT / 'diffusionpde_norm/converted/poisson_hf_first1000.npz', 'v3_dpde_poisson', {'units': ''}),
    ('dpde_darcy_asreleased', 'diagnostic', EXT / 'diffusionpde_norm/converted/darcy_hf_first1000.npz', 'v3_dpde_darcy', {'units': ''}),
]
CONTROL = [
    ('cno_wave', 'control', C / 'WaveData_64x64_IN_t_5_first1000.npz', 'v3_cno_wave', {'units': '', 'time_dependent': True}),
    ('cno_allen_cahn', 'control', C / 'AllenCahn_64x64_IN_first512.npz', 'v3_cno_allen_cahn', {'units': '', 'time_dependent': True}),
    ('cno_navier_stokes', 'control', C / 'NavierStokes_64x64_IN_first1000.npz', 'v3_cno_navier_stokes', {'units': '', 'time_dependent': True}),
    ('cno_cont_transport', 'control', C / 'ContTranslation_64x64_IN_first1000.npz', 'v3_cno_cont_transport', {'units': '', 'time_dependent': True}),
    ('cno_disc_transport', 'control', C / 'DiscTranslation_64x64_IN_first1000.npz', 'v3_cno_disc_transport', {'units': '', 'time_dependent': True}),
    ('pdegym_wave_gauss', 'control', EXT / 'wave_gauss/converted/wave_gauss_first256.npz', 'v3_pdegym_wave_gauss', {'units': '', 'time_dependent': True}),
    ('elado_helmholtz_e1', 'control', EXT / 'elado/converted/E1_test.h5', 'v3_elado_helmholtz', {'units': ''}),
    ('ucsd_convdiff', 'control', EXT / 'ucsd_pde/converted/ConvDiff1.npz', 'v3_ucsd_convdiff', {'units': '', 'head': 1000}),
    ('dpde_helmholtz', 'control', EXT / 'diffusionpde_norm/converted/helmholtz_hf_first1000.npz', 'v3_dpde_helmholtz', {'units': ''}),
]
ALL = INSCOPE + DIAGNOSTIC + CONTROL


def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.run(['git', *a], capture_output=True, text=True, cwd=HERE).stdout.strip()
        except OSError:
            return ''
    fr = rb.RESULTS / 'frozen_thresholds_v2.json'
    return {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'git_head': git('rev-parse', 'HEAD'),
            'git_dirty_tracked': git('status', '--porcelain', '--untracked-files=no'),
            'frozen_file': fr.name, 'frozen_file_sha256': hashlib.sha256(fr.read_bytes()).hexdigest(),
            'code_sha256_now': rb.code_hash(), 'python': platform.python_version(), 'numpy': np.__version__,
            'argv': sys.argv}


def run_one(spec, log):
    name, role = spec[0], spec[1]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.json'
    if path.exists():
        sys.exit(f'refusing: {path} exists. v3 runs are run once (protocol_v3.md section 8).')
    opt, _fr = rb.load_frozen('_v2')                              # exits if any frozen file changed
    prov = provenance()
    if role in ('inscope', 'diagnostic'):
        rec = rb.run_dataset(spec, opt, rb.FAULTS, log)
        recs = [rec]
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
            print(f'{s[1]:11s} {s[0]:28s} {s[3]:26s} {s[2]}  {"[exists]" if s[2].exists() else "[MISSING]"}')
        return 0
    specs = [s for s in ALL if (s[0] == a.run if a.run else s[1] == a.run_all)]
    if not specs:
        sys.exit('no such registered run')
    for s in specs:
        run_one(s, log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
