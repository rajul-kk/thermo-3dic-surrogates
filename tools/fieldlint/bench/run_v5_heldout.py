"""Round 5 (v5) held-out measurement of fieldlint: see protocol_v5.md.

The ONLY code path that may produce v5 held-out results. Re-uses the v2 machinery of run_bench.py unchanged (fault injection for the 14 standard faults,
summarise(), the hash check) with the FROZEN V5 file (bench/results/frozen_thresholds_v5.json), and adds (a) the v5 dataset registry, (b) the new
fault kinds of C5 (F004 duplicate doses) and C6 (F009 pairing arms), (c) a run-once guard, (d) provenance fields in every result file.

  python tools/fieldlint/bench/run_v5_heldout.py --list                       # print the registered runs, run nothing
  python tools/fieldlint/bench/run_v5_heldout.py --run <name>                 # one registered run; refuses if its result file exists
  python tools/fieldlint/bench/run_v5_heldout.py --run-all control|inscope|inscope_nb|diagnostic|control_nb

Result file per run: bench/results/v5_heldout/<name>.json. A run whose result file exists is NEVER repeated (protocol_v5.md section 8).
Mapping configs are the staged presets in bench/presets_v5_staging/ (NOT in src/, so that the v5 hash is untouched), passed as file paths.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import run_bench as rb                                           # same directory; the v2 machinery, unchanged
from fieldlint.model import Dataset

HERE = Path(__file__).resolve().parent
OUT = rb.RESULTS / 'v5_heldout'
D5 = Path(r'D:\fieldlint_data_v5')
PRE = HERE / 'presets_v5_staging'
FROZEN = 'frozen_thresholds_v5.json'
SEED = 54321


def P(name: str) -> str:
    return str(PRE / f'{name}.json')


CD = D5 / 'condiff_v5/converted'
# (name, role, path, preset path, extras).  roles: inscope (binding), inscope_nb (reported, non-binding), diagnostic, control (binding), control_nb
INSCOPE = [(f'condiff_{n}', 'inscope', CD / f'{n}_grid128_test.npz', P('v5_condiff'), {'units': ''})
           for n in ('cubic1.0', 'cubic2.0', 'exponential0.4', 'gaussian0.4', 'cubic0.1', 'exponential1.0')]
INSCOPE_NB = [
    ('chip_fem_a', 'inscope_nb', D5 / 'generated/chip_fem_a.npz', P('v5_chip_fem_a'), {'units': 'K'}),
    ('chip_fem_b', 'inscope_nb', D5 / 'generated/chip_fem_b.npz', P('v5_chip_fem_b'), {'units': 'K'}),
]
DIAGNOSTIC = [
    ('sissa_thermal', 'diagnostic', D5 / 'sissa_thermal_conduction_star/converted/sissa_star_2_3_4_stripes_128.npz', P('v5_sissa_thermal'), {'units': ''}),
]
CONTROL = [
    ('helmholtz_staircase', 'control', D5 / 'helmholtz_staircase/converted/helmholtz_staircase_train_first13_t0.npz', P('v5_helmholtz_staircase'), {'units': ''}),
    ('flowbench_ldcht', 'control', D5 / 'flowbench_ldcht_constRe_128_harmonics/converted/flowbench_ldcht_constRe_128_harmonics_N1000.npz',
     P('v5_flowbench_ldcht'), {'units': ''}),
    ('acoustic_inclusions', 'control', D5 / 'acoustic_inclusions/converted/acoustic_inclusions_test36_37.npz', P('v5_acoustic_inclusions'),
     {'units': '', 'time_dependent': True}),
    ('rayleigh_benard', 'control', D5 / 'rayleigh_benard/converted/rayleigh_benard_first256.npz', P('v5_rayleigh_benard'),
     {'units': '', 'time_dependent': True}),
]
CONTROL_NB = [
    ('advdiff', 'control_nb', D5 / 'generated/advdiff.npz', P('v5_advdiff'), {'units': ''}),
]
ALL = INSCOPE + INSCOPE_NB + DIAGNOSTIC + CONTROL + CONTROL_NB
FAULT_ROLES = ('inscope', 'inscope_nb', 'diagnostic')

C5_FAULTS = ['dup_x1', 'dup_x5', 'dup_within5', 'dupnoise_1e-7', 'dupnoise_1e-4', 'dupnoise_1e-2']      # cross_dup (= dup_x10) is in the standard set
C6_FAULTS = ['shift_index', 'shift_index_5', 'permute20', 'shuffle_all']
HEAD_BLOCK = 64
LEGACY_M = 32


# ── new fault kinds (outside the hashed rule code) ───────────────────────────────────────────────────────────────
def _positions(n):
    cut = int(0.8 * n)                                           # positional split as run_bench.positional_split
    return np.arange(cut), np.arange(cut, n)


def inject_v5(base: Dataset, fault: str):
    n = base.n
    rng = np.random.default_rng(SEED)
    tr, te = _positions(n)
    repl = {}                                                    # position -> function(sample) -> sample
    if fault.startswith('dup_x') or fault.startswith('dupnoise_'):
        pct = float(fault[5:]) if fault.startswith('dup_x') else 5.0
        c = max(2, int(round(pct / 100.0 * n)))
        dst = rng.choice(te, c, replace=False)
        src = rng.choice(tr, c, replace=False)
        rel = float(fault.split('_')[1]) if fault.startswith('dupnoise_') else 0.0
        for j, (d, s) in enumerate(zip(dst, src)):
            repl[int(d)] = (int(s), rel, j)
    elif fault == 'dup_within5':
        c = max(2, int(round(0.05 * n)))
        dst = rng.choice(np.arange(1, len(tr)), c, replace=False)
        for j, d in enumerate(dst):
            repl[int(d)] = (int(rng.integers(0, d)), 0.0, j)
    elif fault in ('shift_index_5', 'shuffle_all', 'permute20'):
        if fault == 'shift_index_5':
            m = {i: (i + 5) % n for i in range(n)}
        elif fault == 'shuffle_all':
            perm = rng.permutation(n)
            m = {i: int(perm[i]) for i in range(n)}
        else:
            pos = np.sort(rng.choice(n, max(2, int(round(0.2 * n))), replace=False))
            perm = rng.permutation(pos)
            m = {int(a): int(b) for a, b in zip(pos, perm)}

        def tf_u(i, s):
            return dataclasses.replace(s, u=base.get(m[i]).u) if i in m else s
        return rb.with_splits(rb.wrap(base, tf_u, f'{base.name}+{fault}'))
    else:
        raise ValueError(fault)

    def tf(i, s):
        if i not in repl:
            return s
        src, rel, j = repl[i]
        o = base.get(src)
        if rel > 0:
            u = np.asarray(o.u, float)
            noise = np.random.default_rng(SEED + 1000 + j).normal(0.0, rel * float(np.ptp(u)), u.shape)
            return dataclasses.replace(o, u=u + noise, id=s.id)
        return dataclasses.replace(o, id=s.id)
    return rb.with_splits(rb.wrap(base, tf, f'{base.name}+{fault}'))


def evenly_spaced_view(ds: Dataset, m: int = LEGACY_M) -> Dataset:
    """A dataset made of the m evenly spaced samples the v4 F009 scored (ds.indices(m)); emulates the v4 sampling exactly."""
    idx = ds.indices(m)

    def loader(j):
        s = ds.get(idx[j])
        if s is None:
            raise RuntimeError(ds.errors.get(idx[j], 'sample failed to load'))
        return s
    out = Dataset(len(idx), loader, ds.name + '+spaced32', ds.units, ds.field_shapes)
    out._keep = getattr(ds, '_keep', None)
    out.time_dependent = ds.time_dependent
    out.operator = getattr(ds, 'operator', '')
    return out


def f009_arms(ds: Dataset, opt):
    """F009 on (A) the dataset as given (shipped rule: the first 64 consecutive samples), (B) its head block of 64 (--head 64), (L) legacy v4 sampling."""
    return {'A_shipped': rb.summarise(ds, opt, ['F009'])['F009'],
            'B_head64': rb.summarise(ds.head(HEAD_BLOCK), opt, ['F009'])['F009'],
            'L_spaced32': rb.summarise(evenly_spaced_view(ds), opt, ['F009'])['F009']}


# ── provenance and running ───────────────────────────────────────────────────────────────────────────────────────
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
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE / 'run_bench.py', HERE / 'run_v5_heldout.py', HERE / 'report_v5.py', HERE / 'convert_v5.py', HERE / 'gen_v5.py', HERE / 'protocol_v5.md', HERE / 'datasets_v5.md') if f.exists()}
    return {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'git_head': git('rev-parse', 'HEAD'),
            'git_dirty_tracked': git('status', '--porcelain', '--untracked-files=no'),
            'frozen_file': fr.name, 'frozen_file_sha256': hashlib.sha256(fr.read_bytes()).hexdigest(),
            'code_sha256_now': rb.code_hash(), 'bench_files_sha256': hashes,
            'staged_presets_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(PRE.glob('*.json'))},
            'python': platform.python_version(), 'numpy': np.__version__, 'h5py': h5v, 'argv': sys.argv}


def run_one(spec, log):
    name, role = spec[0], spec[1]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.json'
    if path.exists():
        sys.exit(f'refusing: {path} exists. v5 held-out runs are run once (protocol_v5.md section 8).')
    opt, _fr = rb.load_frozen('_v5')                              # exits if any frozen file changed
    prov = provenance()
    t0 = time.time()
    if role in FAULT_ROLES:
        rec = rb.run_dataset(spec, opt, rb.FAULTS, log)           # clean + the 14 standard faults, all rules (F009 = shipped rule)
        raw = rb.with_splits(rb.build_clean(spec))
        rec['c5'] = {}
        for f in C5_FAULTS:
            rec['c5'][f] = rb.summarise(inject_v5(raw, f), opt, ['F004'])['F004']
        log(f'  {name}: C5 done ({time.time() - t0:.0f}s)')
        rec['c6'] = {'clean': f009_arms(raw, opt)}
        for f in C6_FAULTS:
            fd = rb.inject(raw, 'shift_index')[0] if f == 'shift_index' else inject_v5(raw, f)
            rec['c6'][f] = f009_arms(fd, opt)
        log(f'  {name}: C6 done ({time.time() - t0:.0f}s)')
        raw.close()
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
    prov['seconds'] = round(time.time() - t0)
    path.write_text(json.dumps({'provenance': prov, 'role': role, 'records': recs}, indent=1, default=float))
    log(f'wrote {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--list', action='store_true')
    g.add_argument('--run')
    g.add_argument('--run-all', choices=['inscope', 'inscope_nb', 'diagnostic', 'control', 'control_nb'])
    a = ap.parse_args(argv)

    def log(m):
        print(m, flush=True)
    if a.list:
        for s in ALL:
            print(f'{s[1]:11s} {s[0]:24s} {Path(s[3]).stem:26s} {"[exists]" if s[2].exists() else "[MISSING]"}  '
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
