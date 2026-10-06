"""Fault-injection benchmark for fieldlint: measured detection and false-positive rates.

Stages (the order matters and is enforced):

  explore   run the TUNING datasets with the current default Options; results go to --out (scratch). Use this to choose
            thresholds. Never touches the held-out sets.
  freeze    write results/frozen_thresholds.json (timestamp, every Options field, SHA-256 of rules.py/adapters.py/presets).
  tuning    run the TUNING datasets and write results/tuning.json (requires the frozen file; checks the code hash).
  heldout   run the HELD-OUT datasets (PDEBench Darcy) with the frozen thresholds; refuses to run if the frozen file is
            missing or the code changed since it was written.
  control   run the out-of-scope controls (PDEBench diffusion-reaction, Burgers) the same way.
  posthoc   NOT a clean measurement: the held-out Darcy sets re-run with op_face_mean='arithmetic', chosen after the frozen
            run showed F008/F009 inapplicable there (written to results/heldout_posthoc_arithmetic.json).
  report    merge tuning.json / heldout.json / control.json into results/bench_results.json and bench_results.md.

Data are read read-only. Faults are injected into the INPUTS (source, k) or the pairing / labels of a clean copy:

  transpose, flip_x, flip_y   in-plane axes of source and k swapped / reversed (u untouched); transpose needs a square grid
  roll1..roll3                source and k circularly shifted by 1..3 cells along x
  shift_index                 u of sample i replaced by u of sample i+1 (pairing / sample-order fault)
  unit_slip                   u reduced by 273.15 while still declared kelvin (Celsius stored as kelvin); K datasets only
  cross_dup                   the last 10% of samples replaced by copies of earlier (train) samples; splits are assigned by
                              position (first 80% train, last 20% test) on every dataset
  <fault>@20                  the same fault on a random 20% of the samples (partial contamination)

A rule 'detects' a fault when it ends 'flagged' with a warning or error finding (info does not count). A skipped rule counts
as a miss but is also tallied separately. The false-positive rate is the same criterion on the clean copy.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from fieldlint import Options, load_config, open_dataset, run
from fieldlint.model import Dataset

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'src' / 'fieldlint'
RESULTS = HERE / 'results'
DATA = Path(r'D:\Work\3D-ICE Thermal-modelling\Thermo\data')
RULE_CODES = ['F001', 'F002', 'F003', 'F004', 'F005', 'F008', 'F009']
HEAD = 400

# name, group, path, preset, extras
TUNING = [
    *[(f'ictherm_s{s}', 'tuning', DATA / '_external/ictherm/datasets' / f'level{s}_steady', 'ictherm',
       {'transpose_inputs': True, 'head': HEAD, 'units': 'K'}) for s in (2, 3, 4, 5)],
    *[(f'thermfm_{n}', 'tuning', DATA / '_external/thermfm/thermal_steady' / n, 'thermfm', {'head': HEAD, 'units': 'K'})
      for n in ('HS_OC_refine1', 'HS_OC_refine2', 'HS_QC_refine1', 'HS_QC_refine2', 'HS_SC_refine1', 'HS_SC_refine2',
                'IND_8C', 'IND_32C')],
    *[(f'3dice_geometry{g}', 'tuning', DATA / '3d-ice' / f'geometry{g}', '3dice', {'units': 'K'})
      for g in ('1', '2a', '3', '4', '5', '6')],
]
HELDOUT = [
    ('darcy_beta1.0', 'heldout', DATA / 'pde-bench/darcy_beta1.0_n1000.npz', 'darcy', {'units': ''}),
    ('darcy_beta0.01', 'heldout', DATA / 'pde-bench/darcy_beta0.01_n1000.npz', 'darcy', {'units': ''}),
]
# Out-of-scope controls. The mapping is a user's most plausible reading of the file: u = final state, source = initial state
# (diffusion-reaction: channel 0 of an assumed (2, 128, 128) layout; Burgers: a 1 x 1024 line).
CONTROL = [
    ('diff_react_t20', 'control', DATA / 'pde-bench/diff_react_t20_n300.npz', None,
     {'units': '', 'cfg': {'format': 'npz', 'u': {'key': 'Y', 'layout': 'NCHW', 'channel': 0, 'reshape': [2, 128, 128]},
                           'source': {'key': 'X', 'layout': 'NCHW', 'channel': 0, 'reshape': [2, 128, 128]}}}),
    ('burgers_nu0.01', 'control', DATA / 'pde-bench/burgers_nu0.01_n400.npz', None,
     {'units': '', 'cfg': {'format': 'npz', 'u': {'key': 'Y', 'layout': 'NHW', 'reshape': [1, 1024]},
                           'source': {'key': 'X', 'layout': 'NHW', 'reshape': [1, 1024]}}}),
    ('burgers_nu0.01_t200', 'control', DATA / 'pde-bench/burgers_nu0.01_t200_n400.npz', None,
     {'units': '', 'cfg': {'format': 'npz', 'u': {'key': 'Y', 'layout': 'NHW', 'reshape': [1, 1024]},
                           'source': {'key': 'X', 'layout': 'NHW', 'reshape': [1, 1024]}}}),
]

FAULTS = ['transpose', 'flip_x', 'flip_y', 'roll1', 'roll2', 'roll3', 'shift_index', 'unit_slip', 'cross_dup',
          'transpose@20', 'flip_x@20', 'flip_y@20', 'roll1@20', 'shift_index@20']


# ── dataset construction ─────────────────────────────────────────────────────────
def build_clean(spec) -> Dataset:
    name, _group, path, preset, extra = spec
    cfg = load_config(preset) if preset else {}
    cfg = {**cfg, **extra.get('cfg', {})}
    cfg['units'] = extra.get('units', cfg.get('units', ''))
    if extra.get('transpose_inputs'):
        for nm in ('source', 'k'):
            if isinstance(cfg.get(nm), dict):
                cfg[nm] = {**cfg[nm], 'transpose': True}
    ds = open_dataset(path, cfg)
    if extra.get('head'):
        ds = ds.head(extra['head'])
    ds.name = name
    return ds


def positional_split(i: int, n: int) -> str:
    return 'train' if i < int(0.8 * n) else 'test'


def wrap(ds: Dataset, transform, name=None) -> Dataset:
    def loader(i):
        s = ds.get(i)
        if s is None:
            raise RuntimeError(ds.errors.get(i, 'sample failed to load'))
        return transform(i, s)
    out = Dataset(ds.n, loader, name or ds.name, ds.units, ds.field_shapes)
    out._keep = getattr(ds, '_keep', None)
    return out


def with_splits(ds: Dataset) -> Dataset:
    return wrap(ds, lambda i, s: dataclasses.replace(s, split=positional_split(i, ds.n)))


def _inputs(s, f):
    return dataclasses.replace(s, **{nm: (None if getattr(s, nm) is None else np.ascontiguousarray(f(getattr(s, nm))))
                                     for nm in ('source', 'k')})


def inject(base: Dataset, fault: str):
    """(faulted dataset, None) or (None, reason) when the fault does not apply to this dataset."""
    kind, _, frac = fault.partition('@')
    frac = float(frac) / 100 if frac else 1.0
    n = base.n
    chosen = np.random.default_rng(12345).random(n) < frac if frac < 1 else np.ones(n, bool)
    s0 = base.get(0)
    if kind == 'transpose' and s0.u.shape[-1] != s0.u.shape[-2]:
        return None, 'non-square grid'
    if kind == 'unit_slip' and (base.units or '').upper() != 'K':
        return None, 'u is not an absolute temperature in K'
    if kind == 'unit_slip' and frac < 1:
        return None, 'n/a'
    if kind == 'cross_dup' and frac < 1:
        return None, 'n/a'
    if kind in ('transpose', 'flip_x', 'flip_y') or kind.startswith('roll'):
        f = {'transpose': lambda a: np.swapaxes(a, -1, -2), 'flip_x': lambda a: a[..., ::-1],
             'flip_y': lambda a: a[..., ::-1, :]}.get(kind) or (lambda a, r=int(kind[4:]): np.roll(a, r, axis=-1))
        tf = lambda i, s: _inputs(s, f) if chosen[i] else s                          # noqa: E731
    elif kind == 'shift_index':
        def tf(i, s):
            if not chosen[i]:
                return s
            nxt = base.get((i + 1) % n)
            return dataclasses.replace(s, u=nxt.u)
    elif kind == 'unit_slip':
        tf = lambda i, s: dataclasses.replace(s, u=np.asarray(s.u, float) - 273.15)   # noqa: E731
    elif kind == 'cross_dup':
        cut = int(0.9 * n)

        def tf(i, s):
            return dataclasses.replace(base.get(i - cut), id=s.id) if i >= cut else s
    else:
        raise ValueError(fault)
    return with_splits(wrap(base, tf, f'{base.name}+{fault}')), None


# ── running ──────────────────────────────────────────────────────────────────────
def summarise(ds: Dataset, opt: Options, rules=RULE_CODES):
    rep = run(ds, rules, opt)
    out = {}
    for r in rep.results:
        sev = max((f.severity for f in r.findings), key={'info': 0, 'warning': 1, 'error': 2}.get, default='none')
        m = {k: v for k, v in r.metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
        out[r.code] = {'status': r.status, 'severity': sev, 'detected': r.status == 'flagged' and sev in ('warning', 'error'),
                       'summary': r.summary, 'metrics': m}
    return out


def run_dataset(spec, opt: Options, faults, log):
    name = spec[0]
    t0 = time.time()
    base = with_splits(build_clean(spec))
    raw = base                                   # injection works on `raw` (labels are re-applied after each fault)
    rec = {'name': name, 'group': spec[1], 'n': base.n, 'units': base.units, 'clean': summarise(base, opt), 'faults': {}}
    log(f'  {name}: clean done ({time.time() - t0:.0f}s)')
    for f in faults:
        fd, why = inject(raw, f)
        if fd is None:
            rec['faults'][f] = {'skipped': why}
            continue
        rec['faults'][f] = summarise(fd, opt)
    log(f'  {name}: {len(faults)} faults done ({time.time() - t0:.0f}s)')
    raw.close()
    return rec


def code_hash() -> dict:
    files = [PKG / 'rules.py', PKG / 'adapters.py', PKG / 'model.py', PKG / 'runner.py', *sorted((PKG / 'presets').glob('*.json'))]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def load_frozen() -> tuple[Options, dict]:
    p = RESULTS / 'frozen_thresholds.json'
    if not p.exists():
        sys.exit('refusing to run: results/frozen_thresholds.json does not exist (run --stage freeze after tuning)')
    fr = json.loads(p.read_text())
    now = code_hash()
    changed = [k for k, v in fr['code_sha256'].items() if now.get(k) != v]
    if changed:
        sys.exit(f'refusing to run: code changed since the thresholds were frozen: {changed}')
    return Options(**fr['options']), fr


def rates(records, rule):
    """per fault: (detected, trials, skipped, not_applicable) pooled over `records`."""
    out = {'clean': [0, 0, 0, 0]}
    for r in records:
        c = r['clean'][rule]
        out['clean'][1] += 1
        out['clean'][0] += c['detected']
        out['clean'][2] += c['status'] == 'skipped'
        for f, res in r['faults'].items():
            o = out.setdefault(f, [0, 0, 0, 0])
            if 'skipped' in res:
                o[3] += 1
                continue
            o[1] += 1
            o[0] += res[rule]['detected']
            o[2] += res[rule]['status'] == 'skipped'
    return out


def any_rate(records, rules):
    out = {}
    for r in records:
        for f, res in [('clean', r['clean'])] + list(r['faults'].items()):
            if 'skipped' in res:
                continue
            o = out.setdefault(f, [0, 0])
            o[1] += 1
            o[0] += any(res[c]['detected'] for c in rules)
    return out


def markdown(blocks: dict, opt: dict, frozen_ts: str) -> str:
    lines = [f'# fieldlint measured detection / false-positive rates', '',
             f'Thresholds frozen {frozen_ts} on the tuning datasets (results/frozen_thresholds.json). Cell = detected / trials '
             f'(skipped by the rule in brackets). Detected = rule flagged with a warning or error. "clean" = false-positive rate.', '']
    for title, recs in blocks.items():
        if not recs:
            continue
        lines += [f'## {title} ({len(recs)} datasets: {", ".join(r["name"] for r in recs)})', '']
        lines += ['| fault | ' + ' | '.join(RULE_CODES) + ' | any rule |', '|---|' + '---|' * (len(RULE_CODES) + 1)]
        per = {c: rates(recs, c) for c in RULE_CODES}
        anyr = any_rate(recs, RULE_CODES)
        for f in ['clean'] + FAULTS:
            if f not in per['F001'] or per['F001'][f][1] == 0:
                continue
            cells = []
            for c in RULE_CODES:
                d, t, sk, _ = per[c][f]
                cells.append(f'{d}/{t}' + (f' ({sk})' if sk else ''))
            lines.append(f'| {f} | ' + ' | '.join(cells) + f' | {anyr[f][0]}/{anyr[f][1]} |')
        lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['explore', 'freeze', 'tuning', 'heldout', 'control', 'posthoc', 'report'])
    ap.add_argument('--only', help='comma separated dataset-name substrings')
    ap.add_argument('--faults', help='comma separated subset of faults')
    ap.add_argument('--out', help='output directory for explore (default: ./bench_explore)')
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    faults = a.faults.split(',') if a.faults else FAULTS
    log = lambda m: print(m, flush=True)                                          # noqa: E731

    def select(specs):
        return [s for s in specs if not a.only or any(t in s[0] for t in a.only.split(','))]

    if a.stage == 'freeze':
        opt = dataclasses.asdict(Options())
        out = {'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'tuned_on': [s[0] for s in TUNING],
               'held_out_not_yet_run': [s[0] for s in HELDOUT + CONTROL], 'options': opt, 'code_sha256': code_hash(),
               'fixed_in_code': {'F003_fire_fractions': [0.1, 0.5], 'F008_fire_fractions': [0.1, 0.5],
                                 'F009_fractions': [opt['pair_warn'], opt['pair_error']]}}
        (RESULTS / 'frozen_thresholds.json').write_text(json.dumps(out, indent=1))
        log(f'frozen at {out["frozen_at"]}')
        return 0
    if a.stage == 'explore':
        opt, outdir = Options(), Path(a.out or 'bench_explore')
        outdir.mkdir(exist_ok=True, parents=True)
        recs = [run_dataset(s, opt, faults, log) for s in select(TUNING)]
        (outdir / 'explore.json').write_text(json.dumps(recs, indent=1, default=float))
        log(markdown({'explore (tuning datasets)': recs}, {}, 'NOT FROZEN'))
        return 0
    if a.stage == 'report':
        fr = json.loads((RESULTS / 'frozen_thresholds.json').read_text())
        now = code_hash()
        changed = [k for k, v in fr['code_sha256'].items() if now.get(k) != v]      # e.g. op_face_mean added for the post-hoc run
        blocks = {}
        for nm, title in (('tuning', 'TUNING datasets (thresholds were fitted here)'),
                          ('heldout', 'HELD-OUT datasets (thresholds frozen before these were run)'),
                          ('heldout_posthoc_arithmetic', 'HELD-OUT datasets, POST-HOC re-run with arithmetic face conductivity (chosen after seeing the frozen run; not a clean held-out measurement)')):
            p = RESULTS / f'{nm}.json'
            blocks[title] = json.loads(p.read_text()) if p.exists() else []
        md = markdown(blocks, fr['options'], fr['frozen_at'])
        if changed:
            md += ('\nNote: files changed after the freeze: ' + ', '.join(changed) + '. The tuning / held-out / control runs were '
                   'made with the frozen code; the only later change is the op_face_mean option (default = the frozen '
                   'harmonic mean), used by the post-hoc block.\n')
        ctl = RESULTS / 'control.json'
        if ctl.exists():
            md += '\n## Out-of-scope controls (clean files, no faults)\n\n| dataset | ' + ' | '.join(RULE_CODES) + ' |\n|---|' + '---|' * len(RULE_CODES) + '\n'
            for r in json.loads(ctl.read_text()):
                md += f'| {r["name"]} | ' + ' | '.join(
                    f'{r["clean"][c]["status"]}' + (f' ({r["clean"][c]["severity"]})' if r['clean'][c]['severity'] != 'none' else '')
                    for c in RULE_CODES) + ' |\n'
        (RESULTS / 'bench_results.md').write_text(md)
        log(md)
        return 0
    if a.stage == 'posthoc':
        fr = json.loads((RESULTS / 'frozen_thresholds.json').read_text())
        opt = Options(**{**fr['options'], 'op_face_mean': 'arithmetic'})
        recs = [run_dataset(s, opt, faults, log) for s in select(HELDOUT)]
        (RESULTS / 'heldout_posthoc_arithmetic.json').write_text(json.dumps(recs, indent=1, default=float))
        return 0
    opt, fr = load_frozen()
    if a.stage == 'tuning':
        recs = [run_dataset(s, opt, faults, log) for s in select(TUNING)]
        (RESULTS / 'tuning.json').write_text(json.dumps(recs, indent=1, default=float))
    elif a.stage == 'heldout':
        recs = [run_dataset(s, opt, faults, log) for s in select(HELDOUT)]
        (RESULTS / 'heldout.json').write_text(json.dumps(recs, indent=1, default=float))
    else:
        recs = []
        for s in select(CONTROL):
            base = with_splits(build_clean(s))
            recs.append({'name': s[0], 'group': 'control', 'n': base.n, 'units': base.units, 'faults': {},
                         'clean': summarise(base, opt)})
            base.close()
            log(f'  {s[0]} done')
        (RESULTS / 'control.json').write_text(json.dumps(recs, indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
