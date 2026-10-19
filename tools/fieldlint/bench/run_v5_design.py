"""Round 5 (v5) design-phase tuning run of fieldlint: F005 scope gate + weak-minimum-principle probe, F009 contiguous block.
NOT a held-out measurement. Re-uses run_v4_design.run_one (the same fault injection / summarise machinery) on every dataset seen up to
the end of v4 (the 40 v4 tuning runs plus the 13 v4 held-out runs, which are tuning data now), with the v5 defaults of `Options`.

  python tools/fieldlint/bench/run_v5_design.py --list
  python tools/fieldlint/bench/run_v5_design.py --run <name> [--force]        # -> results/v5_tuning/<name>.json
  python tools/fieldlint/bench/run_v5_design.py --run-all [--only a,b]
  python tools/fieldlint/bench/run_v5_design.py --compare                     # before (stored v4 results) vs after; lists UNEXPECTED diffs
  python tools/fieldlint/bench/run_v5_design.py --freeze                      # results/frozen_thresholds_v5.json

Allowed differences (decided before the run, plan_v5 section 1.3 + user decision 2026-10-15): F005 on PDEgym Helmholtz and PDEgym Wave-Layer
(flagged and unflagged), and F009 anywhere (intended change). Anything else is UNEXPECTED and means STOP.
Extended AFTER the run (disclosed in the README): an F005 change that only adds an info line ('below-ambient check not applied') to a run declared time-dependent
(status ok, no detection) is accepted, because it is the intended by-product of part A (plan_v5 section 1.3: 'every flagged time-dependent control'); it occurred once
(cno_cont_transport+time_dependent_flag).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import run_bench as rb
import run_v4_design as r4d
import run_v4_heldout as r4h
from fieldlint import Options

HERE = Path(__file__).resolve().parent
OUT = rb.RESULTS / 'v5_tuning'
ALL = r4d.ALL + r4h.ALL
RULES = rb.RULE_CODES
ALLOWED_F005 = ('pdegym_helmholtz', 'pdegym_wave_layer')


def before_path(name):
    base = name.split('+')[0]
    for d in (rb.RESULTS / 'v4_tuning', rb.RESULTS / 'v4_heldout'):
        if (d / f'{base}.json').exists():
            return d / f'{base}.json'
    return None


def load_records(path):
    return {r['name']: r for r in json.loads(Path(path).read_text())['records']}


def key(x):
    return (x['status'], x['severity'], x['detected'])


def diffs():
    """list of (record, condition, rule, before, after, expected: bool)"""
    out = []
    for spec in ALL:
        bp, ap = before_path(spec[0]), OUT / f'{spec[0]}.json'
        if bp is None or not ap.exists():
            out.append((spec[0], '-', '-', 'MISSING', 'MISSING', False))
            continue
        b, a = load_records(bp), load_records(ap)
        for n in sorted(set(b) | set(a)):
            if n not in b or n not in a:
                out.append((n, '-', '-', 'record in only one side', '', False))
                continue
            conds = {'clean': (b[n]['clean'], a[n]['clean'])}
            for f, rb_ in b[n]['faults'].items():
                ra_ = a[n]['faults'].get(f)
                if 'skipped' in rb_ or ra_ is None or 'skipped' in ra_:
                    if ('skipped' in rb_) != (ra_ is None or 'skipped' in ra_):
                        out.append((n, f, '-', 'fault applicability differs', '', False))
                    continue
                conds[f] = (rb_, ra_)
            for c, (rb_, ra_) in conds.items():
                for rule in RULES:
                    if key(rb_[rule]) != key(ra_[rule]):
                        info_only = (rule == 'F005' and not rb_[rule]['detected'] and not ra_[rule]['detected']
                                     and ra_[rule]['severity'] == 'info' and rb_[rule]['severity'] == 'none' and ra_[rule]['status'] == 'ok')
                        exp = rule == 'F009' or (rule == 'F005' and n.split('+')[0] in ALLOWED_F005) or (info_only and n.endswith('+time_dependent_flag'))
                        out.append((n, c, rule, key(rb_[rule]), key(ra_[rule]), exp))
    return out


def compare():
    d = diffs()
    unexpected = [x for x in d if not x[5]]
    md = ['# fieldlint v5: tuning before / after (before = stored v4 results, after = v5 code)', '',
          'v5 changes: F005 below-ambient check honours scope_skip and the weak-minimum-principle probe (osc_v 0.05, fraction 0.5); '
          'F009 scores the first 64 consecutive samples. Tuned on everything seen; NOT held-out numbers.', '',
          f'Records compared: {len(ALL)} runs. Cells changed in status/severity/detected: {len(d)}; UNEXPECTED (not F009, not F005 on '
          f'PDEgym Helmholtz / Wave-Layer): **{len(unexpected)}**.', '']
    if unexpected:
        md += ['## UNEXPECTED differences', '', '| record | condition | rule | before | after |', '|---|---|---|---|---|']
        md += [f'| {n} | {c} | {r} | {b} | {a} |' for n, c, r, b, a, _ in unexpected]
        md.append('')
    md += ['## F005 differences (expected set)', '', '| record | condition | before | after |', '|---|---|---|---|']
    md += [f'| {n} | {c} | {b} | {a} |' for n, c, r, b, a, _ in d if r == 'F005']
    md += ['', '## F009 (intended change): detection / ran / skipped per fault, pooled over in-scope + diagnostic runs', '',
           '| condition | before | after |', '|---|---|---|']
    agg = {}
    for spec in ALL:
        if spec[1] == 'control':
            continue
        bp, ap = before_path(spec[0]), OUT / f'{spec[0]}.json'
        if bp is None or not ap.exists():
            continue
        b, a = load_records(bp)[spec[0]], load_records(ap)[spec[0]]
        for c in ['clean'] + rb.FAULTS:
            rb_ = b['clean'] if c == 'clean' else b['faults'].get(c)
            ra_ = a['clean'] if c == 'clean' else a['faults'].get(c)
            if rb_ is None or ra_ is None or 'skipped' in rb_ or 'skipped' in ra_:
                continue
            t = agg.setdefault(c, [[0, 0, 0], [0, 0, 0]])
            for side, r in ((0, rb_), (1, ra_)):
                x = r['F009']
                t[side][0] += x['detected']
                t[side][1] += x['status'] != 'skipped'
                t[side][2] += x['status'] == 'skipped'
    for c, (x, y) in agg.items():
        md.append(f'| {c} | {x[0]} detected / {x[1]} ran / {x[2]} skipped | {y[0]} / {y[1]} / {y[2]} |')
    md += ['', '## F009 changes on clean copies', '', '| record | before | after |', '|---|---|---|']
    md += [f'| {n} | {b} | {a} |' for n, c, r, b, a, _ in d if r == 'F009' and c == 'clean']
    md += ['', '## Conditions where the any-rule detection (all rules) changed', '']
    anyrows = []
    for spec in ALL:
        bp, ap = before_path(spec[0]), OUT / f'{spec[0]}.json'
        if bp is None or not ap.exists():
            continue
        b, a = load_records(bp), load_records(ap)
        for n in b:
            if n not in a:
                continue
            for c in ['clean'] + rb.FAULTS:
                rb_ = b[n]['clean'] if c == 'clean' else b[n]['faults'].get(c)
                ra_ = a[n]['clean'] if c == 'clean' else a[n]['faults'].get(c)
                if rb_ is None or ra_ is None or 'skipped' in rb_ or 'skipped' in ra_:
                    continue
                x, y = any(rb_[r]['detected'] for r in RULES), any(ra_[r]['detected'] for r in RULES)
                if x != y:
                    anyrows.append(f'* {n} / {c}: any-rule {x} -> {y}')
    md += anyrows or ['(none)']
    (rb.RESULTS / 'bench_results_v5_tuning.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md[:8]))
    print('UNEXPECTED:', len(unexpected))
    for x in unexpected:
        print(x[:5])
    return len(unexpected)


def freeze():
    after = sorted(OUT.glob('*.json'))
    if len(after) != len({s[0] for s in ALL}):
        sys.exit('refusing: not all tuning runs present')
    now = rb.code_hash()
    for p in after:
        if json.loads(p.read_text())['meta']['code_sha256'] != now:
            sys.exit(f'refusing: code changed since {p.name} was produced')
    if compare():
        sys.exit('refusing: unexpected differences')
    opt = dataclasses.asdict(Options())

    def here(n):
        return hashlib.sha256((HERE / n).read_bytes()).hexdigest()
    out = {'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'version': 5,
           'tuned_on': [s[0] for s in ALL], 'options': opt, 'code_sha256': now,
           'bench_code_sha256': {n: here(n) for n in ('run_bench.py', 'run_v3.py', 'run_v4_design.py', 'run_v4_heldout.py', 'run_v5_design.py')},
           'v2_v3_v4_frozen_files_untouched': ['frozen_thresholds.json', 'frozen_thresholds_v2.json', 'frozen_thresholds_v4.json'],
           'new_in_v5': {'F005_scope_gate': 'below-ambient check not applied when scope_skip() gives a reason (time_dependent, operator advective|other, effectively 1D)',
                         'F005_weak_minimum_principle_probe': {
                             'osc_v': opt['osc_v'], 'osc_fraction': opt['osc_fraction'],
                             'rule': 'V = (min over in-plane boundary ring - min u)/ptp(u), max over z-layers; sample oscillatory when V >= osc_v and u.max > ambient + tol; below-ambient warning -> info when oscillatory samples >= osc_fraction of the samples with non-negative source and known ambient'},
                         'F009_contiguous_block': {'pair_samples': opt['pair_samples'],
                                                   'rule': 'the first pair_samples consecutive samples are scored pairwise (v4: 32 evenly spaced)'}},
           'note': 'tuned on everything seen, including the v4 held-out sets and the two v4 control failures; no dataset is held out for v5. Earlier frozen files and results are unchanged.'}
    (rb.RESULTS / 'frozen_thresholds_v5.json').write_text(json.dumps(out, indent=1))
    print('frozen v5 at', out['frozen_at'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--list', action='store_true')
    g.add_argument('--run')
    g.add_argument('--run-all', action='store_true')
    g.add_argument('--compare', action='store_true')
    g.add_argument('--freeze', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only')
    a = ap.parse_args(argv)

    def log(m):
        print(m, flush=True)
    if a.list:
        for s in ALL:
            print(f'{s[1]:11s} {s[0]:28s} {"[exists]" if s[2].exists() else "[MISSING]"} before={before_path(s[0]) is not None}')
        return 0
    if a.compare:
        return 1 if compare() else 0
    if a.freeze:
        freeze()
        return 0
    r4d.OUT = OUT
    specs = [s for s in ALL if s[0] == a.run] if a.run else [s for s in ALL if not a.only or any(t in s[0] for t in a.only.split(','))]
    if not specs:
        sys.exit('no such run')
    for s in specs:
        r4d.run_one(s, log, a.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
