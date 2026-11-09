"""Round 4 (v4) design-phase tuning run of fieldlint: F003 defers to F008, advection probe. NOT a held-out measurement.

v4 is tuned on every dataset seen so far, including the v3 sets and the v3 control failures (see README "Round 4 (v4) design").
Re-uses the v2 machinery of run_bench.py (fault injection, summarise()) and the v3 dataset registry (run_v3.py) unchanged.

  python tools/fieldlint/bench/run_v4_design.py --list
  python tools/fieldlint/bench/run_v4_design.py --run <name> [--force]       # one dataset -> results/v4_tuning/<name>.json
  python tools/fieldlint/bench/run_v4_design.py --run-group tuning|v3|controls
  python tools/fieldlint/bench/run_v4_design.py --freeze                    # write results/frozen_thresholds_v4.json
  python tools/fieldlint/bench/run_v4_design.py --report                    # results/bench_results_v4_tuning.md (before / after tables)

Groups: tuning = the v2 tuning sets (IC-ThermBench, Therm-FM, 3D-ICE, PDEBench Darcy x2) + FNO Darcy 128 and native 421 (v2 held-out,
seen); v3 = the nine v3 in-scope / diagnostic sets with the full fault set; controls = the nine v3 controls, clean only (with the
time_dependent declaration where v3 used it, and without).
"The before" columns come from the stored v2 / v3 result files, which are never rewritten.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import run_bench as rb
import run_v3 as r3
from fieldlint import Options

HERE = Path(__file__).resolve().parent
OUT = rb.RESULTS / 'v4_tuning'
TUNING = rb.TUNING2 + rb.HELDOUT2 + rb.HELDOUT2B
V3 = r3.INSCOPE + r3.DIAGNOSTIC
CONTROLS = r3.CONTROL
GROUPS = {'tuning': TUNING, 'v3': V3, 'controls': CONTROLS}
ALL = TUNING + V3 + CONTROLS
RULES = ['F002', 'F003', 'F008', 'F009']


def extra_files_hash() -> dict:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE / 'run_bench.py', HERE / 'run_v3.py', HERE / 'run_v4_design.py')}


def run_one(spec, log, force=False):
    name = spec[0]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.json'
    if path.exists() and not force:
        log(f'skip {name}: {path.name} exists')
        return
    opt = Options()                                                  # v4 defaults ARE the v4 thresholds (frozen afterwards)
    t0 = time.time()
    if spec[1] == 'control':
        recs = []
        variants = [spec]
        if spec[4].get('time_dependent'):
            variants = [(spec[0] + '+time_dependent_flag', *spec[1:4], {**spec[4], 'cfg': {**spec[4].get('cfg', {}), 'time_dependent': True}}), spec]
        for s in variants:
            base = rb.with_splits(rb.build_clean(s))
            recs.append({'name': s[0], 'group': 'control', 'n': base.n, 'units': base.units, 'faults': {}, 'clean': rb.summarise(base, opt)})
            base.close()
            log(f'  {s[0]} done')
    else:
        recs = [rb.run_dataset(spec, opt, rb.FAULTS, log)]
    meta = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'seconds': round(time.time() - t0), 'python': platform.python_version(),
            'numpy': np.__version__, 'code_sha256': rb.code_hash(), 'options': dataclasses.asdict(opt)}
    path.write_text(json.dumps({'meta': meta, 'role': spec[1], 'records': recs}, indent=1, default=float))
    log(f'wrote {path.name} ({meta["seconds"]} s)')


# ── before / after ───────────────────────────────────────────────────────────────
def load_before() -> dict:
    """name -> record from the stored v2 / v3 result files (read only)."""
    before = {}
    for fn in ('tuning_v2.json', 'heldout_v2.json', 'heldout_native_v2.json'):
        for r in json.loads((rb.RESULTS / fn).read_text()):
            before[r['name']] = r
    for p in sorted((rb.RESULTS / 'v3').glob('*.json')):
        for r in json.loads(p.read_text())['records']:
            before[r['name']] = r
    return before


def load_after() -> dict:
    return {r['name']: r for p in sorted(OUT.glob('*.json')) for r in json.loads(p.read_text())['records']}


def det(res, rules):
    return sorted(c for c in rules if res[c]['detected'])


def cell(res, rules):
    d = det(res, rules)
    return ','.join(c[1:] for c in d) if d else '-'


def dataset_table(name, b, a):
    lines = [f'#### {name}', '', '| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |', '|---|---|---|---|---|']
    anyr = rb.RULE_CODES
    for f in ['clean'] + rb.FAULTS:
        rb_, ra_ = (b['clean'], a['clean']) if f == 'clean' else (b['faults'].get(f), a['faults'].get(f))
        if rb_ is None or ra_ is None or 'skipped' in rb_ or 'skipped' in ra_:
            continue
        ab, aa = bool(det(rb_, anyr)), bool(det(ra_, anyr))
        mark = '' if ab == aa else (' **LOST**' if ab and not aa else ' gained')
        lines.append(f'| {f} | {cell(rb_, RULES)} | {cell(ra_, RULES)} | {"yes" if ab else "no"} | {"yes" if aa else "no"}{mark} |')
    return '\n'.join(lines) + '\n'


def pooled(before, after, names, title):
    out = [f'### {title}', '']
    for lab, src in (('before (v2 code)', before), ('after (v4 code)', after)):
        recs = [src[n] for n in names if n in src]
        body = rb.markdown({lab: recs}, {}, '').split('\n')
        out += [f'**{lab}**', ''] + body[next(i for i, l in enumerate(body) if l.startswith('## ')) + 1:]
    return '\n'.join(out)


def controls_table(before, after):
    lines = ['| run | rule | before | after |', '|---|---|---|---|']
    for s in CONTROLS:
        names = [s[0]] + ([s[0] + '+time_dependent_flag'] if s[4].get('time_dependent') else [])
        for n in names:
            if n not in after or n not in before:
                continue
            for c in RULES + ['F005']:
                cb, ca = before[n]['clean'][c], after[n]['clean'][c]
                f = lambda x: x['status'] + (f" ({x['severity']})" if x['severity'] not in ('none',) else '')       # noqa: E731
                if f(cb) != f(ca):
                    lines.append(f'| {n} | {c} | {f(cb)} | {f(ca)} |')
    return '\n'.join(lines)


def false_alarm_rows(rec_map, names):
    rows = []
    for n in names:
        if n in rec_map:
            bad = [f"{c} {rec_map[n]['clean'][c]['severity']}" for c in RULES + ['F005'] if rec_map[n]['clean'][c]['detected']]
            rows.append((n, bad))
    return rows


def report():
    before, after = load_before(), load_after()
    md = ['# fieldlint v4: tuning results (before = v2/v3 stored results, after = v4 code)', '',
          'v4 is TUNED on every dataset seen so far, including the v3 sets and the v3 control failures. These are tuning numbers, not held-out ones.',
          'Cells list the rules (F-number without the F) that detect each fault; "any-rule" uses all of F001-F009 as in the earlier tables.', '']
    groups = [('tuning', 'v2 tuning sets + v2 held-out sets (all seen)'), ('v3', 'v3 in-scope and diagnostic sets (now seen)')]
    for g, title in groups:
        names = [s[0] for s in GROUPS[g] if s[0] in after]
        md.append(f'## {title}\n')
        md.append(pooled(before, after, names, 'pooled'))
        md.append('\n### per dataset\n')
        for n in names:
            md.append(dataset_table(n, before[n], after[n]))
    md.append('## controls (clean files)\n')
    md.append('Only the cells that changed are listed.\n')
    md.append(controls_table(before, after))
    md.append('\nFalse alarms (warning or error from F002/F003/F005/F008/F009) after v4, per control run:\n')
    cn = [n for s in CONTROLS for n in ([s[0] + '+time_dependent_flag'] if s[4].get('time_dependent') else []) + [s[0]]]
    for n, bad in false_alarm_rows(after, cn):
        bb = dict(false_alarm_rows(before, [n]))[n]
        md.append(f'* {n}: before {bb or "none"}; after {bad or "none"}')
    (rb.RESULTS / 'bench_results_v4_tuning.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md)[:3000])


def freeze():
    after = sorted(OUT.glob('*.json'))
    hashes = {json.loads(p.read_text())['meta']['code_sha256'].__repr__() for p in after}
    now = rb.code_hash()
    if hashes != {now.__repr__()}:
        sys.exit('refusing: the code changed since (some of) the tuning results were produced')
    opt = dataclasses.asdict(Options())
    out = {'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'version': 4,
           'tuned_on': [s[0] for s in ALL], 'held_out_not_yet_run': [], 'options': opt, 'code_sha256': now,
           'bench_code_sha256': extra_files_hash(),
           'v2_v3_frozen_files_untouched': ['frozen_thresholds.json', 'frozen_thresholds_v2.json'],
           'new_in_v4': {'F003_defers_to_F008': {'defer_margin': opt['defer_margin'],
                                                'rule': 'F003 fires -> info if F008 ran, did not flag, and median stored R2 beats the best of transpose/flips by defer_margin'},
                         'advection_probe': {'adv_gain': opt['adv_gain'], 'adv_margin': opt['adv_margin'], 'adv_samples': opt['adv_samples'],
                                             'rule': 'F002/F003/F008/F009 skip when median(R2 with u_x,u_y terms - R2 stored) >= adv_gain and median drifted R2 >= best re-oriented plain R2 + adv_margin'},
                         'config_declaration': 'operator: diffusion | advective | other'},
           'fixed_in_code': {'F003_fire_fractions': [0.1, 0.5], 'F008_fire_fractions': [0.1, 0.5],
                             'F009_fractions': [opt['pair_warn'], opt['pair_error']]},
           'note': 'tuned on everything seen, including the v3 sets and the v3 control failures; no dataset is held out for v4. v2 and v3 frozen files and results are unchanged.'}
    (rb.RESULTS / 'frozen_thresholds_v4.json').write_text(json.dumps(out, indent=1))
    print('frozen v4 at', out['frozen_at'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--list', action='store_true')
    g.add_argument('--run')
    g.add_argument('--run-group', choices=list(GROUPS))
    g.add_argument('--freeze', action='store_true')
    g.add_argument('--report', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', help='comma separated name substrings (with --run-group)')
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)                                            # noqa: E731
    if a.list:
        for s in ALL:
            print(f'{s[1]:11s} {s[0]:28s} {"[exists]" if s[2].exists() else "[MISSING]"}')
        return 0
    if a.freeze:
        freeze()
        return 0
    if a.report:
        report()
        return 0
    specs = [s for s in ALL if s[0] == a.run] if a.run else [s for s in GROUPS[a.run_group] if not a.only or any(t in s[0] for t in a.only.split(','))]
    if not specs:
        sys.exit('no such run')
    for s in specs:
        run_one(s, log, a.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
