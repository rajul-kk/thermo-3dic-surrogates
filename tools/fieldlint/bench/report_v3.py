"""Mechanical report for the v3 measurement (protocol_v3.md sections 6 and 10). Reads bench/results/v3/*.json, applies NO thresholds of its
own except the draft criteria below, which must equal the numbers in protocol_v3.md section 10 (edit both together, before freezing).

  python tools/fieldlint/bench/report_v3.py [--dir bench/results/v3] [--native a,b,c]

`--native` lists the in-scope run names classed native in datasets_v3.md (default below). Output: markdown to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES = ['F001', 'F002', 'F003', 'F004', 'F005', 'F008', 'F009']
WHOLE = ['transpose', 'flip_x', 'flip_y']
NATIVE_DEFAULT = ['pdegym_poisson_gauss', 'cno_poisson', 'elado_a1_test', 'dpde_poisson_denorm', 'ucsd_poisson']
CRIT = {'C1_min_rate': 0.80, 'C2_fp_per_10': 1, 'C4_min_fraction': 0.50}          # = protocol_v3.md section 10


def clopper_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """exact one-sided lower bound of a binomial proportion (bisection on the binomial tail)."""
    if n == 0 or k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        tail = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
        lo, hi = (lo, p) if tail > alpha else (p, hi)
    return hi


def load(d: Path):
    out = {}
    for f in sorted(d.glob('*.json')):
        j = json.loads(f.read_text())
        for r in j['records']:
            out[r['name']] = (j['role'], r)
    return out


def adjusted_hits(rec, fault):
    """rules credited with `fault` under the clean-adjusted rule; also the plain (v2) set."""
    res = rec['faults'].get(fault)
    if res is None or 'skipped' in res:
        return None
    plain = {c for c in RULES if res[c]['detected']}
    clean = {c for c in RULES if rec['clean'][c]['detected']}
    return plain, plain - clean


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(HERE / 'results' / 'v3'))
    ap.add_argument('--native', default=','.join(NATIVE_DEFAULT))
    a = ap.parse_args(argv)
    runs = load(Path(a.dir))
    ins = {k: r for k, (role, r) in runs.items() if role == 'inscope'}
    ctl = {k: r for k, (role, r) in runs.items() if role == 'control'}
    native = [n for n in a.native.split(',') if n in ins]
    L = ['# v3 results (mechanical)', '']

    L += ['## In-scope: clean findings (to be adjudicated by hand, protocol section 7)', '', '| dataset | rules flagged on clean | skipped on clean |', '|---|---|---|']
    fp_sets = []
    for k, r in ins.items():
        fl = [c for c in RULES if r['clean'][c]['detected']]
        sk = [c for c in RULES if r['clean'][c]['status'] == 'skipped']
        L.append(f'| {k} | {", ".join(fl) or "none"} | {", ".join(sk) or "none"} |')
        if fl:
            fp_sets.append(k)
    n = len(ins)
    L += ['', f'datasets with any clean finding before adjudication: {len(fp_sets)} of {n} ({", ".join(fp_sets) or "none"}); allowed false positives after adjudication: {n * CRIT["C2_fp_per_10"] // 10}', '']

    L += ['## In-scope: whole-dataset fault detection, any rule', '', '| fault | v2 count | clean-adjusted count | trials |', '|---|---|---|---|']
    tot_adj = tot_n = 0
    for f in WHOLE:
        p = q = t = 0
        for r in ins.values():
            h = adjusted_hits(r, f)
            if h is None:
                continue
            t += 1
            p += bool(h[0])
            q += bool(h[1])
        L.append(f'| {f} | {p} | {q} | {t} |')
        tot_adj += q
        tot_n += t
    rate = tot_adj / tot_n if tot_n else float('nan')
    c1 = rate >= CRIT['C1_min_rate']
    L += ['', f'C1: clean-adjusted any-rule detection {tot_adj}/{tot_n} = {rate:.2f} (exact one-sided 95% lower bound {clopper_lower(tot_adj, tot_n):.2f}); threshold {CRIT["C1_min_rate"]}: **{"PASS" if c1 else "FAIL"}**', '']

    L += ['## F008 applicability (clean, in scope)', '', '| dataset | native | F008 status | median R2 stored | face mean |', '|---|---|---|---|---|']
    app = 0
    for k, r in ins.items():
        c = r['clean']['F008']
        nat = k in native
        L.append(f'| {k} | {"yes" if nat else "no"} | {c["status"]} | {c["metrics"].get("median_r2_stored", "")} | {c["metrics"].get("face_mean", "")} |')
        app += nat and c['status'] != 'skipped'
    c4 = bool(native) and app / len(native) >= CRIT['C4_min_fraction']
    L += ['', f'C4: F008 applicable on {app}/{len(native)} native in-scope sets; threshold {CRIT["C4_min_fraction"]}: **{"PASS" if c4 else "FAIL"}**', '']

    L += ['## Controls (declined vs false alarm; F002 F003 F005 F008 F009; F004 and F001 listed separately)', '', '| run | ' + ' | '.join(RULES) + ' |', '|---|' + '---|' * len(RULES)]
    alarms_flagged = []
    for k, r in ctl.items():
        cells = []
        bad = [c for c in ('F002', 'F003', 'F005', 'F008', 'F009') if r['clean'][c]['detected']]
        for c in RULES:
            x = r['clean'][c]
            cells.append(x['status'] + (f' ({x["severity"]})' if x['severity'] != 'none' else ''))
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')
        if bad and (k.endswith('+time_dependent_flag') or k.split('+')[0] in {n for n in ctl if not n.endswith('+time_dependent_flag') and (n + '+time_dependent_flag') not in ctl}):
            alarms_flagged.append((k, bad))
    c3 = not alarms_flagged
    L += ['', f'C3 (binding runs = flagged for time-dependent controls, as-is for steady ones): false alarms {alarms_flagged or "none"}: **{"PASS" if c3 else "FAIL"}**', '']
    L += [f'C2 is judged by hand after adjudication (count the datasets above that remain false positives).', '',
          'Verdict rule (protocol section 10): all of C1-C4 -> PAPER-WORTHY; C1 and C3 but not C2/C4 -> PARTIAL; C1 or C3 fail -> NOT PAPER-WORTHY.']
    print('\n'.join(L))


if __name__ == '__main__':
    main()
