"""Mechanical report for the v4 held-out measurement (protocol_v4.md sections 6 and 10). Reads bench/results/v4_heldout/*.json and applies NO
thresholds of its own except the section-10 criteria below, which equal the numbers in protocol_v4.md (edit both together, before freezing).

  python tools/fieldlint/bench/report_v4.py [--dir bench/results/v4_heldout]

Output: markdown to stdout. Adjudication of clean findings (protocol section 7) and the prediction audit are done by hand and are not computed here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES = ['F001', 'F002', 'F003', 'F004', 'F005', 'F008', 'F009']
WHOLE = ['transpose', 'flip_x', 'flip_y']
FAULTS = ['transpose', 'flip_x', 'flip_y', 'roll1', 'roll2', 'roll3', 'shift_index', 'unit_slip', 'cross_dup',
          'transpose@20', 'flip_x@20', 'flip_y@20', 'roll1@20', 'shift_index@20']
# in-scope sets classed NATIVE grid (protocol_v4.md section 10, from datasets_v4.md); BINDING for C4
NATIVE = ['condiff_poisson', 'condiff_gaussian0.1', 'condiff_gaussian1.0', 'condiff_cubic0.4', 'condiff_exponential2.0', 'elado_b1_test']
NATIVE_ALL = NATIVE + ['pdebench_darcy_beta0.1', 'pdebench_darcy_beta10.0', 'pdebench_darcy_beta100.0']       # sensitivity only, not binding
CRIT = {'C1_min_rate': 0.80, 'C2_fp_per_10': 1, 'C4_min_fraction': 0.50}          # = protocol_v4.md section 10
FAMILY = {'condiff': 'ConDiff (FV, random k)', 'elado': 'ELADO B1 (FEM, random a)', 'pdebench': 'PDEBench Darcy (same generator as tuning)'}


def family(name: str) -> str:
    return FAMILY[name.split('_')[0]]


def binom_tail_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """exact one-sided lower bound of a binomial proportion (bisection on the binomial tail)."""
    if n == 0 or k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        lo, hi = (lo, p) if binom_tail_ge(k, n, p) > alpha else (p, hi)
    return hi


def clopper_two_sided(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return float('nan'), float('nan')
    lo = clopper_lower(k, n, alpha / 2)
    if k == n:
        return lo, 1.0
    a, b = 0.0, 1.0                                                  # upper: P(X <= k) = alpha/2
    for _ in range(60):
        p = (a + b) / 2
        cdf = 1 - binom_tail_ge(k + 1, n, p)
        a, b = (p, b) if cdf > alpha / 2 else (a, p)
    return lo, a


def load(d: Path):
    out = {}
    for f in sorted(d.glob('*.json')):
        j = json.loads(f.read_text())
        for r in j['records']:
            out[r['name']] = (j['role'], r)
    return out


def hits(rec, fault):
    """(plain set, clean-adjusted set) of rules credited with `fault`; None if the fault was not applicable."""
    res = rec['faults'].get(fault)
    if res is None or 'skipped' in res:
        return None
    plain = {c for c in RULES if res[c]['detected']}
    clean = {c for c in RULES if rec['clean'][c]['detected']}
    return plain, plain - clean


def fmt_ci(k, n):
    if n == 0:
        return 'n/a'
    lo, hi = clopper_two_sided(k, n)
    return f'{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(HERE / 'results' / 'v4_heldout'))
    a = ap.parse_args(argv)
    runs = load(Path(a.dir))
    ins = {k: r for k, (role, r) in runs.items() if role == 'inscope'}
    dia = {k: r for k, (role, r) in runs.items() if role == 'diagnostic'}
    ctl = {k: r for k, (role, r) in runs.items() if role == 'control'}
    N = len(ins)
    L = ['# v4 held-out results (mechanical)', '', f'N (in-scope datasets that loaded and ran) = {N}; allowed false-positive datasets after adjudication (C2) = {N * CRIT["C2_fp_per_10"] // 10}', '']

    L += ['## In-scope and diagnostic: clean findings (to be adjudicated by hand, protocol section 7)', '', '| dataset | role | rules flagged on clean | skipped on clean |', '|---|---|---|---|']
    fp_sets = []
    for k, r in {**ins, **dia}.items():
        fl = [c for c in RULES if r['clean'][c]['detected']]
        sk = [c for c in RULES if r['clean'][c]['status'] == 'skipped']
        role = 'inscope' if k in ins else 'diagnostic'
        L.append(f'| {k} | {role} | {", ".join(fl) or "none"} | {", ".join(sk) or "none"} |')
        if fl and k in ins:
            fp_sets.append(k)
    L += ['', f'in-scope datasets with any clean finding before adjudication: {len(fp_sets)} of {N} ({", ".join(fp_sets) or "none"})', '']

    L += ['## Clean summary strings (every rule, every in-scope / diagnostic dataset)', '', '| dataset | ' + ' | '.join(RULES) + ' |', '|---|' + '---|' * len(RULES)]
    for k, r in {**ins, **dia}.items():
        L.append(f'| {k} | ' + ' | '.join(f'{r["clean"][c]["status"]}' + (f' ({r["clean"][c]["severity"]})' if r['clean'][c]['severity'] != 'none' else '') for c in RULES) + ' |')
    L.append('')

    L += ['## C1: whole-dataset fault detection, any rule (in-scope only)', '', '| fault | v2-count (plain) | clean-adjusted | trials |', '|---|---|---|---|']
    tot_adj = tot_n = 0
    for f in WHOLE:
        p = q = t = 0
        for r in ins.values():
            h = hits(r, f)
            if h is None:
                continue
            t += 1
            p += bool(h[0])
            q += bool(h[1])
        L.append(f'| {f} | {p} | {q} | {t} |')
        tot_adj += q
        tot_n += t
    rate = tot_adj / tot_n if tot_n else float('nan')
    c1 = bool(tot_n) and rate >= CRIT['C1_min_rate']
    lo = clopper_lower(tot_adj, tot_n)
    L += ['', f'C1: clean-adjusted any-rule detection {fmt_ci(tot_adj, tot_n)} (two-sided exact 95%); one-sided 95% lower bound {lo:.2f}; threshold {CRIT["C1_min_rate"]}: **{"PASS" if c1 else "FAIL"}**', '']

    L += ['### Per-family C1 (clean-adjusted, same three faults)', '', '| family | hits/trials (exact 95% CI) |', '|---|---|']
    for fam in FAMILY.values():
        q = t = 0
        for k, r in ins.items():
            if family(k) != fam:
                continue
            for f in WHOLE:
                h = hits(r, f)
                if h is not None:
                    t += 1
                    q += bool(h[1])
        L.append(f'| {fam} | {fmt_ci(q, t)} |')
    L += ['', '### Per dataset: whole-dataset faults, rules credited (clean-adjusted)', '', '| dataset | ' + ' | '.join(WHOLE) + ' |', '|---|' + '---|' * len(WHOLE)]
    for k, r in ins.items():
        cells = []
        for f in WHOLE:
            h = hits(r, f)
            cells.append('n/a' if h is None else (', '.join(sorted(h[1])) or 'MISS'))
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    L.append('')

    L += ['## Per-fault, per-rule detection (in-scope pooled; plain = flagged on faulted copy; adj = also not flagged on clean)', '',
          '| fault | trials | any (plain) | any (adj) | ' + ' | '.join(f'{c} adj' for c in RULES) + ' |', '|---|---|---|---|' + '---|' * len(RULES)]
    for f in FAULTS:
        t = ap_ = aa = 0
        per = {c: 0 for c in RULES}
        for r in ins.values():
            h = hits(r, f)
            if h is None:
                continue
            t += 1
            ap_ += bool(h[0])
            aa += bool(h[1])
            for c in h[1]:
                per[c] += 1
        L.append(f'| {f} | {t} | {ap_} | {aa} | ' + ' | '.join(str(per[c]) for c in RULES) + ' |')
    L.append('')

    L += ['## F008 applicability and C4', '', '| dataset | native (binding list) | F008 status | median R2 stored | face mean | F009 status |', '|---|---|---|---|---|---|']
    app = app_all = 0
    for k, r in ins.items():
        c = r['clean']['F008']
        nat = k in NATIVE
        L.append(f'| {k} | {"yes" if nat else "no"} | {c["status"]} | {c["metrics"].get("median_r2_stored", "")} | {c["metrics"].get("face_mean", "")} | {r["clean"]["F009"]["status"]} |')
        app += nat and c['status'] != 'skipped'
        app_all += (k in NATIVE_ALL) and c['status'] != 'skipped'
    nn = len([n for n in NATIVE if n in ins])
    c4 = nn > 0 and app / nn >= CRIT['C4_min_fraction']
    L += ['', f'C4: F008 applicable on {app}/{nn} native in-scope sets; threshold {CRIT["C4_min_fraction"]}: **{"PASS" if c4 else "FAIL"}**',
          f'(sensitivity, not binding: {app_all}/{len([n for n in NATIVE_ALL if n in ins])} if PDEBench Darcy is also counted native)', '']

    L += ['## Controls (declined vs false alarm)', '', '| run | binding | ' + ' | '.join(RULES) + ' |', '|---|---|' + '---|' * len(RULES)]
    alarms_flagged, alarms_strict = [], []
    names = list(ctl)
    for k, r in ctl.items():
        base = k.split('+')[0]
        has_flag = (base + '+time_dependent_flag') in ctl
        binding = (k.endswith('+time_dependent_flag')) or not has_flag
        bad = [c for c in ('F002', 'F003', 'F005', 'F008', 'F009') if r['clean'][c]['detected']]
        cells = [r['clean'][c]['status'] + (f' ({r["clean"][c]["severity"]})' if r['clean'][c]['severity'] != 'none' else '') for c in RULES]
        L.append(f'| {k} | {"yes" if binding else "C3-strict only"} | ' + ' | '.join(cells) + ' |')
        if bad:
            (alarms_flagged if binding else alarms_strict).append((k, bad))
    c3 = not alarms_flagged
    L += ['', f'C3 (binding runs = flagged for time-dependent controls, as-is for steady ones): false alarms {alarms_flagged or "none"}: **{"PASS" if c3 else "FAIL"}**',
          f'C3-strict, not binding (unflagged time-dependent runs): false alarms {alarms_strict or "none"}', '']

    L += ['## Control summary strings (clean)', '']
    for k, r in ctl.items():
        L.append(f'* {k}: ' + '; '.join(f'{c} {r["clean"][c]["summary"]}' for c in ('F001', 'F002', 'F003', 'F005', 'F008', 'F009', 'F004')))
    L.append('')

    # F003 -> F008 deferral and advection probe: where do the summaries mention them (clean runs, every dataset)
    L += ['## F003 deferral and advection probe: summaries that report them (clean runs and every faulted copy)', '']
    n_hit = 0
    for k, (role, r) in runs.items():
        for c in ('F002', 'F003', 'F008', 'F009'):
            for tag, res in [('clean', r['clean'])] + [(f, v) for f, v in r['faults'].items() if 'skipped' not in v]:
                s = res[c]['summary']
                if 'confirms the stored orientation' in s or 'out of scope: advective' in s:
                    L.append(f'* {k} [{role}] {c} ({tag}): {s}')
                    n_hit += 1
    L += [f'number of summaries reporting the deferral or the advection probe: {n_hit}', '']
    L += ['C2 is judged by hand after adjudication (count the in-scope datasets above that remain false positives).', '',
          'Verdict rule (protocol section 10): all of C1-C4 -> PAPER-WORTHY; C1 and C3 but not C2/C4 -> PARTIAL; C1 or C3 fail -> NOT PAPER-WORTHY.']
    print('\n'.join(L))


if __name__ == '__main__':
    main()
