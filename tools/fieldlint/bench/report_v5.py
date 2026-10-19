"""Mechanical report for the v5 held-out measurement (protocol_v5.md sections 6 and 10). Reads bench/results/v5_heldout/*.json and applies NO thresholds of
its own except the section-10 criteria below, which equal the numbers in protocol_v5.md (edit both together, before freezing).

  python tools/fieldlint/bench/report_v5.py [--dir bench/results/v5_heldout]      # markdown to stdout

C2 (adjudication of clean findings) and the prediction audit are done by hand and are not computed here.
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
C5_EXACT = ['dup_x1', 'dup_x5', 'cross_dup', 'dup_within5']           # cross_dup = the standard 10% fault (dup_x10)
C5_NOISE_BIND = 'dupnoise_1e-7'
C5_NOISE_REPORT = ['dupnoise_1e-4', 'dupnoise_1e-2']
C6_FAULTS = ['shift_index', 'shift_index_5', 'permute20', 'shuffle_all']
# in-scope BINDING sets classed native grid (protocol_v5.md section 4)
NATIVE = ['condiff_cubic1.0', 'condiff_cubic2.0', 'condiff_exponential0.4', 'condiff_gaussian0.4', 'condiff_cubic0.1', 'condiff_exponential1.0']
CRIT = {'C1_min_rate': 0.80, 'C2_fp_per_10': 1, 'C4_min_fraction': 0.50, 'C5_exact_min': 0.95, 'C5_noise_min': 0.90, 'C6_shift_min': 0.70}   # = protocol_v5.md section 10


def family(name: str) -> str:
    return {'condiff': 'ConDiff (FV, random k)', 'chip': 'self-generated chip FEM (Q1)', 'sissa': 'SISSA thermal (rasterised mesh)'}.get(name.split('_')[0], name)


def binom_tail_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_lower(k: int, n: int, alpha: float = 0.05) -> float:
    if n == 0 or k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        lo, hi = (lo, p) if binom_tail_ge(k, n, p) > alpha else (p, hi)
    return hi


def clopper_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """exact one-sided upper bound (e.g. 0 of 6 controls -> 0.39)."""
    if n == 0:
        return float('nan')
    if k == n:
        return 1.0
    a, b = 0.0, 1.0
    for _ in range(60):
        p = (a + b) / 2
        cdf = 1 - binom_tail_ge(k + 1, n, p)
        a, b = (p, b) if cdf > alpha else (a, p)
    return a


def clopper_two_sided(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return float('nan'), float('nan')
    lo = clopper_lower(k, n, alpha / 2)
    hi = 1.0 if k == n else clopper_upper(k, n, alpha / 2)
    return lo, hi


def fmt_ci(k, n):
    if n == 0:
        return 'n/a'
    lo, hi = clopper_two_sided(k, n)
    return f'{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]'


def load(d: Path):
    out = {}
    for f in sorted(d.glob('*.json')):
        j = json.loads(f.read_text())
        for r in j['records']:
            out[r['name']] = (j['role'], r)
    return out


def hits(rec, fault):
    res = rec['faults'].get(fault)
    if res is None or 'skipped' in res:
        return None
    plain = {c for c in RULES if res[c]['detected']}
    clean = {c for c in RULES if rec['clean'][c]['detected']}
    return plain, plain - clean


def f004_count(x):
    m = x.get('metrics', {})
    return m.get('cross_split', 0) + m.get('within_split', 0)


def c5_detected(rec, fault):
    """flagged on the faulted copy AND more duplicate samples than on the clean copy (protocol_v5.md section 10, C5)."""
    res = rec['faults'].get('cross_dup') if fault == 'cross_dup' else rec['c5'].get(fault)
    if res is None or 'skipped' in res:
        return None
    f4 = res['F004'] if fault == 'cross_dup' else res
    return bool(f4['detected'] and f004_count(f4) > f004_count(rec['clean']['F004']))


def c6_cell(rec, fault, arm):
    """(detected clean-adjusted, skipped, clean_flagged) for an F009 arm."""
    x, c = rec['c6'][fault][arm], rec['c6']['clean'][arm]
    return bool(x['detected'] and not c['detected']), x['status'] == 'skipped', bool(c['detected'])


def section_inscope(L, recs, title, binding):
    N = len(recs)
    L += [f'## {title}', '', f'N = {N}: ' + ', '.join(recs), '']
    return N


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(HERE / 'results' / 'v5_heldout'))
    a = ap.parse_args(argv)
    runs = load(Path(a.dir))
    by = lambda role: {k: r for k, (ro, r) in runs.items() if ro == role}                     # noqa: E731
    ins, nb, dia, ctl, ctl_nb = by('inscope'), by('inscope_nb'), by('diagnostic'), by('control'), by('control_nb')
    N = len(ins)
    L = ['# v5 held-out results (mechanical)', '',
         f'N (binding in-scope datasets that loaded and ran) = {N}; allowed false-positive datasets after adjudication (C2) = {N * CRIT["C2_fp_per_10"] // 10}', '']

    allin = {**ins, **nb, **dia}
    L += ['## Clean findings and skips (every in-scope, non-binding in-scope and diagnostic dataset)', '', '| dataset | role | rules flagged on clean | skipped on clean |', '|---|---|---|---|']
    fp_sets = []
    for k, r in allin.items():
        fl = [c for c in RULES if r['clean'][c]['detected']]
        sk = [c for c in RULES if r['clean'][c]['status'] == 'skipped']
        role = runs[k][0]
        L.append(f'| {k} | {role} | {", ".join(fl) or "none"} | {", ".join(sk) or "none"} |')
        if fl and k in ins:
            fp_sets.append(k)
    L += ['', f'binding in-scope datasets with any clean finding before adjudication: {len(fp_sets)} of {N} ({", ".join(fp_sets) or "none"})', '']
    L += ['## Clean status per rule', '', '| dataset | ' + ' | '.join(RULES) + ' |', '|---|' + '---|' * len(RULES)]
    for k, r in allin.items():
        L.append(f'| {k} | ' + ' | '.join(f'{r["clean"][c]["status"]}' + (f' ({r["clean"][c]["severity"]})' if r['clean'][c]['severity'] != 'none' else '') for c in RULES) + ' |')
    L.append('')

    # ── C1 ──
    def c1_block(pool, label):
        tot_adj = tot_n = 0
        rows = ['| fault | v2-count (plain) | clean-adjusted | trials |', '|---|---|---|---|']
        for f in WHOLE:
            p = q = t = 0
            for r in pool.values():
                h = hits(r, f)
                if h is None:
                    continue
                t += 1
                p += bool(h[0])
                q += bool(h[1])
            rows.append(f'| {f} | {p} | {q} | {t} |')
            tot_adj += q
            tot_n += t
        clu_hit = sum(1 for r in pool.values() if all(hits(r, f) is None or hits(r, f)[1] for f in WHOLE))
        return tot_adj, tot_n, rows, clu_hit
    q, t, rows, clu = c1_block(ins, 'binding')
    rate = q / t if t else float('nan')
    c1 = bool(t) and rate >= CRIT['C1_min_rate']
    L += ['## C1: whole-dataset fault detection, any rule (binding in-scope)', ''] + rows
    L += ['', f'C1: clean-adjusted any-rule detection {fmt_ci(q, t)} (two-sided exact 95%); one-sided 95% lower bound {clopper_lower(q, t):.2f}; threshold {CRIT["C1_min_rate"]}: '
          f'**{"PASS" if c1 else "FAIL"}**',
          f'(dataset-cluster view, optimistic pooled trials aside: {clu} of {N} datasets had all three faults detected; one-sided lower bound {clopper_lower(clu, N):.2f})', '']
    if nb or dia:
        q2, t2, rows2, _ = c1_block({**nb, **dia}, 'non-binding')
        L += ['### C1 on non-binding in-scope and diagnostic sets (reported only)', ''] + rows2 + ['', f'{fmt_ci(q2, t2)}', '']
    L += ['### Per dataset: whole-dataset faults, rules credited (clean-adjusted)', '', '| dataset | role | ' + ' | '.join(WHOLE) + ' |', '|---|---|' + '---|' * len(WHOLE)]
    for k, r in allin.items():
        cells = []
        for f in WHOLE:
            h = hits(r, f)
            cells.append('n/a' if h is None else (', '.join(sorted(h[1])) or 'MISS'))
        L.append(f'| {k} | {runs[k][0]} | ' + ' | '.join(cells) + ' |')
    L.append('')

    L += ['## Per-fault, per-rule detection (binding in-scope pooled)', '', '| fault | trials | any (plain) | any (adj) | ' + ' | '.join(f'{c} adj' for c in RULES) + ' |', '|---|---|---|---|' + '---|' * len(RULES)]
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

    # ── C4 ──
    L += ['## F008 applicability and C4', '', '| dataset | native binding list | F008 status | median R2 stored | face mean | F009 status |', '|---|---|---|---|---|---|']
    app = 0
    for k, r in allin.items():
        c = r['clean']['F008']
        L.append(f'| {k} | {"yes" if k in NATIVE else "no"} | {c["status"]} | {c["metrics"].get("median_r2_stored", "")} | {c["metrics"].get("face_mean", "")} | {r["clean"]["F009"]["status"]} |')
        app += (k in NATIVE and k in ins) and c['status'] != 'skipped'
    nn = len([n for n in NATIVE if n in ins])
    c4 = nn > 0 and app / nn >= CRIT['C4_min_fraction']
    L += ['', f'C4: F008 applicable on {app}/{nn} native binding in-scope sets; threshold {CRIT["C4_min_fraction"]}: **{"PASS" if c4 else "FAIL"}**', '']

    # ── C3 ──
    L += ['## Controls (declined vs false alarm)', '', '| run | role | binding | ' + ' | '.join(RULES) + ' |', '|---|---|---|' + '---|' * len(RULES)]
    alarms_b, alarms_strict, alarms_nb = [], [], []
    for k, (role, r) in runs.items():
        if role not in ('control', 'control_nb'):
            continue
        base = k.split('+')[0]
        has_flag = (base + '+time_dependent_flag') in runs
        binding = role == 'control' and (k.endswith('+time_dependent_flag') or not has_flag)
        bad = [c for c in ('F002', 'F003', 'F005', 'F008', 'F009') if r['clean'][c]['detected']]
        cells = [r['clean'][c]['status'] + (f' ({r["clean"][c]["severity"]})' if r['clean'][c]['severity'] != 'none' else '') for c in RULES]
        L.append(f'| {k} | {role} | {"yes" if binding else ("non-binding" if role == "control_nb" else "C3-strict only")} | ' + ' | '.join(cells) + ' |')
        if bad:
            (alarms_b if binding else (alarms_nb if role == 'control_nb' else alarms_strict)).append((k, bad))
    nbind = len([1 for k, (ro, _) in runs.items() if ro == 'control' and (k.endswith('+time_dependent_flag') or (k + '+time_dependent_flag') not in runs)])
    c3 = not alarms_b
    L += ['', f'C3 (binding runs = flagged for time-dependent controls, as-is for steady ones; {nbind} runs): false alarms {alarms_b or "none"}: **{"PASS" if c3 else "FAIL"}**',
          f'false-alarm rate {len(alarms_b)}/{nbind}, one-sided 95% upper bound {clopper_upper(len(alarms_b), nbind):.2f} (a statement about these files, not a rate)',
          f'C3-strict, not binding (unflagged time-dependent runs): false alarms {alarms_strict or "none"}',
          f'non-binding generated control: false alarms {alarms_nb or "none"}', '']
    L += ['### C3 graded view (descriptive only; defined in protocol_v5.md section 10 before any run; does not change the verdict)', '',
          'For every binding run: the rules with a warning or error, and their severity. Graded reading = PARTIAL-LIKE if every alarm is a single rule across the binding runs, else FAIL-LIKE.', '']
    graded_rules = set()
    for k, bad in alarms_b:
        L.append(f'* {k}: ' + ', '.join(f'{c} {runs[k][1]["clean"][c]["severity"]}' for c in bad))
        graded_rules.update(bad)
    L.append(f'graded reading: {"no alarm" if not alarms_b else ("single rule (" + ",".join(sorted(graded_rules)) + ")" if len(graded_rules) == 1 else "several rules (" + ",".join(sorted(graded_rules)) + ")")}')
    L += ['', '### Control summary strings (clean)', '']
    for k, (role, r) in runs.items():
        if role in ('control', 'control_nb'):
            L.append(f'* {k}: ' + '; '.join(f'{c} {r["clean"][c]["summary"]}' for c in ('F001', 'F002', 'F003', 'F005', 'F008', 'F009', 'F004')))
    L.append('')

    # ── C5 ──
    L += ['## C5: F004 duplicate detection (binding in-scope)', '',
          'detected = F004 flagged on the faulted copy AND (cross-split + within-split duplicate samples) exceeds the clean copy\'s count.', '',
          '| dataset | ' + ' | '.join(C5_EXACT + [C5_NOISE_BIND] + C5_NOISE_REPORT) + ' |', '|---|' + '---|' * (len(C5_EXACT) + 1 + len(C5_NOISE_REPORT))]
    ex_k = ex_n = 0
    nz_k = nz_n = 0
    per_fault = {f: [0, 0] for f in C5_EXACT + [C5_NOISE_BIND] + C5_NOISE_REPORT}
    for k, r in ins.items():
        cells = []
        for f in C5_EXACT + [C5_NOISE_BIND] + C5_NOISE_REPORT:
            d = c5_detected(r, f)
            cells.append('n/a' if d is None else ('hit' if d else 'MISS'))
            if d is not None:
                per_fault[f][0] += d
                per_fault[f][1] += 1
                if f in C5_EXACT:
                    ex_k += d
                    ex_n += 1
                if f == C5_NOISE_BIND:
                    nz_k += d
                    nz_n += 1
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    c5a = ex_n > 0 and ex_k / ex_n >= CRIT['C5_exact_min']
    c5b = nz_n > 0 and nz_k / nz_n >= CRIT['C5_noise_min']
    clean_f4 = [k for k, r in ins.items() if r['clean']['F004']['detected']]
    L += ['', f'C5a exact copies (dup_x1, dup_x5, dup_x10 = cross_dup, dup_within5): {fmt_ci(ex_k, ex_n)}; one-sided lower bound {clopper_lower(ex_k, ex_n):.2f}; threshold {CRIT["C5_exact_min"]}: **{"PASS" if c5a else "FAIL"}**',
          f'C5b {C5_NOISE_BIND}: {fmt_ci(nz_k, nz_n)}; threshold {CRIT["C5_noise_min"]}: **{"PASS" if c5b else "FAIL"}**',
          f'C5c clean copies with an F004 finding (to be adjudicated by hand, P2): {clean_f4 or "none"}  [pass = 0 after adjudication]',
          'per fault: ' + '; '.join(f'{f} {fmt_ci(*per_fault[f])}' for f in per_fault),
          '(dupnoise_1e-4 and dupnoise_1e-2 are above dup_rtol = 1e-5: a documented non-detection, reported, not a criterion)', '']
    c5_clean_ok = not clean_f4                      # provisional; C5c is adjudicated by hand

    # ── C6 ──
    L += ['## C6: F009 pairing detection', '',
          'Arm B (binding): the lint on a head block of 64 consecutive samples of the faulted dataset (`--head 64`). Arm A: the shipped rule on the whole dataset (it scores the first 64 consecutive samples). '
          'Arm L: legacy v4 sampling (32 evenly spaced samples) emulated exactly. Cell = detected (clean-adjusted) / skipped. F009 is "applicable" on a set when it is not skipped on the clean copy in that arm.', '']
    for arm, lab in (('B_head64', 'Arm B (binding)'), ('A_shipped', 'Arm A (shipped rule, whole dataset; non-binding)'), ('L_spaced32', 'Arm L (legacy v4 sampling; reported)')):
        L += [f'### {lab}', '', '| dataset | F009 clean | ' + ' | '.join(C6_FAULTS) + ' |', '|---|---|' + '---|' * len(C6_FAULTS)]
        agg = {f: [0, 0, 0] for f in C6_FAULTS}                      # detected, applicable, total
        clean_flag = 0
        app_sets = 0
        for k, r in ins.items():
            cl = r['c6']['clean'][arm]
            app_sets += cl['status'] != 'skipped'
            clean_flag += cl['detected']
            cells = []
            for f in C6_FAULTS:
                d, sk, _ = c6_cell(r, f, arm)
                cells.append('hit' if d else ('skip' if sk else 'miss'))
                agg[f][2] += 1
                if cl['status'] != 'skipped':
                    agg[f][1] += 1
                    agg[f][0] += d
            L.append(f'| {k} | {cl["status"]} | ' + ' | '.join(cells) + ' |')
        L += ['', f'applicable sets (F009 not skipped on clean): {app_sets}/{N}; F009 findings on clean copies: {clean_flag}',
              'detection over applicable sets: ' + '; '.join(f'{f} {fmt_ci(agg[f][0], agg[f][1])}' for f in C6_FAULTS), '']
        if arm == 'B_head64':
            k_, n_ = agg['shift_index'][0], agg['shift_index'][1]
            c6a = n_ > 0 and k_ / n_ >= CRIT['C6_shift_min']
            c6b = clean_flag == 0
            c6 = c6a and c6b
            L += [f'C6a (Arm B shift_index, clean-adjusted, over applicable sets): {fmt_ci(k_, n_)}; one-sided lower bound {clopper_lower(k_, n_):.2f}; threshold {CRIT["C6_shift_min"]}: **{"PASS" if c6a else "FAIL"}**',
                  f'C6b (F009 findings on clean copies, Arm B): {clean_flag}; threshold 0: **{"PASS" if c6b else "FAIL"}**',
                  f'skip rate on clean copies: {N - app_sets}/{N}', '']
    # ── C7 ──
    L += ['## C7: advection probe', '']
    trig = []
    for k, (role, r) in runs.items():
        if role not in ('inscope',):
            continue
        for c in ('F002', 'F003', 'F008', 'F009'):
            for tag, res in [('clean', r['clean'])] + [(f, v) for f, v in r['faults'].items() if f in WHOLE and 'skipped' not in v]:
                if 'out of scope: advective' in res[c]['summary']:
                    trig.append((k, c, tag))
    n_cond = sum(1 + len([f for f in WHOLE if 'skipped' not in r['faults'].get(f, {'skipped': 1})]) for r in ins.values())
    L += [f'C7a specificity: advection probe triggered on {len(trig)} of {n_cond} (binding in-scope clean + whole-dataset transpose/flip) conditions: {trig or "none"}; required 0: **{"PASS" if not trig else "FAIL"}**', '']
    for k, (role, r) in runs.items():
        if role in ('control', 'control_nb') and ('flowbench' in k or 'advdiff' in k):
            decl = [c for c in ('F002', 'F003', 'F008', 'F009') if 'out of scope: advective' in r['clean'][c]['summary']]
            L.append(f'* C7b sensitivity {k} ({role}): rules declining as advective: {decl or "none"}; ' + '; '.join(f'{c} {r["clean"][c]["summary"][:160]}' for c in ('F002', 'F008')))
    for k, (role, r) in runs.items():
        for c in ('F002', 'F003', 'F008', 'F009'):
            if 'confirms the stored orientation' in r['clean'][c]['summary']:
                L.append(f'* F003 deferral reported on {k} ({role}) clean: {r["clean"][c]["summary"][:160]}')
    L.append('')

    # ── verdict ──
    L += ['## Mechanical verdict (protocol_v5.md section 10)', '',
          f'C1 {"PASS" if c1 else "FAIL"}; C2 by hand (provisional listing above); C3 {"PASS" if c3 else "FAIL"}; C4 {"PASS" if c4 else "FAIL"}; '
          f'C5 {"PASS" if (c5a and c5b and c5_clean_ok) else "FAIL"} (a {"PASS" if c5a else "FAIL"}, b {"PASS" if c5b else "FAIL"}, c provisional); C6 {"PASS" if c6 else "FAIL"}; C7a {"PASS" if not trig else "FAIL"}', '',
          'PAPER-WORTHY: C1-C6 all hold. PARTIAL: C1 and C3 hold but C2, C4, C5 or C6 does not. NOT PAPER-WORTHY: C1 or C3 fails. (C7 is reported, not part of the verdict.)']
    print('\n'.join(L))


if __name__ == '__main__':
    main()
