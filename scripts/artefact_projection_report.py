"""Print the artefact-projection beta table and decision per (architecture, geometry) from the kernel's projection_summary.json.

Usage: python scripts/artefact_projection_report.py path/to/projection_summary.json [--field beta|corr|beta_s|corr_s]
Table: seed-mean beta (equal to the seed-average of per-seed betas), the same-label nulls, and the per-seed single-seed decisions
(O_k - N_k vs the matched N_a - N_b); the headline decision needs all seeds to agree.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.validation import artefact_projection as ap                    # noqa: E402


def fmt(c):
    return f"{c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]"


def report(summary, field='beta'):
    lines = [f"field = {field}   (beta = projection slope of O-N on the true artefact D = T_old - T_new; 1 = reproduced, 0 = not)",
             'decision rule: ' + summary.get('decision_rule', ap.RULE), '']
    meta = summary.get('meta') or {}
    if meta:
        lines += ['meta: ' + json.dumps(meta), '']
    head = f"{'arch':<13}{'geometry':<10}{'n':>3}  {'seed-mean beta [CI]':<24}{'null N-N':<24}{'null O-O':<24}{'per-seed decisions':<34}{'p vs NN':>9}  decision"
    lines += [head, '-' * len(head)]
    for key, e in sorted(summary['results'].items()):
        arch, g = key.split('|')
        a = e['analysis'][field]
        lines.append(f"{arch:<13}{g:<10}{a['n_scenarios']:>3}  {fmt(a['mean']):<24}{fmt(a['null_nn']):<24}{fmt(a['null_oo']):<24}"
                     f"{','.join(s['decision'][:3] + format(s['mean'], '+.2f') for s in a['single_seed']):<34}"
                     f"{a['mean']['wilcoxon_p_vs_null_nn']:>9.2g}  {a['decision']}"
                     f"  (seed-mean vs null: {a['mean']['decision']}; folds {len(e['folds_done'])}/5)")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('summary')
    p.add_argument('--field', default='all', choices=['all', *ap.FIELDS])
    a = p.parse_args()
    s = json.loads(Path(a.summary).read_text())
    for f in (ap.FIELDS if a.field == 'all' else [a.field]):
        print(report(s, f), end='\n\n')


if __name__ == '__main__':
    main()
