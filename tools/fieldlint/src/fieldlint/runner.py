"""Run rules over a Dataset and build / render the report."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .model import Dataset
from .rules import RULES, Finding, Options, RuleResult

SEVERITY_RANK = {'info': 0, 'warning': 1, 'error': 2}
EXIT_CODES = {'none': 0, 'info': 0, 'warning': 1, 'error': 2}


@dataclass
class Report:
    dataset: str
    n_samples: int
    results: List[RuleResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def worst(self) -> str:
        sev = [f.severity for r in self.results for f in r.findings]
        return max(sev, key=SEVERITY_RANK.get) if sev else 'none'

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.worst]

    def to_dict(self):
        return {'dataset': self.dataset, 'n_samples': self.n_samples, 'worst_severity': self.worst,
                'exit_code': self.exit_code, 'seconds': round(self.seconds, 2),
                'rules': [r.to_dict() for r in self.results]}

    def text(self) -> str:
        out = [f'fieldlint: {self.dataset} ({self.n_samples} samples)']
        for r in self.results:
            tag = {'ok': 'ok     ', 'flagged': 'FLAGGED', 'skipped': 'skipped'}[r.status]
            out.append(f'  {r.code} {r.name:<16} {tag}  {r.summary}')
            for f in r.findings:
                ids = f'  [e.g. {", ".join(f.samples[:3])}]' if f.samples else ''
                out.append(f'      {f.severity.upper():<7} {f.message}{ids}')
        c = {s: sum(f.severity == s for r in self.results for f in r.findings) for s in ('error', 'warning', 'info')}
        out.append(f'  result: {c["error"]} error(s), {c["warning"]} warning(s), {c["info"]} info; exit code {self.exit_code}')
        return '\n'.join(out)


def parse_rules(spec: Optional[str]) -> List[str]:
    if not spec:
        return list(RULES)
    codes = [c.strip().upper() for c in spec.split(',') if c.strip()]
    bad = [c for c in codes if c not in RULES]
    if bad:
        raise ValueError(f'unknown rule(s) {bad}; available: {sorted(RULES)}')
    return codes


def run(ds: Dataset, rules: Optional[Iterable[str]] = None, opt: Optional[Options] = None) -> Report:
    """Run the selected rules (default all). F001 runs first whenever selected; if it finds a shape/grid error the
    field-dependent rules are skipped (they would index mismatched arrays)."""
    opt = opt or Options()
    codes = list(rules) if rules else list(RULES)
    codes = sorted(codes, key=lambda c: (c != 'F001', c))
    rep = Report(ds.name, len(ds))
    t0 = time.time()
    fatal = False
    for c in codes:
        if fatal and c != 'F001':
            rep.results.append(RuleResult(c, RULES[c].__name__[5:].replace('_', '-'), 'skipped', summary='skipped: F001 found fatal shape errors'))
            continue
        try:
            r = RULES[c](ds, opt)
        except Exception as exc:                                    # noqa: BLE001 - a rule bug must not hide other rules
            r = RuleResult(c, 'crashed', 'flagged', [Finding(c, 'error', f'rule crashed: {type(exc).__name__}: {exc}')],
                           summary='rule crashed')
        rep.results.append(r)
        if c == 'F001' and any(f.severity == 'error' and 'shape' in f.message for f in r.findings):
            fatal = True
    rep.seconds = time.time() - t0
    return rep
