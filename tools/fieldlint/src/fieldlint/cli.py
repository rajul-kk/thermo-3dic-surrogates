"""Command line: fieldlint PATH [--preset NAME | --config FILE | --u KEY --source KEY --k KEY --layout NHW] [--rules F001,F003]

Exit codes: 0 clean (info only), 1 warnings, 2 errors, 3 usage / load failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .adapters import AdapterError, list_presets, load_config, open_dataset
from .rules import RULES, Options
from .runner import parse_rules, run


def _field(cfg, name, arg, layout, channel, default_layout):
    """CLI shorthand KEY[@FILE] -> field spec."""
    if arg is None:
        return
    key, _, file = arg.partition('@')
    spec = dict(cfg.get(name) or {}) if isinstance(cfg.get(name), dict) else {}
    spec['key'] = key or None
    if file:
        spec['file'] = file
    if layout or default_layout:
        spec['layout'] = layout or default_layout
    if channel is not None:
        spec['channel'] = channel
    cfg[name] = spec


def build_parser():
    ap = argparse.ArgumentParser(prog='fieldlint', description='Physics-aware linter for PDE-surrogate benchmark datasets.')
    ap.add_argument('path', nargs='?', help='dataset file or directory')
    ap.add_argument('--preset', help=f"adapter preset: {', '.join(list_presets())}")
    ap.add_argument('--config', help='adapter config file (.json / .yaml)')
    ap.add_argument('--format', help='auto | npz | npy | mat | h5 | 3dice')
    ap.add_argument('--u', help='output-field key, KEY or KEY@FILE')
    ap.add_argument('--source', help='source-field key')
    ap.add_argument('--k', help='coefficient (conductivity) field key')
    ap.add_argument('--dirichlet', help='boolean mask key (True = u held fixed)')
    ap.add_argument('--flux-out', help='per-sample total boundary heat flux key (layout N)')
    ap.add_argument('--layout', help='axes of the arrays as stored, e.g. NHW, NZHW, NCZYX (default for every field)')
    ap.add_argument('--source-layout'), ap.add_argument('--k-layout')
    ap.add_argument('--source-channel', type=int), ap.add_argument('--k-channel', type=int)
    ap.add_argument('--units', help="units of u: K, C, or '' (rise / unknown)")
    ap.add_argument('--ambient', type=float, help='boundary / ambient level of u')
    ap.add_argument('--spacing', help='grid spacing per spatial axis, comma separated (same order as the axes of u)')
    ap.add_argument('--splits', help="none | from_filename | ictherm | train=0:0.8,test=0.8:1 (fractions of sample order)")
    ap.add_argument('--rules', help='comma separated subset, e.g. F001,F003 (default all)')
    ap.add_argument('--skip', help='comma separated rules to leave out')
    ap.add_argument('--max-samples', type=int, default=400, help='evenly spaced subsample for per-sample rules (0 = all; default 400)')
    ap.add_argument('--neighbourhood', type=int, default=1, help='F002: in-plane dilation radius of the source mask in cells (default 1)')
    ap.add_argument('--head', type=int, help='use only the first N samples (in file order)')
    ap.add_argument('--transpose-inputs', action='store_true', help='swap the in-plane axes of source, k and dirichlet fields after loading')
    ap.add_argument('--json', nargs='?', const='-', metavar='FILE', help='write the JSON report (to FILE, or stdout if omitted)')
    ap.add_argument('--list-rules', action='store_true', help='print the rules with their documentation and exit')
    ap.add_argument('--version', action='version', version=f'fieldlint {__version__}')
    return ap


def _parse_splits(s: str):
    if s in ('none', 'from_filename'):
        return s
    if s == 'ictherm':
        return {'rule': 'ictherm'}
    out = {}
    for part in s.split(','):
        name, _, rng = part.partition('=')
        lo, _, hi = rng.partition(':')
        out[name] = [float(lo), float(hi)]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.list_rules:
        for c, f in RULES.items():
            print((f.__doc__ or '').strip(), '\n')
        return 0
    if not a.path:
        ap.error('PATH is required')
    try:
        cfg = {}
        if a.preset:
            cfg.update(load_config(a.preset))
        if a.config:
            cfg.update(load_config(a.config))
        for k, v in (('format', a.format), ('layout', a.layout), ('units', a.units), ('ambient', a.ambient)):
            if v is not None:
                cfg[k] = v
        if a.spacing:
            cfg['spacing'] = [float(x) for x in a.spacing.split(',')]
        if a.splits:
            cfg['splits'] = _parse_splits(a.splits)
        _field(cfg, 'u', a.u, None, None, a.layout)
        _field(cfg, 'source', a.source, a.source_layout, a.source_channel, a.layout)
        _field(cfg, 'k', a.k, a.k_layout, a.k_channel, a.layout)
        _field(cfg, 'dirichlet_mask', a.dirichlet, None, None, a.layout)
        _field(cfg, 'flux_out', a.flux_out, 'N', None, 'N')
        if a.transpose_inputs:
            for nm in ('source', 'k', 'dirichlet_mask'):
                if isinstance(cfg.get(nm), dict):
                    cfg[nm]['transpose'] = True
        ds = open_dataset(a.path, cfg)
        if a.head:
            ds = ds.head(a.head)
        codes = parse_rules(a.rules)
        if a.skip:
            codes = [c for c in codes if c not in parse_rules(a.skip)]
    except (AdapterError, ValueError, OSError) as exc:
        print(f'fieldlint: {exc}', file=sys.stderr)
        return 3
    rep = run(ds, codes, Options(max_samples=a.max_samples or None, neighbourhood=a.neighbourhood))
    ds.close()
    if a.json == '-':
        print(json.dumps(rep.to_dict(), indent=1, default=float))
    else:
        print(rep.text())
        if a.json:
            with open(a.json, 'w', encoding='utf-8') as fh:
                json.dump(rep.to_dict(), fh, indent=1, default=float)
    return rep.exit_code


if __name__ == '__main__':
    sys.exit(main())
