"""iceforge command line."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import List, Optional

from . import __version__, backends
from .check import check_stack, exit_code, format_findings
from .model import parse_stk, summary, to_dict
from .build import build, load_spec, EXAMPLE_YAML
from .run import run_model
from .snap import snap


def _jdump(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_parse(a) -> int:
    try:
        st = parse_stk(a.model)
    except OSError as e:
        print(f"parse: {e}", file=sys.stderr)
        return 2
    if a.json:
        _jdump(to_dict(st))
    else:
        print(summary(st))
    return 1 if st.missing_files else 0


def cmd_check(a) -> int:
    try:
        st = parse_stk(a.model)
    except OSError as e:
        print(f"check: {e}", file=sys.stderr)
        return 2
    F = check_stack(st)
    code = exit_code(F, a.strict)
    if a.json:
        _jdump(dict(stk=st.path, findings=[f.to_dict() for f in F], exit_code=code))
    else:
        print(format_findings(F))
    return code


def cmd_snap(a) -> int:
    try:
        out, moves, notes = snap(a.model, a.output, a.mode)
    except (OSError, ValueError) as e:
        print(f"snap: {e}", file=sys.stderr)
        return 2
    if a.json:
        _jdump(dict(stk=out, mode=a.mode, moves=[m.to_dict() for m in moves], notes=notes))
    else:
        for n in notes:
            print(f"note: {n}")
        for m in moves:
            print(m)
        print(f"snap ({a.mode}): {len(moves)} change(s); wrote {out}")
    return 0


def cmd_run(a) -> int:
    try:
        code, report = run_model(a.model, a.backend, a.exe, a.output, not a.no_check,
                                 log=(lambda *x: print(*x, file=sys.stderr)) if a.json else print)
    except OSError as e:
        print(f"run: {e}", file=sys.stderr)
        return 2
    if a.json:
        _jdump(report)
    return code


def cmd_build(a) -> int:
    try:
        stk, notes = build(load_spec(a.spec), a.output, snap=a.snap)
    except (OSError, ValueError) as e:      # BuildError is a ValueError
        print(f"build: {e}", file=sys.stderr)
        return 2
    for n in notes:
        print(f"note: {n}")
    print(f"wrote {stk}")
    return 0


def cmd_init(a) -> int:
    target = a.output or ("spec.json" if a.json else "spec.yaml")
    if os.path.exists(target) and not a.force:
        print(f"init: {target} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    text = EXAMPLE_YAML
    if a.json:
        try:
            import yaml
        except ImportError:
            print("init --json needs pyyaml to convert the example", file=sys.stderr)
            return 2
        text = json.dumps(yaml.safe_load(EXAMPLE_YAML), indent=2) + "\n"
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {target}")
    return 0


def cmd_doctor(a) -> int:
    info = dict(iceforge=__version__, python=sys.version.split()[0], platform=platform.platform(), backends={})
    env = os.environ.get("ICE_EXECUTABLE")
    probes = {
        "env ICE_EXECUTABLE": (lambda: backends.from_exe(env) if env else None),
        "native": backends.probe_native,
        "wsl": backends.probe_wsl,
        "docker": backends.probe_docker,
    }
    first = None
    for name, probe in probes.items():
        try:
            b = probe()
        except Exception as e:  # pragma: no cover
            b = None
            info["backends"][name] = dict(ok=False, error=str(e))
            continue
        if b is None:
            info["backends"][name] = dict(ok=False)
            continue
        entry = dict(ok=True, backend=str(b))
        try:
            entry["version"] = backends.emulator_version(b)
        except Exception as e:  # pragma: no cover
            entry["version"] = f"unknown ({e})"
        info["backends"][name] = entry
        first = first or b
    info["auto_choice"] = str(first) if first else None
    if a.json:
        _jdump(info)
    else:
        print(f"iceforge {info['iceforge']}  python {info['python']}  {info['platform']}")
        for name, e in info["backends"].items():
            if e.get("ok"):
                print(f"  [ok]   {name:20s} {e['backend']}  3D-ICE version: {e.get('version')}")
            else:
                print(f"  [--]   {name:20s} {e.get('error', 'not available')}")
        print(f"auto backend: {info['auto_choice'] or 'none found'}")
    return 0 if first else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iceforge", description="Front end for 3D-ICE: parse, check, snap, run.")
    p.add_argument("--version", action="version", version=f"iceforge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("parse", help="parse a .stk and its .flp/.lyt files and print a summary")
    s.add_argument("model")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_parse)

    s = sub.add_parser("check", help="pre-solve lint (rules S001-S009)")
    s.add_argument("model")
    s.add_argument("--json", action="store_true")
    s.add_argument("--strict", action="store_true", help="warnings also give a non-zero exit code")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("snap", help="write a copy with floorplan/layout edges on the cell grid")
    s.add_argument("model")
    s.add_argument("-o", "--output", required=True, help="output directory")
    s.add_argument("--mode", choices=["nearest", "outward"], default="nearest")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_snap)

    s = sub.add_parser("run", help="check, run 3D-ICE, parse Tmap outputs, post-check")
    s.add_argument("model")
    s.add_argument("--backend", choices=["auto", "native", "wsl", "docker"], default="auto")
    s.add_argument("--exe", help="path to 3D-ICE-Emulator (inside WSL for the wsl backend)")
    s.add_argument("-o", "--output", help="output directory (default <model>_iceforge next to the .stk)")
    s.add_argument("--no-check", action="store_true", help="skip the pre-solve check")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("build", help="generate grid-aligned .stk/.flp/.lyt from a YAML/JSON spec")
    s.add_argument("spec")
    s.add_argument("-o", "--output", required=True, help="output directory")
    s.add_argument("--snap", action="store_true", help="snap off-grid die edges instead of refusing")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("init", help="write an example build spec")
    s.add_argument("-o", "--output", help="file to write (default spec.yaml, or spec.json with --json)")
    s.add_argument("--json", action="store_true", help="write JSON instead of YAML (needs pyyaml)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("doctor", help="report which backends work and the 3D-ICE version")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_doctor)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
