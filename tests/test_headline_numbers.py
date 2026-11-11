"""Tripwire for the README "Main findings" numbers (goals T1).

``results/headline_numbers.json`` is written by ``scripts/headline_numbers.py`` from the result files. The README must
contain every recorded value, so changing a digit in either place fails here. Each resolved value is also recomputed
from its result files, which are force-tracked under the otherwise git-ignored ``results/`` folder; a missing file
skips rather than fails. Set ``THERMO_RESULTS`` to point at another results folder.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headline_numbers as hn

RESULTS = Path(os.environ.get("THERMO_RESULTS", ROOT / "results"))
RECORDED = json.loads((ROOT / "results" / "headline_numbers.json").read_text(encoding="utf-8"))
BY_ID = {r["id"]: r for r in RECORDED}


def flat(text):
    return " ".join(text.split())


def main_findings(readme: str) -> str:
    """The "## Main findings" section, so a value elsewhere in the README cannot satisfy the check."""
    start = readme.index("\n## Main findings")
    end = readme.index("\n## ", start + 1)
    return flat(readme[start:end])


FINDINGS = main_findings((ROOT / "README.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", RECORDED, ids=[r["id"] for r in RECORDED])
def test_readme_states_the_recorded_value(row):
    assert flat(row["readme"]) in FINDINGS, f"README Main findings no longer says {row['readme']!r} ({row['id']})"


def test_every_entry_is_recorded():
    expected = [e["id"] for e in hn.ENTRIES] + [e["id"] for e in hn.UNRESOLVED]
    assert [r["id"] for r in RECORDED] == expected


@pytest.mark.parametrize("entry", hn.ENTRIES, ids=[e["id"] for e in hn.ENTRIES])
def test_recorded_value_recomputes_from_its_result_files(entry):
    missing = [s for s in entry["source"] if not (RESULTS / s).exists()]
    if missing:
        pytest.skip(f"result file not on disk: results/{missing[0]}")
    value, _ = entry["derive"](RESULTS)
    assert value == BY_ID[entry["id"]]["value"]


def bump_first_digit(text: str) -> str:
    i = next(i for i, ch in enumerate(text) if ch.isdigit())
    return text[:i] + str((int(text[i]) + 1) % 10) + text[i + 1:]


NUMERIC = [r for r in RECORDED if r["value"] is not None and any(ch.isdigit() for ch in r["value"])]


@pytest.mark.parametrize("row", NUMERIC, ids=[r["id"] for r in NUMERIC])
def test_tripwire_fires_on_a_changed_digit(row):
    changed = flat(row["readme"]).replace(row["value"], bump_first_digit(row["value"]), 1)
    assert changed != flat(row["readme"])
    assert changed not in FINDINGS
