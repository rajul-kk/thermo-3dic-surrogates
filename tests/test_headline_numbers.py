"""Tripwire for the README "Main findings" numbers (goals T1).

``results/headline_numbers.json`` is written by ``scripts/headline_numbers.py`` from the result files. The README must
contain every recorded value, so changing a digit in either place fails here. Where the result files are on disk, the
values are also recomputed. Most of them live under the git-ignored ``results/`` folder, so on CI only the tracked
IC-ThermBench files are recomputed and the rest skip. Set ``THERMO_RESULTS`` to point at another results folder.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headline_numbers as hn  # noqa: E402

RESULTS = Path(os.environ.get("THERMO_RESULTS", ROOT / "results"))
RECORDED = json.loads((ROOT / "results" / "headline_numbers.json").read_text(encoding="utf-8"))
BY_ID = {r["id"]: r for r in RECORDED}


def flat(text):
    return " ".join(text.split())


README = flat((ROOT / "README.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", RECORDED, ids=[r["id"] for r in RECORDED])
def test_readme_states_the_recorded_value(row):
    assert flat(row["readme"]) in README, f"README no longer says {row['readme']!r} ({row['id']})"


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


def test_tripwire_fires_on_a_changed_digit():
    row = BY_ID["icb_s5_fixed"]
    changed = flat(row["readme"]).replace("7.30", "7.31")
    assert changed not in README
