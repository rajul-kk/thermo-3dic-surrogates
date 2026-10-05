import json
import os

import numpy as np
import pytest

from iceforge.cli import main


def stk(fixtures, name):
    return os.path.join(fixtures, "repro", name, "s.stk")


def test_parse_json(fixtures, capsys):
    assert main(["parse", stk(fixtures, "aligned"), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["dims"]["n_cols"] == 40 and d["sink_temperature"] == 300.0


def test_parse_text(fixtures, capsys):
    assert main(["parse", stk(fixtures, "aligned")]) == 0
    assert "40 columns (length) x 40 rows (width)" in capsys.readouterr().out


def test_check_exit_codes(fixtures, capsys):
    assert main(["check", stk(fixtures, "aligned")]) == 0
    assert main(["check", stk(fixtures, "misaligned")]) == 1
    assert main(["check", stk(fixtures, "nolayout")]) == 0
    assert main(["check", stk(fixtures, "nolayout"), "--strict"]) == 1
    capsys.readouterr()
    assert main(["check", stk(fixtures, "misaligned"), "--json"]) == 1
    d = json.loads(capsys.readouterr().out)
    assert "S001" in [f["code"] for f in d["findings"]]


def test_missing_model_is_usage_error(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nope.stk")]) == 2


def test_snap_then_check(fixtures, tmp_path, capsys):
    assert main(["snap", stk(fixtures, "misaligned"), "-o", str(tmp_path / "o")]) == 0
    assert "2 change(s)" in capsys.readouterr().out
    assert main(["check", str(tmp_path / "o" / "s.stk")]) == 0


def test_run_aborts_on_check_error_without_backend(fixtures, tmp_path):
    assert main(["run", stk(fixtures, "misaligned"), "-o", str(tmp_path / "o")]) == 1


def test_doctor_runs(capsys):
    rc = main(["doctor", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert set(d["backends"]) == {"env ICE_EXECUTABLE", "native", "wsl", "docker"}
    assert rc in (0, 1)


@pytest.mark.solver
def test_run_cli_end_to_end(repro, tmp_path):
    out = tmp_path / "o"
    assert main(["run", os.path.join(repro, "aligned", "s.stk"), "-o", str(out)]) == 0
    z = np.load(out / "tmap.npz")
    assert z["T"].shape == (40, 40)
