"""The stages of src/main.py: argument handling, extra-train numbering, generator-only output and the exit code."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import main as pipeline


def _args(*argv):
    return pipeline.build_parser().parse_args(list(argv))


def test_standard_geometries_leave_out_the_pilot():
    assert "geometry7" not in pipeline.STANDARD_GEOMETRIES
    assert len(pipeline.STANDARD_GEOMETRIES) == 6


def test_extra_train_start(tmp_path):
    args = _args("--output", str(tmp_path))
    assert pipeline.extra_train_start(args, "geometry1") == 16
    d = tmp_path / "geometry1"
    d.mkdir()
    for i in (1, 2, 21):
        (d / f"geometry1_train_{i:03d}.npz").touch()
    assert pipeline.extra_train_start(args, "geometry1") == 22
    assert pipeline.extra_train_start(_args("--output", str(tmp_path), "--train-start", "40"), "geometry1") == 40


def test_mock_simulator_needs_no_temp_dir():
    args = _args("--simulator", "mock")
    assert pipeline.make_sim_tmpdir(args) is None
    assert pipeline.make_simulator(args, None) is None


class _Closable:
    closed = False

    def close(self):
        self.closed = True


def test_release_closes_the_simulator_then_removes_its_dir(tmp_path):
    d = tmp_path / "sim"
    (d / "configs").mkdir(parents=True)
    sim = _Closable()
    pipeline.release_simulator(sim, str(d))
    assert sim.closed and not d.exists()
    pipeline.release_simulator(None, None)


def test_temp_dir_is_removed_when_the_run_raises(tmp_path, monkeypatch):
    made = []

    def mkdtemp(prefix):
        made.append(tmp_path / prefix)
        made[-1].mkdir()
        return str(made[-1])

    def boom(*a, **k):
        raise RuntimeError("simulator failed")

    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(pipeline, "run_geometries", boom)
    try:
        pipeline.main(["--simulator", "hotspot", "--geometry", "geometry1", "--output", str(tmp_path / "out")])
    except RuntimeError:
        pass
    assert len(made) == 1 and not made[0].exists()


def test_generator_only_makes_no_simulator(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "make_sim_tmpdir", lambda args: pytest.fail("built a simulator"))
    assert pipeline.main(["--generator-only", "--simulator", "3d-ice", "--output", str(tmp_path)]) == 0


def test_exit_code():
    ok = {"g": {"train": {"a": {}}, "test": {}, "failed": []}}
    partial = {"g": {"train": {"a": {}}, "test": {}, "failed": ["b"]}}
    empty = {"g": {"train": {}, "test": {}, "failed": []}}
    assert pipeline.exit_code(ok, Path(".")) == 0
    assert pipeline.exit_code(partial, Path(".")) == 1
    assert pipeline.exit_code(empty, Path(".")) == 1


def test_generator_only_writes_scenarios(tmp_path):
    assert pipeline.main(["--generator-only", "--geometry", "geometry1", "--extra-train", "2",
                          "--output", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "geometry1" / "scenarios.json").read_text())
    assert (len(data["train"]), len(data["test"])) == (17, 5)
    assert [s["name"] for s in data["train"][-2:]] == ["geometry1_train_016", "geometry1_train_017"]
    assert set(data["train"][0]) == {"name", "geometry", "type", "power_blocks", "htc", "t_ambient", "pattern",
                                     "description", "rdl_joule_fraction", "layer_k_overrides"}
