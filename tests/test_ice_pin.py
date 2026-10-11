"""3D-ICE build pin: the executable must be upstream 4953952 unless THERMO_ALLOW_ANY_3DICE=1."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.simulators import ice_simulator as isim
from src.simulators.ice_simulator import (DEFAULT_ICE_EXECUTABLE, PINNED_ICE_COMMIT, ICESimulator,
                                          verify_ice_executable)

EXE = "wsl /fake/3d-ice-x/bin/3D-ICE-Emulator"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("THERMO_ALLOW_ANY_3DICE", raising=False)
    isim._COMMIT_CACHE.clear()
    isim._LAST_RUN.clear()
    yield
    isim._COMMIT_CACHE.clear()
    isim._LAST_RUN.clear()


def _fake_commit(monkeypatch, value):
    calls = []
    monkeypatch.setattr(isim, "_lookup_ice_commit", lambda e: calls.append(e) or value)
    return calls


def test_pinned_commit_accepted(monkeypatch):
    _fake_commit(monkeypatch, PINNED_ICE_COMMIT + "9" * 33)
    assert verify_ice_executable(EXE) == {"ice_commit": PINNED_ICE_COMMIT + "9" * 33, "ice_executable": EXE}


@pytest.mark.parametrize("commit", ["39ed2a7" + "0" * 33, "unknown"])
def test_other_or_unknown_commit_refused(monkeypatch, commit):
    _fake_commit(monkeypatch, commit)
    with pytest.raises(RuntimeError, match="THERMO_ALLOW_ANY_3DICE"):
        verify_ice_executable(EXE)


def test_override_allows_and_records(monkeypatch):
    _fake_commit(monkeypatch, "unknown")
    monkeypatch.setenv("THERMO_ALLOW_ANY_3DICE", "1")
    assert verify_ice_executable(EXE)["ice_commit"] == "unknown"


def test_commit_cached_per_process(monkeypatch):
    calls = _fake_commit(monkeypatch, PINNED_ICE_COMMIT + "0" * 33)
    verify_ice_executable(EXE)
    verify_ice_executable(EXE)
    assert len(calls) == 1


def test_lookup_without_git_is_unknown(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(isim.subprocess, "run", boom)
    assert isim._lookup_ice_commit(EXE) == "unknown"


def test_run_simulation_refuses_before_launching(monkeypatch, tmp_path):
    _fake_commit(monkeypatch, "deadbee" + "0" * 33)
    launched = []
    monkeypatch.setattr(isim.subprocess, "run", lambda *a, **k: launched.append(a))
    sim = ICESimulator(config_dir=tmp_path, output_dir=tmp_path / "out", executable=EXE)
    sim._nx = sim._ny = 1
    with pytest.raises(RuntimeError, match="pinned"):
        sim.run_simulation("s")
    assert not launched


def test_exporter_records_provenance(monkeypatch, tmp_path):
    from src.core.geometry_builders import build_geometry1
    from src.export.npz_exporter import NPZExporter
    from src.scenario.generator import ScenarioGenerator
    from src.core.mesh import generate_coords_and_indices
    geom = build_geometry1()
    scen = ScenarioGenerator().generate_all_scenarios(geom)[0].to_dict()
    coords, _ = generate_coords_and_indices(geom, uniform_z=False)
    isim._LAST_RUN.update({"ice_commit": "4953952" + "0" * 33, "ice_executable": EXE})
    f = NPZExporter(tmp_path).export_scenario("s", geom, scen, np.full(len(coords), 300.0), coords)
    md = dict(np.load(f, allow_pickle=True)["metadata"][0])
    assert md["ice_commit"].startswith("4953952") and md["ice_executable"] == EXE


def _wsl_ok():
    if shutil.which("wsl") is None:
        return False
    try:
        return subprocess.run(["wsl", "-e", "test", "-x", "/home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator"],
                              timeout=60).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _wsl_ok(), reason="WSL / ~/3d-ice-4.0 not available")
def test_real_default_build_is_the_pinned_one():
    info = verify_ice_executable(DEFAULT_ICE_EXECUTABLE)
    assert info["ice_commit"].startswith(PINNED_ICE_COMMIT)


# ---- the `threedice` wheel (tools/threedice): no .git, the commit comes from <pkg>/_build_info.json

def _wheel_dir(tmp_path, commit, package="threedice"):
    import json
    root = tmp_path / "venv" / "site-packages" / "threedice"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "3D-ICE-Emulator").write_text("")
    (root / "_build_info.json").write_text(json.dumps({"package": package, "ice_commit": commit}))
    return (root / "bin" / "3D-ICE-Emulator").as_posix()      # shlex would eat backslashes


def test_threedice_wheel_commit_read_from_sidecar_not_git(monkeypatch, tmp_path):
    exe = _wheel_dir(tmp_path, PINNED_ICE_COMMIT + "b" * 33)
    monkeypatch.setattr(isim.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("git called")))
    assert verify_ice_executable(exe)["ice_commit"] == PINNED_ICE_COMMIT + "b" * 33


def test_threedice_wheel_with_other_commit_refused(monkeypatch, tmp_path):
    exe = _wheel_dir(tmp_path, "39ed2a7" + "0" * 33)
    with pytest.raises(RuntimeError, match="THERMO_ALLOW_ANY_3DICE"):
        verify_ice_executable(exe)


def test_sidecar_of_another_package_is_ignored(monkeypatch, tmp_path):
    exe = _wheel_dir(tmp_path, PINNED_ICE_COMMIT + "b" * 33, package="other")
    monkeypatch.setattr(isim.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    assert isim._lookup_ice_commit(exe) == "unknown"
