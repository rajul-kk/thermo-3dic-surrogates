"""The `pip` backend: the `threedice` wheel's bundled 3D-ICE, and the pin check for installs without .git."""
import json
import os
import sys
import types

import pytest

from iceforge import backends

PIN40 = backends.PINNED_COMMIT + "a" * (40 - len(backends.PINNED_COMMIT))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(backends.ALLOW_ANY_ENV, raising=False)
    monkeypatch.delenv("ICE_EXECUTABLE", raising=False)
    backends._COMMIT_CACHE.clear()
    yield
    backends._COMMIT_CACHE.clear()


def _fake_threedice(monkeypatch, tmp_path, commit=PIN40):
    exe = tmp_path / "threedice" / "bin" / backends.EMU
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    mod = types.ModuleType("threedice")
    mod.executable = lambda: str(exe)
    mod.ice_commit = lambda: commit
    mod.__version__ = "4.0.0.post1"
    monkeypatch.setitem(sys.modules, "threedice", mod)
    return str(exe)


def _no_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(backends, "_run", lambda cmd, **k: calls.append(cmd) or (_ for _ in ()).throw(AssertionError(cmd)))
    return calls


def test_probe_pip_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "threedice", None)          # import raises ImportError
    assert backends.probe_pip() is None
    with pytest.raises(backends.BackendError, match="'pip' is not available"):
        backends.detect("pip")


def test_probe_pip_found(monkeypatch, tmp_path):
    exe = _fake_threedice(monkeypatch, tmp_path)
    b = backends.probe_pip()
    assert b.kind == "pip" and b.exe == exe and "4.0.0.post1" in b.detail
    assert backends.detect("pip").exe == exe


def test_auto_prefers_pip_over_native_wsl_docker_but_not_env(monkeypatch, tmp_path):
    exe = _fake_threedice(monkeypatch, tmp_path)
    monkeypatch.setattr(backends, "probe_native", lambda: backends.Backend("native", "/usr/bin/" + backends.EMU))
    assert backends.detect("auto").kind == "pip"
    monkeypatch.setenv("ICE_EXECUTABLE", "wsl /home/u/3d-ice-4.0/bin/" + backends.EMU)
    assert backends.detect("auto").kind == "wsl"                   # an explicit ICE_EXECUTABLE still wins


def test_auto_falls_through_without_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "threedice", None)
    monkeypatch.setattr(backends, "probe_native", lambda: backends.Backend("native", "/usr/bin/" + backends.EMU))
    assert backends.detect("auto").kind == "native"


def test_pin_check_reads_package_commit_without_git(monkeypatch, tmp_path):
    _fake_threedice(monkeypatch, tmp_path)
    calls = _no_subprocess(monkeypatch)
    d = backends.verify_pinned(backends.probe_pip())
    assert d["ice_commit"] == PIN40 and d["ice_executable"].startswith("pip:") and not calls


def test_pin_check_refuses_other_package_commit(monkeypatch, tmp_path):
    _fake_threedice(monkeypatch, tmp_path, commit="39ed2a7" + "0" * 33)
    with pytest.raises(backends.BackendError, match="THERMO_ALLOW_ANY_3DICE"):
        backends.verify_pinned(backends.probe_pip())


def test_pip_command_runs_binary_in_stk_dir(monkeypatch, tmp_path):
    exe = _fake_threedice(monkeypatch, tmp_path)
    argv, cwd = backends.command_for(backends.probe_pip(), str(tmp_path / "m" / "s.stk"))
    assert argv == [exe, "s.stk"] and cwd == str(tmp_path / "m")


def _write_sidecar(root, commit, package="threedice"):
    root.mkdir(parents=True)
    (root / "_build_info.json").write_text(json.dumps({"package": package, "ice_commit": commit}))
    (root / "bin").mkdir()
    exe = root / "bin" / backends.EMU
    exe.write_text("")
    return str(exe)


def test_native_exe_inside_wheel_dir_uses_sidecar_not_git(monkeypatch, tmp_path):
    exe = _write_sidecar(tmp_path / "venv" / "site-packages" / "threedice", PIN40)
    calls = _no_subprocess(monkeypatch)
    d = backends.verify_pinned(backends.Backend("native", exe))
    assert d["ice_commit"] == PIN40 and not calls


def test_sidecar_ignored_outside_threedice_dir_and_for_other_packages(monkeypatch, tmp_path):
    other = _write_sidecar(tmp_path / "3d-ice-x", PIN40)                    # not a threedice dir: git is asked
    seen = []

    class R:
        returncode, stdout, stderr = 0, "0" * 40, ""
    monkeypatch.setattr(backends, "_run", lambda cmd, **k: seen.append(cmd) or R())
    assert backends.ice_commit(backends.Backend("native", other)) == "0" * 40 and seen
    backends._COMMIT_CACHE.clear()
    seen.clear()
    wrong = _write_sidecar(tmp_path / "x" / "threedice", PIN40, package="something-else")
    assert backends.ice_commit(backends.Backend("native", wrong)) == "0" * 40 and seen


def test_wsl_exe_inside_wheel_dir_reads_sidecar_through_wsl(monkeypatch):
    class R:
        returncode, stderr = 0, ""
        stdout = json.dumps({"package": "threedice", "ice_commit": PIN40})
    calls = []
    monkeypatch.setattr(backends, "_run", lambda cmd, **k: calls.append(cmd) or R())
    b = backends.Backend("wsl", "/home/u/venv/lib/python3.12/site-packages/threedice/bin/" + backends.EMU)
    assert backends.verify_pinned(b)["ice_commit"] == PIN40
    assert calls == [["wsl", "-e", "cat", "/home/u/venv/lib/python3.12/site-packages/threedice/_build_info.json"]]


# ------------------------------------------------------------------ real wheel (skipped when not installed)

def _real_threedice():
    try:
        import threedice
        return threedice if os.path.isfile(threedice.executable()) and threedice.ice_commit() else None
    except Exception:
        return None


@pytest.mark.skipif(_real_threedice() is None, reason="the threedice wheel is not installed")
def test_real_wheel_solves_repro_case_and_passes_pin(repro, tmp_path):
    from iceforge.run import run_model
    code, rep = run_model(os.path.join(repro, "aligned", "s.stk"), backend="pip", outdir=str(tmp_path / "o"),
                          log=lambda *a: None)
    assert code == 0, rep
    assert rep["backend"].startswith("pip:") and rep["ice_commit"].startswith(backends.PINNED_COMMIT)
    assert rep["tmaps"][0]["max_rise"] == pytest.approx(11.130, abs=0.01)
