"""iceforge refuses to solve with a 3D-ICE build other than the pinned commit."""
import pytest

from iceforge import backends


def _b():
    return backends.Backend("wsl", "/fake/3d-ice-x/bin/3D-ICE-Emulator")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(backends.ALLOW_ANY_ENV, raising=False)
    backends._COMMIT_CACHE.clear()
    yield
    backends._COMMIT_CACHE.clear()


def _fake(monkeypatch, out, rc=0):
    class R:
        returncode, stdout, stderr = rc, out, ""
    calls = []
    monkeypatch.setattr(backends, "_run", lambda cmd, **k: calls.append(cmd) or R())
    return calls


def test_pinned_ok_and_cached(monkeypatch):
    calls = _fake(monkeypatch, backends.PINNED_COMMIT + "a" * 33 + "\n")
    d = backends.verify_pinned(_b())
    backends.verify_pinned(_b())
    assert d["ice_commit"].startswith(backends.PINNED_COMMIT) and len(calls) == 1
    assert calls[0][:5] == ["wsl", "-e", "git", "-C", "/fake/3d-ice-x"]


def test_other_commit_refused_then_override(monkeypatch):
    _fake(monkeypatch, "39ed2a7" + "0" * 33)
    with pytest.raises(backends.BackendError, match="THERMO_ALLOW_ANY_3DICE"):
        backends.verify_pinned(_b())
    monkeypatch.setenv(backends.ALLOW_ANY_ENV, "1")
    assert backends.verify_pinned(_b())["ice_commit"].startswith("39ed2a7")


def test_git_failure_is_unknown_and_refused(monkeypatch):
    _fake(monkeypatch, "", rc=128)
    with pytest.raises(backends.BackendError, match="unknown"):
        backends.verify_pinned(_b())


def test_run_model_reports_unpinned(monkeypatch, tmp_path):
    from iceforge.run import run_model
    import shutil, pathlib
    fx = pathlib.Path(__file__).parent / "fixtures" / "geometry1"
    shutil.copytree(fx, tmp_path / "m")
    _fake(monkeypatch, "0" * 40)
    code, rep = run_model(str(tmp_path / "m" / "s.stk"), exe="wsl /fake/3d-ice-x/bin/3D-ICE-Emulator",
                          do_check=False, log=lambda *a: None)
    assert code == 2 and rep["status"] == "unpinned-build"
