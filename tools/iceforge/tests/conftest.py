import os
import shutil
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
# always test the package next to these tests, even if another checkout is pip-installed in editable mode
sys.path.insert(0, os.path.dirname(HERE))


@pytest.fixture
def fixtures():
    return FIX


@pytest.fixture
def repro(tmp_path):
    """Writable copy of the repro cases (the solver writes its output next to the .stk)."""
    dst = tmp_path / "repro"
    shutil.copytree(os.path.join(FIX, "repro"), dst)
    return str(dst)


def _backend_available():
    try:
        from iceforge import backends
        return backends.detect("auto")
    except Exception:
        return None


def _hotspot_unavailable():
    """None when a HotSpot build that passes iceforge's convergence self-test is available, else the reason."""
    try:
        from iceforge import hotspot
        err, _ = hotspot.preflight()
        return err
    except Exception as e:
        return str(e)


def pytest_collection_modifyitems(config, items):
    if any("solver" in it.keywords for it in items):
        if _backend_available() is None:
            skip = pytest.mark.skip(reason="no 3D-ICE backend available")
            for it in items:
                if "solver" in it.keywords:
                    it.add_marker(skip)


_HS = {}


def pytest_runtest_setup(item):
    """Skip a hotspot-marked test (lazily, so the self-test only runs when one is about to run)."""
    if "hotspot" in item.keywords:
        if "why" not in _HS:
            _HS["why"] = _hotspot_unavailable()
        if _HS["why"]:
            pytest.skip("no usable HotSpot (set HOTSPOT_EXECUTABLE to a SUPERLU=1 build): " + _HS["why"][:120])
