import os
import shutil
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")


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


def pytest_collection_modifyitems(config, items):
    if any("solver" in it.keywords for it in items):
        if _backend_available() is None:
            skip = pytest.mark.skip(reason="no 3D-ICE backend available")
            for it in items:
                if "solver" in it.keywords:
                    it.add_marker(skip)
