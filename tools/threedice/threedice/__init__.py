"""The 3D-ICE 4.0 thermal simulator (3D-ICE-Emulator), pinned to upstream commit 4953952, as a wheel."""
import json
import os
import stat
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
EXE_NAME = "3D-ICE-Emulator"

try:
    from importlib.metadata import version as _version
    __version__ = _version("threedice")
except Exception:                                  # not installed (source checkout)
    __version__ = "unknown"


def build_info():
    """The contents of _build_info.json: upstream commit, build date, toolchain, applied patches."""
    with open(os.path.join(_HERE, "_build_info.json")) as f:
        return json.load(f)


def ice_commit():
    """Full sha of the upstream esl-epfl/3d-ice commit this binary was built from."""
    return build_info()["ice_commit"]


def executable():
    """Absolute path of the bundled 3D-ICE-Emulator binary."""
    p = os.path.join(_HERE, "bin", EXE_NAME)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} is missing: this build of threedice has no binary for {sys.platform}")
    if os.name == "posix" and not os.access(p, os.X_OK):
        try:                                       # some installers drop the exec bit
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
    return p
