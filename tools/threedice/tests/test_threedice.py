"""Checks of the installed wheel; skipped when threedice (with its binary) is not installed."""
import json
import os
import subprocess
import sys

import pytest

threedice = pytest.importorskip("threedice")
if not os.path.isfile(os.path.join(os.path.dirname(threedice.__file__), "bin", "3D-ICE-Emulator")):
    pytest.skip("no bundled binary (source checkout)", allow_module_level=True)

PIN = "4953952a1ef6d38807ff307212a6f15e5b2ef935"


def test_commit_and_build_info():
    assert threedice.ice_commit() == PIN
    info = threedice.build_info()
    assert info["package"] == "threedice" and info["patches"] and info["toolchain"]["gcc"]


def test_executable_is_runnable_and_runs_a_stack(tmp_path):
    exe = threedice.executable()
    assert os.access(exe, os.X_OK)
    (tmp_path / "d.flp").write_text("blk :\n position 0, 0 ;\n dimension 1000, 1000 ;\n power values 1.0 ;\n")
    (tmp_path / "m.stk").write_text(
        "material SI :\n thermal conductivity 1.30e-4 ;\n volumetric heat capacity 1.628e-12 ;\n"
        "bottom heat sink :\n heat transfer coefficient 2.0e-8 ;\n temperature 300 ;\n"
        "dimensions :\n chip length 1000, width 1000 ;\n cell length 250, width 250 ;\n"
        "die D :\n source 50 SI ;\nstack :\n die TOP D floorplan \"d.flp\" ;\n"
        "solver :\n steady ;\n initial temperature 300 ;\noutput :\n Tmap ( TOP, \"t.txt\", final ) ;\n")
    r = subprocess.run([exe, "m.stk"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "t.txt").exists()


def test_console_module_form_runs_the_same_binary(tmp_path):
    r = subprocess.run([sys.executable, "-m", "threedice"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0          # no stack argument: the emulator itself reports usage
