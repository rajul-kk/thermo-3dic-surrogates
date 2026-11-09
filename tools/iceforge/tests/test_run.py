import json
import os

import numpy as np
import pytest

from iceforge import backends
from iceforge.model import parse_stk
from iceforge.run import parse_tmap, post_checks, run_model


def test_wsl_path_translation_with_spaces():
    assert backends.to_wsl_path(r"D:\Work\3D-ICE Thermal-modelling\Thermo\x.stk") == \
        "/mnt/d/Work/3D-ICE Thermal-modelling/Thermo/x.stk"
    assert backends.to_wsl_path("\\\\wsl.localhost\\Ubuntu\\home\\u\\m") == "/home/u/m"


def test_from_exe_conventions():
    assert backends.from_exe("wsl /home/u/3d-ice/bin/3D-ICE-Emulator").kind == "wsl"
    assert backends.from_exe("C:/tools/ice.exe").kind == "native"


def test_parse_tmap_orientation_and_shape(tmp_path):
    # file rows run along chip WIDTH: 3 rows (width idx) x 4 columns (length idx)
    f = tmp_path / "t.txt"
    arr = np.arange(12, dtype=float).reshape(3, 4)
    f.write_text("\n".join("  ".join(f"{v:7.3f}" for v in row) for row in arr) + "\n")
    T, F = parse_tmap(str(f), n_rows=3, n_cols=4)
    assert F == [] and T.shape == (4, 3)
    assert T[2, 1] == arr[1, 2]                      # T[length_index, width_index] == file[row=width, col=length]
    T2, F2 = parse_tmap(str(f), n_rows=4, n_cols=3)  # transposed grid expectation -> P001
    assert T2 is None and F2[0].code == "P001"


def test_parse_tmap_takes_last_block(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("1 1\n1 1\n2 2\n2 2\n")
    T, _ = parse_tmap(str(f), 2, 2)
    assert T.max() == 2


def test_post_check_p002_flags_peak_away_from_source(repro):
    st = parse_stk(os.path.join(repro, "aligned", "s.stk"))
    T = np.full((40, 40), 300.0)
    T[10, 10] = 320.0            # cell centre (2625, 2625): inside the die (1500..7500)
    assert post_checks(st, T, "x")[0] == []
    T[2, 2] = 330.0              # centre (625, 625): outside
    F, hot = post_checks(st, T, "x")
    assert [f.code for f in F] == ["P002"]


def test_run_aborts_on_check_error(repro, tmp_path):
    code, rep = run_model(os.path.join(repro, "misaligned", "s.stk"), outdir=str(tmp_path / "o"), log=lambda *a: None)
    assert code == 1 and rep["status"] == "aborted"


# ------------------------------------------------------------------ integration (3D-ICE required)

def _run(repro, name, tmp_path, no_check=False):
    out = tmp_path / ("out_" + name)
    code, rep = run_model(os.path.join(repro, name, "s.stk"), outdir=str(out), do_check=not no_check,
                          log=lambda *a: None)
    return code, rep, out


@pytest.mark.solver
@pytest.mark.parametrize("name,expected,p002", [
    ("aligned", 11.130, False),
    ("misaligned", 14.562, True),
    ("snapped", 11.118, False),
])
def test_repro_max_rise(repro, tmp_path, name, expected, p002):
    code, rep, out = _run(repro, name, tmp_path, no_check=True)   # misaligned would be aborted by check
    assert rep["status"] in ("ok", "post-check-error"), rep
    tm = rep["tmaps"][0]
    assert tm["max_rise"] == pytest.approx(expected, abs=0.01)
    fired = any(f["code"] == "P002" for f in tm["findings"])
    assert fired is p002
    z = np.load(tm["npz"])
    assert z["T"].shape == (40, 40) and str(z["units"]) == "K"
    side = json.load(open(os.path.splitext(tm["npz"])[0] + ".json"))
    assert side["max_rise"] == pytest.approx(expected, abs=0.01)


@pytest.mark.solver
def test_nolayout_runs_without_artefact(repro, tmp_path):
    code, rep, _ = _run(repro, "nolayout", tmp_path)       # S002 warning only, so check passes
    assert code == 0
    assert not any(f["code"] == "P002" for f in rep["tmaps"][0]["findings"])


@pytest.mark.solver
def test_orientation_asymmetric_die(tmp_path):
    """Rectangular chip (length 10000, width 6000) with an off-centre die: the hottest cell of the
    npz must sit under the die in (length, width) coordinates."""
    d = tmp_path / "asym"
    d.mkdir()
    (d / "die.flp").write_text("blk :\n position 6000.0, 500.0 ;\n dimension 3000.0, 1500.0 ;\n power values 10.0 ;\n")
    (d / "s.stk").write_text("""material SI :
   thermal conductivity 1.48e-4 ;
   volumetric heat capacity 1.628e-12 ;
bottom heat sink :
   heat transfer coefficient 2.0e-8 ;
   temperature 300.0 ;
dimensions :
   chip length 10000.0 , width 6000.0 ;
   cell length 250.0 , width 250.0 ;
layer SUB :
   height 300.0 ;
   material SI ;
die D :
   source 50.0 SI ;
stack :
   die TOPD D floorplan "die.flp" ;
   layer BOT SUB ;
solver :
   steady ;
   initial temperature 300.0 ;
output :
   Tmap ( TOPD, "tmap.txt", final ) ;
""")
    code, rep = run_model(str(d / "s.stk"), outdir=str(tmp_path / "o"), log=lambda *a: None)
    assert code == 0, rep
    z = np.load(rep["tmaps"][0]["npz"])
    T, x, y = z["T"], z["x_um"], z["y_um"]
    assert T.shape == (40, 24) and len(x) == 40 and len(y) == 24
    i, j = np.unravel_index(np.argmax(T), T.shape)
    assert 6000 <= x[i] <= 9000 and 500 <= y[j] <= 2000, (x[i], y[j])
    # and the raw file really is (width rows) x (length columns)
    raw = np.loadtxt(str(d / "tmap.txt"), comments="%")
    assert raw.shape == (24, 40) and np.array_equal(raw.T, T)
