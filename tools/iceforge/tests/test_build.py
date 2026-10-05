import copy
import json
import os

import pytest

from iceforge import cli
from iceforge.build import BuildError, build, load_spec, EXAMPLE_YAML
from iceforge.check import check_stack
from iceforge.model import parse_stk
from iceforge.run import run_model

REPRO = dict(
    chip=dict(length_um=10000, width_um=10000, cell_um=250),
    heat_sink=dict(side="bottom", htc_W_m2K=20000, t_ambient_K=300),
    materials=dict(SI=dict(k_W_mK=148, rho_cp_J_m3K=1.628e6), GAP=dict(k_W_mK=0.7, rho_cp_J_m3K=1.628e6)),
    layers=[
        dict(name="SUB", height_um=300, material="SI"),
        dict(name="SRC", height_um=50, material="GAP",
             dies=[dict(name="blk", material="SI", x_um=1500, y_um=1500, length_um=6000, width_um=6000, power_W=10)]),
    ],
    solver="steady",
)

TWO_DIE = dict(
    chip=dict(length_um=10000, width_um=8000, cell_um=250),
    heat_sink=dict(side="top", htc_W_m2K=10000, t_ambient_K=318.15),
    materials=dict(SI=dict(k_W_mK=148, rho_cp_J_m3K=1.628e6), UF=dict(k_W_mK=0.7, rho_cp_J_m3K=1.628e6),
                   TIM=dict(k_W_mK=4, rho_cp_J_m3K=2e6)),
    layers=[
        dict(name="BASE", height_um=200, material="SI"),
        dict(name="SRC", height_um=20, material="UF", dies=[
            dict(name="cpu", material="SI", x_um=1000, y_um=1000, length_um=3000, width_um=3000, power_W=20),
            dict(name="hbm", material="SI", x_um=5000, y_um=1000, length_um=2000, width_um=3000, power_W=5)]),
        dict(name="TIM", height_um=40, material="TIM"),
    ],
)


def test_repro_files_and_units(tmp_path):
    stk, notes = build(REPRO, str(tmp_path))
    assert notes == []
    st = parse_stk(stk)
    assert check_stack(st) == [] or all(f.severity != "error" for f in check_stack(st))
    text = open(stk).read()
    assert "1.48e-05" in text or "0.000148" in text            # 148 W/m/K -> 1.48e-4 W/um/K
    assert text.index("layer L_SUB") < text.index("die D_SRC") < text.index("stack :")
    stack = text[text.index("stack :"):]
    assert stack.index("die   SRC") < stack.index("layer SUB")  # top to bottom
    assert open(tmp_path / "SRC.lyt").read().strip().startswith("SI :")


def test_check_passes_two_die_top_sink(tmp_path):
    stk, _ = build(TWO_DIE, str(tmp_path))
    F = check_stack(parse_stk(stk))
    assert not [f for f in F if f.severity == "error"], F
    text = open(stk).read()
    assert text.index("top heat sink") >= 0
    stack = text[text.index("stack :"):]
    assert stack.index("layer TIM") < stack.index("die   SRC") < stack.index("layer BASE")
    assert text.count("layout") == 1 and "cpu :" in open(tmp_path / "SRC.flp").read()


def test_off_grid_refused_with_nearest(tmp_path):
    s = copy.deepcopy(REPRO)
    s["layers"][1]["dies"][0]["x_um"] = 1550
    with pytest.raises(BuildError) as e:
        build(s, str(tmp_path))
    assert "x_um=1500" in str(e.value) and "--snap" in str(e.value)


def test_snap_flag_snaps(tmp_path):
    s = copy.deepcopy(REPRO)
    s["layers"][1]["dies"][0].update(x_um=1550, y_um=1480)
    stk, notes = build(s, str(tmp_path), snap=True)
    assert len(notes) == 1
    assert not [f for f in check_stack(parse_stk(stk)) if f.severity == "error"]


def test_overlap_and_bad_material(tmp_path):
    s = copy.deepcopy(TWO_DIE)
    s["layers"][1]["dies"][1]["x_um"] = 3000
    with pytest.raises(BuildError, match="overlaps"):
        build(s, str(tmp_path))
    s = copy.deepcopy(REPRO)
    s["layers"][0]["material"] = "NOPE"
    with pytest.raises(BuildError, match="NOPE"):
        build(s, str(tmp_path))


def test_cli_init_build_json(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(REPRO))
    assert cli.main(["build", str(spec), "-o", str(tmp_path / "o")]) == 0
    assert os.path.exists(tmp_path / "o" / "model.stk")
    bad = copy.deepcopy(REPRO)
    bad["layers"][1]["dies"][0]["x_um"] = 1551
    spec.write_text(json.dumps(bad))
    assert cli.main(["build", str(spec), "-o", str(tmp_path / "o2")]) == 2


def test_cli_init_yaml_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init"]) == 0
    assert cli.main(["init"]) == 2                       # refuses to overwrite
    assert load_spec("spec.yaml") == REPRO
    assert cli.main(["build", "spec.yaml", "-o", "out"]) == 0


@pytest.mark.solver
def test_built_repro_solves_to_11_130(tmp_path):
    stk, _ = build(REPRO, str(tmp_path / "m"))
    code, rep = run_model(stk, outdir=str(tmp_path / "o"), log=lambda *a: None)
    assert code == 0, rep
    assert rep["tmaps"][0]["max_rise"] == pytest.approx(11.130, abs=0.01)


@pytest.mark.solver
def test_built_two_die_solves(tmp_path):
    stk, _ = build(TWO_DIE, str(tmp_path / "m"))
    code, rep = run_model(stk, outdir=str(tmp_path / "o"), log=lambda *a: None)
    assert code == 0, rep
    assert rep["tmaps"][0]["max_rise"] > 0
