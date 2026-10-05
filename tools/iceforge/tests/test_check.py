import json
import os
import shutil

import pytest

from iceforge.check import check_stack, exit_code, edge_offset
from iceforge.model import parse_stk


def codes(path):
    return [f.code for f in check_stack(parse_stk(path))]


def case(fixtures, name):
    return os.path.join(fixtures, "repro", name, "s.stk")


def test_edge_offset():
    assert edge_offset(1500, 250) == pytest.approx(0)
    assert edge_offset(1394.7, 250) == pytest.approx(-0.4212, abs=1e-3)


def test_aligned_is_clean(fixtures):
    assert codes(case(fixtures, "aligned")) == []


def test_s001_only_on_misaligned(fixtures):
    assert "S001" in codes(case(fixtures, "misaligned"))
    for name in ("aligned", "nolayout", "snapped"):
        assert "S001" not in codes(case(fixtures, name)), name


def test_s001_reports_element_edge_offset(fixtures):
    f = [f for f in check_stack(parse_stk(case(fixtures, "misaligned"))) if f.code == "S001"][0]
    assert f.severity == "error" and f.detail["element"] == "blk"
    offs = {e["edge"]: e["offset_cells"] for e in f.detail["edges"]}
    assert offs["x_min"] == pytest.approx(-0.4212, abs=1e-3)
    assert "heated insulator" in f.message


def test_misaligned_also_flags_layout_s003(fixtures):
    assert "S003" in codes(case(fixtures, "misaligned"))


def test_nolayout_is_only_a_warning(fixtures):
    st = parse_stk(case(fixtures, "nolayout"))
    F = check_stack(st)
    assert [f.code for f in F] == ["S002"] and F[0].severity == "warning"
    assert exit_code(F) == 0 and exit_code(F, strict=True) == 1


def test_snapped_case_has_only_s002(fixtures):
    assert codes(case(fixtures, "snapped")) == ["S002"]


def test_exit_code_on_error(fixtures):
    assert exit_code(check_stack(parse_stk(case(fixtures, "misaligned")))) == 1


def _write(tmp_path, flp, lyt=None, chip="10000.0 , width 10000.0", cell="250.0 , width 250.0", extra_mat="",
           k="1.48e-4", htc="2e-8", height="50.0"):
    (tmp_path / "d.flp").write_text(flp)
    layout = ""
    if lyt is not None:
        (tmp_path / "f.lyt").write_text(lyt)
        layout = '   layout "f.lyt" ;\n'
    (tmp_path / "s.stk").write_text(f"""
material SI : thermal conductivity {k} ; volumetric heat capacity 1.6e-12 ;
material GAP : thermal conductivity 7e-7 ; volumetric heat capacity 1.6e-12 ;
{extra_mat}
bottom heat sink : heat transfer coefficient {htc} ; temperature 300 ;
dimensions : chip length {chip} ; cell length {cell} ;
layer SRC : height {height} ; material GAP ;
{layout}
die D : source SRC ;
stack : die T D floorplan "d.flp" ;
solver : steady ; initial temperature 300 ;
output : Tmap (T, "t.txt", final) ;
""")
    return str(tmp_path / "s.stk")


FLP = "a :\n position 1500, 1500 ;\n dimension 6000, 6000 ;\n power values 10 ;\n"
LYT = "SI :\n rectangle ( 1500, 1500, 6000, 6000 ) ;\n"


def test_s004_element_outside_layout(tmp_path):
    flp = "a :\n position 1500, 1500 ;\n dimension 6000, 6000 ;\n power values 10 ;\n" \
          "b :\n position 8000, 8000 ;\n dimension 1000, 1000 ;\n power values 5 ;\n"
    p = _write(tmp_path, flp, LYT)
    F = [f for f in check_stack(parse_stk(p)) if f.code == "S004"]
    assert len(F) == 1 and F[0].detail["element"] == "b"


def test_s005_outside_and_overlap(tmp_path):
    flp = ("a :\n position 0, 0 ;\n dimension 6000, 6000 ;\n power values 1 ;\n"
           "b :\n position 5000, 5000 ;\n dimension 6000, 6000 ;\n power values 1 ;\n")
    p = _write(tmp_path, flp)
    msgs = [f.message for f in check_stack(parse_stk(p)) if f.code == "S005"]
    assert any("outside" in m for m in msgs) and any("overlap" in m for m in msgs)


def test_s006_chip_not_multiple(tmp_path):
    p = _write(tmp_path, FLP, chip="10100.0 , width 10000.0")
    assert "S006" in codes(p)


def test_s007_units(tmp_path):
    p = _write(tmp_path, FLP, k="148.0", htc="2.0e-2", height="0.5")
    F = [f for f in check_stack(parse_stk(p)) if f.code == "S007"]
    text = " ".join(f.message for f in F)
    assert "k = 148" in text and "HTC" in text and "height" in text


def test_s008_missing_file(tmp_path):
    p = _write(tmp_path, FLP)
    os.remove(tmp_path / "d.flp")
    F = check_stack(parse_stk(p))
    assert any(f.code == "S008" and f.severity == "error" for f in F)


def test_s009_zero_power(tmp_path):
    p = _write(tmp_path, "a :\n position 0, 0 ;\n dimension 1000, 1000 ;\n power values 0 ;\n")
    assert "S009" in codes(p)


def test_nonuniform_skips_grid_rules(fixtures):
    F = check_stack(parse_stk(os.path.join(fixtures, "3dice_examples", "example_steady_material.stk")))
    assert [f.code for f in F] == ["I001"]


def test_3dice_examples_have_no_errors(fixtures):
    for n in ("example_steady.stk", "example_transient_nonuniform.stk", "example_steady_material.stk"):
        F = check_stack(parse_stk(os.path.join(fixtures, "3dice_examples", n)))
        assert exit_code(F) == 0, (n, [str(f) for f in F])
