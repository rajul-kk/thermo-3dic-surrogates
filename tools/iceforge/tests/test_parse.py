import glob
import os

import pytest

from iceforge.model import parse_stk, parse_flp, parse_lyt, tokenize, strip_comments, summary, to_dict


def test_comments_stripped_but_not_in_strings():
    t = strip_comments('a // x\nb /* y\n z */ c "p//q" ;')
    assert 'x' not in t and 'y' not in t and '"p//q"' in t
    assert tokenize('chip length 10.5e3 , width 4 ;') == ['chip', 'length', '10.5e3', ',', 'width', '4', ';']


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                                               "fixtures", "3dice_examples", "*.stk"))))
def test_every_3dice_example_parses(path):
    st = parse_stk(path)
    assert st.dims.chip_l > 0 and st.dims.cell_l > 0
    assert st.stack and st.dies
    assert not st.missing_files, st.missing_files
    assert st.floorplans
    assert summary(st)
    assert to_dict(st)["dims"]["n_cols"] == st.dims.n_cols


def test_example_steady_details(fixtures):
    st = parse_stk(os.path.join(fixtures, "3dice_examples", "example_steady.stk"))
    assert (st.dims.chip_l, st.dims.chip_w, st.dims.cell_l) == (10000, 10000, 100)
    assert st.dims.n_cols == 100 and not st.dims.non_uniform
    assert st.materials["SILICON"].k == [1.30e-4]
    assert st.heatsinks[0].side == "top" and st.heatsinks[0].temperature == 300
    assert [s.instance for s in st.stack][0] == "MEMORY_DIE"
    assert st.stack[0].floorplan_path.endswith("mem.flp")
    tm = [o for o in st.outputs if o.kind == "Tmap"]
    assert tm and tm[0].instance == "CORE_DIE" and tm[0].when == "final"
    assert st.solver.mode == "steady"
    top = st.dies["TOP_IC"]
    assert [l.is_source for l in top.layers] == [True, False]
    assert st.sink_temperature == 300


def test_non_uniform_is_reported_not_fatal(fixtures):
    st = parse_stk(os.path.join(fixtures, "3dice_examples", "example_steady_material.stk"))
    assert st.dims.non_uniform
    assert any("non-uniform grid: off-grid checks skipped" in w for w in st.warnings)
    assert st.materials["SILICON"].k == [1.3e-4] * 3        # anisotropic form
    fp = list(st.floorplans.values())[0]
    assert fp.elements[0].rects[0].l == 5000


def test_flp_rectangle_list_and_lyt(fixtures):
    d = os.path.join(fixtures, "repo_generated")
    fp = parse_flp(os.path.join(d, "floorplan_layer1.flp"))
    assert [e.name for e in fp.elements] == ["cpu", "hbm"]
    assert len(fp.elements[1].rects) == 2 and fp.elements[1].power == [15.0]
    ly = parse_lyt(os.path.join(d, "layout_footprint_active.lyt"))
    r = ly.shapes["silicon"][0]
    assert (r.x, r.y, r.l, r.w) == (2000, 1000, 8000, 6000)


def test_repo_generated_stk(fixtures):
    st = parse_stk(os.path.join(fixtures, "repo_generated", "stack.stk"))
    assert (st.dims.chip_l, st.dims.chip_w) == (12000, 8000)
    assert (st.dims.n_cols, st.dims.n_rows) == (48, 32)
    src = st.layers["type_layer_2_src"]
    assert src.layout_path.endswith("layout_footprint_active.lyt") and os.path.isabs(src.layout_path)
    assert len(st.dies["type_die_2"].layers) == 3
    assert not st.missing_files
    assert st.sink_temperature == 318.15


def test_missing_files_recorded(tmp_path):
    (tmp_path / "m.stk").write_text(
        'material S : thermal conductivity 1e-4 ; volumetric heat capacity 1e-12 ;\n'
        'dimensions : chip length 100, width 100 ; cell length 10, width 10 ;\n'
        'die D : source 10 S ;\nstack : die X D floorplan "nope.flp" ;\nsolver : steady ;\n')
    st = parse_stk(str(tmp_path / "m.stk"))
    assert st.missing_files and st.missing_files[0][0] == "floorplan"
