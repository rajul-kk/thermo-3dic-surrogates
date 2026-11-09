import os

from iceforge.check import check_stack, exit_code
from iceforge.model import parse_stk
from iceforge.snap import snap


def stk(fixtures, name):
    return os.path.join(fixtures, "repro", name, "s.stk")


def test_snap_nearest_fixes_misaligned(fixtures, tmp_path):
    src = stk(fixtures, "misaligned")
    before = open(os.path.join(os.path.dirname(src), "die.flp")).read()
    out, moves, _ = snap(src, str(tmp_path / "o"), "nearest")
    assert open(os.path.join(os.path.dirname(src), "die.flp")).read() == before   # input untouched
    assert len(moves) == 2                                                        # flp element + lyt rect
    assert all(m.new[:2] == (1500.0, 1500.0) and m.new[2:] == (6000.0, 6000.0) for m in moves)
    F = check_stack(parse_stk(out))
    assert F == [], [str(f) for f in F]


def test_snap_outward_grows_layout_only(fixtures, tmp_path):
    out, moves, _ = snap(stk(fixtures, "misaligned"), str(tmp_path / "o"), "outward")
    assert len(moves) == 1 and moves[0].new == (1250.0, 1250.0, 6250.0, 6250.0)
    F = check_stack(parse_stk(out))
    assert exit_code(F) == 0 and "S001" not in [f.code for f in F]
    st = parse_stk(out)
    assert list(st.floorplans.values())[0].elements[0].rects[0].x == 1394.7     # power untouched


def test_snap_aligned_is_noop(fixtures, tmp_path):
    out, moves, _ = snap(stk(fixtures, "aligned"), str(tmp_path / "o"))
    assert moves == []


def test_snap_nearest_falls_back_when_out_of_chip(tmp_path):
    (tmp_path / "d.flp").write_text("a :\n position 8900, 0 ;\n dimension 1000, 1000 ;\n power values 1 ;\n")
    (tmp_path / "s.stk").write_text("""
material SI : thermal conductivity 1e-4 ; volumetric heat capacity 1e-12 ;
dimensions : chip length 9800, width 9800 ; cell length 200, width 200 ;
die D : source 10 SI ;
stack : die T D floorplan "d.flp" ;
solver : steady ;
""")
    out, moves, _ = snap(str(tmp_path / "s.stk"), str(tmp_path / "o"))
    st = parse_stk(out)
    r = list(st.floorplans.values())[0].elements[0].rects[0]
    assert r.x1 <= 9800 and r.x % 200 == 0


def test_snap_names_do_not_collide(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "d.flp").write_text("e :\n position 0, 0 ;\n dimension 100, 100 ;\n power values 1 ;\n")
    (tmp_path / "s.stk").write_text("""
material SI : thermal conductivity 1e-4 ; volumetric heat capacity 1e-12 ;
dimensions : chip length 1000, width 1000 ; cell length 100, width 100 ;
die D : source 10 SI ;
stack : die T1 D floorplan "a/d.flp" ; die T2 D floorplan "b/d.flp" ;
solver : steady ;
""")
    out, _, _ = snap(str(tmp_path / "s.stk"), str(tmp_path / "o"))
    st = parse_stk(out)
    assert len(st.floorplans) == 2 and not st.missing_files
