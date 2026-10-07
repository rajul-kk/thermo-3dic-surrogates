"""Non-uniform grids (``non-uniform true``): mesh derivation, lint rules, Tmap reading, diff.

The recorded files under fixtures/nonuniform/*/recorded and fixtures/3dice_examples/recorded_steady_material are
what 3D-ICE 4.0 itself wrote for these inputs.
"""
import os
import shutil

import numpy as np
import pytest

from iceforge.check import check_stack
from iceforge.model import parse_stk, non_uniform_cells, Rect
from iceforge.run import collect_outputs, parse_tmap_nu, read_xyaxis

NU = "nonuniform"


def _copy(fixtures, tmp_path, name, recorded=True):
    dst = tmp_path / name
    shutil.copytree(os.path.join(fixtures, NU, name), dst)
    if recorded:
        for f in os.listdir(dst / "recorded"):
            shutil.copy(dst / "recorded" / f, dst / f)
    return str(dst / "s.stk")


def _codes(findings):
    return sorted({f.code for f in findings})


# ------------------------------------------------------------------ mesh derivation

@pytest.mark.parametrize("name", ["conforming", "nonconforming"])
def test_mesh_matches_the_xyaxis_files_3dice_wrote(fixtures, name):
    st = parse_stk(os.path.join(fixtures, NU, name, "s.stk"))
    assert st.dims.non_uniform
    for se in st.stack:
        rec = read_xyaxis(os.path.join(fixtures, NU, name, "recorded", f"xyaxis_{se.instance}.txt"))
        cells = non_uniform_cells(st, se)
        mine = np.array([[c.x, c.y, c.l, c.w] for c in cells])
        assert mine.shape == rec.shape
        assert np.abs(mine - rec).max() < 0.011          # the file has two decimals


def test_mesh_matches_the_shipped_steady_example(fixtures):
    d = os.path.join(fixtures, "3dice_examples")
    st = parse_stk(os.path.join(d, "example_steady_material.stk"))
    rec = read_xyaxis(os.path.join(d, "recorded_steady_material", "xyaxis_MEMORY_DIE.txt"))
    cells = non_uniform_cells(st, st.stack[0])
    assert len(cells) == 400                              # 4 elements x discretization 10 x 10
    assert np.abs(np.array([[c.x, c.y, c.l, c.w] for c in cells]) - rec).max() < 0.011


def test_mesh_counts_of_the_shipped_transient_example(fixtures):
    st = parse_stk(os.path.join(fixtures, "3dice_examples", "example_transient_nonuniform.stk"))
    n = {se.instance: len(non_uniform_cells(st, se)) for se in st.stack}
    # TOP_DIE: 4 elements x (stack-line discretization 4 5); BOTTOM_DIE: 1x1 + 2x3 + 3x4 + default (10 x 10);
    # CONN_TO_PCB: plain layer, discretization 2 2
    assert n == {"TOP_DIE": 4 * 20, "BOTTOM_DIE": 1 + 6 + 12 + 100, "CONN_TO_PCB": 4}


def test_cell_order_is_x_fastest_within_an_element(fixtures):
    st = parse_stk(os.path.join(fixtures, NU, "nonconforming", "s.stk"))
    cells = non_uniform_cells(st, st.stack[0])
    a = cells[:160]                                       # element A: discretization 8 x 20
    assert [c.x for c in a[:8]] == [500.0 * i for i in range(8)] and {c.y for c in a[:8]} == {0.0}
    assert a[8].y == 500.0 and a[8].x == 0.0
    b = cells[160:]                                       # element B: die default 10 x 10 -> 600 x 1000 cells
    assert (b[0].x, b[0].l, b[0].w) == (4000.0, 600.0, 1000.0)


def test_only_the_first_rectangle_of_an_element_is_meshed(tmp_path):
    (tmp_path / "a.flp").write_text("e :\n rectangle (0, 0, 1000, 1000) ;\n rectangle (2000, 0, 1000, 1000) ;\n"
                                    " power values 1.0 ;\n")
    (tmp_path / "s.stk").write_text(
        "material M :\n thermal conductivity 1e-4 ;\n volumetric heat capacity 1e-12 ;\n"
        "top heat sink :\n heat transfer coefficient 1e-8 ;\n temperature 300 ;\n"
        "dimensions :\n chip length 1000, width 1000 ;\n cell length 500, width 500 ;\n non-uniform true ;\n"
        "die D :\n source 10 M ;\nstack :\n die X D floorplan \"a.flp\" ;\n"
        "solver :\n steady ;\n initial temperature 300 ;\n")
    st = parse_stk(str(tmp_path / "s.stk"))
    cells = non_uniform_cells(st, st.stack[0])
    assert len(cells) == 4 and max(c.x1 for c in cells) == 1000.0
    assert "N004" in _codes(check_stack(st))


# ------------------------------------------------------------------ lint rules

def test_clean_non_uniform_models_have_no_findings_beyond_the_note(fixtures):
    for name in ("conforming", "nonconforming"):
        f = check_stack(parse_stk(os.path.join(fixtures, NU, name, "s.stk")))
        assert _codes(f) == ["I001"]
        assert "TOPD 400" in f[0].message or "TOPD 260" in f[0].message
    f = check_stack(parse_stk(os.path.join(fixtures, "3dice_examples", "example_transient_nonuniform.stk")))
    assert not any(x.severity in ("error", "warning") for x in f)


def test_uniform_off_grid_rules_are_not_applied_to_non_uniform_models(fixtures, tmp_path):
    stk = _copy(fixtures, tmp_path, "conforming", recorded=False)
    s = open(stk).read().replace("dimensions :", "dimensions :", 1)
    # a floorplan edge at 1394.7 would be S001/S002 on a regular grid; here it is a cell edge by construction
    flp = open(os.path.join(os.path.dirname(stk), "die.flp")).read().replace("4000", "1394.7", 3)
    open(os.path.join(os.path.dirname(stk), "die.flp"), "w").write(flp)
    codes = _codes(check_stack(parse_stk(stk)))
    assert not ({"S001", "S002", "S003", "S004"} & set(codes))


def test_layout_is_flagged_as_ignored(fixtures, tmp_path):
    stk = _copy(fixtures, tmp_path, "conforming", recorded=False)
    d = os.path.dirname(stk)
    open(os.path.join(d, "fp.lyt"), "w").write("SI :\n rectangle (0, 0, 5000, 10000) ;\n")
    s = open(stk).read().replace("   material SI ;\n\ndie D", "   material GAP ;\n   layout \"fp.lyt\" ;\n\ndie D")
    open(stk, "w").write(s)
    f = [x for x in check_stack(parse_stk(stk)) if x.code == "N001"]
    assert len(f) == 1 and f[0].severity == "error"
    assert f[0].detail["n_rectangles"] == 1 and f[0].detail["stack_element"] == "TOPD"


def test_floorplan_that_does_not_cover_the_die_is_flagged(fixtures, tmp_path):
    stk = _copy(fixtures, tmp_path, "conforming", recorded=False)
    d = os.path.dirname(stk)
    lines = open(os.path.join(d, "die.flp")).read().splitlines()
    open(os.path.join(d, "die.flp"), "w").write("\n".join(lines[:6]) + "\n")      # drop element B
    f = [x for x in check_stack(parse_stk(stk)) if x.code == "N002"]
    assert len(f) == 1 and f[0].detail["covered_fraction"] == pytest.approx(0.4)


def test_bottom_only_heat_sink_is_flagged(fixtures, tmp_path):
    stk = _copy(fixtures, tmp_path, "conforming", recorded=False)
    s = open(stk).read().replace("top heat sink", "bottom heat sink")
    open(stk, "w").write(s)
    f = [x for x in check_stack(parse_stk(stk)) if x.code == "N003"]
    assert len(f) == 1 and f[0].severity == "error" and "top heat sink" in f[0].message
    assert "N005" not in _codes(check_stack(parse_stk(stk)))


def test_no_heat_sink_is_flagged(fixtures):
    f = check_stack(parse_stk(os.path.join(fixtures, NU, "repro_topsink_null", "crash_no_sink.stk")))
    assert "N003" in _codes(f) and "N005" not in _codes(f)


def test_top_only_heat_sink_is_clean(fixtures):
    f = check_stack(parse_stk(os.path.join(fixtures, NU, "repro_topsink_null", "ok_top_only.stk")))
    assert not ({"N003", "N005"} & set(_codes(f)))


def test_top_and_bottom_heat_sinks_are_flagged_as_the_offset_bug(fixtures):
    base = os.path.join(fixtures, NU, "repro_topsink_null")
    for p in (("matrix", "nu_both.stk"), ("issue3", "i_nu_both.stk")):
        f = check_stack(parse_stk(os.path.join(base, *p)))
        assert "N005" in _codes(f) and "N003" not in _codes(f)
        assert [x for x in f if x.code == "N005"][0].severity == "error"
    # the same stacks on a uniform grid are fine
    f = check_stack(parse_stk(os.path.join(base, "matrix", "u_both.stk")))
    assert not ({"N003", "N005"} & set(_codes(f)))


def test_layout_ignored_on_the_strip_repro(fixtures):
    f = check_stack(parse_stk(os.path.join(fixtures, NU, "repro_lyt_ignore_strip", "nu_lyt.stk")))
    n1 = [x for x in f if x.code == "N001"]
    assert len(n1) == 1 and n1[0].severity == "error"
    f = check_stack(parse_stk(os.path.join(fixtures, NU, "repro_lyt_ignore_strip", "nu_gap.stk")))
    assert "N001" not in _codes(f)


# ------------------------------------------------------------------ the real solver on the repro fixtures
# These pin the CURRENT (buggy) 3D-ICE 4.0 behaviour that the rules above describe.  If one of them fails after a
# 3D-ICE upgrade, the bug may be fixed: re-check the rule and update the rule table / README.

def _solve(fixtures, tmp_path, sub, stk):
    from iceforge import backends
    dst = tmp_path / sub
    shutil.copytree(os.path.join(fixtures, NU, sub), dst)
    r = backends.run_emulator(backends.detect("auto"), str(dst / stk), timeout=300)
    return r.returncode, dst


def _numbers(path):
    out = []
    for line in open(path):
        for tok in line.split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return np.array(out)


@pytest.mark.solver
@pytest.mark.parametrize("stk", ["crash_bottom_only.stk", "crash_no_sink.stk"])
def test_solver_segfaults_without_a_top_sink(fixtures, tmp_path, stk):
    rc, _ = _solve(fixtures, tmp_path, "repro_topsink_null", stk)
    assert rc in (139, 11, -11), "3D-ICE no longer segfaults without a top heat sink: revisit N003"


@pytest.mark.solver
def test_solver_top_only_runs(fixtures, tmp_path):
    rc, _ = _solve(fixtures, tmp_path, "repro_topsink_null", "ok_top_only.stk")
    assert rc == 0


@pytest.mark.solver
def test_solver_non_uniform_with_both_sinks_adds_a_constant_offset(fixtures, tmp_path):
    rc, d = _solve(fixtures, tmp_path / "a", "repro_topsink_null", "matrix/nu_both.stk")
    assert rc == 0
    rc, d2 = _solve(fixtures, tmp_path / "b", "repro_topsink_null", "matrix/nu_top.stk")
    assert rc == 0
    both = _numbers(d / "matrix" / "nu_both_t.txt")
    top = _numbers(d2 / "matrix" / "nu_top_t.txt")
    off = both - top
    assert off.mean() > 1000.0, "no offset any more: upstream #3 may be fixed, revisit N005"
    assert np.ptp(off) < 1e-3 * off.mean()          # a constant: the field shape is intact


@pytest.mark.solver
def test_solver_ignores_the_layout_in_non_uniform_mode(fixtures, tmp_path):
    rc1, d = _solve(fixtures, tmp_path / "a", "repro_lyt_ignore_strip", "nu_lyt.stk")
    rc2, d2 = _solve(fixtures, tmp_path / "b", "repro_lyt_ignore_strip", "nu_gap.stk")
    assert rc1 == rc2 == 0
    assert np.array_equal(_numbers(d / "nu_lyt_t.txt"), _numbers(d2 / "nu_gap_t.txt")),         "Tmaps now differ: 3D-ICE honours the layout, revisit N001"
    # while on a uniform grid the same layout matters (5.5 K)
    rc3, d3 = _solve(fixtures, tmp_path / "c", "repro_lyt_ignore_strip", "u_lyt.stk")
    rc4, d4 = _solve(fixtures, tmp_path / "d", "repro_lyt_ignore_strip", "u_gap.stk")
    assert rc3 == rc4 == 0
    assert np.abs(_numbers(d3 / "u_lyt_t.txt") - _numbers(d4 / "u_gap_t.txt")).max() > 1.0


def test_non_integer_discretization_is_flagged(fixtures, tmp_path):
    stk = _copy(fixtures, tmp_path, "conforming", recorded=False)
    s = open(stk).read().replace("discretization 20 20", "discretization 20.5 20")
    open(stk, "w").write(s)
    assert "N004" in _codes(check_stack(parse_stk(stk)))


# ------------------------------------------------------------------ Tmap reading

def test_parse_tmap_nu_counts_values(fixtures):
    f = os.path.join(fixtures, NU, "conforming", "recorded", "tmap.txt")
    T, F = parse_tmap_nu(f, 400)
    assert T.shape == (400,) and not F and 300 < T.min() < T.max() < 320
    T, F = parse_tmap_nu(f, 399)
    assert T is None and F[0].code == "P001"


def test_collect_outputs_writes_cells_and_checks_xyaxis(fixtures, tmp_path):
    stk = _copy(fixtures, tmp_path, "nonconforming")
    st = parse_stk(stk)
    out = tmp_path / "out"
    out.mkdir()
    res = {r.instance: r for r in collect_outputs(st, str(out))}
    assert set(res) == {"TOPD", "BOT"}
    r = res["TOPD"]
    assert r.shape == (260,) and not [f for f in r.findings if f.severity != "info"]
    z = np.load(r.npz)
    assert z["T"].shape == (260,) and z["cells"].shape == (260, 4) and str(z["grid"]) == "non-uniform"
    assert r.hottest["under"] == "A"                     # element A has the higher power density
    assert res["BOT"].shape == (400,)


def test_a_mismatching_xyaxis_file_is_reported(fixtures, tmp_path):
    stk = _copy(fixtures, tmp_path, "conforming")
    p = os.path.join(os.path.dirname(stk), "xyaxis_TOPD.txt")
    a = read_xyaxis(p)
    a[5, 0] += 50.0
    np.savetxt(p, a, fmt="%.2f", delimiter="\t")
    st = parse_stk(stk)
    out = tmp_path / "out"
    out.mkdir()
    res = {r.instance: r for r in collect_outputs(st, str(out))}
    assert any(f.code == "P003" for f in res["TOPD"].findings)


# ------------------------------------------------------------------ diff (needs scipy)

scipy = pytest.importorskip("scipy")

from iceforge import diff as D  # noqa: E402
from iceforge import refsolve as R  # noqa: E402


def _diff(stk, tmp_path):
    st = parse_stk(stk)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    collect_outputs(st, str(out))
    return D.run_diff(stk, r=4, ice_npz=str(out))


def test_diff_conforming_mesh_agrees_with_the_reference(fixtures, tmp_path):
    rep = _diff(_copy(fixtures, tmp_path, "conforming"), tmp_path)
    assert rep["verdict"] == "AGREE"
    for c in rep["layers"]:
        assert c["grid"] == "non-uniform" and c["n_cells"] == 400 and c["n_mesh_mismatch_cells"] == 0
        assert c["max_abs_dT"] < 0.15                      # measured 0.10 K of a 7.5 K rise


def test_diff_flags_cells_that_do_not_conform_to_the_other_mesh(fixtures, tmp_path):
    rep = _diff(_copy(fixtures, tmp_path, "nonconforming"), tmp_path)
    assert rep["verdict"] == "DISAGREE (die-edge)" or rep["layers"][0]["verdict"] == "DISAGREE (die-edge)"
    for c in rep["layers"]:
        assert c["max_abs_dT"] > 0.5                       # measured 0.78 / 0.87 K of a 7.5 K rise
        assert c["n_flagged"] > 0 and c["n_flagged_interior"] == 0
        assert c["n_mesh_mismatch_cells"] > 0
    # peak rise barely moves: the error is local, which is why it is easy to miss
    assert abs(rep["layers"][0]["peak_diff"]) < 0.05


def test_diff_on_the_shipped_steady_non_uniform_example(fixtures, tmp_path):
    d = tmp_path / "ex"
    shutil.copytree(os.path.join(fixtures, "3dice_examples"), d)
    shutil.copy(d / "recorded_steady_material" / "output2.txt", d / "output2.txt")
    stk = str(d / "example_steady_material.stk")
    rep = _diff(stk, tmp_path)
    assert rep["verdict"] == "AGREE" and rep["layers"][0]["n_cells"] == 400
    assert rep["layers"][0]["max_abs_dT"] < 0.1


def test_cell_average_is_exact():
    xe = np.linspace(0, 10, 11)
    F = np.add.outer(np.arange(10.0), 10.0 * np.arange(10.0))      # F[i, j] = i + 10 j
    cells = np.array([[0.0, 0.0, 10.0, 10.0], [2.5, 0.0, 5.0, 10.0], [0.0, 0.0, 1.0, 1.0]])
    got = R.cell_average(F, xe, xe, cells)
    assert got[0] == pytest.approx(F.mean())
    assert got[1] == pytest.approx(49.5)       # x in [2.5, 7.5] -> mean i = 4.5; y -> mean 10 j = 45
    assert got[2] == F[0, 0]


def test_mesh_mismatch_mask_on_a_hand_built_case():
    a = np.array([[0, 0, 5, 10], [5, 0, 5, 10]], float)           # two 5 x 10 cells
    b = np.array([[0, 0, 4, 10], [4, 0, 6, 10]], float)           # edge at x = 4 cuts a[0]
    m = D.mesh_mismatch_mask(a, [b])
    assert m.tolist() == [True, True]                              # a[0] is cut; a[1]'s edge x=5 cuts b[1]
    assert not D.mesh_mismatch_mask(a, [a.copy()]).any()


# ------------------------------------------------------------------ solver

@pytest.mark.solver
def test_3dice_reproduces_the_recorded_non_uniform_output(fixtures, tmp_path):
    from iceforge.run import run_model
    stk = _copy(fixtures, tmp_path, "conforming", recorded=False)
    code, rep = run_model(stk, outdir=str(tmp_path / "out"), log=lambda *a: None)
    assert code == 0 and rep["status"] == "ok"
    T = np.load(rep["tmaps"][0]["npz"])["T"]
    rec, _ = parse_tmap_nu(os.path.join(fixtures, NU, "conforming", "recorded", "tmap.txt"), 400)
    assert np.abs(T - rec).max() < 2e-3                           # the file has three decimals
