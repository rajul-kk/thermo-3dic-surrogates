"""HotSpot as a third solver: translation, refusals, field handling, and (marked) real HotSpot runs."""
import os
import re
import shutil

import numpy as np
import pytest

from iceforge import hotspot as H
from iceforge.model import parse_stk
from iceforge import refsolve as R

scipy = pytest.importorskip("scipy")

from iceforge import diff as D  # noqa: E402


def _stk(repro, name):
    return os.path.join(repro, name, "s.stk")


def _write(tmp_path, stk_text, flp_text=None):
    (tmp_path / "s.stk").write_text(stk_text)
    if flp_text is not None:
        (tmp_path / "d.flp").write_text(flp_text)
    return str(tmp_path / "s.stk")


BASE = """\
material SI :
   thermal conductivity     1.4800e-04 ;
   volumetric heat capacity 1.628e-12 ;
{sinks}
dimensions :
   chip length {L} , width {W} ;
   cell length 250.0 , width 250.0 ;
{extra}
layer SUB :
   height 300.0 ;
   material SI ;

die D :
   source 50 SI ;

stack :
   die TOPD D floorplan "d.flp" ;
   layer BOT SUB ;

solver:
   steady ;
   initial temperature 300.0 ;

output:
   Tmap (TOPD, "t.txt", final) ;
"""
BOTTOM = "bottom heat sink :\n   heat transfer coefficient 2e-8 ;\n   temperature 300 ;\n"
TOP = "top heat sink :\n   heat transfer coefficient 2e-8 ;\n   temperature 300 ;\n"
FLP = "blk :\n position 1000, 1000 ;\n dimension 2000, 2000 ;\n power values 5.0 ;\n"


def _model(tmp_path, sinks=BOTTOM, L=4000, W=4000, extra="", flp=FLP):
    return parse_stk(_write(tmp_path, BASE.format(sinks=sinks, L=L, W=W, extra=extra), flp))


# ------------------------------------------------------------------ translation

def test_translate_misaligned_repro(repro):
    c = H.translate(parse_stk(_stk(repro, "misaligned")))
    assert len(c.layers) == 4                       # two half-layers per 3D-ICE layer
    assert [(L.instance, L.role) for L in c.layers] == [("TOPD", "far"), ("TOPD", "mid"), ("BOT", "far"),
                                                        ("BOT", "mid")]
    assert c.sink_side == "bottom" and c.heterogeneous
    assert (c.rows, c.cols) == (64, 64) and (c.n_cols, c.n_rows) == (40, 40)   # power-of-two grid
    assert c.lcf_index(0) == 1 and c.lcf_index(1) == 3
    # power: all of it (10 W) in the mid half of the source layer, split over the tiles by area
    names, vals = [l.split("\t") for l in c.files["power.ptrace"].strip().splitlines()]
    assert len(names) == c.layers[1].n_units == 9 and sum(float(v) for v in vals) == pytest.approx(10.0)
    assert max(float(v) for v in vals) == pytest.approx(10.0)         # the die is one tile of the 3 x 3 tiling
    # layout tiles carry their own resistivity (7-column flp lines); the centre tile is silicon
    rows = [l.split("\t") for l in c.files["L1.flp"].splitlines() if not l.startswith("#")]
    assert all(len(r) == 7 for r in rows)
    res = sorted(float(r[6]) for r in rows)
    assert res[0] == pytest.approx(1 / 148.0)
    # the film: 1 / (h A) with the sink size, divided by 1.5 for the -detailed_3D film factor
    argv = dict(zip(c.argv[::2], c.argv[1::2]))
    sink = 0.01 * H.RIM_FACTOR ** 2
    assert float(argv["-r_convec"]) == pytest.approx(1 / (2e4 * sink * sink) / H.DET_FILM_FACTOR)
    assert argv["-detailed_3D"] == "on" and argv["-grid_rows"] == "64" and argv["-model_secondary"] == "0"


def test_homogeneous_model_uses_the_plain_mode(repro):
    c = H.translate(parse_stk(_stk(repro, "nolayout")))
    argv = dict(zip(c.argv[::2], c.argv[1::2]))
    assert not c.heterogeneous and argv["-detailed_3D"] == "off"
    assert all(len(l.split("\t")) == 5 for l in c.files["L1.flp"].splitlines() if not l.startswith("#"))
    assert float(argv["-r_convec"]) == pytest.approx(1 / (2e4 * (0.01 * H.RIM_FACTOR ** 2) ** 2))


def test_lcf_order_runs_away_from_the_sink(tmp_path):
    bot = H.translate(_model(tmp_path, BOTTOM))
    assert [L.instance for L in bot.layers] == ["TOPD", "TOPD", "BOT", "BOT"]
    top = H.translate(_model(tmp_path, TOP))
    assert [L.instance for L in top.layers] == ["BOT", "BOT", "TOPD", "TOPD"] and top.sink_side == "top"
    assert [L.thickness_m for L in top.layers] == pytest.approx([150e-6, 150e-6, 25e-6, 25e-6])


def test_grid_size_is_the_next_power_of_two_and_overridable(repro):
    st = parse_stk(_stk(repro, "nolayout"))
    assert H._pow2_at_least(100) == 128 and H._pow2_at_least(64) == 64 and H._pow2_at_least(1) == 1
    assert H.translate(st, r=2).rows == 128           # 40 x 2 = 80 -> 128
    assert H.translate(st, grid=32).cols == 32
    with pytest.raises(H.HotspotRefused):
        H.translate(st, grid=50)


def test_tiles_split_a_power_element_at_layout_edges(tmp_path):
    (tmp_path / "fp.lyt").write_text("SI :\n rectangle (1500, 1500, 1000, 1000) ;\n")
    stk = BASE.format(sinks=BOTTOM, L=4000, W=4000, extra="").replace(
        "die D :\n   source 50 SI ;", "layer SRC :\n   height 50 ;\n   material SI ;\n   layout \"fp.lyt\" ;\n\ndie D :\n   source SRC ;")
    (tmp_path / "s.stk").write_text(stk)
    (tmp_path / "d.flp").write_text(FLP)
    c = H.translate(parse_stk(str(tmp_path / "s.stk")))
    names, vals = [l.split("\t") for l in c.files["power.ptrace"].strip().splitlines()]
    vals = [float(v) for v in vals]
    assert sum(vals) == pytest.approx(5.0)
    assert len([v for v in vals if v > 0]) == 9      # element 1000..3000 cut at 1500 and 2500 -> 3 x 3 powered tiles
    centre = max(vals)
    assert centre == pytest.approx(5.0 * 1000 * 1000 / (2000 * 2000))


# ------------------------------------------------------------------ refusals

def test_refuses_both_sinks(tmp_path):
    with pytest.raises(H.HotspotRefused, match="both a top and a bottom"):
        H.translate(_model(tmp_path, BOTTOM + TOP))


def test_refuses_non_square_chip(tmp_path):
    with pytest.raises(H.HotspotRefused, match="square"):
        H.translate(_model(tmp_path, BOTTOM, L=4000, W=3000, flp="blk :\n position 0, 0 ;\n dimension 1000, 1000 ;\n"
                                                                 " power values 1.0 ;\n"))


def test_refuses_anisotropic_conductivity(tmp_path):
    txt = BASE.format(sinks=BOTTOM, L=4000, W=4000, extra="").replace(
        "thermal conductivity     1.4800e-04", "thermal conductivity     1.4800e-04, 1.4800e-04, 1.0e-04")
    with pytest.raises(H.HotspotRefused, match="anisotropic"):
        H.translate(parse_stk(_write(tmp_path, txt, FLP)))


def test_refuses_what_the_reference_refuses(tmp_path):
    with pytest.raises(R.RefusedModel, match="transient"):
        txt = BASE.format(sinks=BOTTOM, L=4000, W=4000, extra="").replace("steady ;", "transient step 0.1, slot 1 ;")
        H.translate(parse_stk(_write(tmp_path, txt, FLP)))


def test_refuses_non_uniform_grids(tmp_path):
    with pytest.raises(H.HotspotRefused, match="non-uniform"):
        H.translate(_model(tmp_path, TOP, extra="   non-uniform true ;"))


def test_refuses_overlapping_layout_rectangles(tmp_path):
    (tmp_path / "fp.lyt").write_text("SI :\n rectangle (0, 0, 2000, 2000) ;\n rectangle (1000, 1000, 2000, 2000) ;\n")
    stk = BASE.format(sinks=BOTTOM, L=4000, W=4000, extra="").replace(
        "die D :\n   source 50 SI ;", "layer SRC :\n   height 50 ;\n   material SI ;\n   layout \"fp.lyt\" ;\n\ndie D :\n   source SRC ;")
    (tmp_path / "s.stk").write_text(stk)
    (tmp_path / "d.flp").write_text(FLP)
    with pytest.raises(H.HotspotRefused, match="overlap"):
        H.translate(parse_stk(str(tmp_path / "s.stk")))


def test_diff_reports_a_refusal_instead_of_raising(tmp_path):
    st = _model(tmp_path, BOTTOM + TOP)
    rep = D.hotspot_opinion(st, None, [], 0.02)
    assert rep["status"] == "refused" and "both a top and a bottom" in rep["reason"]
    assert "refused" in D.format_hotspot(rep)[0]


# ------------------------------------------------------------------ fields

def test_parse_grid_steady_puts_row_zero_at_the_top():
    case = H.HotspotCase(files={}, layers=[], rows=2, cols=4, r=1, n_cols=4, n_rows=2, side_m=1.0, ambient=0.0,
                         sink_side="bottom", argv=[])
    # idx = row * cols + col; temperature = 10 * row + col
    text = "Layer 0:\n" + "".join(f"{r * 4 + c}\t{10 * r + c}\n" for r in range(2) for c in range(4))
    T = H.parse_grid_steady(text, case)[0]
    assert T.shape == (4, 2)                                    # (length index, width index)
    assert T[1, 1] == 1.0 and T[1, 0] == 11.0                   # row 0 is the top (largest y): width index 1
    assert T[3, 0] == 13.0 and T[0, 1] == 0.0


def test_remap_is_exact_and_conserves_the_mean():
    rng = np.random.default_rng(0)
    F = rng.random((64, 64))
    G = H.remap(F, 40, 40)
    assert G.shape == (40, 40) and G.mean() == pytest.approx(F.mean())
    assert H.remap(F, 64, 64) == pytest.approx(F)
    assert H.remap(F, 32, 32) == pytest.approx(F.reshape(32, 2, 32, 2).mean(axis=(1, 3)))


def test_find_hotspot_parses_the_executable_string(monkeypatch):
    assert H.find_hotspot("wsl /home/u/HotSpot/hotspot") == ("wsl", "/home/u/HotSpot/hotspot")
    assert H.find_hotspot("C:/tools/hotspot.exe")[0] == "native"
    monkeypatch.setenv("HOTSPOT_EXECUTABLE", "wsl /x/hotspot")
    assert H.find_hotspot() == ("wsl", "/x/hotspot")


# ------------------------------------------------------------------ the 3-way verdict

def test_pair_removes_only_an_affine_difference():
    rng = np.random.default_rng(1)
    b = 300 + rng.random((20, 20)) * 10
    a = 2.0 * b + 3.0
    p = D._pair(a, b, 0.2)
    assert p["max_abs"] > 100 and p["fit_max_abs"] < 1e-9
    assert p["fit_scale"] == pytest.approx(2.0) and p["fit_offset_K"] == pytest.approx(3.0)
    bump = a.copy()
    bump[3, 3] += 5.0                                       # a local error survives the fit
    assert D._pair(bump, b, 0.2)["fit_max_abs"] > 4.0


def _p(fit):
    return dict(fit_max_abs=fit)


def test_hotspot_side_table():
    tol = 0.2
    # 3D-ICE and the reference disagree (the artefact case)
    assert D.hotspot_side(_p(0.05), _p(7.0), _p(7.0), False, tol) == "HotSpot sides with the reference"
    assert D.hotspot_side(_p(7.0), _p(0.05), _p(7.0), False, tol) == "HotSpot sides with 3D-ICE"
    assert D.hotspot_side(_p(1.0), _p(7.0), _p(7.0), False, tol) == \
        "HotSpot is closer to the reference (neither within tolerance)"
    assert D.hotspot_side(_p(3.0), _p(3.5), _p(7.0), False, tol) == "inconclusive"
    # 3D-ICE and the reference agree
    assert D.hotspot_side(_p(0.1), _p(0.3), _p(0.28), True, tol) == "all three agree"
    assert D.hotspot_side(_p(0.1), _p(2.0), _p(0.28), True, tol) == \
        "HotSpot agrees with the reference, not with 3D-ICE"
    assert D.hotspot_side(_p(2.0), _p(2.0), _p(0.28), True, tol) == "HotSpot differs from both (its own modelling difference)"


# ------------------------------------------------------------------ real HotSpot (marker: hotspot)

def _ref_and_hs(stk_path, r=2):
    st = parse_stk(stk_path)
    case = H.translate(st)
    fields = H.hotspot_fields(case)
    model = case.model
    li = model.tmap_layer(st.outputs[0].instance)
    sol = R.solve(model, r=r)
    ref = R.block_average(sol.layer_field(li), r)
    hs = H.remap(fields[case.lcf_index(li)], *ref.shape)
    return hs, ref, sol.t_ref


@pytest.mark.hotspot
@pytest.mark.parametrize("name", ["aligned", "misaligned", "nolayout", "snapped"])
def test_hotspot_matches_the_reference_on_the_repro_cases(repro, name):
    hs, ref, t0 = _ref_and_hs(_stk(repro, name))
    rise = ref.max() - t0
    # measured: 0.058 K (misaligned), 0.125 K (aligned) of an ~11 K rise; allow 2 %
    assert np.abs(hs - ref).max() < 0.02 * rise
    assert abs(hs.max() - ref.max()) < 0.02 * rise


@pytest.mark.hotspot
def test_hotspot_does_not_see_the_die_edge_artefact(repro):
    """3D-ICE's misaligned answer is 14.562 K above ambient; HotSpot, like the reference, stays at ~11.06 K."""
    hs, ref, t0 = _ref_and_hs(_stk(repro, "misaligned"))
    assert hs.max() - t0 < 11.5          # not 14.5


@pytest.mark.hotspot
def test_preflight_passes_on_a_superlu_build():
    err, det = H.preflight()
    assert err is None and det is None


def test_geometry1_fixture_translates(fixtures):
    c = H.translate(parse_stk(os.path.join(fixtures, "geometry1", "s.stk")))
    assert len(c.layers) == 24 and not c.heterogeneous and (c.rows, c.cols) == (128, 128)
    assert c.sink_side == "bottom"
    vals = [float(v) for v in c.files["power.ptrace"].splitlines()[1].split("\t")]
    assert sum(vals) == pytest.approx(4 * 2.03125)
