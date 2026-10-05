"""Tests for the reference solver and `iceforge diff`."""
import json
import os

import numpy as np
import pytest

from iceforge import refsolve as R
from iceforge.model import Rect, parse_stk

scipy = pytest.importorskip("scipy")

from iceforge import diff as D  # noqa: E402
from iceforge.cli import main  # noqa: E402

# 3D-ICE 4.0 results for the repro cases (max rise above 300 K, see README)
ICE_RISE = {"aligned": 11.130, "misaligned": 14.562, "nolayout": 11.027, "snapped": 11.118}


# ------------------------------------------------------------------ exact geometry

def test_rect_overlap_area_is_exact():
    xe = np.arange(0.0, 5.0)           # cells of 1 um: edges 0..4
    ye = np.arange(0.0, 5.0)
    r = Rect(0.5, 1.25, 2.0, 1.5)      # x 0.5..2.5, y 1.25..2.75
    A = R.rect_overlap_area(xe, ye, r)
    assert A.sum() == pytest.approx(r.l * r.w)
    assert A[0, 1] == pytest.approx(0.5 * 0.75)    # cell x0..1, y1..2
    assert A[1, 1] == pytest.approx(1.0 * 0.75)
    assert A[2, 2] == pytest.approx(0.5 * 0.75)    # cell x2..3, y2..3
    assert A[3, :].sum() == 0.0


def test_area_fractions_partial_and_full_cells():
    xe = np.arange(0.0, 5.0)
    f = R.area_fractions(xe, xe, [Rect(0.5, 0.0, 2.0, 4.0)])
    assert f[0, 0] == pytest.approx(0.5)
    assert f[1, 3] == pytest.approx(1.0)
    assert f[2, 0] == pytest.approx(0.5)
    assert f[3, 0] == 0.0
    # clipped to 1 where rectangles overlap
    g = R.area_fractions(xe, xe, [Rect(0, 0, 2, 2), Rect(1, 1, 2, 2)])
    assert g.max() == pytest.approx(1.0)


def test_edge_crossing_mask_only_cells_cut_by_an_edge():
    xe = np.arange(0.0, 6.0) * 10      # 5 cells of 10
    m = D.edge_crossing_mask(xe, xe, [Rect(15, 10, 20, 30)])   # x 15..35, y 10..40
    # edges at x=15 (inside cell 1) and x=35 (cell 3); y edges on cell boundaries
    assert m[1, 1] and m[1, 2] and m[1, 3]
    assert m[3, 1] and m[3, 2] and m[3, 3]
    assert not m[1, 0] and not m[1, 4]            # y-span of the edge does not reach there
    assert not m[2].any()                         # cell 2 is wholly inside


# ------------------------------------------------------------------ 1D slab, closed form

SLAB = """\
blk :
   position 0.0, 0.0 ;
   dimension 1000.0, 1000.0 ;
   power values 1.0e-3 ;
"""

STK = """\
material SI :
   thermal conductivity {k1:e} ;
   volumetric heat capacity 1.6e-12 ;
material LOW :
   thermal conductivity {k2:e} ;
   volumetric heat capacity 1.6e-12 ;
bottom heat sink :
   heat transfer coefficient {h:e} ;
   temperature 300.0 ;
dimensions :
   chip length 1000.0 , width 1000.0 ;
   cell length 250.0 , width 250.0 ;
layer SUB :
   height {h2} ;
   material LOW ;
die D :
   source {h1} SI ;
stack :
   die TOP D floorplan "d.flp" ;
   layer BOT SUB ;
solver:
   steady ;
   initial temperature 300.0 ;
output:
   Tmap (TOP, "t.txt", final) ;
"""


def test_slab_matches_closed_form(tmp_path):
    k1, k2, h, h1, h2, P = 1.48e-4, 2.0e-5, 1.0e-8, 100.0, 200.0, 1.0e-3
    (tmp_path / "d.flp").write_text(SLAB)
    (tmp_path / "s.stk").write_text(STK.format(k1=k1, k2=k2, h=h, h1=h1, h2=h2))
    m = R.build_model(parse_stk(str(tmp_path / "s.stk")))
    sol = R.solve(m, r=1, nz=20, method="direct")
    q0 = P / (1000.0 * 1000.0)                      # W/um^2, uniform over the whole chip
    exact_mid = q0 / h + q0 * h2 / k2 + q0 * 3 * h1 / (8 * k1)
    got = sol.layer_field(m.tmap_layer("TOP")) - 300.0
    assert got.std() < 1e-9                         # laterally uniform
    assert got[0, 0] == pytest.approx(exact_mid, rel=2e-3)
    top = sol.T[0, 0, 0] - 300.0                    # top of the source layer: + q0 h1 / (2k) above the bottom
    exact_top = q0 / h + q0 * h2 / k2 + q0 * h1 / (2 * k1)
    assert top == pytest.approx(exact_top, rel=2e-3)
    assert sol.info["heat_out"] == pytest.approx(P, rel=1e-9)


def test_slab_top_sink_mirror(tmp_path):
    """Same slab, sink on the top face of the first element: the die's source layer is next to the sink."""
    k1, h, h1, P = 1.48e-4, 1.0e-8, 100.0, 1.0e-3
    stk = STK.format(k1=k1, k2=k1, h=h, h1=h1, h2=50.0).replace("bottom heat sink", "top heat sink")
    (tmp_path / "d.flp").write_text(SLAB)
    (tmp_path / "s.stk").write_text(stk)
    m = R.build_model(parse_stk(str(tmp_path / "s.stk")))
    assert m.top is not None and m.bottom is None
    sol = R.solve(m, r=1, nz=20, method="direct")
    q0 = P / 1e6
    # z = depth below the cooled top face; the flux toward the sink at depth z is the heat generated
    # above it, q0 z / H, so T(z) = Ta + q0/h + q0 z^2 / (2 k H). The layer below is adiabatic.
    z = h1 / 2
    exact_mid = q0 / h + q0 * z * z / (2 * k1 * h1)
    got = sol.layer_field(m.tmap_layer("TOP"))[0, 0] - 300.0
    assert got == pytest.approx(exact_mid, rel=2e-3)


# ------------------------------------------------------------------ refusals

def test_refuses_microchannel_and_nonuniform(tmp_path):
    base = open(os.path.join(os.path.dirname(__file__), "fixtures", "repro", "aligned", "s.stk")).read()
    (tmp_path / "die.flp").write_text(open(os.path.join(os.path.dirname(__file__), "fixtures", "repro",
                                                        "aligned", "die.flp")).read())
    (tmp_path / "fp.lyt").write_text(open(os.path.join(os.path.dirname(__file__), "fixtures", "repro",
                                                       "aligned", "fp.lyt")).read())
    (tmp_path / "nu.stk").write_text(base.replace("cell length 250.0 , width  250.0 ;",
                                                  "cell length 250.0 , width  250.0 ;\n   non-uniform true ;"))
    with pytest.raises(R.RefusedModel, match="non-uniform"):
        R.build_model(parse_stk(str(tmp_path / "nu.stk")))
    (tmp_path / "ch.stk").write_text(base.replace("   layer BOT  SUB ;", "   layer BOT  SUB ;\n   channel CH ;"))
    with pytest.raises(R.RefusedModel, match="microchannel"):
        R.build_model(parse_stk(str(tmp_path / "ch.stk")))
    (tmp_path / "tr.stk").write_text(base.replace("steady ;", "transient step 0.01, slot 0.1 ;"))
    with pytest.raises(R.RefusedModel, match="steady"):
        R.build_model(parse_stk(str(tmp_path / "tr.stk")))


# ------------------------------------------------------------------ against 3D-ICE's published numbers

def _model(repro, case):
    return R.build_model(parse_stk(os.path.join(repro, case, "s.stk")))


@pytest.mark.parametrize("case", ["aligned", "nolayout", "snapped"])
def test_emulation_reproduces_3dice(repro, case):
    """r=1, one cell per layer and 3D-ICE's end-layer vertical rule give 3D-ICE's own number to 0.005 K.

    This pins the whole bulk difference between 3D-ICE and the converged reference on aligned models
    to one documented rule, and validates the reference solver's assembly and boundary conditions.
    """
    m = _model(repro, case)
    sol = R.solve(m, r=1, nz=1, emulate_3dice_ends=True, method="direct")
    rise = sol.layer_field(m.tmap_layer("TOPD")).max() - 300.0
    assert rise == pytest.approx(ICE_RISE[case], abs=0.005)


def test_reference_ignores_die_edge_position(repro):
    """Exact area fractions: a die shifted by 0.42 cell must not change the peak (3D-ICE: +3.4 K)."""
    peaks = {}
    for case in ("aligned", "misaligned"):
        m = _model(repro, case)
        peaks[case] = R.solve(m, r=2).layer_field(m.tmap_layer("TOPD")).max() - 300.0
    assert abs(peaks["misaligned"] - peaks["aligned"]) < 0.05
    assert 10.9 < peaks["aligned"] < 11.2


# ------------------------------------------------------------------ the diff command without a solver

def _fake_ice(repro, case, outdir):
    """A 3D-ICE-like Tmap from the emulation (r=1, one cell per layer), written as iceforge run does."""
    m = _model(repro, case)
    sol = R.solve(m, r=1, nz=1, emulate_3dice_ends=True, method="direct")
    T = sol.layer_field(m.tmap_layer("TOPD"))
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "tmap.npz"), T=T)
    return T


def test_diff_agrees_with_emulated_aligned_run(repro, tmp_path, capsys):
    npz = str(tmp_path / "ice")
    _fake_ice(repro, "aligned", npz)
    code = main(["diff", os.path.join(repro, "aligned", "s.stk"), "--r", "2", "--ice-npz", npz, "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["verdict"] == "AGREE"
    lay = out["layers"][0]
    assert lay["n_explained_by_3dice_vertical_rule"] == lay["n_flagged_raw"]
    assert lay["ref_r_vs_2r_max_change"] < 0.05


def test_diff_flags_an_edge_hot_spot_as_die_edge(repro, tmp_path, capsys):
    npz = str(tmp_path / "ice")
    T = _fake_ice(repro, "misaligned", npz)     # exact-fraction emulation: no artefact of its own
    T = T.copy()
    # heated-insulator signature: hot cells in the column cut by the left die edge (x = 1394.7 um, cell 5)
    T[5, 10:20] += 3.0
    np.savez(os.path.join(npz, "tmap.npz"), T=T)
    code = main(["diff", os.path.join(repro, "misaligned", "s.stk"), "--r", "2", "--ice-npz", npz, "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["verdict"] == "DISAGREE (die-edge)"
    lay = out["layers"][0]
    assert lay["n_flagged"] == 10 and lay["n_flagged_die_edge"] == 10


def test_diff_flags_a_bulk_offset_as_diffuse(repro, tmp_path, capsys):
    npz = str(tmp_path / "ice")
    T = _fake_ice(repro, "aligned", npz)
    np.savez(os.path.join(npz, "tmap.npz"), T=T + 1.0)
    code = main(["diff", os.path.join(repro, "aligned", "s.stk"), "--r", "2", "--ice-npz", npz, "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["verdict"] == "DISAGREE (diffuse)"


def test_diff_refusal_exit_code(repro, tmp_path, capsys):
    p = os.path.join(repro, "aligned", "s.stk")
    s = open(p).read().replace("steady ;", "transient step 0.01, slot 0.1 ;")
    q = os.path.join(repro, "aligned", "tr.stk")
    open(q, "w").write(s)
    assert main(["diff", q, "--ice-npz", str(tmp_path)]) == 2
    assert "refused" in capsys.readouterr().err


# ------------------------------------------------------------------ against the real solver

@pytest.mark.solver
@pytest.mark.parametrize("case,verdict,code", [
    ("aligned", "AGREE", 0),
    ("nolayout", "AGREE", 0),
    ("snapped", "AGREE", 0),
    ("misaligned", "DISAGREE (die-edge)", 1),
])
def test_diff_repro_cases(repro, case, verdict, code, capsys):
    rc = main(["diff", os.path.join(repro, case, "s.stk"), "--r", "2", "--json",
               "-o", os.path.join(repro, case, "out")])
    out = json.loads(capsys.readouterr().out)
    lay = out["layers"][0]
    assert lay["max_rise_ice"] == pytest.approx(ICE_RISE[case], abs=0.005)
    assert out["verdict"] == verdict and rc == code
    assert lay["ref_r_vs_2r_max_change"] < 0.05           # reference is converged
    if case == "misaligned":
        assert lay["max_rise_ref"] == pytest.approx(11.07, abs=0.1)
        assert lay["peak_diff"] > 3.0
        assert lay["n_flagged_interior"] == 0
        assert lay["n_flagged_die_edge"] > 0
    else:
        assert lay["max_rise_ref"] == pytest.approx(ICE_RISE[case], abs=0.2)
