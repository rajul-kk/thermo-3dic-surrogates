"""Regenerate the README "Main findings" numbers from their result files.

    python scripts/headline_numbers.py [--results-root results] [--out results/headline_numbers.json]

Each entry names the README text it backs, the result files it reads and how the value is derived. Entries whose
result file has not been located carry ``source: null`` and a note; they are still pinned by the README tripwire in
``tests/test_headline_numbers.py`` but are not recomputed. ``results/`` is git-ignored, but every source file named
below is force-tracked, as is the JSON written here, so CI recomputes every resolved entry.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5_GEOMS = ("geometry1", "geometry2a", "geometry3", "geometry4", "geometry5", "geometry6")
SNAP_GEOMS = ("geometry4-snap", "geometry5-snap", "geometry6-snap")
ICB = "icb_thermono_fixed/icb_fixed_results/icb_thermono_level4_v2_seed{}_fixed.json"
LAYOUT_GEOMS = ("geometry4", "geometry5", "geometry6")
THERMONO_KAGGLE = "thermono_kaggle/summary.json"
PROJECTION = "artefact_projection/projection_summary.json"
ORIENTATION = "icb_orientation/baselines_{}.json"


def _load(root, rel):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def ridge_spatial_r2(root):
    vals = [_load(root, f"v5/baselines_{g}.json")["summary"]["ridge"]["spatial_r2"] for g in V5_GEOMS]
    return f"{min(vals):.2f}–{max(vals):.2f}", {"min": min(vals), "max": max(vals)}


def g6_ridge_median(root):
    v = _load(root, "v5/layout_cv_geometry6.json")["results"]["ridge"]["r2_median"]
    return ("negative" if v < 0 else "not negative"), {"r2_median": v}


def _snap(x):
    """Drop float representation noise (15.000000000000002 -> 15.0) before floor or ceil."""
    return round(x, 9)


def backbone_ms(root):
    be = _load(root, "backbone_eval.json")
    ms = [be[g]["backbone_sec_per_solve"] * 1000 for g in SNAP_GEOMS]
    lo, hi = 5 * math.floor(_snap(min(ms) / 5)), 5 * math.ceil(_snap(max(ms) / 5))
    return f"{lo}–{hi} ms", {"min_ms": min(ms), "max_ms": max(ms), "rounding": "outward to 5 ms"}


def backbone_peak_err(root):
    be = _load(root, "backbone_eval.json")
    v = [be[g]["models"]["backbone"]["abs_peak_err_K"] for g in SNAP_GEOMS]
    return f"{min(v):.2f}–{max(v):.2f} K", {"min_K": min(v), "max_K": max(v)}


def backbone_loc(root):
    be = _load(root, "backbone_eval.json")
    v = [be[g]["models"]["backbone"]["loc_err_um_median"] for g in SNAP_GEOMS]
    return f"{min(v):.0f}–{max(v):.0f} µm", {"min_um": min(v), "max_um": max(v)}


def _icb(root, level):
    vals = [_load(root, ICB.format(s))[level]["rmse"] for s in range(3)]
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f} K", {"seeds": vals}


def thermono_prefix_advantage(root):
    summary = _load(root, THERMONO_KAGGLE)
    ratios = {}
    for g in LAYOUT_GEOMS:
        backbone = statistics.mean(s["backbone"]["abs_peak_err_K"] for s in summary[g])
        thermono = statistics.mean(s["thermono"]["abs_peak_err_K"] for s in summary[g])
        ratios[g] = backbone / thermono
    return f"{min(ratios.values()):.1f}–{max(ratios.values()):.1f}×", {"ratio_by_geometry": ratios}


def _projection_beta(root, model, geoms):
    results = _load(root, PROJECTION)["results"]
    return {g: results[f"{model}|{g}"]["analysis"]["beta"]["mean"]["mean"] for g in geoms}


def label_swap_beta(root):
    betas = _projection_beta(root, "thermono", LAYOUT_GEOMS)
    return f"β {min(betas.values()):.2f}–{max(betas.values()):.2f}", {"beta_by_geometry": betas}


def fno_label_swap_beta(root):
    betas = _projection_beta(root, "fno", ("geometry4", "geometry6"))
    return f"β {betas['geometry4']:.2f} and {betas['geometry6']:.2f}", {"beta_by_geometry": betas}


def icb_ridge_orientation_change(root):
    old, fixed = _load(root, ORIENTATION.format("old")), _load(root, ORIENTATION.format("fixed"))
    pairs = [(level, model) for level in old.keys() & fixed.keys()
             for model in old[level].keys() & fixed[level].keys() if model.startswith("ridge")]
    if not pairs:
        raise ValueError("no ridge baseline appears in both orientation files")
    changes = {f"{level}/{model}": abs(fixed[level][model]["rmse"] - old[level][model]["rmse"]) / old[level][model]["rmse"]
               for level, model in sorted(pairs)}
    bound = math.ceil(_snap(100 * max(changes.values()) * 100)) / 100
    return f"≤ {bound:.2f}%", {"relative_rmse_change": changes, "rounding": "up to 0.01%, since the text states a bound"}


ENTRIES = [
    {"id": "ridge_spatial_r2_v5", "readme": "spatial R² {value}", "finding": 1,
     "source": [f"v5/baselines_{g}.json" for g in V5_GEOMS], "derive": ridge_spatial_r2,
     "how": "min and max of summary.ridge.spatial_r2 over the six v5 geometries"},
    {"id": "g6_ridge_median_r2", "readme": "on geometry6 its median R² is {value}", "finding": 2,
     "source": ["v5/layout_cv_geometry6.json"], "derive": g6_ridge_median,
     "how": "sign of results.ridge.r2_median (5-fold CV, n=45)"},
    {"id": "backbone_ms", "readme": "about {value}", "finding": 3,
     "source": ["backbone_eval.json"], "derive": backbone_ms,
     "how": "backbone_sec_per_solve on the grid-aligned chiplet geometries, rounded outward to 5 ms"},
    {"id": "backbone_peak_err", "readme": "hotspot to {value}", "finding": 3,
     "source": ["backbone_eval.json"], "derive": backbone_peak_err,
     "how": "models.backbone.abs_peak_err_K on the grid-aligned chiplet geometries"},
    {"id": "backbone_loc_err", "readme": "peak error and {value} location", "finding": 3,
     "source": ["backbone_eval.json"], "derive": backbone_loc,
     "how": "models.backbone.loc_err_um_median on the grid-aligned chiplet geometries"},
    {"id": "icb_s4_fixed", "readme": "S4 {value}", "finding": 5,
     "source": [ICB.format(s) for s in range(3)], "derive": lambda r: _icb(r, "level4"),
     "how": "mean ± sample sd of level4.rmse, v2 seeds 0-2, fixed orientation"},
    {"id": "icb_s5_fixed", "readme": "S5 zero-shot {value}", "finding": 5,
     "source": [ICB.format(s) for s in range(3)], "derive": lambda r: _icb(r, "level5"),
     "how": "mean ± sample sd of level5.rmse (S4-trained, zero-shot), v2 seeds 0-2, fixed orientation"},
    {"id": "thermono_prefix_advantage", "readme": "ThermoNO's apparent {value} hotspot advantage", "finding": 4,
     "source": [THERMONO_KAGGLE], "derive": thermono_prefix_advantage,
     "how": "per geometry, backbone abs_peak_err_K over the seed-mean ThermoNO abs_peak_err_K (seeds 0-2, pre-fix data); "
            "min and max over geometry4-6"},
    {"id": "label_swap_beta", "readme": "amplitude ({value}, all 3 seeds", "finding": 4,
     "source": [PROJECTION], "derive": label_swap_beta,
     "how": "results['thermono|<g>'].analysis.beta.mean.mean (3-seed mean over 5 folds), min and max over geometry4-6"},
    {"id": "fno_label_swap_beta", "readme": "geometry4/6 ({value}, CI above zero)", "finding": 4,
     "source": [PROJECTION], "derive": fno_label_swap_beta,
     "how": "results['fno|<g>'].analysis.beta.mean.mean for geometry4 and geometry6"},
    {"id": "icb_ridge_orientation_change", "readme": "Ridge-type baselines change by {value} when fixed", "finding": 5,
     "source": [ORIENTATION.format("old"), ORIENTATION.format("fixed")], "derive": icb_ridge_orientation_change,
     "how": "largest |fixed - old| / old test RMSE over the ridge-* baselines at every level present in both files "
            "(levels 2-4), rounded up"},
]

UNRESOLVED = [
    {"id": "field_vs_vector_r2", "readme": "R² 0.95 from the per-cell power field and −0.67 from the block vector",
     "finding": 2, "note": "report 9.15c (re-measured 2026-09-24) quotes field 0.936/0.951/0.954 at pca_k 8/16/32 and "
     "compact -0.67, printed by a CV script that writes no file. The nearest result file, results/v5/layout_cv_geometry4.json "
     "(v5 snapped data), gives ridge r2_mean -0.663, i.e. -0.66: the README value may predate the v5 snap"},
    {"id": "pcg_iterations", "readme": "20–50 iterations", "finding": 3,
     "note": "results/cost_accuracy.json holds PCG at 10/20/30/50 iterations. No single tolerance gives 20-50: matching the "
     "direct FV solve's |peak| and det.MAE to 1e-4 K gives 30-50 (g4 50, g5 30, g6 30); to 5e-4 or 1e-3 K gives 20-30"},
    {"id": "artefact_peak_inflation", "readme": "spurious peaks up to 14 K", "finding": 4,
     "note": "report 9.32 gives medians 0.5 / 2.1 / 4.1 K and a 14 K maximum (g4 / g5 / g6). results/artefact_learning/ "
     "reproduces the g5 and g6 medians (truth_old.excess_K: 2.10, 4.13; maxima 7.5 and 6.6 K) but has no geometry4 file, "
     "where the 14 K must come from"},
]


def build(results_root):
    root = Path(results_root)
    out = []
    for e in ENTRIES:
        value, detail = e["derive"](root)
        out.append({"id": e["id"], "finding": e["finding"], "value": value,
                    "readme": e["readme"].format(value=value), "source": [f"results/{s}" for s in e["source"]],
                    "how": e["how"], "detail": detail})
    for e in UNRESOLVED:
        out.append({"id": e["id"], "finding": e["finding"], "value": None, "readme": e["readme"],
                    "source": None, "note": e["note"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "headline_numbers.json"))
    args = ap.parse_args(argv)
    rows = build(args.results_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # values carry β, µ, ×, ±, which a cp1252 console cannot print
    for r in rows:
        print(f"{r['id']:32s} {r['value'] if r['value'] is not None else '(unresolved)'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
