"""Regenerate the README "Main findings" numbers from their result files.

    python scripts/headline_numbers.py [--results-root results] [--out results/headline_numbers.json]

Each entry names the README text it backs, the result files it reads and how the value is derived. Entries whose
result file has not been located carry ``source: null`` and a note; they are still pinned by the README tripwire in
``tests/test_headline_numbers.py`` but are not recomputed. Most source files live under the git-ignored ``results/``
folder, so recomputation needs a local checkout with results; the JSON written here is tracked.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5_GEOMS = ("geometry1", "geometry2a", "geometry3", "geometry4", "geometry5", "geometry6")
SNAP_GEOMS = ("geometry4-snap", "geometry5-snap", "geometry6-snap")
ICB = "icb_thermono_fixed/icb_fixed_results/icb_thermono_level4_v2_seed{}_fixed.json"


def _load(root, rel):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def ridge_spatial_r2(root):
    vals = [_load(root, f"v5/baselines_{g}.json")["summary"]["ridge"]["spatial_r2"] for g in V5_GEOMS]
    return f"{min(vals):.2f}–{max(vals):.2f}", {"min": min(vals), "max": max(vals)}


def g6_ridge_median(root):
    v = _load(root, "v5/layout_cv_geometry6.json")["results"]["ridge"]["r2_median"]
    return ("negative" if v < 0 else "not negative"), {"r2_median": v}


def backbone_ms(root):
    be = _load(root, "backbone_eval.json")
    ms = [be[g]["backbone_sec_per_solve"] * 1000 for g in SNAP_GEOMS]
    lo, hi = 5 * math.floor(min(ms) / 5), 5 * math.ceil(max(ms) / 5)
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
]

UNRESOLVED = [
    {"id": "field_vs_vector_r2", "readme": "R² 0.95 from the per-cell power field and −0.67 from the block vector",
     "finding": 2, "note": "report 9.15c quotes 'about 0.95' over CV values 0.936/0.951/0.954; the -0.67 file was not located"},
    {"id": "pcg_iterations", "readme": "20–50 iterations", "finding": 3,
     "note": "scripts/backbone_pcg.py prints its table; no result file is written"},
    {"id": "artefact_peak_inflation", "readme": "spurious peaks up to 14 K", "finding": 4,
     "note": "report 9.32; result file not located"},
    {"id": "thermono_prefix_advantage", "readme": "2.6–5.4×", "finding": 4,
     "note": "report 9.26 and 9.32, pre-fix data; derived from results/thermono_kaggle/, rule not yet written down"},
    {"id": "label_swap_beta", "readme": "β 0.77–0.95", "finding": 4,
     "note": "report 9.32; results/artefact_learning/ holds geometry5 and geometry6 seed 0 only"},
    {"id": "fno_label_swap_beta", "readme": "β 0.35 and 0.26", "finding": 4,
     "note": "report 9.32; results/fno_artefact/ is not readable from this account"},
    {"id": "icb_ridge_orientation_change", "readme": "≤ 0.34%", "finding": 5,
     "note": "report 9.13 table, ridge-per-geom on S3; results/icb_orientation/baselines_{old,fixed}.json, rule not yet written down"},
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
    Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['id']:32s} {r['value'] if r['value'] is not None else '(unresolved)'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
