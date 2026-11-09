"""Tier-C reference check for IC-ThermBench S2: re-run HotSpot on the paper's Table 8 stack and
compare with the released output.mat temperature fields.

This is a release-completeness / reference-ceiling check (see docs/report.md §9.35), not a fair
competitor: a generator-config reproduction matches its own labels by construction.

Inputs  : data/ic-thermbench/datasets/level2_steady/{input,output}.mat  (read-only)
HotSpot : WSL build (default 'wsl /home/rajul/HotSpot_slu/hotspot'), grid model, -detailed_3D on.
Output  : results/icb_hotspot_repro/<variant>.json (+ fields .npz)

Everything here is reconstructed from the paper (Table 8, Sec. 4.3) and the upstream ATPlace2.5D
public generator's file formats; the ICB authors' own config files are NOT in the Drive release.
Assumptions are listed in ASSUMPTIONS below and in the report.

Usage: python scripts/icb_hotspot_repro.py --variant at|t8 [--r-convec 0.01] [--ambient 298.15]
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "iceforge"))
from iceforge import hotspot as ihs  # noqa: E402

DATA = os.path.join(ROOT, "data", "ic-thermbench", "datasets", "level2_steady")
HOTSPOT = "wsl /home/rajul/HotSpot_slu/hotspot"
N = 64
# Recorded BEFORE any HotSpot run: 10 evenly spaced samples in the last 20 % (the benchmark's test
# block); 301 is coprime to 10 so the ten samples cover all ten cases (case = index % 10).
SAMPLES = [12000 + 301 * j for j in range(10)]

ASSUMPTIONS = [
    "Interposer extent W x H (mm) = max(grid_x) x max(grid_y) of the sample (integers, constant per case); "
    "the 64x64 HotSpot grid then has pitch W/64 (grid_x is linspace(0,W,64), so this is a guess).",
    "Chiplet footprints = connected components of equal non-zero chiplet_power cells, tiled exactly at "
    "cell level (adjacent chiplets with equal power would merge; 4-connectivity).",
    "Chiplet power (W) = the broadcast value (nominal chiplet power), applied as a total over the footprint.",
    "Package from the upstream ATPlace2.5D public config: spreader side = W+H (m), sink side = 2(W+H), "
    "r_convec = 0.01 K/W (the upstream script's rescale branch never fires), c_convec 140.4, spreader/sink "
    "k=400, interface 20 um k=4, secondary path off.  Ambient 298.15 K is NOT from the paper (S2 has no "
    "ambient channel); it is inferred from the lowest released temperatures (302.0 K).",
    "Filler (non-chiplet) cells of the chip and micro-bump layers = epoxy underfill (cap 2.32e6, res 0.625).",
    "Variant 't8': Table 8 numbers verbatim, uniform layers.  Variant 'at': composite C4 / TSV / micro-bump "
    "materials as in the upstream generator's .flp writer (Cu-in-underfill / Cu-in-Si area mixing).",
    "Upstream rim ('Edge_*') units of half-granularity width are omitted.",
]

# Table 8 (bottom to top): name, power, heat capacity, resistivity, thickness (m)
T8 = [
    ("Substrate", "N", 1.06e6, 3.33, 200e-6),
    ("C4Layer", "N", 2.32e6, 0.625, 70e-6),
    ("Interposer", "N", 1.75e6, 0.01, 110e-6),
    ("UbumpLayer", "N", 2.32e6, 0.625, 10e-6),
    ("ChipLayer", "Y", 1.75e6, 0.01, 150e-6),
    ("TIM", "N", 4.00e6, 0.25, 20e-6),
]
UF = (2.32e6, 0.625)
SI = (1.75e6, 0.01)


def composite():
    """Upstream generator's mixed materials (areas ratio of pitch^2/diameter^2 - 1)."""
    cu_c, cu_r = 3494400.0, 0.0025
    uf_c, uf_r = 2320000.0, 0.625
    si_c, si_r = 1750000.0, 0.01

    def mix(edge, dia, c_o, r_o):
        a = (edge / dia) ** 2 - 1
        return ((cu_c + a * c_o) / (1 + a), (1 + a) * cu_r * r_o / (r_o + a * cu_r))
    return {"C4": mix(0.0006, 0.00025, uf_c, uf_r),
            "TSV": mix(0.00005, 0.00001, si_c, si_r),
            "ubump": mix(0.000045, 0.000025, uf_c, uf_r)}


CONFIG = """\
-t_chip\t0.00015
-k_chip\t100.0
-p_chip\t1.75e6
-thermal_threshold\t354.95
-c_convec\t140.4
-r_convec\t{r_convec}
-s_sink\t{s_sink}
-t_sink\t0.0069
-k_sink\t400.0
-p_sink\t3.55e6
-s_spreader\t{s_spreader}
-t_spreader\t0.001
-k_spreader\t400.0
-p_spreader\t3.55e6
-t_interface\t2.0e-05
-k_interface\t4.0
-p_interface\t4.0e6
-model_secondary\t0
-ambient\t{ambient}
-init_temp\t{ambient}
-model_type\tgrid
-leakage_used\t0
-grid_rows\t64
-grid_cols\t64
-grid_map_mode\tcenter
-package_model_used\t0
"""


def load(i):
    with h5py.File(os.path.join(DATA, "input.mat"), "r") as f:
        x = f["data"][i, :, 0]
    with h5py.File(os.path.join(DATA, "output.mat"), "r") as f:
        t = f["data"][i, 0].astype(np.float64)
    return x.astype(np.float64), t


def d8(a, k):
    """8 dihedral transforms; returns (array, swaps_axes)."""
    ops = [(lambda z: z, False), (lambda z: z.T, True), (np.flipud, False), (np.fliplr, False),
           (lambda z: np.rot90(z, 1), True), (lambda z: np.rot90(z, 2), False),
           (lambda z: np.rot90(z, 3), True), (lambda z: np.rot90(z, 2).T, True)]
    f, sw = ops[k]
    return np.ascontiguousarray(f(a)), sw


D8_NAMES = ["identity", "transpose", "flip_rows", "flip_cols", "rot90", "rot180", "rot270", "antitranspose"]


def tile_mask(mask, r0=0):
    """Rectangles (r, c, h, w) in cells tiling the boolean mask: row runs merged vertically."""
    rects, open_ = [], {}
    for r in range(mask.shape[0] + 1):
        runs = {}
        if r < mask.shape[0]:
            row = np.concatenate([[0], mask[r].astype(int), [0]])
            d = np.diff(row)
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                runs[(s, e)] = True
        for key in list(open_):
            if key not in runs:
                r_start = open_.pop(key)
                rects.append((int(r_start), int(key[0]), int(r - r_start), int(key[1] - key[0])))
        for key in runs:
            open_.setdefault(key, r)
    return rects


def build_case(Q, Wc, Hr, variant, ambient, r_convec):
    """Q[row, col] chiplet power (rows = y up, cols = x); Wc, Hr = extent (mm) along cols / rows."""
    dx, dy = Wc * 1e-3 / N, Hr * 1e-3 / N
    Wm, Hm = Wc * 1e-3, Hr * 1e-3
    comps = []
    units = []  # (name, rect, power_W, material)
    l = 0
    for v in np.unique(Q[Q > 0]):          # components are labelled per power value (touching chiplets differ)
        lab, nl = ndimage.label(Q == v, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        for k in range(1, nl + 1):
            m = lab == k
            l += 1
            rects = tile_mask(m)
            comps.append({"cells": int(m.sum()), "power_W": float(v),
                          "rect": bool(m.sum() == (np.ptp(np.where(m)[0]) + 1) * (np.ptp(np.where(m)[1]) + 1))})
            for j, (r, c, h, w) in enumerate(rects):
                units.append((f"Chiplet_{l}_{j}", (r, c, h, w), float(v) * h * w / float(m.sum()), "chip"))
    filler = tile_mask(Q <= 0)
    for j, rect in enumerate(filler):
        units.append((f"WS_{j}", rect, 0.0, "fill"))

    def flp_line(name, rect, mat=None):
        r, c, h, w = rect
        s = f"{name}\t{w*dx!r}\t{h*dy!r}\t{c*dx!r}\t{r*dy!r}"
        if mat is not None:
            s += f"\t{mat[0]!r}\t{mat[1]!r}"
        return s + "\n"

    hdr = "# all dimensions are in meters\n"
    full = (0, 0, N, N)
    mats = composite() if variant == "at" else {}
    files = {}
    # single-unit layers
    for idx, (nm, pw, cap, res, th) in enumerate(T8):
        fname = f"L{idx}.flp"
        if nm in ("Substrate", "TIM"):
            txt = hdr + flp_line(nm, full)
        elif nm == "C4Layer":
            txt = hdr + flp_line(nm, full, mats["C4"] if variant == "at" else None)
        elif nm == "Interposer":
            txt = hdr + flp_line(nm, full, mats["TSV"] if variant == "at" else None)
        elif nm == "UbumpLayer":
            if variant == "at":
                txt = hdr + "".join(flp_line(u[0], u[1], mats["ubump"] if u[3] == "chip" else UF) for u in units)
            else:
                txt = hdr + "".join(flp_line(u[0], u[1], None if u[3] == "chip" else UF) for u in units)
            # t8 variant: micro-bump layer is uniform underfill (lcf values), one unit
            if variant == "t8":
                txt = hdr + flp_line(nm, full)
        else:  # chip layer
            txt = hdr + "".join(flp_line(u[0], u[1], SI if u[3] == "chip" else UF) for u in units)
        files[fname] = txt
    sim = "".join(flp_line(u[0], u[1]) for u in units)
    files["sim.flp"] = hdr + sim
    lcf = "# layers\n"
    for idx, (nm, pw, cap, res, th) in enumerate(T8):
        lcf += f"\n{idx}\nY\n{pw}\n{cap:.6E}\n{res}\n{th!r}\nL{idx}.flp\n"
    files["layers.lcf"] = lcf
    names = [u[0] for u in units]
    files["power.ptrace"] = "\t".join(names) + "\n" + "\t".join(repr(u[2]) for u in units) + "\n"
    sp = Wm + Hm
    files["hotspot.config"] = CONFIG.format(r_convec=r_convec, s_sink=2 * sp, s_spreader=sp, ambient=ambient)
    argv = ["-c", "hotspot.config", "-f", "L4.flp", "-p", "power.ptrace", "-steady_file", "out.steady",
            "-grid_steady_file", "out.grid.steady", "-detailed_3D", "on", "-grid_layer_file", "layers.lcf"]
    case = ihs.HotspotCase(files=files, layers=[], rows=N, cols=N, r=1, n_cols=N, n_rows=N, side_m=Wm,
                           ambient=ambient, sink_side="", argv=argv)
    return case, comps


def parse_layer(text, layer):
    vals, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Layer"):
            cur = int(s.split()[1].rstrip(":"))
            continue
        if cur == layer and s:
            p = s.split()
            if len(p) == 2:
                vals.append(float(p[1]))
    a = np.array(vals).reshape(N, N)   # [row from TOP, col]
    return a[::-1].copy()              # rows increasing in y


def metrics(p, t):
    e = p - t
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "max_abs": float(np.abs(e).max()),
            "peak_err": float(p.max() - t.max()), "bias": float(e.mean()),
            "r2": float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)),
            "corr": float(np.corrcoef(p.ravel(), t.ravel())[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["at", "t8"], default="at")
    ap.add_argument("--ambient", type=float, default=298.15)
    ap.add_argument("--r-convec", type=float, default=0.01)
    ap.add_argument("--orients", default="all", help="'all' or comma list of D8 indices")
    ap.add_argument("--samples", default="", help="comma list overriding SAMPLES")
    ap.add_argument("--tag", default="")
    ap.add_argument("--exe", default=HOTSPOT)
    a = ap.parse_args()
    samples = [int(s) for s in a.samples.split(",")] if a.samples else SAMPLES
    orients = list(range(8)) if a.orients == "all" else [int(s) for s in a.orients.split(",")]
    err, _ = ihs.preflight(a.exe)
    if err:
        sys.exit(err)
    out = {"variant": a.variant, "ambient": a.ambient, "r_convec": a.r_convec, "samples": samples,
           "assumptions": ASSUMPTIONS, "rows": []}
    fields = {}
    for i in samples:
        x, t = load(i)
        Lx, Ly = float(x[1].max()), float(x[2].max())
        Q0 = x[0].T            # raw stored input is transposed vs the output: Q0[row=y, col=x]
        per = []
        for k in orients:
            Q, sw = d8(Q0, k)
            Wc, Hr = (Ly, Lx) if sw else (Lx, Ly)
            case, comps = build_case(Q, Wc, Hr, a.variant, a.ambient, a.r_convec)
            t0 = time.time()
            txt, _ = ihs.run_hotspot(case, a.exe)
            p = parse_layer(txt, 4)
            m = metrics(p, t)
            m.update(orient=D8_NAMES[k], k=k, secs=round(time.time() - t0, 1))
            per.append(m)
            fields[f"s{i}_o{k}"] = p
        best = min(per, key=lambda d: d["rmse"])
        out["rows"].append({"index": i, "case": i % 10, "Lx": Lx, "Ly": Ly, "n_components": len(comps),
                            "all_rect": all(c["rect"] for c in comps), "total_power_W": float(sum(c["power_W"] for c in comps)),
                            "peak_true": float(t.max()), "min_true": float(t.min()),
                            "best": best, "all_orients": per})
        print(i, "case", i % 10, f"{Lx:.0f}x{Ly:.0f}", "chips", len(comps), "best", best["orient"],
              f"rmse {best['rmse']:.3f} max {best['max_abs']:.3f} peak {best['peak_err']:+.3f} r2 {best['r2']:.4f}",
              "| identity-orient rmse", f"{per[0]['rmse']:.2f}" if orients[0] == 0 else "", flush=True)
    od = os.path.join(ROOT, "results", "icb_hotspot_repro")
    os.makedirs(od, exist_ok=True)
    stem = f"{a.variant}{a.tag}"
    with open(os.path.join(od, stem + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(od, stem + "_fields.npz"), **fields)


if __name__ == "__main__":
    main()
