"""Guard-off re-solve checks. MUST BE RUN BY THE USER BY HAND.

The ICEFORGE_GUARD=0 bypass reproduces how the unsnapped (moving-geometry4) and TSV-map
files were originally made. Agents are not permitted to set it, so this script is run by the
user, not by an agent. The variable is set only in the child-process environment.

Run from anywhere:  python scripts/guard_off_checks.py [--out DIR]
  F2  realdata diff re-run under pinned 3D-ICE, compared with results/iceforge_realdata
  C   data/3d-ice-moving-geometry4 (unsnapped): 3 files re-solved, compared with stored
  E   data/3d-ice/geometry{2a,5,6} (TSV maps): 2 files each re-solved, compared with stored
Output: prints a summary and writes guard_off_checks.log into the output directory
(default logs/guard_off_checks/). Roughly 1-2 h of CPU.

Context and the pending items this closes: results/3dice_version_check/summary.md
"""
import argparse, glob, os, shutil, subprocess, sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S = str(REPO / "logs" / "guard_off_checks")
LOG = os.path.join(S, "guard_off_checks.log")
ENV = dict(os.environ, ICEFORGE_GUARD="0")


def pick_first_last(data_dir, geometry):
    """First and last (sorted) npz under data/3d-ice/geometry{g}/**, excluding plots/."""
    pat = os.path.join(str(data_dir), "3d-ice", f"geometry{geometry}", "**", "*.npz")
    fs = sorted(f for f in glob.glob(pat, recursive=True)
                if "plots" not in Path(f).relative_to(data_dir).parts)
    return [fs[0], fs[-1]]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(args):
    log("$ " + " ".join(args))
    r = subprocess.run([sys.executable] + args, cwd=str(REPO), env=ENV, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines()[-15:]:
        log("  " + line)
    return r.returncode


def resolve_and_compare(tag, files):
    inp, out = os.path.join(S, tag, "in"), os.path.join(S, tag, "out")
    shutil.rmtree(os.path.join(S, tag), ignore_errors=True)
    os.makedirs(inp)
    for f in files:
        shutil.copy(f, inp)
    run(["scripts/resolve_from_metadata.py", inp, "--out", out])
    for f in files:
        name = os.path.basename(f)
        new = glob.glob(os.path.join(out, "**", name), recursive=True)
        if not new:
            log(f"{tag} {name}: NO OUTPUT"); continue
        a, b = np.load(f, allow_pickle=True), np.load(new[0], allow_pickle=True)
        ta, tb = np.asarray(a["temp"], float), np.asarray(b["temp"], float)
        if ta.shape != tb.shape:
            log(f"{tag} {name}: shape {ta.shape} vs {tb.shape}"); continue
        d = np.abs(ta - tb)
        commit = b["metadata"].item().get("ice_commit", "?") if "metadata" in b.files else "?"
        log(f"{tag} {name}: max|dT|={d.max():.3e} K mean|dT|={d.mean():.3e} K "
            f"max rel={d.max() / (ta.max() - ta.min()):.3e} ice_commit={commit}")


def main():
    global S, LOG
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=S, help="scratch/output dir and log location (default: logs/guard_off_checks/)")
    S = os.path.abspath(ap.parse_args().out)
    LOG = os.path.join(S, "guard_off_checks.log")
    os.makedirs(S, exist_ok=True)
    open(LOG, "w").close()
    D = REPO / "data"
    # C: moving-geometry4 unsnapped
    mg = sorted(glob.glob(os.path.join(D, "3d-ice-moving-geometry4", "geometry4", "*.npz")))
    resolve_and_compare("C_moving_g4", [mg[0], mg[len(mg) // 2], mg[-1]])
    # E: TSV-map fixed-placement geometries
    for g in ("2a", "5", "6"):
        resolve_and_compare(f"E_g{g}", pick_first_last(D, g))
    # F2: realdata diff re-run
    rd = os.path.join(S, "realdata_rerun")
    run(["scripts/iceforge_realdata_diff.py", "--out", rd])
    run(["scripts/compare_realdata_runs.py", "results/iceforge_realdata", rd])
    log("DONE")


if __name__ == "__main__":
    main()
