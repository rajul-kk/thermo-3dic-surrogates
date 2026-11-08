"""NativeICESimulator against ICESimulator on scenarios that share one stack (same geometry, HTC and ambient).

usage: python scripts/check_ice_native.py [--geometry geometry1] [--htc 10000] [--t-amb 25]
          [--ice-executable "wsl ..."] [--batch-executable "wsl .../3D-ICE-Batch"]
Prints per-scenario wall times, factorisations, and the largest temperature difference (both write 3-decimal Tmaps).
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.geometry_builders import get_geometry_by_name  # noqa: E402
from src.simulators.ice_native import NativeICESimulator  # noqa: E402
from src.simulators.ice_simulator import DEFAULT_ICE_EXECUTABLE, ICESimulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="geometry1")
    ap.add_argument("--htc", type=float, default=10000.0)
    ap.add_argument("--t-amb", type=float, default=25.0)
    ap.add_argument("--ice-executable", default=DEFAULT_ICE_EXECUTABLE)
    ap.add_argument("--batch-executable", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = yaml.safe_load((ROOT / "configs" / "scenarios" / f"{a.geometry}_scenarios.yaml").read_text(encoding="utf-8"))
    scens = [s for s in spec["scenarios"] if s["htc"] == a.htc and s["t_ambient"] == a.t_amb]
    if len(scens) < 2:
        raise SystemExit(f"fewer than two scenarios share htc={a.htc}, t_ambient={a.t_amb}")
    geom = get_geometry_by_name(a.geometry)
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for name, cls in (("emulator", ICESimulator), ("native", NativeICESimulator)):
            cfg, out = Path(td) / name / "cfg", Path(td) / name / "out"
            cfg.mkdir(parents=True)
            out.mkdir()
            kw = {"batch_executable": a.batch_executable} if cls is NativeICESimulator else {}
            sim = cls(cfg, out, a.ice_executable, **kw)
            times, temps = [], []
            for s in scens:
                t = time.perf_counter()
                r = sim.simulate(geom, s, s["name"])
                times.append(time.perf_counter() - t)
                temps.append(np.asarray(r["temperature"] if "temperature" in r else r["temp"], dtype=np.float64))
            res[name] = {"seconds": times, "temps": temps, "stats": getattr(sim, "stats", None)}
            if hasattr(sim, "close"):
                sim.close()
            print(f"{name:<8} " + "  ".join(f"{t:6.1f} s" for t in times) + f"   total {sum(times):6.1f} s"
                  + (f"   {sim.stats}" if hasattr(sim, "stats") else ""), flush=True)
    diff = max(float(np.max(np.abs(x - y))) for x, y in zip(res["emulator"]["temps"], res["native"]["temps"]))
    print(f"{len(scens)} scenarios sharing one stack; max |native - emulator| = {diff:.4f} K (both print 3 decimals)")
    print(f"speed-up {sum(res['emulator']['seconds']) / sum(res['native']['seconds']):.1f}x over the group")
    if a.out:
        Path(a.out).write_text(json.dumps({"geometry": a.geometry, "htc": a.htc, "t_ambient": a.t_amb,
                                           "scenarios": [s["name"] for s in scens], "max_abs_diff_K": diff,
                                           "emulator_s": res["emulator"]["seconds"], "native_s": res["native"]["seconds"],
                                           "native_stats": res["native"]["stats"]}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
