#!/usr/bin/env bash
# Re-run every downstream analysis on the v5 datasets (docs/report.md 9.23). Sequential, one log.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/v5
for g in geometry1 geometry2a geometry3 geometry4 geometry5 geometry6; do
  echo "=== 9.1a baselines $g"; python scripts/baselines.py --geometry $g --data data/3d-ice --output results/v5/baselines_$g.json 2>&1 | tail -12
done
for g in geometry1 geometry4 geometry5 geometry6; do
  echo "=== 9.15b layout_cv $g"; python scripts/layout_cv.py --geometry $g --data data/3d-ice-layout-$g --output results/v5/layout_cv_$g.json 2>&1 | tail -8
done
echo "=== 9.17 hotspot"; HOTSPOT_OUT=results/v5/hotspot_remeasured.json python -u scripts/hotspot_remeasure.py 2>&1 | tail -12
echo "=== 9.19 geometry conditioning"; python scripts/geometry_conditioning_test.py 2>&1 | tail -8
echo "=== 9.20 conformal"; python scripts/conformal_hotspot_test.py 2>&1 | tail -24
echo "=== 9.22 quadratic"; python scripts/quadratic_manifold_test.py 2>&1 | tail -20
echo "REMEASURE V5 DONE"
