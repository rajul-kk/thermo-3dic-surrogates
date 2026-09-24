#!/usr/bin/env bash
# Re-run the pilot-dataset analyses (docs/report.md 9.8-9.11, 9.15) on the v5 pilots. One log.
set -u
cd "$(dirname "$0")/.."
echo "=== 9.8 throttle baselines"; python scripts/baselines.py --geometry geometry1 --data data/3d-ice-throttle-pilot 2>&1 | tail -9
echo "=== 9.8 microchannel baselines"; python scripts/baselines.py --geometry geometry6_microchannel --data data/3d-ice-microchannel-pilot 2>&1 | tail -9
echo "=== 9.9 interface linearity"; python scripts/analyze_interface_linearity.py 2>&1 | tail -30
echo "=== 9.9 interface uncertainty"; python scripts/analyze_interface_uncertainty.py 2>&1 | tail -20
echo "=== 9.9 ridge tim-k feature"; python scripts/test_ridge_tim_k_feature.py 2>&1 | tail -15
echo "=== 9.10 leakage convergence"; python scripts/analyze_leakage_convergence.py 2>&1 | tail -20
echo "=== 9.10 leakage classification"; python scripts/classify_leakage_convergence.py 2>&1 | tail -20
echo "=== 9.10 leakage baselines"; python scripts/baselines_leakage.py --data data/3d-ice-leakage-pilot --geometry geometry1 2>&1 | tail -15
echo "=== 9.11 geometry7 baselines"; python scripts/baselines.py --geometry geometry7 --data data/3d-ice-geometry7-pilot 2>&1 | tail -9
echo "=== 9.11 geometry7 hotspot"; python scripts/hotspot_eval.py --geometry geometry7 --data data/3d-ice-geometry7-pilot 2>&1 | tail -10
for g in geometry1 geometry4; do
  echo "=== 9.15 rigid translation $g"; python scripts/layout_cv.py --geometry $g --data data/3d-ice-moving-$g 2>&1 | tail -8
done
echo "REMEASURE PILOTS DONE"
