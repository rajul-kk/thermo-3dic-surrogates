#!/usr/bin/env bash
# Regenerate the pilot datasets on the v5 pipeline (docs/report.md 9.24). Sequential: parallel
# 3D-ICE jobs have exhausted WSL resources before. Outputs go to <dir>-v5 for validation first.
set -u
cd "$(dirname "$0")/.."
ICE="wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator"
echo "=== validate geometry7"; python -u scripts/validate_3dice.py --check B --geometries geometry7 2>&1 | grep -E "rise|Trace|Error"
for d in geometry7-pilot geometry7-material-sweep interface-multi interface-pilot interface-pilot-hi \
         interface-pilot-median moving-geometry1 moving-geometry4; do
  echo "=== resolve $d"; python -u scripts/resolve_from_metadata.py data/3d-ice-$d 2>&1 | tail -2
done
echo "=== leakage"; python scripts/gen_leakage_pilot.py --n 45 --extra-train 25 --output data/3d-ice-leakage-pilot-v5 2>&1 | tail -2
echo "=== throttle"; python src/main.py --simulator 3d-ice --geometry geometry1 --throttle --ice-executable "$ICE" \
    --output data/3d-ice-throttle-pilot-v5 2>&1 | tail -2
echo "=== microchannel"; python scripts/gen_microchannel_pilot.py --n 12 --output data/3d-ice-microchannel-pilot-v5 2>&1 | tail -2
echo "PILOTS V5 DONE"
