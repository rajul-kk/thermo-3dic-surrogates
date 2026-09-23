# Regenerate the fixed-placement dataset (data/3d-ice) after the 2026-09-24 physics fixes
# (docs/report.md 9.23): power maps confined to blocks on their own layer, and active layers
# >100 um split into three z-nodes. Same scenario counts and seeds as regen_v4_final.ps1.
# Writes to data/3d-ice-v5; validate (scripts/validate_3dice.py) before swapping it in.
# Two passes per geometry: --extra-train forces skip_test in main.py.

$ErrorActionPreference = 'Continue'
$ICE = "wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator"
$OUT = "data/3d-ice-v5"
$plan = [ordered]@{ 'geometry1' = 25; 'geometry2a' = 10; 'geometry3' = 25; 'geometry4' = 25; 'geometry5' = 35; 'geometry6' = 35 }

foreach ($g in $plan.Keys) {
    python src/main.py --simulator 3d-ice --geometry $g --extra-train $plan[$g] --power-map mixed `
        --ice-executable $ICE --output $OUT 2>&1 | Select-Object -Last 2
    python src/main.py --simulator 3d-ice --geometry $g --skip-train --power-map mixed `
        --ice-executable $ICE --output $OUT 2>&1 | Select-Object -Last 2
    $n = (Get-ChildItem "$OUT/$g/*.npz" -ErrorAction SilentlyContinue).Count
    Write-Host "=== $g DONE : $n files ==="
}
Write-Host "ALL GEOMETRIES COMPLETE"
