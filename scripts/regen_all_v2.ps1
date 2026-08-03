# Regenerate every geometry under the revised operating regime (2026-07-31) into data_v2.
#
# Two passes per geometry: --extra-train silently forces skip_test=True in main.py,
# so the test split needs its own --skip-train run afterwards.

$ErrorActionPreference = 'Continue'
$ICE = "wsl /home/rajul/3d-ice/bin/3D-ICE-Emulator"
$OUT = "data_v2"

# geometry -> extra training scenarios beyond the 15-scenario base pool
$plan = [ordered]@{
    'geometry2a' = 10
    'geometry2b' = 10
    'geometry2c' = 10
    'geometry3'  = 25
    'geometry4'  = 25
    'geometry1'  = 25
    'geometry5'  = 35
    'geometry6'  = 35
}

foreach ($g in $plan.Keys) {
    $extra = $plan[$g]
    Write-Host "=== $g : train (+$extra extra) ==="
    python src/main.py --simulator 3d-ice --geometry $g --extra-train $extra `
        --ice-executable $ICE --output $OUT 2>&1 | Select-Object -Last 2

    Write-Host "=== $g : test ==="
    python src/main.py --simulator 3d-ice --geometry $g --skip-train `
        --ice-executable $ICE --output $OUT 2>&1 | Select-Object -Last 2

    $n = (Get-ChildItem "$OUT/$g/*.npz" -ErrorAction SilentlyContinue).Count
    Write-Host "=== $g DONE : $n files ==="
}
Write-Host "ALL GEOMETRIES COMPLETE"
