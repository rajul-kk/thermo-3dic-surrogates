# Final regeneration under 3D-ICE 4.0 with every fidelity change applied:
#   - per-cell power maps (already default via --power-map mixed)
#   - spatially varying TSV density fields (now auto-wired in main.py)
#   - die-footprint Si/underfill layouts for geometry4/5/6 (auto, no flag)
#   - z-subdivision (MAX_SUBLAYER_UM, already in ice_simulator.py)
#
# Two passes per geometry: --extra-train silently forces skip_test=True in
# main.py, so the test split needs its own --skip-train run afterwards.

$ErrorActionPreference = 'Continue'
$ICE = "wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator"
$OUT = "data/3d-ice-v4"

# geometry -> extra training scenarios beyond the 15-scenario base pool
# (matches the counts already validated in data/3d-ice, for a like-for-like
# regeneration: 45/30/30/30/45/45/55/55 = 335 total scenarios)
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
    python src/main.py --simulator 3d-ice --geometry $g --extra-train $extra --power-map mixed `
        --ice-executable $ICE --output $OUT 2>&1 | Select-Object -Last 2

    Write-Host "=== $g : test ==="
    python src/main.py --simulator 3d-ice --geometry $g --skip-train --power-map mixed `
        --ice-executable $ICE --output $OUT 2>&1 | Select-Object -Last 2

    $n = (Get-ChildItem "$OUT/$g/*.npz" -ErrorAction SilentlyContinue).Count
    Write-Host "=== $g DONE : $n files ==="
}
Write-Host "ALL GEOMETRIES COMPLETE"
