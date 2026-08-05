# Targeted regeneration of geometry5/6 only, after fixing a bug where both
# builders never passed tsv_density= to Geometry(), so geometry.tsv_density
# silently stayed 0.0 and ScenarioGenerator.attach_tsv_maps's guard
# (`geometry.tsv_density <= 0`) no-opped for every scenario ever generated --
# TSV zone was always one uniform material, never a spatial field, and
# tsv_frac was a constant-zero PINN input / ridge feature for these two
# geometries. See tests/test_tsv_maps.py::test_every_tsv_geometry_declares_a_nonzero_scalar.

$ErrorActionPreference = 'Continue'
$ICE = "wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator"
$OUT = "data/3d-ice"

$plan = [ordered]@{
    'geometry5' = 35
    'geometry6' = 35
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
Write-Host "GEOMETRY5/6 TSV FIX REGENERATION COMPLETE"
