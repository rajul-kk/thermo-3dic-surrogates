# Pre-registration: mesh-convergence and sub-cell hotspot study

**Status: written 2026-10-22 BEFORE any run.** Provenance tags: [LIT], [OWN], [UNV]. Predictions below are [OWN] guesses, stated so the
results can be compared with them honestly; they are not derived from data.

## 1. Question
Our geometries resolve heat at 100 um (geometry1, 2a) and 250 um (geometry3-6) cells with one 150 um silicon die layer. Real chips have power
structure down to the um scale (report scoping note, 2026-10-22). How much error does the coarse grid cause, and does it change a claim?
- **M1 (discretisation):** is the 3D-ICE solution at its native lateral resolution converged for our block-power maps?
- **M2 (sub-cell power):** how much does averaging power over a cell understate the local peak, as a function of hotspot size and strength?

## 2. Method
Reference: `iceforge diff` reference solver (independent discretisation: lateral cell / r, vertical sub-layers, exact power overlap; pyamg-CG;
README "Reference solver"). 3D-ICE: pinned build 4953952, iceforge guard ON, uniform grid, no non-uniform mode.
Vertical discretisation is held fixed for the lateral sweep, with the heat sink and spreader sub-layers capped via `--nz` to keep memory below
~12 GB (the earlier r=4 out-of-memory came from 25 um sub-layers in the 5 mm heat sink). A separate vertical check doubles nz once.
- **M1.** geometry1 and geometry5 (3D-on-2.5D; 250 um cells), the standard 4 power blocks, 3 nominal scenarios each (low, mid, high power).
  Reference at r in {1, 2, 4, 8} (geometry1: 100, 50, 25, 12.5 um). 3D-ICE at its native cell and, if one solve fits in memory (gate G2), at half
  the native cell. Metrics: peak temperature, hotspot location, field RMSE against the finest reference; observed order of convergence and a
  Richardson-extrapolated peak; 3D-ICE native error against the extrapolated value.
- **M2.** geometry1 with the block-1 3 mm power split into a hot rectangle of side s in {10, 25, 50, 100, 200, 400} um (cell-aligned, centred in
  the block) at power density multiplier m in {3, 10, 30} of the block mean, block total power unchanged (the rest of the block lowered). The
  coarse-grid value is 3D-ICE at 100 um (power averaged over the cell by overlap area) and the reference at r fine enough that the cell is <= s/4.
  Metric: relative peak-rise underestimation `(rise_fine - rise_coarse) / rise_fine` as a function of s/(die thickness 150 um) and m. A misaligned
  placement (offset 0.5 cell) is run once per (s, m) as a labelled check of power smearing (iceforge rule S002). No `.lyt` layout is used, so the
  die-edge artefact (S001) cannot interfere.

## 3. Predictions (fixed now, [OWN])
- **P1:** 3D-ICE at native resolution is within 1% of the extrapolated converged peak rise on block-power maps (consistent with prior iceforge
  diff AGREE on geometry1/5); hotspot location within one cell.
- **P2:** underestimation is monotone in 1/s; below 5% for s >= 150 um (the die thickness, because silicon spreads heat laterally over about its
  thickness); above 20% for s = 10-25 um at m >= 10; the misaligned placement changes the coarse peak by less than the aligned-vs-fine gap.
- **P3 (decision):** if P1 holds, "the resolution is adequate for block-level claims" is stated; if P2 holds, peak temperatures at 100-250 um
  are reported as lower bounds with the measured relation as the limitation. If P1 fails, the discretisation error is added to the error budget
  of every classical-vs-learned comparison and the largest-error geometry is reported.
Everything is reported whatever it shows; predictions are not edited after the runs.

## 4. Gates (record as dated amendments before the sweep)
- **G1** Confirm `iceforge diff` can be driven programmatically for r, nz and custom `.flp` files, and that the reference solver fits in memory for
  geometry1 at r=8 with the capped nz (run r=2 first).
- **G2** One 3D-ICE solve of geometry1 at 50 um cells (200 x 200) on a quiet machine: record time and memory. If it fails, 3D-ICE is compared at
  native resolution only and the convergence evidence comes from the reference solver alone (stated as a weaker result).
- **G3** The reference reproduces 3D-ICE at native resolution on the standard geometry1 scenario within the existing iceforge tolerance before any
  sweep is interpreted.

## 5. Cost
Reference solves: minutes each (about 7 s per 360k unknowns, 25 s per 1.4M); roughly 60-80 reference runs. 3D-ICE: ~30-40 s per native solve,
~10-30 min per 50 um solve if it fits. About 3-6 CPU-hours in total, no GPU.

## 6. Stated weaknesses
Block-aggregate power inputs only; steady state; the reference solver shares the same physics model and boundary conditions as 3D-ICE, so this tests
discretisation, not model fidelity (no measured chip temperatures exist to validate against). Real BEOL and TSV detail stays homogenised.
