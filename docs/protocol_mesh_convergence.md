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

## 7. Amendment M3 (2026-10-22, before any run): coarse-to-fine downscaling (block -> cell scale)
**Why.** Real chips span transistor -> cell -> block -> die -> package. We model die-to-package at 100-250 um. The block -> cell step (10-100 um) can
be bridged by a one-way "zoom": the local hotspot rise adds on top of the coarse field because silicon spreads heat laterally over about its
thickness (150 um) [OWN]. The cell -> transistor step is NOT simulated: no public sub-block power data exists, the wiring stack is anisotropic and
unmodelled, and below the phonon mean free path (hundreds of nm in silicon, [UNV] recalled, not sourced here) Fourier conduction, which 3D-ICE and
our solvers assume, under-predicts the local resistance. It stays a stated limitation with a literature-based note.
**Method.** Coarse field: reference solver at r=1 (100 um) on the full stack. Window of side w in {0.5, 1, 2, 4} mm centred on the M2 hotspot; local
fine model (cell <= 12.5 um) on the window domain with the full vertical stack, lateral boundary conditions interpolated from the coarse field
(Dirichlet T; a Neumann-flux variant is run as a labelled sensitivity), top and bottom conditions as in the global model. Truth: the domain-wide
fine solve from M1/M2. Metrics: peak-rise error and field RMSE inside the window against the truth; wall time of the zoom vs the full fine solve.
**Predictions [OWN].** P4: peak error < 2% when the window half-width >= 3 die thicknesses (450 um) with Dirichlet boundaries from the coarse field;
> 10% when the half-width < 1 die thickness. P5: Dirichlet and Neumann variants agree within 2% for windows >= 2 mm.
**Learned downscaler (M3b).** A small U-Net maps (coarse window field, coarse power map) to the fine window field, trained on synthetic hotspot
families (size, strength, position, number) and tested on held-out families. It "pays" only if it matches the local solve's error at <= 1/10 of its
latency (batch 1, equal hardware, 3 seeds). Expected: it does NOT beat the local solve on accuracy (a window solve takes seconds). The result is
reported whichever way it falls; the synthetic hotspot families are not real chip data and the claim is about the method, not about real chips.
**Gate G4.** The window solver reproduces the full fine solve to solver tolerance when the window covers the whole die (and to 1e-6 relative on the
peak), and the interpolated-boundary consistency check passes before any sweep.
## 8. Gate results (2026-10-22; gates G1-G3 run once, before any sweep; log `results/mesh_convergence/gates.log`, untracked)
Scenario: geometry1, uniform pattern, HTC 20000, ambient 25 C; peak about 310.05 K, i.e. a peak rise of about 11.9 K (single low-power scenario).
- **G1 (reference solver feasibility): PASS with a limit.** Peak (K) by lateral refinement r (cell = 100 um / r) and sub-layers per layer nz:
  r=1,nz=2: 310.0458 (5 s, 0.25 GB); r=2,nz=2: 310.0472 (20 s, 0.78 GB); r=2,nz=4: 310.0420 (33 s, 1.5 GB); r=4,nz=4: 310.0424
  (227 s, 7.7M unknowns, 5.4 GB). **r=8,nz=4 fails: MemoryError** at 30.7M unknowns (8.9 GB resident at failure on this 16 GB machine).
  So domain-wide refinement reaches 25 um at nz=4; 12.5 um needs nz=2 (15.4M unknowns, untested) or a local window.
- **G3 (reference vs 3D-ICE at native resolution): PASS.** `iceforge diff` at r=2, nz=2, tol 0.02: AGREE on all 10 Tmap layers (250 s, 2.8 GB).
- **G2 (3D-ICE at 50 um cells): PASS.** The 200 x 200 solve fits and completes: 85 s versus 12.9 s at native 100 um (6.6x for 4x the cells), peak
  310.0650 K vs 310.0640 K native (difference 0.001 K).
- **Early observation (not the M1 result, one scenario):** on this block-power map the lateral change in the reference peak between r=1 and r=4 is
  at most about 0.005 K (about 0.04% of the 11.9 K rise) and the nz=2 -> 4 change is about 0.005 K, so vertical discretisation is the larger of the two;
  3D-ICE and the reference differ by about 0.02 K at the peak (the known end-layer convention). Prediction P1 (within 1%) is therefore expected to hold
  trivially for block-aggregate power; M2/M3 are where discretisation can matter.
- **Timing correction.** A native 3D-ICE solve took 12.9 s on a quiet machine, against 27-60 s in earlier loaded runs; the regime-A compute estimate
  (docs/protocol_nonlinear_A.md section 9) should be refreshed at its gate G1 using ~13-15 s per native solve measured on a quiet machine, with the
  loaded 40-60 s as the pessimistic case.
**Consequence for M2 (amendment).** M2 asks for hotspots down to s = 10 um, which needs cells <= s/4 = 2.5 um. Domain-wide refinement stops at 12.5 um
(and only at nz=2), so M2 for s <= 50 um CANNOT use a domain-wide fine solve. It requires the local window solver of M3 (gate G4). The order is
therefore changed: build and validate the window solver (G4) against the domain-wide solves that do fit (r=4, nz=4, and r=8, nz=2 if it fits) first;
M2 for s >= 100 um uses domain-wide solves; M2 for s < 100 um uses the validated window solver, and its results are labelled as such.

## 9. Gate G4 result (2026-10-22): PASS
`iceforge.refsolve.solve_window` (local fine window, full vertical stack, Dirichlet or Neumann lateral data from a coarse r=1 solve; chip-edge sides
adiabatic) and `face_T_from_solution`; tests `tools/iceforge/tests/test_window.py` (7 pass; with `test_diff.py` 27 pass).
- A window covering the whole die reproduces the domain-wide r=2 solve to < 1e-6 K.
- With exact interface temperatures from the domain-wide solve, four interior/edge windows (including a 4 x 4-coarse-cell window) reproduce it inside
  the window to < 1e-6 K.
- Early observation, fixture only (iceforge `repro/aligned`, 250 um coarse cells, peak rise about 11 K, windows centred on the die; NOT the M3 result):
  max |dT| vs the domain-wide solve for half-widths 1000 / 2500 / 4000 um: Dirichlet 0.057 / 0.184 / 0.033 K, Neumann 0.019 / 0.044 / 0.017 K.
  The Dirichlet error is not monotone in window size here (the 2500 um boundary sits near a block edge, where the coarse field is least accurate),
  and Neumann was the more accurate on this fixture. M3 will report both and does not assume monotonicity; P4 and P5 stay as stated and are judged
  on the geometry1 runs.

**Link to regime A.** The leakage law uses each block's mean temperature; leakage is convex in temperature, so by Jensen's inequality sub-block hot
spots make the true block leakage larger than the block-mean law gives. The zoom model quantifies this bias; this is [OWN] reasoning, untested.

## 10. M3b detailed pre-registration (2026-10-24, written BEFORE any M3b sample is generated or any model trained)
Question: can a learned downscaler stand in for the classical zoom (the 1 mm window solve) on the block -> cell step?
**Setup.** geometry1, scenario 0, base power as in M2 (block1 centred at (2500, 2500) um, 3 mm square, uniform density `d_b`). Fine grid 12.5 um (r=8),
vertical sub-layers `nz_fine(12.5)`. Per sample: add N square hotspots of side s on the source layer as EXTRA density `(m-1) d_b` (total power is not
conserved here; additive hotspots), placed uniformly (continuous, not cell-aligned) so each lies inside the central 1 mm region [2000, 3000]^2.
- **Truth:** window [1500, 3500]^2 (2 mm), r=8, Dirichlet data from the coarse (r=1) solve of the same sample; the source-layer mid-plane temperature on
  the central 1 mm (80 x 80 fine cells). Truth gate **G5** (run first, 5 samples): the same field from a 3 mm window must agree with the 2 mm window within
  0.005 K (max |dT|); if not, the half-width is increased and G5 repeated before anything else.
- **Classical zoom (the thing to match):** window [2000, 3000]^2 (1 mm), r=8, Dirichlet data from the same coarse solve; same output field. Its error against
  the truth and its wall time (assembly + solve, coarse solve excluded for every method) are measured, not assumed.
- **No-downscaling baseline:** the coarse source-layer field bilinearly upsampled to the fine grid.
- **Learned downscaler:** fully convolutional residual CNN (3x3 convs, 32 channels, dilations 1, 2, 4, 2, 1, about 60k parameters). Input on a 96 x 96 fine grid covering
  [1900, 3100]^2: channel 1 the coarse source-layer temperature of the 12 x 12 coarse cells (bilinear, in K minus its mean), channel 2 the exact fine power-density map
  divided by `d_b`. Output: residual (fine T minus upsampled coarse T) on the central 80 x 80. Loss MSE; Adam lr 1e-3, batch 16, up to 200 epochs, early stopping
  patience 20 on validation; data augmentation by the 8 flips/rotations of the plane (approximate symmetry: the surroundings and the coarse boundary data are not exactly
  symmetric, stated); seeds 0, 1, 2. CPU, no GPU.
**Families (parameter ranges fixed now).** Train/val F0: N in {1, 2}, s log-uniform [25, 200] um, m log-uniform [3, 30]. Test families (40 samples each): T0 same as F0;
T1 size shift, N = 1, s in [250, 400]; T2 count shift, N in {4, 5, 6}, s in [25, 200]; T3 strength shift, N in {1, 2}, s in [25, 200], m in [30, 60]. Train 240, val 40.
**Metrics.** Per sample: peak error (K) and field RMSE (K) against the truth over the central 1 mm; median over samples per family and seed. Latency: classical window solve
vs learned forward pass, batch 1, one CPU thread, same machine.
**Decision rule (fixed now).** The learned downscaler "pays" iff, for ALL four test families and ALL three seeds: median |peak error| <= 2 x the classical zoom's median |peak error|
AND median field RMSE <= 2 x the classical zoom's, AND median latency <= 1/10 of the classical zoom's. Otherwise it does not pay; each condition is reported separately.
**Expectation [OWN].** Latency passes by orders of magnitude (the window solve takes seconds); accuracy fails, because the classical zoom is near the truth by construction
(M3: peak error 0.001-0.05 K). The learned model should still beat the no-downscaling baseline on the training family. Reported whichever way it falls.
**Scope and weaknesses.** Synthetic additive hotspots; one geometry; source-layer field only; 12.5 um resolution (not the transistor scale); the truth is itself a solver
(same physics model, no measured chip data); 240 training samples is small, so a failure may reflect data volume, not the method.
**Compute.** About 25 s per sample (coarse ~4 s, 2 mm truth ~15 s, 1 mm classical ~3 s, + overhead); 440 samples ~3 h on one core, ~1.5 h on two.
