# Pre-registration: non-linear regime A, leakage-temperature feedback (geometry1)

**Status: written 2026-10-22 BEFORE any data is generated under this protocol.** Plan and reasoning: `notes/2026-10-22-nonlinear-plan.md`
(untracked). Regime B (per-block DVFS) is NOT covered here: its V-f table and trip temperatures are not yet sourced (gate G4).
Provenance tags: [LIT] literature, [OWN] our reasoning/assumption, [UNV] unverified.

## 1. Question
On the linear steady core, classical methods win (report 9.25, 9.30). Does that survive a legitimately non-linear, closed-loop
physics, and is there a regime where a learned surrogate lands on the cost-accuracy frontier? Thesis line: "do learned
surrogates beat the right classical baseline, and when?"

## 2. Physics (ground-truth model)
Per power block b of geometry1 (4 blocks): `P_b(T_b) = P_dyn,b + P_leak,b(Tref) * 2^((T_b - Tref)/k_d)`, `Tref = 85 C`.
`T_b` = mean temperature over the block footprint on the source layer (v5 export, orientation-fixed). Exponential form
[LIT: report ref 18, Chen et al. ICCAD 2006, not rechecked]; the 2^(dT/k_d) parametrisation is [OWN].
- Leakage share of nominal block power at Tref: drawn from [0.10, 0.35], per-block jitter +-20% [OWN assumption; patents and a
  course slide bracket whole-chip leakage at ~10% rising to ~50% for unnamed nodes, so this is context, not a source].
- `k_d` drawn from [10, 25] C [OWN assumption]. The Sky130 anchor (inv_1: 217x between 25 and 100 C, i.e. doubling per 9.7 C) is
  [UNV: LLM-summarised, units unconfirmed, recheck failed on 2026-10-22] and is NOT used to set this range. See gate G2.
- OOD law (second law): `P_leak ~ T^2 exp(-Ea/(k T))`, fitted to equal the exponential law's value and slope at 85 C [OWN].
Legacy variant: the global-peak law of `src/scenario/leakage.py` is kept only as a labelled comparison to report 9.10; no hypothesis
here uses it.

## 3. Ground-truth generation
- Simulator: 3D-ICE pinned build 4953952, iceforge guard ON, geometry1 (uniform layers, uniform grid, no non-uniform mode), 100 um cells.
- Outer Picard loop on block powers: `P^(n+1) = (1-w) P^n + w P(T(P^n))`, `w = 0.5`, then 1.0 once the estimated loop gain < 0.5.
  Stop when `max_b |P_b(T_b) - P_b used| / P_b < 1e-3` (not on peak change); cap 20 iterations.
- Runaway: any cell > 300 C, or the residual grows for 3 consecutive iterations. Runaway scenarios are kept and labelled, never dropped.
- Each file stores nominal power, parameters, delivered power, iteration count, final residual, estimated loop gain, runaway flag.
- Verification: (i) self-consistency: recompute `P(T_final)` from the exported field; must equal the delivered power to 1e-3.
  (ii) Independent fixed point: the same loop on the 9.23 finite-volume solver for a fixed 30-scenario subset; compare with 3D-ICE.
  (iii) `iceforge diff` (HotSpot third solver) on the final linear solve of 10 scenarios. (iv) Energy balance < 1e-4 on every converged file.
  Caveat stated in advance: the FV solver shares 3D-ICE's partial-cell conventions (report 9.32), so (ii) checks the loop, not the discretisation.

## 4. Scenarios and splits (sampling seed 20261201, fixed now)
- Pre-screen with the FV solver (cheap) so the estimated loop gain (fraction of critical) is spread over 0.1-0.8 for train/iid and
  0.7-0.9 for OOD. Nominal heat pattern, block powers, HTC, ambient, shares and k_d vary as in the existing generator ranges.
- Converged scenarios: 240 train / 30 iid test / 30 OOD-gain (gain > 0.7) / 30 OOD-law (T^2 exp(-Ea/kT) law).
  Runaway set: 30 runaway + 30 converged near-critical scenarios, held out, for classification (secondary result).
- Non-linearity measure per scenario: `100 * (peak rise at fixed point - peak rise at nominal power) / peak rise at nominal`.
  Target median 15-30%, range 5-60%; report the realised distribution. If the realised median is below 10%, the study is declared
  "weakly non-linear" and the A2 conclusions are not drawn from it.
- Exclusion rules: only the runaway and non-convergence rules above; every excluded scenario is counted and listed.

## 5. Methods compared (equal hardware, same inputs)
Classical: C0 linearised superposition (one Newton step, ~2 solves); C1 Picard with backbone-PCG inner solves; C2 Newton-Krylov
with the backbone as preconditioner (Jacobian = conduction matrix + diagonal `-dP/dT`), which also gives the stability test
(spectral radius of `G diag(dP/dT)` < 1). Error-vs-wall-time curves from truncating each at k iterations, CPU and GPU.
Learned: ThermoNO, plain FNO, LT-FNO, learned-conductance solver. Inputs: NOMINAL power map, k map, HTC, ambient and the leakage
parameters (share, k_d, Tref) as constant channels; never the delivered power. Control run per model with delivered power as input
(answer leaked) to show the inflation (as in the 9.8 bug). Seeds 0, 1, 2; training budgets copied from the 9.27/9.31 configs,
chosen once, and any run that hits a cap is reported as cap-limited.
Metrics: peak-temperature error (K), field RMSE and MAE (K), block-mean temperature error, wall time (batch size 1, CPU and GPU
reported separately). Mean +- SD over seeds; paired bootstrap CIs over scenarios (n=30 test sets, so CIs will be wide and are reported).

## 6. Hypotheses and decision rules (fixed before generation)
- **A1** (classical extends): C2 reproduces the 3D-ICE fixed point with peak error <= 0.3 K on >= 95% of converged iid + OOD scenarios
  at <= 1/5 of the 3D-ICE loop wall time. FAIL otherwise; then no "classical is solved for non-linear" claim is made.
- **A2** (learning beats the weak baseline only where it should): the best learned model's field MAE is below C0's on scenarios with
  deviation > 20% (paired bootstrap CI of the difference excludes 0), and is not claimed below 5%. Both bins are reported.
- **A3** (the main test): "learning pays" only if a learned model's peak error is below the classical error-vs-time frontier at the
  learned model's own latency, on iid AND on OOD-gain, with seed ranges not overlapping the frontier. Expected: FAIL.
- **A4** (misspecification): on OOD-law, a learned model's error exceeds 2x its iid error; a classical solver given the true law is
  unchanged; a classical solver given the wrong law errs by about the law mismatch. Reported as the trade-off; this is the only
  place a learning win is plausible (an unknown law, learned from data).
- **A5** (runaway): the spectral-radius test classifies converged/runaway at >= 98% on the held-out set; a learned classifier does
  not beat it.
Everything is reported whatever it shows. Thresholds are not changed after generation.

## 7. What each outcome allows the thesis to say
A1 pass + A3 fail: "even under legitimately non-linear closed-loop physics a Newton backbone-PCG matches 3D-ICE and no learned model
is on the cost-accuracy frontier." A3 pass (iid or OOD): "learning pays under feedback, here", stated with its exact latency and
regime. A4 as stated. A1 fail: the classical ladder does not extend; reported as a boundary.

## 8. Gates that must be closed (and recorded as dated amendments) before any data is generated
- **G1** Re-time 3D-ICE solves on a quiet machine (measured 2026-10-22: mean 60.5 s over 5 loaded solves; ~27-30 s steady per solve in a
  profile, of which 8.3 s was a one-time WSL build lookup) and fix N scenarios and the budget.
- **G2** Source or label the leakage parameters: recheck the Sky130 `sky130_fd_sc_hd` tt_025C/tt_100C cell leakage with units, or find
  a published leakage-vs-temperature curve. If no primary source is found, the `k_d` and share ranges stay [OWN] assumptions and
  every result is labelled parametric.
- **G3** Implementation tests pass before generation: the per-block loop reproduces a single linear solve when leakage share = 0; a
  one-block analytic fixed point; FV-vs-3D-ICE agreement (<0.5 K peak) on the final linear solve of 5 scenarios.
- **G4** (regime B only) Source a real V-f/OPP table and trip temperatures: not retrievable on 2026-10-22 (kernel.org anti-bot page, GitHub
  raw rate limit, blob 404, `gh` not installed).

## 9. Compute (estimate, to be refreshed at G1)
About 2100 3D-ICE solves at 30-40 s = ~18-24 CPU-h (~9-12 h wall on 2 jobs); training 4 models x 3 seeds + controls ~15 Kaggle GPU-h.

## 10. Stated weaknesses and prior work
- Synthetic leakage parameters and laws, one geometry, uniform grid only, claims limited to this family of laws.
- Prior work (extended search 2026-10-22, abstracts and one primary read): no paper found comparing a learned thermal surrogate against a
  Newton/Picard solver on leakage feedback. The nearest, arXiv 2603.29268, embeds a learned electrical model in an electro-thermal loop
  with an analytical thermal solver, on one 4x4 case, with no loop wall-time comparison. Not enough for a novelty claim beyond
  "we found no prior comparison".

## 11. Amendment 2026-10-24: gates G1-G3 run (before any data is generated)
- **G3 closed.** `src/scenario/leakage_perblock.py` + `tests/test_leakage_perblock.py` (6 tests pass: zero share = one linear solve; one-block analytic fixed point; multi-block root-find; damping does not move the fixed point; runaway flagged; Arrhenius law equals the exponential law's value and slope at Tref). FV-vs-3D-ICE on the final linear solve of 5 scenarios (`scripts/leakage_g3_check.py`, native geometry1, loop run on the FV solver): 3D-ICE peak minus FV peak = +0.054 to +0.070 K (limit 0.5 K), consistent with the known end-layer offset. Loop converged in 9-10 Picard iterations, self-consistency residual 5.7e-4 to 9.6e-4 (< 1e-3).
- **G1 closed.** Quiet-machine 3D-ICE solve (WSL, pinned build, includes launch and parse): 9.1, 13.0, 12.2, 12.6, 11.9 s (median 12.2 s). The 60 s of 2026-10-22 was contention. Budget: ~2100 solves x ~12 s = ~7 CPU-h (about 3.5 h wall on 2 jobs) if the loop runs on 3D-ICE directly.
- **Finding 1 (changes the plan): the FV solver is NOT a cheap pre-screen.** A full FV Picard loop took 258-368 s per scenario (~30 s per r=1 solve), slower than 3D-ICE. Section 4's "pre-screen with the FV solver (cheap)" is replaced by: pre-screen with the backbone-PCG solver of report 9.30 (0.5-3 s) or with the unit-response matrix (4 blocks -> 4 solves once per HTC/ambient), whichever is built first; the gain estimate needs only the unit-response matrix.
- **Finding 2 (changes the sampling): the scenarios were too cool to be non-linear in the intended direction.** With the native geometry1 stack and nominal powers of 2-5 W per block, block temperatures were 42-48 C, far below Tref = 85 C, so the leakage multiplier was < 1 and delivered power was 11-21% BELOW nominal (10.3-14.7 W vs 12.4-16.5 W). The loop gain was ~0.5 (damped ratio), convergent. A study defined as "heating raises power" needs scenarios whose block temperatures are near or above Tref. The sampling in section 4 therefore must also draw HTC and ambient (and total power) so that block temperatures span roughly 70-130 C, and the non-linearity measure (section 4) is reported with its sign. Until this is checked the "median 15-30%" target is not assumed achievable. These 5 scenarios are discarded as a check set and are not part of any dataset.
- **G2 (partial).** SkyWater PDK sky130_fd_sc_hd inv_1, cell_leakage_power: tt_025C_1v80 = 0.005326682, tt_100C_1v80 = 1.158223 (primary .lib.json files; units not stated in the JSON), ratio 217.4, doubling per 9.7 C. This is one 1.8 V inverter, not chip leakage. A patent (not a measurement) says 85 -> 10 C reduces subthreshold leakage 3-4x (doubling per ~37-47 C). The two differ by ~4x, so no citable chip-level k_d exists: `k_d` stays [OWN]; widen the drawn range to 10-45 C so both sources are inside it. All results are labelled parametric.
- **Still open before generation:** the sampling rule of Finding 2 (a dated amendment), the unit-response/PCG pre-screen of Finding 1, and the user's go-ahead for ~7 CPU-h of generation and ~15 Kaggle GPU-h of training. G4 (regime B) remains unsourced.

## 12. Amendment 2026-10-24 (written BEFORE the plan or any scenario is generated): sampling and ground-truth procedure
The user approved the CPU generation on 2026-10-24 after section 11. Changes to sections 3-4, all made before data exists:
- **Ground-truth procedure.** The plant is linear and 3D-ICE material properties are constant, so block-mean temperatures are `T_b = T_amb + sum_j R_bj d_j` exactly. `R` (4 x 4, K per W/cm^2) is measured with 3D-ICE (pinned build, pipeline `ICESimulator`) by one unit solve per block, for each of 5 discrete HTC values {2000, 3500, 6000, 10000, 20000} W/m^2K (so the HTC of a scenario is drawn from this set, not continuously; ambient is additive). A linearity check (a 2-block solve vs superposition) is recorded. The fixed point is computed on `R` by monotone iteration from the dynamic power to 1e-12 (the minimal fixed point), then ONE real 3D-ICE solve at the fixed-point power produces the exported v5 field. Check (i) of section 3 is applied to that exported field (block means recomputed from the file, `P(T_export)` vs delivered power, must be < 1e-3). On a validation subset (every 30th train/iid scenario, about 12) the real 3D-ICE Picard loop from nominal power is also run and compared with the fixed point; this gives the real loop iteration count and wall time used by A1. The per-scenario "3D-ICE loop wall time" for A1 elsewhere is `picard_iterations x median 3D-ICE solve time`, labelled estimated.
- **Picard cap.** The recorded Picard iteration count (to 1e-3, damping rule of section 3) uses a cap of 200, not 20: at loop gain up to 0.9 a 20-iteration cap cannot converge, which would mislabel genuinely convergent scenarios. Convergence of the ground truth itself is not capped (monotone iteration, 20000 steps).
- **Temperature range (Finding 2 of section 11).** Per scenario a target maximum block temperature at the fixed point is drawn U(70, 130) C and the mean power-density scale is found by bisection to hit it within 1 C; candidates with no fixed point, or with any block above 250 W/cm^2 nominal, are rejected. Ambient U(20, 60) C. Relative block pattern: 4 x Dirichlet(2,2,2,2). Leakage share per block U(0.10, 0.35) x U(0.8, 1.2); `k_d` U(10, 45) C [OWN; widened from 10-25 after G2]; Tref 85 C.
- **Loop-gain bins.** Gain = spectral radius of `diag(dP/dT) R` at the fixed point. Train and iid: gain in [0.1, 0.7], 80 train and 10 iid per bin [0.1,0.3), [0.3,0.5), [0.5,0.7] (so 240 / 30). OOD-gain: gain in (0.7, 0.9], 30. (The 0.7-0.8 overlap in section 4 is removed.) OOD-law: 30 scenarios in the train gain range solved under the second law. Near-critical converged: gain in (0.9, 0.97), 30. Runaway: 30 scenarios made by raising the power scale until no fixed point exists; these are not solved in 3D-ICE (no steady field exists) and only their parameters and the runaway label are stored in the plan.
- **Pre-screen** is the 4x4 `R` model (milliseconds), replacing the FV pre-screen (section 11, Finding 1). It is exact for this plant; the A1 comparison of "classical solver vs 3D-ICE loop" is therefore a comparison with the full-field solvers (C0-C2 of section 5), not with this 4x4 model, which is a privileged reference.
- **Not part of this CPU run:** the 30-scenario FV loop check (ii) is dropped (the FV loop costs ~5 min per scenario and G3 already compared the final solve); the `iceforge diff` HotSpot check (iii) on 10 final solves and the energy-balance check (iv) are run after generation. This is stated now so it is not read as an omission later.
- Realised non-linearity (signed, `100 (peak rise at fixed point - peak rise at nominal)/peak rise at nominal`) and the gain distribution are reported per split after planning; the "weakly non-linear" rule of section 4 stays.

## 13. Amendment 2026-10-24 (after a 56-candidate pre-screen on the measured response matrices, before the plan or any scenario was generated)
- The pre-screen with the section 12 sampling (target block temperature U(70,130) C) gave a signed non-linearity median of +7.3% (percentiles 0/10/25/75/90/100: -13.6/-4.8/-1.1/15.8/20.2/36.6%) and a loop-gain median 0.33 (0.04-0.96). The median is below the 10% "weakly non-linear" bar because scenarios under Tref = 85 C have delivered power below nominal. The target temperature range is therefore changed to **U(85, 135) C**, so that leakage raises power in (almost) all scenarios. This is chosen after seeing the pre-screen's non-linearity distribution, not before; the bar of section 4 is unchanged and the realised distribution of the final plan is reported whatever it is.
- The bisection fixed-point tolerance is relaxed to 1e-8 (max 3000 iterations, 32 steps) for speed; the stored fixed point is recomputed to 1e-12.
