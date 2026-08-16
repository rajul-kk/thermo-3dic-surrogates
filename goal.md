# Goal

## What this repo is

An open benchmark for neural thermal surrogates in 3D-IC packaging: **275 real 3D-ICE
simulations** across **6 package geometries** (single die → CoWoS + 6×HBM), **5 surrogate
model families** (PINN, FNO/WHNO/CNO-FNO, PI-DeepONet, ARO, Therm-FM), a shared XAI
toolkit, and a FastAPI app. Ground truth is real finite-element 3D-ICE output, not
synthetic data.

**The actual contribution is a negative result, not the models.** Closed-form ridge
regression — no training, no GPU — already solves this benchmark's temperature field at
spatial R² 0.92–0.99. That means block-level thermal simulation of fixed-k packages is
close to linear in power/HTC/ambient, and no published architecture on this class of
benchmark has been shown to beat that baseline. The paper (`docs/report.md`) is about
*why* the field's usual evaluation protocol hides this, and what would make a benchmark
in this domain actually require a learned operator.

## What changed this session

1. **Upgraded 3D-ICE 3.0.0 → 4.0** — the only version that supports per-element
   materials, anisotropic k, and spatial layouts. Everything below depends on it.
2. **Spatially varying TSV density** (`src/scenario/tsv_maps.py`) replacing one scalar
   per TSV layer, with anisotropic lateral/vertical effective k (a TSV is a copper
   cylinder — conducts along its axis, resists across phases).
3. **Closed the geometry4/5/6 underfill inconsistency** (`assumptions.md` §6.1): 3D-ICE's
   ground truth now emits a real Si/underfill layout instead of uniform silicon, matching
   what the PINN's PDE loss always assumed. This was a genuine train/target mismatch, not
   just a simplification.
4. **Found and fixed three bugs during audit**, all silent (no error, wrong output):
   - `attach_tsv_maps` was built and tested but never called by the CLI orchestrator —
     every prior run used the old scalar regardless.
   - 3D-ICE 4.0 heap-corrupts ("corrupted size vs. prev_size") when a footprint layout and
     a full-mesh per-cell power floorplan land on the same die layer. Fixed by capping
     power-map resolution to 64 for footprint-bearing geometries.
   - `build_geometry5`/`build_geometry6` built their TSV material at 3% but never passed
     `tsv_density=` to the `Geometry` constructor, so it silently stayed at the dataclass
     default of `0.0`. Every guard that reads `geometry.tsv_density` (the field mechanism,
     the PINN's `tsv_frac` input, the ridge baseline's TSV feature) had been a no-op for
     these two geometries their **entire history**, not just this session.
5. **Regenerated the full dataset** under 4.0 with all of the above active (335/335
   scenarios, zero failures), then regenerated geometry5/6 a second time once the
   `tsv_density` bug was caught.
6. **Removed geometry2b/2c** (2026-08-06): near-duplicates of geometry2a whose
   ridge-regression baselines were bit-identical to geometry2a's (spatial R²=0.991,
   MAE=2.207 K, all three to 3 decimals), with TSV density not exposed as a model input
   anywhere. Benchmark is now 6 geometries / 275 scenarios. Every reference to
   geometry2b/2c across code, tests, docs, configs and the Kaggle notebook was removed
   or updated in the same pass; `docs/report.md` §1–4/§10–11 were also rewritten to
   drop the retracted "first PINN" framing and re-measured against the current data.

## Net effect on the benchmark

- **Ridge still wins.** Spatial R² 0.918–0.991 across all 6 remaining geometries post-fix. Expected
  and correct — none of the above were meant to break linearity; they were data-fidelity
  fixes to the training set, not new physics.
- **Per-cell power remains the one change that mattered.** It doesn't move field R² much,
  but it collapses hotspot-localisation error from single digits to thousands of microns
  of true spatial ambiguity — the one place an operator can currently earn its cost over
  ridge.
- **TSV field and underfill layout are correctness fixes, not novelty.** Both close
  train/target gaps; neither is a defensible research contribution on its own (lateral
  heterogeneity in 3D-IC thermal sims is settled prior art — see `docs/references.md`).

## TSV field exposure — closed 2026-08-06, partially

The TSV spatial field affected the 3D-ICE ground truth but no model received it as an
input. Fixed for the layer that matters most: `src/core/mesh.py:generate_tsv_field`
samples the per-scenario map onto every point, `NPZExporter` now saves it as `tsv_frac`
in every `.npz`, and `src/fno/model.py` (`_as_field`/`_as_scalar`) consumes it as a real
per-cell channel in FNO3d/CondFNO3d/CNOFNOHybrid — the model can now condition on *where*
TSV density is high, not just how much there is on average. FourierPINN, DeepONet and ARO
data loaders were upgraded to at least use the field's true per-scenario mean (previously
a geometry-constant scalar that didn't vary within a geometry at all) instead of the field
itself — full per-point conditioning for those three point-based architectures would need
their model signatures changed to accept a per-point tensor instead of a per-scenario
scalar (touches ~20 call sites across trainer/evaluate/explain/sampling/physics for PINN
alone) and was judged too large a surgery to do untested in the same pass. Covered by
`tests/test_tsv_field_export.py`.

**Data note:** the mechanism is validated end-to-end (export round-trip, FNO forward/
backward pass with a real field), but `data/3d-ice` was last regenerated before this
change, so existing `.npz` files fall back to an all-zero `tsv_frac` array (correct,
non-breaking, but not yet carrying real signal). One more regeneration pass populates it.

## Microchannel liquid cooling — mechanism built and validated 2026-08-06

`Geometry.coolant_layer_name` + `ICESimulator` cooling_mode='microchannel_2rm' replaces
the idealised `bottom heat sink` BC with a real 3D-ICE 4.0 `microchannel 2rm` coolant
layer (grammar confirmed against 3D-ICE 4.0's own `test/mc2rm/steady/*.stk` examples).
`_plan_sublayers` forces the coolant layer to a single stack element (a channel doesn't
sub-divide); `_generate_stack_file` emits the `microchannel 2rm :` block, a `channel`
stack entry in place of the layer, and skips Tmap output for it (see below). Fails loudly
if `cooling_mode='microchannel_2rm'` is requested on a geometry with no
`coolant_layer_name` set, rather than silently building an ill-posed problem with no
heat-rejection boundary. Covered by `tests/test_microchannel.py`.

**Validated against the real 3D-ICE 4.0 binary** on a geometry6-based test config (a thin
copper base plate added below the channel — 3D-ICE rejects a channel as the bottom-most
stack element, and real cold plates have a base plate anyway): stable and physically
plausible at 29–455 W (geometry6's core-fraction TDP ceiling), peak 167.4 °C, well under
Si's 1414 °C melting point. This directly resolves the standing caution about whether
3D-ICE's channel model stays numerically stable at accelerator-class power.

**Honest caveat:** the illustrative coolant parameters used in that validation (channel
200 µm high, coolant HTC top=25000/bottom=5000 W/m²K) ran *hotter* than an air-cooled
comparison at the same power (167.4 °C vs. 151.5 °C, HTC=50000). This proves the
simulation mechanism is correct, not that those parameters make a good cooler — real
microchannel cold-plate design (channel/wall dimensions, flow rate, HTC split) is a
separate tuning exercise, not attempted here.

**Correction (2026-08-09): the "coords/export gap" described here previously does not
actually exist — verified empirically, not just re-reasoned.** `process_scenario`'s real
(non-mock) path never uses `generate_coords_and_indices`'s output; it sources `coords`
directly from `ICESimulator.parse_results()`, which only includes `output_inst_N.txt`
files for stack elements that were actually given a `Tmap(...)` request in the `.stk`
file — and `_generate_stack_file` already skips that request for the coolant element.
Ran a full `process_scenario` → `NPZExporter` round trip on a microchannel-cooled
geometry6 variant: succeeded cleanly, 103,488 consistently-shaped points, layer index 1
(the coolant-replaced `heat_sink`) correctly absent from `layer`/`coords`/`temp`, all 11
other layers present. **What's actually missing is CLI wiring, not a pipeline bug**: none
of the 6 registered geometries have `coolant_layer_name` set, so there's no way to select
a microchannel-cooled run through `main.py` yet without constructing the geometry
ad hoc the way the validation script did. Scope: a small dedicated generation script
(same pattern as `scripts/regen_v4_final.ps1`) building a microchannel-enabled geometry
variant and looping `process_scenario` over a scenario sweep, then re-run the ridge
baseline — if ridge still solves it, that's a stronger result than anything currently in
the paper; if it doesn't, that's the benchmark's reason to exist.

## Thermal throttling (DVFS) — implemented and validated 2026-08-06

`ScenarioGenerator.attach_throttling` + `src/scenario/throttling.py:apply_throttling`
iteratively solve-and-derate: run 3D-ICE, check peak T against a threshold (default
95 °C), reduce power proportionally to the overshoot if exceeded, re-solve, repeat until
convergence or a power floor is hit. Wired into `main.py`'s `process_scenario` — when a
scenario has `throttle_enabled=True`, `scenario_params['power_blocks']` is mutated to the
FINAL derated values, so npz export, statistics and plots all see the power that was
actually delivered, not the nominal request. Throttle state (`throttle_triggered`,
`throttle_derate_factor`, `throttle_iterations`) is exported as npz metadata. Covered by
`tests/test_throttling.py` (stub-simulator unit tests: graduated derate, pinned-at-floor,
max-iteration cap, convergence math).

**Validated against the real 3D-ICE 4.0 binary** on geometry3 (server CPU — Turbo
Boost/thermal throttling is exactly this chip's real behavior) at two overshoot levels:
a moderate case converged to derate=0.724 in 2 solves (peak 81.06 °C, under the 90 °C
threshold), and an aggressive case correctly pinned at the 0.3 power floor in 2 solves
(peak 68.54 °C — under-throttled relative to the 90 °C target, an expected consequence of
an aggressive default gain rather than a bug). **Both converged in 2 solves, not the
3–5 estimated before implementing** — steady-state conduction being linear in power means
one proportional correction from the first solve's overshoot lands very close to the
target, so the real cost is closer to **2× a normal solve**, not 3–5×.

**Why this one is different from every other change so far:** power becomes a function of
the temperature field being solved for — a genuine closed feedback loop. Per-cell power,
TSV fields, underfill layouts, and even microchannel cooling at a fixed flow rate all
still map a scenario-fixed source to a temperature field; this doesn't.

**Not yet done:** not wired into the shipped dataset (no throttled scenarios in
`data/3d-ice` yet — `attach_throttling` exists but nothing calls it from `main.py`'s CLI
path the way `attach_tsv_maps` is auto-wired), and the model architectures don't yet know
to expect a *derated* power field distinct from the nominal request (they just see
whatever `power`/`temp` pair ends up in the npz, which is fine for training but means
there's no way to ask a model "what would this scenario have looked like unthrottled").

## Other recommended next steps

**1. Wire microchannel and throttling into the shipped dataset.** Both mechanisms are
built and validated against the real binary; neither has produced a single scenario in
`data/3d-ice` yet. Microchannel's pipeline already works (see correction above) and just
needs a generation script; throttling just needs `main.py` to call `attach_throttling` the way it already
auto-calls `attach_tsv_maps`, plus a decision on which geometries get it (geometry1 and
geometry3 are the physically-motivated choices — see the geometry-by-geometry discussion
in this conversation).

**2. Do not add more geometry-fidelity fixes for their own sake.** Three bugs found
2026-08-05/06 were all in the "make the existing linear regime slightly more correct"
category — worth doing because they were silently wrong, not because they were expected
to change the paper's conclusion, and they didn't. Microchannels and throttling are
qualitatively different (both break the linearity assumption itself); further
TSV/underfill/thickness-style refinements are not, and shouldn't be prioritized over
finishing the two mechanisms that already exist.

**3. Full per-point TSV conditioning for PINN/DeepONet/ARO** (see above) — the field
mechanism now exists and FNO consumes it; extending point-based architectures to the same
depth is bounded, understood work, just deferred for scope/risk reasons this session.

**4. HBM/chiplet-count sweep on geometry6** (2/4/6-HBM variants) — cheap for FNO (shared
epoch loop, ~20% more wall time for ~20% more scenarios), notably more expensive for PINN
(a full new per-geometry training run per variant, ~37-90 min each, no shared
amortization — see the training-cost discussion in this conversation). Do this as an
FNO-first exercise if pursued.

## Next phase: testing, audit, development

Three different kinds of work remain, and they don't compete for the same time — pick
based on what's actually blocking progress, not necessarily in this order.

### Testing

1. **Wire microchannel + throttling into `main.py`'s CLI path** and generate a *pilot*
   batch (10-15 scenarios each, not a full 50+ regeneration) for each — enough to prove
   the mechanisms produce real, exportable, trainable `.npz` files end to end, not just
   pass the unit tests that mock/stub around the real pipeline.
2. **Re-run `scripts/baselines.py` on the pilot batches.** This is the actual test that
   matters for the paper: does ridge still solve microchannel-cooled or throttled
   scenarios? If yes, that's stronger evidence for the negative result than anything
   currently in the paper. If no, that's the benchmark's reason to exist — either result
   is a real finding, not a blocker.
3. **Smoke-test one real FNO training run** (few epochs, geometry1 or the pilot batch) on
   the new data to confirm tensor shapes and the training loop work against real (not
   synthetic) throttled/microchannel data before committing to a full run.

### Audit

1. **Apply this session's own lesson to the new mechanisms.** Three real silent bugs were
   found this session by auditing regenerated data, not by reading code. Once pilot
   batches exist: check for NaN/Inf, verify `throttle_derate_factor` in the metadata
   actually matches the ratio of exported `power` to the nominal (pre-throttle) request,
   and verify microchannel scenarios' temperature fields are self-consistent with the
   coolant inlet temperature used.
2. **Confirm DeepONet/ARO data loaders don't choke on the new scenario fields.**
   `throttle_enabled`/`coolant_layer_name` were added without updating those two loaders —
   they should be harmless (loaders read specific named keys, not the whole dict), but
   this hasn't been verified against a real throttled/microchannel `.npz` file yet, only
   reasoned about.
3. **Full top-to-bottom re-read of `docs/report.md`** for self-consistency. It's had many
   targeted edits this session (geometry counts, dataset numbers, two new mechanisms); a
   single continuous read-through would catch cross-section contradictions a sequence of
   local edits can miss, the way the geometry2b/2c pass did for the old draft.
4. **Flag the Kou→Zhou citation as blocked on human access**, not something further
   automated work can resolve — IEEE Xplore blocks the fetch this session needed to
   confirm the specific dimensional figures, not just the citation's existence.

### Development

1. Write a dedicated microchannel-scenario generation script (the pipeline itself is
   already correct — verified 2026-08-09, see correction above).
2. Wire `attach_throttling` into `main.py`'s CLI the way `attach_tsv_maps` is auto-wired —
   as an opt-in flag rather than on-by-default, since it multiplies solve cost per
   scenario and shouldn't silently slow down every future regeneration.
3. Full per-point TSV conditioning for PINN (§ above) — lower risk to attempt now that the
   field-export mechanism itself is proven and tested against real FNO forward/backward
   passes.
4. **Train an actual model.** After all of this session's infrastructure work, `checkpoints/`
   is still empty and `docs/report.md` §9.7 still reads "*None. No checkpoint has been
   trained.*" Once the data pipeline is stable (it now is, for the 6-geometry/275-scenario
   baseline), running the first real FNO training pass on geometry1 — the cheapest,
   simplest case — is arguably the single most overdue item in this project, independent
   of whichever of the above gets prioritized first.

## Plan: nonlinearity and generalization readiness (2026-08-09)

Neither "useful for nonlinearity" nor "useful for generalization" is currently a
defensible claim — not because the architectures are wrong, but because the data and the
conditioning mechanism needed to test each claim don't exist yet. Two independent tracks,
plus the standing "train something" prerequisite both depend on.

**Prerequisite, blocks nothing else and should happen first regardless of track:**
Train baseline FNO on geometry1 with the *existing* 275-scenario dataset (§ above, item 4).
This validates the training loop, loss functions, and checkpoint/eval plumbing on the
simple case before either track below adds more complexity on top of an unproven
pipeline. Cheap: ~6.7 min on a single T4 per the earlier cost estimate.

### Track A — Nonlinearity (throttling / microchannel) — A1/A2 done 2026-08-09

**A1. Done.** Microchannel's pipeline was already correct (§ correction above); wrote
`scripts/gen_microchannel_pilot.py`. Throttling wired into `main.py` as `--throttle`
(off by default).

**A2. Done — first real measurement, and it moved.** Generated a 12-scenario microchannel
pilot (geometry6, flow rate swept 80–280 mL/min) and a 20-scenario throttled pilot
(geometry1, 6/20 scenarios actually triggered, derate factors 0.34–0.72).

- **Microchannel:** ridge spatial R² = **0.906** — the weakest of any geometry measured
  this whole session (every other geometry scored ≥0.918). More strikingly, **ridge's
  hotspot localisation (780 µm) is worse than plain nearest-neighbour (616 µm) and kNN
  (595 µm)** — the first time in this project ridge has lost to a dumber baseline on any
  metric. Small pilot (8 train/4 test); treat as a signal to follow up, not a settled
  result.
- **Throttling — found and fixed a real methodology bug in the process.** First measurement
  gave ridge spatial R² = 0.989 on throttled data, *higher* than the same geometry
  without throttling (0.970) — suspicious, and correctly so: the exported
  `block_power_*` metadata holds the **delivered** (already-derated) power, so ridge was
  being handed the closed loop's resolved answer as an input feature, not asked to
  represent the loop at all. Fixed by exporting `nominal_block_power_*` (the pre-throttle
  request) and making `scripts/baselines.py::collect_block_keys` prefer it whenever
  `throttle_enabled` is set (fails loudly if that field is missing rather than silently
  falling back). Corrected result: **spatial R² = 0.890, det.MAE = 0.831 K** — real
  degradation from 0.970, and knn (R²=0.891) now edges out ridge on det.MAE too. Still
  not a rout, but the first clean evidence that a genuinely closed-loop nonlinearity
  measurably erodes ridge's advantage. This finding — and the bug that nearly hid it — is
  worth the fix on its own.

**A3. Partial — one baseline FNO run on the throttled pilot, not the full comparison
matrix.** `checkpoints/fno/geometry1_throttled` (same `--cpu-fast` config as the
prerequisite run). The full CondFNO/CNO-FNO+attention comparison this item originally
scoped was not completed in this pass — time-boxed given everything else in this
session; a fair architecture comparison needs GPU-scale training anyway (§ below), not
another CPU smoke test. Remains open.

**Apples-to-apples fix, done 2026-08-09.** The A3 FNO run above was comparing an easier
problem than ridge's: `NPZExporter` only ever wrote the *delivered* (post-throttle)
per-cell power, so FNO's `Q_norm` saw the closed loop's resolved answer while ridge was
correctly fed the pre-throttle `nominal_block_power_*` request. Fixed by exporting a new
`power_nominal` per-cell field (recomputed from `throttle_nominal_power_blocks` when
throttling fired; identical to `power` otherwise, so a no-op for the rest of the dataset)
and making `FNODataset` build `Q_norm` from it. Regenerated the throttled geometry1 pilot
(fresh 3D-ICE draw, 6/20 triggered, derate 0.48–0.84) and retrained `geometry1_throttled`
on the corrected data with the same plain-FNO/`--cpu-fast` config as the original run.
Result: FNO now trails ridge by a wide margin on every metric (det.MAE 1.795 K vs.
0.707 K, spatial R² 0.551 vs. 0.919, hotspot loc. err 5139 µm vs. 0 µm) — consistent with
the paper's central finding, now without the earlier caveat that the comparison wasn't
like-for-like. Still a capacity/epoch-limited CPU run, so this isn't evidence a
properly-resourced operator couldn't do better, only that this one doesn't. Covered by
`tests/test_nominal_power_export.py` (export round-trip, delivered-vs-nominal
divergence when throttled, `FNODataset` consumption). See `docs/report.md` §9.7/§9.8.

**A3 GPU notebook prepared, not yet run — 2026-08-11.**
`notebooks/kaggle_a3_throttled_arch_comparison.ipynb` trains all four architectures
(`fno`, `cond-fno`, `cno-fno`, `cno-fno`+attention/SAU-FNO) on the throttled geometry1
pilot at GPU scale (32 channels, 4 blocks, 400 epochs — vs. the CPU smoke test's 16
channels/100 epochs), evaluated with the same detrended metrics ridge/kNN were scored on
so the comparison is direct. Requires a Kaggle GPU session and a separate dataset upload
(`data/3d-ice-throttle-pilot/geometry1/`, distinct from the standard `3dice-thermal-data`
dataset other notebooks use) — see `notebooks/KAGGLE_SETUP.md`. No local GPU is available
in this environment, so this is prepared for the user to run, not executed here.
Result pending; update this section and `docs/report.md` §9.7/§9.8 once run.

**A4. Decision gate, unchanged:** only pursue a genuine nonlinear-interference
architecture extension if a *properly resourced* A3 (GPU, full capacity, multiple
architectures) shows the existing variants clearly failing. The CPU pilot runs in this
session are too capacity- and epoch-limited to license that conclusion either way.

### Track B — Cross-geometry generalization

**B1. Rescope the claim now — cheap, do immediately.** The current mechanism
(`target_grid` trilinear resampling + a single `geom_extent_norm` scalar) is what current
literature would call brute-force grid alignment, not geometry-aware encoding (contrast
with GINO/PI-GANO-style SDF or graph-based geometry conditioning, which report <3% error
on genuinely unseen shapes). Update `README.md`/`docs/report.md` to state plainly that
this repo's cross-geometry mechanism supports *interpolation among the 6 trained
geometries*, not zero-shot transfer to an unseen shape — a documentation fix, not an
engineering one, and it removes an overclaim risk immediately regardless of which other
track gets prioritized.

**B2. Built and validated 2026-08-09 — result is genuinely mixed, not a clear win.**
Implemented `generate_distance_to_power_block_field` (`src/core/mesh.py`): per-cell
distance to the nearest power block, normalised by die diagonal, purely geometric (no
simulation data needed, unlike the TSV field — computed once per geometry, same for
every scenario). Wired opt-in through `FNODataset(geometries=...)` →
`FNO3d(use_geometry_field=True)` (a 6th input channel) → `FNOTrainer`, all off by default
so nothing already trained this session is affected. `scripts/experiment_geometry_aware.py`
runs the validation gate this item specified: train on 5 geometries (`--common-grid`
resampled to 32×32×8), test zero-shot on the 6th (geometry4 held out), compare with vs.
without the field.

**Result (single run, seed 42, 60 epochs, CPU, small model — 198k params):**

| metric | baseline | +geometry field |
|---|---|---|
| MAE (K) | 3.777 | **3.096** |
| det.MAE (K) | 1.241 | **1.144** |
| spatial R² | **0.326** | 0.258 |
| hotspot loc. err (normalised units — mislabeled µm in the script's own output, fix before reusing) | 0.510 | **0.475** |

Three of four metrics improve modestly with the geometry-aware field; spatial R² gets
*worse*. The geometry-field run's validation curve was also visibly less stable across
training (val_MAE oscillated 5.53→4.16→4.34→5.23→5.13 K vs. the baseline's smoother
4.43→4.25→4.27→4.22→4.46 K) — plausibly more capacity to overfit with only 205 training
scenarios and 60 epochs. **This is a promising first signal, not a validated
result** — one run, one held-out geometry, one seed. Before drawing any conclusion:
repeat with multiple seeds, hold out a different geometry (geometry4 is a reasonable but
arbitrary choice), and fix the coordinate-unit bug in the experiment script's own
hotspot-distance calculation (it uses a normalised unit cube, not real µm, despite the
field name — harmless for the baseline-vs-field comparison since both runs share it, but
mislabeled if reused for anything reported in real units).

**B2 multi-seed/multi-holdout follow-up, done 2026-08-10 — the single-run signal does
not generalize.** Extended `scripts/experiment_geometry_aware.py` with `--seeds` and
`--holdouts` to repeat the whole experiment (3 seeds × 2 holdouts × 2 conditions = 12
FNO training runs, same 60-epoch/198K-param config as the single run above).

| holdout | metric | baseline (mean±std, n=3) | geom-field (mean±std, n=3) | winner |
|---|---|---|---|---|
| geometry4 | MAE (K) | 3.597±0.423 | 3.572±0.454 | field (barely) |
| geometry4 | det.MAE (K) | 1.258±0.093 | 1.161±0.096 | field |
| geometry4 | spatial R² | 0.299±0.159 | 0.397±0.139 | field |
| geometry4 | hotspot loc. err (µm)* | 5991±1377 | 4871±220 | field |
| geometry1 | MAE (K) | 9.389±0.499 | 8.965±1.236 | field |
| geometry1 | det.MAE (K) | 1.812±0.026 | 1.786±0.049 | field |
| geometry1 | spatial R² | 0.464±0.018 | 0.378±0.081 | **base** |
| geometry1 | hotspot loc. err (µm)* | 4875±692 | 5856±866 | **base** |

*(hotspot distances are in the experiment script's normalised-cube units mislabeled
"_um", see the caveat above — not fixed, since only the baseline-vs-field comparison
within a run matters here, not the absolute value.)*

**Verdict: mixed, and not in the way that's easy to spin.** The geometry-aware field
helps on geometry4 (all 4 metrics, including spatial R² and hotspot error — the two
metrics that matter most for this paper's framing) but *hurts* on geometry1 on those same
two metrics, while still helping MAE/det.MAE there. Per-seed values show the direction
isn't even stable within a holdout: geometry4/seed42 has the field *hurting* spatial R²
(0.326→0.258, exactly the original single-seed B2 result above), while seed43 and seed44
both show large gains (0.479→0.586, 0.092→0.346) — the original B2 run happened to land
on the one seed out of three where the field looked worst on that metric. **Conclusion:**
the geometry-aware distance-to-power-block field is not a validated generalization
mechanism at this scale (205 training scenarios, 60 epochs, 198K params, one grid
resolution) — its effect is geometry-dependent and noisy enough that a single run in
either direction would have been misleading. This is a real result, not a null one: it
means claims of the form "geometry-aware conditioning helps zero-shot generalization"
need either a properly-resourced run (more scenarios, more epochs, more holdouts) or a
different conditioning mechanism (SDF/graph-based, per B1's literature contrast) before
they're defensible on this benchmark — not further single-run experiments on this
architecture.

### Suggested sequencing — status 2026-08-09

1. ✅ Baseline FNO trained on geometry1.
2. ✅ B1 (rescoped the generalization claim in README/report.md).
3. ✅ A1 → A2 (throttling CLI + microchannel pipeline verified/scripted; pilot batches
   generated and measured — see Track A above).
4. ◐ A3 partial — one baseline-FNO run each on throttled/un-throttled data, apples-to-apples
   as of 2026-08-09; the full CondFNO/CNO-FNO+attention GPU comparison has a notebook
   prepared (2026-08-11) but not yet run — see A3 above.
5. ✅ B2 done, including the multi-seed/multi-holdout follow-up (2026-08-10) — mechanism
   built and validated end-to-end; the single-run signal (3/4 metrics improved) did NOT
   hold up across seeds/holdouts. Settled result: **not validated as a generalization
   mechanism at this scale**, geometry-dependent and seed-noisy rather than a clean win
   or loss. See Track B above for the full breakdown.

**What's still open, in priority order:** (a) ✅ done 2026-08-10 — multi-seed/multi-holdout
repeats of B2, result: not validated, see Track B, (b) ✅ done 2026-08-09 — exported
`power_nominal` so FNO's throttled comparison is apples-to-apples with ridge's (Track A3
above), (c) ◐ notebook prepared 2026-08-11, not yet run — the full A3 GPU architecture
comparison (`notebooks/kaggle_a3_throttled_arch_comparison.ipynb`); needs a Kaggle GPU
session this environment doesn't have, execution is on the user.

## Documentation audit and novelty action plan (2026-08-16)

Two parallel passes: (1) a top-to-bottom fact-check of every doc against real code/data —
this repo had never had one across ALL docs simultaneously, only targeted fixes per
session; (2) a fresh literature scoop-check plus a search for the field's own stated
pressing open problems, to ground novelty claims in current (2026) sources rather than
this project's own accumulated assumptions.

### Audit findings and fixes

**Real errors found and fixed, all in `docs/geometry_reference.md` and
`docs/references.md` — none in `docs/report.md`'s own body text.** The drift was
concentrated in the two reference docs, which get touched less often than the paper:

1. `docs/geometry_reference.md`'s Overview table used a `layers × nx × ny` formula that
   doesn't match 3D-ICE's actual adaptive z-grid (verified 8/8 point-count claims by
   loading real files in `data/3d-ice/*/` directly — e.g. geometry6 was listed as
   470,400 pts/file, a raw mesh-cell product; the real npz files contain 141,120). Fixed
   the Overview table and five detail-section restatements of the same wrong formula.
2. The same file's "Dataset Statistics" table still said 335 files / 8 geometries — the
   pre-2026-08-06 count, contradicting its own "Dataset Summary" table 60 lines above it
   (which correctly said 275/6). Fixed, and pruned leftover geometry2b/2c rows from the
   TSV-properties and Material Library tables.
3. `docs/references.md` §4 and §6's HTC/ambient-temperature comparison tables described
   the *original* scenario-generator design (HTC 500–200,000, ambient 25–85°C) as if
   current. Verified the live dataset's actual range directly from all 275 files' metadata
   (HTC 2,000–50,000 W/m²·K, ambient 25–45°C — matches `docs/report.md`'s own numbers)
   and added dated correction notes rather than deleting the historical comparison, since
   it still serves its original purpose (checking initial design choices against
   literature).
4. `docs/compute.md`'s GPU cost estimates key off the *declared* mesh resolution, not the
   real per-file grid FNO actually trains on (confirmed from training logs: FNO uses the
   real npz shape by default, e.g. geometry1 trains on grid (100,100,10) despite the
   geometry declaring (100,100,40)) — flagged as a likely-conservative-overestimate
   discrepancy rather than re-deriving every cost number, since these are estimates for
   future runs, not settled results.
5. **Closed a real reproducibility gap**: `docs/report.md` §9.7/§9.8's throttled-FNO
   numbers (det.MAE 1.795 K, spatial R² 0.551, hotspot 5139 µm) were real — computed
   in-session — but had never been saved as a re-runnable artifact, unlike the ridge
   numbers next to them. Added `scripts/eval_fno_throttled.py`; re-ran it, got the
   identical numbers (confirms they weren't wrong, just unarchived), output now at
   `results/fno_throttled_eval.json`.

**Known, not fully closed**: most §9.1/§9.3/§9.4/§9.5 result tables in `docs/report.md`
still rest on session-only computation with no saved metric JSON in `results/` to
re-verify against (only the train/test split membership is saved, not the computed
MAE/R²/hotspot numbers). Two spot-checked tables (`results/baselines_geometry1.json`,
`results/baselines_geometry6.json`) matched the paper exactly to 3 decimals, which is
reassuring but not a substitute for closing the gap properly — **next action**: write a
single `scripts/regenerate_all_results.py` that re-runs `scripts/baselines.py` across
every geometry/OOD-split/power-map condition the paper cites and saves full metric JSONs,
not just split membership. Not done this pass — scoped as the next concrete task.

### Novelty / scoop check (fresh 2026 literature pass)

**No scoop risk found** on any of this project's three most novelty-load-bearing claims:
(a) a linear/ridge baseline beating or matching a neural PDE surrogate specifically in
chip/package thermal simulation — nothing found makes this claim, only adjacent generic
PDE-surrogate benchmarking (e.g. "Operator Boosting Produces Pareto-Efficient PDE
Surrogates," arXiv:2606.17460) that doesn't compare against a closed-form baseline
either; (b) post-hoc FNO/WHNO interpretability via reading trained spectral weights
directly (no forward pass) — closest adjacent work ("Neural Interpretable PDEs,"
arXiv:2505.23106) achieves interpretability via architecture design instead, a different
mechanism; (c) testing WHNO's claimed Gibbs-ringing advantage over FNO on a new domain —
the origin paper (arXiv:2511.07347) validates only on synthetic Darcy flow/Burgers/heat
conduction, not real 3D-IC data.

**Strongest concrete evidence found for the paper's central argument**: "Self-Attention
to Operator Learning-based 3D-IC Thermal Simulation" (SAU-FNO, arXiv:2510.15968, Oct
2025) — already cited in this repo as close prior art — reports 842× speedup and >50%
MSE reduction **without stating a linear/ridge baseline comparison**. That's the
evaluation gap this paper's central finding diagnoses, demonstrated in a specific,
very-recent, directly-on-domain paper, not just asserted as a general pattern. **Action**:
cite this explicitly by name in `docs/report.md` §10's "Implications for the field"
paragraph — currently that paragraph makes the general claim without a concrete recent
example; SAU-FNO is one.

### Pressing open problems in the field (arXiv:2604.03290, Barua/Udoy/Aziz 2026 — now
fully confirmed, see `docs/references.md` §7)

Four of the six problems this survey identifies are ones this repo's *existing, already-
completed* work substantively responds to — not yet explicitly connected in the paper's
own positioning:

1. **Robustness under distribution shift, need for explainability** (§VIII-E: "no longer
   raw prediction speed, but robustness under shift... more uncertainty aware, more
   explainable") — directly answered by Track B's leave-one-geometry-out result and the
   MC-Dropout miscalibration finding (§10). **Action**: cite this survey's framing when
   introducing those results, rather than presenting them as self-motivated.
2. **No standardized, uncertainty-aware TBR (thermal boundary resistance) interface
   library** (§VIII) — this repo does **not** address this. Genuine candidate for actual
   new work, not just reframing, but a substantial separate project (would need either a
   literature-compiled TBR database or new interface measurements) — not a quick add.
3. **UQ not propagated through multiscale/reduced-order model chains** (§VIII-B/C) — same
   alignment as point 1 (MC-Dropout finding).
4. **Incomplete multiphysics coupling** (thermal-mechanical-electrical) (§VIII/conclusion)
   — this repo's throttling (electro-thermal feedback) and microchannel (fluid-thermal)
   mechanisms are concrete steps toward exactly this, already built and pilot-measured
   (Track A above). **Action**: connect explicitly in §10 rather than leaving Track A
   framed only as "does ridge still win" — it's also a direct answer to a field-identified
   gap.
5. Two other findings (no single framework suffices; general PDE-surrogate OOD/temporal
   generalization failure) support the paper's framing generally but aren't specific
   action items.

### Track C — Interface-property uncertainty as a noise-floor argument (started and measured 2026-08-16)

**Motivation.** Two independent findings converged. (1) The multiscale-3D-IC review
[Barua/Udoy/Aziz, arXiv:2604.03290 §VIII] identifies the absence of standardized,
uncertainty-aware thermal-interface data as an open problem, noting reported TBR values
vary substantially between measurement groups. (2) McGreivy & Hakim [Nature Mach. Intell.
2024, arXiv:2407.07218] show 79% of ML-for-PDE papers compare against weak baselines.
Combining them suggests a sharper question than either: **if plausible uncertainty in the
interface properties fed to the simulator moves the answer by more than the model-vs-model
differences being optimized, the entire surrogate accuracy race is running inside the
noise floor of its own inputs.** That is a stronger and more general form of this repo's
existing "ridge already solves it" result, and it is directly measurable with what the
repo already has.

**Method.** `scripts/gen_interface_uncertainty_pilot.py`. Critically — and unlike the TIM
k-sweep already present in the geometry5/6 training data, which varies power pattern, HTC
*and* TIM k simultaneously and therefore isolates nothing — every scenario holds power
pattern, HTC and ambient **fixed** and varies exactly one interface conductivity at a
time, across ranges taken from `docs/assumptions.md` rather than invented. Run at two
operating points, because interface resistance only matters in proportion to the heat flux
crossing it: a low-power control (`uniform`, ΔT≈7 K) and a high-power case
(`split_chiplet_a_hot`, 237 W/cm², HTC 50000). 30 real 3D-ICE solves, 0 failures.
Analysis: `scripts/analyze_interface_uncertainty.py` → `results/interface_uncertainty_{low,high}power.json`.

**Result — peak-junction-temperature spread from one interface property alone:**

| interface | k range (W/m·K) | low-power spread | **high-power spread** | hotspot shift |
|---|---|---|---|---|
| `tim_sink` (thermal grease) | 1–8 | 5.23 K | **28.25 K** | 504 µm |
| `tim_top` (TIM1, indium) | 5–80 | 0.52 K | 4.32 K | 496 µm (low-p) |
| `hybrid_bonding` | 60–400 | 0.01 K | 0.02 K | 248 µm (low-p) |

Reference line: the measured ridge-vs-FNO detrended-MAE gap on throttled data is
**1.09 K** (`docs/report.md` §9.7) — i.e. the size of difference this literature competes
over.

**Three findings, in order of importance:**

1. **Thermal-grease conductivity uncertainty alone moves peak T by 28 K at a realistic
   high-power operating point — ~26× the model-vs-model gap.** k=1→8 W/m·K is not a
   contrived range; it is the ordinary literature spread for thermal greases, and a real
   package's grease also degrades over its lifetime. Even the low-power control (5.23 K)
   already exceeds the surrogate gap by ~5×.
2. **The effect scales with heat flux, as physics requires** (5.23 K → 28.25 K from the
   low- to high-power case), which is a useful internal consistency check: this is a real
   thermal-resistance effect, not a numerical artifact.
3. **Interface uncertainty moves the hotspot *location*, not just its temperature**
   (248–504 µm from a pure material-property change, with power held fixed). This matters
   because §9.4 established hotspot localisation as the metric that actually discriminates
   surrogates — and it is being perturbed by an input nobody varies or reports.

**Bonus self-check**: `assumptions.md` §2.3 flags the hybrid-bonding layer as a known ~28×
overestimate of real Cu-Cu bond resistance but judges it "low priority" because the
absolute error is small. This sweep confirms that judgment empirically for the first time:
sweeping it 60→400 W/m·K moves peak T by 0.02 K, i.e. genuinely negligible. A documented
modelling assumption, now measured rather than argued.

**Honest caveats, to carry into any writeup:** one geometry (geometry5), one scenario per
power level, one parameter varied at a time (so no interaction effects), and 3D-ICE is
itself the ground truth — this measures *model-input sensitivity*, not validated
real-hardware variation. It establishes that the benchmark's answer is highly sensitive to
an unreported input, which is the claim being made; it does not establish what the true
physical spread is on real silicon.

**Why this is a genuine contribution rather than a reframing:** it is the one item in this
plan that produces a *new measured result* rather than better positioning of existing
ones, it required no new modelling machinery (`layer_k_overrides` was already wired
end-to-end), and no paper found in the 2026-08-16 literature pass propagates interface-
property uncertainty through a chip thermal model and compares the resulting spread
against surrogate model error.

**Novelty correction (2026-08-17), stated plainly rather than left implicit:** the
*technique* here — Monte Carlo / parametric propagation of TIM and interface-resistance
uncertainty through a package thermal model — is established practice, not new (Electronics
Cooling, 2018; arXiv:2606.26176 runs N=2,000-trial Monte Carlo for process variation in
3.5D packages, 2026). What is not established, and is what this section actually claims,
is using that propagated spread as a *reference line against neural-surrogate accuracy
claims specifically* — nobody found in the literature pass asks "does this exceed the
gap the surrogate literature competes over."

**Multi-geometry extension (2026-08-17).** Repeated the single-parameter sweep on
geometry1, geometry4 and geometry6 (98 solves, 0 failures;
`data/3d-ice-interface-multi/`), at each geometry's own highest-power `hotspot` scenario
(HTC 50000) rather than geometry5's `split_chiplet_a_hot`:

| geometry | interface | k range | peak-T spread | hotspot shift |
|---|---|---|---|---|
| geometry1 | tim_top | 5–80 | **40.62 K** | 424 µm |
| geometry1 | tim_bottom | 1–8 | 36.01 K | 141 µm |
| geometry4 | **tim_die** | 1–8 | **75.98 K** | — (no shift) |
| geometry4 | tim_sink | 1–8 | 25.65 K | 556 µm |
| geometry6 | tim_sink | 1–8 | 12.01 K | 82 µm |

Every value is a physically sane, monotonic function of k (verified by hand, e.g.
geometry4's `tim_die`: k=1→143.0°C, k=2→104.4°C, k=4→80.6°C, k=6→71.7°C, k=8→67.0°C — not
a fluke). geometry4's 75.98 K is now the largest single-interface spread measured in this
project, ~70× the 1.09 K reference gap. geometry6's spread (12.01 K) is smaller than the
others because its die area is much larger (42×14 mm) and spreads heat further for the
same block power density, not because the mechanism is weaker there — a useful reminder
that this sensitivity is geometry-dependent, not a fixed property of the benchmark.

**Interaction effects, tested and corrected (2026-08-17).** Ran a 3×3 joint sweep of
`tim_sink` × `tim_top` on geometry6 (9 scenarios) to test whether two interface
uncertainties compound super- or sub-additively — the main caveat on the single-parameter
result. **A real bug was caught and fixed in the analysis before trusting the result**:
the first version of `scripts/analyze_interface_uncertainty.py` identified the "both at
nominal" baseline point by checking whether the two override *values* were equal to each
other — which is never true here (`tim_sink` and `tim_top` have non-overlapping sweep
ranges), so it silently fell back to an arbitrary grid corner and would have reported a
false **9/9 "super-additive"** result. Fixed by defining nominal-k per layer explicitly
and locating the true double-nominal point in the grid. Corrected result:
**0/9 super-additive — the two effects compound linearly (additively) at this operating
point**, agreeing with the sum of their individual single-parameter effects to within
0.01 K on every one of the 9 grid points. This is the more scientifically useful finding:
it means a first-order, independent treatment of these two interface uncertainties would
have been adequate here, not an oversimplification. Tested at one (lower-power) operating
point on one geometry; a higher-power interaction test, where the individual effects are
larger, is a natural follow-up but not yet run.

### Track D — Leakage/temperature positive feedback (built and measured 2026-08-17)

**Motivation.** Throttling (Track A) is *negative* feedback: hotter → less power →
cooler. It is self-limiting and degraded ridge only modestly (spatial R² 0.970 → 0.890/
0.919). Subthreshold leakage is *positive* feedback: hotter → more leakage power →
hotter. It is self-amplifying and, above a critical loop gain, has no steady state at all
(thermal runaway) — structurally the sharpest available test of this benchmark's central
linearity claim, since the map from requested to delivered power is no longer even
guaranteed to be well-defined, let alone affine.

**Novelty is in the test, not the mechanism.** Self-consistent leakage-temperature
solving to predict thermal runaway is established — Chen et al., *ICCAD* 2006 ("Leakage
power dependent temperature estimation to predict thermal runaway"), and current work
(ATSim3D, arXiv:2601.11050, Jan 2026) explicitly builds this into a fast 3D-IC thermal
simulator. What's new here is asking whether a *linear baseline* still solves the
CONVERGED portion of such a dataset, which nothing found in the literature pass asks.

**Method.** `src/scenario/leakage.py` (`apply_leakage_feedback`) — damped fixed-point
iteration on `P_total(T) = P_dynamic + P_leak_ref · 2^((T−T_ref)/k_double)`, referenced to
a nominal *junction* temperature (85°C default — a real bug was caught before running any
real solves: an earlier version referenced ambient (25°C), which would have spuriously
amplified every realistically-hot scenario; fixed and covered by 10 unit tests against a
stub simulator, including an explicit high-loop-gain test that must report runaway rather
than a false steady state). Wired into `main.py`/`NPZExporter` mirroring throttling's
convention exactly: delivered (feedback-amplified) power in `block_power_*`, the original
request preserved in `nominal_block_power_*`, and `scripts/baselines.py::collect_block_keys`
extended to prefer nominal power whenever `leakage_enabled` is set (the same
answer-vs-question bug throttling had, fixed proactively this time rather than found
after the fact).

**Result** (`scripts/gen_leakage_pilot.py`, geometry1, 20 scenarios, leakage settings
swept from benign to aggressive; `scripts/baselines_leakage.py` for the fit, which
excludes runaway/non-converged scenarios before scoring — including a 506,854°C numerical
divergence would test "does ridge survive garbage," not the real question):

- **13/20 converged, 6/20 ran away, 1/20 neither** (still slowly climbing at the
  iteration cap, excluded from both categories). A 30% runaway rate at these settings.
- **Among converged scenarios (9 train / 4 test), ridge still wins**: det.MAE 0.248 K,
  spatial R² **0.972** — matching the un-throttled baseline (0.970) and beating the
  throttled result (0.919). Positive feedback, when it settles at all, does not degrade
  the linear approximation any further than the ordinary dataset already sits at.
- **The real finding is what ridge's regression framing doesn't even attempt**: 30% of
  scenarios have no steady-state temperature field to predict at all. That's not a
  regression-accuracy question — it's closer to a stability/classification problem
  (will this scenario converge or run away?), and nothing in this benchmark's current
  metrics tests it.

**Honest caveats:** one geometry, small converged sample (9 train/4 test) from a
20-scenario pilot, one damping/gain schedule. The direction of the regression result
(ridge still wins on convergent data) is clear, but the exact numbers would tighten with
a larger or gentler batch — not generated in this pass since the qualitative answer was
already unambiguous and a bigger run costs another 20–30 min of real 3D-ICE solves.

**What this changes about the paper's central claim:** it survives, and gets a sharper
boundary. Every mechanism tested so far that keeps the problem well-posed (throttling,
per-cell power, TSV fields, underfill layouts, now leakage-when-convergent) leaves ridge
winning or close to it. The one place this benchmark has found real nonlinearity is where
the problem stops being well-posed at all (leakage runaway; and, more mildly, the
microchannel pilot's hotspot-localisation loss to nearest-neighbour, Track A above) — not
in any of the smooth closed-loop feedback mechanisms tried so far.

### Prioritized action list

1. **Cite arXiv:2604.03290 and arXiv:2510.15968 explicitly in `docs/report.md` §10**,
   connecting Track A/B/XAI results to the field's own stated open problems rather than
   presenting them as self-motivated — cheap, high-value, not yet done.
2. **Close the `results/` reproducibility gap** for the remaining un-archived tables
   (§9.1/9.3/9.4/9.5) — write `scripts/regenerate_all_results.py`. Medium effort, directly
   strengthens the audit trail a reviewer would check first.
3. **Run the prepared A3 GPU notebook** (`notebooks/kaggle_a3_throttled_arch_comparison.ipynb`)
   — still blocked on GPU access, unchanged from before this pass.
4. **TBR standardization** (open-problem #2 above) — flagged as a real, distinct
   opportunity, explicitly scoped as future work rather than attempted now; would need
   its own brainstorming/design pass before starting.
