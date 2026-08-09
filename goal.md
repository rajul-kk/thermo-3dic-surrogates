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

### Suggested sequencing — status 2026-08-09

1. ✅ Baseline FNO trained on geometry1.
2. ✅ B1 (rescoped the generalization claim in README/report.md).
3. ✅ A1 → A2 (throttling CLI + microchannel pipeline verified/scripted; pilot batches
   generated and measured — see Track A above).
4. ◐ A3 partial — one baseline-FNO run each on throttled/un-throttled data; the full
   CondFNO/CNO-FNO+attention comparison this item specified is still open.
5. ◐ B2 done as a first pass — mechanism built, validated end-to-end, one
   leave-one-geometry-out run completed with a genuinely mixed result (3/4 metrics
   improved modestly, spatial R² got worse). Not yet multi-seed or multi-holdout, so not
   a settled result either way.

**What's still open, in priority order:** (a) multi-seed/multi-holdout repeats of B2
before drawing a conclusion — in progress, see below, (b) ✅ done 2026-08-09 — exported
`power_nominal` so FNO's throttled comparison is apples-to-apples with ridge's (Track A3
above), (c) the full A3 architecture comparison, ideally GPU-scale rather than another
CPU smoke test.
