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

**Not yet done — the integration gap.** The mechanism is not wired into `main.py` or
`NPZExporter`: the coolant stack element has no Tmap output, so `generate_coords_and_indices`
(which derives coordinates from `geometry.layers` independent of what the simulator
actually emits) would produce coordinates for a z-region 3D-ICE never returns temperatures
for — a real mismatch that needs the coords/export pipeline to know about coolant layers
before this can produce a shippable geometry7 or an alternate-cooling geometry6 variant.
Scope for that: teach `generate_coords_and_indices`/`NPZExporter` to skip (or separately
handle) coolant sublayers, then sweep flow rate as a scenario parameter and re-run the
ridge baseline — if ridge still solves it, that's a stronger result than anything
currently in the paper; if it doesn't, that's the benchmark's reason to exist.

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
`data/3d-ice` yet. Microchannel needs the coords/export pipeline fix described above;
throttling just needs `main.py` to call `attach_throttling` the way it already
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
