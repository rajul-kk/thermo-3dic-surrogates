# fieldlint round 5 (v5): pre-registered held-out measurement

**STATUS: DRAFT, NOT FROZEN (not committed as a freeze; section 12 is empty until the lead confirms).** Written 2026-10-19 by the Phase-C agent under the user's approvals of 2026-10-15 (the eight decisions are listed in section 2). Nothing in this
document has been used to run a rule on a v5 held-out dataset. After the freeze commit the only permitted edits are the entries in section 12 (deviation log).

The data cards (datasets, mapping, documentation, per-dataset predictions) are in `datasets_v5.md` in this directory and are part of this pre-registration. Sections 6, 7 and 11
(metrics, adjudication, post-hoc rules) and the criteria C1-C4 are carried over from `protocol_v4.md` unchanged in substance; sections 5 and 10 add the new fault kinds and criteria C5-C7.
The plan that led here is `plan_v5.md` (a draft, not frozen; its diagnoses and its section 3 are superseded by this document where they differ).

## 1. Question

Do the **frozen v5 rules** (a) detect the input faults they were built for on steady diffusion-type datasets from files no fieldlint version has run on, (b) stay quiet on those datasets
as released, (c) decline, rather than raise false alarms, on steady and time-dependent controls (now including a steady advection-dominated thermal flow, a steady Helmholtz with a
real point source, and two time-dependent sets, one of them thermal convection), (d) find exact duplicates across and within splits down to a 1% dose, and (e) find a sample-pairing
(index shift) fault with F009 on a contiguous block of 64 samples?

v5 differs from v4 in three rule changes made on the v4 failures and a v4 limitation (F005 honours the scope gate; F005's below-ambient warning yields to a weak-minimum-principle probe;
F009 scores a contiguous block of 64 samples instead of 32 evenly spaced ones). They were tuned on **every** dataset seen up to Phase A (README "Round 5 (v5) design", `results/bench_results_v5_tuning.md`),
including the v4 held-out sets and the two v4 control failures. The v4 sets are therefore no longer held out.

## 2. Decisions taken before this document (user, 2026-10-15)

1. Fix scope A + B: F005's below-ambient check honours `scope_skip` (A); the weak-minimum-principle probe `V >= 0.05` on at least 50% of the checked samples turns the warning into an info line (B). No extension into other rules.
2. F009 samples a contiguous block (the first 64 samples) as the shipped rule. The old evenly spaced behaviour (v4: 0 of 9 `shift_index`) is reported honestly (Arm L below).
3. Source-less Helmholtz controls keep the invented uniform source (value 1).
4. Thermal sets: this project's own layout-varied 3D-ICE solves (plan T5) are excluded from the binding sets; independently generated sets (T6) are non-binding; SISSA thermal-conduction (T1) may be used for evaluation without redistribution (its Hugging Face licence field is blank: recorded). No author was contacted.
5. Advective control: FlowBench lid-driven cavity with heat transfer (A1; one file pair) plus a generated advection-diffusion set (A5, non-binding).
6. C3 stays binary; a graded view is descriptive only and is defined in section 10 before any run.
7. C5 (F004 duplicates) is binding; C6 Arm B is binding; the shipped-F009 Arm A is reported, non-binding.
8. Download budget about 9-12 GB plus ranged reads inside large files; data under `D:\fieldlint_data_v5\`, one new directory per download.

## 3. What Phase B looked at (disclosure)

Phase B acquired the data and wrote the mapping. **No fieldlint rule of any kind was run on any held-out dataset, not even the F001 integrity check** (v4 had run F001 as a loader smoke test; v5 did not). Loadability was
checked by the converters themselves (numpy/h5py/pyarrow shape and dtype reads and the copying). Inspected: file names, sizes, sha256 against the server ids, array shapes and dtypes, the README/paper/code of each dataset (citations in `datasets_v5.md`), and the following statistics, each
read only to settle a mapping question and written into the card: FlowBench (first 50 samples): per-channel min/max, number of distinct values in sample 0 and whether a channel is constant per sample (to identify the channels, which do not match the paper's table);
SISSA (first sample of the 2_stripes.star split): coordinate bounds, the number of unique x/y values, value ranges of `constraints_values`, `conductivity` and `steady`, and whether points/cells/constraint indices are the same in all 100 samples of each split. Nothing else was computed. The generated sets (T6, A5) were generated from a frozen script (`gen_v5.py`, parameters fixed in its docstring before generation) and not examined.
The staged presets were written before any rule ran on any held-out file; the Phase A agent that tuned v5 was given no information about these data.

## 4. Datasets and roles

Roles and reasons are in `datasets_v5.md`; this table is the registry (the same list is in `run_v5_heldout.py`).

| role | meaning | counted in section 10? |
|---|---|---|
| **inscope** | steady diffusion-type data with a documented equation of the form -div(k grad u) = f, Dirichlet boundary, 2D | yes (C1, C2, C4, C5, C6, C7a) |
| **inscope_nb** | in-scope by construction but generated by this project (T6) | reported, non-binding |
| **diagnostic** | a processed or under-documented set | reported, non-binding |
| **control** | out of scope by documentation (steady non-diffusion, advective, or time-dependent) | yes (C3, C7b) |
| **control_nb** | generated advective control (A5) | reported, non-binding |

**In scope, binding (N = 6).** ConDiff 128 x 128 test files, 200 samples each (all of the file), mapping `v5_condiff`: `condiff_cubic1.0`, `condiff_cubic2.0`, `condiff_exponential0.4`, `condiff_gaussian0.4`, `condiff_cubic0.1`, `condiff_exponential1.0`.
These are fresh files of one generator family and one solver (ConDiff; the five v4 files were the same family). **No in-scope held-out set from a second, independent diffusion family was obtainable** (section 3 of `datasets_v5.md`); this is the main limitation of the round.

**In scope, non-binding (2).** `chip_fem_a` (k varying, power rectangles, pure conduction) and `chip_fem_b` (k = 1, Gaussian hot spots, lumped vertical loss), 300 samples each, generated by `gen_v5.py` with a Q1 finite-element solver (not 3D-ICE, not HotSpot, not the 5-point stencil of F008), units K, ambient 300.

**Diagnostic (1).** `sissa_thermal`: SISSA thermal-conduction, the splits 2_, 3_, 4_stripes.star (300 snapshots, rasterised from an unstructured mesh to 128 x 128). Processed, source-less, equation undocumented.

**Controls, binding (4 files, 6 runs).** `helmholtz_staircase` (steady Helmholtz, real point source whose position is not stored, mapped with the invented uniform source), `flowbench_ldcht` (steady, advection-dominated heat transfer; mapping per card), and the time-dependent
`acoustic_inclusions` and `rayleigh_benard`, each run twice: with `time_dependent: true` (**binding**) and without (**C3-strict, reported only**).
**Control, non-binding (1).** `advdiff` (generated advection-diffusion).

**Native grid (for C4), decided from the documentation and fixed here.** The six ConDiff files (cell-centred finite volume on exactly the stored 128 x 128 cells; as in `datasets_v4.md` 2.1): **6 sets, binding for C4**. `chip_fem_*` are native (the FEM node grid) but non-binding. SISSA is processed.

**Sample counts.** First N <= 1000 samples in file order, fixed in advance: ConDiff 200 (all), chip 300, SISSA 300 (100 per split, in the order 2, 3, 4 stripes), FlowBench 1000, Helmholtz-staircase 208 (the first 13 trajectories of each of the 16 train files, omega-major order),
acoustic 200 (test chunks 36 and 37), Rayleigh-Benard 256 (first 256 of test files then valid files, sorted by name), advdiff 300.

**Registry closed.** A dataset that failed to load or run stays in the table as "not run" and counts as not satisfying whichever criteria it appears in. Sought but not obtainable or not taken: see `datasets_v5.md` section 3 (the other candidates of plan section 2, with reasons). This project's own layout-varied 3D-ICE solves (plan T5) are not registered (decision 4; each directory has 40-55 samples, below the 200-sample minimum).

## 5. Fault set

**Standard faults (identical to v2-v4)**, from `run_bench.FAULTS`, injected by `run_bench.inject`: `transpose`, `flip_x`, `flip_y`, `roll1`, `roll2`, `roll3`, `shift_index`, `unit_slip`, `cross_dup`, and the 20% variants `transpose@20`, `flip_x@20`, `flip_y@20`, `roll1@20`, `shift_index@20`
(seed 12345). Splits are assigned by position (first 80% train, last 20% test). `cross_dup` replaces the last 10% of the samples by copies of earlier samples (= dose `dup_x10` of C5). `unit_slip` applies to the K datasets only (the two chip sets); `transpose` needs a square grid (all in-scope grids are square).
Rule options: `max_samples = 400` (frozen).

**C5 fault kinds (`run_v5_heldout.inject_v5`, seed 54321), applied to every in-scope / non-binding in-scope / diagnostic dataset, F004 only:**
`dup_x1`, `dup_x5` (1% / 5% of the samples, at least 2, in the test region replaced by whole-sample copies of random earlier train samples), `dup_within5` (5% of the train positions replaced by copies of random earlier train samples),
`dupnoise_1e-7`, `dupnoise_1e-4`, `dupnoise_1e-2` (5% test-region copies of train samples to which Gaussian noise with standard deviation 1e-7 / 1e-4 / 1e-2 of the sample's range was added to u; inputs unchanged).

**C6 fault kinds (u of sample i replaced; inputs unchanged):** `shift_index` (u_i := u_{i+1}, the standard fault), `shift_index_5` (u_i := u_{i+5}), `permute20` (a seeded random 20% of the positions have their u permuted among themselves; called `swap_blocks20` in `plan_v5.md`, renamed because no blocks are involved), `shuffle_all` (a seeded permutation of all positions).
Arms (F009 only): **B** = the lint of the first 64 consecutive samples of the faulted dataset (`ds.head(64)`, the documented way to lint a block); **A** = the shipped rule applied to the whole dataset (it scores the first 64 consecutive samples; practically equal to B, reported as the "shipped" arm of decision 7);
**L** = the legacy v4 sampling, emulated exactly by presenting the 32 evenly spaced samples that v4 scored (`ds.indices(32)`).

## 6. Metrics

A rule **detects** a fault (or fires on clean data) when its result is `flagged` with a warning or error finding (`summarise()`; info does not count); `skipped` counts as a miss and is tallied separately. Rules considered: F001, F002, F003, F004, F005, F008, F009 (F006 informational, F007 needs flux).
Per-rule and any-rule rates per fault, clean false positives, skip rates, control behaviour (declined vs false alarm; F004 literal duplicates are adjudicated like clean findings; F001 must be ok), and F008 applicability, exactly as `protocol_v4.md` section 6.
**Clean-adjusted detection** (a rule is credited with a fault only if it is not flagged on the clean copy of the same dataset; the any-rule rate likewise) is the basis of C1 and C6. C5 uses its own definition (section 10). Both numbers are computed offline by `report_v5.py` (committed with this document).
Confidence intervals: exact Clopper-Pearson (two-sided 95% for rates, one-sided 95% lower/upper bounds for pass/fail statements); the **pooled-trial interval is the optimistic one** (three faults on one dataset are not independent), so the dataset-cluster view (k of k datasets with all three detected) is reported next to it.

## 7. Adjudication of findings on clean, as-released data

As `protocol_v4.md` section 7, unchanged: a finding on a clean in-scope copy is a TRUE fault (not a false positive) only through (a) documentation, (b) a physics check stated in advance (P1: for -div(k grad u) = f with f >= 0, k > 0, u = 0 on the boundary, u >= 0; P2: bit-identical u in two samples is a duplicate; P3: non-finite or inconsistent arrays are a fault) or (c) a maintainer statement; everything else is a false positive;
severity classes benign/material; sign conventions fixed from documentation and code before the runs (`datasets_v5.md`). Pre-stated: ConDiff has a sign-changing source (P1 does not apply to it).

## 8. Exact commands and the run-once rule

From the repository root, with the frozen environment (versions are written into each result file):

```
python tools/fieldlint/bench/run_v5_heldout.py --list                  # prints each registered run and whether its file exists; runs nothing
python tools/fieldlint/bench/run_v5_heldout.py --run-all control       # one result file per dataset in bench/results/v5_heldout/
python tools/fieldlint/bench/run_v5_heldout.py --run-all control_nb
python tools/fieldlint/bench/run_v5_heldout.py --run-all inscope
python tools/fieldlint/bench/run_v5_heldout.py --run-all inscope_nb
python tools/fieldlint/bench/run_v5_heldout.py --run-all diagnostic
python tools/fieldlint/bench/report_v5.py                              # mechanical tables C1, C3-C7; C2 and the prediction audit are by hand
```

* **Run once.** `run_v5_heldout.py` writes `bench/results/v5_heldout/<name>.json` and **refuses** to run a name whose file exists. A crash before the file is written may be re-launched; the re-launch and its cause are logged in section 12. A result is never re-run because of its content, a different conclusion, a bug found in the converter or the preset, or a changed mind about the mapping.
* At most 2 CPU jobs at a time. Results are not inspected between runs except for crash detection. Order: controls (cheap), then in-scope (clean + 14 standard faults + C5 + C6 stages).
* Each result file records: git HEAD, uncommitted tracked changes, the sha256 of the frozen file, the current code hashes, the hashes of the bench files and staged presets, library versions, argv.
* The hash check of `bench/results/frozen_thresholds_v5.json` passes at every run (the runner exits otherwise).
* No threshold, rule, mapping, role or criterion may change after the first v5 held-out result exists.

## 9. Pre-registered predictions

Per-dataset predictions (with the documentation each follows from) are in `datasets_v5.md`. They are **predictions, not criteria**: a wrong prediction does not fail the protocol but is reported as wrong. Summary:

* ConDiff (6): F008 applicable (the five v4 ConDiff files, now seen, had median R^2 0.88-1.00); F002 skipped, F003 weak; F004/F005 ok; whole-dataset transpose/flip detected by F008 (and F003 where it works); `shift_index` detected by F009 in Arm B on the applicable sets; copies detected by F004.
* `chip_fem_*`: F008 applicable with R^2 below ConDiff's (Q1 elements give a 9-point operator, not the 5-point stencil); everything else as ConDiff, plus `unit_slip` detected by F005.
* `sissa_thermal`: most rules skip (no source; the problem is boundary-driven); not informative about detection.
* Controls: `helmholtz_staircase`: F002/F003/F008/F009 skipped or silent; F005 info (weak-minimum-principle probe) rather than a warning; `acoustic_inclusions` and `rayleigh_benard` flagged: the four steady rules skip as out of scope and F005 reports "not applied"; unflagged: F002/F003 may fire, F005 info;
  `flowbench_ldcht`: **uncertain** (the object mask is a user's reading of a cold Dirichlet body as a source; F002 can fire if the advection probe does not decline; F005 may warn about an undershoot of about 0.2% of the range below ambient 0, which is above the 0.1% tolerance and below the 5% probe threshold); `advdiff`: declined as advective.
* The advection probe is expected to trigger on `advdiff` and, with less confidence, on `flowbench_ldcht`; on no in-scope set.

## 10. Criteria and verdict

Let N be the number of **inscope** (binding) datasets that loaded and ran, and "applicable trial" an (in-scope dataset, whole-dataset fault) pair for which `inject` produced a faulted copy. All six criteria are binding for the verdict (decisions 1-7).

| id | criterion | threshold | why this number |
|---|---|---|---|
| **C1** | clean-adjusted any-rule detection of whole-dataset `transpose`, `flip_x`, `flip_y`, pooled over the binding in-scope applicable trials | **>= 80%** (point estimate); exact one-sided 95% lower bound and the dataset-cluster view are reported | unchanged since v3 (0.86, 1.00 passed); not changed after a pass. With N = 6 (18 trials) all-hit gives lower bound 0.85, one miss 0.74 |
| **C2** | clean false positives on binding in-scope datasets after adjudication | **at most floor(N/10)**, i.e. **0 for N < 10** | unchanged strict reading |
| **C3** | no false alarm among F002, F003, F005, F008, F009 on any binding control run (time-dependent controls with `time_dependent: true`; steady ones as they are) | **0** | unchanged and **binary**. The unflagged runs and the generated control are reported, not binding |
| **C4** | F008 applicable (clean `status != skipped`) on the native binding in-scope sets (6 ConDiff files) | **>= 50%** (at least 3 of 6) | unchanged |
| **C5** | F004 duplicate detection, F004 only. "Detected" = F004 flagged on the faulted copy AND its cross-split + within-split duplicate-sample count exceeds the clean copy's count. (a) exact copies `dup_x1`, `dup_x5`, `cross_dup` (= `dup_x10`), `dup_within5` over the N binding in-scope datasets (4N cells); (b) `dupnoise_1e-7`; (c) clean copies | (a) **>= 95%**, (b) **>= 90%**, (c) **0** clean F004 findings after adjudication (P2); `dupnoise_1e-4` and `dupnoise_1e-2` (above `dup_rtol` = 1e-5) are a documented non-detection, reported, not a criterion | the hash path is deterministic; v2-v4 pooled `cross_dup` detection was 18/18, 20/20, 9/9; 95% leaves room for one miss in 20 and still catches a broken path. With N = 6 and 24 cells, 24/24 gives lower bound 0.88 |
| **C6** | F009, Arm B (head block of 64): (a) clean-adjusted `shift_index` detection over the binding in-scope sets on which F009 is applicable on the clean copy; (b) F009 findings on clean copies; skip rate reported | (a) **>= 70%**, (b) **0** | tuning: `shift_index` was detected on 12 of 12 sets on which F008 ran (v2 tables). Arms A and L, `shift_index_5`, `permute20`, `shuffle_all`: reported, not criteria. Prediction for Arm L: about 0% (v4: 0 of 9) |
| **C7** | advection probe: (a) specificity: it triggers on **0** binding in-scope clean copies and **0** of their whole-dataset transpose/flip copies; (b) sensitivity on `flowbench_ldcht` (and `advdiff`, non-binding): reported as k of n | (a) 0 triggers. (b) pre-registered prediction "declined as advective", the opposite is a probe failure and is stated so | a probe claim may be made only if (a) holds and (b) was run held-out. C7 is **reported, not part of the verdict** (it needs a second advective family to say anything general) |

**Verdict, decided mechanically from the results and not by the author:**

* **PAPER-WORTHY (as a tool paper with a measured scope)**: C1, C2, C3, C4, C5 and C6 all hold.
* **PARTIAL (negative-result / limits-of-lint paper, or workshop note)**: C1 and C3 hold but C2, C4, C5 or C6 does not.
* **NOT PAPER-WORTHY (as measured)**: C1 fails, or C3 fails.

**C3 graded view (descriptive only; defined here, before any run; it never changes the verdict).** For the binding control runs: the list of rules with a warning or error and their severity, the number of alarmed runs of the binding runs, the false-alarm upper bound (one-sided 95%), and the label "no alarm", "single rule" (every alarm is the same rule) or "several rules".
The binary C3 stands as the criterion.

Not criteria, but reported with exact confidence intervals: per-rule detection for all faults, roll and index-shift faults, the 20% variants, the non-binding sets, the C3-strict runs, the Arm A and Arm L F009 results, the F008 applicability list, the number of TRUE faults found on clean data under section 7.

**Scope caveat that goes with any verdict.** All binding in-scope sets are fresh files of **one** generator family (ConDiff, a generic variable-coefficient diffusion family, native finite volume). No held-out application field (heat conduction with a real chip geometry) is among the binding sets: the thermal sets are generated by this project or are processed and source-less (non-binding). The study cannot support a claim about "PDE datasets in general", nor about 3D-IC data outside the tuning geometries.
Independence caveat: the six ConDiff files share one generator and one solver; results are also reported per family and per dataset cluster.

## 11. Post-hoc analysis

Any analysis not specified in sections 5-8 (re-mapping, other face-mean conventions, a re-run after a converter fix, which samples fired, ablations, extra fault types, other files) is **post hoc**. It may be done after all registered runs are complete, must be labelled "POST HOC" in every table and sentence that reports it, must not alter any section-10 number, and its result files go to `bench/results/v5_heldout_posthoc/`. Explicitly pre-announced post-hoc diagnostics: the unset-source version of the Helmholtz controls (decision 3 keeps the uniform source binding).

## 12. Freeze and deviation log

* Frozen by: user pre-approval 2026-10-15 + agent (Phase C), at **the commit that adds this line** (its hash is in the git log; the commit message begins "fieldlint v5: FREEZE held-out protocol"). The freeze commit contains this protocol, `run_v5_heldout.py`, `report_v5.py`, `gen_v5.py`, `convert_v5.py` and the staged presets, and precedes every held-out run.
* Conditions of the pre-approval, all met: C1-C4 definitions and thresholds unchanged; C3 binary; adjudication and run-once rule unchanged; v5 hash check at every run; staged presets passed as configs (not moved into `src/`); N and the native list written explicitly; limitations of section 10 stated.
* Deviations after freeze (re-launches, datasets that could not be run, etc.): none yet.

## Appendix: references

* Rules and thresholds: `tools/fieldlint/README.md` ("Round 5 (v5) design"), `src/fieldlint/rules.py`; `bench/results/frozen_thresholds_v5.json`, `bench/results/bench_results_v5_tuning.md`.
* v4 protocol and results: `bench/protocol_v4.md`, `bench/results/bench_results_v4_heldout.md`. Plan: `bench/plan_v5.md`.
