# fieldlint round 4 (v4): pre-registered held-out measurement

**STATUS: FROZEN (see section 12).** Written 2026-10-09 by the Phase-B/C agent under the lead's pre-approval; nothing in this document has been used to run a rule on a v4
held-out dataset. After the freeze commit the only permitted edits are the entries in section 12 (deviation log).

The data cards (datasets, mapping, documentation, per-dataset predictions) are in `datasets_v4.md` in this directory and are part of this pre-registration. Sections 5, 6, 7
(except its sign-convention paragraph), 10 (criteria) and 11 are carried over from `protocol_v3.md` unchanged in substance: the metrics, the clean-adjusted scoring, the
adjudication routes (a)/(b)/(c) with the physics checks P1-P3 and the benign/material classes, the run-once rule, the post-hoc rules and the four criteria with their thresholds. Only the
dataset-specific lists (which sets are in scope, which are native, the value of N) are new.

## 1. Question

Do the **frozen v4 rules** (a) detect the input faults they were built for on steady diffusion-type datasets from generators and files that no fieldlint version has ever run on, (b) stay
quiet on those datasets as released, and (c) decline, rather than raise false alarms, on steady and time-dependent data outside their stated scope?

v4 differs from v3 in two rule changes made on the v3 failures (F003 defers to F008 when the operator fit confirms the stored orientation; an advection probe makes F002/F003/F008/F009 skip on
advective data) and was tuned on **every** dataset seen up to Phase A, including the v3 held-out sets (README "Round 4 (v4) design", `results/bench_results_v4_tuning.md`). The v3 sets are therefore
no longer held out; this round uses fresh files only (`datasets_v4.md`).

## 2. What Phase B looked at (disclosure)

Phase B acquired the data and wrote the mapping (commit `ab1f720`). It did **not** run any fieldlint rule except F001 (integrity), through the CLI with the working-tree code after the v4 freeze commit
`bf27d68` existed (`--rules F001 --max-samples 20`, 13 loads, all ok); and no correlation, R^2, fit, residual or other statistic of the fields was computed. Inspected: file names, sizes, md5/sha256, HDF5 key
names, shapes, dtypes, the README/paper/code of each dataset (see the sign-convention citations in `datasets_v4.md`), and one min/max: the first 50 samples of each converted PDEBench Darcy file
(`tensor` strictly positive, `nu` in [0.1, 1.0]) to confirm the sign convention. The staged presets were written before any rule other than F001 was run; the Phase A agent that tuned v4 was given no information about
these data.

## 3. Frozen rule version

* Rules, options and code: **v4**, `bench/results/frozen_thresholds_v4.json` (frozen_at 2026-10-09T23:45:11+0800, commit `bf27d68`). `run_bench.load_frozen('_v4')` covers `rules.py`, `adapters.py`, `model.py`, `runner.py` and
  every preset JSON in `src/fieldlint/presets/` (28 files); it must pass at **every** run (it exits otherwise). It passes at the time of writing.
* The sha256 of the frozen file is recorded in every result file, with the git HEAD, uncommitted tracked changes, Python / numpy / h5py versions and the sha256 of every staged preset.
* **No edit of any hashed file, no change of any `Options` value, no change to the fault code in `run_bench.py` is permitted.** (`run_bench.py` is not in the code hash but its sha256 is recorded in `frozen_thresholds_v4.json` under `bench_code_sha256`; it is unchanged.)
* **The staged presets are NOT moved into `src/`.** The v4 hash covers all preset JSON in `src/fieldlint/presets/`, so adding files there would break the check. The eight mapping configs in `bench/presets_v4_staging/` are passed by file path
  (`run_bench.build_clean` calls `load_config(<path>)`), as `fno_darcy_native.json` was in v2. **No config sets `operator:`; detection is automatic** (the declaration is not used for any registered run).
* New files outside the hash: `convert_v4.py`, `presets_v4_staging/*.json`, `run_v4_heldout.py`, `report_v4.py`, `datasets_v4.md`, this document.
* The converted data are in `data/_external_v4/` with `*.MANIFEST.json` files; sha256 of every file is in `datasets_v4.md` section 4.

Code revision of the v4 tooling at freeze time: the commit that adds this file (section 12).

## 4. Datasets and roles

Roles and reasons are in `datasets_v4.md`; this table is the registry (the same list is in `run_v4_heldout.py`).

| role | meaning | counted in section 10? |
|---|---|---|
| **inscope** | steady diffusion-type data with a documented equation of the form -div(k grad u) = f, Dirichlet boundary, 2D | yes |
| **diagnostic** | an alternative mapping of an in-scope file that deliberately gives the rules less than they need | reported, not counted |
| **control** | out of scope by documentation (Helmholtz, time-dependent) | yes (section 10, C3) |

**In scope (N = 9).** Clean data plus the fault set of section 5:

| run name | file | family |
|---|---|---|
| `condiff_poisson` | ConDiff `poisson_grid128` test, k = 1, 200 samples | ConDiff |
| `condiff_gaussian0.1`, `condiff_gaussian1.0`, `condiff_cubic0.4`, `condiff_exponential2.0` | ConDiff 128 x 128 test files, 200 samples each, primary mapping `v4_condiff` (k := `k_cell`, the solver's own cell coefficient, derived) | ConDiff |
| `elado_b1_test` | ELADO Poisson B1_test, 1000 samples, 512 x 512 | ELADO |
| `pdebench_darcy_beta0.1`, `..._beta10.0`, `..._beta100.0` | PDEBench 2D Darcy, first 1000 of 10000 each, 128 x 128; **same generator as the tuning sets (beta 1.0, 0.01), fresh files** | PDEBench Darcy |

**Diagnostic (1, not counted).** `condiff_gaussian0.1_nok`: the same Gaussian-0.1 file with k omitted (`v4_condiff_nok`).

**Controls (3 files, 4 runs).** `pdegym_helmholtz` (PDEgym Helmholtz, steady) and `elado_helmholtz_c1` (ELADO C1_test, steady Helmholtz) are binding as they are (no time flag exists for steady data).
`pdegym_wave_layer` (PDEgym Wave-Layer, time-dependent) runs **twice**: with `time_dependent: true`, which the documentation (wave equation, trajectories) prescribes and which is **binding**, and without it (**C3-strict, reported only**).

**Native grid (for C4), decided from the documentation and fixed here.** Native = the stored `u` / `source` grid is the solver's own grid, and the documentation or code says so: the five ConDiff files (cell-centred finite volume on
exactly the 128 x 128 cells; `condiff.py` lines 121-160) and ELADO B1 (P1 FEM on the vertex grid; `poisson_dirichlet.py`): **6 sets, binding for C4**. The three PDEBench Darcy files are **not** in the binding list: the paper states 128 x 128 cells
and a finite-volume time-evolution to steady state, but its data-generation code is not public, so nativeness cannot be confirmed from code. C4 is also reported with them counted as native (sensitivity, not binding).

**Limitations, stated plainly.**

* The fresh in-scope sets are again generic Poisson / Darcy / variable-k families (ConDiff, ELADO, PDEBench Darcy), **not application fields**. Nothing here tests heat conduction, electrostatics or groundwater data (nothing obtainable, `datasets_v4.md` section 3).
* PDEBench Darcy **shares its generator with tuning data** (beta 1.0 and 0.01 were tuning sets; v2 matched them by an arithmetic face mean). These three files test file-to-file stability, not a new generator.
* **No fresh advective control was obtainable** (UCSD has no ConvDiff variant beyond `ConvDiff1.npz`, which is tuning data now; no other advection-diffusion or reaction-dominated steady grid set was found). The v4 advection probe therefore gets **no held-out test** in this round; its margins (`adv_gain` 0.20, `adv_margin` 0.05) rest on one advective dataset (UCSD ConvDiff1).
* The synthetic Helmholtz (DiffusionPDE) still fires F002 under v4 (a known weakness recorded in Phase A); the two Helmholtz controls here are different generators, and if they also fire F002 the C3 verdict reports it.
* ELADO C1 stores y whose meaning (|u| or a component) the paper does not state; the control is mapped as in v3.

The registry is closed. A dataset that failed to load by freeze time stays in the table as "not run" and counts as not satisfying whichever criteria it appears in. **Sought but not obtainable** (nothing run, not replaced): UCSD MTNN variants beyond
Poisson4/ConvDiff1 (do not exist); further PDEgym steady sets (only Poisson-Gauss, used in v3, and Helmholtz exist); PDEBench diffusion-reaction and shallow water as time-dependent controls (opening the large HDF5 by range requests did not complete; Wave-Layer taken instead); ELADO D1 (PDE unstated) and A1_tail / A1_train (same family as v3's A1_test) were not taken; a fresh advective control (see above).

Datasets are subsets by sample count only: the first N samples in file order (N <= 1000; selected before looking at any statistics; the ConDiff test files have 200 samples in all).

## 5. Fault set (identical to v2 and v3)

From `run_bench.FAULTS`, injected by `run_bench.inject` into the inputs (source, k) or the pairing of a clean copy; splits assigned by position (first 80% train, last 20% test) as in v2:

`transpose`, `flip_x`, `flip_y`, `roll1`, `roll2`, `roll3`, `shift_index`, `unit_slip`, `cross_dup`, and the partial-contamination variants
`transpose@20`, `flip_x@20`, `flip_y@20`, `roll1@20`, `shift_index@20` (random 20% of samples, seed 12345).

Applicability is decided by `inject` as in v2: `transpose` needs a square grid (all v4 grids are square); `unit_slip` needs `units == 'K'` (no v4 dataset is in kelvin, so it is
"n/a" everywhere, as for the two Darcy sets in v2); `@20` variants of `unit_slip` and `cross_dup` do not exist. A fault that does not change the input (for example a flip of a
uniform source when there is no `k`) is still scored as v2 scores it: detected or not detected. Rule options: `max_samples = 400` evenly spaced samples (frozen), so per-dataset scoring uses at most 400 samples.

## 6. Metrics

A rule **detects** a fault (or fires on clean data) when its result is `flagged` with a warning or error finding (`summarise()` of `run_bench.py`; info does not count). A rule that is
`skipped` counts as a miss and is tallied separately, exactly as in `bench_results_v2.md`. Rules considered: F001, F002, F003, F004, F005, F008, F009 (F006 is informational, F007 needs flux data).

For each run set:

1. per-rule detection rate per fault, any-rule detection rate per fault (any of the seven fires), pooled over the in-scope datasets (and reported per dataset and per generator family);
2. **clean false positives** on the as-released data: number of in-scope datasets for which any rule fires on the clean copy, after adjudication (section 7);
3. **skip rates**: per rule, the fraction of clean in-scope runs on which it is skipped, with the skip reason string;
4. **control behaviour**: for each control and rule, `declined` (skipped, or ok with nothing flagged) versus `false alarm` (warning or error). F004 findings that are literal duplicates in the data are adjudicated like clean findings. F001 must be `ok` or the control is "not loadable".
5. For F008: the fitted face mean and the clean median R^2 (from `metrics`), so that applicability can be audited against the 0.3 gate.

6. **Clean-adjusted detection** (introduced in v3). v2 scoring credits a rule with a fault when the rule is flagged on the faulted copy, without checking
   whether it is already flagged on the clean copy. If a rule fires on the clean copy of a dataset (as F005 did on CNO Darcy in v3, whose stored output has the opposite sign), it would be credited with every fault of that
   dataset. Both numbers are therefore reported: the v2 count, and the **clean-adjusted** count in which a rule is credited with a fault only if it is *not* flagged on the clean copy of the same dataset. The any-rule rate is
   computed the same way (any rule that is not already flagged on clean). **C1 below is judged on the clean-adjusted any-rule rate.** Both are computed offline from the result files by a script written before the runs
   (`bench/report_v4.py`, committed with this document; its only thresholds are the section-10 numbers, which must be edited together with this document).

Neither the thresholds nor the metric definitions change between the first and last dataset.

## 7. Adjudication of findings on clean, as-released data

A warning or error from any rule on the clean copy of an in-scope dataset is a **TRUE fault** (and then not a false positive) **only if** one of the following holds, and
the citation is written into the results before the finding is looked at again:

* (a) **documentation**: the dataset's own documentation, paper, README, issue tracker or code states the property that the rule reports (for example that inputs and outputs are stored on different axis orders, that the output was given the opposite sign, or that two splits share samples);
* (b) **physics check stated in advance**: a check written into this protocol before the run, listed here: **(P1)** for the documented equation -div(k grad u) = f with f >= 0, k > 0 and u = 0 on the boundary, the weak maximum principle gives u >= 0, so a field that is negative in the interior contradicts the documented equation; **(P2)** for a literal duplicate (F004), bit-identical u in two samples is a duplicate regardless of cause; **(P3)** a non-finite or shape-inconsistent array (F001) is a fault. No other physics check is allowed after the run;
* (c) **maintainer statement**: a public statement by the dataset authors or maintainers (issue, erratum, e-mail quoted with permission) that the property is a known defect.

Everything else is a **false positive**, including a finding that later turns out to be "real" through an analysis done after the run (that analysis is reported, labelled post hoc, and may be written up as a discovery, but does not change the false-positive count).
Findings that depend on a standardisation documented in the dataset's own README (the **diagnostic** role) are not adjudicated; they are reported with the role label.

**Severity class of every TRUE finding** (assigned when it is adjudicated, from the documentation alone):

* **benign**: a sign or scale convention that does not affect what a model learns (for example the stored sign of u, a documented standardisation);
* **material**: a fault that changes what a model learns: orientation, pairing, duplicates across splits, wrong units, non-finite values.

Discovery counts are reported per class. A benign finding is never counted as a headline discovery, and no sentence of the report may present one as such.

**Sign conventions are fixed before the runs, from code and documentation only.** The mapping of each dataset to fieldlint's convention -div(k grad u) = s was decided from the generating code or the paper (citations in `datasets_v4.md`) and not from any rule output: ConDiff: source = +rhs, u = +x (`condiff.py` lines 26-29, 83-85, 121-160: a positive-definite -div(k grad) system), no negation; ELADO B1: source +1 (`poisson_dirichlet.py`); PDEBench Darcy: -div(a grad u) = beta with beta > 0 and u >= 0 (paper App. D.4; no code), u kept as stored. The Helmholtz and Wave-Layer controls carry no sign decision beyond a user's plausible reading (`datasets_v4.md`).

Pre-stated expectation of what (b) applies to: nothing in particular. ConDiff has a sign-changing source (P1 does not apply to it); a clean flag on PDEBench Darcy or ELADO B1 would have to be justified by (a), (b) or (c) after the fact, otherwise it is a false positive.

## 8. Exact commands and the run-once rule

All commands are run from the repository root, with the frozen environment (Python 3.10; versions are written into each result file).

```
python tools/fieldlint/bench/run_v4_heldout.py --list                  # prints each registered run and whether its file exists; runs nothing
python tools/fieldlint/bench/run_v4_heldout.py --run-all inscope       # one result file per dataset in bench/results/v4_heldout/
python tools/fieldlint/bench/run_v4_heldout.py --run-all diagnostic
python tools/fieldlint/bench/run_v4_heldout.py --run-all control
python tools/fieldlint/bench/report_v4.py                              # mechanical tables (C1, C3, C4, per-family, per-fault); C2 and the prediction audit are by hand
```

or one dataset: `python tools/fieldlint/bench/run_v4_heldout.py --run <name>`.

* **Run once.** `run_v4_heldout.py` writes `bench/results/v4_heldout/<name>.json` and **refuses** to run a name whose file exists. A crash before the file is written may be re-launched; the re-launch and its cause are logged in section 12. A result is never re-run because of its content, a different conclusion, a bug found in the converter or the preset, or a changed mind about the mapping. If an error is found after a run, the run stands, the error is reported, and a corrected run is a separate, labelled post-hoc analysis (section 11).
* At most 2 CPU jobs at a time. Results are not inspected between runs except for crash detection.
* Each result file records: git HEAD, uncommitted tracked changes, the sha256 of the frozen file, the current code hashes, the staged-preset hashes, library versions, argv, start time.
* The hash check of the frozen v4 file passes at every run.
* No threshold, rule, mapping, role or criterion may change after the first v4 held-out result exists.

## 9. Pre-registered predictions

Per-dataset predictions (rules expected to apply or skip, with the documentation they follow from) are in `datasets_v4.md`. They are **predictions, not criteria**: a wrong prediction does not fail the protocol but is reported as wrong. Summary:

* F008 is expected applicable on ConDiff Poisson and Gaussian-0.1, ELADO B1 and the three PDEBench Darcy files; uncertain on ConDiff Gaussian-1.0, cubic-0.4 and especially exponential-2.0 (rough, high contrast; the solver's node-mean face coefficients differ from means of `k_cell`). The `nok` diagnostic is expected to skip or fit poorly.
* ConDiff's source is sign-changing white noise (F002 skipped, F003 weak); ELADO and Darcy have uniform sources (F002/F003 skipped), so only F008 sees a transposed / flipped k there, and a skip of F008 would be a miss of the any-rule detection.
* Controls: F008 expected to skip on both Helmholtz sets and on Wave-Layer; Wave-Layer flagged: the four steady rules skip as out of scope; unflagged runs may raise false alarms on F002/F003 (u0 >= 0). The v4 changes are expected to leave the three controls quiet; this is the first test of those changes on files the tuning did not see.
* No advective data: the advection probe is not expected to trigger on any registered set (a trigger on an in-scope set would be a false decline and is reported as such).

## 10. Criteria for a "paper-worthy" verdict (the v3 criteria, approved by the lead and carried over unchanged)

Let N be the number of **inscope** datasets that loaded and ran, and let "applicable trial" be an (in-scope dataset, whole-dataset fault) pair for which `inject` produced a faulted copy.

| id | criterion | threshold | why this number |
|---|---|---|---|
| **C1** | clean-adjusted any-rule detection (section 6, item 6) of whole-dataset `transpose`, `flip_x`, `flip_y` pooled over the in-scope applicable trials | **>= 80%** (point estimate), and the exact one-sided 95% Clopper-Pearson lower bound is reported | the v2 tuning sets gave 17/17, 20/20 and 18/20 (92-100%), the tuning rates are optimistic, and one native-grid held-out gave 3/3; 80% asks for most of that to survive on unseen generators while allowing the Darcy-type sets to be missed. With N = 9 registered in-scope sets and 3 faults (27 trials) the exact one-sided 95% lower bound is about 0.89 at 27/27 and about 0.65 at 22/27 (computed with `report_v4.clopper_lower`), a weak result that the paper would have to state |
| **C2** | clean false positives on in-scope datasets after adjudication | **at most 1 per 10 datasets**: the count of false-positive datasets <= floor(N/10), i.e. **0 for N < 10** (the lead chose the strict reading, review 2026-10-09; it is binding) | v2 had 1/20 on tuning (true duplicates, would be a TRUE finding under section 7) and 0/1 held-out; a tool whose output is read as a warning must be quiet on clean data. |
| **C3** | no false alarm on any control run **with the README's scope declaration** (Wave-Layer with `time_dependent: true`; the two steady Helmholtz controls as they are); the unflagged runs are reported but not binding | **0** false alarms among F002, F003, F005, F008, F009 | v2 controls: Burgers 1D gave false alarms in v1 and none in v2; diffusion-reaction none. The README tells users to set the flag. C3-strict (also the unflagged runs) is reported and is **not binding** (lead decision, review 2026-10-09) |
| **C4** | F008 applicable (clean `status != skipped`) on in-scope datasets classed **native grid** in section 4 of this document (6 registered: ConDiff Poisson, Gaussian-0.1, Gaussian-1.0, cubic-0.4, exponential-2.0, ELADO B1) | **>= 50%** of them, i.e. at least 3 of 6 | F008 is the only rule that sees k-only datasets; v2 had it applicable on 1/1 native and 0/1 processed grids |

Verdicts, decided mechanically from the results and not by the author:

* **PAPER-WORTHY (as a tool paper with a measured scope)**: C1, C2, C3 and C4 all hold.
* **PARTIAL (negative-result / limits-of-lint paper, or workshop note)**: C1 and C3 hold but C2 or C4 does not.
* **NOT PAPER-WORTHY (as measured)**: C1 fails, or C3 fails.

Not criteria, but reported with exact confidence intervals: per-rule detection for all faults, the roll and index-shift faults (v2 showed they are mostly invisible), the 20%-contaminated variants, cross-split duplicates (F004), the number of TRUE faults found on clean data under section 7 (a discovery is a separate claim and needs the independent confirmation of section 7).

**Scope caveat that goes with any verdict.** All in-scope sets are again generic Poisson / Darcy / variable-k benchmark families (ConDiff, ELADO, PDEBench Darcy). The study tests generalisation across generators, solvers (finite volume, FEM) and files, **not across distinct application fields**; PDEBench Darcy shares a generator with tuning data; and no fresh advective control was obtainable, so the v4 advection probe has no held-out test (section 4).

Independence caveat that goes with any verdict: the three PDEBench Darcy files share one generator (and with the tuning files); the five ConDiff files share one generator and one solver; results are reported per generator family as well as pooled, and the pooled number is not a sample of "PDE datasets in general".


## 11. Post-hoc analysis

Any analysis not specified in sections 5-8 (re-mapping, other face-mean conventions, using the denormalised variant for a dataset that was run as released, re-running a dataset after a converter fix, looking at which samples fired, ablations, correlations, extra fault types, other ELADO / PDEgym / ConDiff files) is **post hoc**. It may be done after all registered runs are complete, must be labelled "POST HOC" in every table and sentence that reports it, must not alter any section-10 number, and its result files go to `bench/results/v4_heldout_posthoc/`.

## 12. Freeze and deviation log

* Frozen by: lead pre-approval 2026-10-09 + agent (Phase C), on 2026-10-09 at 23:53 +0800, at **the commit that adds this line** (its hash is in the git log; the commit message begins "fieldlint v4: FREEZE held-out protocol"). The freeze commit contains this protocol, `run_v4_heldout.py` and `report_v4.py` and precedes every held-out run.
* Conditions of the pre-approval, all met in this document: criteria C1-C4 and their thresholds unchanged; clean-adjusted scoring, section-7 adjudication and run-once rule unchanged; v4 hash check at every run; staged presets passed as configs (not moved into `src/`, no `operator:`); N written explicitly (9); native list written explicitly (6); limitations of section 4 stated.
* Deviations after freeze (run re-launched after a crash, dataset that could not be run, etc.): *none yet.*

## Appendix: references for the frozen-version statement

* Rules and thresholds: `tools/fieldlint/README.md` (v1, v2 and "Round 4 (v4) design" sections), `src/fieldlint/rules.py`; `bench/results/frozen_thresholds_v4.json`, `bench/results/bench_results_v4_tuning.md`.
* v3 protocol and results: `bench/protocol_v3.md`, `bench/results/bench_results_v3.md`.

