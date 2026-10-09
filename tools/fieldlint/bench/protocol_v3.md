# fieldlint round 3 (v3): pre-registered held-out measurement

**STATUS: DRAFT, not yet frozen.** Written 2026-10-09 by the Phase-1 (acquisition and protocol) agent. Nothing in this document has
been used to run a rule on a v3 dataset. The lead reviews it, edits it, and freezes it by committing a "FROZEN" line (section 12) before
Phase 2 starts. After that line exists, the only permitted edits are the entries in section 12 (deviation log).

The data cards (datasets, mapping, documentation, per-dataset predictions) are in `datasets_v3.md` in this directory and are part of
this pre-registration.

## 1. Question

Is fieldlint paper-worthy? Operationally: do the **frozen v2 rules** (a) detect the input faults they were built for on steady
diffusion-type datasets from fields and generators they were never tuned on, (b) stay quiet on those datasets as released, and
(c) decline, rather than raise false alarms, on data outside their stated scope?

Previous evidence is thin: v2 had one fresh held-out file whose rules mostly skipped (resampled Darcy) and one native-grid file (Li et al. Darcy 421, 100
samples) where F008 ran with a margin of 0.31 against a gate of 0.3. Round 3 widens this to several independent generators, solvers
and application fields (see `datasets_v3.md`), and adds out-of-scope controls with the README's scope declaration both set and unset.

## 2. What Phase 1 looked at (disclosure)

Phase 1 acquired the data and wrote the mapping. It did **not** run any fieldlint rule except F001 (integrity: shapes agree, values finite,
every sample loads) through the CLI on the converted files, and it did not compute any correlation, R^2, fit, residual or other statistic of
the fields. What was inspected, exhaustively:

* file names, sizes, sha256 (and md5 where the host publishes it), HDF5 / Arrow / npz key names, array shapes, dtypes, chunking, file-level
  attributes and the file-level constants that ship in the files (e.g. `min_out`, `max_out`);
* the minimum and maximum of arrays, in the following places only: PDEgym Poisson-Gauss `source` and `solution` over the first 200 samples
  (source in [0, 2.37], solution in [0, 0.0269]); DiffusionPDE-normalized poisson/darcy/helmholtz channels over the first 200 rows; every converted
  CNO file (`input`, `output`, whole converted subset, including the unregistered Euler airfoil file); ELADO `x` and `y` of the first 3 (A1) and 5 (E1) samples;
  the UCSD `Poisson4.npz` and `ConvDiff1.npz` arrays over the whole file; the Wave-Gauss subset over all 256 samples; plus whatever the data providers publish
  (`metadata.json` stats of the DiffusionPDE sets, the `min_*`/`max_*` constants inside the CNO files). These min/max values were needed to confirm normalisation and sign conventions;
  the UCSD X/Y coordinate arrays were looked at to see along which axis each varies (mapping documentation);
* a look for the `unique` values of the first 50 samples of CNO `input` and `output` to detect binary fields (Darcy `input` in {3, 12}).

Two facts learned this way shape the pre-registered predictions and are therefore stated up front: (i) the stored CNO Darcy `output` is
<= 0 everywhere (min -0.023, max 0.0) although the documented equation, -div(a grad u) = 1 with u = 0 on the boundary, has u >= 0;
(ii) the DiffusionPDE-normalized arrays are standardised (a = 3 / 12 maps to -0.5 / +0.5, README and `metadata.json`).

## 3. Frozen rule version

The rules, options and code are those of v2, unchanged:

* `bench/results/frozen_thresholds_v2.json` (frozen_at 2026-10-07T02:15:55+0800; sha256 of the file is recorded in every v3 result file).
* The hash check of `run_bench.load_frozen('_v2')` covers `rules.py`, `adapters.py`, `model.py`, `runner.py` and the five presets
  `3dice.json`, `darcy.json`, `fno_darcy.json`, `ictherm.json`, `thermfm.json`. Their sha256 at the time of writing equal the frozen ones
  (checked: `load_frozen('_v2')` passes).
* **No edit of any of those files, no change of any `Options` value, no change to the fault code in `run_bench.py` (`inject`, `FAULTS`,
  `with_splits`, `summarise`) is permitted.** `run_v3.py` imports them. If the hash check fails the run refuses to start (v2 behaviour).
* New files are outside the hash: the `v3_*.json` presets (mapping only: keys, layouts, affine undo of a documented standardisation, a
  uniform source value), `convert_v3.py` (lossless re-packaging) and `run_v3.py`. `adapters.py` was **not** edited: NetCDF, Arrow and the
  CNO per-sample HDF5 groups are converted to `.npz` in `data/_external/<name>/converted/`; ELADO's `.h5.zst` is decompressed to the `.h5` the adapter
  already reads. Each conversion writes a `*.MANIFEST.json` (sha256 of source and output, dtype, shapes).
* Precedent: `fno_darcy_native.json` was added to the v2 held-out under the same arrangement.

Code revision of the v3 tooling at freeze time: *to be filled in by the lead when freezing (git commit hash of this file).*

## 4. Datasets and roles

The roles and the reason for each are in `datasets_v3.md`; this table is the registry (the same list is in `run_v3.py`).

| role | meaning | counted in section 10? |
|---|---|---|
| **inscope** | steady diffusion-type data with a documented equation of the form -div(k grad u) = f, Dirichlet or relaxing boundary, 2D | yes |
| **diagnostic** | the same data **as released** where the release standardises the fields, so that a premise of the rules (non-negative k, absolute level of u) is violated by the documented transformation | reported, not counted |
| **control** | out of scope by documentation (time-dependent, hyperbolic, Helmholtz, nonlinear) | yes (section 10, C3) |

In-scope and diagnostic runs are the clean data plus the fault set of section 5. Control runs are clean only (no faults), each time-dependent control twice: with the README's
scope declaration (`time_dependent: true`) and without it.

The registry is closed. A dataset that failed to download or load by freeze time is listed as such in `datasets_v3.md` and is **not** replaced
after freeze; a dataset that cannot be loaded in Phase 2 stays in the table as "not run" and counts as not satisfying whichever criteria it appears in.

Datasets are subsets by sample count only: the first N samples in file order (N <= 1000; selected before looking at any statistics).

## 5. Fault set (identical to v2)

From `run_bench.FAULTS`, injected by `run_bench.inject` into the inputs (source, k) or the pairing of a clean copy; splits assigned by position (first 80% train, last 20% test) as in v2:

`transpose`, `flip_x`, `flip_y`, `roll1`, `roll2`, `roll3`, `shift_index`, `unit_slip`, `cross_dup`, and the partial-contamination variants
`transpose@20`, `flip_x@20`, `flip_y@20`, `roll1@20`, `shift_index@20` (random 20% of samples, seed 12345).

Applicability is decided by `inject` as in v2: `transpose` needs a square grid (all v3 grids are square); `unit_slip` needs `units == 'K'` (no v3 dataset is in kelvin, so it is
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

6. **Clean-adjusted detection** (added in v3; the v2 tables did not need it). v2 scoring credits a rule with a fault when the rule is flagged on the faulted copy, without checking
   whether it is already flagged on the clean copy. If a rule fires on the clean copy of a dataset (for example F005 on CNO Darcy, whose stored output has the opposite sign), it would be credited with every fault of that
   dataset. Both numbers are therefore reported: the v2 count, and the **clean-adjusted** count in which a rule is credited with a fault only if it is *not* flagged on the clean copy of the same dataset. The any-rule rate is
   computed the same way (any rule that is not already flagged on clean). **C1 below is judged on the clean-adjusted any-rule rate.** Both are computed offline from the result files by a script written before the runs
   (`bench/report_v3.py`, committed with this draft; its only thresholds are the section-10 numbers, which must be edited together with this document).

Neither the thresholds nor the metric definitions change between the first and last dataset.

## 7. Adjudication of findings on clean, as-released data

A warning or error from any rule on the clean copy of an in-scope dataset is a **TRUE fault** (and then not a false positive) **only if** one of the following holds, and
the citation is written into the results before the finding is looked at again:

* (a) **documentation**: the dataset's own documentation, paper, README, issue tracker or code states the property that the rule reports (for example that inputs and outputs are stored on different axis orders, that the output was given the opposite sign, or that two splits share samples);
* (b) **physics check stated in advance**: a check written into this protocol before the run, listed here: **(P1)** for the documented equation -div(k grad u) = f with f >= 0, k > 0 and u = 0 on the boundary, the weak maximum principle gives u >= 0, so a field that is negative in the interior contradicts the documented equation; **(P2)** for a literal duplicate (F004), bit-identical u in two samples is a duplicate regardless of cause; **(P3)** a non-finite or shape-inconsistent array (F001) is a fault. No other physics check is allowed after the run;
* (c) **maintainer statement**: a public statement by the dataset authors or maintainers (issue, erratum, e-mail quoted with permission) that the property is a known defect.

Everything else is a **false positive**, including a finding that later turns out to be "real" through an analysis done after the run (that analysis is reported, labelled post hoc, and may be written up as a discovery, but does not change the false-positive count).
Findings that depend on a standardisation documented in the dataset's own README (the **diagnostic** role) are not adjudicated; they are reported with the role label.

Pre-stated expectation of what (b) will apply to: CNO Darcy (P1) - see `datasets_v3.md`.

## 8. Exact commands and the run-once rule

All commands are run from the repository root, with the frozen environment (Python 3.10, numpy 2.2.6, h5py 3.14.0 as at Phase 1; the versions are written into each result file).

```
python tools/fieldlint/bench/run_v3.py --list                  # prints each registered run and whether its file exists; runs nothing
python tools/fieldlint/bench/run_v3.py --run-all inscope       # one result file per dataset in bench/results/v3/
python tools/fieldlint/bench/run_v3.py --run-all diagnostic
python tools/fieldlint/bench/run_v3.py --run-all control
```

or one dataset: `python tools/fieldlint/bench/run_v3.py --run <name>` (names are in `datasets_v3.md` and `--list`).

* **Run once.** `run_v3.py` writes `bench/results/v3/<name>.json` and **refuses** to run a name whose file exists. A crash before the file is written may be re-launched; the re-launch
  and its cause are logged in section 12. A result is never re-run because of its content, a different conclusion, a bug found in the converter or the preset, or a changed mind about the
  mapping. If a converter or preset error is found after a run, the run stands as is, the error is reported, and a corrected run is a separate, labelled post-hoc analysis (section 11).
* Each result file records: git HEAD of the repository, uncommitted tracked changes, the sha256 of the frozen file, the current code hashes, library versions, argv, start time.
* The order of runs does not matter. Results are not inspected between runs except for crash detection.
* Conversions (`convert_v3.py`) happened in Phase 1 and are part of the registry; their manifests are in the `converted/` folders.
* No threshold, rule, mapping, role or criterion may change after the first v3 result exists.

## 9. Pre-registered predictions

Per-dataset predictions (rules expected to apply or skip, with the documentation they follow from) are in `datasets_v3.md`. They are **predictions, not criteria**: a wrong prediction does not fail the
protocol, but it is reported as a wrong prediction. The summary the lead should know:

* F008 is expected to be applicable (clean median R^2 >= 0.3) on the native-grid Poisson-type sets (PDEgym Poisson-Gauss, ELADO Poisson, CNO Poisson) and expected to be skipped, or marginal, on the sets generated through the FNO Darcy scripts (CNO Darcy, DiffusionPDE Darcy), which interpolate between cell-centred and nodal grids; F002 and F003 skip where the source is uniform or sign-changing.
* On sets with a uniform source, only F008 can detect a transposed / flipped input, so a skip of F008 there is a miss of the any-rule detection. The expected any-rule detection is therefore below 100% on the Darcy-type sets and near 100% where F003 or F008 apply.
* Two predicted problems are stated in advance: (1) the DiffusionPDE Poisson convention `Laplace(u) = a` gives a negative correlation between source and u, which can make F003 prefer a wrong orientation (a false positive of the type seen on Therm-FM HS_SC); (2) F005 is expected to flag the sign of CNO Darcy's output (a documented-equation violation, adjudicated TRUE by P1) and the negative `k` of the as-released standardised DiffusionPDE Darcy (diagnostic role).

## 10. Criteria for a "paper-worthy" verdict (draft numbers for the lead to approve or change)

Let N be the number of **inscope** datasets that loaded and ran, and let "applicable trial" be an (in-scope dataset, whole-dataset fault) pair for which `inject` produced a faulted copy.

| id | criterion | draft threshold | why this number |
|---|---|---|---|
| **C1** | clean-adjusted any-rule detection (section 6, item 6) of whole-dataset `transpose`, `flip_x`, `flip_y` pooled over the in-scope applicable trials | **>= 80%** (point estimate), and the exact one-sided 95% Clopper-Pearson lower bound is reported | the v2 tuning sets gave 17/17, 20/20 and 18/20 (92-100%), the tuning rates are optimistic, and one native-grid held-out gave 3/3; 80% asks for most of that to survive on unseen generators while allowing the Darcy-type sets to be missed. With N = 7 registered in-scope sets and 3 faults (21 trials) the exact one-sided 95% lower bound is 0.62 at 17/21 and 0.87 at 21/21 (computed with `report_v3.clopper_lower`), a weak result that the paper would have to state |
| **C2** | clean false positives on in-scope datasets after adjudication | **at most 1 per 10 datasets**: the count of false-positive datasets <= floor(N/10), i.e. **0 for N < 10** | v2 had 1/20 on tuning (true duplicates, would be a TRUE finding under section 7) and 0/1 held-out; a tool whose output is read as a warning must be quiet on clean data. The lead may prefer "<= 1 in total", which is looser for N < 20 |
| **C3** | no false alarm on any control run **with the README's scope declaration** (time-dependent controls with `time_dependent: true`); the unflagged runs are reported but not binding | **0** false alarms among F002, F003, F005, F008, F009 | v2 controls: Burgers 1D gave false alarms in v1 and none in v2; diffusion-reaction none. The README tells users to set the flag. A stricter reading (also the unflagged runs) is stated as C3-strict and reported; if the lead prefers it as binding, change it before freezing |
| **C4** | F008 applicable (clean `status != skipped`) on in-scope datasets classed **native grid** in `datasets_v3.md` (5 registered: PDEgym Poisson-Gauss, CNO Poisson, ELADO A1, UCSD Poisson, DiffusionPDE Poisson denormalised) | **>= 50%** of them, i.e. at least 3 of 5 | F008 is the only rule that sees k-only datasets; v2 had it applicable on 1/1 native and 0/1 processed grids |

Verdicts, decided mechanically from the results and not by the author:

* **PAPER-WORTHY (as a tool paper with a measured scope)**: C1, C2, C3 and C4 all hold.
* **PARTIAL (negative-result / limits-of-lint paper, or workshop note)**: C1 and C3 hold but C2 or C4 does not.
* **NOT PAPER-WORTHY (as measured)**: C1 fails, or C3 fails.

Not criteria, but reported with exact confidence intervals: per-rule detection for all faults, the roll and index-shift faults (v2 showed they are mostly invisible), the 20%-contaminated variants, cross-split duplicates (F004), the number of TRUE faults found on clean data under section 7 (a discovery is a separate claim and needs the independent confirmation of section 7).

Independence caveat that goes with any verdict: the Darcy-type sets (CNO, DiffusionPDE, and the earlier PDEBench / FNO sets of v2) share the FNO generating scripts; the Poisson-type sets
share the `-Laplace u = f` operator. Results are reported per generator family as well as pooled; the pooled number is not a sample of "PDE datasets in general".

## 11. Post-hoc analysis

Any analysis not specified in sections 5-8 (re-mapping, other face-mean conventions, using the denormalised variant for a dataset that was run as released, re-running a dataset after a converter fix, looking at which samples fired, ablations, correlations, extra fault types, other ELADO/PDEgym files) is **post hoc**. It may be done after all registered runs are complete, must be labelled "POST HOC" in every table and sentence that reports it, must not alter any section-10 number, and its result files go to `bench/results/v3_posthoc/`.

## 12. Freeze and deviation log

* Frozen by: *(lead)*  on: *(date)*  at commit: *(hash)*  - **NOT YET FROZEN.**
* Deviations after freeze (run re-launched after a crash, dataset that could not be run, etc.): *none yet.*

## Appendix: references for the frozen-version statement

* Rules and thresholds: `tools/fieldlint/README.md` (v1, v2 sections), `src/fieldlint/rules.py`.
* v2 results: `bench/results/bench_results_v2.md`, `tuning_v2.json`, `heldout_v2.json`, `heldout_native_v2.json`, `control_v2.json`.
