# fieldlint v4 held-out results (round 4)

Protocol: `bench/protocol_v4.md` (frozen in commit `187662b`, before any run). Data cards and predictions: `bench/datasets_v4.md`. Runner `bench/run_v4_heldout.py`, result files `bench/results/v4_heldout/*.json` (13 files, 14 recorded runs, each run once, no crash, no re-launch; every file records git HEAD `187662b36dda79aca80389513026aad7219db4f4`, the frozen-file sha256 `df7851af...a4a0` and the hash check of `frozen_thresholds_v4.json` passed at each start). The tables from "Mechanical report" down are the unedited output of `bench/report_v4.py`.

## Verdict (mechanical, protocol section 10)

| criterion | result | verdict |
|---|---|---|
| **C1** clean-adjusted any-rule detection of whole-dataset transpose / flip_x / flip_y, N = 9 in-scope sets | **27/27 = 1.00**, exact two-sided 95% CI [0.87, 1.00], one-sided lower bound 0.89 (threshold 0.80) | **PASS** |
| **C2** clean false positives after adjudication (allowed 0 for N = 9) | **0**: no rule fired (warning or error) on any of the 9 in-scope clean copies, so there is nothing to adjudicate | **PASS** |
| **C3** false alarms on binding controls | **2 of 3 binding control runs have a false alarm, both F005 (warning)**: `pdegym_helmholtz` and `pdegym_wave_layer` with `time_dependent: true`. `elado_helmholtz_c1`: no false alarm (F001, F004 and F005 ran and were ok; F002, F003, F008 and F009 were skipped). | **FAIL** |
| **C4** F008 applicable on native in-scope sets (6 binding) | **6/6** (9/9 if PDEBench Darcy is also counted native) | **PASS** |

**Verdict by the section-10 rule: NOT PAPER-WORTHY (as measured)**, because C3 fails (the rule is "C1 fails, or C3 fails"). C1, C2 and C4 pass. The failure is a single rule (F005) on two controls, whose cause is shown below; nothing here was changed or re-run after seeing it.

## What failed, and why (reported, not repaired)

* **F005 fires on PDEgym Helmholtz** (`u in [-4.706, 4.593]`, the field spans both signs; every one of the 1000 converted samples has some interior `u` < 0, and the rule checked 400 of them): the finding text is "400 sample(s) have u below the ambient level although the source is non-negative (a passive problem cannot cool below ambient)". The preset maps a user's plausible reading, `source_uniform = 1.0` (there is no source in the data; the forcing is the nonzero Dirichlet value b in [0.25, 0.5]), so F005's premise (non-negative source, relaxation to ambient) is satisfied by the mapping alone, and Helmholtz `u` oscillates in sign.
* **F005 fires on Wave-Layer with `time_dependent: true`** (`u(T)` in [-0.311, 0.394], source = `u(t0) >= 0`): the time flag skips F002, F003, F008, F009 but **not F005**, so the wave field's negative values are flagged against ambient 0.
* Both findings are exactly the "false alarm" of protocol C3 (a warning from F002/F003/F005/F008/F009 on a binding control). The post-hoc finding texts above were obtained with `fieldlint --rules F005` on the same files and are labelled POST HOC; they change no number.
* **POST HOC note on PDEgym Helmholtz (descriptive; changes no number, criterion or verdict).** A later verification pass found that the stored data do not match the dataset card's stated equation, -Lap(u) - (5/2)^2 a u = 0: the 5-point residual on 50 samples has a median relative value of about 0.95. Also, `a` lies in [1, 2], not the card's [0, 1]. A fit of -Lap(u) = kappa*a*u gives kappa about 119 (IQR 116-120) with a residual of about 0.08; this is a rough 50-sample estimate and was not checked against the Poseidon code. The pre-registered prediction (u > 0) relied on the card's coefficient, so the F005 warning may reflect a card/data inconsistency rather than only a rule weakness. The preset also supplied an invented uniform source = 1, which the data card disclosed ("a reading, not a documented source"). Under protocol v4, section 7, this does NOT change the C3 count or the verdict; it is reported only.
* **Disclosure on the pre-freeze smoke test.** An F001-only integrity smoke test ran on all 13 loads before the freeze, as disclosed in `datasets_v4.md` section 5 and the protocol. No other rule was run. See the protocol section 12 deviation log.

## Per-run adjudication (protocol section 7)

* **In-scope and diagnostic clean copies: no warning or error from any rule** on any of the 10 sets (info notes only: F005 reports "negative source values (sinks)" as info on the five ConDiff sets, because the sources are N(0,1)). There are therefore **no clean findings to adjudicate**, no TRUE-fault discoveries (benign: 0, material: 0) and no false positives. F004 found no duplicate in any clean set.
* Controls: the two F005 alarms above are false alarms by the C3 definition (the data are out of scope by documentation; section 7's adjudication routes apply to in-scope data only).

## Predictions audit (from `datasets_v4.md`; predictions, not criteria)

Right:
* ConDiff Poisson and Gaussian-0.1: F008 applicable at median R^2 1.000 / 1.000 (predicted near 1); F002 skipped (sign-changing source); F003 applicable and ok; F004 ok; F005 ok.
* ELADO B1: F008 applicable at 0.99999 (arithmetic face mean); F002/F003 skipped; F005 ok.
* PDEBench Darcy beta 0.1 / 10 / 100: F008 applicable at 0.99994 / 0.9999998 / 0.9999998 (arithmetic face mean, as in v2); F002/F003 skipped; F005 ok; detection of transposed / flipped k by F008.
* ELADO C1 Helmholtz: F008/F009 skipped (R^2 0.12), F005 ok (u >= 0), F002/F003 skipped.
* PDEgym Helmholtz: F002/F003 skipped, F008/F009 skipped (R^2 0.04).
* Wave-Layer flagged: F002, F003, F008, F009 all skipped; unflagged: F002 (error) and F003 (warning) false alarms, as the card allowed; F008 skipped.
* Neither the F003->F008 deferral nor the advection probe triggered on any run, clean or faulted (the report counts summaries containing "confirms the stored orientation" or "out of scope: advective": 0). No in-scope set was declined as advective, as predicted. Because the registry has no advective data, this round does **not** test whether the probe fires when it should.

Wrong or uncertain-resolved:
* **PDEgym Helmholtz F005: predicted ok, was flagged** (I predicted u > 0 from a positive-definiteness argument; the stored u spans [-4.7, 4.6], so the prediction was wrong, and the data card's claim that "omega^2 a <= 6.25 is below the smallest Dirichlet eigenvalue so u > 0" does not describe the stored field).
* **Wave-Layer F005 with the time flag**: not anticipated by the card (the card predicted a quiet flagged run); it fired.
* **ConDiff `nok` diagnostic: predicted F008 skipped or poor, was applicable** (median R^2 0.91; face mean harmonic): for Gaussian-0.1 (contrast <= 15) a uniform-k operator already explains the field, so omitting k loses little. Whole-dataset transposed / flipped source were detected by F003 and F008 (as on the primary), rolls by F008.
* ConDiff Gaussian-1.0, cubic-0.4, exponential-2.0 (marked uncertain, decline possible): F008 was applicable on all (0.9996, 0.9926, 0.884).
* F003 on ConDiff was predicted weak: clean corr(smoothed source, u) = 0.40-0.41 and it detected transpose / flips on 5/5 (as F003 is credited on the five ConDiff sets in the per-dataset table); it did not detect them on ELADO and Darcy (skipped: uniform source), as predicted.
* F008 face mean: "auto" chose **harmonic** on all five ConDiff sets (the solver's node-mean coefficients are not an arithmetic mean of `k_cell`) and **arithmetic** on ELADO and PDEBench Darcy; this was not specified in the cards.

## Notes the reader should have

* The only fault that is mostly missed is `shift_index` (1/9 any-rule detections, F003 on one set; F008 is skipped there at R^2 ~ 0), which is not part of C1. `roll1` is missed on one set (8/9). Cross-split duplicates (`cross_dup`, `shift_index@20`) are detected by F004 on 9/9.
* The ConDiff `k_cell` mapping is the derived array of `datasets_v4.md` (the solver's own cell coefficient), not lossless.
* PDEBench Darcy shares its generator with the tuning data; the three C1 / C4 successes there test file-to-file stability. The per-family C1 is 15/15 (ConDiff), 3/3 (ELADO) and 9/9 (PDEBench Darcy).
* Limitations of protocol section 4 stand: generic Poisson / Darcy families, no application field, no advective control.

## POST HOC candidates (listed only; none run, none counted)

* Whether F005 should be gated by the `time_dependent` declaration, or by a non-uniform / documented source, so that steady Helmholtz and wave data decline (the v4 design left F005 unconditional).
* Whether a mapping with no source term should leave `source` unset rather than uniform 1 for Helmholtz-type controls (F005's premise then does not hold).
* Why the harmonic face mean was selected on ConDiff by `auto` and whether the arithmetic face mean on `k_cell` changes the exponential-2.0 R^2 (0.88).
* An advective fresh control (ConvDiff-type) to test the advection probe held-out.

## Mechanical report (`bench/report_v4.py`, unedited)

### (mechanical tables)

N (in-scope datasets that loaded and ran) = 9; allowed false-positive datasets after adjudication (C2) = 0

## In-scope and diagnostic: clean findings (to be adjudicated by hand, protocol section 7)

| dataset | role | rules flagged on clean | skipped on clean |
|---|---|---|---|
| condiff_cubic0.4 | inscope | none | F002 |
| condiff_exponential2.0 | inscope | none | F002 |
| condiff_gaussian0.1 | inscope | none | F002 |
| condiff_gaussian1.0 | inscope | none | F002 |
| condiff_poisson | inscope | none | F002 |
| elado_b1_test | inscope | none | F002, F003 |
| pdebench_darcy_beta0.1 | inscope | none | F002, F003 |
| pdebench_darcy_beta10.0 | inscope | none | F002, F003 |
| pdebench_darcy_beta100.0 | inscope | none | F002, F003 |
| condiff_gaussian0.1_nok | diagnostic | none | F002 |

in-scope datasets with any clean finding before adjudication: 0 of 9 (none)

## Clean summary strings (every rule, every in-scope / diagnostic dataset)

| dataset | F001 | F002 | F003 | F004 | F005 | F008 | F009 |
|---|---|---|---|---|---|---|---|
| condiff_cubic0.4 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_exponential2.0 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_gaussian0.1 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_gaussian1.0 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_poisson | ok | skipped | ok | ok | ok (info) | ok | ok |
| elado_b1_test | ok | skipped | skipped | ok | ok | ok | ok |
| pdebench_darcy_beta0.1 | ok | skipped | skipped | ok | ok | ok | ok |
| pdebench_darcy_beta10.0 | ok | skipped | skipped | ok | ok | ok | ok |
| pdebench_darcy_beta100.0 | ok | skipped | skipped | ok | ok | ok | ok |
| condiff_gaussian0.1_nok | ok | skipped | ok | ok | ok (info) | ok | ok |

## C1: whole-dataset fault detection, any rule (in-scope only)

| fault | v2-count (plain) | clean-adjusted | trials |
|---|---|---|---|
| transpose | 9 | 9 | 9 |
| flip_x | 9 | 9 | 9 |
| flip_y | 9 | 9 | 9 |

C1: clean-adjusted any-rule detection 27/27 = 1.00 [0.87, 1.00] (two-sided exact 95%); one-sided 95% lower bound 0.89; threshold 0.8: **PASS**

### Per-family C1 (clean-adjusted, same three faults)

| family | hits/trials (exact 95% CI) |
|---|---|
| ConDiff (FV, random k) | 15/15 = 1.00 [0.78, 1.00] |
| ELADO B1 (FEM, random a) | 3/3 = 1.00 [0.29, 1.00] |
| PDEBench Darcy (same generator as tuning) | 9/9 = 1.00 [0.66, 1.00] |

### Per dataset: whole-dataset faults, rules credited (clean-adjusted)

| dataset | transpose | flip_x | flip_y |
|---|---|---|---|
| condiff_cubic0.4 | F003, F008 | F003, F008 | F003, F008 |
| condiff_exponential2.0 | F003, F008 | F003, F008 | F003, F008 |
| condiff_gaussian0.1 | F003, F008 | F003, F008 | F003, F008 |
| condiff_gaussian1.0 | F003, F008 | F003, F008 | F003, F008 |
| condiff_poisson | F003, F008 | F003, F008 | F003, F008 |
| elado_b1_test | F008 | F008 | F008 |
| pdebench_darcy_beta0.1 | F008 | F008 | F008 |
| pdebench_darcy_beta10.0 | F008 | F008 | F008 |
| pdebench_darcy_beta100.0 | F008 | F008 | F008 |

## Per-fault, per-rule detection (in-scope pooled; plain = flagged on faulted copy; adj = also not flagged on clean)

| fault | trials | any (plain) | any (adj) | F001 adj | F002 adj | F003 adj | F004 adj | F005 adj | F008 adj | F009 adj |
|---|---|---|---|---|---|---|---|---|---|---|
| transpose | 9 | 9 | 9 | 0 | 0 | 5 | 0 | 0 | 9 | 0 |
| flip_x | 9 | 9 | 9 | 0 | 0 | 5 | 0 | 0 | 9 | 0 |
| flip_y | 9 | 9 | 9 | 0 | 0 | 5 | 0 | 0 | 9 | 0 |
| roll1 | 9 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 8 | 0 |
| roll2 | 9 | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| roll3 | 9 | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| shift_index | 9 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| unit_slip | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_dup | 9 | 9 | 9 | 0 | 0 | 0 | 9 | 0 | 0 | 0 |
| transpose@20 | 9 | 9 | 9 | 0 | 0 | 5 | 0 | 0 | 9 | 0 |
| flip_x@20 | 9 | 9 | 9 | 0 | 0 | 5 | 0 | 0 | 9 | 0 |
| flip_y@20 | 9 | 9 | 9 | 0 | 0 | 5 | 0 | 0 | 9 | 0 |
| roll1@20 | 9 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 8 | 0 |
| shift_index@20 | 9 | 9 | 9 | 0 | 0 | 0 | 9 | 0 | 0 | 0 |

## F008 applicability and C4

| dataset | native (binding list) | F008 status | median R2 stored | face mean | F009 status |
|---|---|---|---|---|---|
| condiff_cubic0.4 | yes | ok | 0.9926371330968725 | harmonic | ok |
| condiff_exponential2.0 | yes | ok | 0.88398939958508 | harmonic | ok |
| condiff_gaussian0.1 | yes | ok | 0.9999638701151303 | harmonic | ok |
| condiff_gaussian1.0 | yes | ok | 0.9996335435506367 | harmonic | ok |
| condiff_poisson | yes | ok | 1.0 | harmonic | ok |
| elado_b1_test | yes | ok | 0.9999950239666042 | arithmetic | ok |
| pdebench_darcy_beta0.1 | no | ok | 0.9999399155334456 | arithmetic | ok |
| pdebench_darcy_beta10.0 | no | ok | 0.9999997885086158 | arithmetic | ok |
| pdebench_darcy_beta100.0 | no | ok | 0.9999998413682963 | arithmetic | ok |

C4: F008 applicable on 6/6 native in-scope sets; threshold 0.5: **PASS**
(sensitivity, not binding: 9/9 if PDEBench Darcy is also counted native)

## Controls (declined vs false alarm)

| run | binding | F001 | F002 | F003 | F004 | F005 | F008 | F009 |
|---|---|---|---|---|---|---|---|---|
| elado_helmholtz_c1 | yes | ok | skipped | skipped | ok | ok | skipped | skipped |
| pdegym_helmholtz | yes | ok | skipped | skipped | ok | flagged (warning) | skipped | skipped |
| pdegym_wave_layer+time_dependent_flag | yes | ok | skipped | skipped | ok | flagged (warning) | skipped | skipped |
| pdegym_wave_layer | C3-strict only | ok | flagged (error) | flagged (warning) | ok | flagged (warning) | skipped | skipped |

C3 (binding runs = flagged for time-dependent controls, as-is for steady ones): false alarms [('pdegym_helmholtz', ['F005']), ('pdegym_wave_layer+time_dependent_flag', ['F005'])]: **FAIL**
C3-strict, not binding (unflagged time-dependent runs): false alarms [('pdegym_wave_layer', ['F002', 'F003', 'F005'])]

## Control summary strings (clean)

* elado_helmholtz_c1: F001 400 samples checked; no problems; F002 nothing to check (400 vacuous / no-heating, 0 with sinks); F003 only 0 usable sample(s); F005 u in [2.487e-23, 0.09153] (units unspecified); 400 samples; no problems; F008 not applicable: the operator model explains only median R2 = 0.12 in any orientation (< 0.3); the field is probably time-dependent, advective or strongly coupled between layers; F009 not applicable: the best match of each row has median R2 0.28 (< 0.3); the operator model does not describe this field; F004 1000 samples compared: 0 cross-split, 0 within-split duplicate(s)
* pdegym_helmholtz: F001 400 samples checked; no problems; F002 nothing to check (400 vacuous / no-heating, 0 with sinks); F003 only 0 usable sample(s); F005 u in [-4.706, 4.593] (units unspecified); 400 samples; 1 note(s); F008 not applicable: the operator model explains only median R2 = 0.04 in any orientation (< 0.3); the field is probably time-dependent, advective or strongly coupled between layers; F009 not applicable: the best match of each row has median R2 0.05 (< 0.3); the operator model does not describe this field; F004 1000 samples compared: 0 cross-split, 0 within-split duplicate(s)
* pdegym_wave_layer+time_dependent_flag: F001 256 samples checked; no problems; F002 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F003 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F005 u in [-0.3106, 0.3935] (units unspecified); 256 samples; 1 note(s); F008 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F009 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F004 256 samples compared: 0 cross-split, 0 within-split duplicate(s)
* pdegym_wave_layer: F001 256 samples checked; no problems; F002 161/256 samples (62.9%) have their maximum away from every source; F003 corr(smoothed source, u): stored -0.07, transpose -0.04, flip_x -0.03, flip_y -0.02, flip_xy -0.03; alternative better in 48% of 256 samples; F005 u in [-0.3106, 0.3935] (units unspecified); 256 samples; 1 note(s); F008 not applicable: the operator model explains only median R2 = 0.02 in any orientation (< 0.3); the field is probably time-dependent, advective or strongly coupled between layers; F009 not applicable: the best match of each row has median R2 0.13 (< 0.3); the operator model does not describe this field; F004 256 samples compared: 0 cross-split, 0 within-split duplicate(s)

## F003 deferral and advection probe: summaries that report them (clean runs and every faulted copy)

number of summaries reporting the deferral or the advection probe: 0

C2 is judged by hand after adjudication (count the in-scope datasets above that remain false positives).

Verdict rule (protocol section 10): all of C1-C4 -> PAPER-WORTHY; C1 and C3 but not C2/C4 -> PARTIAL; C1 or C3 fail -> NOT PAPER-WORTHY.
