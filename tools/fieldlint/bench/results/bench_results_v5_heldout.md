# v5 held-out results (mechanical)

N (binding in-scope datasets that loaded and ran) = 6; allowed false-positive datasets after adjudication (C2) = 0

## Clean findings and skips (every in-scope, non-binding in-scope and diagnostic dataset)

| dataset | role | rules flagged on clean | skipped on clean |
|---|---|---|---|
| condiff_cubic0.1 | inscope | none | F002 |
| condiff_cubic1.0 | inscope | none | F002 |
| condiff_cubic2.0 | inscope | none | F002 |
| condiff_exponential0.4 | inscope | none | F002 |
| condiff_exponential1.0 | inscope | none | F002 |
| condiff_gaussian0.4 | inscope | none | F002 |
| chip_fem_a | inscope_nb | none | none |
| chip_fem_b | inscope_nb | none | none |
| sissa_thermal | diagnostic | none | F002, F003, F008, F009 |

binding in-scope datasets with any clean finding before adjudication: 0 of 6 (none)

## Clean status per rule

| dataset | F001 | F002 | F003 | F004 | F005 | F008 | F009 |
|---|---|---|---|---|---|---|---|
| condiff_cubic0.1 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_cubic1.0 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_cubic2.0 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_exponential0.4 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_exponential1.0 | ok | skipped | ok | ok | ok (info) | ok | ok |
| condiff_gaussian0.4 | ok | skipped | ok | ok | ok (info) | ok | ok |
| chip_fem_a | ok | ok | ok | ok | ok | ok | ok |
| chip_fem_b | ok | ok | ok | ok | ok | ok | ok |
| sissa_thermal | ok | skipped | skipped | ok | ok | skipped | skipped |

## C1: whole-dataset fault detection, any rule (binding in-scope)

| fault | v2-count (plain) | clean-adjusted | trials |
|---|---|---|---|
| transpose | 6 | 6 | 6 |
| flip_x | 6 | 6 | 6 |
| flip_y | 6 | 6 | 6 |

C1: clean-adjusted any-rule detection 18/18 = 1.00 [0.81, 1.00] (two-sided exact 95%); one-sided 95% lower bound 0.85; threshold 0.8: **PASS**
(dataset-cluster view, optimistic pooled trials aside: 6 of 6 datasets had all three faults detected; one-sided lower bound 0.61)

### C1 on non-binding in-scope and diagnostic sets (reported only)

| fault | v2-count (plain) | clean-adjusted | trials |
|---|---|---|---|
| transpose | 2 | 2 | 3 |
| flip_x | 2 | 2 | 3 |
| flip_y | 2 | 2 | 3 |

6/9 = 0.67 [0.30, 0.93]

### Per dataset: whole-dataset faults, rules credited (clean-adjusted)

| dataset | role | transpose | flip_x | flip_y |
|---|---|---|---|---|
| condiff_cubic0.1 | inscope | F003, F008 | F003, F008 | F003, F008 |
| condiff_cubic1.0 | inscope | F003, F008 | F003, F008 | F003, F008 |
| condiff_cubic2.0 | inscope | F003, F008 | F003, F008 | F003, F008 |
| condiff_exponential0.4 | inscope | F003, F008 | F003, F008 | F003, F008 |
| condiff_exponential1.0 | inscope | F003, F008 | F003, F008 | F003, F008 |
| condiff_gaussian0.4 | inscope | F003, F008 | F003, F008 | F003, F008 |
| chip_fem_a | inscope_nb | F002, F003, F008 | F002, F003, F008 | F002, F003, F008 |
| chip_fem_b | inscope_nb | F002, F003, F008 | F002, F003, F008 | F002, F003, F008 |
| sissa_thermal | diagnostic | MISS | MISS | MISS |

## Per-fault, per-rule detection (binding in-scope pooled)

| fault | trials | any (plain) | any (adj) | F001 adj | F002 adj | F003 adj | F004 adj | F005 adj | F008 adj | F009 adj |
|---|---|---|---|---|---|---|---|---|---|---|
| transpose | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 6 | 0 |
| flip_x | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 6 | 0 |
| flip_y | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 6 | 0 |
| roll1 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| roll2 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| roll3 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| shift_index | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unit_slip | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_dup | 6 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| transpose@20 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 6 | 0 |
| flip_x@20 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 6 | 0 |
| flip_y@20 | 6 | 6 | 6 | 0 | 0 | 6 | 0 | 0 | 6 | 0 |
| roll1@20 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| shift_index@20 | 6 | 6 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |

## F008 applicability and C4

| dataset | native binding list | F008 status | median R2 stored | face mean | F009 status |
|---|---|---|---|---|---|
| condiff_cubic0.1 | yes | ok | 0.9981865839004302 | harmonic | ok |
| condiff_cubic1.0 | yes | ok | 0.9804443284004363 | harmonic | ok |
| condiff_cubic2.0 | yes | ok | 0.956839296678287 | harmonic | ok |
| condiff_exponential0.4 | yes | ok | 0.981071614741668 | harmonic | ok |
| condiff_exponential1.0 | yes | ok | 0.9480402564165964 | harmonic | ok |
| condiff_gaussian0.4 | yes | ok | 0.999855652521795 | harmonic | ok |
| chip_fem_a | no | ok | 0.9886413966929442 | arithmetic | ok |
| chip_fem_b | no | ok | 0.9989030938434165 | harmonic | ok |
| sissa_thermal | no | skipped |  |  | skipped |

C4: F008 applicable on 6/6 native binding in-scope sets; threshold 0.5: **PASS**

## Controls (declined vs false alarm)

| run | role | binding | F001 | F002 | F003 | F004 | F005 | F008 | F009 |
|---|---|---|---|---|---|---|---|---|---|
| acoustic_inclusions+time_dependent_flag | control | yes | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| acoustic_inclusions | control | C3-strict only | ok | flagged (error) | flagged (warning) | ok | ok (info) | skipped | skipped |
| advdiff | control_nb | non-binding | ok | skipped | skipped | ok | ok | skipped | skipped |
| flowbench_ldcht | control | yes | ok | ok | ok | ok | flagged (warning) | skipped | skipped |
| helmholtz_staircase | control | yes | ok | skipped | skipped | ok (info) | ok (info) | skipped | skipped |
| rayleigh_benard+time_dependent_flag | control | yes | ok | skipped | skipped | ok (info) | ok (info) | skipped | skipped |
| rayleigh_benard | control | C3-strict only | ok | skipped | ok | ok (info) | flagged (warning) | ok | flagged (error) |

C3 (binding runs = flagged for time-dependent controls, as-is for steady ones; 4 runs): false alarms [('flowbench_ldcht', ['F005'])]: **FAIL**
false-alarm rate 1/4, one-sided 95% upper bound 0.75 (a statement about these files, not a rate)
C3-strict, not binding (unflagged time-dependent runs): false alarms [('acoustic_inclusions', ['F002', 'F003']), ('rayleigh_benard', ['F005', 'F009'])]
non-binding generated control: false alarms none

### C3 graded view (descriptive only; defined in protocol_v5.md section 10 before any run; does not change the verdict)

For every binding run: the rules with a warning or error, and their severity. Graded reading = PARTIAL-LIKE if every alarm is a single rule across the binding runs, else FAIL-LIKE.

* flowbench_ldcht: F005 warning
graded reading: single rule (F005)

### Control summary strings (clean)

* acoustic_inclusions+time_dependent_flag: F001 200 samples checked; no problems; F002 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F003 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F005 u in [-4.898, 3.47] (units unspecified); 200 samples; 2 note(s); F008 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F009 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F004 200 samples compared: 0 cross-split, 0 within-split duplicate(s)
* acoustic_inclusions: F001 200 samples checked; no problems; F002 100/101 samples (99.0%) have their maximum away from every source; F003 corr(smoothed source, u): stored -0.04, transpose 0.00, flip_x 0.04, flip_y 0.04, flip_xy 0.07; alternative better in 25% of 200 samples; F005 u in [-4.898, 3.47] (units unspecified); 200 samples; 2 note(s); F008 not applicable: the operator model explains only median R2 = 0.01 in any orientation (< 0.3); the field is probably time-dependent, advective or strongly coupled between layers; F009 not applicable: the best match of each row has median R2 0.07 (< 0.3); the operator model does not describe this field; F004 200 samples compared: 0 cross-split, 0 within-split duplicate(s)
* advdiff: F001 300 samples checked; no problems; F002 out of scope: advective (first-derivative terms explain the residual: median R2 0.18 -> 1.00 in the stored orientation with a uniform drift, against 0.19 for the best re-oriented diffusion fit); F003 out of scope: advective (first-derivative terms explain the residual: median R2 0.18 -> 1.00 in the stored orientation with a uniform drift, against 0.19 for the best re-oriented diffusion fit); F005 u in [3.523e-32, 0.4716] (units unspecified); 300 samples; no problems; F008 out of scope: advective (first-derivative terms explain the residual: median R2 0.18 -> 1.00 in the stored orientation with a uniform drift, against 0.19 for the best re-oriented diffusion fit); F009 out of scope: advective (first-derivative terms explain the residual: median R2 0.18 -> 1.00 in the stored orientation with a uniform drift, against 0.19 for the best re-oriented diffusion fit); F004 300 samples compared: 0 cross-split, 0 within-split duplicate(s)
* flowbench_ldcht: F001 400 samples checked; no problems; F002 0/400 samples (0.0%) have their maximum away from every source; F003 corr(smoothed source, u): stored 0.33, transpose 0.32, flip_x 0.32, flip_y 0.32, flip_xy 0.32; alternative better in 0% of 400 samples; F005 u in [-0.002814, 0.9956] (units unspecified); 400 samples; 1 note(s); F008 not applicable: the operator model explains only median R2 = 0.12 in any orientation (< 0.3); the field is probably time-dependent, advective or strongly coupled between layers; F009 not applicable: the best match of each row has median R2 0.14 (< 0.3); the operator model does not describe this field; F004 1000 samples compared: 0 cross-split, 0 within-split duplicate(s)
* helmholtz_staircase: F001 208 samples checked; no problems; F002 nothing to check (208 vacuous / no-heating, 0 with sinks); F003 only 0 usable sample(s); F005 u in [-1.878, 1.913] (units unspecified); 208 samples; 1 note(s); F008 no sample has a varying source or k (nothing to fit); F009 only 0 usable sample(s) (needs 4); F004 208 samples compared: 0 cross-split, 0 within-split duplicate(s)
* rayleigh_benard+time_dependent_flag: F001 256 samples checked; no problems; F002 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F003 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F005 u in [-0.3528, 1.008] (units unspecified); 256 samples; 1 note(s); F008 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F009 out of scope: the dataset is flagged time-dependent (snapshots of a trajectory), not a steady solution; F004 256 samples compared: 0 cross-split, 0 within-split duplicate(s)
* rayleigh_benard: F001 256 samples checked; no problems; F002 nothing to check (256 vacuous / no-heating, 0 with sinks); F003 corr(smoothed source, u): stored 0.58, flip_x -0.58, flip_y 0.58, flip_xy -0.58; alternative better in 0% of 256 samples; F005 u in [-0.3528, 1.008] (units unspecified); 256 samples; 1 note(s); F008 face mean harmonic; median R2 of s ~ aL(u)+b(u-c): stored 0.35, flip_x 0.35, flip_y 0.35, flip_xy 0.35; alternative better in 0% of 256 samples; F009 26/32 rows (81%) match another sample's u better than their own; F004 256 samples compared: 0 cross-split, 0 within-split duplicate(s)

## C5: F004 duplicate detection (binding in-scope)

detected = F004 flagged on the faulted copy AND (cross-split + within-split duplicate samples) exceeds the clean copy's count.

| dataset | dup_x1 | dup_x5 | cross_dup | dup_within5 | dupnoise_1e-7 | dupnoise_1e-4 | dupnoise_1e-2 |
|---|---|---|---|---|---|---|---|
| condiff_cubic0.1 | hit | hit | hit | hit | hit | MISS | MISS |
| condiff_cubic1.0 | hit | hit | hit | hit | hit | MISS | MISS |
| condiff_cubic2.0 | hit | hit | hit | hit | hit | MISS | MISS |
| condiff_exponential0.4 | hit | hit | hit | hit | hit | MISS | MISS |
| condiff_exponential1.0 | hit | hit | hit | hit | hit | MISS | MISS |
| condiff_gaussian0.4 | hit | hit | hit | hit | hit | MISS | MISS |

C5a exact copies (dup_x1, dup_x5, dup_x10 = cross_dup, dup_within5): 24/24 = 1.00 [0.86, 1.00]; one-sided lower bound 0.88; threshold 0.95: **PASS**
C5b dupnoise_1e-7: 6/6 = 1.00 [0.54, 1.00]; threshold 0.9: **PASS**
C5c clean copies with an F004 finding (to be adjudicated by hand, P2): none  [pass = 0 after adjudication]
per fault: dup_x1 6/6 = 1.00 [0.54, 1.00]; dup_x5 6/6 = 1.00 [0.54, 1.00]; cross_dup 6/6 = 1.00 [0.54, 1.00]; dup_within5 6/6 = 1.00 [0.54, 1.00]; dupnoise_1e-7 6/6 = 1.00 [0.54, 1.00]; dupnoise_1e-4 0/6 = 0.00 [0.00, 0.46]; dupnoise_1e-2 0/6 = 0.00 [0.00, 0.46]
(dupnoise_1e-4 and dupnoise_1e-2 are above dup_rtol = 1e-5: a documented non-detection, reported, not a criterion)

## C6: F009 pairing (DESCRIPTIVE ONLY, not a criterion, not in the verdict)

F009 is the v4 rule (32 evenly spaced samples). Arms: A = the shipped rule on the whole dataset; H64 = `--head 64` (every second sample is scored); H32 = `--head 32` (all 32 consecutive: a neighbour is in the matrix). Cell = detected (clean-adjusted) / skipped / miss. "applicable" = F009 not skipped on the clean copy in that arm.

### Arm A (shipped rule, whole dataset)

| dataset | F009 clean | shift_index | shift_index_5 | permute20 | shuffle_all |
|---|---|---|---|---|---|
| condiff_cubic0.1 | ok | skip | skip | miss | skip |
| condiff_cubic1.0 | ok | skip | skip | miss | skip |
| condiff_cubic2.0 | ok | skip | skip | miss | skip |
| condiff_exponential0.4 | ok | skip | skip | miss | skip |
| condiff_exponential1.0 | ok | skip | skip | miss | skip |
| condiff_gaussian0.4 | ok | skip | skip | miss | skip |

applicable sets: 6/6; F009 findings on clean copies: 0
detection over applicable sets: shift_index 0/6 = 0.00 [0.00, 0.46]; shift_index_5 0/6 = 0.00 [0.00, 0.46]; permute20 0/6 = 0.00 [0.00, 0.46]; shuffle_all 0/6 = 0.00 [0.00, 0.46]

### Arm H64 (--head 64)

| dataset | F009 clean | shift_index | shift_index_5 | permute20 | shuffle_all |
|---|---|---|---|---|---|
| condiff_cubic0.1 | ok | skip | skip | miss | skip |
| condiff_cubic1.0 | ok | skip | skip | miss | skip |
| condiff_cubic2.0 | ok | skip | skip | miss | skip |
| condiff_exponential0.4 | ok | skip | skip | miss | skip |
| condiff_exponential1.0 | ok | skip | skip | miss | skip |
| condiff_gaussian0.4 | ok | skip | skip | miss | skip |

applicable sets: 6/6; F009 findings on clean copies: 0
detection over applicable sets: shift_index 0/6 = 0.00 [0.00, 0.46]; shift_index_5 0/6 = 0.00 [0.00, 0.46]; permute20 0/6 = 0.00 [0.00, 0.46]; shuffle_all 0/6 = 0.00 [0.00, 0.46]

### Arm H32 (--head 32)

| dataset | F009 clean | shift_index | shift_index_5 | permute20 | shuffle_all |
|---|---|---|---|---|---|
| condiff_cubic0.1 | ok | hit | hit | miss | skip |
| condiff_cubic1.0 | ok | hit | hit | miss | skip |
| condiff_cubic2.0 | ok | hit | hit | miss | skip |
| condiff_exponential0.4 | ok | hit | hit | miss | skip |
| condiff_exponential1.0 | ok | hit | hit | miss | skip |
| condiff_gaussian0.4 | ok | hit | hit | miss | skip |

applicable sets: 6/6; F009 findings on clean copies: 0
detection over applicable sets: shift_index 6/6 = 1.00 [0.54, 1.00]; shift_index_5 6/6 = 1.00 [0.54, 1.00]; permute20 0/6 = 0.00 [0.00, 0.46]; shuffle_all 0/6 = 0.00 [0.00, 0.46]

## C7: advection probe

C7a specificity: advection probe triggered on 0 of 24 (binding in-scope clean + whole-dataset transpose/flip) conditions: none; required 0: **PASS**

* C7b sensitivity advdiff (control_nb): rules declining as advective: ['F002', 'F003', 'F008', 'F009']; F002 out of scope: advective (first-derivative terms explain the residual: median R2 0.18 -> 1.00 in the stored orientation with a uniform drift, against 0.19 for th; F008 out of scope: advective (first-derivative terms explain the residual: median R2 0.18 -> 1.00 in the stored orientation with a uniform drift, against 0.19 for th
* C7b sensitivity flowbench_ldcht (control): rules declining as advective: none; F002 0/400 samples (0.0%) have their maximum away from every source; F008 not applicable: the operator model explains only median R2 = 0.12 in any orientation (< 0.3); the field is probably time-dependent, advective or strongly couple

## Mechanical verdict (protocol_v5.md section 10)

C1 PASS; C2 by hand (provisional listing above); C3 FAIL; C4 PASS; C5 PASS (a PASS, b PASS, c provisional); C6 descriptive; C7a PASS

PAPER-WORTHY: C1-C5 all hold. PARTIAL: C1 and C3 hold but C2, C4 or C5 does not. NOT PAPER-WORTHY: C1 or C3 fails. (C6 and C7 are reported, not part of the verdict.)
