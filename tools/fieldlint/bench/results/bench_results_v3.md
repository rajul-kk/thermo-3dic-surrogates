# fieldlint v3 results (pre-registered, run once)

Protocol: `protocol_v3.md`, frozen at commit f118382 (2026-10-09). Rules and thresholds: the v2 frozen code (hash check passed in every run, recorded in each `results/v3/*.json`).
Runs: `run_v3.py --run-all inscope|control|diagnostic`, one run per name, no crash, no re-launch. Raw: `results/v3/*.json`.
Numbers come from `report_v3.py` and `run_bench.markdown` unchanged, plus the adjudication of section 4 (finding texts, and for the checks fixed in advance, the converted files). No post-hoc analysis was run in this pass.

## 1. Criteria and mechanical verdict

| id | criterion (protocol section 10) | result | exact 95% CI (Clopper-Pearson) | pass |
|---|---|---|---|---|
| C1 | clean-adjusted any-rule detection of whole-dataset transpose / flip_x / flip_y, in-scope, >= 80% | 18/21 = 0.857 (v2 count, not adjusted: 21/21) | two-sided 0.637-0.970; one-sided lower bound 0.671 | **PASS** |
| C2 | clean false positives after adjudication <= floor(7/10) = 0 | 0 of 7 (2 of 7 had findings before adjudication, all adjudicated TRUE, section 4) | 0/7: two-sided 0-0.41, one-sided upper 0.35 | **PASS** |
| C3 | no false alarm (F002/F003/F005/F008/F009) on the binding control runs | **3 false alarms in 2 binding runs**: `dpde_helmholtz` F003 error; `ucsd_convdiff` F003 error and F008 error | | **FAIL** |
| C4 | F008 applicable on >= 50% of the native in-scope sets (3 of 5) | 5/5 (median R^2 0.9999 to 1.0000 on all five) | two-sided 0.48-1; one-sided lower 0.55 | **PASS** |

Binding runs for C3: the flagged runs (time-dependent controls with `time_dependent: true`) and the steady controls (Helmholtz x2, ConvDiff), which have no flag.

**Mechanical verdict (protocol section 10): NOT PAPER-WORTHY as measured, because C3 fails.** C1, C2 and C4 hold. The verdict rule is "C1 or C3 fails -> NOT PAPER-WORTHY".
The scope caveat of the protocol applies: the in-scope sets are generic Poisson/Darcy families, so the numbers speak to generalisation across generators, resolutions and releases, not across application fields.

## 2. In-scope and diagnostic detection (v2 scoring; "any rule" here is NOT clean-adjusted)


### v3 in-scope (7 datasets) (7 datasets: cno_darcy, cno_poisson, dpde_darcy_denorm, dpde_poisson_denorm, elado_a1_test, pdegym_poisson_gauss, ucsd_poisson)

| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/7 | 0/7 (6) | 0/7 (3) | 1/7 | 2/7 | 0/7 (1) | 0/7 (1) | 2/7 |
| transpose | 0/7 | 1/7 (6) | 3/7 (3) | 1/7 | 2/7 | 6/7 (1) | 4/7 (3) | 7/7 |
| flip_x | 0/7 | 1/7 (6) | 4/7 (3) | 1/7 | 2/7 | 6/7 (1) | 2/7 (5) | 7/7 |
| flip_y | 0/7 | 1/7 (6) | 4/7 (3) | 1/7 | 2/7 | 6/7 (1) | 3/7 (4) | 7/7 |
| roll1 | 0/7 | 0/7 (6) | 0/7 (3) | 1/7 | 2/7 | 3/7 (1) | 1/7 (2) | 4/7 |
| roll2 | 0/7 | 0/7 (6) | 0/7 (3) | 1/7 | 2/7 | 5/7 (1) | 1/7 (3) | 6/7 |
| roll3 | 0/7 | 0/7 (6) | 0/7 (3) | 1/7 | 2/7 | 5/7 (1) | 1/7 (3) | 6/7 |
| shift_index | 0/7 | 1/7 (6) | 3/7 (3) | 1/7 | 2/7 | 1/7 (6) | 3/7 (3) | 7/7 |
| cross_dup | 0/7 | 0/7 (6) | 0/7 (3) | 7/7 | 2/7 | 0/7 (1) | 0/7 (1) | 7/7 |
| transpose@20 | 0/7 | 1/7 (6) | 3/7 (3) | 1/7 | 2/7 | 6/7 (1) | 0/7 (1) | 7/7 |
| flip_x@20 | 0/7 | 1/7 (6) | 4/7 (3) | 1/7 | 2/7 | 6/7 (1) | 0/7 (1) | 7/7 |
| flip_y@20 | 0/7 | 1/7 (6) | 4/7 (3) | 1/7 | 2/7 | 6/7 (1) | 0/7 (1) | 7/7 |
| roll1@20 | 0/7 | 0/7 (6) | 0/7 (3) | 1/7 | 2/7 | 3/7 (1) | 0/7 (1) | 4/7 |
| shift_index@20 | 0/7 | 1/7 (6) | 0/7 (3) | 7/7 | 2/7 | 1/7 (1) | 0/7 (1) | 7/7 |

### v3 diagnostic, as released (2 datasets: dpde_darcy_asreleased, dpde_poisson_asreleased)

| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/2 | 0/2 (2) | 0/2 (1) | 0/2 | 1/2 | 0/2 (1) | 0/2 (1) | 1/2 |
| transpose | 0/2 | 0/2 (2) | 1/2 (1) | 0/2 | 1/2 | 1/2 (1) | 1/2 (1) | 2/2 |
| flip_x | 0/2 | 0/2 (2) | 1/2 (1) | 0/2 | 1/2 | 1/2 (1) | 1/2 (1) | 2/2 |
| flip_y | 0/2 | 0/2 (2) | 1/2 (1) | 0/2 | 1/2 | 1/2 (1) | 1/2 (1) | 2/2 |
| roll1 | 0/2 | 0/2 (2) | 0/2 (1) | 0/2 | 1/2 | 0/2 (1) | 0/2 (1) | 1/2 |
| roll2 | 0/2 | 0/2 (2) | 0/2 (1) | 0/2 | 1/2 | 1/2 (1) | 0/2 (1) | 2/2 |
| roll3 | 0/2 | 0/2 (2) | 0/2 (1) | 0/2 | 1/2 | 1/2 (1) | 0/2 (1) | 2/2 |
| shift_index | 0/2 | 0/2 (2) | 1/2 (1) | 0/2 | 1/2 | 0/2 (2) | 1/2 (1) | 2/2 |
| cross_dup | 0/2 | 0/2 (2) | 0/2 (1) | 2/2 | 1/2 | 0/2 (1) | 0/2 (1) | 2/2 |
| transpose@20 | 0/2 | 0/2 (2) | 1/2 (1) | 0/2 | 1/2 | 1/2 (1) | 0/2 (1) | 2/2 |
| flip_x@20 | 0/2 | 0/2 (2) | 1/2 (1) | 0/2 | 1/2 | 1/2 (1) | 0/2 (1) | 2/2 |
| flip_y@20 | 0/2 | 0/2 (2) | 1/2 (1) | 0/2 | 1/2 | 1/2 (1) | 0/2 (1) | 2/2 |
| roll1@20 | 0/2 | 0/2 (2) | 0/2 (1) | 0/2 | 1/2 | 0/2 (1) | 0/2 (1) | 1/2 |
| shift_index@20 | 0/2 | 0/2 (2) | 0/2 (1) | 2/2 | 1/2 | 1/2 (1) | 0/2 (1) | 2/2 |


### Per dataset: rules that detect each fault

| dataset | clean flagged | transpose | flip_x | flip_y | roll1 | roll2 | roll3 | shift_index | unit_slip | cross_dup | transpose@20 | flip_x@20 | flip_y@20 | roll1@20 | shift_index@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cno_darcy | F004,F005 | F004,F005 | F004,F005 | F004,F005 | F004,F005 | F004,F005 | F004,F005 | F004,F005 | n/a | F004,F005 | F004,F005 | F004,F005 | F004,F005 | F004,F005 | F004,F005 |
| cno_poisson | - | F003,F008 | F003,F008 | F003,F008 | F008 | F008 | F008 | F003 | n/a | F004 | F003,F008 | F003,F008 | F003,F008 | F008 | F004 |
| dpde_darcy_denorm | F005 | F005,F008 | F005,F008 | F005,F008 | F005,F008 | F005,F008 | F005,F008 | F005 | n/a | F004,F005 | F005,F008 | F005,F008 | F005,F008 | F005,F008 | F004,F005 |
| dpde_poisson_denorm | - | F003,F008,F009 | F003,F008,F009 | F003,F008,F009 | - | F008 | F008 | F003,F009 | n/a | F004 | F003,F008 | F003,F008 | F003,F008 | - | F004,F008 |
| elado_a1_test | - | F008,F009 | F008,F009 | F008,F009 | - | - | - | F009 | n/a | F004 | F008 | F008 | F008 | - | F004 |
| pdegym_poisson_gauss | - | F002,F003,F008,F009 | F002,F003,F008 | F002,F003,F008,F009 | - | F008 | F008 | F002,F003,F009 | n/a | F004 | F002,F003,F008 | F002,F003,F008 | F002,F003,F008 | - | F002,F004 |
| ucsd_poisson | - | F008,F009 | F003,F008 | F003,F008 | F008,F009 | F008,F009 | F008,F009 | F008 | n/a | F004 | F008 | F003,F008 | F003,F008 | F008 | F004 |
| dpde_darcy_asreleased | F005 | F005 | F005 | F005 | F005 | F005 | F005 | F005 | n/a | F004,F005 | F005 | F005 | F005 | F005 | F004,F005 |
| dpde_poisson_asreleased | - | F003,F008,F009 | F003,F008,F009 | F003,F008,F009 | - | F008 | F008 | F003,F009 | n/a | F004 | F003,F008 | F003,F008 | F003,F008 | - | F004,F008 |


Clean-adjusted any-rule count for whole-dataset faults (a rule is credited only if it is not already flagged on the clean copy): transpose 6/7, flip_x 6/7, flip_y 6/7. The one miss per fault is `cno_darcy`, where the only rules that fire (F004, F005) already fire on the clean data.

### F008 on clean in-scope data

| dataset | native? | F008 | median R^2 stored | face mean |
|---|---|---|---|---|
| pdegym_poisson_gauss | yes | ok | 1.0000 | harmonic |
| cno_poisson | yes | ok | 0.9999 | harmonic |
| dpde_poisson_denorm | yes | ok | 1.0000 | harmonic |
| elado_a1_test | yes | ok | 1.0000 | arithmetic |
| ucsd_poisson | yes | ok | 0.9999 | arithmetic |
| dpde_darcy_denorm | no | ok | 0.60 | arithmetic |
| cno_darcy | no | skipped | 0.24 (< 0.3 gate) | harmonic |

### Per generator family (clean-adjusted whole-dataset detection, 3 faults each)

| family | datasets | detected | note |
|---|---|---|---|
| -Laplace u = s, k = 1 (varying source) | pdegym_poisson_gauss, cno_poisson, dpde_poisson_denorm | 9/9 | F003 and F008 both carry it |
| variable k, uniform source (Darcy type) | elado_a1_test, cno_darcy, dpde_darcy_denorm | 6/9 | elado 3/3 (F008, F009), dpde_darcy_denorm 3/3 (F008), cno_darcy 0/3 (F008 skipped) |
| variable k and variable source | ucsd_poisson | 3/3 | |

## 3. Controls (clean only; declined = skipped or nothing flagged; false alarm = warning or error)

### Controls (declined vs false alarm; F002 F003 F005 F008 F009; F004 and F001 listed separately)

| run | F001 | F002 | F003 | F004 | F005 | F008 | F009 |
|---|---|---|---|---|---|---|---|
| cno_allen_cahn+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| cno_allen_cahn | ok | skipped | ok | ok | ok (info) | ok | ok |
| cno_cont_transport+time_dependent_flag | ok | skipped | skipped | flagged (error) | ok | skipped | skipped |
| cno_cont_transport | ok | ok | flagged (warning) | flagged (error) | ok | skipped | skipped |
| cno_disc_transport+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| cno_disc_transport | ok | skipped | flagged (warning) | ok | ok (info) | skipped | skipped |
| cno_navier_stokes+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| cno_navier_stokes | ok | skipped | ok | ok | ok (info) | ok | flagged (error) |
| cno_wave+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| cno_wave | ok | skipped | flagged (error) | ok | ok (info) | flagged (error) | flagged (error) |
| dpde_helmholtz | ok | skipped | flagged (error) | ok | ok (info) | ok | ok |
| elado_helmholtz_e1 | ok | skipped | skipped | ok | ok | skipped | skipped |
| pdegym_wave_gauss+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| pdegym_wave_gauss | ok | flagged (error) | flagged (warning) | ok | ok (info) | skipped | skipped |
| ucsd_convdiff | ok | skipped | flagged (error) | ok | ok | flagged (error) | skipped |


Reading. With the scope declaration set, every time-dependent control is declined by F002/F003/F008/F009 ("out of scope"). The only finding on a flagged run is F004 on `cno_cont_transport` (true duplicates, section 4).
Without the declaration (C3-strict, reported, not binding): false alarms on `cno_wave` (F003, F008, F009 errors), `cno_navier_stokes` (F009 error), `cno_cont_transport` (F003 warning), `cno_disc_transport` (F003 warning), `pdegym_wave_gauss` (F002 error, F003 warning); `cno_allen_cahn` unflagged: nothing flagged.
Steady controls (no flag exists): `elado_helmholtz_e1` declined (F003, F008, F009 skipped; F005 ok); `dpde_helmholtz` false alarm, F003 error (stored correlation -0.82, flip_xy 0.31); `ucsd_convdiff` false alarms, F003 error and F008 error (stored R^2 0.18 against 0.67 for flip_xy).

## 4. Adjudication of every clean finding (protocol section 7)

In-scope runs:

| dataset | rule | finding | route | class | verdict |
|---|---|---|---|---|---|
| cno_darcy | F004 error | 512 samples: 2 cross-split, 9 within-split near-duplicates (rule's positional 80/20 split) | **P2**: checked on the raw file `Darcy_64x64_IN.h5` and the converted copy: 11 pairs of samples with bit-identical `output` and `input`, e.g. samples 27 and 493 (`np.array_equal` True on the raw groups) | **material** (duplicate samples) | TRUE |
| cno_darcy | F005 warning | u in [-0.021, 0]: output <= 0 everywhere | **P1** (documented -div(a grad u)=1, u\|dD=0, f=1 gives u >= 0; sign decision in `datasets_v3.md` 2.3: nothing in loader or paper negates it) | **benign** (sign convention) | TRUE |
| dpde_darcy_denorm | F005 warning | the rule reports u in [-4.0e-05, 0.0175] | **P1**: negative cells exist (in the 400 evenly spaced samples: 85 negative cells of 6.55 M, 48 off the outer ring, in 21 samples, minimum -0.0035 against a maximum 0.0175). The generator interpolates by splines between cell centres and nodes (`solve_gwf.m` lines 8-12), and the denormalisation uses the global metadata mean and std; either can produce small undershoots | **benign** (does not change what a model learns) | TRUE under P1 as worded; flagged to the lead, see section 6 item 4 |
| all other in-scope | none | | | | |

False positives after adjudication: **0 of 7**. TRUE findings by class (in-scope): material 1 (CNO Darcy duplicates), benign 2 (CNO Darcy sign, DiffusionPDE Darcy small negative values). Benign findings are not discoveries.

Controls, F004 (the protocol treats literal duplicates like clean findings):

| control | finding | route | class | verdict |
|---|---|---|---|---|
| cno_cont_transport (flagged and unflagged runs) | F004 error (duplicates; the earlier cross/within-split counts are superseded by the P2 verification) | **P2**: in `ContTranslation_64x64_IN.h5`, sample i and sample i+512 have bit-identical input and output for every i in 0..511 (512 distinct of 1024). The CNO paper (appendix C.3) specifies 512 train, 256 validation and 256 in-distribution test samples; under the loader's offsets (0, 512, 768) the validation set duplicates train samples 0–255 and the test set duplicates train samples 256–511 (e.g. raw groups `Sample_5` and `Sample_517`, `np.array_equal` True) | **material** | TRUE |

Diagnostic (as released; not adjudicated, not counted): `dpde_darcy_asreleased` F005 error (k in {-0.5, +0.5} and standardised u, by construction); `dpde_poisson_asreleased` nothing flagged on clean (F005 info).

POST HOC (descriptive, not part of any criterion): the CNO Darcy duplicates fall across the CNO loader's own default splits (`CNOBenchmarks.py` `DarcyDataset`: train indices 0-255, validation 256-383, test 384-511): under train 0-255 / validation 256-383 / test 384-511, of the 11 pairs 6 are train/validation, 3 are train/test and 2 are validation/validation. For ContTranslation, sample i and sample i+512 are bit-identical for every i in 0..511 (512 distinct of 1024; verified 2026-10-14 on the raw file); with the paper's 512/256/256 split and the loader's offsets (0, 512, 768), the validation set duplicates train samples 0-255 and the test set duplicates train samples 256-511.

POST HOC verification on raw files (2026-10-09; descriptive, changes no pre-registered number above):

* `Darcy_64x64_IN.h5` (sha256 e460c2d9...): 11 bit-identical pairs, so 501 distinct of 512 samples. Pairs: (27,493), (59,379), (79,283), (82,304), (87,309), (98,351), (203,398), (209,412), (246,346), (269,349), (296,381). Under the split that `TrainCNO.py` uses (training_samples=256, `CNO2d_classic/TrainCNO.py` line 26): train 0-255, validation 256-383, test 384-511. Then 6 pairs are train/validation, 3 are train/test and 2 are validation/validation.
* `ContTranslation_64x64_IN.h5` (sha256 1c116871...): 512 pairs (i, i+512), so only 512 distinct of 1024 samples. Under the paper's split (512 train, 256 validation, 256 test; offsets 0, 512, 768): 256 train/validation pairs and 256 train/test pairs. Under `TrainCNO.py`'s training_samples=256 (train 0-255, validation 512-767, test 768-1023) the test set duplicates indices 256-511, which that run does not train on.
* Prior reports: none found in the Zenodo record 10406879 description, the README, or the 11 GitHub issues.
* Loader quirk: the `DarcyDataset` class default (512) requests samples 512-639, which do not exist in this 512-sample file.

## 5. Predictions (datasets_v3.md) against the outcome

Right: PDEgym Poisson-Gauss F002/F003/F008 applicable and clean; CNO Poisson F002 skipped, F003 ok, F008 applicable; CNO Darcy F002/F003/F008/F009 skipped, F005 flagged (as a warning), clean-adjusted detection 0/3; DiffusionPDE Poisson (with source = -a) F002 skipped, F003 ok, F008 applicable at R^2 1.00; DiffusionPDE Darcy as released F005 error; ELADO F002/F003 skipped, F008 applicable (arithmetic, R^2 1.00); UCSD F003 ok, F008 applicable, roll faults detected; all time-dependent controls declined with the flag; ELADO Helmholtz declined; `dpde_helmholtz` F003 alarm (pre-registered risk, materialised); unflagged transports F003 warnings (pre-registered possibility); Wave-Gauss unflagged F002/F003 alarms (possibility).

Wrong: (1) `dpde_darcy_denorm` F008 predicted skipped, actually applicable (R^2 0.60, arithmetic) and it detects transpose, flips and rolls; (2) `dpde_darcy_denorm` F005 predicted ok, actually a warning (small negative u); (3) `ucsd_poisson` F002 predicted applicable, actually skipped (the source covers every cell, "vacuous"); (4) `ucsd_convdiff` predicted at worst a decline or a lower F008 R^2, actually F003 and F008 errors; (5) the CNO Darcy F004 duplicates were not predicted; (6) `cno_poisson` F008 predicted R^2 of order 0.9, actually 0.9999.

## 6. Post-hoc candidates (NOT run; for the lead to decide)

1. Why `ucsd_convdiff` fires F003/F008 (stored R^2 0.18, flip_xy 0.67) and `dpde_helmholtz` F003 (stored correlation -0.82): a property of the control mappings (orientation, sign) or of the data; look at convection direction and at the sign convention of the Helmholtz mapping.
2. F003 behaviour when the stored correlation is large and negative (it errs on `dpde_helmholtz`): a rule change, so a new frozen version, not a patch.
3. Re-score C3 with sign-corrected Helmholtz and ConvDiff mappings (a mapping change, so a labelled new run).
4. P1 fired on a numerical-size negative value in DiffusionPDE Darcy; a tolerance would be a protocol change.
5. Check the other CNO files (OUT files, Wave, the 10k Navier-Stokes) for the same duplicate pattern and report the CNO Darcy and ContTranslation duplicates to the maintainers (the independent confirmation, route c, and a possible stand-alone data finding).
6. Why F008 works on DiffusionPDE Darcy (R^2 0.60) but not on CNO Darcy (0.24): face-mean conventions and the sign of the CNO output.
7. More samples (other ELADO files, remaining DiffusionPDE shards, other UCSD files) to tighten the C1 interval.
