# fieldlint v5: tuning before / after (before = stored v4 results, after = v5 code)

v5 changes: F005 below-ambient check honours scope_skip and the weak-minimum-principle probe (osc_v 0.05, fraction 0.5); F009 scores the first 64 consecutive samples. Tuned on everything seen; NOT held-out numbers.

Records compared: 53 runs. Cells changed in status/severity/detected: 69; UNEXPECTED (not F009, not F005 on PDEgym Helmholtz / Wave-Layer): **0**.

## F005 differences (expected set)

| record | condition | before | after |
|---|---|---|---|
| cno_cont_transport+time_dependent_flag | clean | ('ok', 'none', False) | ('ok', 'info', False) |
| pdegym_helmholtz | clean | ('flagged', 'warning', True) | ('ok', 'info', False) |
| pdegym_wave_layer | clean | ('flagged', 'warning', True) | ('ok', 'info', False) |
| pdegym_wave_layer+time_dependent_flag | clean | ('flagged', 'warning', True) | ('ok', 'info', False) |

## F009 (intended change): detection / ran / skipped per fault, pooled over in-scope + diagnostic runs

| condition | before | after |
|---|---|---|
| clean | 0 detected / 34 ran / 7 skipped | 1 / 34 / 7 |
| transpose | 12 detected / 12 ran / 26 skipped | 14 / 14 / 24 |
| flip_x | 15 detected / 15 ran / 26 skipped | 19 / 19 / 22 |
| flip_y | 9 detected / 12 ran / 29 skipped | 13 / 16 / 25 |
| roll1 | 1 detected / 20 ran / 21 skipped | 1 / 20 / 21 |
| roll2 | 1 detected / 18 ran / 23 skipped | 2 / 19 / 22 |
| roll3 | 2 detected / 19 ran / 22 skipped | 3 / 20 / 21 |
| shift_index | 16 detected / 17 ran / 24 skipped | 31 / 32 / 9 |
| unit_slip | 0 detected / 14 ran / 4 skipped | 1 / 14 / 4 |
| cross_dup | 0 detected / 34 ran / 7 skipped | 1 / 34 / 7 |
| transpose@20 | 0 detected / 31 ran / 7 skipped | 1 / 30 / 8 |
| flip_x@20 | 6 detected / 34 ran / 7 skipped | 1 / 33 / 8 |
| flip_y@20 | 4 detected / 34 ran / 7 skipped | 1 / 33 / 8 |
| roll1@20 | 0 detected / 34 ran / 7 skipped | 1 / 33 / 8 |
| shift_index@20 | 7 detected / 34 ran / 7 skipped | 1 / 34 / 7 |

## F009 changes on clean copies

| record | before | after |
|---|---|---|
| thermfm_IND_32C | ('ok', 'none', False) | ('flagged', 'warning', True) |
| elado_helmholtz_c1 | ('skipped', 'none', False) | ('flagged', 'error', True) |

## Conditions where the any-rule detection (all rules) changed

* thermfm_HS_SC_refine1 / flip_y: any-rule False -> True
* thermfm_HS_SC_refine1 / roll2: any-rule False -> True
* thermfm_HS_SC_refine2 / flip_y: any-rule False -> True
* thermfm_HS_SC_refine2 / roll3: any-rule False -> True
* thermfm_IND_32C / clean: any-rule False -> True
* thermfm_IND_32C / roll1: any-rule False -> True
* thermfm_IND_32C / roll1@20: any-rule False -> True
* darcy_beta1.0 / shift_index: any-rule False -> True
* fno_darcy_native421 / shift_index: any-rule False -> True
* condiff_poisson / shift_index: any-rule False -> True
* condiff_gaussian0.1 / shift_index: any-rule False -> True
* condiff_cubic0.4 / shift_index: any-rule False -> True
* condiff_exponential2.0 / shift_index: any-rule False -> True
* elado_b1_test / shift_index: any-rule False -> True
* pdebench_darcy_beta10.0 / shift_index: any-rule False -> True
* pdebench_darcy_beta100.0 / shift_index: any-rule False -> True
* condiff_gaussian0.1_nok / shift_index: any-rule False -> True
* pdegym_helmholtz / clean: any-rule True -> False
* elado_helmholtz_c1 / clean: any-rule False -> True
* pdegym_wave_layer+time_dependent_flag / clean: any-rule True -> False
