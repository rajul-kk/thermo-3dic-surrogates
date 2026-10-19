# fieldlint v5: tuning before / after (before = stored v4 results, after = v5 code)

v5 changes: F005 below-ambient check honours scope_skip and the weak-minimum-principle probe (osc_v 0.05, fraction 0.5); F009 is unchanged from v4 (the 64-sample block variant was tried and reverted). Tuned on everything seen; NOT held-out numbers.

Records compared: 53 runs. Cells changed in status/severity/detected: 4; UNEXPECTED (not F009, not F005 on PDEgym Helmholtz / Wave-Layer): **0**.

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
| clean | 0 detected / 34 ran / 7 skipped | 0 / 34 / 7 |
| transpose | 12 detected / 12 ran / 26 skipped | 12 / 12 / 26 |
| flip_x | 15 detected / 15 ran / 26 skipped | 15 / 15 / 26 |
| flip_y | 9 detected / 12 ran / 29 skipped | 9 / 12 / 29 |
| roll1 | 1 detected / 20 ran / 21 skipped | 1 / 20 / 21 |
| roll2 | 1 detected / 18 ran / 23 skipped | 1 / 18 / 23 |
| roll3 | 2 detected / 19 ran / 22 skipped | 2 / 19 / 22 |
| shift_index | 16 detected / 17 ran / 24 skipped | 16 / 17 / 24 |
| unit_slip | 0 detected / 14 ran / 4 skipped | 0 / 14 / 4 |
| cross_dup | 0 detected / 34 ran / 7 skipped | 0 / 34 / 7 |
| transpose@20 | 0 detected / 31 ran / 7 skipped | 0 / 31 / 7 |
| flip_x@20 | 6 detected / 34 ran / 7 skipped | 6 / 34 / 7 |
| flip_y@20 | 4 detected / 34 ran / 7 skipped | 4 / 34 / 7 |
| roll1@20 | 0 detected / 34 ran / 7 skipped | 0 / 34 / 7 |
| shift_index@20 | 7 detected / 34 ran / 7 skipped | 7 / 34 / 7 |

## F009 changes on clean copies

| record | before | after |
|---|---|---|

## Conditions where the any-rule detection (all rules) changed

* pdegym_helmholtz / clean: any-rule True -> False
* pdegym_wave_layer+time_dependent_flag / clean: any-rule True -> False
