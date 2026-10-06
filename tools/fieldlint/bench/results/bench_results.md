# fieldlint measured detection / false-positive rates

Thresholds frozen 2026-10-06T16:16:02+0800 on the tuning datasets (results/frozen_thresholds.json). Cell = detected / trials (skipped by the rule in brackets). Detected = rule flagged with a warning or error. "clean" = false-positive rate.

## TUNING datasets (thresholds were fitted here) (18 datasets: ictherm_s2, ictherm_s3, ictherm_s4, ictherm_s5, thermfm_HS_OC_refine1, thermfm_HS_OC_refine2, thermfm_HS_QC_refine1, thermfm_HS_QC_refine2, thermfm_HS_SC_refine1, thermfm_HS_SC_refine2, thermfm_IND_8C, thermfm_IND_32C, 3dice_geometry1, 3dice_geometry2a, 3dice_geometry3, 3dice_geometry4, 3dice_geometry5, 3dice_geometry6)

| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/18 | 0/18 (2) | 0/18 | 1/18 | 0/18 | 0/18 (5) | 0/18 (5) | 1/18 |
| transpose | 0/15 | 10/15 (2) | 15/15 | 1/15 | 0/15 | 10/15 (5) | 7/15 (8) | 15/15 |
| flip_x | 0/18 | 10/18 (2) | 18/18 | 1/18 | 0/18 | 13/18 (5) | 12/18 (6) | 18/18 |
| flip_y | 0/18 | 10/18 (2) | 15/18 | 1/18 | 0/18 | 13/18 (5) | 5/18 (10) | 16/18 |
| roll1 | 0/18 | 0/18 (2) | 0/18 | 1/18 | 0/18 | 8/18 (5) | 0/18 (5) | 8/18 |
| roll2 | 0/18 | 0/18 (2) | 0/18 | 1/18 | 0/18 | 12/18 (5) | 0/18 (6) | 12/18 |
| roll3 | 0/18 | 5/18 (2) | 0/18 | 1/18 | 0/18 | 13/18 (5) | 1/18 (5) | 15/18 |
| shift_index | 0/18 | 6/18 (2) | 12/18 | 1/18 | 6/18 | 4/18 (12) | 12/18 (6) | 16/18 |
| unit_slip | 0/18 | 0/18 (8) | 0/18 | 1/18 | 15/18 | 0/18 (5) | 0/18 (5) | 15/18 |
| cross_dup | 0/18 | 0/18 (2) | 0/18 | 18/18 | 0/18 | 0/18 (5) | 0/18 (5) | 18/18 |
| transpose@20 | 0/15 | 9/15 (2) | 12/15 | 1/15 | 0/15 | 10/15 (5) | 0/15 (5) | 13/15 |
| flip_x@20 | 0/18 | 9/18 (2) | 16/18 | 1/18 | 0/18 | 13/18 (5) | 7/18 (5) | 16/18 |
| flip_y@20 | 0/18 | 10/18 (2) | 12/18 | 1/18 | 0/18 | 13/18 (5) | 4/18 (5) | 16/18 |
| roll1@20 | 0/18 | 0/18 (2) | 0/18 | 1/18 | 0/18 | 7/18 (5) | 0/18 (5) | 7/18 |
| shift_index@20 | 0/18 | 6/18 (2) | 1/18 | 18/18 | 4/18 | 6/18 (5) | 7/18 (5) | 18/18 |

## HELD-OUT datasets (thresholds frozen before these were run) (2 datasets: darcy_beta1.0, darcy_beta0.01)

| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| transpose | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| flip_x | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| flip_y | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| roll1 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| roll2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| roll3 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| shift_index | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| cross_dup | 0/2 | 0/2 (2) | 0/2 (2) | 2/2 | 0/2 | 0/2 (2) | 0/2 (2) | 2/2 |
| transpose@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| flip_x@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| flip_y@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| roll1@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| shift_index@20 | 0/2 | 0/2 (2) | 0/2 (2) | 2/2 | 0/2 | 0/2 (2) | 0/2 (2) | 2/2 |

## HELD-OUT datasets, POST-HOC re-run with arithmetic face conductivity (chosen after seeing the frozen run; not a clean held-out measurement) (2 datasets: darcy_beta1.0, darcy_beta0.01)

| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| transpose | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 (2) | 2/2 |
| flip_x | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 (2) | 2/2 |
| flip_y | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 (2) | 2/2 |
| roll1 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 (2) | 2/2 |
| roll2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 (2) | 2/2 |
| roll3 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 (2) | 2/2 |
| shift_index | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 |
| cross_dup | 0/2 | 0/2 (2) | 0/2 (2) | 2/2 | 0/2 | 0/2 | 0/2 | 2/2 |
| transpose@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 | 2/2 |
| flip_x@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 | 2/2 |
| flip_y@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 | 2/2 |
| roll1@20 | 0/2 | 0/2 (2) | 0/2 (2) | 0/2 | 0/2 | 2/2 | 0/2 | 2/2 |
| shift_index@20 | 0/2 | 0/2 (2) | 0/2 (2) | 2/2 | 0/2 | 0/2 | 0/2 | 2/2 |

Note: files changed after the freeze: rules.py. The tuning / held-out / control runs were made with the frozen code; the only later change is the op_face_mean option (default = the frozen harmonic mean), used by the post-hoc block.

## Out-of-scope controls (clean files, no faults)

| dataset | F001 | F002 | F003 | F004 | F005 | F008 | F009 |
|---|---|---|---|---|---|---|---|
| diff_react_t20 | ok | skipped | ok | ok | ok (info) | skipped | skipped |
| burgers_nu0.01 | ok | flagged (error) | flagged (warning) | ok | ok (info) | skipped | skipped |
| burgers_nu0.01_t200 | ok | flagged (error) | flagged (warning) | ok | ok (info) | skipped | skipped |
