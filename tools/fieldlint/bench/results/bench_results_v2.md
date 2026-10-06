# fieldlint measured detection / false-positive rates

Thresholds frozen 2026-10-07T02:15:55+0800 on the tuning datasets (results/frozen_thresholds.json). Cell = detected / trials (skipped by the rule in brackets). Detected = rule flagged with a warning or error. "clean" = false-positive rate.

## v2 TUNING datasets (v1 tuning sets + both PDEBench Darcy files) (20 datasets: ictherm_s2, ictherm_s3, ictherm_s4, ictherm_s5, thermfm_HS_OC_refine1, thermfm_HS_OC_refine2, thermfm_HS_QC_refine1, thermfm_HS_QC_refine2, thermfm_HS_SC_refine1, thermfm_HS_SC_refine2, thermfm_IND_8C, thermfm_IND_32C, 3dice_geometry1, 3dice_geometry2a, 3dice_geometry3, 3dice_geometry4, 3dice_geometry5, 3dice_geometry6, darcy_beta1.0, darcy_beta0.01)

| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/20 | 0/20 (4) | 0/20 (2) | 1/20 | 0/20 | 0/20 (4) | 0/20 (4) | 1/20 |
| transpose | 0/17 | 10/17 (4) | 15/17 (2) | 1/17 | 0/17 | 13/17 (4) | 7/17 (10) | 17/17 |
| flip_x | 0/20 | 10/20 (4) | 18/20 (2) | 1/20 | 0/20 | 16/20 (4) | 12/20 (8) | 20/20 |
| flip_y | 0/20 | 10/20 (4) | 15/20 (2) | 1/20 | 0/20 | 16/20 (4) | 5/20 (12) | 18/20 |
| roll1 | 0/20 | 0/20 (4) | 0/20 (2) | 1/20 | 0/20 | 13/20 (4) | 0/20 (7) | 13/20 |
| roll2 | 0/20 | 0/20 (4) | 0/20 (2) | 1/20 | 0/20 | 15/20 (4) | 0/20 (8) | 15/20 |
| roll3 | 0/20 | 5/20 (4) | 0/20 (2) | 1/20 | 0/20 | 16/20 (4) | 1/20 (7) | 17/20 |
| shift_index | 0/20 | 6/20 (4) | 12/20 (2) | 1/20 | 6/20 | 4/20 (14) | 12/20 (8) | 16/20 |
| unit_slip | 0/18 | 0/18 (8) | 0/18 | 1/18 | 15/18 | 0/18 (4) | 0/18 (4) | 15/18 |
| cross_dup | 0/20 | 0/20 (4) | 0/20 (2) | 20/20 | 0/20 | 0/20 (4) | 0/20 (4) | 20/20 |
| transpose@20 | 0/17 | 9/17 (4) | 12/17 (2) | 1/17 | 0/17 | 13/17 (4) | 0/17 (4) | 15/17 |
| flip_x@20 | 0/20 | 9/20 (4) | 16/20 (2) | 1/20 | 0/20 | 16/20 (4) | 6/20 (4) | 18/20 |
| flip_y@20 | 0/20 | 10/20 (4) | 12/20 (2) | 1/20 | 0/20 | 16/20 (4) | 4/20 (4) | 18/20 |
| roll1@20 | 0/20 | 0/20 (4) | 0/20 (2) | 1/20 | 0/20 | 12/20 (4) | 0/20 (4) | 12/20 |
| shift_index@20 | 0/20 | 6/20 (4) | 1/20 (2) | 20/20 | 4/20 | 6/20 (4) | 7/20 (4) | 20/20 |

## v2 FRESH HELD-OUT dataset (FNO-paper Darcy, Zenodo 12784353; never looked at before the v2 freeze) (1 datasets: fno_darcy_128)

| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| transpose | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| flip_x | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| flip_y | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| roll1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| roll2 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| roll3 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| shift_index | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| cross_dup | 0/1 | 0/1 (1) | 0/1 (1) | 1/1 | 0/1 | 0/1 (1) | 0/1 (1) | 1/1 |
| transpose@20 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| flip_x@20 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| flip_y@20 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| roll1@20 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 | 0/1 | 0/1 (1) | 0/1 (1) | 0/1 |
| shift_index@20 | 0/1 | 0/1 (1) | 0/1 (1) | 1/1 | 0/1 | 0/1 (1) | 0/1 (1) | 1/1 |

## v2 out-of-scope controls (clean files, no faults)

| dataset | F001 | F002 | F003 | F004 | F005 | F008 | F009 |
|---|---|---|---|---|---|---|---|
| diff_react_t20 | ok | skipped | ok | ok | ok (info) | skipped | skipped |
| burgers_nu0.01 | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| burgers_nu0.01_t200 | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| diff_react_t20+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| burgers_nu0.01+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
| burgers_nu0.01_t200+time_dependent_flag | ok | skipped | skipped | ok | ok (info) | skipped | skipped |
