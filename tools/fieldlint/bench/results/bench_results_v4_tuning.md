# fieldlint v4: tuning results (before = v2/v3 stored results, after = v4 code)

v4 is TUNED on every dataset seen so far, including the v3 sets and the v3 control failures. These are tuning numbers, not held-out ones.
Cells list the rules (F-number without the F) that detect each fault; "any-rule" uses all of F001-F009 as in the earlier tables.

## v2 tuning sets + v2 held-out sets (all seen)

### pooled

**before (v2 code)**


| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 0/22 (5) | 0/22 (5) | 1/22 |
| transpose | 0/19 | 10/19 (6) | 15/19 (4) | 1/19 | 0/19 | 14/19 (5) | 7/19 (12) | 18/19 |
| flip_x | 0/22 | 10/22 (6) | 18/22 (4) | 1/22 | 0/22 | 17/22 (5) | 12/22 (10) | 21/22 |
| flip_y | 0/22 | 10/22 (6) | 15/22 (4) | 1/22 | 0/22 | 17/22 (5) | 5/22 (14) | 19/22 |
| roll1 | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 14/22 (5) | 0/22 (9) | 14/22 |
| roll2 | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 16/22 (5) | 0/22 (10) | 16/22 |
| roll3 | 0/22 | 5/22 (6) | 0/22 (4) | 1/22 | 0/22 | 17/22 (5) | 1/22 (9) | 18/22 |
| shift_index | 0/22 | 6/22 (6) | 12/22 (4) | 1/22 | 6/22 | 4/22 (16) | 12/22 (10) | 16/22 |
| unit_slip | 0/18 | 0/18 (8) | 0/18 | 1/18 | 15/18 | 0/18 (4) | 0/18 (4) | 15/18 |
| cross_dup | 0/22 | 0/22 (6) | 0/22 (4) | 22/22 | 0/22 | 0/22 (5) | 0/22 (5) | 22/22 |
| transpose@20 | 0/19 | 9/19 (6) | 12/19 (4) | 1/19 | 0/19 | 13/19 (6) | 0/19 (5) | 15/19 |
| flip_x@20 | 0/22 | 9/22 (6) | 16/22 (4) | 1/22 | 0/22 | 16/22 (6) | 6/22 (5) | 18/22 |
| flip_y@20 | 0/22 | 10/22 (6) | 12/22 (4) | 1/22 | 0/22 | 16/22 (6) | 4/22 (5) | 18/22 |
| roll1@20 | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 12/22 (6) | 0/22 (5) | 12/22 |
| shift_index@20 | 0/22 | 6/22 (6) | 1/22 (4) | 22/22 | 4/22 | 6/22 (6) | 7/22 (5) | 22/22 |

**after (v4 code)**


| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 0/22 (5) | 0/22 (5) | 1/22 |
| transpose | 0/19 | 10/19 (6) | 15/19 (4) | 1/19 | 0/19 | 14/19 (5) | 7/19 (12) | 18/19 |
| flip_x | 0/22 | 10/22 (6) | 18/22 (4) | 1/22 | 0/22 | 17/22 (5) | 12/22 (10) | 21/22 |
| flip_y | 0/22 | 10/22 (6) | 15/22 (4) | 1/22 | 0/22 | 17/22 (5) | 5/22 (14) | 19/22 |
| roll1 | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 14/22 (5) | 0/22 (9) | 14/22 |
| roll2 | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 16/22 (5) | 0/22 (10) | 16/22 |
| roll3 | 0/22 | 5/22 (6) | 0/22 (4) | 1/22 | 0/22 | 17/22 (5) | 1/22 (9) | 18/22 |
| shift_index | 0/22 | 6/22 (6) | 12/22 (4) | 1/22 | 6/22 | 4/22 (16) | 12/22 (10) | 16/22 |
| unit_slip | 0/18 | 0/18 (8) | 0/18 | 1/18 | 15/18 | 0/18 (4) | 0/18 (4) | 15/18 |
| cross_dup | 0/22 | 0/22 (6) | 0/22 (4) | 22/22 | 0/22 | 0/22 (5) | 0/22 (5) | 22/22 |
| transpose@20 | 0/19 | 9/19 (6) | 12/19 (4) | 1/19 | 0/19 | 13/19 (6) | 0/19 (5) | 15/19 |
| flip_x@20 | 0/22 | 9/22 (6) | 16/22 (4) | 1/22 | 0/22 | 16/22 (6) | 6/22 (5) | 18/22 |
| flip_y@20 | 0/22 | 10/22 (6) | 12/22 (4) | 1/22 | 0/22 | 16/22 (6) | 4/22 (5) | 18/22 |
| roll1@20 | 0/22 | 0/22 (6) | 0/22 (4) | 1/22 | 0/22 | 12/22 (6) | 0/22 (5) | 12/22 |
| shift_index@20 | 0/22 | 6/22 (6) | 1/22 (4) | 22/22 | 4/22 | 6/22 (6) | 7/22 (5) | 22/22 |


### per dataset

#### ictherm_s2

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | yes | yes |
| transpose | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_x | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y | 002,003,008,009 | 002,003,008,009 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 002,003,009 | 002,003,009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y@20 | 002,003,008 | 002,003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | 002,008,009 | 002,008,009 | yes | yes |

#### ictherm_s3

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008 | 002,003,008 | yes | yes |
| flip_x | 002,003,008 | 002,003,008 | yes | yes |
| flip_y | 002,003,008 | 002,003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 002,003,009 | 002,003,009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y@20 | 002,003,008 | 002,003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | 002,008,009 | 002,008,009 | yes | yes |

#### ictherm_s4

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008 | 002,003,008 | yes | yes |
| flip_x | 002,003,008 | 002,003,008 | yes | yes |
| flip_y | 002,003,008 | 002,003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 002,003 | 002,003 | yes | yes |
| unit_slip | - | - | no | no |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_y@20 | 002,003,008 | 002,003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | 002,008,009 | 002,008,009 | yes | yes |

#### ictherm_s5

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008 | 002,003,008 | yes | yes |
| flip_x | 002,003,008 | 002,003,008 | yes | yes |
| flip_y | 002,003,008 | 002,003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 002,008 | 002,008 | yes | yes |
| shift_index | 002,003 | 002,003 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_y@20 | 002,003,008 | 002,003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | 002 | 002 | yes | yes |

#### thermfm_HS_OC_refine1

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003 | 002,003 | yes | yes |
| flip_x | 003 | 003 | yes | yes |
| flip_y | 002,003 | 002,003 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | - | - | no | no |
| roll3 | - | - | no | no |
| shift_index | 003 | 003 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003 | 002,003 | yes | yes |
| flip_x@20 | 003 | 003 | yes | yes |
| flip_y@20 | 002,003 | 002,003 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | - | - | yes | yes |

#### thermfm_HS_OC_refine2

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003 | 002,003 | yes | yes |
| flip_x | 003 | 003 | yes | yes |
| flip_y | 002,003 | 002,003 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | - | - | no | no |
| roll3 | - | - | no | no |
| shift_index | 003 | 003 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003 | 002,003 | yes | yes |
| flip_x@20 | 003 | 003 | yes | yes |
| flip_y@20 | 002,003 | 002,003 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | - | - | yes | yes |

#### thermfm_HS_QC_refine1

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003,008,009 | 003,008,009 | yes | yes |
| flip_x | 003,008,009 | 003,008,009 | yes | yes |
| flip_y | 003,008,009 | 003,008,009 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 003,008,009 | 003,008,009 | yes | yes |
| unit_slip | - | - | no | no |
| cross_dup | - | - | yes | yes |
| transpose@20 | 003,008 | 003,008 | yes | yes |
| flip_x@20 | 003,008,009 | 003,008,009 | yes | yes |
| flip_y@20 | 003,008,009 | 003,008,009 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | 008,009 | 008,009 | yes | yes |

#### thermfm_HS_QC_refine2

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003,008,009 | 003,008,009 | yes | yes |
| flip_x | 003,008,009 | 003,008,009 | yes | yes |
| flip_y | 003,008,009 | 003,008,009 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 003,008,009 | 003,008,009 | yes | yes |
| unit_slip | - | - | no | no |
| cross_dup | - | - | yes | yes |
| transpose@20 | 003,008 | 003,008 | yes | yes |
| flip_x@20 | 003,008,009 | 003,008,009 | yes | yes |
| flip_y@20 | 003,008,009 | 003,008,009 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | 008,009 | 008,009 | yes | yes |

#### thermfm_HS_SC_refine1

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003 | 003 | yes | yes |
| flip_x | 003,009 | 003,009 | yes | yes |
| flip_y | - | - | no | no |
| roll1 | - | - | no | no |
| roll2 | - | - | no | no |
| roll3 | 009 | 009 | yes | yes |
| shift_index | - | - | no | no |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | - | - | no | no |
| flip_x@20 | - | - | no | no |
| flip_y@20 | - | - | no | no |
| roll1@20 | - | - | no | no |
| shift_index@20 | - | - | yes | yes |

#### thermfm_HS_SC_refine2

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003 | 003 | yes | yes |
| flip_x | 003,009 | 003,009 | yes | yes |
| flip_y | - | - | no | no |
| roll1 | - | - | no | no |
| roll2 | - | - | no | no |
| roll3 | - | - | no | no |
| shift_index | - | - | no | no |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | - | - | no | no |
| flip_x@20 | - | - | no | no |
| flip_y@20 | - | - | no | no |
| roll1@20 | - | - | no | no |
| shift_index@20 | - | - | yes | yes |

#### thermfm_IND_8C

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_x | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y | 002,003,008,009 | 002,003,008,009 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | - | - | no | no |
| roll3 | 002,008 | 002,008 | yes | yes |
| shift_index | 002,003,008,009 | 002,003,008,009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y@20 | 002,003,008,009 | 002,003,008,009 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | 002,003,008,009 | 002,003,008,009 | yes | yes |

#### thermfm_IND_32C

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_x | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y | 002,003,008 | 002,003,008 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 002,008 | 002,008 | yes | yes |
| shift_index | 002,003,009 | 002,003,009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y@20 | 002,003,008,009 | 002,003,008,009 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | 002,009 | 002,009 | yes | yes |

#### 3dice_geometry1

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003,008,009 | 003,008,009 | yes | yes |
| flip_x | 003,008,009 | 003,008,009 | yes | yes |
| flip_y | 003,008,009 | 003,008,009 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 003,008,009 | 003,008,009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 008 | 008 | yes | yes |
| flip_x@20 | 003,008 | 003,008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### 3dice_geometry2a

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_x | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y | 002,003,008 | 002,003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 009 | 009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 003,008 | 003,008 | yes | yes |
| flip_x@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_y@20 | 002,003,008 | 002,003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### 3dice_geometry3

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008 | 002,003,008 | yes | yes |
| flip_x | 003,008,009 | 003,008,009 | yes | yes |
| flip_y | 002,003,008 | 002,003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 002,008 | 002,008 | yes | yes |
| shift_index | 009 | 009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 003,008 | 003,008 | yes | yes |
| flip_y@20 | 002,003,008 | 002,003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### 3dice_geometry4

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| flip_x | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y | 003,008 | 003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 003,009 | 003,009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| flip_x@20 | 003,008 | 003,008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### 3dice_geometry5

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| flip_x | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_y | 008 | 008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 009 | 009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| flip_x@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### 3dice_geometry6

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| flip_x | 002,003,008 | 002,003,008 | yes | yes |
| flip_y | 003,008 | 003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 002,008 | 002,008 | yes | yes |
| shift_index | 009 | 009 | yes | yes |
| unit_slip | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| flip_x@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### darcy_beta1.0

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 008 | 008 | yes | yes |
| flip_x | 008 | 008 | yes | yes |
| flip_y | 008 | 008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | - | - | no | no |
| cross_dup | - | - | yes | yes |
| transpose@20 | 008 | 008 | yes | yes |
| flip_x@20 | 008 | 008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### darcy_beta0.01

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 008 | 008 | yes | yes |
| flip_x | 008 | 008 | yes | yes |
| flip_y | 008 | 008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | - | - | no | no |
| cross_dup | - | - | yes | yes |
| transpose@20 | 008 | 008 | yes | yes |
| flip_x@20 | 008 | 008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### fno_darcy_128

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | - | - | no | no |
| flip_x | - | - | no | no |
| flip_y | - | - | no | no |
| roll1 | - | - | no | no |
| roll2 | - | - | no | no |
| roll3 | - | - | no | no |
| shift_index | - | - | no | no |
| cross_dup | - | - | yes | yes |
| transpose@20 | - | - | no | no |
| flip_x@20 | - | - | no | no |
| flip_y@20 | - | - | no | no |
| roll1@20 | - | - | no | no |
| shift_index@20 | - | - | yes | yes |

#### fno_darcy_native421

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 008 | 008 | yes | yes |
| flip_x | 008 | 008 | yes | yes |
| flip_y | 008 | 008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | - | - | no | no |
| cross_dup | - | - | yes | yes |
| transpose@20 | - | - | no | no |
| flip_x@20 | - | - | no | no |
| flip_y@20 | - | - | no | no |
| roll1@20 | - | - | no | no |
| shift_index@20 | - | - | yes | yes |

## v3 in-scope and diagnostic sets (now seen)

### pooled

**before (v2 code)**


| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 0/9 (2) | 0/9 (2) | 3/9 |
| transpose | 0/9 | 1/9 (8) | 4/9 (4) | 1/9 | 3/9 | 7/9 (2) | 5/9 (4) | 9/9 |
| flip_x | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 3/9 (6) | 9/9 |
| flip_y | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 4/9 (5) | 9/9 |
| roll1 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 3/9 (2) | 1/9 (3) | 5/9 |
| roll2 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 6/9 (2) | 1/9 (4) | 8/9 |
| roll3 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 6/9 (2) | 1/9 (4) | 8/9 |
| shift_index | 0/9 | 1/9 (8) | 4/9 (4) | 1/9 | 3/9 | 1/9 (8) | 4/9 (4) | 9/9 |
| cross_dup | 0/9 | 0/9 (8) | 0/9 (4) | 9/9 | 3/9 | 0/9 (2) | 0/9 (2) | 9/9 |
| transpose@20 | 0/9 | 1/9 (8) | 4/9 (4) | 1/9 | 3/9 | 7/9 (2) | 0/9 (2) | 9/9 |
| flip_x@20 | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 0/9 (2) | 9/9 |
| flip_y@20 | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 0/9 (2) | 9/9 |
| roll1@20 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 3/9 (2) | 0/9 (2) | 5/9 |
| shift_index@20 | 0/9 | 1/9 (8) | 0/9 (4) | 9/9 | 3/9 | 2/9 (2) | 0/9 (2) | 9/9 |

**after (v4 code)**


| fault | F001 | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|---|
| clean | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 0/9 (2) | 0/9 (2) | 3/9 |
| transpose | 0/9 | 1/9 (8) | 4/9 (4) | 1/9 | 3/9 | 7/9 (2) | 5/9 (4) | 9/9 |
| flip_x | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 3/9 (6) | 9/9 |
| flip_y | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 4/9 (5) | 9/9 |
| roll1 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 3/9 (2) | 1/9 (3) | 5/9 |
| roll2 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 6/9 (2) | 1/9 (4) | 8/9 |
| roll3 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 6/9 (2) | 1/9 (4) | 8/9 |
| shift_index | 0/9 | 1/9 (8) | 4/9 (4) | 1/9 | 3/9 | 1/9 (8) | 4/9 (4) | 9/9 |
| cross_dup | 0/9 | 0/9 (8) | 0/9 (4) | 9/9 | 3/9 | 0/9 (2) | 0/9 (2) | 9/9 |
| transpose@20 | 0/9 | 1/9 (8) | 4/9 (4) | 1/9 | 3/9 | 7/9 (2) | 0/9 (2) | 9/9 |
| flip_x@20 | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 0/9 (2) | 9/9 |
| flip_y@20 | 0/9 | 1/9 (8) | 5/9 (4) | 1/9 | 3/9 | 7/9 (2) | 0/9 (2) | 9/9 |
| roll1@20 | 0/9 | 0/9 (8) | 0/9 (4) | 1/9 | 3/9 | 3/9 (2) | 0/9 (2) | 5/9 |
| shift_index@20 | 0/9 | 1/9 (8) | 0/9 (4) | 9/9 | 3/9 | 2/9 (2) | 0/9 (2) | 9/9 |


### per dataset

#### pdegym_poisson_gauss

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 002,003,008,009 | 002,003,008,009 | yes | yes |
| flip_x | 002,003,008 | 002,003,008 | yes | yes |
| flip_y | 002,003,008,009 | 002,003,008,009 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 002,003,009 | 002,003,009 | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_x@20 | 002,003,008 | 002,003,008 | yes | yes |
| flip_y@20 | 002,003,008 | 002,003,008 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | 002 | 002 | yes | yes |

#### cno_poisson

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003,008 | 003,008 | yes | yes |
| flip_x | 003,008 | 003,008 | yes | yes |
| flip_y | 003,008 | 003,008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 003 | 003 | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 003,008 | 003,008 | yes | yes |
| flip_x@20 | 003,008 | 003,008 | yes | yes |
| flip_y@20 | 003,008 | 003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### cno_darcy

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | yes | yes |
| transpose | - | - | yes | yes |
| flip_x | - | - | yes | yes |
| flip_y | - | - | yes | yes |
| roll1 | - | - | yes | yes |
| roll2 | - | - | yes | yes |
| roll3 | - | - | yes | yes |
| shift_index | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | - | - | yes | yes |
| flip_x@20 | - | - | yes | yes |
| flip_y@20 | - | - | yes | yes |
| roll1@20 | - | - | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### elado_a1_test

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 008,009 | 008,009 | yes | yes |
| flip_x | 008,009 | 008,009 | yes | yes |
| flip_y | 008,009 | 008,009 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | - | - | no | no |
| roll3 | - | - | no | no |
| shift_index | 009 | 009 | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 008 | 008 | yes | yes |
| flip_x@20 | 008 | 008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | - | - | yes | yes |

#### dpde_poisson_denorm

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003,008,009 | 003,008,009 | yes | yes |
| flip_x | 003,008,009 | 003,008,009 | yes | yes |
| flip_y | 003,008,009 | 003,008,009 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 003,009 | 003,009 | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 003,008 | 003,008 | yes | yes |
| flip_x@20 | 003,008 | 003,008 | yes | yes |
| flip_y@20 | 003,008 | 003,008 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | 008 | 008 | yes | yes |

#### dpde_darcy_denorm

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | yes | yes |
| transpose | 008 | 008 | yes | yes |
| flip_x | 008 | 008 | yes | yes |
| flip_y | 008 | 008 | yes | yes |
| roll1 | 008 | 008 | yes | yes |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 008 | 008 | yes | yes |
| flip_x@20 | 008 | 008 | yes | yes |
| flip_y@20 | 008 | 008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### ucsd_poisson

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 008,009 | 008,009 | yes | yes |
| flip_x | 003,008 | 003,008 | yes | yes |
| flip_y | 003,008 | 003,008 | yes | yes |
| roll1 | 008,009 | 008,009 | yes | yes |
| roll2 | 008,009 | 008,009 | yes | yes |
| roll3 | 008,009 | 008,009 | yes | yes |
| shift_index | 008 | 008 | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 008 | 008 | yes | yes |
| flip_x@20 | 003,008 | 003,008 | yes | yes |
| flip_y@20 | 003,008 | 003,008 | yes | yes |
| roll1@20 | 008 | 008 | yes | yes |
| shift_index@20 | - | - | yes | yes |

#### dpde_poisson_asreleased

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | no | no |
| transpose | 003,008,009 | 003,008,009 | yes | yes |
| flip_x | 003,008,009 | 003,008,009 | yes | yes |
| flip_y | 003,008,009 | 003,008,009 | yes | yes |
| roll1 | - | - | no | no |
| roll2 | 008 | 008 | yes | yes |
| roll3 | 008 | 008 | yes | yes |
| shift_index | 003,009 | 003,009 | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | 003,008 | 003,008 | yes | yes |
| flip_x@20 | 003,008 | 003,008 | yes | yes |
| flip_y@20 | 003,008 | 003,008 | yes | yes |
| roll1@20 | - | - | no | no |
| shift_index@20 | 008 | 008 | yes | yes |

#### dpde_darcy_asreleased

| fault | before (rules F002..F009 detecting) | after | any-rule before | any-rule after |
|---|---|---|---|---|
| clean | - | - | yes | yes |
| transpose | - | - | yes | yes |
| flip_x | - | - | yes | yes |
| flip_y | - | - | yes | yes |
| roll1 | - | - | yes | yes |
| roll2 | - | - | yes | yes |
| roll3 | - | - | yes | yes |
| shift_index | - | - | yes | yes |
| cross_dup | - | - | yes | yes |
| transpose@20 | - | - | yes | yes |
| flip_x@20 | - | - | yes | yes |
| flip_y@20 | - | - | yes | yes |
| roll1@20 | - | - | yes | yes |
| shift_index@20 | - | - | yes | yes |

## controls (clean files)

Only the cells that changed are listed.

| run | rule | before | after |
|---|---|---|---|
| ucsd_convdiff | F003 | flagged (error) | skipped |
| ucsd_convdiff | F008 | flagged (error) | skipped |
| dpde_helmholtz | F003 | flagged (error) | ok (info) |

False alarms (warning or error from F002/F003/F005/F008/F009) after v4, per control run:

* cno_wave+time_dependent_flag: before none; after none
* cno_wave: before ['F003 error', 'F008 error', 'F009 error']; after ['F003 error', 'F008 error', 'F009 error']
* cno_allen_cahn+time_dependent_flag: before none; after none
* cno_allen_cahn: before none; after none
* cno_navier_stokes+time_dependent_flag: before none; after none
* cno_navier_stokes: before ['F009 error']; after ['F009 error']
* cno_cont_transport+time_dependent_flag: before none; after none
* cno_cont_transport: before ['F003 warning']; after ['F003 warning']
* cno_disc_transport+time_dependent_flag: before none; after none
* cno_disc_transport: before ['F003 warning']; after ['F003 warning']
* pdegym_wave_gauss+time_dependent_flag: before none; after none
* pdegym_wave_gauss: before ['F002 error', 'F003 warning']; after ['F002 error', 'F003 warning']
* elado_helmholtz_e1: before none; after none
* ucsd_convdiff: before ['F003 error', 'F008 error']; after none
* dpde_helmholtz: before ['F003 error']; after none
