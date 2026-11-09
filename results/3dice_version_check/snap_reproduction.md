# Does the committed `--snap` option reproduce the 2026-09-30 snapped datasets? (2026-10-08)

Question (flag F3): the snapped layout datasets `data/3d-ice-layout-geometry{4,5,6}` and `data/3d-ice-moving-geometry4-snap` were produced on 2026-09-30 by an uncommitted script. Does the committed `--snap` option of `scripts/resolve_from_metadata.py` give the same files? Method: 3 unsnapped source files per dataset (test_041, train_018, train_040; layout sources from `data/_archive_pre_snap_20260930/3d-ice-layout-geometry{4,5,6}`, moving source from `data/3d-ice-moving-geometry4`) re-solved with

`python scripts/resolve_from_metadata.py <src> --snap --out <out>`

into a scratch directory, one job at a time, under the pinned build (upstream 4953952, WSL ~/3d-ice-4.0). `ICEFORGE_GUARD` was not set or changed and the guard refused nothing. Each output was compared with the same-named file in the stored snapped dataset: array keys, `coords`, `temp`, `power`, `power_nominal`, `layer`, `tsv_frac` (max abs difference) and every metadata key.

| file | stored peak (C) | max abs dT (K) | max abs d(power) | metadata keys differing |
|---|---|---|---|---|
| 3d-ice-layout-geometry4/geometry4_test_041 | 107.5 | 0 | 0 | 0 |
| 3d-ice-layout-geometry4/geometry4_train_018 | 37.3 | 0 | 0 | 0 |
| 3d-ice-layout-geometry4/geometry4_train_040 | 97.1 | 0 | 0 | 0 |
| 3d-ice-layout-geometry5/geometry5_test_041 | 108.9 | 0 | 0 | 0 |
| 3d-ice-layout-geometry5/geometry5_train_018 | 37.7 | 0 | 0 | 0 |
| 3d-ice-layout-geometry5/geometry5_train_040 | 83.8 | 0 | 0 | 0 |
| 3d-ice-layout-geometry6/geometry6_test_041 | 114.2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry6/geometry6_train_018 | 37.9 | 0 | 0 | 0 |
| 3d-ice-layout-geometry6/geometry6_train_040 | 91.5 | 0 | 0 | 0 |
| 3d-ice-moving-geometry4-snap/geometry4_test_041 | 107.8 | 0 | 0 | 0 |
| 3d-ice-moving-geometry4-snap/geometry4_train_018 | 38.0 | 0 | 0 | 0 |
| 3d-ice-moving-geometry4-snap/geometry4_train_040 | 95.4 | 0 | 0 | 0 |

**12 files in 4 datasets: temperature, coordinates, power, layer and tsv_frac arrays are bit-identical (max abs difference 0), the array key sets match, and all metadata keys match exactly, including the snapped offsets, `orig_placement_*`, `placement_snapped_2026_09_30` and block positions.** F3 is closed for this sample: the committed `--snap` code reproduces the stored snapped data.

Caveats: 3 files per dataset, not the full 45 (or more); the sample is the same one used in the other re-solve checks. As in `resolve_archive_pre_snap.md`, this shows agreement with the pinned build, not which build originally produced the stored files. The moving-geometry4 source (`data/3d-ice-moving-geometry4`) is the unsnapped counterpart and re-solved to different temperatures than it stores (peaks e.g. 107.8 vs 113.4 C for test_041), as expected, because snapping moves the dies.
