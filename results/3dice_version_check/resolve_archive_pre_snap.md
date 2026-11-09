# Re-solve of archived pre-snap g4-g6 files under the pinned 3D-ICE (2026-10-08)

Question: do the archived pre-snap layout datasets `data/_archive_pre_snap_20260930/3d-ice-layout-geometry{4,5,6}` (geometry6 dated 2026-09-11, build not recorded) equal what the pinned build (upstream 4953952, WSL ~/3d-ice-4.0) produces? Method: 3 files per geometry (test_041, train_018, train_040) re-solved with

`ICEFORGE_GUARD=0 python scripts/resolve_from_metadata.py <src> --out <out>`

(no `--snap`; stored offsets used as-is) into a scratch directory; stored `temp` compared with the new `temp`.

`ICEFORGE_GUARD=0` was required: these layouts are deliberately unsnapped (die edges off the cell grid), and the iceforge guard (S001/S003) refuses to solve unsnapped layouts. Without it nothing runs.

| file | peak (C) | max abs dT (K) | mean abs dT (K) |
|---|---|---|---|
| geometry4_test_041 | 107.6 | 0 | 0 |
| geometry4_train_018 | 37.8 | 0 | 0 |
| geometry4_train_040 | 96.7 | 0 | 0 |
| geometry5_test_041 | 109.0 | 0 | 0 |
| geometry5_train_018 | 41.1 | 0 | 0 |
| geometry5_train_040 | 83.9 | 0 | 0 |
| geometry6_test_041 | 114.2 | 0 | 0 |
| geometry6_train_018 | 43.4 | 0 | 0 |
| geometry6_train_040 | 91.6 | 0 | 0 |

**9 files in 3 datasets: stored and re-solved temperature fields are bit-identical (max abs dT = 0 K).** (A max of 0 implies a mean of 0; the mean was printed for geometry4_test_041 only.)

Caveat: this shows the stored fields equal what the pinned build B produces. It does not show which build originally produced them. Builds A, B and C behave identically on the tested defects (see `summary.md`), so the original build cannot be identified this way. Do not read this as "the archive was made with B".

Pinned commit: 4953952a1ef6d38807ff307212a6f15e5b2ef935, confirmed through `verify_ice_executable(DEFAULT_ICE_EXECUTABLE)` as in `resolve_sample.md`.
