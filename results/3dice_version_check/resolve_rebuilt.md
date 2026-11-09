# Re-solve of previously unrebuildable datasets under the pinned 3D-ICE (2026-10-08)

Question: with the new rebuild support, do the files that `resolve_sample.md` could not compare equal what the pinned build (upstream 4953952a1ef6d38807ff307212a6f15e5b2ef935) produces? Method: 2-3 files per dataset re-solved into a scratch directory (nothing written under `data/`), stored `temp` compared with the new `temp`. Fixed-placement and microchannel files with `scripts/resolve_from_metadata.py <in> --out <out>` (no `--snap`); leakage and throttle files with `scripts/resolve_feedback_pilot.py` (`--dry-run` generator cross-check first: all 5 OK). Rise = max(stored T) - ambient. Every output carries `ice_commit` = 4953952a1ef6d38807ff307212a6f15e5b2ef935; the stored files have none.

| dataset | file | pattern | max abs dT (K) | mean abs dT (K) | max dT / rise | same shape / coords |
|---|---|---|---|---|---|---|
| fixed geometry1 | test_001 | uniform | 0 | 0 | 0 | yes / yes |
| fixed geometry1 | train_040 | checkerboard | 0 | 0 | 0 | yes / yes |
| fixed geometry3 | test_001 | uniform | 0 | 0 | 0 | yes / yes |
| fixed geometry3 | train_040 | checkerboard | 1.007e-3 | 2.0e-7 | 1.4e-5 | yes / yes |
| fixed geometry4 | test_001 | uniform | 1.007e-3 | 3.8e-7 | 1.3e-4 | yes / yes |
| fixed geometry4 | train_040 | split_chiplet_b_hot | 0 | 0 | 0 | yes / yes |
| microchannel pilot | train_000 | uniform | 0 | 0 | 0 | yes / yes |
| microchannel pilot | train_011 | uniform | 0 | 0 | 0 | yes / yes |
| leakage pilot geometry1 | test_017 | hotspot | 0 | 0 | 0 | yes / yes |
| leakage pilot geometry1 | train_003 | checkerboard | 0 | 0 | 0 | yes / yes |
| leakage pilot geometry1 | train_030 | gradient | 0 | 0 | 0 | yes / yes |
| throttle pilot geometry1 | test_002 | hotspot | 0 | 0 | 0 | yes / yes |
| throttle pilot geometry1 | train_005 | dual_hotspot | 0 | 0 | 0 | yes / yes |

Feedback loops: iteration counts and multipliers/derate factors equal the stored ones exactly (leakage test_017 8/8, x66.669; train_003 7/7, x0.7545; train_030 2/2, x5.290 runaway in both; throttle test_002 2/2, derate 0.8390; train_005 2/2, derate 0.7186). Rebuilt `power` equals stored `power` exactly (max abs diff 0) in all 13 files, so the sub-block power maps are recovered correctly (this was the failure in `resolve_sample.md`, where differences were 1.6-13 K).

**13 files, 4 datasets: 11 bit-identical; 2 differ by one last-digit step.** geometry3 train_040 differs at 22 of 110000 nodes and geometry4 test_001 at 21 of 56000, every difference being 0.00098 or 0.00101 K, i.e. one unit of the third decimal of the stored text-readback temperatures (stored 327.791 vs new 327.790). Max 1.007e-3 K is at the 1e-3 K investigation threshold, but the pattern (all differences are exactly one print quantum, at ~0.02-0.04% of nodes, power and grid identical) is a rounding flip at a print boundary, not a wrong rebuild. Not attributed further (no pre-pinned binary was re-run).

## Not compared
- **data/3d-ice/geometry2a, geometry5, geometry6 (TSV)**: re-solving needs `ICEFORGE_GUARD=0` (S003 refuses the 32x32 TSV tiles that fall mid-cell); reserved for the user. Pending.
