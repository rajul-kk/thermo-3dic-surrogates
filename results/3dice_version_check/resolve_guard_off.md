# Guard-off re-solves and realdata re-run under the pinned 3D-ICE (2026-10-09)

Question: do the three groups left pending in `summary.md` (C unsnapped `data/3d-ice-moving-geometry4`, E TSV-map fixed-placement geometry2a/5/6, F2 the 2026-10-06 realdata diff) agree with what the pinned build (upstream 4953952a1ef6d38807ff307212a6f15e5b2ef935, `wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator`) produces? Run by the user with `ICEFORGE_GUARD=0` via `scripts/guard_off_checks.py`; every re-solved output carries `ice_commit` = 4953952a1ef6d38807ff307212a6f15e5b2ef935 (the stored files have none). Method as in `resolve_rebuilt.md`: `scripts/resolve_from_metadata.py <in> --out <out>` into a scratch directory (nothing written under `data/`), stored `temp` compared with the new `temp`.

## C and E: stored fields vs re-solve

| dataset | file | max abs dT (K) | mean abs dT (K) | max dT / rise |
|---|---|---|---|---|
| C moving-geometry4 (unsnapped) | geometry4_test_041 | 0 | 0 | 0 |
| C moving-geometry4 (unsnapped) | geometry4_train_018 | 0 | 0 | 0 |
| C moving-geometry4 (unsnapped) | geometry4_train_040 | 0 | 0 | 0 |
| E fixed geometry2a | test_001 | 0 | 0 | 0 |
| E fixed geometry2a | train_025 | 0 | 0 | 0 |
| E fixed geometry5 | test_001 | 0 | 0 | 0 |
| E fixed geometry5 | train_050 | 1.007e-3 | 3.332e-7 | 4.592e-5 |
| E fixed geometry6 | test_001 | 1.007e-3 | 1.347e-7 | 1.503e-4 |
| E fixed geometry6 | train_050 | 1.007e-3 | 4.816e-7 | 2.525e-5 |

Peak temperatures printed by the re-solve equal the stored peaks to 0.1 C in all 9 files.

- **C: 3 files, 1 dataset: 3 bit-identical** (max abs dT = 0 K).
- **E: 6 files, 3 datasets: 3 bit-identical (geometry2a x2, geometry5 test_001) and 3 differing by 1.007e-3 K** (geometry5 train_050, geometry6 test_001 and train_050). 1.007e-3 K is one step of the stored 3-decimal text readback, the same quantum as the two differences in `resolve_rebuilt.md`; mean abs dT is 1.3e-7 to 4.8e-7 K. Rebuilt `tsv_frac` and power were already bit-identical to the stored arrays in pure Python for all 140 TSV files; this closes the solve.
- Combined C+E: 9 files, 6 bit-identical, 3 at one print quantum.

## F2: realdata diff re-run (r=2, nz=2)

`scripts/iceforge_realdata_diff.py --r 2 --nz 2 --out logs/guard_off_checks/realdata_rerun` under the pinned build (the script refuses any other commit), then `scripts/compare_realdata_runs.py results/iceforge_realdata <rerun dir>`. 48 cases (geometry4/5/6 x 8 files x orig/snapped), same parameters as the original 2026-10-06 run (r=2, nz=2). A first attempt in the same session used the script's then-default `--r 4` and failed with a MemoryError (4.81 GiB allocation); it produced no results and is not used. The script's default is now `--r 2`, and `scripts/guard_off_checks.py` passes `--r 2 --nz 2`.

| comparison | count | max abs diff |
|---|---|---|
| JSON numbers | 72,972 | 1.297e-10 (geometry4_train_034_orig, `layers[1]/flagged[29]/T_emulated`) |
| field arrays (`*_fields.npz`) | 1,280 | 1.228e-11 (geometry5_train_021_orig, `inst_12_dT`); 0 shape or key mismatches |
| verdict changes | 0 | |
| other non-numeric differences | 48 | all `<case>.json/hotspot` reported "missing" (key absent in the stored file, `null` in the re-run) |

The 48 "missing" entries are a `hotspot` key that the code from the HotSpot third-solver commit (cc67239, 2026-10-07) writes to every case file, as `null` when HotSpot is not requested; the 2026-10-06 run predates that code, so its stored files have no such key, and the re-run did not request HotSpot. Checked over all 48 case pairs: all 48 stored files lack `hotspot`, and all 48 re-run files have `hotspot` equal to `null`. `geometry4_test_041_orig_r4.json` is the old r=4 spot check and was not re-run (listed as present in one directory only).

**F2 closed: the 2026-10-06 realdata results reproduce under the pinned build** (differences at the 1e-10 level, no verdict changes). `results/iceforge_realdata/` is kept as is, as the original record of the 2026-10-06 run; the re-run differs only by the empty `hotspot` placeholder key.

Caveat: this shows the stored data equal what 4953952 produces, not which build originally made them.
