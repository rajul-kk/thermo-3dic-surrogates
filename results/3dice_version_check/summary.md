# 3D-ICE defect re-verification across versions (2026-10-08)

Question: is every 3D-ICE defect we report real in each relevant 3D-ICE build, rather than fixed upstream or specific to an old build?

## Builds
| id | path | source | built | notes |
|---|---|---|---|---|
| A | `~/3d-ice` | 39ed2a7 (2026-05-18), `git describe` = 4.0-5-g39ed2a7 | 2026-05-29 | working-tree patches: `LUIndex_t` long->int, printf `%ld`->`%d` in system_matrix.c, `-Werror` removed |
| B | `~/3d-ice-4.0` | 4953952 (2026-07-02) = upstream HEAD (`git fetch` and `ls-remote origin HEAD` re-checked today: unchanged) | cloned 2026-08-04, rebuilt 2026-08-18 | shallow (depth 1) clone; same `LUIndex_t` int and no `-Werror` patches |
| C | `~/3d-ice-tag40` (scratch) | tag `4.0` = e0bb6850c5e446363e26936586d625270c87f224 (2025-12-01) | 2026-10-08 | same patches as A/B; SuperLU_MT taken from A's build (the tag's zip is byte-identical to A's, md5 0330b733...; B's zip differs, see below) |

## A -> B source diff (`git log --oneline 39ed2a7..4953952 -- sources/ include/ bison/ flex/`)
One commit touches solver code: **4953952 "update local server mode"**, `sources/thermal_data.c` only (adds `set_openblas_threads()`, i.e. `openblas_set_num_threads(1)` before `set_parallel_cores`, plus a trailing newline). The other six commits in the range (2021ff9 SuperLU_MT zip "U matrix storage problem", 0eaf90c / 766c314 example .flp files, f1855a8 / d69cc99 / 08d0e54 SoftHier STK generator) touch only the SuperLU_MT zip, `makefile.def`, `bin/` examples and a generator script. **Nothing touches `.lyt` layout handling, non-uniform mode, or heat sinks** (bison/, flex/ and include/ are unchanged A -> B).
For completeness, tag 4.0 -> A (5 commits) changes `include/types.h`, `sources/inspection_point.c`, `network_socket.c`, `output.c` (server / inspection-point features, Tflp min/max); again no layout, non-uniform or heat-sink code.

## Results (identical on A, B and C)
Tmax = maximum of the written Tmap in K; exit codes are the emulator's. Fixtures from `tools/iceforge/tests/fixtures/`, CRLF->LF, fresh scratch dir per case.

| # | Defect | Case | A | B | C |
|---|---|---|---|---|---|
| 1 | `.lyt` die-edge artefact (uniform) | misaligned (die at 1394.7 um) | 314.562 K (rise 14.562), exit 0 | 314.562 | 314.562 |
| 1 | | aligned (1500 um) | 311.130 (rise 11.130) | 311.130 | 311.130 |
| 1 | | snapped layout | 311.118 | 311.118 | 311.118 |
| 1 | | no layout (uniform Si) | 311.027 | 311.027 | 311.027 |
| 2 | non-uniform ignores `.lyt` | uniform, lyt vs gap max abs diff | 5.522 K | 5.522 K | 5.522 K |
| 2 | | non-uniform, lyt vs gap max abs diff | 0 (n=180, Tmax 314.988) | 0 | 0 |
| 3 | non-uniform segfault, no top sink | crash_bottom_only.stk | exit 139 | exit 139 | exit 139 |
| 3 | | crash_no_sink.stk | exit 139 | exit 139 | exit 139 |
| 3 | | matrix/nu_bottom.stk | exit 139 | exit 139 | exit 139 |
| 3 | | issue3/i_nu_bottom.stk | exit 139 | exit 139 | exit 139 |
| 3 | | ok_top_only.stk (control) | exit 0, 350.068 K | exit 0, 350.068 | exit 0, 350.068 |
| 4 | non-uniform top+bottom sink offset (upstream #3) | matrix/nu_both.stk | 223057.507 K | 223057.507 | 223057.507 |
| 4 | | matrix/u_both.stk (uniform control) | 303.766 | 303.766 | 303.766 |
| 4 | | matrix/nu_top, u_top, u_bottom | 307.507 / 307.502 / 307.547 | same | same |
| 4 | | issue3/i_nu_both.stk (top / bottom die) | 390620.015 / 391220.031 | same | same |
| 4 | | issue3/i_u_both.stk; i_nu_top.stk | 310.007; 320.015 | same | same |

Verdict: all four defects **reproduced on A, B and C**. No "feature absent" outcomes: even the 4.0 release itself parses `.lyt`, `non-uniform true` and `bottom heat sink`. (3D-ICE 3.x was not tested; it has neither `.lyt` nor non-uniform mode.) Nothing failed to reproduce on any version. The defects are in the 4.0 release, were not fixed by any commit up to upstream HEAD, and are not specific to our build. Caveat: all three builds carry the same local portability patches (`LUIndex_t` int, no `-Werror`); none touches the affected code. The SuperLU_MT zip differs in B (upstream 2021ff9); results agree to the printed precision, so it does not matter here.

## Data provenance
- The wrapper `src/simulators/ice_simulator.py` takes the executable as an argument (`--ice-executable`) and hard-codes no path. `scripts/resolve_from_metadata.py:23` sets `ICE_EXE = wsl /home/rajul/3d-ice-4.0/bin/3D-ICE-Emulator` (B). `app/main.py:26`, `scripts/regen_v4_final.ps1:11`, `scripts/regen_pilots_v5.sh:6` and `warpage/scripts/edge_artifact_3dice.sh:11` ("the build the benchmark data was made with") also point to B. The `.npz` metadata records no executable or version.
- Timeline from WSL file times and logs: A (`~/3d-ice`) was built 2026-05-29; archived early runs (2026-06-07 / 06-10, `data/_archive/3d-ice_pre_tdp_regime`, logs show `/home/rajul/3d-ice/bin/...`) and the 2026-08-01 `regen_v2` pass used A. B (`~/3d-ice-4.0`) was cloned and built 2026-08-04 (rebuilt 2026-08-18 for `numofcores`). The generation scripts from `regen_v4_final` (2026-08-05) onward point to B, but the build actually used was not recorded at the time, so that is not evidence.
- **What is evidenced (updated 2026-10-08).** Under the pinned build B (4953952) the stored fields were re-solved and compared with the stored `temp`. Counting distinct stored files (the 8 snapped files that appear in both `resolve_sample` and `snap_reproduction` are counted once; the 6 `invalid_rebuild` fixed-placement rows of the first sample are excluded and replaced by the rebuilt re-solve): **56 files in 24 datasets, 54 bit-identical (max abs dT = 0 K) and 2 differing by 1.007e-3 K**, one step of the stored 3-decimal readback at about 20 nodes with power and grid identical. There are 64 re-solve comparisons in all (30 + 9 + 12 + 13).

| dataset group | files re-solved | result | record file |
|---|---|---|---|
| 15 live datasets (geometry7 material-sweep and pilot, interface x4, layout geometry1/2a/3/4/5/6/7, moving-geometry1, moving-geometry4-snap), first and last file each | 30 | all bit-identical | `resolve_sample.md` |
| archived pre-snap `_archive_pre_snap_20260930/3d-ice-layout-geometry{4,5,6}` (run with `ICEFORGE_GUARD=0`) | 9 | all bit-identical | `resolve_archive_pre_snap.md` |
| `--snap` reproduction of snapped layout geometry4-6 and moving-geometry4-snap (commit 33e2f11); temp, coords, power, layer, tsv_frac and all metadata keys compared | 12 | all bit-identical | `snap_reproduction.md` |
| fixed-placement geometry1/3/4 (sub-block power maps recovered) | 6 | 4 bit-identical; geometry3 train_040 and geometry4 test_001 differ by 1.007e-3 K (one print quantum) | `resolve_rebuilt.md` |
| microchannel pilot | 2 | bit-identical | `resolve_rebuilt.md` |
| leakage pilot (geometry1) | 3 | bit-identical; iteration counts and multipliers equal | `resolve_rebuilt.md` |
| throttle pilot (geometry1) | 2 | bit-identical; iteration counts and derate factors equal | `resolve_rebuilt.md` |

Caveat: this shows the stored fields equal what B produces, not which build originally produced them; the original build cannot be identified this way. Outputs of `resolve_from_metadata.py` and the pilot re-solvers now carry `ice_commit` (and `ice_executable`), as the generators do; stored files written before 2026-10-08 have none.
- **Provenance audit flags.** F1 (archived pre-snap layout data of unknown build): **closed** by the 9-file archive re-solve (with the caveat above). F3 (the 2026-09-30 snap re-solve used an uncommitted script): **closed** for the 12-file sample; the committed `--snap` reproduces the stored snapped data bit-identically. F2 (the build used for the 2026-10-06 realdata run is unrecorded): **pending**; the realdata script now verifies the pin (014b894) and the re-run is ready.
- **Pending the user's guard-off run** (needs `ICEFORGE_GUARD=0`, which the permission system reserves for the user; one `guard_off_checks.py` script covers all three): (F2) re-run the 2026-10-06 realdata diff under the pin; (C) `data/3d-ice-moving-geometry4` unsnapped re-solve; (E) TSV-map fixed-placement `geometry2a/5/6` re-solve (rebuilt `tsv_frac` and power are already bit-identical to the stored arrays in pure Python for all 140 files; only the solve is pending). Until these run, those three groups are unverified.
