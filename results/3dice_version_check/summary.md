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
- **What is evidenced:** 30 sampled shipped files in 15 datasets (`resolve_sample.md`) plus 9 archived pre-snap geometry4-6 files (`resolve_archive_pre_snap.md`, run with `ICEFORGE_GUARD=0` because those layouts are deliberately unsnapped) re-solve bit-identically under B (max abs dT = 0 K). Because A, B and C give identical numbers on every tested defect, the choice does not affect any defect claim. Caveat: this shows the stored fields equal what B produces, not which build originally produced them; the original build cannot be identified this way.
- **Provenance audit flags.** F1 (archived pre-snap layout data of unknown build, on which the old arm of the projection test, the g6 shelf row of report 9.15 and `results/v5/layout_cv_geometry6.json` depend): **resolved** in the sense above (9 of the archived files re-solve bit-identically under B), with the caveat above. F2 (the build used for the 2026-10-06 realdata run is unrecorded): **remains**. F3 (the 2026-09-30 snap re-solve used an uncommitted script): **remains**.
- **Not verifiable (cannot be rebuilt by `resolve_from_metadata.py`, so unverified):** fixed-placement geometry1/3/4 (`invalid_rebuild`: sub-block power maps), geometry2a/5/6 (nonzero `tsv_frac`), leakage and throttle pilots (feedback scenarios), microchannel pilot (unknown geometry) and moving-geometry4 unsnapped (`3d-ice-moving-geometry4`; the archive re-solve covered only `_archive_pre_snap_20260930/3d-ice-layout-geometry{4,5,6}`, not this dataset).
