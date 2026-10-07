3D-ICE 4.0 segfaults in non-uniform mode (`non-uniform true ;`) whenever the stack has NO `top heat sink`.
The bottom sink is not the cause; it is just the usual way to have no top sink. Reproduced 2026-10-07.

Run (cwd = this directory):  3D-ICE-Emulator crash_bottom_only.stk     -> exit 139 (SIGSEGV)
  crash_bottom_only.stk   1 die, 1 cell, bottom heat sink only, non-uniform        -> SIGSEGV
  crash_no_sink.stk       same, no heat sink at all                                -> SIGSEGV
  ok_top_only.stk         same, top heat sink instead                              -> exit 0
  matrix/                 the 2x3 matrix {uniform, non-uniform} x {top, bottom, both} on a 10x10 mm, two-element die:
      u_top u_bottom u_both nu_top nu_both  -> exit 0 ;  nu_bottom -> exit 139 (writes only the Tmap header line)
Input validity: the bottom-sink block is exactly the grammar of bison/stack_description_parser.y (`bottomsink`, optional),
and the same syntax is used by 3D-ICE's own test/solid/steady/bottomsink.stk (uniform mode), which runs. A stack with no
heat sink at all is also grammatical (`heatsink_opt` may be empty). So this is a crash on VALID input.
Root cause: see asan_backtrace.txt (unguarded `First->Data.TopSink->SinkModel` in update_number_of_cells).
Workaround: in non-uniform mode always declare a top heat sink (a negligible HTC is fine), or use uniform mode.
iceforge N003 ("non-uniform grid with a bottom heat sink") is therefore too narrow: the condition is "no top heat sink".

Related, DISTINCT defect (esl-epfl/3d-ice issue #3, "Bugs that may exist", 2026-05-18): top + bottom sink + non-uniform gives abnormal
temperatures (no crash). Reproduced with issue3/ (a two-die rebuild of example_steady_material.stk, 5000 um cells; the issue's exact
files were not available, only its description):
  Tmax (K)   u_top 320.0 | u_both 310.0 | nu_top 320.0 | nu_both 390620.0 (constant +390300 K offset, field shape intact)
  matrix/ stack (10x10 mm, 1 mm cells): u_both 303.8 | nu_both 223057.5 (constant +223050 K offset) -- exit code 0, no warning.
So non-uniform + BOTH sinks silently adds a huge constant to every temperature; non-uniform + bottom-only (issue3/i_nu_bottom.stk,
matrix/nu_bottom.stk) segfaults. Different symptom and different code path (power_grid.c bottom-sink source term vs thermal_data.c:153).

----- Patch test (2026-10-07; scratch copy of the 3D-ICE 4.0 sources, never committed; apply with `patch -p1` in the source root) -----
topsink_null_guard.patch  : NULL-guards TopSink at thermal_data.c:153 and :672 (the only unguarded TopSink dereferences; the others in
  power_grid.c / thermal_grid.c are already inside `if (... != NULL)`).
  Result: the segfault is gone, but the guard alone is not enough. crash_bottom_only.stk then fails with "SuperLu factorization error 1"
  (singular matrix), and matrix/nu_bottom.stk exits 0 with garbage (-3.8e18 K). crash_no_sink.stk exits 1 the same way (no heat path at
  all: a singular system is the correct outcome there; uniform mode also has no solution).
bottom_sink_nonuniform.patch : (apply AFTER the first) cause of the bottom-sink failure AND of the issue #3 offset, found by reading the code:
  1. system_matrix.c add_solid_column_non_uniform adds the ambient conductance to the matrix diagonal only for *_CONNECTED_TO_AMBIENT
     layers (top sink). There is no case for TDICE_LAYER_*_CONNECTED_TO_PCB, so the bottom sink's conductance never enters the matrix.
  2. power_grid.c update_source_vector DOES add T_ambient * HeatSinkBottomTcs to the right-hand side for the PCB layer. So the RHS has
     a source with no matching sink: bottom-only -> singular / garbage; top + bottom -> a constant offset of about
     T_amb * g_bottom / g_top (g_bottom ~ k A/(h/2) = 14.8 W/K per cell, g_top ~ HTC A = 0.02 W/K: 300 * 740 ~ 2.2e5 K, which is the
     observed +223050 K on matrix/nu_both.stk).
  3. power_grid.c power_grid_fill (the non-uniform bottom Tcs loops) also omits the series ambient HTC: it uses k A/(h/2)
     where the uniform get_conductance_bottom and the top-sink code use 2 k HTC A / (h HTC + 2 k).
  The patch adds the missing diagonal term (same series formula as the top sink) and fixes the Tcs formula.
Verification (patched build, 10x10 mm, two-element die, 1 mm cells; die.flp discretization changed to 4x10 / 6x10 and BOT to 10x10 so that
the non-uniform mesh is exactly the uniform 10x10 grid; 100 cells compared through xyaxis_TOPD.txt):
  top-only    : uniform vs non-uniform max|diff| = 0.000 K (Tmax 307.502)
  bottom-only : max|diff| = 0.000 K (Tmax 307.547)                          [unpatched: segfault / garbage]
  top+bottom  : max|diff| = 0.000 K (Tmax 303.766)                          [unpatched: +223050 K offset]
  crash_bottom_only.stk (1 cell, 1 W, HTC 2e-8 * 1e6 um^2 = 0.02 W/K): 350.068 K, = 300 + 1/0.02 + half-cell drop, as expected.
(Differences are below the 3-decimal Tmap resolution.)  These patches are a diagnosis aid, not an upstream-quality fix: only the
solid/source-at-layer-0 cases are handled, and only steady state was checked (the transient capacity terms are untouched).
