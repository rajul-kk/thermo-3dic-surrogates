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
