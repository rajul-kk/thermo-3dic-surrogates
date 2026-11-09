Second independent case of "3D-ICE 4.0 ignores `layout` in non-uniform mode" (2026-10-07).
10x10 mm chip, die source layer SRC = GAP material (k 7e-7) with strip.lyt = SI rectangle (7000,1000,2000,8000) um,
two floorplan elements P (6000 um wide, 3 W) and Q (4000 um wide, 7 W), top heat sink, steady state.
  u_lyt / u_gap    uniform grid (500 um cells), with / without the layout      : max|dT| = 5.522 K  (layout honoured)
  nu_lyt / nu_gap  non-uniform (P 10x10, Q 8x10 cells, layer 20x20), with / without : max|dT| = 0 (byte-identical Tmap)
Run each .stk with cwd = this directory; *_t.txt are the files 3D-ICE wrote. `awk -v F1=a -v F2=b -f cmp.awk` prints max|diff|.
Different from the first case: other strip position/size, other power split, other discretisation.
