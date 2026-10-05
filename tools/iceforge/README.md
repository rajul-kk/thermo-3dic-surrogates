# iceforge

A standalone command-line front end for [3D-ICE](https://github.com/esl-epfl/3d-ice) 4.0. It parses the
hand-written `.stk` / `.flp` / `.lyt` inputs, lints them before a solve, snaps edges to the cell grid,
runs the solver on Linux, WSL or docker, and checks the output afterwards. Only dependency: numpy (`diff` needs the optional extra `iceforge[diff]`).
It does not import anything from the surrounding repository.

```
pip install -e tools/iceforge        # console script: iceforge   (or: python -m iceforge)
pip install -e "tools/iceforge[yaml]"  # optional: pyyaml, for YAML build specs (JSON specs work without it)
```

Units are 3D-ICE's: lengths in um, conductivity in W/um/K, heat-transfer coefficient in W/um^2/K.

## Why

3D-ICE silently accepts inputs that give wrong answers:

1. **Die-edge artefact.** A floorplan element's power goes to cells by *overlap area*, but a `layout`
   (`.lyt`) assigns material by *cell centre*. If a die edge falls mid-cell, the edge cell is heated but
   its centre lies outside the footprint, so it takes the layer's gap material: a heated insulator.
   (6x6 mm, 10 W die, 250 um cells: max rise 11.130 K aligned, 14.562 K misaligned by 0.42 cell, 11.118 K
   after snapping. Repro: `notes/3dice_repro/edge_artifact_3dice.sh`.)
2. **Tmap orientation.** Each line of a Tmap file is one *row*, and rows run along the chip **width**
   (`n_width_cells` lines of `n_length_cells` values). Parsers that assume rows run along the length
   transpose every field on non-square chips.
3. **Windows.** 3D-ICE is Linux-only; WSL needs `/mnt/<drive>/...` paths, which break easily with spaces.

## Commands

| command | what it does |
|---|---|
| `iceforge parse model.stk [--json]` | summary (or JSON) of the stack, materials, grid, dies, floorplans, layouts |
| `iceforge check model.stk [--json] [--strict]` | pre-solve lint, exit 1 on errors (and on warnings with `--strict`) |
| `iceforge snap model.stk -o outdir [--mode nearest\|outward]` | write a gridded copy, report every move |
| `iceforge run model.stk [--backend auto\|native\|wsl\|docker] [--exe PATH] [-o outdir] [--no-check]` | check, solve, parse Tmaps to `.npz`, post-check |
| `iceforge build spec.yaml -o outdir [--snap]` | generate grid-aligned `model.stk` + `.flp` + `.lyt` from a small spec (see Build) |
| `iceforge init [-o spec.yaml] [--json] [--force]` | write an example spec (the repro case) |
| `iceforge diff model.stk [--r 4] [--nz N] [--tol 0.02] [--ice-npz PATH] [--json] [--plot out.png]` | run 3D-ICE and an independently discretised reference solver, compare per Tmap layer, give a verdict (needs `iceforge[diff]`) |
| `iceforge doctor` | which backends work, which 3D-ICE was found |

### Check rules

| code | severity | meaning |
|---|---|---|
| S001 | error | floorplan element edge off the grid and a layout makes the powered edge cell a gap-material cell (heated insulator) |
| S002 | warning | element edge off the grid, no artefact (power smearing only, or no layout cell affected) |
| S003 | error | `.lyt` rectangle edge off the grid (material follows the cell centre, not the drawn shape) |
| S004 | error | cells under a powered element take gap material because the layout does not cover them |
| S005 | error | element outside the chip, or overlapping another element in the same floorplan |
| S006 | warning | chip length/width not an integer multiple of the cell size |
| S007 | warning | unit sanity: k outside [1e-9, 1e-2] W/um/K, HTC outside [1e-12, 1e-3], layer height < 1 or > 1e5 um |
| S008 | error | referenced file missing |
| S009 | warning | total floorplan power <= 0 |

"Gap material" means the layer's own material when it is much less conductive (below half the largest k
used by the layout), the usual footprint-layout convention. Layouts over a uniform-conductivity base (for
example TSV density maps) are not treated as gaps. With `non-uniform true` grids, the off-grid rules
(S001-S004, S006) are skipped and reported as `I001`.

### Snap

`nearest` moves each floorplan element and each layout rectangle to the nearest cell boundary, keeping its
size when that is a whole number of cells (otherwise both edges snap and the size is rounded to whole
cells), falling back to the other neighbouring boundary if the result leaves the chip or overlaps another
element. `outward` grows only the `.lyt` rectangles to whole cells and leaves the power edges alone, which
is the repro's "snapped" case (S002 stays as a note). Inputs are never edited; the copy goes to `outdir`
with `.flp`/`.lyt` files regenerated in canonical form (comments are not preserved).

### Build

`iceforge build` writes inputs that are on the cell grid by construction, instead of linting hand-written ones.
`iceforge init` writes a starting spec. Spec units are SI (converted on output: k x1e-6 to W/um/K,
rho_cp x1e-18, HTC x1e-12); lengths stay in um.

```yaml
chip: {length_um: 10000, width_um: 10000, cell_um: 250}   # or cell_length_um + cell_width_um
heat_sink: {side: bottom, htc_W_m2K: 20000, t_ambient_K: 300}
materials:
  SI:  {k_W_mK: 148, rho_cp_J_m3K: 1.628e+6}   # YAML needs the sign in the exponent: 1.628e+6
  GAP: {k_W_mK: 0.7, rho_cp_J_m3K: 1.628e+6}
layers:                       # bottom -> top
  - {name: SUB, height_um: 300, material: SI}
  - name: SRC                 # a layer with `dies` is a source layer
    height_um: 50
    material: GAP             # material around the dies
    dies:
      - {name: blk, material: SI, x_um: 1500, y_um: 1500, length_um: 6000, width_um: 6000, power_W: 10}
solver: steady                # only steady is supported
```

Each source layer becomes a die with `<layer>.flp` (one element per die) and `<layer>.lyt` (a footprint of each
die's material), and a `Tmap` named `tmap_<layer>.txt`. The stack is emitted top to bottom with all layer
definitions before die definitions, as the 3D-ICE grammar requires. A die edge or size that is not a whole
number of cells is refused with the nearest valid values; `--snap` applies them (same rule as `iceforge snap`)
and prints each move. Dies in one layer may not overlap or leave the chip. Without pyyaml, a `.json` spec
(same schema) still works. Then `iceforge check outdir/model.stk` and `iceforge run outdir/model.stk`.

### Run and outputs

`run` executes 3D-ICE with the working directory set to the stk's directory (so relative paths in the
model resolve as 3D-ICE expects). Backend `auto` tries `$ICE_EXECUTABLE` (a path, or `wsl /path`), a
native `3D-ICE-Emulator` on PATH (non-Windows), WSL (`~/3d-ice-4.0/bin/...`, `~/3d-ice/bin/...`), then the
docker image `iceforge:latest`.

For each `Tmap` directive, `<outdir>/<name>.npz` holds `T` shaped `(length_index, width_index)` in kelvin,
`x_um` (cell-centre coordinates along the length), `y_um` (along the width), `units`, and `t_ref`; a
`<name>.json` sidecar adds shape, cell and chip size, maximum temperature and the maximum rise above the
heat-sink temperature. Post-solve checks: `P001` shape matches the grid, `P002` the hottest cell must lie
under a powered floorplan element (otherwise it warns; this is the artefact signature). Transient runs
skip P002; non-uniform grids skip Tmap parsing.

### Diff: a second opinion from an independent discretisation

`iceforge diff` runs 3D-ICE (or reuses a result with `--ice-npz`, a `.npz` file or the output directory of
`iceforge run`) and `iceforge/refsolve.py`, a steady-state finite-volume solver that shares none of
3D-ICE's discretisation choices, so a disagreement localises a discretisation bug instead of repeating a
shared assumption. (The repository's own validator, `src/validation/fv_solver.py`, copied 3D-ICE's rules and
could never have flagged the die-edge artefact.)

```
pip install -e "tools/iceforge[diff]"      # scipy + pyamg; --plot also needs matplotlib
iceforge diff model.stk                    # exit 0 AGREE, 1 DISAGREE, 2 refused / failed
```

Reference solver:

| | 3D-ICE | reference |
|---|---|---|
| lateral grid | .stk cell | cell / r (default r = 4), and again at 2r |
| vertical | one cell per layer | sub-layers of at most 25 um (min 2 per layer), or `--nz` |
| material of a partial cell | the cell centre decides | exact area fraction of each `.lyt` rectangle |
| power | overlap area, on the coarse grid | exact overlap, uniform over the element and the source thickness |
| mixed-cell k | n/a | arithmetic area-weighted in x, y and z (exact for z: side-by-side columns conduct in parallel; for in-plane flow it is the parallel bound, the r vs 2r check covers the sensitivity) |

Boundary conditions follow 3D-ICE 4.0 (checked in `thermal_grid.c`/`system_matrix.c`): `top heat sink` is a
film on the top face of the first stack element, `bottom heat sink` on the bottom face of the last one
(conductance `2 k h A / (dz h + 2 k)`, half a cell in series with the film); every other face is adiabatic.
Anisotropic `k` (three values, x y z) is honoured. Refused with a message: non-uniform grids,
microchannel layers, pluggable heat sinks, transient solves, models with no power.
Linear solver: pyamg-preconditioned CG if pyamg is installed (about 7 s for 360 k unknowns, 25 s for 1.4 M),
otherwise sparse LU up to 150 k unknowns and Jacobi CG above (slow; install pyamg).

Comparison: the reference at 2r is block-averaged onto the 3D-ICE grid (the mid-plane of the die's source
layer, where 3D-ICE's node sits). A cell is *flagged* when |dT| exceeds both `--tol` (default 2 %) of the
reference's max rise and three times the reference's own r vs 2r change in that cell. Flagged cells are
classified `die-edge` (a floorplan or layout edge crosses the cell interior), `edge-adjacent` (neighbour of
such a cell; the artefact conducts into it) or `interior`.

**3D-ICE's end-layer vertical rule.** A stack-end layer without a heat sink (here the top die layer)
couples to its neighbour through its full height instead of half (`get_conductance_top/bottom` use `k A / h`
for the first/last layer). On aligned models this is the whole difference to the converged reference: the
r = 1, one-cell-per-layer solver with that rule (`refsolve.solve(..., emulate_3dice_ends=True)`) reproduces
3D-ICE's own result to 0.005 K on all three artefact-free repro cases, and the extra rise scales as `q h / 2k`
(0.046 K at h = 50 um, 0.36 K at h = 400 um, measured against 3D-ICE). `diff` therefore calls a flagged cell
*explained* when it matches that emulation (within a quarter of the tolerance) and only unexplained cells
decide the verdict. The explanation is reported, not hidden: the output lists how many cells were explained
and the largest |3D-ICE - emulation|. A thick end layer is a real, if small, 3D-ICE inaccuracy.

Verdicts: `AGREE` (no unexplained flagged cell); `DISAGREE (die-edge)` (at least 70 % of the unexplained cells
are die-edge or edge-adjacent: partial-cell artefact suspected, run `check` and `snap`); `DISAGREE (diffuse)`
(otherwise: model or boundary-condition mismatch, e.g. wrong heat-sink side, layer order, units).

Repro cases (3D-ICE 4.0 through WSL, reference at r = 4 and 8, tolerance 2 %; rises in K above 300 K):

| case | 3D-ICE max rise | reference max rise | max abs dT | reference r vs 2r change (max cell / peak) | verdict |
|---|---|---|---|---|---|
| aligned | 11.130 | 11.044 | 0.298 (all 60 cells over tolerance explained by the vertical rule) | 0.011 / -0.002 | AGREE |
| misaligned | 14.562 | 11.067 | 7.372 | 0.012 / 0.000 | DISAGREE (die-edge): 90 cells, 49 die-edge + 41 edge-adjacent, 0 interior |
| nolayout | 11.027 | 10.970 | 0.090 | 0.000 / 0.000 | AGREE |
| snapped | 11.118 | 11.047 | 0.155 | 0.007 / -0.001 | AGREE |

The reference does not care where the die edge sits (11.044 aligned vs 11.067 misaligned), 3D-ICE does
(11.130 vs 14.562). Run on the project's generated geometry1 (aligned blocks, no layouts, r = 1, nz = 3):
AGREE, max |dT| 0.004 K of 1.21 K rise. Run on geometry5 (three dies, three layouts, r = 1, nz = 4): AGREE
on all three source layers, peak differences 0.007-0.018 K of about 10.5 K (the reference's own r vs 2r change
is 0.05 K there, so r = 1 is coarse for that stack; use the default r = 4 for a real check).

Argmax displacement is reported but is not used in the verdict: on a flat plateau the argmax is arbitrary.

Future work: HotSpot as a third, independently written solver (out of scope for this build).

### Docker

```
docker build -t iceforge:latest tools/iceforge    # builds 3D-ICE from source, installs iceforge
docker run --rm -v "$PWD":/work iceforge:latest check model.stk
```

The docker backend of `run` mounts the stk's directory at `/work` and calls the bundled emulator, so
model files must use relative paths. Verified (Docker Desktop, Windows): `iceforge run --backend docker` on the
repro fixtures gives max rise 11.130 K (aligned) and 14.562 K (misaligned, P002 fires).

### Guarding the repository pipeline

`src/simulators/ice_simulator.py` runs `iceforge.check` on the `stack.stk` it generates (end of
`generate_config_files`) when iceforge is importable. S001, S003, S004 and S005 raise `RuntimeError`; other
findings are logged as warnings. `ICEFORGE_GUARD=0` disables it; without iceforge installed it is skipped.
On Windows, iceforge maps the `/mnt/<drive>/...` paths of WSL-targeted stacks back to drive paths to read them.

## Tests

```
python -m pytest tools/iceforge/tests -q
```

Tests marked `solver` need a working backend and are skipped otherwise. Fixtures in
`tests/fixtures/3dice_examples` are copies of GPL-3.0-or-later example files from the 3D-ICE source tree.
