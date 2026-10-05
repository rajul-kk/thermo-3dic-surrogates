# iceforge

A standalone command-line front end for [3D-ICE](https://github.com/esl-epfl/3d-ice) 4.0. It parses the
hand-written `.stk` / `.flp` / `.lyt` inputs, lints them before a solve, snaps edges to the cell grid,
runs the solver on Linux, WSL or docker, and checks the output afterwards. Only dependency: numpy.
It does not import anything from the surrounding repository.

```
pip install -e tools/iceforge        # console script: iceforge   (or: python -m iceforge)
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

### Docker

```
docker build -t iceforge:latest tools/iceforge    # builds 3D-ICE from source, installs iceforge
docker run --rm -v "$PWD":/work iceforge:latest check model.stk
```

The docker backend of `run` mounts the stk's directory at `/work` and calls the bundled emulator, so
model files must use relative paths.

## Tests

```
python -m pytest tools/iceforge/tests -q
```

Tests marked `solver` need a working backend and are skipped otherwise. Fixtures in
`tests/fixtures/3dice_examples` are copies of GPL-3.0-or-later example files from the 3D-ICE source tree.
