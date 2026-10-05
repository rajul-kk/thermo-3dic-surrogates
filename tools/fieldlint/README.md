# fieldlint

A linter for PDE-surrogate benchmark datasets whose targets are steady diffusion / heat-type fields,
`-div(k grad u) = s` with `u` held at (or relaxed towards) a boundary level. It checks the things that
train without error but make the benchmark wrong: misaligned or transposed inputs, a maximum in the wrong
place, train/test duplicates, unit slips, and benchmarks that a linear model already solves.

It is generalised from the 3D-ICE-specific linter of the thermal-surrogate benchmark repository it grew out of
(rules L001-L011 there) and has no dependency on that repository. Core: numpy only.

```
pip install -e tools/fieldlint              # numpy only
pip install -e "tools/fieldlint[all]"       # + h5py (HDF5, MATLAB v7.3), scipy (MATLAB v5), pyyaml (YAML configs)
```

## Quick start

```
# arrays in an .npz, layout (N, H, W)
fieldlint data.npz --u temp --source power --k kmap --layout NHW --units K

# channel-stacked inputs: source is channel 0 of an (N, C, H, W) array
fieldlint data.npz --u temp --layout NHW --source power --source-layout NCHW --source-channel 0

# ready-made adapters
fieldlint path/to/level3_steady --preset ictherm
fieldlint path/to/HS_OC_refine1 --preset thermfm
fieldlint path/to/3d-ice-npz-dir --preset 3dice

# own mapping file (JSON or YAML), JSON report, a subset of rules
fieldlint path --config mapping.yaml --rules F001,F002,F003 --json report.json
fieldlint --list-rules                     # every rule with its justification and when it does not apply
```

Python:

```python
from fieldlint import dataset_from_arrays, run
rep = run(dataset_from_arrays(u, source, k, units='K', splits=split_names))   # u, source: (N, [Z,] Y, X)
print(rep.text()); rep.exit_code
```

Exit codes: `0` clean (info only), `1` warnings, `2` errors, `3` could not load / bad usage.
Per-sample rules look at an evenly spaced subsample (`--max-samples`, default 400, `0` = all);
`--head N` uses the first N samples in file order.

## Mapping config

```yaml
format: auto            # auto | npz | npy | mat | h5 | 3dice
units: K                # units of u: K, C, or '' (a rise / unknown)
ambient: 300.0          # optional boundary level of u
spacing: [1.0, 1.0]     # optional, same axis order as u
u:      {key: temp,  layout: NHW}
source: {key: power, layout: NHW}
k:      {key: kmap,  layout: NHW, optional: true}
dirichlet_mask: {key: fixed, layout: NHW}      # True where u is held fixed
flux_out: {key: q_out, layout: N}              # total boundary heat flux per sample (enables F007)
splits: from_filename   # none | from_filename | {rule: ictherm} | {train: [0, 0.8], test: [0.8, 1]}
```

A field may add `file:` (relative to the dataset directory), `channel:` (index along `C`) and `transpose: true`
(swap in-plane axes after loading, to test a suspected fix). Layout letters describe the array as the reader returns it:
`N` sample, `C` channel, `Z`/`D`/`L` depth, `H`/`Y` rows, `W`/`X` columns; omit `N` for one-sample-per-file
directories. MATLAB v7.3 files are read with h5py, whose axis order is the reverse of MATLAB's. Single-layer `Z`
axes are squeezed, so fields are `(Y, X)` or `(Z, Y, X)`.

Presets: `ictherm` (IC-ThermBench S2-S5 as released: `input.mat` channel 0 power, channel 3 conductivity when present,
`output.mat` temperature, the benchmark's own index split), `thermfm` (Therm-FM steady HotSpot sets, channel 0 power),
`3dice` (this project's `.npz` files: scattered node coordinates are scattered onto the `(z, y, x)` tensor grid; split
from the file name).

## Rules

| Code | Name | Checks | Severity |
|------|------|--------|----------|
| F001 | integrity | fields finite; u, source, k, masks share one grid; shapes consistent across samples; every sample loads | error (warning for mixed grid sizes) |
| F002 | max-principle | with source >= 0 and no sinks, the maximum of u lies on the source (within 1 cell) or on a Dirichlet node; reports the fraction of violating samples | warning < 20% of samples, error >= 20% |
| F003 | orientation | corr(Gaussian-smoothed source, u) as stored vs transposed vs flipped (x, y, both); fires if an alternative wins by a margin on both the plain and the band-passed correlation | warning >= 10% of samples, error >= 50% |
| F004 | duplicates | identical / near-identical u (exact hash, then max abs difference <= 1e-5 of range) within and across splits | error across splits, warning within |
| F005 | range-units | constant fields; K/C range plausibility; u below ambient with non-negative source; k <= 0; zero source | error / warning |
| F006 | linearity-probe | held-out R^2 of a ridge fit from the source field to u (informational) | info only |
| F007 | energy-balance | heat out / heat in = 1 within 1%; only when `flux_out` is supplied | error |

Each rule's docstring (`fieldlint --list-rules`) gives its physical justification and the cases where it does not apply.
Rules never guess: a rule that cannot run (no source field, every cell powered, no flux data) reports `skipped`.

## Example: IC-ThermBench S2-S5

The public IC-ThermBench release stores the spatial input channels (chiplet power, conductivity) with the two in-plane
axes swapped relative to the temperature field. `fieldlint ... --preset ictherm --head 400` on the first 400 samples of
each scope measures:

| scope | corr(smoothed power, T) as stored | transposed | samples violating F002 as stored (0-cell / 1-cell neighbourhood) | after `--transpose-inputs` |
|-------|------|------|------|------|
| S2 (level2) | 0.41 | 0.78 | 84 / 57 of 400 | 0 |
| S3 (level3) | 0.33 | 0.76 | 108 / 69 | 0 |
| S4 (level4) | 0.35 | 0.77 | 100 / 66 | not run |
| S5 (level5) | 0.42 | 0.70 | 147 / 59 | 0 |

F003 reports `transpose` as the better orientation in 81-94% of samples per scope and exits with code 2.
The F002 counts depend on the tolerance and on the neighbourhood radius (`--neighbourhood`); an earlier hand-written check
with a fixed 0.01 K tolerance and no neighbourhood counted 101-154 of 400. Reading the data with the in-plane axes of the
power and conductivity channels swapped (`--transpose-inputs`) removes every F002 violation and restores a correlation of
0.70-0.78 in the stored orientation. On the same scopes F006 gives a held-out ridge R^2 of about 0.46 (S3), i.e. not near-linear
while conductivity varies.

Public Therm-FM steady sets (HS_SC/QC/OC refine1+2, IND_8C, IND_32C; 300 samples each) lint clean for F001-F005.
Two HS_SC sets receive an info note: their power maps are everywhere positive and dominated by wide low-contrast regions, so
the plain smoothed correlation is negative as stored, but the band-passed (hot-spot) correlation is positive as stored
and no orientation fault is claimed. F006 on Therm-FM gives R^2 of about 1.000: those sets are solved by a linear map from power.

## Limitations

* Scope is steady, non-negative-source diffusion-type problems. Time-dependent, advective or sink-dominated problems
  violate the premises of F002/F003/F006/F007; the rules say so in their docstrings.
* F003 can only test orientations of the source; k is assumed to share the source's orientation.
* F004 finds copies, not augmentations or perturbations; leakage is only testable when split labels are available.
* F006 is a screening heuristic on a block-averaged, subsampled problem, not a certificate of linearity.
* The `3dice` adapter reads pickled metadata (`allow_pickle`): only use it on files you trust.

## Tests

```
pip install -e "tools/fieldlint[test]"
python -m pytest tools/fieldlint/tests -q
```

Synthetic tests solve a small 2D Poisson problem with numpy to produce physically valid samples, then transpose or flip the
inputs, inject an off-source maximum, and duplicate samples across splits. Real-data tests (marked `realdata`) read
IC-ThermBench, Therm-FM and 3D-ICE files read-only and skip when absent; set `FIELDLINT_DATA_ROOT` to the `data/` directory.
