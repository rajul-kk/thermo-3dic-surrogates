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
fieldlint path/to/darcy_beta1.0_n1000.npz --preset darcy

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
from the file name), `darcy` (PDEBench 2D Darcy `.npz`: flattened `X` = k and `Y` = u reshaped to 128 x 128, the uniform
source beta read from the file name). Field specs also accept `reshape: [H, W]` for flattened arrays, and the config accepts
`source_uniform` (a number, or `{filename_regex: ...}`) for a constant source.

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
| F008 | operator-residual | fit of s ~ a(-div k grad u) + b(u - c) (conservative 5-point stencil, harmonic face k) per z-layer; R^2 as stored vs transposed / flipped / shifted 1-3 cells; also works when the source is uniform and only k varies | warning >= 10% of samples, error >= 50% |
| F009 | pairing | M = 32 samples: F008's R^2 for every (inputs_i, u_j); the diagonal must be the best match of its row | warning >= 25% of rows, error >= 50% |

F008 and F009 skip (with the reason) when the operator model explains a median R^2 below 0.3 in every orientation, when
neither the source nor k varies, and for fields that are not 2D or z-stacked 2D. The face conductivity is a convention
(`Options.op_face_mean`: `harmonic`, the default, or `arithmetic`); see the Darcy result below.

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

## Measured detection and false-positive rates

`bench/run_bench.py` injects faults into clean copies of real datasets and records which rules fire (warning or error;
info does not count). Faults: transpose / flip_x / flip_y of source and k, a roll of 1-3 cells, an off-by-one sample index
(u of sample i replaced by u of sample i+1), a C-to-K unit slip (u - 273.15, still declared kelvin; K datasets only), the last
10% of the samples replaced by copies of train samples (splits assigned by position on every dataset), and "20% of samples" =
the same fault on a random 20% of the samples. Cell = detected / datasets; `(n)` = rule skipped on n of them (counted as a
miss). Data are read read-only (IC-ThermBench is read with its inputs transposed so that it is clean; first 400 samples).
Thresholds were frozen on the tuning datasets (`bench/results/frozen_thresholds.json`, 2026-10-06T16:16, with SHA-256 of the
code) before the held-out sets were run, and the held-out stage refuses to run if the code has changed. Full tables per rule
and the raw per-dataset results: `bench/results/`.

**Tuning datasets (18: IC-ThermBench S2-S5, 8 Therm-FM steady sets, 6 of this project's 3D-ICE geometries).** The rules were
developed and the thresholds chosen on these, so the rates are optimistic.

| fault | F002 | F003 | F004 | F005 | F008 | F009 | any rule |
|---|---|---|---|---|---|---|---|
| clean (false positives) | 0/18 (2) | 0/18 | 1/18 | 0/18 | 0/18 (5) | 0/18 (5) | 1/18 |
| transpose (15 square grids) | 10/15 (2) | 15/15 | 1/15 | 0/15 | 10/15 (5) | 7/15 (8) | 15/15 |
| flip_x | 10/18 (2) | 18/18 | 1/18 | 0/18 | 13/18 (5) | 12/18 (6) | 18/18 |
| flip_y | 10/18 (2) | 15/18 | 1/18 | 0/18 | 13/18 (5) | 5/18 (10) | 16/18 |
| roll 1 cell | 0/18 (2) | 0/18 | 1/18 | 0/18 | 8/18 (5) | 0/18 (5) | 8/18 |
| roll 2 cells | 0/18 (2) | 0/18 | 1/18 | 0/18 | 12/18 (5) | 0/18 (6) | 12/18 |
| roll 3 cells | 5/18 (2) | 0/18 | 1/18 | 0/18 | 13/18 (5) | 1/18 (5) | 15/18 |
| sample index shift by 1 | 6/18 (2) | 12/18 | 1/18 | 6/18 | 4/18 (12) | 12/18 (6) | 16/18 |
| C/K unit slip | 0/18 (8) | 0/18 | 1/18 | 15/18 | 0/18 (5) | 0/18 (5) | 15/18 |
| cross-split duplicates | 0/18 (2) | 0/18 | 18/18 | 0/18 | 0/18 (5) | 0/18 (5) | 18/18 |
| transpose, 20% of samples | 9/15 (2) | 12/15 | 1/15 | 0/15 | 10/15 (5) | 0/15 (5) | 13/15 |
| flip_x, 20% of samples | 9/18 (2) | 16/18 | 1/18 | 0/18 | 13/18 (5) | 7/18 (5) | 16/18 |

The one clean F004 hit is IC-ThermBench S2, whose first 400 samples contain 3 cross-split and 3 within-split identical u
fields (a property of the file, not a false alarm). F001 and F005 never fire on a clean copy. F008 / F009 skip on 5 of the 18
clean tuning datasets (IC-ThermBench S5 and four Therm-FM HS sets) because the lumped operator model explains a median R^2 of
only 0.19-0.29 there. Where F008 runs it catches every transpose (10/10) and flip (13/13) it can score; its misses are skips.
A one-cell roll is invisible to every rule wherever the operator R^2 is around 0.5 (IC-ThermBench S3-S5, IND_32C,
3D-ICE geometries 5-6). F005 misses the unit slip on IC-ThermBench S4 and the two HS_QC sets: the slipped maximum is still above 150.
The 20%-of-samples cross-check of F004 in `bench_results.md` (shift_index@20, 18/18) fires because the shifted samples duplicate their neighbours.

**Held-out datasets (PDEBench Darcy, beta = 1.0 and 0.01, 1000 samples each: k varies, uniform source, u = 0 boundary).**
Run once with the frozen thresholds. **F008 and F009 did not work as frozen.** They skipped on both sets and every fault:
with the harmonic face-conductivity stencil the operator explains only R^2 = 0.19 (beta 1.0) and 0.16 (beta 0.01), below the
0.3 applicability gate. F002 and F003 skip (no varying source). Only F004 acted: cross-split duplicates 2/2. False positives
0/2, but only because nothing ran.

**Post-hoc diagnosis (not a clean held-out measurement).** PDEBench's Darcy data match the arithmetic mean of k on cell
faces: with `op_face_mean = arithmetic` the operator fit gives R^2 = 1.00 on the stored data (0.02-0.03 when transposed or
flipped). Re-running the held-out sets with that option (chosen after seeing the failure, so a convention fix and no longer a
held-out estimate): F008 detects transpose, flip_x, flip_y, roll 1-3 and the 20%-contaminated variants on 2/2 datasets each,
0/2 false positives. F009 still detects nothing on the shifted data: when the pairing is wrong every (inputs_i, u_j) R^2 is
near zero, and the applicability gate reads that as "operator model does not apply" and skips. The same gate costs F009 the
shifted IC-ThermBench S4/S5 and HS_SC sets in the tuning table. The face-averaging convention of the generating solver must be
known; F008 is only as good as that match.

**Out-of-scope controls (clean files, no faults).** PDEBench diffusion-reaction (t = 20; u = species 0 at the final time,
source = species 0 initially): F002 skipped (negative values count as sinks), F003 ok, F008 / F009 skipped (median R^2 = 0.00),
no finding. PDEBench Burgers (read as a 1 x 1024 line, two files): F008 / F009 skip, but **F002 reports an error and F003 a
warning** (28% of samples): F002 and F003 give false alarms on a 1D advective problem pointed at them, so treat their output
on non-diffusion data as meaningless. The new rules stay out of the way.

**HS_SC (Therm-FM).** Its negative plain correlation (-0.31 stored, +0.37 after flipping y) remains unexplained. F008 rules out
a transposition (R^2 0.006 against 0.19 stored) but cannot adjudicate flips: stored 0.19 against 0.20-0.21 for the three
flips, a difference inside the noise of a model that explains under a quarter of the variance.

## Limitations

* Scope is steady, non-negative-source diffusion-type problems. Time-dependent, advective or sink-dominated problems
  violate the premises of F002/F003/F006/F007; the rules say so in their docstrings.
* F003 can only test orientations of the source; k is assumed to share the source's orientation. F008 tests source and k together.
* F008/F009 need the operator to explain the field (median R^2 >= 0.3) and a matching face-conductivity convention; thin dies with strong vertical coupling, and solvers that average k differently, fall outside.
* The measured rates come from three tuning families and two held-out Darcy files: small samples that say little about other data.
* F004 finds copies, not augmentations or perturbations; leakage is only testable when split labels are available.
* F006 is a screening heuristic on a block-averaged, subsampled problem, not a certificate of linearity.
* The `3dice` adapter reads pickled metadata (`allow_pickle`): only use it on files you trust.

## Tests

```
pip install -e "tools/fieldlint[test]"
python -m pytest tools/fieldlint/tests -q
```

Fault-injection benchmark: `python tools/fieldlint/bench/run_bench.py --stage explore|freeze|tuning|heldout|control|posthoc|report` (see its docstring; about an hour per stage on one core).

Synthetic tests solve a small 2D Poisson problem with numpy to produce physically valid samples, then transpose or flip the
inputs, inject an off-source maximum, and duplicate samples across splits. Real-data tests (marked `realdata`) read
IC-ThermBench, Therm-FM and 3D-ICE files read-only and skip when absent; set `FIELDLINT_DATA_ROOT` to the `data/` directory.
