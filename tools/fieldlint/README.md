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
from the file name), `fno_darcy` (FNO-paper Darcy, NeuralOperator release; see the v2 section), `darcy` (PDEBench 2D Darcy `.npz`: flattened `X` = k and `Y` = u reshaped to 128 x 128, the uniform
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
| F008 | operator-residual | fit of s ~ a(-div k grad u) + b(u - c) (conservative 5-point stencil, face k harmonic / arithmetic / geometric chosen per dataset) per z-layer; R^2 as stored vs transposed / flipped / shifted 1-3 cells; also works when the source is uniform and only k varies | warning >= 10% of samples, error >= 50% |
| F009 | pairing | M = 32 samples: F008's R^2 for every (inputs_i, u_j); the diagonal must be the best match of its row | warning >= 25% of rows, error >= 50% |

F008 and F009 skip (with the reason) when the operator model explains a median R^2 below 0.3 in every orientation, when
neither the source nor k varies, and for fields that are not 2D or z-stacked 2D. The face conductivity is a convention
(`Options.op_face_mean`: since v2 `auto`, the default, which fits `harmonic` / `arithmetic` / `geometric` per dataset and reports the
choice; or a fixed one); see the Darcy results below. F002, F003, F008 and F009 also skip with an explicit "out of scope" reason
when the field is effectively 1D (an in-plane axis shorter than 8 cells) or the dataset is flagged time-dependent
(`time_dependent: true` in the mapping config; no shipped preset loads trajectories), and, since v4, when the advection probe finds the data advective (see the Round 4 section).

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

## Round 2 (v2): fitted discretisation, scope gate, fresh held-out

The v1 numbers above stand as measured: the frozen v1 rules failed on their held-out set. v2 changes (all frozen on 2026-10-07T02:15,
`bench/results/frozen_thresholds_v2.json` with the code SHA-256; the v1 file is untouched) and was then run once on a dataset nobody had looked at:

* F008/F009 choose the face mean per dataset (`auto`): on 32 samples each of harmonic / arithmetic / geometric is scored by the median
  best-orientation R^2 and the best wins. (Chosen on the best orientation rather than the stored one, so a mis-oriented input cannot decide
  it; it is still independent of the orientation vote.) It picks arithmetic on both PDEBench Darcy files (R^2 1.00 / 0.99 against 0.19 / 0.17 harmonic).
* Scope gate: F002/F003/F008/F009 skip as "out of scope" for time-dependent or effectively 1D data.
* F009 gate unchanged: skipped, not warned. A wrong pairing and an inapplicable operator model both give a diagonal indistinguishable from the
  off-diagonal scores, so the score matrix alone cannot separate them; a warning there would be a guess.
* Tuning set = the v1 tuning sets plus both PDEBench Darcy files (seen in v1). Thresholds are otherwise the v1 ones.

| | v1 (frozen, held-out = PDEBench Darcy) | v2 (frozen, fresh held-out = FNO-paper Darcy) |
|---|---|---|
| tuning false positives, any rule | 1/18 (IC-ThermBench S2 true duplicates) | 1/20 (same) |
| F008 transpose / flip_x / flip_y on tuning | 10/15, 13/18, 13/18 (5 skips) | 13/17, 16/20, 16/20 (4 skips) |
| F008 roll 1 / 2 / 3 cells on tuning | 8/18, 12/18, 13/18 | 13/20, 15/20, 16/20 |
| F009 index shift on tuning | 12/18 (6 skips) | 12/20 (8 skips) |
| any-rule on tuning: transpose / flip_x / flip_y / index shift / unit slip / duplicates | 15/15, 18/18, 16/18, 16/18, 15/18, 18/18 | 17/17, 20/20, 18/20, 16/20, 15/18, 20/20 |
| held-out F008 / F009 | skipped on both files (R^2 0.19 / 0.16 < gate): **failed** | **skipped on the one file (R^2 0.03): not informative** |
| held-out false positives | 0/2 (nothing ran) | 0/1 (nothing ran) |
| controls: Burgers 1 x 1024 | F002 error, F003 warning (false alarms) | all skipped, "out of scope: effectively 1D" |
| controls: diffusion-reaction | F008/F009 skipped | F002/F008/F009 skipped; with `time_dependent` also F003 skipped |

(Full v2 tables: `bench/results/bench_results_v2.md`; v1: `bench/results/bench_results.md`. The v1 stage code path is kept, but its
hash check now fails because the rules changed.)

**Fresh held-out, FNO-paper Darcy** (Li et al. 2020; NeuralOperator release, Zenodo record 12784353, DOI 10.5281/zenodo.12784353, CC-BY-4.0,
`darcy_128.tgz`, `darcy_test_128.pt`, 1000 samples, converted to `.npz`; stored under `data/_external/fno_darcy/`). F008 and F009 skipped: the operator
explains a median R^2 of only 0.03 with any face mean. The released low-resolution file is processed (a is binary, u rescaled, the 128 grid is
resampled from the 421 grid the PDE was solved on); a post-hoc probe with a = 3 + 9x and a = 12 - 9x and all three face means gave R^2 <= 0.033, so it is
not a convention problem the fitted choice can absorb. F002/F003 skip (uniform source). Only F004 acts (cross-split duplicates 1/1). This held-out
test therefore says nothing about F008's detection rate; the only held-out evidence that F008 works on k-only data remains the post-hoc PDEBench result above.

**Second fresh held-out, native grid** (run under the v2 hash check, frozen v2 thresholds unchanged; the only code-side addition is the new
preset file `fno_darcy_native.json`, which the hash check does not cover as it only verifies the frozen files). Source: the original Li et al. 2020
Darcy flow `piececonst_r421_N1024_smooth1` on its native 421 x 421 solver grid, as redistributed in the Hugging Face dataset
`LDA1020/codepde-data`, file `darcy/piececonst_r421_N1024_smooth1_sample100.hdf5` (100 samples; sha256
`7032793a1a61f33f6945031252ad8edeeea32b930037ed2bf2ad6bb905486a07`, equal to the repository's LFS id). Checked as native: `coeff` is uint8 with values
{3, 12} only, `sol` is float64, both 421 x 421, solution near zero on the border. Result (1 dataset, so each cell is 0/1 or 1/1): the fitted face mean is
arithmetic; F008's median R^2 on the stored data is only 0.31 (0.02 transposed or flipped), just above the 0.3 gate, so the original solver's discretisation is
matched only loosely. False positives 0/1. F008 detects transpose, flip_x, flip_y and roll 1, 2, 3 (each 1/1). It skips (R^2 below the gate) on
the 20%-contaminated variants and on the index shift, and F009 skips on every fault. So on data on its solver grid F008 works for whole-dataset faults, with a
margin so thin that partial contamination drops it below the applicability gate. Full numbers: `bench/results/bench_results_v2.md`.

**Scope statement.** F008/F009 apply only to data stored on the solver's native grid, with a known (or fitted) face-conductivity convention, and only
while the operator explains a median R^2 of at least 0.3. The first fresh held-out file (the NeuralOperator 128 x 128 release, resampled from the 421 grid
with rescaled fields) violates the first condition, which is why that test was uninformative; the native file above does not.

## Round 3 (v3): pre-registered held-out measurement

Frozen v2 rules and thresholds, unchanged (hash check against `bench/results/frozen_thresholds_v2.json` passed in every run). Protocol `bench/protocol_v3.md` and data cards
`bench/datasets_v3.md` were frozen (commit f118382) before any rule was run on the new data; each of 18 runs was made once (`bench/run_v3.py`); results `bench/results/v3/`, tables `bench/results/bench_results_v3.md`.
Seven in-scope sets (PDEgym Poisson-Gauss, CNO Poisson and Darcy, DiffusionPDE Poisson and Darcy denormalised with source = -a for Poisson, ELADO Poisson A1_test, UCSD/LLNL MTNN Poisson; all generic Poisson/Darcy families, not distinct application fields),
two as-released diagnostics, and nine out-of-scope controls.

| criterion | result |
|---|---|
| C1: clean-adjusted any-rule detection of whole-dataset transpose / flip_x / flip_y, in scope (draft threshold 80%) | 18/21 = 0.86 (exact 95% CI 0.64-0.97); 21/21 without the clean adjustment |
| C2: clean false positives after adjudication (threshold 0 of 7) | 0 of 7 (two sets had findings before adjudication, all confirmed: CNO Darcy has 11 bit-identical duplicate pairs and a stored output of the opposite sign to its documented equation; DiffusionPDE Darcy has a few small negative values) |
| C3: no false alarm on controls, with the `time_dependent` declaration set for time-dependent data, steady controls as is | **fails**: DiffusionPDE Helmholtz (F003 error) and UCSD convection-diffusion (F003 and F008 errors) |
| C4: F008 applicable on at least half of the native-grid sets | 5/5 (median operator R^2 0.9999-1.0000) |

Mechanical verdict of the protocol: not paper-worthy as measured (C3 fails). F008 explained the clean fields almost exactly on all five native sets (and detected every whole-dataset transpose and flip there);
on the two Darcy-lineage sets it skipped on CNO Darcy (R^2 0.24) and ran at R^2 0.60 on DiffusionPDE Darcy. With the scope declaration set, every time-dependent control was declined by F002/F003/F008/F009; without it,
five of the six time-dependent controls drew false alarms. F004 found bit-identical duplicate samples in two CNO files (Darcy: 11 pairs; ContTranslation, a control: sample i equals sample i + 512).

## Round 4 (v4) design: F003 defers to F008, advection probe

**v4 is tuned on everything seen, including the v3 sets and the two v3 control failures that triggered it. No dataset is held out for v4: the v3 numbers are no longer held-out evidence for the v4 code.** The v3 verdict (C3 failed) and the v2 / v3 frozen files and results are unchanged. The v4 numbers below are tuning numbers; a held-out measurement needs data nobody has looked at (Phase B).

Two changes, both motivated by the physics of the two failures, not by fitting them:

1. **F003 defers to F008.** The correlation of a smoothed source with u is a monotone proxy for the operator, valid when the field is 'hot where powered'. DiffusionPDE Helmholtz is not (u oscillates, a term k^2 u): the correlation is -0.82 as stored and +0.31 flipped, while F008's reaction term b(u - c) absorbs k^2 u and fits the stored orientation with median R^2 1.00 (best of transpose / flips: 0.24). When F003 would fire, F008 ran on the same data, did not flag it, and its stored-orientation median R^2 beats the best of transpose / flip_x / flip_y / flip_xy by `defer_margin` = 0.10, F003 reports info ("F008 confirms the stored orientation"). When F008 skips, flags, or is ambiguous, F003 stands alone (tested on a noisy flipped field where F008 skips).
2. **Advection probe** (`advection_probe`, run as part of the scope gate `operator_skip` of F002 / F003 / F008 / F009). Advection adds v . grad u to the equation and shifts u downstream of the source, so a flipped diffusion fit can beat the stored one (UCSD ConvDiff: F008 median R^2 0.18 stored, 0.67 flip_xy). On 32 evenly spaced samples the probe refits F008's model in the stored orientation with two extra regressors u_x, u_y (one drift vector per sample and z-layer). The data are called advective, and the four rules skip with "out of scope: advective (first-derivative terms explain the residual ...)", when (a) the median R^2 gain is at least `adv_gain` = 0.20 and (b) the drifted stored fit beats the best re-oriented plain fit by at least `adv_margin` = 0.05. Condition (b) is the guard against faults: a flipped or transposed field is not rescued by a uniform drift in its stored orientation, because the correctly oriented plain fit still explains it better. Optional declaration in the mapping config: `operator: diffusion` (skip the probe), `advective` or `other` (rules report out of scope); detection is automatic by default.

**Margins, chosen on the tuning set only.** Over all seen datasets, clean and with whole-dataset transpose / flips / roll / index-shift faults (explored directly on 27 datasets; all through the full run): ConvDiff clean gain 0.70, drifted 0.89 against 0.67 best alternative (margin +0.22). Every other clean set has gain <= 0.065. Faulted sets can have a large gain (UCSD Poisson flip_y: stored R^2 0.10 -> 0.81 with drift) but the drifted fit stays at least 0.19 below the correct orientation (1.00), and the largest faulted gain elsewhere is 0.21 (3D-ICE geometry 3 flip_y, drifted 0.22 against 0.99). Any gain threshold between 0.07 and 0.64 and any margin between -0.19 and +0.22 separates the seen cases; 0.20 and 0.05 (equal to `op_margin`) are round values inside those gaps, so they are not tight-fitted. The separation is wide on seen data but rests on one advective dataset.

**Tuning results** (`bench/run_v4_design.py`, results `bench/results/v4_tuning/`, tables `bench/results/bench_results_v4_tuning.md`, frozen `bench/results/frozen_thresholds_v4.json`): 31 in-scope datasets x 15 conditions (clean + 14 faults), 446 runs. **No rule's status or detection changed in any run**, so every pooled tuning table is identical before and after (no detection lost, no clean false positive added; the one clean F004 hit is still IC-ThermBench S2's true duplicates). The nine v3 controls: ConvDiff F003 / F008 errors gone (declined as advective), Helmholtz F003 error gone (info). Unchanged false alarms on controls: with no time_dependent declaration, cno_wave (F003, F008, F009), cno_navier_stokes (F009), cno_cont_transport and cno_disc_transport (F003 warnings), pdegym_wave_gauss (F002 error, F003 warning); with the declaration all declined. The synthetic Helmholtz test field still makes F002 fire (negative lobes of u away from the source); DiffusionPDE Helmholtz escapes only because its source has negative cells.

Code changes alter the SHA-256 that `frozen_thresholds_v2.json` records, so the v2 / v3 harness refuses to re-run on this code; their stored results stand.

## Round 4 (v4) held-out

Frozen v4 rules (`bench/results/frozen_thresholds_v4.json`, hash check passed at every run). Protocol `bench/protocol_v4.md` and data cards `bench/datasets_v4.md` frozen (commit 187662b) before any run; 13 result files (14 runs), each made once (`bench/run_v4_heldout.py`), no crash; results `bench/results/v4_heldout/`, tables `bench/results/bench_results_v4_heldout.md`.
Nine in-scope sets (ConDiff Poisson, Gaussian-0.1, Gaussian-1.0, cubic-0.4, exponential-2.0; ELADO B1_test; PDEBench Darcy beta 0.1, 10, 100, same generator as tuning), one diagnostic (ConDiff Gaussian-0.1 without k), three controls (PDEgym Helmholtz, ELADO C1 Helmholtz, PDEgym Wave-Layer flagged and unflagged). No fresh advective control was obtainable.

| criterion | result |
|---|---|
| C1: clean-adjusted any-rule detection of whole-dataset transpose / flip_x / flip_y (threshold 80%) | 27/27 = 1.00 (exact 95% CI 0.87-1.00; one-sided lower bound 0.89) |
| C2: clean false positives after adjudication (threshold 0 of 9) | 0 of 9 (no rule flagged any clean in-scope copy) |
| C3: no false alarm on binding controls | **fails**: F005 warning on PDEgym Helmholtz and on Wave-Layer with `time_dependent: true`; ELADO C1 drew no false alarm (F001, F004, F005 ok; F002, F003, F008, F009 skipped) |
| C4: F008 applicable on native sets (6 binding) | 6/6 (median operator R^2 0.884-1.000); 9/9 with PDEBench Darcy counted native |

Mechanical verdict of the protocol: not paper-worthy as measured (C3 fails). The F003 deferral and the advection probe did not trigger on any run. Unflagged Wave-Layer drew F002 (error), F003 and F005 (warnings).

## Round 5 (v5) design (tuning on seen data; no held-out claim)

**v5 is tuned on everything seen, including the v4 held-out sets and the two v4 control failures that triggered it. Its numbers are tuning numbers.** Frozen: `bench/results/frozen_thresholds_v5.json` (2026-10-19, with the code SHA-256); the v2/v3/v4 frozen files and results are unchanged (their hash checks fail on v5 code). Decisions approved by the user on 2026-10-15; plan: `bench/plan_v5.md`.

1. **F005 honours the scope gate.** The below-ambient check (only) is not applied when `scope_skip` gives a reason (`time_dependent`, `operator: advective|other`, effectively 1D); F005 reports an info line instead. The integrity checks (constant field, K/C range, k <= 0, zero source) stay unconditional.
2. **F005 weak-minimum-principle probe.** For samples with a non-negative source and a known ambient that are not sign-definite, `V = (min over the in-plane boundary ring - min u) / ptp(u)` (maximum over z-layers) is the interior undershoot of the boundary minimum, impossible for `-div(k grad u) + c(u - ambient) = s` with `s >= 0`. A sample is oscillatory when `V >= osc_v = 0.05`; when at least `osc_fraction = 0.5` of the checked samples are oscillatory the below-ambient warning becomes an info line ("not a passive diffusion solution"). Isolated oscillatory samples still warn; sign-definite fields (CNO Darcy) and constant offsets (as-released DiffusionPDE Darcy) are untouched. The thresholds come from the pre-run diagnosis in `plan_v5.md` (35 diffusion sets: max V 0.014; oscillatory controls: min V 0.09) and were not fitted further; on all 60 clean tuning records the oscillatory fraction is 0 except PDEgym Helmholtz and Wave-Layer (1.0).
3. **F009 scores a contiguous block.** `pair_samples = 64`: the first 64 consecutive samples (v4: 32 evenly spaced, whose neighbours were never in the matrix, so an off-by-one pairing fault was invisible on datasets longer than about 64; v4 held-out `shift_index`: 0 of 9).

**Tuning check** (`bench/run_v5_design.py`; 53 runs = the 40 v4 tuning runs plus the 13 v4 held-out runs; before = the stored v4 results, never rewritten; `bench/results/v5_tuning/`, `bench/results/bench_results_v5_tuning.md`):
no status, severity or detection changed for F001, F002, F003, F004, F008, or for F005 on any set except (i) PDEgym Helmholtz and Wave-Layer flagged and unflagged (warning -> info, as intended), and (ii) **one** info line added to the declared time-dependent run `cno_cont_transport+time_dependent_flag` ("below-ambient check not applied": the by-product of part 1 that the plan anticipated for every declared time-dependent control; status ok, nothing detected). The comparison script's allow-list was widened to accept (ii) **after** seeing it (disclosed in `run_v5_design.py`). F005 still flags CNO Darcy, DiffusionPDE Darcy (denormalised and as released) and every unit-slip / shift_index detection exactly as before.
F009, the intended change, moves a lot, **including two new false findings that the lead should weigh**:

| F009, pooled over in-scope + diagnostic tuning runs | v4 (32 evenly spaced) | v5 (first 64 consecutive) |
|---|---|---|
| clean findings | 0 / 34 ran (7 skipped) | **1 / 34 (7 skipped)**: Therm-FM IND_32C (warning) |
| `shift_index` detected / ran / skipped | 16 / 17 / 24 | **31 / 32 / 9** |
| transpose / flip_x / flip_y detected / ran | 12/12, 15/15, 9/12 | 14/14, 19/19, 13/16 |
| `*@20` variants detected (transpose, flip_x, flip_y, roll1, shift_index) | 0, 6, 4, 0, 7 (of ~34) | 1, 1, 1, 1, 1 (of ~33) |

On the controls, F009 now **flags the ELADO Helmholtz control C1 with an error** (v4: skipped by the R^2 gate), a steady out-of-scope file seen in v4; so a false alarm on an out-of-scope steady control follows from the F009 change (the wider block gives more columns to out-score the diagonal by `pair_tie` = 0.10 at moderate R^2). Not worked around: `pair_tie`, the gates and every other option are unchanged. These two findings make a clean F009 finding (C6b of the v5 plan) and a control false alarm (C3) more likely in the held-out round than the plan's tuning evidence (0/18 clean) suggested.

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

## Status

* **Detection on fresh data:** v4 held-out C1 27/27 (one-sided 95% lower bound 0.89), C2 0/9, C4 6/6 native sets.
* **Out-of-scope data:** both pre-registered rounds failed C3. v3 failed on F003/F008 for steady Helmholtz and convection-diffusion; v4 failed on F005 for a steady Helmholtz and a declared time-dependent wave set.
* So fieldlint reliably detects orientation faults in in-scope data, but does not yet reliably decline out-of-scope data, and users must declare scope.
* The advection probe has no held-out test.
