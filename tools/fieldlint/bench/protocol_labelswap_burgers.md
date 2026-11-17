# Label-swap test on PDEBench 1D Burgers (pre-registration)

**STATUS: RUN, outcome in section 6** (frozen when the commit adding the run's first prediction file lands; this file's text at that commit
is binding). Written 2026-11-16, before any model was trained on these labels. Script: `scripts/labelswap_burgers.py`.
Method: `fieldlint.labelswap`, spec `tools/fieldlint/docs/labelswap.md`.

## 1. Question

Does a 1D Fourier neural operator trained on PDEBench's Burgers data (nu = 0.001) reproduce the error of the solver
that made the data? The test trains the same FNO on PDEBench's stored labels and on corrected labels for the same
inputs, and projects the difference of its held-out predictions onto the label difference. This is the thermal
study's section 9.32 test applied, unchanged, to an outside benchmark.

PDEBench nu = 0.01 was checked first and rejected before any training. Its stored solution matches an accurate
re-solve to rms 9e-5 (about 0.03% of the field), below the reference's own refinement change, so it has no artefact
to test (`notes/2026-11-16-u1-label-swap.md`). nu = 0.001 has fronts narrower than the 1024-point grid, where a
finite-volume solver is expected to smear them.

## 2. Data

* PDEBench `1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5` (DaRUS file id 268190), first 400 trajectories, read by
  HTTP range request. Input `u(x, 0)`; output `u(x, t)` at stored time index 20 (`t-coordinate` gives t = 0.2).
  1024 points on [0, 1), periodic.
* Old labels: the stored `u(x, 0.2)`.
* New labels: the same trajectories re-solved from the stored `u(x, 0)` for `u_t + (u^2/2)_x = (nu/pi) u_xx`
  (PDEBench's convention, confirmed on nu = 0.01: rms mismatch 0.0029 at t = 0.2 against 0.044-0.049 at t = 0.19 and
  0.21). The solver is a dealiased (2/3 rule) pseudo-spectral integrating-factor RK4 on M = 16384 points with
  dt = 2e-5 (stability limit about 5e-5), sampled at the stored points. The initial condition is spectrally interpolated from the stored points.
* `D = old - new`. Label-only checks, made before freezing (section 3.2), are in
  `results/labelswap_burgers/label_checks.json`.

## 3. Design

### 3.1 Model and training (fixed; no tuning on test folds)

* FNO-1D: inputs `u(x, 0)` and `x` lifted to width 32; 4 spectral layers with 16 modes, each plus a pointwise
  convolution, GELU between layers; projection 32 -> 128 -> 1.
* AdamW, learning rate 1e-3 with a one-cycle schedule, weight decay 1e-4, batch 20, 300 epochs. No early stopping, so
  that the stopping point cannot depend on the label set. Loss: MSE on the raw output.
* 5-fold CV, folds from `np.random.default_rng(0)` shuffling the 400 indices (`np.array_split`). Seeds 0, 1, 2. Each
  (fold, seed) is trained once on old and once on new labels: 30 trainings. All 400 samples are held out exactly once,
  so n = 400.
* CPU only, on three Kaggle CPU sessions, one per seed (`train --seeds S --threads 4`: all 5 folds and both arms
  of that seed), joined with `merge`, which refuses a unit that differs between files. Both arms of a (fold, seed)
  unit train in the same session, so a paired difference never mixes machines or library versions. Each session
  records its torch version and CPU.

### 3.2 Label-only checks before freezing

All of these come from labels alone, computed before any model was trained:

| check | value | requirement |
|---|---|---|
| median relative size of D (`|D| / |old - mean|`) | 0.0050 | the artefact exists: above the reference's refinement change |
| reference refinement change, M 16384 -> 32768, dt halved (first 16 samples) | 5.6e-8 (max \|D\| there 0.107) | at most 10% of max \|D\| on the same samples |
| max \|D\| | 0.163 | |
| support nodes per sample at \|D\| > 0.01 (median) | 7 | at least 3 |

All requirements pass. The refinement change is about 5e-7 of max \|D\|, so D is the stored solver's error, not
the reference's. Values from `results/labelswap_burgers/label_checks.json`, computed on a Kaggle CPU session
(numpy 2.1.3, 8094 s); a local re-solve of samples 0 and 399 matches those labels to 1.9e-14.

If a requirement fails, the run does not go ahead and that is reported as the outcome.

### 3.3 Statistic and decision (unchanged from section 9.32)

* Primary field `beta`. Secondary `corr`, `beta_s`, `corr_s`, with support `|D| > 0.01` (units of u).
* Decision: **learned** if the 95% bootstrap CI of beta (4000 resamples, seed 0) lies above the same-label null CI and
  its lower bound is > 0.2; **not learned** if the upper bound is < 0.1 and it overlaps the null CI; otherwise
  **inconclusive**. Headline: per-seed single-seed comparison against the matched same-label pair; all three seeds must
  agree. The seed-mean beta is reported beside it.
* The minimum detectable beta from the same-label null pairs is reported with the verdict. If the MDB is above 0.5, an
  "inconclusive" verdict is to be read as lack of power, not as evidence against learning.

### 3.4 Reported in every outcome

The decision; the seed-mean and per-seed betas with CIs; both nulls; the MDB; each arm's RMSE against its own labels;
the label checks. Nothing is dropped or re-run selectively. A crash is rerun from the saved units with the same
settings and reported.

## 4. Expectations (stated before the run, not a hypothesis test)

Either outcome is informative. "Learned" would show a neural operator reproducing a published benchmark's solver
error at this resolution, the Burgers counterpart of section 9.32 and of Patel (arXiv 2610.09255) on a real
benchmark. "Not learned" or "inconclusive" with a low MDB would mean this FNO does not copy the error at this size.
The FNO family was underpowered in section 9.32 (MDB 0.54-1.67 at n = 45). n = 400 is chosen to shrink the MDB by
about 3x for the same seed noise.

## 5. Amendments

None.

## 6. Outcome (2026-11-17)

Run as frozen: 30 trainings on three Kaggle CPU sessions (torch 2.11.0+cpu, numpy 2.1.3, 4 cores; seed 0 and 2 on
Intel Xeon 2.20 GHz, 4244 s and 4203 s; seed 1 on AMD EPYC 7B12, 2994 s), merged with no missing or conflicting
unit. Nothing was rerun. Numbers from `results/labelswap_burgers/verdict.json` and `session_s*.json`.

**Decision: inconclusive, from lack of power.** The minimum detectable beta is 103, 197 and 127 for the three null
pairs, far above the 0.5 that section 3.3 sets for reading "inconclusive" as evidence against learning.

| field | mean (95% CI) | same-label null N-N (95% CI) | per-seed decisions | decision |
|---|---|---|---|---|
| `beta` (primary) | +1.68 [-0.14, +4.11] | -91.6 [-218.8, -4.8] | inconclusive x3 | inconclusive |
| `corr` | +0.051 [+0.033, +0.069] | -0.025 [-0.035, -0.015] | inconclusive x3 | inconclusive |
| `beta_s` (support, 272 samples) | +0.017 [+0.003, +0.030] | -0.52 [-0.95, -0.12] | not learned, inconclusive, not learned | inconclusive |
| `corr_s` (support) | +0.026 [-0.019, +0.069] | -0.030 [-0.057, -0.004] | not learned, inconclusive, inconclusive | inconclusive |

RMSE against its own labels: 0.130 (old) and 0.131 (new). The label difference has rms 0.0024 against 0.68 for the
field, so the model's own error is about 54 times the artefact it was asked to copy. With all 1024 nodes in the
projection, seed-to-seed noise swamps D, hence the null betas near -90 and the MDB above 100. On the support nodes
(`|D| > 0.01`) the null is tighter and beta_s is small and positive, +0.017 [+0.003, +0.030]: well under the 0.2
"learned" threshold, but its interval lies above the null's, so it is not "not learned" either.

Reading: at n = 400 and this model size, the test cannot tell whether this FNO reproduces PDEBench's Burgers
nu = 0.001 solver error, because the error is about 2% of the FNO's own. The secondary fields point the same way
(at most a few percent of D reproduced on the support), but the rule makes no call on them. A decisive test needs
a model whose error is well below the artefact (a longer schedule or a larger FNO), which would be a new
pre-registration, not a rerun of this one.
