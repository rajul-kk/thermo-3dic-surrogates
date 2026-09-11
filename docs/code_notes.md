# Code notes (migrated from docstrings)

Docstrings in this repo were collapsed to at most two lines on 2026-09-11.
The prose removed is preserved here verbatim, keyed by `file :: symbol`, because
much of it records corrections and measurements this project has repeatedly
relied on rather than merely describing what the code does.

`third_party/` was excluded: it is a verbatim vendored copy with a recorded md5.

---


## `app/job_queue.py`

### FunctionDef:__init__

```text
Args:
            simulator_factory: (config_dir, output_dir, executable) -> a
                ThermalSimulator-like object with .simulate(). Defaults to
                constructing a real ICESimulator. Overriding this is what
                makes JobQueue unit-testable without a real 3D-ICE/WSL
                install -- see tests/test_app.py's stub simulator.
```

## `data/ic-thermbench/datasets/tools/convert_csv_to_mat.py`

### module

```text
Convert per-point CSVs into the `input.mat` / `output.mat` the training scripts read.

Channels are inferred from the columns actually present in the CSV, so there is no need
to name the level by hand:
    always present        chiplet_power, grid_x, grid_y                    -> P=3
    plus local_thermal_k                                                    -> P=4
    plus ambient_c / h_w_m2k / r_convec_k_per_w                             -> P=7
Temperature and ambient_c are converted from degC to K (+273.15). The remaining columns
(chiplet_id, occupancy_mask, edge_mask, coord_*_norm) do not enter the tensors.

Sample ordering (`--layout`):
  interleave (default, for training sets) -- samples are round-robin interleaved by case
      c1s1, c2s1, ..., cNs1, c1s2, ...
      Every framework takes the leading 80% as train+val (then 9:1 for val) and the
      trailing 20% as test. Only after interleaving do all three segments cover every
      case. Stacked by case instead, the leading 80% would miss the last case entirely
      while test would consist of it -- a model evaluated on a case it never saw. The
      divisibility assertion below guards against that.
  sequential (for pure evaluation sets) -- stacked by case, convenient for per-case use.

Also writes `manifest.json` (one `{case, file}` record per sample), which drives per-case
metrics and is the only record of how .mat sample order maps back to the CSV files.

Usage:
    python convert_csv_to_mat.py --csv <CSV_ROOT> --out <OUT_DIR>
    python convert_csv_to_mat.py --csv raw_csv/level5 --out datasets/level5_steady --layout sequential

<CSV_ROOT> must contain `CaseN/*.csv`, one sample per CSV, 4096 rows (64x64).
```

## `scripts/analyze_interface_linearity.py`

### module

```text
Track C found that peak temperature is highly sensitive to TIM/interface k --
up to 75.98K on geometry4. But *sensitive* isn't the same question this
benchmark's central claim is about: is that sensitivity LINEAR?

Naively fitting peak-T against k directly is the wrong functional form: a TIM
layer of thickness t and conductivity k contributes a series thermal
resistance R = t/k to the stack, and for a fixed heat flux Q crossing it,
T = T_upstream + Q*R = T_upstream + Q*t/k. That is linear in 1/k, not in k.
If the paper's "ridge already solves this" argument extends to the interface-
property axis too, peak-T vs 1/k should fit a line far better than peak-T vs
k does -- i.e. the same physics (steady-state conduction is linear in the
inputs that actually enter linearly) should hold here as well, once expressed
in the right variable.

This also directly speaks to whether a ridge baseline that has never been fed
TIM k as an input (scripts/baselines.py's feature_vector has no k feature for
ANY geometry, including g5/g6 where the training set already varies TIM k
internally per assumptions.md 2.2) could, in principle, absorb it as trivially
as it absorbs 1/htc -- since the resistance form T = T_amb + sum_b A_b*Q_b
already includes a 1/h term for exactly this reason (see baselines.py docstring
on `feature_vector`).

Uses ONLY existing sweep data already on disk (data/3d-ice-interface-multi,
data/3d-ice-interface-pilot-hi, data/3d-ice-interface-pilot-median) -- no new
3D-ICE solves.

Usage:
    python scripts/analyze_interface_linearity.py
```

## `scripts/analyze_interface_uncertainty.py`

### FunctionDef:load

```text
Returns (single_rows, interaction_rows). A file whose layer_k_overrides has
    TWO keys (from --interactions) is an interaction run, not a single-parameter
    sweep point -- folding it into a single layer's group by taking only its
    first key (an earlier bug in this function) silently corrupts that group's
    peak-T spread with data from a different, two-variable experiment.
```

### module

```text
Quantify how much interface-property uncertainty moves the answer, and compare
that spread against the model-vs-model differences surrogate papers optimise.

The question this answers: neural thermal surrogates compete over sub-Kelvin
field-error improvements. If plausible uncertainty in the interface properties
fed to the simulator moves peak junction temperature (or hotspot location) by
MORE than those differences, the accuracy race is being run inside the noise
floor of its own inputs.

Reads the sweep produced by scripts/gen_interface_uncertainty_pilot.py, in which
power/HTC/ambient are held fixed and exactly one interface conductivity varies
per scenario. Writes results/interface_uncertainty.json.

Usage:
    python scripts/analyze_interface_uncertainty.py
```

## `scripts/analyze_leakage_convergence.py`

### module

```text
Does the leakage pilot's 30% runaway rate come from (leakage_fraction, k_double_c) --
the two parameters the pilot actually swept -- or from the underlying scenario's own
power/HTC/pattern, which varied incidentally across the 4 repeats of each (frac,
k_double) pair?

Motivated by eyeballing gen_leakage_pilot.py's output: each of the 5 (leakage_fraction,
k_double_c) settings appears 4 times, paired with a different base power scenario each
time, and the outcome (converged/runaway) is NOT consistent within a setting -- e.g.
(0.15, 25.0) converges 3/4 times and runs away once. That means loop gain alone doesn't
determine stability here; something about the base scenario does too.

Usage:
    python scripts/analyze_leakage_convergence.py --data data/3d-ice-leakage-pilot/geometry1
```

## `scripts/analyze_operator_variation.py`

### module

```text
Why is our benchmark linear-solvable and IC-ThermBench's not? (docs/report.md 9.14)

Two diagnostics, both structural properties of the data rather than of any model:

1. Error decomposition -- does the linear model's error concentrate at hotspots, and does it
   concentrate MORE on IC-ThermBench than on ours? (Answer: no. 2.1x vs 2.4x. The
   "their benchmark scores hotspots, which is ridge's weakness" hypothesis is refuted.)

2. Source-support and domain-scale variation -- see analyze_source_variation() below and
   docs/report.md 9.14 for the numbers that actually explain the difference: our heat sources
   never move (one support pattern per geometry, IoU 1.000) while theirs move per sample
   (IoU 0.449), and their physical cell pitch varies ~2x across samples while ours is fixed.

Run: python scripts/analyze_operator_variation.py
```

## `scripts/baselines.py`

### FunctionDef:collect_block_keys

```text
Union of block-power metadata keys, sorted for determinism.

    Prefers `nominal_block_power_*` (the requested power) over
    `block_power_*` (the delivered power after any closed-loop feedback)
    whenever any scenario carries throttle OR leakage metadata. Both
    mechanisms make delivered power a function of the temperature being
    solved for, so both must hand the baseline the request, not the outcome. Using the delivered power as ridge's
    feature hands it the already-resolved answer for throttled scenarios --
    it no longer has to represent the closed feedback loop at all, and scores
    a misleadingly high R^2 that has nothing to do with whether the map is
    actually linear. Found 2026-08-09: ridge scored *better* on throttled
    data (spatial R^2 0.989) than on the same geometry without throttling
    (0.970) until this was fixed -- using nominal power instead correctly
    shows real degradation (0.890).
```

### FunctionDef:feature_vector

```text
Build the scenario parameter vector.

    Includes 1/htc alongside htc: convective thermal resistance is proportional
    to 1/h, so 1/h is the term that enters the temperature field linearly. Giving
    the linear models this feature is what makes `ridge` a fair — rather than
    strawman — baseline.

    `pos_keys` are per-block `block_x_*` / `block_y_*` positions, added 2026-09-11 for the
    moving-source datasets (docs/report.md §9.14). Once chiplet placement varies per
    scenario, a baseline that only sees per-block POWER cannot know where the heat went, and
    beating it would prove nothing. They are constant for fixed-placement datasets, where
    the zero-variance guard in `predict_all` neutralises them.
```

### FunctionDef:metrics

```text
Raw AND spatially-detrended error.

    Detrending matters on this dataset. Each scenario's field is dominated by a
    near-uniform offset set by ambient temperature and total power, while the
    spatial gradient within a scenario is ~1 K. Raw MAE therefore mostly scores
    how well a model copies the ambient input, and a model that predicts a
    constant per scenario can post a respectable MAE having learned no spatial
    structure whatsoever.

    `mae_detrended_K` removes each field's own mean from both prediction and
    truth, so it measures only the spatial structure -- the part a thermal
    surrogate actually exists to predict. `spatial_r2` is R^2 on that same
    detrended field. Report both; judge architectures on the detrended pair.
```

### FunctionDef:power_pca_features

```text
Project each scenario's full power field onto its leading principal components.

    Needed to keep ridge a FAIR baseline once power becomes a per-cell field. With
    block-scalar power the whole source is 4-13 numbers and ridge can consume it
    directly; with a per-cell map the source has ~10,000 degrees of freedom and
    handing ridge only the block means would rig the comparison by hiding most of
    the input. PCA gives the linear model the best n_comp-dimensional summary of
    the source that exists, so if it still loses, it loses on the merits.
```

### FunctionDef:predict_all

```text
Fit every baseline on `train` and return raw predicted fields for `test`.

    Returns one dict per test scenario: {baseline_name: predicted (N,) field in K}.

    Split out of `fit_predict` (2026-09-09) so callers that need different metrics
    -- scripts/hotspot_eval.py scores hotspot-specific quantities rather than the
    detrended field metrics below -- reuse these exact baseline implementations
    instead of reimplementing ridge/kNN. Two copies of a baseline that silently
    drift apart would invalidate every comparison in this project that relies on
    them, and this repo has already been bitten once by a duplicated metric
    definition living in a notebook.
```

### module

```text
Non-neural baselines for the 3D-IC thermal benchmark.

Every learned surrogate in this repo must beat these before its architecture can
be credited for anything. They need no GPU and no training loop; the whole suite
runs in seconds.

Baselines
---------
mean   Predict the training-set mean temperature field. The floor: any model
       that does not beat this has learned nothing.

nn     Copy the temperature field of the single nearest training scenario in
       normalised parameter space. Zero-parameter memorisation.

knn    Inverse-distance-weighted blend of the k nearest training scenarios.
       This is "what you get for free" from the dataset without any model.

ridge  Per-point ridge regression on the scenario parameter vector. This is the
       baseline that matters most scientifically. Steady-state conduction with
       fixed k is LINEAR in volumetric power and in ambient temperature:

           T(x) = T_amb + sum_b A_b(x) * Q_b

       where A_b is the (scenario-independent) thermal impedance from block b.
       The only genuine nonlinearities are k(T) and the convective boundary term
       (which enters through 1/h, supplied here as an explicit feature). A ridge
       fit therefore recovers the exact physics of the linear regime, and any
       neural surrogate must beat it to justify its cost. If ridge is already at
       ~1 K MAE, the nonlinear capacity is not what is buying accuracy.

Usage
-----
    python scripts/baselines.py --geometry geometry1 --data data/3d-ice
    python scripts/baselines.py --geometry geometry1 --data data/3d-ice \
        --test-split ood --output results/baselines_geometry1_ood.json
```

## `scripts/baselines_leakage.py`

### module

```text
Ridge-vs-leakage-feedback test: the key question this pilot exists to answer.

Reuses scripts/baselines.py's fit_predict/metrics machinery directly rather than
duplicating it, but with one necessary filter the generic CLI doesn't have:
runaway scenarios are excluded before fitting or scoring anything.

Runaway is not "hard data," it's a numerical divergence -- some pilot scenarios
hit peak temperatures of several hundred thousand degrees, which is not a
temperature field a surrogate should be scored against (or a physically
meaningful state; silicon melts at 1414 C). Testing ridge against that would be
answering "does ridge cope with garbage," a different and uninteresting question.
The question this pilot asks is whether ridge still solves the CONVERGED portion
of a positive-feedback electrothermal problem -- i.e. whether the mechanism
degrades the baseline even where a physically sensible steady state exists.

Usage:
    python scripts/baselines_leakage.py --data data/3d-ice-leakage-pilot --geometry geometry1
```

## `scripts/benchmark_linearity_audit.py`

### FunctionDef:_spatial_r2

```text
Detrended (per-sample mean removed) R^2, averaged over samples.

    This is the metric the rest of this project judges on (docs/report.md 9.1a) and the one
    these numbers must be comparable to. Raw pooled R^2 is dominated by each sample's mean
    offset, which a constant predictor can match, so it flatters and misleads.
```

### FunctionDef:participation_ratio

```text
Effective number of degrees of freedom in an ensemble of input fields.

    exp(Shannon entropy of the normalised PCA eigenvalue spectrum). Equals 1 when every
    sample is a scalar multiple of one pattern (pure amplitude variation) and rises toward
    the number of retained components as the ensemble becomes genuinely high-dimensional.
    Preferred over "number of components to reach 99% variance" because it is continuous and
    does not depend on an arbitrary threshold.
```

### module

```text
Is this PDE-surrogate benchmark linear-solvable? A portable diagnostic.

Generalises docs/report.md §9.14-§9.15 beyond chip thermal. Those sections established, on
this project's own data and on IC-ThermBench, that:

  * a benchmark that holds the PDE operator fixed and varies only the source AMPLITUDE is
    solved exactly by a fixed linear operator -- it cannot discriminate between surrogate
    architectures, and in chip thermal it is the regime classical power blurring has solved
    to ~1 K since 2007 (Kemper et al., arXiv:0709.1850);
  * source *movement* is not the thing that breaks this. Rigid translation left a linear fit
    essentially intact even at a support overlap lower than IC-ThermBench's. What breaks it
    is the DIMENSIONALITY of the varying structure.

This script applies that as a benchmark-agnostic audit. Given input fields X and output
fields Y for any benchmark, it reports three numbers:

  D1  linear solvability   Pooled R^2 of a dense ridge operator fitted X -> Y, plus the
                           standard relative-L2 metric. This is the headline: steady linear
                           PDEs with a fixed operator are EXACTLY of this form, so a high
                           R^2 means the benchmark does not require a neural operator, it
                           requires a matrix.
  D2  effective input DOF  Participation ratio of the input ensemble's PCA spectrum,
                           exp(H) of the normalised eigenvalue distribution. A benchmark
                           whose inputs are amplitude rescalings of one pattern has DOF ~1;
                           independently placed sources have DOF in the tens or hundreds.
  D3  trivial floor        R^2 of predicting the training-mean field, so D1 can be read
                           against what costs nothing.

Reading the result: D1 high AND D2 low is the failure mode -- a benchmark that looks like
a hard field-prediction task but is a small linear problem in disguise. D1 low is a
benchmark that has earned the architectures evaluated on it.

Ridge is fitted on the train split only and lambda is chosen on a validation split, never
on test. The dense operator is the physically correct model class for a linear PDE, so this
is a strong baseline rather than a strawman; a PCA-restricted variant is also reported so a
large parameter count cannot be blamed for the result.

Usage:
    python scripts/benchmark_linearity_audit.py --datasets ours icthermbench
    python scripts/benchmark_linearity_audit.py --list
```

## `scripts/classify_leakage_convergence.py`

### module

```text
Turn the eyeballed "total nominal power >~271W predicts leakage runaway" observation
(scripts/analyze_leakage_convergence.py) into an actual fitted, cross-validated
classifier, using only features known BEFORE the leakage feedback loop runs (total
nominal power, HTC, ambient temperature, leakage_fraction, k_double_c) -- i.e. features
that would let you predict "will this operating point even have a steady state?"
without paying for the iterative 3D-ICE solve at all.

Deliberately excludes leakage_iterations/leakage_multiplier -- those are OUTPUTS of the
feedback loop, not predictors; a classifier trained on them would be cheating.

Small-sample caveat stated up front: n=20 (growing to n=45 as
data/3d-ice-leakage-pilot/geometry1 fills in, see gen_leakage_pilot.py --extra-train).
Leave-one-out CV is used specifically because a train/test split would be meaningless at
this sample size.

Usage:
    python scripts/classify_leakage_convergence.py --data data/3d-ice-leakage-pilot/geometry1
```

## `scripts/eval_fno_throttled.py`

### module

```text
Evaluate the geometry1_throttled FNO checkpoint with the same detrended metrics
scripts/baselines.py scores ridge with, so the two are directly comparable.

Written 2026-08-16 to close a reproducibility gap found during a documentation
audit: docs/report.md §9.7/§9.8 and goal.md quoted this checkpoint's det.MAE/
spatial R^2/hotspot error, but the eval that produced those numbers had only
been run inline and was never saved as a re-runnable artifact. Output is
written to results/fno_throttled_eval.json.

Usage:
    python scripts/eval_fno_throttled.py
```

## `scripts/eval_pinn.py`

### module

```text
CLI entry point for PINN evaluation.

Usage:
    python scripts/eval_pinn.py \
        --checkpoint checkpoints/geometry1/geometry1_best.pt \
        --data data/3d-ice \
        --output results/geometry1/ \
        [--plots]

Outputs:
    metrics.json         per-scenario and aggregate metrics
    comparison_*.png     z-slice comparison plots (if --plots)
    z_profile_*.png      z-axis profile plots (if --plots)
```

## `scripts/experiment_geometry_aware.py`

### module

```text
Track B validation gate: leave-one-geometry-out.

Train FNO on 5 geometries (resampled onto a common grid), test zero-shot on
the 6th, with and without the distance-to-power-block geometry-aware field
(src/core/mesh.py:generate_distance_to_power_block_field). This is the actual
test goal.md's Track B specified before committing to the mechanism being
useful -- the field existing and being wired in (already done) doesn't by
itself demonstrate it helps zero-shot generalization.

Usage:
    python scripts/experiment_geometry_aware.py --holdout geometry4 --epochs 60
```

## `scripts/explain_fno.py`

### module

```text
FNO/WHNO spectral mode-importance XAI.

Compares the learned spectral weight distributions of two trained
checkpoints (typically FNO3d vs WHNO3d on the SAME geometry) to test
whether the Walsh-Hadamard basis concentrates more importance in
finer/high-sequency z-bands than Fourier does at low-frequency truncation
-- the mechanism-level evidence for the Gibbs-ringing argument, checked
against real trained weights rather than the synthetic step-function demo.

Usage
-----
python scripts/explain_fno.py     --model-a checkpoints/fno/geometry1_best.pt --label-a FNO     --model-b checkpoints/whno/geometry1_best.pt --label-b WHNO     --geometry geometry1 --model-a-type fno --model-b-type whno     --output results/fno_whno_mode_importance
```

## `scripts/explain_pinn.py`

### module

```text
CLI entry point for PINN explainability analysis.

Runs three complementary explainability methods on a trained checkpoint:

  Option 1 — PDE residual maps: where does the model violate the heat equation?
  Option 2 — Engineering sensitivity maps: power block influence + HTC sensitivity
  Option 5 — MC Dropout uncertainty: where is the model uncertain?

Usage:
    python scripts/explain_pinn.py \
        --checkpoint checkpoints/geometry1/geometry1_best.pt \
        --data data/3d-ice \
        --output results/explain/geometry1 \
        --split test

    # Only specific methods:
    python scripts/explain_pinn.py \
        --checkpoint checkpoints/geometry1/geometry1_best.pt \
        --data data/3d-ice \
        --output results/explain/geometry1 \
        --methods residual sensitivity

    # Control MC Dropout samples (fewer = faster but noisier):
    python scripts/explain_pinn.py ... --mc-samples 50

Outputs per scenario (--split test, default):
    pde_residual_<scenario>.png        PDE residual at die z-slice
    sensitivity_block_<b>_<scenario>.png   per-block dT/dQ influence map
    sensitivity_htc_<scenario>.png     cooling effectiveness map
    uncertainty_<scenario>.png         MC Dropout predictive std
    explain_summary.json               aggregate statistics

Notes:
    MC Dropout requires the model to have been trained with dropout_p > 0
    (the default in build_model). Models trained with dropout_p=0 will
    produce near-zero uncertainty estimates.
```

## `scripts/explain_therm_fm.py`

### FunctionDef:compute_weight_drift

```text
Per-parameter drift: L2 norm of the difference (absolute) and relative
    drift (||delta|| / ||pretrained||). Returns a list of per-parameter
    records, each also tagged with its coarse group.

    Some parameters (e.g. FiLM's final layer, GroupNorm biases) are
    deliberately zero-initialized so the layer starts as an identity
    transform. For these, ||pretrained|| ~ 0 makes relative drift explode
    to a meaningless huge number even for a tiny absolute update. Such
    tensors are flagged 'near_zero_init' and excluded from relative-drift
    aggregation — absolute drift is the only meaningful metric for them.
```

### FunctionDef:plot_drift_summary

```text
Two-panel bar chart: absolute drift norm (always defined) and mean
    relative drift (NaN for groups made entirely of near-zero-init tensors,
    shown as a gap rather than a misleading zero or huge spike).
```

### module

```text
Therm-FM weight-drift XAI: diff a pretrained CNOFNOHybrid checkpoint against
its few-shot fine-tuned result to see WHAT changed to adapt to a new
geometry.

Pure post-hoc checkpoint analysis — no forward passes, no retraining, no
compute cost beyond loading two .pt files. Bucketed by the same parameter
groups finetune_therm_fm.py freezes/unfreezes (encoder, film_gen, decoder,
proj, tail latent_blocks), so drift in the frozen encoder should be exactly
zero — this doubles as a correctness check that freezing actually worked.

Usage
-----
python scripts/explain_therm_fm.py     --pretrained checkpoints/cno_fno/cno_fno_best.pt     --finetuned  checkpoints/therm_fm/therm_fm_best.pt     --output     results/therm_fm_drift
```

## `scripts/finetune_therm_fm.py`

### module

```text
Therm-FM: few-shot fine-tuning of a pre-trained CNOFNOHybrid for a new geometry.

Strategy
--------
Given a CNOFNOHybrid pre-trained on 6-7 geometries (encoder has learned
structural features: conduction paths, TSV influence, boundary layers), we
freeze ~90% of parameters and only update:
  - FiLM generator (maps scenario BCs to per-block gamma/beta modulations)
  - Last 2 FNO blocks of the latent trunk (fine spatial adjustment)
  - Decoder (reconstructs T from the FNO latent)
  - Final projection layer (if any)

This is ~10% of total parameters, allowing 5-20 shot fine-tuning in
<1 minute on a T4 without catastrophic forgetting of the source geometries.

Usage
-----
# 10-shot fine-tune for a new geometry:
python scripts/finetune_therm_fm.py     --pretrained  checkpoints/cno_fno/cno_fno_best.pt     --new-data    data/new_geometry     --geometry    geometry_new     --output      checkpoints/therm_fm     --shots       10     --epochs      100

# All 5 shots (fast iteration):
python scripts/finetune_therm_fm.py     --pretrained checkpoints/cno_fno/cno_fno_best.pt     --new-data   data/new_geometry     --geometry   geometry_new     --shots      5  --epochs 50 --output checkpoints/therm_fm
```

## `scripts/fno_modes_channels_sweep.py`

### module

```text
Modes-vs-channels compute-optimal sweep for baseline FNO.

Motivation (2026-09-08 research session): the neural-operator scaling literature
(e.g. the "optimal Fourier cutoff" line of work summarized against this project's own
config choices) reports that at a FIXED parameter budget, there is a real accuracy
tradeoff between spending that budget on more spectral modes vs. more hidden channels,
and the optimal split is not universal across PDE types. This project has used the same
hardcoded `channels=32, modes=(16,16,12)` for every FNO run on every geometry
(`scripts/train_fno.py`'s default) without ever checking whether that split is actually
a good one relative to other splits at a similar total parameter count. This script is
that check.

Design: hold total parameter count roughly fixed within two tiers (~4.6-4.8M and
~6.1-6.7M params -- close enough for a first-pass comparison, not exactly matched; the
project's own default falls in the second tier), vary (channels, modes) within each
tier, train every candidate for the SAME epoch budget on the SAME data with the SAME
seed (isolates architecture shape as the only variable, same convention as
kaggle_pinn_sampling_comparison.ipynb), and score with the same detrended metrics
scripts/baselines.py uses so results are directly comparable to the ridge reference
(docs/report.md Sec 9.1a, re-measured 2026-09-09: geometry1 ridge det.MAE=0.407,
spatial R^2=0.970, hotspot localisation error 7399 um).

This is a fast, CPU-scale first pass, not a publication run -- same caveat this
project's other CPU-scale FNO smoke tests carry (docs/report.md Sec 9.7 etc.): a
config that loses here might still win at GPU scale/full epoch budget. The question
this script answers is narrower and cheaper: at matched capacity and a matched
(short) training budget, does the project's existing default sit anywhere near the
best point on the modes/channels tradeoff curve, or is it leaving accuracy on the
table for free (same params, different split)?

Usage:
    python scripts/fno_modes_channels_sweep.py
    python scripts/fno_modes_channels_sweep.py --epochs 80 --geometry geometry1
```

## `scripts/gen_geometry7_material_sweep.py`

### module

```text
Interface-uncertainty-style sweep for geometry7's two invented material constants:
organic_substrate (the bare CoWoS-L field, k=0.5 nominal, literature range 0.3-0.8
W/m*K per material.py) and lsi_bridge_via (the bridge/via composite under each
compute die, k=60 nominal, "matching hybrid_bonding's existing precedent, not
independently derived" per material.py -- the most genuinely uncertain constant in
this geometry).

Same methodology as scripts/gen_interface_uncertainty_pilot.py (Track C): hold
power pattern, HTC and ambient FIXED at one real operating point and vary exactly
one material constant at a time, so the resulting spread isolates that constant's
effect rather than conflating it with anything else.

Fixed operating point: the actual hottest scenario from the (post-fix) 40-scenario
geometry7 pilot (geometry7_train_040: uniform pattern, htc=5000, t_amb=45C, peak
181.2C) -- a real, already-validated operating point, not an invented one.

Usage:
    python scripts/gen_geometry7_material_sweep.py --output data/3d-ice-geometry7-material-sweep
```

## `scripts/gen_geometry7_pilot.py`

### module

```text
Generate a 40-scenario pilot dataset for geometry7 (CoWoS-L reticle-stitched
package, see src/core/geometry_builders.py build_geometry7 docstring).

geometry7 is NOT part of the standard 6-geometry benchmark dataset -- it is a
pilot testing whether this benchmark's central "ridge already solves this"
finding survives a structurally new mechanism: sparse high-k (silicon LSI
bridge) islands in an otherwise low-k (organic substrate) passive spreading
layer, replacing every other 2.5D geometry's uniform-material interposer.

40 scenarios = the same 15 base-train + 5 test scenarios every geometry gets
(ScenarioGenerator.generate_all_scenarios), plus 20 extra-train scenarios
(ScenarioGenerator.generate_extra_training_scenarios) drawing denser HTC/power
coverage from the 2.5D-geometry pool -- the same recipe used to grow
geometry1..6 from their base 20 to the full per-geometry counts recorded in
docs/geometry_reference.md.

Usage:
    python scripts/gen_geometry7_pilot.py --output data/3d-ice-geometry7-pilot
```

## `scripts/gen_interface_uncertainty_pilot.py`

### module

```text
Controlled interface-property (TBR-proxy) uncertainty sweep.

Motivation (goal.md "Documentation audit and novelty action plan", 2026-08-16):
the multiscale-3D-IC review [Barua, Udoy & Aziz, arXiv:2604.03290] identifies
the absence of standardized, uncertainty-aware thermal-interface-property data
as an open problem, noting that reported TBR values vary substantially between
measurement groups. Independently, packaging practice reports that modest
interface/packaging changes can shift hotspot location or move peak junction
temperature by double-digit degrees.

If that is true, it bears directly on this benchmark's central argument. Neural
thermal surrogates compete over sub-Kelvin field-error improvements. If plausible
uncertainty in the *interface properties fed to the simulator* moves the answer by
more than the model-vs-model differences being optimized, then that accuracy race
is being run inside the noise floor of its own inputs -- a stronger and more
general version of "ridge already solves this."

This script measures that directly. Crucially, and unlike the TIM k-sweep already
present in the geometry5/6 training data (which varies power pattern, HTC and TIM k
*simultaneously* and therefore cannot isolate anything), every scenario here holds
power pattern, HTC and ambient FIXED and varies exactly one interface property at a
time across its documented uncertainty range.

Ranges are taken from `docs/assumptions.md`, not invented:
  - tim_top (TIM1, indium solder): 5-80 W/m.K, the documented pump-out/degradation
    lifecycle range already used by the generator's k-sweep.
  - tim_sink (thermal grease): 1-8 W/m.K, the literature range for greases
    (assumptions.md gives 4.0 as the nominal, 1-8 as the spread).
  - hybrid_bonding: 60 -> 400 W/m.K. assumptions.md 2.3 states the repo's 60 W/m.K
    value overestimates real Cu-Cu hybrid-bond resistance by ~28x, with real
    bondlines at k ~ 300-400. This sweep spans "what we model" to "what hardware
    actually does" -- i.e. a known modelling-uncertainty axis, not a guess.

Usage:
    python scripts/gen_interface_uncertainty_pilot.py --output data/3d-ice-interface-pilot
```

## `scripts/gen_leakage_pilot.py`

### module

```text
Generate a leakage-feedback pilot batch: the sharpest available test of this
benchmark's central linearity claim.

Throttling (goal.md Track A) is *negative* feedback -- self-limiting, converges in
~2 solves, and degraded the linear baseline only modestly (spatial R^2 0.970 ->
0.890/0.919). Leakage is *positive* feedback: hotter -> more leakage -> hotter, with
no steady state at all above a critical loop gain. If ridge still solves the dataset
under positive electrothermal feedback, the linearity finding is robust rather than
an artifact of a benign regime. If it does not, that is the first mechanism in this
benchmark that genuinely requires a learned operator.

Sweeps leakage aggressiveness across the batch so the dataset spans benign to
near-runaway rather than sitting at one operating point -- the interesting behaviour
is concentrated near the critical gain, and a single setting would miss it.

Usage:
    python scripts/gen_leakage_pilot.py --geometry geometry1 --n 20
```

## `scripts/gen_microchannel_pilot.py`

### module

```text
Generate a pilot batch of microchannel-cooled geometry6 scenarios.

The process_scenario -> NPZExporter pipeline was already confirmed correct for
microchannel-cooled geometries (2026-08-09, see goal.md) -- what was actually
missing was a way to select a microchannel-cooled geometry at all, since none
of the 6 registered geometries have coolant_layer_name set. This builds one
ad hoc (geometry6 + a thin copper base plate below the channel, matching the
pattern already validated against the real 3D-ICE 4.0 binary) and runs a
small scenario sweep through the real pipeline, varying coolant flow rate --
the axis that actually introduces advection, i.e. the one thing in this
benchmark that isn't confined to the linear-conduction regime.

Usage:
    python scripts/gen_microchannel_pilot.py --n 12 --output data/3d-ice-microchannel-pilot
```

## `scripts/gen_moving_source_pilot.py`

### module

```text
Moving-source pilot: the fix for the defect in docs/report.md §9.14, and a controlled test
of the mechanism claimed there.

§9.14 measured that every geometry in this benchmark has exactly ONE heat-source support
pattern across all its scenarios (mean pairwise IoU 1.000). Sources never move, so the
thermal operator is fixed and the solution map is exactly a linear model's hypothesis class.
This script regenerates a geometry's scenarios with per-scenario chiplet placement
(`src/core/placement.py`), which is what IC-ThermBench varies and this benchmark did not.

It is set up as a CONTROLLED experiment, not just a harder dataset, because "move the
sources" is not by itself sufficient to break linearity and it matters to show which part
does the work:

  geometry1  laterally homogeneous silicon die. Moving the power blocks changes where heat
             enters but NOT the medium, so the operator is unchanged and T remains a fixed
             linear functional of the per-cell power map.
             PREDICTION: a linear model given the per-cell power map should stay strong.

  geometry4  2.5D assembly: each chiplet IS a material region (silicon island in k=0.7
             underfill). Moving it rearranges the medium, so the operator changes per
             scenario and no single fixed linear map suffices.
             PREDICTION: the linear model should degrade markedly.

If both degrade equally, the mechanism claimed in §9.14 (operator variation, not source
translation) is wrong and the section needs rewriting. That is the point of running both.

Every scenario is a real 3D-ICE solve; the placement is applied to the geometry handed to
the simulator, so the .stk file, the material map and the exported per-cell power field all
reflect the moved chiplets.

Usage:
    python scripts/gen_moving_source_pilot.py --geometry geometry4 --n 45
```

## `scripts/generate_lf_data.py`

### module

```text
Generate low-fidelity analytical thermal data for all (or selected) geometries.

Runs the 1D-resistance + 2D-Gaussian simulator for every train/test scenario
produced by the standard ScenarioGenerator.  Output is written to data/lf/
in the same NPZ format as the 3D-ICE HF files — drop-in compatible with the
ARO dataset loader.

Usage
-----
# All geometries:
python scripts/generate_lf_data.py --output data/lf

# Specific geometries:
python scripts/generate_lf_data.py --output data/lf     --geometries geometry1 geometry4 geometry5

# Quiet run (no progress bar):
python scripts/generate_lf_data.py --output data/lf --quiet
```

## `scripts/hotspot_eval.py`

### module

```text
Hotspot-focused evaluation, scored with k-fold cross-validation.

Why this exists (2026-09-09). `docs/report.md` Sec 9.1a showed every non-neural baseline
failing hotspot localisation on every geometry (5.0-31.7 mm errors), and Sec 9.12b showed
two CPU-budget FNOs beating ridge on that metric -- the first metric in this project on
which a neural operator wins. Both readings rest on foundations too weak to publish:

  1. n = 5 test scenarios per geometry.
  2. A single argmax-distance number, which is fragile when a field has several
     near-equal hot regions. A diagnostic on the shipped test splits showed this is
     geometry-dependent and matters a lot:

       geometry1: the 100 hottest cells lie within ~300-400 um of the peak. The hotspot
                  is a sharp, well-posed target, so a 7 mm error is a real failure.
       geometry6: the 100 hottest cells spread 2.7-22 mm (median), up to 41.8 mm on a
                  42 mm package -- several near-equal hot regions. Here argmax distance
                  is substantially noise: a model can pick a different-but-nearly-as-hot
                  stack and be scored as catastrophically wrong.

This script fixes both. It pools each geometry's train+test scenarios and runs k-fold CV
so every scenario is scored once as a held-out point (45 test points on geometry1 rather
than 5), and it reports hotspot quality with metrics that survive multi-modal fields:

  peak_temp_err_K       max(pred) - max(true). The thermally critical number -- design
                        point and throttle thresholds key off predicted peak, and this is
                        well-posed no matter how ambiguous the peak's *location* is.
  hotspot_temp_err_K    pred at the TRUE hotspot cell minus truth there.
  hotspot_loc_err_um    argmax-to-argmax distance. Kept for continuity with Sec 9.1/9.12b,
                        but read it against the flatness diagnostic this script prints.
  top1pct_recall        |top-1% hottest predicted cells INTERSECT top-1% hottest true
                        cells| / |top-1% true|. Robust to argmax flipping between
                        near-equal peaks, and the closest thing here to "did the model
                        find the hot regions".
  hit_within_1mm/2mm    fraction of scenarios whose predicted peak lands within tolerance.

Also reported per geometry, so a reader can judge whether hotspot_loc_err_um means
anything on that geometry:

  peak_spread_um        median distance of the 100 hottest TRUE cells from the true peak.
  dT_top100_K           temperature gap between the hottest and 100th-hottest true cell.

Baselines come from scripts/baselines.py::predict_all -- the same fitted implementations
scored in Sec 9.1a, not a reimplementation.

Usage:
    python scripts/hotspot_eval.py --geometry geometry1
    python scripts/hotspot_eval.py --geometry geometry1 geometry6 --folds 5
    python scripts/hotspot_eval.py --geometry geometry7 --data data/3d-ice-geometry7-pilot
```

## `scripts/hotspot_eval_fno_cv.py`

### module

```text
k-fold cross-validated FNO vs. ridge/kNN on hotspot metrics.

Companion to scripts/hotspot_eval.py, which does the same thing for the non-neural
baselines only. This one trains an FNO per fold, so it is expensive (hours on CPU) and
must be run in the background.

Why: docs/report.md Sec 9.12b recorded two CPU-budget FNOs beating ridge on hotspot
localisation -- the first metric in this project on which a neural operator wins -- but
on only 5 test scenarios, and scored on argmax distance alone. Re-scoring those same
checkpoints with the fuller metric set (2026-09-09) showed the win is real but much
narrower than it looked:

    model                   |peak err| K   loc err um   top1% recall   <=2mm
    ridge                          2.132         7169          0.032    0.00
    kNN                            1.777         7616          0.038    0.00
    FNO-default                   14.860         4469          0.025    0.40
    FNO-tierB-wide-narrow          8.060         3324          0.054    0.20

i.e. FNO localises the hotspot better (3.3-4.5 mm vs ridge's 7.2 mm, and the only
non-zero hit rates within 2 mm) while being 4-7x WORSE at predicting the peak
temperature. Knowing where the hotspot is but being 15 K wrong about how hot it is, is
not straightforwardly more useful than the reverse. Both halves need saying.

Those numbers are n=5. This script gets the sample size up by training one FNO per fold
on the SAME fold split scripts/hotspot_eval.py uses (same seed, same kfold_indices call),
so FNO and the baselines are scored on identical held-out scenarios.

Usage (expect ~6 h on CPU for the defaults; run it in the background):
    python scripts/hotspot_eval_fno_cv.py --geometry geometry1 --folds 5 --epochs 150
```

## `scripts/ic_thermbench_baselines.py`

### ClassDef:FittedRidge

```text
Everything needed to predict on new data with a fitted ridge-green model.

    Kept so S5 can be scored zero-shot: their S5 protocol freezes an S4-trained model
    and applies it to unseen cases with unchanged preprocessing statistics, so the PCA
    bases and standardisation must come from S4 and must NOT be refitted on S5.
```

### FunctionDef:fit_pca

```text
Leading principal components of centred training fields. Returns (mean, basis).

    Randomised range-finder rather than a full SVD: we only ever want the leading
    ~256 directions of a 10,800 x 4,096 matrix, and a full economy SVD costs
    O(n*m*min(n,m)) ~ 1.8e11 flops against ~1.1e10 here. Uses one power iteration,
    which is ample for the strongly-decaying spectra these thermal fields have.
```

### FunctionDef:geometry_keys

```text
A stable id per sample for its substrate geometry, from the grid_x/grid_y channels.

    These encode the (non-uniform) grid the case is discretised on, and measurement shows
    they take only ~20 distinct values across thousands of samples while the power map
    varies per sample. Hashing the pair therefore recovers the layout/case grouping that
    the released files do not label directly.
```

### FunctionDef:ridge_gram

```text
Precompute the lambda-independent normal-equation blocks, in chunks.

    AtA and AtY are the expensive parts (O(n*p^2) and O(n*p*c)) and neither depends on
    lambda, so they are computed once and reused across the whole lambda sweep. Only the
    p x p solve repeats -- a 5x saving on the dominant cost for a 5-value sweep.

    Accumulated over row chunks so the float64 copy of A never exists in full. At S4's
    8,709 features a full float64 A is ~750 MB on top of everything else, and this
    machine has ~4 GB free; chunking keeps the transient at ~100 MB while giving
    bit-comparable results (the sum order changes, nothing else). A itself may be
    float32 -- the accumulation is done in float64 regardless, which is what matters
    for the conditioning of the normal equations.
```

### FunctionDef:run_transfer

```text
Zero-shot transfer: fit on `source`, evaluate on `target` without refitting.

    Mirrors their S5 protocol -- a frozen source-trained model, unchanged preprocessing
    statistics, scored on all target samples.
```

### module

```text
Non-neural baselines on IC-ThermBench S2-S5.

See docs/ic_thermbench_plan.md. IC-ThermBench's eight baselines are all deep networks;
this supplies the missing closed-form row, scored with their own metric code
(`third_party/ic_thermbench/metrics.py`, vendored verbatim) on their own splits
(`scripts/ic_thermbench_data.py`, verified array-exact against their loader).

Baselines
---------
ridge-green   The physically-motivated linear model: a dense ridge operator from the
              4,096-cell power map (plus the per-cell conductivity map where the scope
              has one, plus per-sample scalars) to the 4,096-cell temperature field.
              Steady-state conduction IS linear in the source, so for a fixed operator
              this is the exact solution form, not an approximation:
                  T(x) = T_amb + sum_x' G(x,x') Q(x')
              The intercept absorbs any fixed offset. This is the number that matters.

ridge-pca     The same idea with the spatial maps compressed to their leading principal
              components. Far fewer parameters; guards against the objection that
              ridge-green only wins because it is large.

knn / nn      Inverse-distance-weighted / single nearest training field, in PCA feature
              space (raw 4,096-d distances over 10,800 training samples are needlessly
              slow and no more meaningful).

mean          Training-set mean field. The floor.

Design notes
------------
- Ridge lambda is selected on THEIR validation split, mirroring the validation-best
  checkpoint selection their models use. Not tuned on test.
- Features are standardised; the intercept is never penalised (same convention as
  scripts/baselines.py).
- `grid_x` / `grid_y` are per-cell geometry encodings that vary per sample (they carry
  layout identity). They are included as PCA components rather than raw 4,096-dim blocks
  by default: including them raw pushes the feature count past the training-set size,
  which is a different and less interpretable regime. `--raw-grid` opts into it.

Usage
-----
    python scripts/ic_thermbench_baselines.py --scope level2
    python scripts/ic_thermbench_baselines.py --scope level2 level3 level4 --output results/ic_thermbench_baselines.json
```

## `scripts/ic_thermbench_data.py`

### FunctionDef:load_scope

```text
Load one scope.

    `split_data=False` returns everything in the test slot with train/val empty --
    the correct handling for S5, which their docs describe as a pure evaluation
    dataset that must not go through the train_ratio split.
```

### FunctionDef:verify_against_upstream

```text
Check our arrays match theirs exactly, using their code as the reference.

    This is the parity test docs/ic_thermbench_plan.md §6 step 2 asks for. It needs
    their repo checked out (and torch, which their loader returns tensors from), so
    it is a separate opt-in function rather than an import-time dependency.

    Returns True on exact match; raises AssertionError describing the first
    disagreement otherwise.
```

### module

```text
Loader for IC-ThermBench S2-S5, reproducing their split exactly.

See docs/ic_thermbench_plan.md for why this experiment exists. This module is
deliberately small and does one thing: turn their released `.mat` files into
train/val/test arrays that are *index-for-index identical* to what their own
`data_provider/data_loader.py` produces, without depending on their repo at runtime.

Their pipeline, reproduced here (verified against their source 2026-09-10):

    load_mat_pair()          input.mat["data"]  (B, P, Z, Y, X) float32
                             output.mat["data"] (B,    Z, Y, X) float32, kelvin
                             transposed to      (B, X, Y, Z, P) and (B, X, Y, Z)

    split_train_val_test()   train_ratio = 0.8 (data_factory.TRAIN_RATIO, "shared by
                             the whole benchmark; changing it invalidates every result")
                               trainval = first int(0.8 * B) samples
                               test     = everything after that      -> last 20%
                               n_train  = int(0.9 * trainval)
                               train    = trainval[:n_train]
                               val      = trainval[n_train:]

The split slices by index and never shuffles. Their archives store samples
case-interleaved so each segment stays case-balanced; restacking by case before
splitting would silently change the semantics while leaving sample counts intact.
`verify_against_upstream()` checks our arrays against theirs when their repo is
available, so this claim is tested rather than asserted.

Channel semantics (measured, docs/ic_thermbench_plan.md §2): `chiplet_power`,
`grid_x`, `grid_y` and `local_thermal_k` are genuine per-cell maps; `ambient_K`,
`h_w_m2k` and `r_convec_k_per_w` are constant within a sample and vary across
samples, i.e. per-sample scalars.
```

## `scripts/make_ood_split.py`

### module

```text
Build out-of-distribution (OOD) evaluation splits from scenario metadata.

Why this exists
---------------
The shipped `*_test_*.npz` files are interpolation points inside the same
parameter sweep as training: same power patterns, HTC and ambient values drawn
from the middle of the training ranges. They measure interpolation, not
generalisation, and every model looks good on them.

These splits hold out a whole *axis* instead, and write a manifest rather than
copying NPZ files (the data is large; the split is just a list of names).

Axes
----
pattern    Train on the smooth patterns (uniform, checkerboard, gradient);
           test on the peaked ones (hotspot, dual_hotspot, extreme_hotspot).
           Tests whether a model extrapolates to unseen spatial structure.

power      Train on the low-power half, test on the high-power half. This is
           the axis that can actually break a linear model: steady-state
           conduction is linear in power only while k(T) is constant, so
           nonlinearity appears exactly where self-heating is largest.

htc        Train on mid-range cooling, test on the extremes. Temperature depends
           on 1/h, so this probes extrapolation in a reciprocal coordinate.

ambient    Train on low ambient, test on high ambient. Expected to be easy --
           ambient enters as a near-exact additive offset -- and included as a
           control: a split that everything passes.

Usage
-----
    python scripts/make_ood_split.py --geometry geometry1 --axis power
    python scripts/baselines.py --geometry geometry1 \
        --split-manifest results/ood_geometry1_power.json
```

## `scripts/patch_tsv_power.py`

### module

```text
One-time patch: zero out power at TSV-region blocks in geometry2/geometry5 NPZ files.

Bug: generate_power_density_field() did not check block.is_tsv_region, so TSV-region
blocks (is_tsv_region=True) were assigned nonzero power in NPZ files even though
3D-ICE assigns them zero heat (they're passive conductors, excluded from floorplans).
This inconsistency corrupted the PINN PDE loss at those points.

Fix applied in mesh.py (generate_power_density_field) and generator.py (block_names).
This script patches existing NPZ files by recomputing the power field with the fix.
```

## `scripts/regenerate_failed.py`

### module

```text
Regenerate NPZ files where 3D-ICE silently fell back to synthetic temperatures.

Detection: files whose point count != the 3D-ICE Tmap grid count
  (n_layers * nx * ny, i.e. 1 z-point per layer).

Each bad file's scenario parameters are preserved from its metadata so the
same scenario ID keeps the same power pattern, HTC, and ambient temperature.
The bad file is deleted only after the new 3D-ICE run succeeds.
```

## `scripts/test_ice_connection.py`

### module

```text
Test 3D-ICE connectivity: verify the executable is reachable and generate a
minimal config to confirm the simulator runs without error.

Usage:
    # With 3D-ICE-Emulator in PATH
    python scripts/test_ice_connection.py

    # With WSL-wrapped binary
    python scripts/test_ice_connection.py --executable "wsl /home/user/3d-ice/bin/3D-ICE-Emulator"

    # Full smoke test (generates config + runs simulation)
    python scripts/test_ice_connection.py --executable "wsl ..." --run
```

## `scripts/test_ridge_tim_k_feature.py`

### module

```text
Follow-up to analyze_interface_linearity.py's finding that peak-T is linear in 1/k_tim,
not k. That result predicts a specific, testable claim: a ridge model given 1/k_tim as an
explicit feature (mirroring the 1/htc feature scripts/baselines.py already uses, and for
the same physical reason -- a series thermal resistance enters T linearly through its
reciprocal conductivity) should recover the interface-uncertainty axis almost exactly,
where a ridge model blind to k cannot represent it at all and is wrong by construction
whenever k differs from whatever the (single, fixed) training value was.

This was flagged as "untested here" in docs/report.md Sec 9.9 and is tested directly now,
using ONLY existing sweep data (data/3d-ice-interface-multi + data/3d-ice-interface-pilot-
median geometry4/tim_die, 10 scenarios: 5 k-values x {high-power, median-power}) -- no new
3D-ICE solves. Leave-one-scenario-out CV (only 10 points -- a real train/test split would
leave too little data at either end of the k range to be meaningful).

Three feature sets compared:
  blind    -- same features scripts/baselines.py's ridge already uses (block powers, htc,
              1/htc, t_ambient, tsv_density). No k signal at all.
  +k       -- blind, plus the raw k value.
  +1/k     -- blind, plus 1/k -- the physically-motivated feature.

Usage:
    python scripts/test_ridge_tim_k_feature.py
```

## `scripts/train_aro.py`

### module

```text
Train the Autoregressive Operator (ARO) thermal surrogate.

Multi-fidelity training pipeline:
  1. Pre-train on LF (analytical) + HF (3D-ICE) data with teacher forcing
  2. Fine-tune on HF only with teacher forcing disabled

Example
-------
# All geometries, multi-fidelity (LF pretrain + HF fine-tune):
python scripts/train_aro.py     --hf-data  data/3d-ice     --lf-data  data/lf     --geometries geometry1 geometry2a geometry3     --pretrain-epochs 200     --finetune-epochs 100     --output checkpoints/aro

# HF only (no LF):
python scripts/train_aro.py     --hf-data  data/3d-ice     --geometries geometry1     --pretrain-epochs 300     --finetune-epochs 0     --output checkpoints/aro_hf_only
```

## `scripts/train_deeponet.py`

### module

```text
CLI entry point for PI-DeepONet training.

One model — all 5 uniform-stack geometries simultaneously.
geometry4/5 (2.5D chiplet assemblies with lateral conductivity variation) are
excluded: the trunk's (x,y,z,layer_id) coordinates cannot represent sharp
temperature gradients at chiplet boundaries without explicit region encoding.
geometry4/5 use per-geometry CNO-FNO models instead.

Usage
-----
# Train cross-geometry PI-DeepONet on all 5 stack geometries (default)
python scripts/train_deeponet.py \
    --data data/3d-ice \
    --output checkpoints/deeponet/ \
    --epochs 1000 \
    --pde-weight 0.1

# Data-only baseline (no physics loss) — pure DeepONet
python scripts/train_deeponet.py \
    --data data/3d-ice \
    --output checkpoints/deeponet/ \
    --pde-weight 0.0

# Subset of geometries (e.g. to compare against single-geometry FNO)
python scripts/train_deeponet.py \
    --geometries geometry1 geometry3 \
    --data data/3d-ice \
    --output checkpoints/deeponet/

# CPU-fast (smaller model, fewer epochs)
python scripts/train_deeponet.py \
    --data data/3d-ice \
    --output checkpoints/deeponet/ \
    --cpu-fast

Key advantages over per-geometry FNO
-------------------------------------
- Single model: train once, evaluate on all 5 stack geometries
- Arbitrary query resolution: trunk evaluates at any (x,y,z), not fixed grid
- Better parametric extrapolation via physics loss on trunk gradients
- ~3x cheaper per epoch than FNO (no 3D FFT overhead)
```

## `scripts/train_fno.py`

### module

```text
CLI entry point for FNO training.

Model variants (--model):
    fno       Baseline FNO3d. Fast, data-only, no physics or BC conditioning.
    cond-fno  FiLM-conditioned FNO. HTC/T_amb/TSV_frac modulate spectral
              filters via a hypernetwork. Better BC extrapolation than fno.
    cno-fno   CNO-FNO hybrid (default). CNN encoder/decoder around a
              FiLM-FNO in latent space. Best accuracy: sharp interfaces
              (CNO) + global spreading (FNO) + BC conditioning (FiLM).
              Physics loss enabled by default.

Usage:
    # Recommended (CNO-FNO + physics, default model):
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice

    # Higher capacity (publication runs):
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --channels 64

    # Baseline FNO for ablation:
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --model fno

    # All geometries:
    python scripts/train_fno.py --all-geometries --data data/3d-ice

    # CPU-only (reduced defaults):
    python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --cpu-fast
```

## `scripts/train_pinn.py`

### module

```text
CLI entry point for PINN training.

Usage:
    python scripts/train_pinn.py \
        --geometry geometry1 \
        --data data/3d-ice \
        --output checkpoints/ \
        --epochs 8000 \
        --n-col 20000 \
        --fourier-sigma 10.0 \
        [--device cuda]

The script:
  1. Loads train/test .npz files for the given geometry
  2. Computes or loads normalization statistics
  3. Trains a FourierPINN with curriculum staging
  4. Saves best checkpoint + norm_stats.json to --output
```

## `scripts/validate_dataset.py`

### module

```text
Validate a generated NPZ dataset for shape, key, and physical correctness.

Usage:
    python scripts/validate_dataset.py data/mock
    python scripts/validate_dataset.py data/3d-ice --strict
```

## `src/__init__.py`

### module

```text
3D-IC Thermal PINN Benchmark System

A comprehensive data generation pipeline for Physics-Informed Neural Network (PINN)
training on 3D integrated circuit thermal problems.
```

## `src/aro/data_loader.py`

### ClassDef:ARODataset

```text
Dataset of layered (Q, T, k) stacks for ARO training.

    Each item is a dict with:
        Q_stack:   (n_layers, H, W)  normalised volumetric power density
        T_stack:   (n_layers, H, W)  normalised temperature
        k_norms:   (n_layers,)       normalised thermal conductivity
        cond:      (4,)              BCs: htc_norm, t_amb_norm, tsv_frac, tim_k_norm
        is_lf:     bool              True if item comes from LF dataset
        geom_name: str
        scenario_name: str
        n_layers:  int
```

### module

```text
Dataset for Autoregressive Operator (ARO) training.

Loads NPZ files (3D-ICE HF or LF analytical) and returns per-geometry
stacks of shape (n_layers, H, W) for each of Q, T, k_norm.

Supports mixed HF + LF datasets:
    hf_files: List[Path]  — real 3D-ICE NPZ files
    lf_files: List[Path]  — analytical LF NPZ files

Each NPZ item carries:
    Q_stack:   (n_layers, H, W) float32  — normalised power per layer
    T_stack:   (n_layers, H, W) float32  — normalised temperature per layer
    k_norms:   (n_layers,) float32       — normalised thermal conductivity
    cond:      (4,) float32              — (htc_norm, t_amb_norm, tsv_frac, tim_k_norm)
    is_lf:     bool                      — True if this item is low-fidelity
```

## `src/aro/model.py`

### ClassDef:ARO

```text
Autoregressive Operator: shared AROBlock applied sequentially per z-layer.

    Forward inputs (batch of scenarios):
        Q_stack:    (B, n_layers, H, W)  normalised power per layer
        k_norms:    (B, n_layers)        normalised thermal conductivity per layer
        cond:       (B, cond_dim)        BCs (htc_norm, t_amb_norm, tsv_frac, tim_k_norm)
        region_map: (B, H, W) optional  integer region IDs; embedded if provided

    Forward output:
        T_stack:    (B, n_layers, H, W)  normalised temperature per layer

    Teacher forcing (training):
        Pass `T_gt_stack` to replace predicted T[i-1] with ground-truth.
```

### ClassDef:AROBlock

```text
Shared 2D FiLM-FNO block, weight-tied across all z-layers.

    Input  channels (in_ch):
        0   : Q[i]    (normalised power)
        1   : T[i-1]  (temperature from layer below; 0 at bottom)
        2   : k[i]    (layer thermal conductivity, broadcast from scalar)
        (optional) 3: region_id map for chiplet geometries

    Output channels (out_ch = 1): T̂[i] (normalised temperature prediction)
```

### FunctionDef:forward_windowed_rollout

```text
RNO-style training pass (Recurrent Neural Operators, Yang et al. 2025):
        recursively feed the model its OWN predictions (no ground truth) over a
        window of `window` consecutive layers, keeping gradients attached through
        the whole window (short BPTT). This exposes the model to its own
        compounding error DURING training, closing the train/inference gap that
        plain teacher forcing leaves open.

        The window start is chosen randomly; T[start-1] (input to the window) is
        taken from ground truth (or zeros at start=0) — only the interior of the
        window is self-fed. Layers before `start` are not computed (cheaper than
        a full n_layers rollout every step).

        Returns:
            T_window:  (B, window, H, W) predictions for layers [start, start+window)
            start:     window start index (for indexing T_gt_stack when computing loss)
            window:    actual window length (clamped to n_layers - start)
```

### module

```text
Autoregressive Operator (ARO) for 3D-IC thermal prediction.

Architecture
------------
ARO processes the chip stack layer by layer in the z-direction (bottom → top).
A single shared 2D FiLM-FNO block predicts the temperature slice T[i] from:
  - Q[i]:   power map at layer i         (nx, ny)
  - T[i-1]: temperature map from layer i-1  (nx, ny)  [zeros at i=0]
  - k[i]:   per-layer normalised thermal conductivity  (scalar)
  - bc:     scenario BCs: (htc_norm, t_amb_norm, tsv_frac, tim_k_norm)  (4,)

Layer-to-layer autoregression captures vertical heat spreading that a single
forward-pass 2D model would miss.  Because the block is shared (weight-tied)
across all layers, parameter count is O(ch²) instead of O(n_layers × ch²).

Why 2D FNO (not 3D)?
---------------------
3D FNO requires the full volumetric grid in memory and scales O(nx·ny·nz) in
activation storage.  ARO splits the 3D problem into nz sequential 2D steps,
each O(nx·ny), with only two 2D slices (T[i-1], Q[i]) on the GPU at a time.
This lets geometry6 (56×168 footprint, 11 layers) run on a 16 GB T4 with
ch=32.

Multi-fidelity usage
--------------------
Pre-train on large LF dataset (generate_lf_dataset in lf_simulator.py),
then fine-tune on the smaller 3D-ICE HF dataset.  Because the LF and HF
data share the same NPZ format and spatial coordinates, no dataset
adapter is needed — just different file lists.

Parameter count (ch=32): ~0.85 M  (suitable for a single T4 in 1-2 h)
```

## `src/aro/trainer.py`

### ClassDef:AROTrainer

```text
Trainer for the Autoregressive Operator.

    Args:
        model:            ARO instance
        train_dataset:    ARODataset (HF + optional LF)
        val_dataset:      ARODataset (HF validation scenarios)
        output_dir:       where to save checkpoints
        batch_size:       training batch size
        lr:               Adam peak learning rate
        lr_finetune_factor: factor to reduce LR for the fine-tune stage
        val_interval:     validate every N epochs
        log_interval:     print training loss every N epochs
        grad_clip:        max gradient norm
        amp:              use automatic mixed precision (recommended on T4)
```

### FunctionDef:_stage_train

```text
Each step optimises a combined loss:
            L = L_teacher_forced + rno_weight * L_windowed_rollout

        L_teacher_forced: standard full-stack pass (as before), tf_ratio decaying.
        L_windowed_rollout: RNO-style pass — the model predicts `window`
            consecutive layers using only its OWN prior outputs (see
            ARO.forward_windowed_rollout). Window length grows linearly from
            rno_window_min to rno_window_max over the stage (curriculum: short
            windows are easy/stable early, full-length windows match inference
            exactly by the end of fine-tuning).
```

### module

```text
ARO Trainer — multi-fidelity RNO-style training loop.

Training stages
---------------
1. Pre-train on LF + HF combined (or LF only if lf_only_pretrain=True).
   Each epoch mixes two objectives:
     (a) Standard teacher-forced full-stack pass — tf_ratio decays 1.0 -> 0
         over pretrain_epochs, as before.
     (b) RNO windowed self-rollout pass (Yang et al., "Recurrent Neural
         Operators: Stable Long-Term PDE Prediction", 2025) — the model
         predicts a window of `rno_window` consecutive layers using ONLY its
         own prior predictions (no ground truth injected inside the window),
         with gradients flowing through the whole window. This exposes the
         model to compounding self-error DURING training, which is the
         documented root cause of exposure bias in autoregressive rollout
         (train/inference mismatch: teacher-forced training vs self-fed
         inference). Window length grows via `rno_window_schedule` as
         training progresses (curriculum: short windows early, full-stack
         windows late).

2. Fine-tune on HF only with teacher forcing disabled (tf_ratio=0) and
   windowed rollout at full window length (== n_layers, i.e. matches
   inference exactly). Learning rate is reduced by lr_finetune_factor.

Usage::

    from src.aro.trainer import AROTrainer
    trainer = AROTrainer(model, train_ds, val_ds, output_dir='checkpoints/aro')
    trainer.train(pretrain_epochs=200, finetune_epochs=100)
```

## `src/core/geometry.py`

### ClassDef:DiePrint

```text
Footprint of a single chiplet on a shared interposer (2p5d_stack geometries).

    Defines which (x, y) region of the die_layer_name layer contains Si die
    material vs underfill/gap material.
```

### ClassDef:Geometry

```text
Complete geometry specification for thermal simulation.

    Attributes:
        name: Geometry identifier (e.g., 'geometry1', 'geometry2a')
        geometry_type: Type identifier ('2d_stack' or '3d_stack')
        layers: List of Layer objects from bottom to top
        power_blocks: List of PowerBlock objects
        die_width: Die width in μm
        die_length: Die length in μm
        mesh_resolution: Grid resolution (nx, ny, nz)
        tsv_density: TSV area fraction (0.0-1.0), 0 for no TSVs
```

### ClassDef:Layer

```text
Represents a single layer in the thermal stack.

    Attributes:
        name: Layer identifier (e.g., 'die', 'tim_top', 'spreader')
        material: Material name (references MaterialLibrary)
        thickness: Layer thickness in micrometers (μm)
        k_thermal: Thermal conductivity in W/m·K
        volumetric_heat_capacity: Volumetric heat capacity in J/m³·K
        z_bottom: Bottom z-coordinate in μm (set during stack assembly)
        z_top: Top z-coordinate in μm (set during stack assembly)
        is_active: Whether this layer has power dissipation
```

### ClassDef:PowerBlock

```text
Represents a power dissipation region in a die layer.

    Attributes:
        name: Block identifier
        x: Bottom-left x-coordinate in μm
        y: Bottom-left y-coordinate in μm
        width: Block width in μm
        height: Block height in μm
        power_density: Power density in W/cm² (set by scenario)
        is_tsv_region: True if this block represents a TSV array region
        die_index: Die index for multi-die stacks (0-based)
        layer_name: Name of the associated die layer
```

### FunctionDef:get_layer_at_z

```text
Find the layer containing the given z-coordinate.

        Uses half-open intervals [z_bottom, z_top) for interior layers.
        The top surface of the last layer (z == total height) is included
        so that convective BC points are correctly assigned.

        Args:
            z: Z-coordinate in μm

        Returns:
            Layer object or None if not found
```

### FunctionDef:get_layer_index_at_z

```text
Get the index of the layer containing z-coordinate.

        Uses half-open intervals [z_bottom, z_top) for interior layers.
        The top surface of the last layer (z == total height) is included.

        Args:
            z: Z-coordinate in μm

        Returns:
            Layer index (0-based) or -1 if not found
```

### FunctionDef:get_power_block_at_xy

```text
Find the power block containing point (x, y).

        Args:
            x, y: Coordinates in μm

        Returns:
            PowerBlock object or None if point is not in any block
```

### FunctionDef:power_watts

```text
Calculate total power in Watts.

        Args:
            power_density_wcm2: Override default power density (W/cm²)

        Returns:
            Total power in Watts
```

### FunctionDef:validate

```text
Raise ValueError if the geometry has any structural inconsistency.

        Checks (fatal):
          1. Layer z-ranges are contiguous (no gaps or overlaps)
          2. No duplicate layer names
          3. PowerBlock.layer_name references exist in the layer stack
          4. Power blocks lie within the die footprint

        Checks (warning only):
          5. Power block pairs that overlap by more than 5% of the smaller block's area
```

### module

```text
Geometry data structures and builders for thermal simulations.

Defines the core classes for representing thermal stack geometries:
- Layer: Single material layer in the thermal stack
- PowerBlock: Power dissipation region
- Geometry: Complete geometry specification with layers and power map
```

## `src/core/geometry_builders.py`

### FunctionDef:build_all_geometries

```text
Build all benchmark geometries.

    Returns:
        List: [geometry1, geometry2a, geometry3, geometry4, geometry5, geometry6]
```

### FunctionDef:build_geometry1

```text
Build Geometry 1: 2D Cross-Sectional Stack.

    Layer stack (bottom to top):
    1. Heat Sink (Cu): 5000 μm   -- convective (HTC) ground-truth BC applied
       here by 3D-ICE ("bottom heat sink" directive in ice_simulator.py);
       this layer genuinely removes heat to ambient in the real simulation.
    2. TIM (bottom): 100 μm
    3. Heat Spreader (Cu): 1000 μm
    4. TIM (top): 100 μm
    5. Silicon Die: 150 μm

    NOTE: src/pinn/trainer.py's own physics-loss BC term (bc_residual_top,
    via self.top_layer_id) currently enforces a convective condition at the
    OPPOSITE end of the stack (topmost layer, near the die) instead of here
    at heat_sink -- a real mismatch with the ground-truth physics above,
    not yet fixed. See project notes for details before trusting PDE/BC loss
    diagnostics from PINN training on this geometry.

    Floorplan:
    - Die: 10mm × 10mm
    - 4 power blocks: 3mm × 3mm each (positioned in corners)
```

### FunctionDef:build_geometry2

```text
Build Geometry 2 variants: 3D Stack with TSV.

    Args:
        tsv_density: TSV area fraction (only 'geometry2a' at 0.03 is built; the
            function keeps the density parameter so a variant is one call away)
        variant_name: Variant identifier (only 'geometry2a' is currently used)

    Layer stack (bottom to top):
    1. Heat Sink (Cu): 5000 μm
    2. TIM: 100 μm
    3. Heat Spreader (Cu): 1000 μm
    4. TIM: 100 μm
    5. Die 2 Active (Si): 50 μm
    6. Die 2 TSV Region: 100 μm (enhanced k)
    7. Hybrid Bonding Layer: 5 μm (Cu-Cu pillar array, k_eff=60 W/m·K)
    8. Die 1 TSV Region: 100 μm (enhanced k)
    9. Die 1 Active (Si): 50 μm

    Floorplan:
    - Die: 8mm × 8mm
    - Per die: 2 cores (2.5mm × 2.5mm) + 1 TSV region (2mm × 2mm)
```

### FunctionDef:build_geometry2a

```text
Build Geometry 2a: 3D Stack with 3% TSV density.

    geometry2b (5%) and geometry2c (10%) were removed 2026-08-06: they were the
    same base 3D-TSV-stack geometry differing only in this one scalar, which
    (a) is not exposed as a model input anywhere in the pipeline and (b) produced
    bit-identical ridge-regression baselines to geometry2a (spatial R^2=0.991,
    MAE=2.207 to 3 decimals, all three) -- three near-zero-marginal-information
    copies of one benchmark. TSV density variation is still exercised as a
    spatial field within this single geometry via ScenarioGenerator.attach_tsv_maps.
```

### FunctionDef:build_geometry3

```text
Build Geometry 3: Server-class single die (25mm × 25mm).

    Represents a high-end CPU/GPU die at 4× the area of geometry1.
    Same layer structure as geometry1 (plus TIM2) but with a thicker spreader
    and 8 power blocks arranged as two rows of 4 core clusters separated by an
    IO/interconnect gap — a topology absent from the mobile-class geometries.

    Layer stack (bottom to top):
    1. Heat Sink (Cu): 5000 µm
    2. TIM (bottom): 100 µm
    3. Heat Spreader (Cu): 2000 µm  (thicker than geometry1 for server thermal budget)
    4. TIM (top): 100 µm
    5. Die (Si): 200 µm
    6. TIM2: 50 µm  (die-to-cooler interface)

    Power blocks — 2 rows × 4 cols, each 4mm × 4mm:
      Row 1 (y=2000): x = 2000, 7500, 13000, 18500
      Row 2 (y=16000): same x positions
      Gap y∈[6000,16000] = 10mm represents IO/crossbar area
```

### FunctionDef:build_geometry4

```text
Geometry 4: 2.5D chiplet assembly — two chiplets on a silicon interposer.

    Interposer footprint: 25mm × 14mm
    Chiplet A (compute, 10×12mm): x=2mm, y=1mm — 4 power blocks (2×2)
    Chiplet B (IO,     8×12mm): x=15mm, y=1mm — 4 power blocks (2×2)
    Gap between chiplets: 3mm (x: 12mm to 15mm)

    Layer stack (bottom to top):
      heat_sink   5000µm Cu   — full interposer footprint
      tim_sink     100µm TIM  — full
      spreader    1000µm Cu   — full
      tim_die      100µm TIM  — full
      die_zone     150µm      — Si inside chiplet footprints, underfill (k=0.7) in gap
      interposer   100µm Si   — full (shared routing substrate)

    Total height: 6450µm   Mesh: 100×56×40 = 224k points (250µm/cell xy)
```

### FunctionDef:build_geometry5

```text
Geometry 5 (Tier 0+1 upgraded): CoWoS-style HBM-stack + compute die on shared interposer.

    Same interposer footprint as geometry4 (25mm × 14mm).
    Chiplet A (compute, 10×12mm, x=2mm, y=1mm): single 50µm low-k Si die — die_zone_1 only.
    Chiplet B (HBM-style, 8×12mm, x=15mm, y=1mm): two-die TSV stack:
      die1=50µm (die_zone_1) + TSV=100µm composite (tsv_zone) + die2=50µm (die_zone_2).

    Tier 0 upgrades applied:
      - RDL power layer (rdl_layer, 5µm, active) added above interposer
      - Die k reduced to 80 W/m·K (low-k dielectric composite at N5/N3 node)

    Tier 1 upgrades applied:
      - C4 bump array (c4_bumps, 100µm, k=15 W/m·K) between RDL and die_zone_1
      - TIM1 updated to indium solder (tim_top, 50µm, k=80 W/m·K)
      - TIM2 thickness updated to 125µm (tim_sink)
      - Interposer thickness updated to 300µm (Kou 2022 published dims)

    Layer stack (bottom to top):
      heat_sink   5000µm Cu         — full
      tim_sink     125µm TIM grease — full  (TIM2, Tier 1)
      spreader    1000µm Cu         — full
      tim_top       50µm In solder  — full  (TIM1, Tier 1; k=80)
      interposer   300µm Si         — full  (Tier 1 published dims)
      rdl_layer      5µm Si low-k   — full, active  (Tier 0 RDL self-heating)
      c4_bumps     100µm k=15       — full  (Tier 1 C4 bump array)
      die_zone_1    50µm Si low-k   — active; A+B footprints, underfill in gap
      tsv_zone     100µm TSV        — B only, underfill elsewhere
      die_zone_2    50µm Si low-k   — active; B only, underfill elsewhere

    Total height: 6785µm   Mesh: 100×56×50 ≈ 280k points (11 layers, +hybrid_bonding 5µm)
```

### FunctionDef:build_geometry6

```text
Geometry 6: geometry5 + 6 HBM stacks — CoWoS-style with full HBM complement.

    Extends geometry5 to 6 HBM stacks side-by-side on the interposer, matching the
    HBM count of AMD MI300X (6× HBM3).  Chiplet A (compute) is unchanged.
    Each HBM stack is 4×12mm with its own die_zone_1 / hybrid_bonding / tsv_zone /
    die_zone_2 floorplan blocks.

    Footprint: 42mm × 14mm
      chipA  : x=1mm,  y=1mm, 10×12mm
      hbm1   : x=12mm, y=1mm,  4×12mm
      hbm2   : x=17mm, y=1mm,  4×12mm
      hbm3   : x=22mm, y=1mm,  4×12mm
      hbm4   : x=27mm, y=1mm,  4×12mm
      hbm5   : x=32mm, y=1mm,  4×12mm
      hbm6   : x=37mm, y=1mm,  4×12mm  (right margin 1mm → 42mm total)

    Layer stack: identical to upgraded geometry5 (11 layers, Tier 0+1 + hybrid bonding).
    Total height: 6785µm   Mesh: 56×168×50 ≈ 470k points (250µm/cell in x and y)
```

### FunctionDef:build_geometry7

```text
Geometry 7: CoWoS-L-class reticle-stitched package (Rubin/Rubin-Ultra-like).

    Structurally distinct from geometry4/5/6, which all model CoWoS-S (a single
    monolithic silicon interposer spanning the whole package). Modern
    disaggregated GPU packages (NVIDIA Rubin/Rubin Ultra class, per public
    reporting as of 2026-08: 2 near-reticle compute dies + I/O dies on a
    multi-reticle CoWoS-L interposer with 8 HBM4 stacks) instead use LOCAL
    silicon interconnect (LSI) bridges and power/ground via-dense regions
    embedded in a much lower-conductivity organic (ABF/BT) substrate, placed
    only where die-to-die routing or power delivery density actually requires
    it. That is a genuine lateral-conductivity structure no other geometry in
    this benchmark has: a passive spreading layer that is MOSTLY low-k organic
    film with a higher-k composite only under specific regions, rather than
    either a uniform material or an active die layer's silicon-vs-underfill
    pattern.

    **Revision 2026-08-18**: the first version covered only narrow seams
    (compute-compute reticle stitch, HBM near-edges) with pure silicon
    (k=148), leaving each compute die's own footprint almost entirely over
    bare organic substrate. Under a power-concentrating scenario
    (`split_chiplet_a_hot`), that forced ~180 W/cm2 through 300um of k=0.5
    W/m*K material, producing a simulated 1125C peak -- confirmed as the real
    mechanism (not a solver bug) via a back-of-envelope series-resistance
    estimate matching the simulated rise within 1.5% (docs/compute.md). Fixed
    by giving each compute die's FULL footprint bridge/via coverage (real
    packages route dense copper power/ground vias through the organic
    substrate under high-current compute dies, not just narrow signal
    bridges) at a recalibrated, intermediate conductivity (k=60, the
    'lsi_bridge_via' material -- see its docstring in material.py for why not
    148) rather than either leaving the gap uncovered or overcorrecting to
    full silicon, which would erase the CoWoS-L-vs-CoWoS-S distinction this
    geometry exists to test. I/O dies and most of the inter-HBM field remain
    bare organic substrate -- they were never the source of the extreme
    values (I/O dies are low-power; HBM power is capped by the generator's
    existing HBM density ceiling), so widening their coverage too would dilute
    the comparison without fixing anything.

    Footprint: 62mm x 14mm (868 mm^2 -- ~1.48x geometry6's 588 mm^2, reflecting
    a larger reticle-stitched package; NOT a literal match to any specific real
    package's exact dimensions, which are not public).

    Layout (y: 1-13mm for all blocks; x from left edge):
      chipA (compute)  : x=1mm,    9x12mm   -- 2x2 sub-block grid
      chipB (compute)  : x=10.5mm, 9x12mm   -- 2x2 sub-block grid (0.5mm seam to chipA)
      io1 (I/O die)    : x=20mm,   3x5.5mm  (upper)
      io2 (I/O die)    : x=20mm,   3x5.5mm  (lower)
      hbm1..hbm8        : x=24.5, 29, 33.5, 38, 42.5, 47, 51.5, 56mm; 4x12mm each,
                          4.5mm pitch (0.5mm gap between stacks)

    LSI bridge / power-via islands (substrate_organic layer, k=60 composite
    footprints in an organic-substrate k=0.5 field -- see 'lsi_bridge_via' in
    material.py):
      bridge_chipA     : x=1mm,   9x12mm  -- full compute-die coverage (power/ground vias)
      bridge_chipB     : x=10.5mm, 9x12mm -- full compute-die coverage (power/ground vias)
                          (0.5mm true reticle-stitch gap x=10-10.5mm stays bare organic)
      bridge_hbm{1..8} : x=hbm[n]-0.5mm, 1x12mm  -- each HBM's D2D bridge to the
                          compute cluster (spans the 0.5mm gap + 0.5mm under the
                          stack's near edge)

    Layer stack (bottom to top) -- same 11-layer count and thicknesses as
    geometry5/6, isolating the CoWoS-L change to exactly one layer:
      heat_sink         5000um Cu           -- full
      tim_sink           125um TIM grease   -- full
      spreader           1000um Cu           -- full
      tim_top              50um In solder    -- full
      substrate_organic   300um ABF/BT organic (k=0.5) + Si bridge islands (k=148)
                                              -- REPLACES geometry5/6's uniform-Si
                                                 'interposer' layer; the one
                                                 structural change this geometry
                                                 tests
      rdl_layer             5um Si low-k     -- full, active
      c4_bumps            100um k=15         -- full
      die_zone_1            50um Si low-k    -- active; chipA/B, io1/2, hbm d1
                                                 footprints, underfill elsewhere
      hybrid_bonding         5um k=60         -- full
      tsv_zone             100um             -- hbm tsv footprints, underfill elsewhere
      die_zone_2            50um Si low-k    -- active; hbm d2 footprints only

    Total height: 6785um (identical to geometry5/6). Mesh: 56x248x50 (die_length
    14000/56=250um, die_width 62000/248=250um -- same 250um cell pitch as geometry6).
```

### FunctionDef:get_geometry_by_name

```text
Get geometry by name.

    Args:
        name: One of 'geometry1', 'geometry2a', 'geometry3'..'geometry7'
              ('geometry7' is a CoWoS-L pilot, not part of the standard
              6-geometry benchmark dataset -- see build_geometry7 docstring)

    Returns:
        Geometry object

    Raises:
        ValueError: If geometry name is invalid
```

### module

```text
Geometry builders for benchmark geometries.

Provides functions to create the 4 standard geometries:
- Geometry 1: 2D cross-sectional stack
- Geometry 2a: 3D stack with 3% TSV density
- Geometry 2b: 3D stack with 5% TSV density
- Geometry 2c: 3D stack with 10% TSV density
```

## `src/core/material.py`

### ClassDef:Material

```text
Thermal material properties.

    Attributes:
        name: Material identifier
        k_thermal: Thermal conductivity in W/m·K
        volumetric_heat_capacity: ρCp in J/m³·K
        density: Density in kg/m³ (optional, for reference)
        description: Human-readable description
```

### ClassDef:MaterialLibrary

```text
Library of standard thermal materials for 3D-IC packaging.

    All properties at room temperature (300K) unless otherwise noted.
```

### FunctionDef:add

```text
Add or update a material in the library.

        Args:
            material: Material object to add
```

### FunctionDef:compute_tsv_effective_k

```text
Calculate effective thermal conductivity for TSV region.

        Uses arithmetic mean (parallel conductors, upper bound):
        k_eff = (1 - φ) * k_Si + φ * k_Cu

        Args:
            k_silicon: Silicon thermal conductivity (W/m·K)
            k_copper: Copper thermal conductivity (W/m·K)
            tsv_fraction: TSV area fraction (0.0-1.0)

        Returns:
            Effective thermal conductivity (W/m·K)
```

### FunctionDef:create_tsv_material

```text
Create a custom TSV-enhanced silicon material.

        Args:
            tsv_density: TSV area fraction (0.0-1.0)
            name: Custom material name (auto-generated if None)

        Returns:
            Material object with equivalent properties
```

### FunctionDef:get

```text
Retrieve material by name.

        Args:
            name: Material identifier

        Returns:
            Material object

        Raises:
            KeyError: If material not found
```

### FunctionDef:k_silicon_temp_dependent

```text
Temperature-dependent silicon thermal conductivity.

        Model: k(T) = k_300K × (300/T)^α
        where α = 1.3 for 300K < T < 1000K

        Reference: Glassbrenner & Slack (1964), Physical Review 134(4A)

        Args:
            T_kelvin: Temperature in Kelvin

        Returns:
            Thermal conductivity in W/m·K

        Examples:
            >>> k_300 = MaterialLibrary.k_silicon_temp_dependent(300)
            >>> print(f"{k_300:.1f}")  # 148.0 W/m·K
            >>> k_400 = MaterialLibrary.k_silicon_temp_dependent(400)
            >>> print(f"{k_400:.1f}")  # ~80 W/m·K (47% drop)
```

### module

```text
Material property library for thermal simulations.

Defines material thermal properties for common 3D-IC packaging materials:
- Silicon (die substrate)
- Copper (heat spreader, TSVs, heat sink)
- TIM (Thermal Interface Material)
- TSV-enhanced silicon (equivalent conductivity)
- Bonding materials
```

## `src/core/mesh.py`

### FunctionDef:_power_field_from_maps

```text
Sample per-cell power maps onto arbitrary coordinates.

    Maps are indexed (along-length, along-width) to match the 3D-ICE floorplan
    convention, while coords are (x=width, y=length) in the PINN convention --
    hence the axis swap when looking up a cell.
```

### FunctionDef:compute_cell_volumes

```text
Compute cell volumes for finite volume discretization.

    Args:
        X, Y, Z: Meshgrid arrays of shape (nx, ny, nz)

    Returns:
        Array of cell volumes in μm³ with shape (nx-1, ny-1, nz-1)
```

### FunctionDef:create_uniform_grid_summary

```text
Create a summary of the mesh for a geometry.

    Args:
        geometry: Geometry object

    Returns:
        Formatted string summary
```

### FunctionDef:generate_adaptive_z_coords

```text
Generate z-coordinates that align with layer boundaries.

    Distributes more points in thin layers (like TIM) and fewer in thick layers
    (like heat sink) while ensuring layer interfaces are captured.

    Args:
        geometry: Geometry object
        nz: Total number of z-points desired

    Returns:
        1D array of z-coordinates
```

### FunctionDef:generate_cartesian_mesh

```text
Generate a 3D Cartesian mesh for the geometry.

    Args:
        geometry: Geometry object
        uniform_z: If True, use uniform z-spacing. If False, adapt to layer boundaries.

    Returns:
        Tuple of (X, Y, Z) meshgrid arrays with shape (nx, ny, nz)
```

### FunctionDef:generate_coords_and_indices

```text
Generate coordinate array and corresponding layer indices.

    Args:
        geometry: Geometry object
        uniform_z: Whether to use uniform z-spacing

    Returns:
        Tuple of:
        - coords: (N, 3) array of (x, y, z) coordinates in μm
        - layer_indices: (N,) array of layer indices (0-based)
```

### FunctionDef:generate_distance_to_power_block_field

```text
Per-point distance to the nearest power block's footprint, normalised by
    the die diagonal, in [0, ~1].

    Purely geometric -- constant for every scenario of a given geometry, no
    simulation data required (unlike `generate_tsv_field`, which depends on a
    per-scenario map). This is the right-sized geometry-aware conditioning
    signal for a benchmark whose geometries are all structured Cartesian
    grids: not a full SDF/graph encoder (GINO/PI-GANO-style), just per-cell
    information about *where within the geometry-specific floorplan* a point
    sits, which the current `geom_extent_norm` global scalar cannot express.
    See goal.md Track B.

    Points inside a block (or on its edge) get 0. TSV-region blocks are
    excluded (they are passive conductors, not power sources -- see
    `_power_field_from_maps`'s docstring for the same exclusion).
```

### FunctionDef:generate_power_density_field

```text
Generate volumetric power density field for given coordinates.

    Args:
        coords: (N, 3) array of (x, y, z) coordinates in μm
        geometry: Geometry object
        power_scenario: Dictionary mapping block names to power densities (W/cm²)
        power_map_by_layer: Optional {layer_name: (n_l, n_w) array of WATTS per
            cell}. When given it REPLACES the block decomposition, because it is
            what the simulator actually used. Falling back to blocks here would
            train a model on a coarse approximation of the source that produced
            its own targets.

    Returns:
        (N,) array of volumetric power densities (W/m³)
```

### FunctionDef:generate_tsv_field

```text
Sample per-cell TSV density maps onto arbitrary coordinates.

    Mirrors `_power_field_from_maps`: same (length, width) map indexing and the
    same x/y axis swap, but the field is dimensionless TSV area fraction phi in
    [0, 1] rather than a power density. Points outside any TSV-bearing layer get
    phi=0. This is what makes the spatial TSV mechanism (`src/scenario/tsv_maps.py`)
    a real per-point model input instead of only affecting the 3D-ICE ground truth
    with no corresponding conditioning signal (see assumptions.md).
```

### FunctionDef:interpolate_field_to_coords

```text
Interpolate a field defined on meshgrid to arbitrary coordinates.

    Args:
        field: Field values on meshgrid, shape (nx, ny, nz)
        X, Y, Z: Meshgrid coordinate arrays
        target_coords: Target coordinates, shape (N, 3)

    Returns:
        Interpolated field values at target_coords, shape (N,)
```

### FunctionDef:meshgrid_to_coords

```text
Convert meshgrid arrays to coordinate array.

    Args:
        X, Y, Z: Meshgrid arrays of shape (nx, ny, nz)

    Returns:
        Coordinate array of shape (N, 3) where N = nx*ny*nz
```

### module

```text
Mesh generation utilities for thermal simulations.

Provides functions to generate structured meshes for thermal geometries:
- 3D Cartesian grids
- Coordinate arrays for exporting
- Layer index mapping
```

## `src/core/placement.py`

### FunctionDef:associate_blocks_to_dies

```text
Map each DiePrint name -> names of the PowerBlocks lying inside it.

    Containment, not naming: a block belongs to the footprint that contains its centre.
    Blocks in no footprint (and all blocks of non-2.5D geometries, which have no footprints)
    come back under the key ''.
```

### FunctionDef:lateral_chiplets

```text
Group DiePrints that occupy the SAME lateral rectangle on different layers.

    A stacked chiplet (geometry6's `hbm1_die1` / `hbm1_die2`, geometry5's TSV stack) is one
    physical object appearing once per layer it spans. Those footprints must translate
    together: moving them independently would shear a die stack apart, and treating them as
    separate objects makes them look mutually overlapping, which made rejection sampling fail
    outright on geometry6.

    Returns [((x, y, w, h), [dieprint names])], one entry per physical chiplet.
```

### FunctionDef:place_chiplets

```text
Return a copy of `geometry` with each named chiplet translated by its (dx, dy) in µm.

    A chiplet is its DiePrint plus the PowerBlocks contained in it; both move together so
    the material region and its heat source stay registered. Use the key '' to translate
    blocks that belong to no footprint (the only option for non-2.5D geometries).

    Raises ValueError if any moved element would leave the package footprint; the caller is
    expected to sample offsets that fit (see `random_placement`).
```

### FunctionDef:random_block_placement

```text
Place every power block INDEPENDENTLY anywhere in the footprint, not as a rigid group.

    Added 2026-09-11 after the first moving-source pilot failed to make the benchmark
    discriminative (docs/report.md §9.15). Translating a fixed arrangement rigidly gives the
    layout only 2 degrees of freedom per chiplet, and because the temperature field is
    laterally smooth, a linear model handed the displacement as a feature covers that with a
    first-order response — ridge's error relative to the signal barely moved even at a
    source-overlap IoU of 0.472, which matches IC-ThermBench's 0.449.

    IC-ThermBench's layouts come from a placement optimiser: genuinely reconfigurable
    arrangements, not translations of one arrangement. This mode reproduces that property by
    sampling each block's position independently (2 DOF per block, so 8 for geometry1's four
    blocks), which changes the *relative* geometry of the sources and not just their common
    offset.
```

### FunctionDef:random_placement

```text
Sample per-chiplet offsets that keep everything inside the package and non-overlapping.

    `max_shift_um` bounds the translation per axis. `margin_um` is the minimum clearance kept
    between chiplets (one mesh cell at geometry4's 250 µm pitch), so moved chiplets never
    touch — a benchmark where dies sometimes abut and sometimes do not would confound
    placement with contact.

    Rejection sampling; returns all-zero offsets if no valid configuration is found within `max_tries`,
    so a caller always gets a usable (if unmoved) geometry rather than an exception.
```

### module

```text
Per-scenario chiplet placement — the fix for the defect diagnosed in docs/report.md §9.14.

Measurement showed every geometry in this benchmark has exactly ONE heat-source support
pattern across all of its scenarios (mean pairwise IoU 1.000): the sources never move, only
their amplitudes change. With the medium and mesh also fixed, the solution map is

    T(x) = T_amb + sum_b A_b(x) * Q_b        with A_b fixed

which is exactly, and only, the hypothesis class of a linear model — and the regime classical
power blurring has solved to ~1 K since 2007 (Kemper et al., arXiv:0709.1850). A benchmark
confined to it cannot discriminate between surrogate architectures.

This module moves the sources. It translates a *chiplet* — a `DiePrint` together with the
`PowerBlock`s sitting on it — as a rigid unit, so that in 2.5D geometries the silicon island
moves with its heat, changing the lateral material distribution and therefore the thermal
operator itself. IC-ThermBench has that property; this benchmark lacked it.

CORRECTION (2026-09-11, docs/report.md §9.15): rigid translation alone turned out NOT to be
enough. Translating chiplets gives the layout only ~2 degrees of freedom each, and because
the temperature field is laterally smooth a linear model handed the displacement covers it
to first order -- ridge's error relative to the signal moved by <10% even at a support
overlap of IoU 0.472, which is MORE movement than IC-ThermBench's 0.449. What actually
breaks a linear fit is layout DIMENSIONALITY: `random_block_placement` below, which places
each block independently (8 DOF on geometry1), drops ridge's spatial R^2 from 0.970 to
0.770 and doubles its relative error. Support-overlap metrics order these datasets wrongly;
degrees of freedom order them correctly.

The distinction matters and is the point of the experiment in
`scripts/gen_moving_source_pilot.py`:

  * In a laterally HOMOGENEOUS geometry (e.g. geometry1, a uniform silicon die), moving a
    power block changes where heat enters but not the medium. The operator is unchanged, so
    T is still a fixed linear functional of the per-cell power map. A linear model given the
    power map should still do well.
  * In a 2.5D geometry (geometry4/5/6/7), a chiplet IS a material region — silicon in
    underfill. Moving it rearranges the medium, so the operator changes per scenario and no
    single fixed linear map suffices.

Blocks are associated to dies by geometric containment rather than by name, so this works
across every geometry in the project without a naming convention.
```

## `src/deeponet/__init__.py`

### module

```text
Physics-Informed DeepONet for multi-geometry 3D-IC thermal surrogate.

A single model trained across all 7 benchmark geometries.

Reference: Lu et al., "Learning Nonlinear Operators via DeepONet Based on
the Universal Approximation Theorem of Operators", Nature Machine Intelligence,
2021.  Physics-informed extension: Wang et al., "Improved Architectures and
Training Algorithms for Deep Operator Networks", 2022.

Modules:
    model       -- BranchNet, TrunkNet, PIDeepONet
    data_loader -- MultiGeomDataset (all geometries in one dataset)
    trainer     -- Training loop with optional PI loss
```

## `src/deeponet/cno_model.py`

### ClassDef:CNOBranchEncoder

```text
Cross-geometry CNO-FNO encoder: (B,5,nx,ny,nz) → (B, n_basis).

    The 5 input channels mirror CNOFNOHybrid (Q_norm, layer_id_norm,
    htc, t_amb, tsv broadcast) so the branch has the same physics signals.

    Args:
        ch:           channel width (recommend 64 for cross-geometry training)
        n_cno_layers: stride-2 downsampling stages (2 → 4× spatial reduction)
        n_fno_blocks: FiLM-FNO blocks at the fixed (8,8,8) latent
        n_basis:      branch output dimension (must match trunk)
```

### ClassDef:PICNODeepONet

```text
PI-CNO-DeepONet: full spatial CNO-FNO branch + coordinate trunk.

    T(x) = CNOBranch(Q_grid) · Trunk(x) + bias

    Drop-in replacement for PIDeepONet — same trainer, same PDE loss,
    same trunk. Only the branch changes: from a 540D MLP to a spatial
    encoder that sees the full 3D power field.

    Args:
        n_basis:       dot-product dimension (default 128)
        ch:            branch channel width (default 64)
        n_fno_blocks:  FiLM-FNO blocks in branch latent
        n_cno_layers:  CNN encoder downsampling stages
        trunk_hidden:  trunk MLP width
        trunk_layers:  trunk MLP depth
        fourier_sigma: trunk Fourier encoding bandwidth
```

### module

```text
PI-CNO-DeepONet: CNO-FNO spatial branch encoder + coordinate trunk.

Replaces the fixed 16×16 sensor MLP branch of PI-DeepONet with a full
spatial CNO-FNO encoder, eliminating information loss from point sampling.

Architecture
------------
  Branch  —  CNOBranchEncoder
    Lift (5→ch) → CNN encoder (stride-2 × 2) → AdaptiveAvgPool3d to
    fixed (LF, LF, LF) latent → FiLM-FNO blocks → global pool → Linear
    → n_basis coefficients.

    FiLM is conditioned on (htc_norm, t_amb_norm, tsv_frac, geom_desc),
    giving the branch full geometry + scenario awareness via two pathways:
      (a) spatial Q_grid input (power field structure)
      (b) FiLM scalar conditioning (geometry descriptor + BCs)

    AdaptiveAvgPool3d normalises variable grid shapes (g1: 100×100×40,
    g6: 56×168×50, etc.) to a fixed (8,8,8) latent before the spectral
    blocks, making ONE branch work across all 8 geometries.

  Trunk  —  TrunkNet (reused from model.py)
    Fourier encoding of (x_norm, y_norm, z_norm, layer_id_norm) → n_basis.
    Trunk-only autograd for PDE loss: ∂T/∂x_i = branch · ∂trunk/∂x_i.

  Output
    T(x) = branch(Q_grid) · trunk(x) + bias   shape: (B, N)

Advantages over PI-DeepONet
----------------------------
  • No sensor information loss: full 3D power field, not 16×16 samples
  • Cross-geometry: AdaptiveAvgPool3d handles any (nx, ny, nz) → one model
  • geometry4/5/6 support: lateral k variation is implicitly encoded in the
    Q_grid spatial structure + layer_id channel
  • PDE cost identical: branch coefficients are fixed per scenario, so only
    the trunk is differentiated during the physics loss
  • ~3-5× better branch expressiveness vs MLP on the same n_basis

Parameter count (ch=64, n_basis=128, n_fno_blocks=4)
------------------------------------------------------
  CNN encoder (~640k)  +  latent FNO (~1.4M)  +  FiLM (~0.2M)
  + proj head (~16k)   +  trunk (~0.5M)
  Total: ~2.8M  vs  ~0.5M for PIDeepONet (richer, justified by 275 train scenarios)
```

## `src/deeponet/data_loader.py`

### ClassDef:MultiGeomDataset

```text
Dataset covering multiple geometries for cross-geometry DeepONet training.

    Each item is a dict with:
        branch_input:  (BRANCH_DIM,) float32
        coords_norm:   (N, 4) float32  — trunk query coords + layer id
        T_norm:        (N,) float32    — normalised target temperature
        power_norm:    (N,) float32    — normalised power (for PDE RHS)
        geom_name:     str
        scenario_name: str
```

### FunctionDef:_build_q_grid_3d

```text
Reconstruct a (nx, ny, nz) normalised Q grid from flat NPZ arrays.

    Handles both formats:
      - Full mesh (mock simulator): N = nx*ny*nz — direct reshape.
      - Per-layer-slice (real 3D-ICE): N = n_layers*nx*ny — broadcast each
        layer's 2D Q map to the z-bins closest to that layer's z-midpoint.
```

### module

```text
Multi-geometry dataset for PI-DeepONet training.

Unlike FNODataset (one model per grid shape), this dataset supports ALL 7
geometries in a single pass.  Each item carries both the branch input (power
sensors + BCs + geometry descriptor) and flat coordinate arrays (for trunk
evaluation and PDE loss).

Memory layout
-------------
Each item stores ALL data-grid points as flat arrays:
    coords_norm:  (N, 4) float32  — (x, y, z, layer_id)  all in [0,1]
    T_norm:       (N,)   float32  — normalised target temperature
    power_norm:   (N,)   float32  — normalised volumetric power density
    branch_input: (BRANCH_DIM,)   — branch network input (pre-built at load)

For geometry1 N = 60,000; geometry5 N = 44,800 — all fit comfortably in RAM
with 140 scenarios × ~60k × 4 × 4 bytes ≈ 130 MB.

Data stays on CPU; the trainer samples random subsets of N for each training
step (collocation sampling for PDE loss, and random subsets for data loss to
fit in GPU memory per step).
```

## `src/deeponet/model.py`

### ClassDef:BranchNet

```text
MLP that encodes (power sensors + scenario BCs + geometry descriptor)
    into N_basis coefficient scalars.

    input_dim:  BRANCH_DIM = 534
    output_dim: N_basis
```

### ClassDef:FourierEncoding4D

```text
Random Fourier feature encoding for 4D trunk input (x, y, z, layer_id).

    B matrix: (4, N_freq) sampled from N(0, σ²) once and fixed.
    Output dim: 2 * N_freq * 4 = 128 for N_freq=16.
```

### ClassDef:PIDeepONet

```text
Physics-Informed Deep Operator Network.

    T(x) = branch(u) · trunk(x)  +  bias

    Args:
        n_basis:       Number of basis functions (depth of dot product)
        branch_hidden: Width of branch MLP
        trunk_hidden:  Width of trunk MLP
        branch_layers: Depth of branch MLP
        trunk_layers:  Depth of trunk MLP
        fourier_sigma: Bandwidth for trunk Fourier encoding
```

### ClassDef:TrunkNet

```text
MLP that maps (x_norm, y_norm, z_norm, layer_id_norm) to N_basis spatial
    basis functions.

    Fourier encoding of the 4 coordinates gives the trunk good spectral
    properties for representing smooth thermal fields.

    Inputs flow through the trunk with requires_grad=True so autograd can
    compute ∂trunk/∂x for the PDE physics loss.
```

### FunctionDef:encode_branch_from_item

```text
Unified branch encoding interface used by DeepONetTrainer.
        Returns (1, n_basis) branch coefficients for a single dataset item.
        Trainer calls this on any DeepONet variant — subclasses override.
```

### FunctionDef:forward

```text
Evaluate T at N query points for a BATCH of B scenarios.

        Returns (B, N) — temperature at all query points for all scenarios.
        Each scenario gets the same query locations but different branch coefficients.
```

### FunctionDef:geometry_descriptor

```text
Build a fixed-size (GEOM_DESC_DIM,) float32 tensor from a Geometry object.

    Padding to GEOM_LAYERS_MAX layers with zeros for shorter stacks.
    Normalisation keeps all features in ~[0, 1].
```

### module

```text
PI-DeepONet: Physics-Informed Deep Operator Network.

Architecture
------------
T(x) = Σ_i  branch_i(u)  ×  trunk_i(x)   +  bias

  Branch network  — encodes the input *function* (power distribution + BCs
                    + geometry descriptor) into N_basis coefficients.

  Trunk network   — evaluates N_basis spatial basis functions at any query
                    point (x_norm, y_norm, z_norm, layer_id_norm).

  Bias            — learnable scalar added after the dot product.

The trunk takes CONTINUOUS coordinates, so a single trained model can
evaluate temperatures at arbitrary (x,y,z) across all 5 uniform-stack
geometries (geometry1/2a/2b/2c/3) without retraining.  This is the key
architectural advantage over FNO (which requires separate models per grid).
geometry4/5 (2.5D chiplet assemblies) are excluded: their per-layer thermal
conductivity varies laterally by chiplet region — a signal absent from the
trunk's (x,y,z,layer_id) coordinate space. FNO per-geometry covers those.

Branch input layout (BRANCH_DIM = 534)
--------------------------------------
  [0:512]    Power sensor values: 2 active layers × 16×16 bilinear
             downsampling of the Q_norm grid, zero-padded to 512 if only
             1 active layer (geometry1/3/4).
  [512:515]  Scenario scalars: htc_norm, t_amb_norm, tsv_frac
  [515:534]  Geometry descriptor (19 features):
               - 8 × layer_thickness_norm  (µm / max_thickness)
               - 8 × layer_k_norm          (W/m·K / 400)
               - underfill_k_norm           (k / 400; 0 for non-2p5d)
               - die_width_norm             (µm / 30000)
               - die_length_norm            (µm / 30000)

Trunk input (4 features, Fourier-encoded)
-----------------------------------------
  (x_norm, y_norm, z_norm, layer_id_norm)  ∈ [0, 1]
  Fourier encoding: 2 × N_freq × 4 = 128 features (N_freq=16, sigma=10)

Physics loss
------------
  ∂T/∂x, ∂T/∂y, ∂T/∂z are computed via autograd through the TRUNK only
  (branch coefficients are fixed per scenario).  With N_basis=128 and a
  2-layer trunk MLP, differentiating the trunk is O(N_col × trunk_cost)
  — far cheaper than differentiating through a full PINN.

  PDE:  ∇·(k ∇T) + Q = 0   →  L_pde = mean[(∇·(k∇T) + Q)²]
```

## `src/deeponet/trainer.py`

### ClassDef:DeepONetTrainer

```text
Trains PIDeepONet across multiple geometries simultaneously.

    Args:
        model:        PIDeepONet instance
        geometries:   dict of {geom_name: Geometry} (for layer tensors)
        norm_stats:   global normalisation statistics
        train_data:   MultiGeomDataset
        val_data:     MultiGeomDataset
        output_dir:   checkpoint directory
        batch_size:   scenarios per step
        epochs:       training epochs
        lr:           learning rate
        pde_weight:   λ for physics loss (0 = pure data-driven)
        n_data:       data points per scenario per step (random subset of N)
        n_col:        collocation points per scenario per step (PI loss)
        device:       training device
```

### FunctionDef:_pde_residual_deeponet

```text
Compute PDE residual ∇·(k∇T) + Q = 0 at collocation points via
    autograd through the trunk network only.

    branch_coeffs: pre-computed (1, n_basis) — identical for both PIDeepONet
    and PICNODeepONet. Passing it in avoids computing the branch twice and
    keeps the interface model-agnostic.

    Returns scalar loss.
```

### module

```text
PI-DeepONet training loop.

Training strategy
-----------------
Each epoch iterates over all scenarios (batch_size=4 scenarios at a time).
For each scenario, the trunk is evaluated at a random subset of the N data
points for the data loss, and at a separate set of collocation points for the
PDE residual (when pde_weight > 0).

Physics loss via trunk autograd
--------------------------------
The key advantage over PI-FNO and standalone PINN:

  T(x) = branch(u) · trunk(x)
  ∂T/∂x_i = branch(u) · ∂trunk/∂x_i          (chain rule, branch has no x)

So the PDE only differentiates the TRUNK (a small MLP, ~200k params) through
N_col collocation points.  Branch is fixed per scenario — no need to
backpropagate through the sensor/geometry encoding.  This is ~5-10× cheaper
than the PINN PDE loss per step, and requires no create_graph for the spatial
derivatives.

The data loss is MSE(T_pred, T_true) at n_data randomly sampled grid points.
Using random subsets avoids loading all N=60k points to GPU per scenario;
n_data=4096 points per scenario per step captures sufficient signal while
keeping GPU memory manageable.
```

## `src/export/__init__.py`

### module

```text
Data export utilities for thermal simulations.

Provides functionality to:
- Export simulation results to NumPy .npz format (PINN-ready)
- Compute thermal statistics (min/max/mean, hotspots)
- Generate dataset summaries
```

## `src/export/npz_exporter.py`

### ClassDef:NPZExporter

```text
Export thermal simulation results to NumPy .npz format.

    Output Format:
    - coords: (N, 3) array of (x, y, z) coordinates in micrometers
    - temp: (N,) array of temperatures in Kelvin
    - power: (N,) array of volumetric power density in W/m³
    - layer: (N,) array of layer indices (0-based)
```

### FunctionDef:__init__

```text
Initialize exporter.

        Args:
            output_dir: Directory for output .npz files
```

### FunctionDef:batch_export

```text
Export multiple scenarios efficiently.

        Args:
            scenarios: List of ScenarioParameters objects
            geometries: Dict mapping geometry names to Geometry objects
            simulator_results: Dict mapping scenario names to (coords, temps) tuples
            verbose: Print progress messages

        Returns:
            List of output file paths
```

### FunctionDef:create_dataset_summary

```text
Create summary of all .npz files in a directory.

        Args:
            output_dir: Directory containing .npz files

        Returns:
            Dictionary with dataset statistics
```

### FunctionDef:export_scenario

```text
Export a single simulation scenario to .npz format.

        Args:
            scenario_name: Scenario identifier (e.g., 'geometry1_train_001')
            geometry: Geometry object defining the physical structure
            scenario_params: Scenario parameters dict with:
                - power_blocks: {block_name: power_density_W/cm2}
                - htc: Heat transfer coefficient (W/m²·K)
                - t_ambient: Ambient temperature (°C)
            temperature_field: (N,) array of temperatures in K from simulator
            coords: Optional (N, 3) coordinate array. If None, generated from geometry.

        Returns:
            Path to saved .npz file
```

### FunctionDef:get_file_info

```text
Get summary information about a .npz file without loading all data.

        Args:
            npz_file: Path to .npz file

        Returns:
            Dictionary with file info
```

### FunctionDef:load_scenario

```text
Load a scenario from .npz file.

        Args:
            npz_file: Path to .npz file

        Returns:
            Dictionary with keys:
                - coords: (N, 3) array
                - temp: (N,) array
                - power: (N,) array
                - layer: (N,) array
                - tsv_frac: (N,) array (zeros for files exported before this field existed)
                - metadata: Dict with scenario information
```

### module

```text
NumPy data exporter for thermal simulation results.

Converts 3D-ICE/HotSpot simulation output to PINN-ready NumPy format.
Exports (coords, temp, power, layer) arrays as compressed .npz files.
```

## `src/export/statistics.py`

### ClassDef:StatisticsCalculator

```text
Compute thermal statistics from simulation results.

    Calculates:
    - Min/max/mean/median temperature
    - Temperature standard deviation
    - Hotspot location and peak temperature
    - Temperature distribution percentiles
    - Per-layer statistics
```

### FunctionDef:_compute_distribution

```text
Compute temperature distribution histogram.

        Args:
            temperatures: (N,) temperature array

        Returns:
            Dictionary with bin counts
```

### FunctionDef:_compute_gradients

```text
Compute temperature gradients in x, y, z directions.

        Args:
            coords: (N, 3) coordinate array
            temperatures: (N,) temperature array
            geometry: Geometry object

        Returns:
            Dictionary with gradient statistics
```

### FunctionDef:batch_compute

```text
Compute statistics for multiple scenarios.

        Args:
            scenarios: List of ScenarioParameters objects
            geometries: Dict mapping geometry names to Geometry objects
            simulation_data: Dict mapping scenario names to (coords, temps, power, layer) tuples
            output_dir: Optional directory to save JSON files
            verbose: Print progress messages

        Returns:
            Dictionary mapping scenario names to statistics
```

### FunctionDef:compute_scenario_stats

```text
Compute comprehensive statistics for a single scenario.

        Args:
            scenario_name: Scenario identifier
            coords: (N, 3) array of (x, y, z) coordinates in μm
            temperatures: (N,) array of temperatures in K
            power_density: (N,) array of volumetric power density in W/m³
            layer_indices: (N,) array of layer indices
            geometry: Geometry object

        Returns:
            Dictionary with comprehensive statistics
```

### FunctionDef:create_summary_stats

```text
Create aggregate statistics across multiple scenarios.

        Args:
            all_stats: Dictionary of per-scenario statistics
            by_geometry: Whether to group by geometry

        Returns:
            Dictionary with aggregate statistics
```

### FunctionDef:load_stats

```text
Load statistics from JSON file.

        Args:
            input_file: Path to JSON file

        Returns:
            Statistics dictionary
```

### FunctionDef:save_stats

```text
Save statistics to JSON file.

        Args:
            output_file: Path to output JSON file
            stats: Statistics dictionary
            pretty: Whether to pretty-print JSON
```

### module

```text
Thermal statistics calculator for benchmark scenarios.

Computes temperature statistics and hotspot locations from simulation data.
Exports results as JSON for validation and analysis.
```

## `src/fno/__init__.py`

### module

```text
Fourier Neural Operator (FNO) for 3D-IC thermal surrogate modelling.

Maps power density field Q(x,y,z) → temperature field T(x,y,z) for all four
benchmark geometries in a single model. Unlike the geometry-specific PINNs,
the FNO learns the thermal operator directly and generalises across TSV densities.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial Differential
Equations", ICLR 2021. https://arxiv.org/abs/2010.08895

Modules:
    model       -- SpectralConv3d, FNOBlock, FNO3d
    data_loader -- FNODataset (flat .npz → 3D grid tensors)
    trainer     -- Training loop
```

## `src/fno/data_loader.py`

### ClassDef:FNODataset

```text
Dataset of 3D field tensors for FNO training.

    Each item:
        Q_norm:         (nx, ny, nz) float32 — normalised power density
        layer_id_norm:  (nx, ny, nz) float32 — layer_index / (n_layers-1)
        T_norm:         (nx, ny, nz) float32 — normalised temperature (target)
        htc_norm:       scalar float
        t_amb_norm:     scalar float
        tsv_frac:       scalar float
        name:           str scenario identifier

    Data is kept on CPU; move to device in training loop to avoid pinning issues
    on Windows where pin_memory has known reliability problems with large tensors.
```

### FunctionDef:__init__

```text
Args:
            expected_grid: Native grid used when `target_grid` is None. All files
                must match it, and non-matching files are skipped.
            target_grid: If given, each file is built at its OWN native resolution
                (read from its metadata) and resampled onto this shared grid,
                enabling one FNO across geometries with different mesh shapes.
            geometries: Optional {geometry_name: Geometry}. When given, each item
                also carries `dist_to_block_norm` -- per-cell distance to the
                nearest power block, normalised by die diagonal (Track B
                geometry-aware conditioning, see src/core/mesh.py). Purely
                geometric (no simulation data needed), zero-filled if omitted,
                for backward compatibility with callers that don't pass it.
```

### FunctionDef:_resample

```text
Resample a (nx, ny, nz) grid onto `target`.

    `mode` is 'trilinear' for continuous fields (power, temperature) or 'nearest'
    for categorical ones (layer identity).
```

### FunctionDef:predict_to_flat

```text
Run inference on a single item, return flat (N,) arrays in Kelvin.

    Useful for evaluation code that works with flat coordinate arrays.
```

### module

```text
FNO data loader: reshapes flat .npz arrays into 3D grid tensors.

The .npz files store coords/temp/power as flat (N,) or (N,3) arrays produced
by the mesh generator with indexing='ij' and ravel() in C order. Reshaping to
(nx, ny, nz) is therefore safe as long as the mesh resolution matches what's
stored in the file metadata — which it always does if the file was produced by
the standard export pipeline.

Each dataset item is a dict so the trainer can access fields by name without
positional index bugs. No custom collation required; torch's default collate
handles dicts of same-shape tensors correctly.

Multi-geometry training
-----------------------
The eight benchmark geometries produce five incompatible grid shapes:

    (100, 100,  6)  geometry1, geometry3
    ( 80,  80, 10)  geometry2a/b/c
    (100,  56,  6)  geometry4
    (100,  56, 11)  geometry5
    ( 56, 168, 11)  geometry6

FNO's spectral convolution takes an FFT over the spatial dims, so a single model
cannot span these natively. Two options: (a) one dataset and training loop per
grid shape, or (b) resample everything onto a common grid. BOTH are supported.

Pass `target_grid` to enable (b). Every file is then built at its own native
resolution and trilinearly resampled onto the shared grid, so one FNO can train
across all eight geometries.

Resampling alone would be lossy in a way that matters: an 8x8 mm die and a
42x14 mm die resampled to the same array are indistinguishable, and the operator
would be asked to learn contradictory mappings. Each item therefore also carries
`geom_extent_norm` — the physical (width, length, height) of the package — so the
model conditions on real spatial scale rather than array indices. That is what
keeps each geometry's specific physics learnable after resampling.
```

## `src/fno/hybrid.py`

### ClassDef:CorrectionPINN

```text
Point-wise residual network that predicts δT = T_true - T_FNO.

    Identical architecture to FourierPINN except:
    - One extra scalar input: T_FNO_norm (the FNO prediction at this point)
    - Output is unconstrained (correction can be positive or negative)
    - Shallower by default (4 residual blocks instead of 6) because the
      correction field is smoother than the full temperature field

    Input dim = 32 (Fourier) + 8 (layer emb) + 4 (scenario scalars) + 1 (T_FNO) = 45
```

### ClassDef:HybridModel

```text
CNOFNOHybrid (frozen, preferred) or FNO3d (frozen) + CorrectionPINN (trained).

    The FNO provides the coarse grid prediction; the PINN refines it at
    arbitrary query points. Freezing the FNO keeps memory low and avoids
    destabilising a converged model. Using CNOFNOHybrid as the base gives a
    better coarse solution so the correction residual is smaller (< 2 K vs
    up to 10 K with FNO3d), making the PINN's job easier.

    Args:
        fno:            Pre-trained CNOFNOHybrid (recommended) or FNO3d.
                        Weights are frozen at construction.
        n_layers:       Number of geometry layers (for layer embedding size).
        fourier_sigma:  Fourier feature scale for CorrectionPINN.
        hidden_dim:     CorrectionPINN MLP width.
        n_res_blocks:   CorrectionPINN depth.
```

### FunctionDef:compute_hybrid_loss

```text
Compute hybrid loss components.

    Returns a dict with scalar tensors:
        total, correction_data, pde (if w_pde>0), bc (if w_bc>0)

    L_correction_data: MSE(T_total - T_true) at query points (primary signal)
    L_pde:             PDE residual on T_total at collocation points
    L_bc:              Convective BC residual on T_total at top-surface points

    The grid-level FNO loss is NOT recomputed here — it was already minimised
    during FNO pre-training. The correction loss drives the PINN to close the
    remaining gap.
```

### FunctionDef:forward_points

```text
Return (T_total_norm, T_fno_interp_norm), both shape (N,).

        T_total = T_FNO (interpolated) + δT (CorrectionPINN)
```

### FunctionDef:interpolate_fno

```text
Trilinear interpolation of the FNO grid at arbitrary normalised coords.

        grid_sample expects input in [-1,1] with (D,H,W) = (z,y,x) convention.
        We remap: normalised [0,1] -> grid_sample [-1,1].
```

### module

```text
Hybrid FNO + PINN correction model.

Motivation
----------
The FNO predicts T on the Cartesian grid (fast, global, data-driven).
The CorrectionPINN predicts a residual δT at *arbitrary* query points so the
combined prediction satisfies the heat equation:

    T_total(x) = T_FNO(x) + δT(x)

where T_FNO(x) is the FNO output trilinearly interpolated from the grid to
point x, and δT is a small correction learned by a point-wise PINN.

Why this is useful
------------------
- The FNO handles coarse global structure with O(N log N) cost; the PINN
  handles fine-scale physics residuals and boundary layers.
- The correction is typically small (< 5 K) so the PINN trains much faster
  than a standalone PINN that must learn the full temperature field.
- The PDE loss is applied to T_total, not δT alone, so the physics constraint
  acts on the meaningful quantity.

Training protocol
-----------------
1. Pre-train FNO on grid data (FNOTrainer) until convergence.
2. Freeze FNO weights.
3. Train CorrectionPINN with:
     L = L_correction_data + λ_pde * L_pde(T_total) + λ_bc * L_bc(T_total)
   using the same curriculum staging as the standalone PINN.

The FNO is frozen during step 3 — its activations are not backpropagated
through. This keeps the hybrid training memory budget close to PINN-only.

Coordinate convention
---------------------
FNO operates on the integer grid [0..nx-1] x [0..ny-1] x [0..nz-1] mapped to
physical µm coordinates. The CorrectionPINN receives normalised coords [0,1]³,
as does the standalone FourierPINN. Interpolation is bilinear in the normalised
grid (grid_norm indices = coord_norm * (grid_size - 1)).

Usage example
-------------
    fno = CNOFNOHybrid(grid_shape=(100,100,40), ...)  # preferred base
    # load pretrained weights
    model = HybridModel(fno, n_layers=5, fourier_sigma=10.0)
    loss = compute_hybrid_loss(model, batch, weights, norm_stats, ...)
```

## `src/fno/interpret.py`

### FunctionDef:_get_spectral_blocks

```text
Locate the sequence of spectral-conv-containing blocks on a model.

    Supports FNO3d/WHNO3d (`.blocks`, each with `.spectral`) and
    CNOFNOHybrid (`.latent_blocks`, each with `.spectral` OR an axial-
    attention wrapper around one -- see AxialAttentionFiLMBlock).
```

### FunctionDef:ablate_and_measure

```text
Causal importance: zero out successive low-sequency/low-frequency BANDS
    of one block's spectral weight (band 0 = lowest modes ... band n-1 =
    highest retained modes), re-measure `forward_fn()` after each ablation,
    restore the original weight before returning.

    Returns a list of `val_loss_after_ablating_band_i` for i in [0, n_bands).
    Compare against the baseline (unablated) forward_fn() value to see which
    bands are load-bearing vs redundant -- a causal complement to the
    magnitude-based importance above (magnitude tells you where the model
    PUT weight; ablation tells you where it actually MATTERS for accuracy).

    Cost: n_bands forward passes over the val set. Cheap for FNO/WHNO
    (inference is fast — seconds, not hours) but O(n_bands) wall time.
```

### FunctionDef:block_mode_importance

```text
Per-mode importance for a single spectral block's weight tensor.

    weight: (in_ch, out_ch, mx, my, mz), real or complex.
    Returns (mx, my, mz) real-valued Frobenius-norm-over-channels importance
    map. Works identically for FNO's complex weights (via .abs()) and WHNO's
    real weights.
```

### FunctionDef:compare_z_decay

```text
Return (decay_a, decay_b) z-axis marginal importance profiles for two
    models (e.g. FNO3d vs WHNO3d trained on the SAME geometry) so their
    energy-vs-mode-index decay curves can be compared directly.

    Only meaningful on trained checkpoints — on randomly-initialised models
    both profiles will be roughly flat/noisy since there's no learned signal
    yet.
```

### FunctionDef:marginal_importance

```text
Collapse a (mx,my,mz) importance map to a 1D profile along one axis by
    summing over the other two — "how much total energy sits at each mode
    index along this axis, regardless of the other two axes."

    axis: 0=x, 1=y, 2=z (z is the axis carrying material-interface structure
    in these layered-stack geometries — most informative for the interface
    hypothesis).
```

### module

```text
Spectral mode-importance XAI for FNO / WHNO / CNO-FNO.

Unlike the PINN's post-hoc XAI (gradient-based, requires forward passes),
an operator with a spectral basis has interpretability NATIVE to its
architecture: the learned `weight` tensor of each spectral block IS a
per-mode importance map, readable directly off the trained parameters with
zero forward passes.

Both FNO's SpectralConv3d and WHNO's WalshConv3d store their truncated
`weight` as (in_ch, out_ch, mx, my, mz), already ordered low-index=smooth
to high-index=oscillatory (FFT's rfft ordering for Fourier; sequency
ordering — see whno.hadamard_to_sequency_perm — for Walsh, already applied
before truncation). So mode index 0 along any axis is the smoothest
component retained, with no extra basis-specific conversion needed to
compare "does this model emphasise fine detail or bulk smoothness" between
the two bases.

Use case: cross-reference this against known material-interface locations
(see src/fno/physics.py's interface z-index detection) to test the
hypothesis that WHNO concentrates more importance in finer z-bands (because
interfaces are piecewise-constant discontinuities the Walsh basis
represents natively) while FNO's importance decays faster and is truncated
before it can represent the interface (the Gibbs-ringing argument, already
validated on a synthetic step function in whno.py's docstring — this module
lets you check whether that pattern shows up in ACTUAL TRAINED WEIGHTS, not
just the toy 1D demo).
```

## `src/fno/model.py`

### ClassDef:AxialAttentionFiLMBlock

```text
FiLMFNOBlock + axial self-attention (x → y → z) in the latent space.

    Full 3D self-attention on 4k–7k tokens costs ~830 MB/sample for the
    attention matrix alone (e.g. g2/g6 at 7200 tokens × 4 heads).
    Axial factorisation attends along each spatial axis independently:
      x-axis: ly*lz parallel sequences of length lx  (max 25 tokens)
      y-axis: lx*lz parallel sequences of length ly  (max 42 tokens for g6)
      z-axis: lx*ly parallel sequences of length lz  (max 18 tokens)
    Peak attention matrix: 42² × 16 bytes × 4 heads = 112 KB/sample regardless
    of geometry. Captures full 3D correlations via the x→y→z composition.

    On PyTorch 2.x, nn.MultiheadAttention dispatches to
    F.scaled_dot_product_attention (FlashAttention-2) on CUDA automatically
    when need_weights=False — no extra package required.

    Args:
        channels: hidden channel width (must be divisible by n_heads)
        modes:    spectral modes (passed through to SpectralConv3d)
        n_heads:  attention heads; default 4 → head_dim = channels/4
```

### ClassDef:CNOFNOHybrid

```text
CNO encoder + FiLM-FNO latent + CNO decoder — the combined best-of-three model.

    Forward signature matches FNO3d / CondFNO3d: same inputs, same output shape.
    Works as a drop-in replacement in FNOTrainer with PI loss enabled.

    Args:
        grid_shape:   (nx, ny, nz) full-resolution geometry mesh
        ch:           channel width throughout (encoder, FNO latent, decoder)
        n_fno_blocks: FNO blocks applied in the latent space
        n_cno_layers: stride-2 downsampling stages (2 → 4× spatial reduction)
```

### ClassDef:CondFNO3d

```text
Physics-parameter-conditioned FNO3d.

    Improvements implemented:
      #2 — BC encoding: HTC/T_amb drive spectral filter modulation (not just
           broadcast channels), giving the operator a physically-structured
           boundary-condition response.
      #4 — Physical parameter conditioning: FiLMGenerator maps the full
           (htc_norm, t_amb_norm, tsv_frac) tuple to per-block (γ, β) pairs.

    The baseline FNO still receives the three scalars as input channels (5-ch
    input unchanged), so the FiLM modulation is *additional* — it fine-tunes
    the spectral response rather than replacing the information path.

    n_parameters ≈ base_FNO + 64*64 + 64*64 + 64*(2*n_blocks*hidden_ch)
                ≈ base + ~400k for default (4 blocks, 32 ch) — small overhead.
```

### ClassDef:FNO3d

```text
Fourier Neural Operator for 3D thermal field prediction.

    Takes the power density + layer/scenario features on the geometry grid,
    outputs the normalised temperature field on the same grid.

    Args:
        grid_shape:   (nx, ny, nz) of the target geometry mesh
        modes:        (mx, my, mz) spectral modes to retain — clamped to
                      floor(grid_dim/2) automatically; defaults suit both
                      geometry1 (100×100×40) and geometry2 (80×80×50)
        hidden_ch:    width of the hidden representation (32 for fast runs,
                      64 for publication accuracy)
        n_blocks:     number of FNO layers; 4 is the standard from Li et al.
```

### ClassDef:FNOBlock

```text
Single FNO layer: spectral path + local (pointwise) path + residual.

    spectral(x) captures long-range correlations via truncated FFT.
    local(x) = Conv3d(kernel=1) handles high-frequency and local features
               that the spectral path drops when modes are truncated.

    The two paths are summed (not concatenated) before activation.
    Activation: GELU — smooth, performs better than ReLU for operator learning.
```

### ClassDef:FiLMGenerator

```text
Small MLP that maps physics parameters → (γ, β) scaling vectors for each
    FNO block.  FiLM (Feature-wise Linear Modulation) conditions the spectral
    filters on HTC, T_amb and TSV_frac rather than broadcasting them as plain
    input channels.

    Why this is better than input-channel broadcasting:
    - The spectral filters explicitly adapt their frequency response to the
      cooling and material conditions — a physically meaningful inductive bias.
    - Extrapolation to unseen (HTC, T_amb) pairs is constrained to the
      parametric manifold learned by the generator, not blind spectral blending.

    param_dim: 3  (htc_norm, t_amb_norm, tsv_frac)
    output:    2 * n_blocks * hidden_ch  (γ and β per block per channel)
```

### ClassDef:SpectralConv3d

```text
3D spectral convolution via truncated FFT.

    Learns complex weights R of shape (in_ch, out_ch, mx, my, mz) in frequency
    space. The forward pass FFTs the input, multiplies the low-mode block by R,
    then IFFTs back. High-frequency modes are zeroed (not learned).

    Weight init: uniform on the unit circle, scaled by 1/sqrt(in_ch * out_ch).
    This keeps activations stable at init regardless of mode count.
```

### FunctionDef:_as_field

```text
Expand a conditioning input to (B, 1, nx, ny, nz).

    Accepts either a real per-cell field already shaped (B, nx, ny, nz) -- as
    tsv_frac now is, see src/fno/data_loader.py -- or a scalar/per-batch scalar,
    which is broadcast uniformly as htc_norm/t_amb_norm still are. Old files/
    call sites that only ever had a scalar TSV density keep working unchanged.
```

### FunctionDef:_as_scalar

```text
Reduce a conditioning input to (B,) for use in a FiLM-style scalar summary.

    A real per-cell field (B, nx, ny, nz) is mean-pooled to its per-scenario
    average; a scalar passes through unchanged. Used where a spatial signal
    (tsv_frac) needs to additionally feed a per-scenario modulation vector, not
    just the stacked input channels.
```

### FunctionDef:build_cno_fno

```text
Build the combined CNO + FiLM-FNO best model.

    Args:
        use_attention: Add axial self-attention (SAU-FNO style) to each latent
                       FiLM-FNO block. Uses x→y→z factorised attention so the
                       attention matrix stays under 112 KB/sample regardless of
                       geometry. Dispatches to FlashAttention on PyTorch 2.x CUDA.
        n_heads:       Attention heads. Must divide ch evenly (default 4 → head_dim=ch/4).
```

### module

```text
FNO3d: Fourier Neural Operator for 3D steady-state heat conduction.

The operator learns  F: (Q, params) -> T  where:
  Q      -- volumetric power density field, shape (nx, ny, nz)
  params -- scenario scalars: htc_norm, t_amb_norm, tsv_frac
  T      -- temperature field, shape (nx, ny, nz)

All fields are on the geometry's Cartesian mesh. Coordinates are NOT passed as
input — the FNO encodes spatial structure through the spectral basis implicitly.
If you want coordinate-conditioned output, add them as extra input channels;
for this thermal problem they hurt more than they help because the mesh is
fixed per geometry and the spectral modes already encode position.

Input channels per voxel (IN_CH = 5):
  0: Q_norm          normalised power density
  1: layer_id_norm   layer index / (n_layers-1), encodes material discontinuities
  2: htc_norm        broadcast scalar
  3: t_amb_norm      broadcast scalar
  4: tsv_frac        real per-cell TSV area fraction field where the geometry
                     carries TSVs (zero elsewhere), not a broadcast scalar --
                     see src/scenario/tsv_maps.py and _as_field() below. Falls
                     back to broadcasting when given a genuine scalar, for
                     compatibility with data predating this field.

Hidden channels: 32 (sufficient for smooth thermal fields; increase to 64 for
publication accuracy runs).

Mode counts: clamped at init to floor(grid_dim/2) per Nyquist. The defaults
(16, 16, 12) cover the dominant thermal modes for both geometry1 (100×100×40)
and geometry2 (80×80×50) without aliasing.
```

## `src/fno/physics.py`

### FunctionDef:_harmonic_mean

```text
Two-point harmonic mean face conductivity: 2*ka*kb / (ka+kb).

    This is the standard finite-volume face-conductivity choice for
    heterogeneous diffusion (Patankar, "Numerical Heat Transfer and Fluid
    Flow", 1980) — it exactly reproduces flux continuity for a piecewise-
    constant conductivity field with a two-point flux approximation.
    Arithmetic mean (ka+kb)/2, used previously, has O(1) relative error
    when ka/kb is large — exactly the regime at every material interface in
    this dataset (Si k=148 vs TIM k=4 W/m·K, a 37x ratio).
```

### FunctionDef:build_k_grid

```text
Map layer-id grid to conductivity grid (B, nx, ny, nz) in W/m·K.

    layer_id_norm ∈ [0,1] → layer index = round(layer_id_norm * (n_layers-1))
```

### FunctionDef:fd_divergence

```text
Compute ∇·(k ∇T) via central FV differences on a regular grid.

    Returns (B, nx, ny, nz) [W/m³].  Interior cells only — boundary rows
    use one-sided differences (forward/backward) so the output has the same
    shape as the input without cropping.

    Face conductivity uses the HARMONIC mean of adjacent cells (see
    _harmonic_mean) — required for flux accuracy across the large
    conductivity contrasts at material interfaces in this dataset.
```

### FunctionDef:grid_spacings

```text
Return (dx, dy, dz) in metres for a geometry's Cartesian mesh.

    dx = die_width  / nx
    dy = die_length / ny
    dz = total_height / nz   (mean cell — valid approximation for our
                               adaptive-z meshes where variation is < 2×)
```

### FunctionDef:interface_flux_loss

```text
Flux-continuity residual at each material interface, using one-sided
    finite differences computed INDEPENDENTLY on each side of the boundary
    (not the blended harmonic-mean face conductivity fd_divergence uses for
    the bulk PDE residual). This directly checks the interface condition
    k_lower * dT/dz|- == k_upper * dT/dz|+ using two separately-estimated
    one-sided derivatives, rather than a single symmetric stencil straddling
    the boundary — a more direct, independently-reportable diagnostic of
    interface behaviour, and lets flux-continuity be weighted separately
    from (and typically much higher than) the diluted whole-volume PDE loss.

    Interface z-indices are located PER-SAMPLE from layer_id_norm at runtime
    (not hardcoded), by scanning where the layer index changes along z.
    layer_id_norm is assumed constant across (x,y) within a z-slice, which
    holds for all layered-stack geometries in this dataset (2d_stack,
    3d_stack, 2p5d_stack all assign layer index purely by z-range; lateral
    heterogeneity in 2p5d_stack geometries is a separate k-override
    mechanism, not a layer-index change).

    An interface is skipped (not scored) if either adjacent layer has fewer
    than 2 z-cells in the mesh (too thin to form a one-sided difference) —
    this can happen for very thin layers (e.g. a 5µm hybrid-bonding layer)
    at coarse z-resolution.

    Returns a scalar (mean squared flux mismatch, W²/m⁴, batch- and
    interface-averaged). NOT detached — gradients flow through to T_norm's
    source (the model), unlike pde_loss_fd's current call site.
```

### FunctionDef:pde_loss_fd

```text
Finite-difference PDE residual loss for PI-FNO.

    L_pde = mean_over_batch_and_grid( (∇·(k∇T) + Q)² )

    All quantities are converted to physical units before differencing so the
    loss has units of (W/m³)² — consistent across geometries.

    Gradient is NOT tracked through this loss (no create_graph) — it serves
    purely as a regulariser. AMP autocast is compatible: operates in float32
    because F.pad and float arithmetic stay in fp32 under autocast.
```

### module

```text
PI-FNO: physics residual via finite differences on the FNO output grid.

Unlike the PINN which uses autograd second derivatives through a point-wise MLP,
the FNO already produces a full 3D grid at each forward pass.  We can therefore
evaluate the heat equation cheaply with central finite differences on that grid
— no computation graph retention, no third-order autograd, just tensor ops.

PDE (steady-state heat):   ∇·(k ∇T) + Q = 0

Finite-difference approximation on a regular Cartesian mesh (dx, dy uniform;
dz uniform approximation using mean layer thickness):

    ∂/∂x (k ∂T/∂x) ≈  [k_{i+½}(T_{i+1}-T_i) - k_{i-½}(T_i-T_{i-1})] / dx²

    where  k_{i+½} = (k_i + k_{i+1}) / 2  (arithmetic mean at face)

This conservative finite-volume stencil preserves energy balance at material
interfaces (k discontinuities between layers).

Coordinate convention (physical, metres):
    dx = die_width_m  / nx
    dy = die_length_m / ny
    dz = total_height_m / nz   (mean cell; valid approximation for regularised
                                  meshes where adaptive z-spacing is < 2× uniform)
```

## `src/fno/trainer.py`

### ClassDef:FNOTrainer

```text
Trains FNO3d (or CondFNO3d) on a fixed-geometry dataset.

    PI-FNO support: set pde_weight > 0 to add the finite-difference PDE
    residual loss.  Requires `geometry` to be passed so cell spacings and
    layer conductivities can be computed.  The PI loss is computed in FP32
    regardless of AMP setting (F.pad and float arithmetic stay in FP32 under
    autocast, so this is free).

    Args:
        model:        FNO3d or CondFNO3d instance
        norm_stats:   global normalisation constants (shared with PINN)
        train_data:   FNODataset with training scenarios
        val_data:     FNODataset with test scenarios
        output_dir:   directory for checkpoints and logs
        batch_size:   scenarios per training step (4 is the standard)
        epochs:       training epochs
        lr:           initial learning rate
        weight_decay: AdamW weight decay (regularises spectral weights)
        device:       training device
        pde_weight:   λ for whole-volume finite-difference PDE loss (0 = pure
                      data-driven baseline FNO)
        flux_weight:  λ for interface-isolated flux-continuity loss (0 =
                      disabled). Independent of pde_weight — can be used
                      alone or combined. See interface_flux_loss() docstring.
        geometry:     geometry object (required when pde_weight > 0 or
                      flux_weight > 0)
```

### FunctionDef:relative_l2_loss

```text
Relative L2 loss per sample, averaged over batch.

    T_pred, T_true: (B, nx, ny, nz)
```

### module

```text
FNO training loop.

Unlike the PINN, this is pure data-driven training — no PDE collocation, no
autograd second derivatives, no curriculum staging. The FNO is a regression
model: minimise MSE(T_pred, T_3DICE) over the full 3D field.

The relative L2 loss (rL2) normalises by ||T_true||₂ per sample so that
low-power scenarios (small absolute temperatures) contribute equally to
high-power scenarios. This is standard for neural operator training and
prevents the model from ignoring cold scenarios.

Loss: L = mean_over_batch(||T_pred - T_true||₂ / ||T_true||₂)

Optimiser: AdamW with weight decay 1e-4 (FNO complex weights need regularisation
to prevent spectral overfitting). Cosine LR schedule, no warm restarts needed
since the loss landscape is smooth compared to the PINN.

Mixed precision: enabled for CUDA. The complex spectral weights stay in float32;
only the real-valued activations use float16 (torch handles this automatically
via autocast).

Memory note: one 3D field at float32 for grid (100,100,40) = 1.6MB. Batch of 4
= 6.4MB. Negligible compared to the 200MB activation memory for the FNO layers
themselves. Batch size of 4 fits comfortably on 8GB GPU.
```

## `src/fno/whno.py`

### ClassDef:WHNO3d

```text
Walsh-Hadamard Neural Operator — drop-in alternative to FNO3d.

    Same input/output contract as FNO3d.forward: (Q_norm, layer_id_norm,
    htc_norm, t_amb_norm, tsv_frac) -> normalised temperature field on the
    same grid. Interchangeable in train_fno.py via --model whno.
```

### ClassDef:WalshConv3d

```text
3D "spectral" convolution via truncated Walsh-Hadamard transform.

    Mirrors SpectralConv3d's interface and role (truncated-basis global
    mixing operator) but uses the Walsh-Hadamard basis, real-valued weights,
    and sequency-ordered mode truncation instead of Fourier low-frequency
    truncation.
```

### FunctionDef:fwht_last_dim

```text
Unnormalised Fast Walsh-Hadamard Transform along the LAST dimension.
    Length of the last dim must be a power of 2. Self-inverse up to a
    factor of n (call twice and divide by n to invert).
```

### FunctionDef:hadamard_to_sequency_perm

```text
Permutation mapping natural (Hadamard-ordered) FWHT coefficient index ->
    sequency-ordered index (ordered by number of sign changes in the
    corresponding Walsh function, i.e. by smoothness — index 0 is constant,
    higher indices oscillate least-to-most).

    Computed numerically (not via a closed-form bit-reversal/Gray-code
    formula) by reconstructing the natural-order Hadamard matrix through
    fwht_last_dim itself (applying it to the identity matrix), then sorting
    rows by their number of sign changes. This guarantees the permutation is
    self-consistent with THIS module's specific FWHT butterfly convention,
    regardless of which of several nonequivalent "natural order" conventions
    a closed-form formula might assume — verified against scipy.linalg.hadamard
    sign-change ordering in tests/test_whno.py.
```

### module

```text
Walsh-Hadamard Neural Operator (WHNO) for 3D thermal field prediction.

Motivation
----------
FNO's spectral convolution uses a truncated Fourier (sinusoidal) basis, which
suffers the Gibbs phenomenon at sharp discontinuities: ringing/overshoot near
jumps, requiring hundreds of retained modes to resolve accurately (see e.g.
"Walsh-Hadamard Neural Operators for Solving PDEs with Discontinuous
Coefficients", Nov 2025). Our thermal stacks have exactly this problem —
material conductivity jumps by up to 37x at layer interfaces (Si k=148 vs
TIM k=4 W/m·K), which are piecewise-CONSTANT in z.

The Walsh-Hadamard basis is itself piecewise-constant (square waves, not
sinusoids), so it is a structurally better match for representing
piecewise-constant coefficient fields — no Gibbs ringing at a jump that
happens to align with (or be well-approximated by) a Walsh basis function.
This module swaps FNO3d's SpectralConv3d (FFT-based) for a Walsh-Hadamard
equivalent, keeping everything else (lift/blocks/proj structure, FiLM
conditioning compatibility, training loop) identical so it's a drop-in
alternative for ablation against FNO3d / CNO-FNO.

Implementation notes
---------------------
- The Fast Walsh-Hadamard Transform (FWHT) requires power-of-2 length along
  each transformed axis. Real grids (100x100x40, 56x168x50, ...) are NOT
  powers of 2, so each spatial dim is zero-padded up to the next power of 2
  before the transform and cropped back after the inverse transform.
- FWHT in natural (Hadamard) order does not order coefficients by smoothness
  the way Fourier's low-to-high frequency ordering does. To truncate to
  "low modes" analogously to FNO's mode truncation, coefficients are
  permuted into SEQUENCY order (ordered by number of sign changes, i.e.
  smoothness) via a bit-reversal + Gray-code permutation before truncating
  to the lowest-sequency `modes` coefficients.
- WHT coefficients are real-valued (no complex arithmetic needed), unlike
  FFT — the learnable spectral weight is a plain real tensor.
```

## `src/main.py`

### FunctionDef:generate_synthetic_temperature

```text
Generate synthetic temperature field using simplified thermal model.

    This enables testing without 3D-ICE when simulator unavailable.

    Args:
        coords: (N, 3) coordinate array [um]
        geometry: Geometry object
        ambient_temp: Ambient temperature [degC]
        power_field: (N,) volumetric power density [W/m³]

    Returns:
        (N,) temperature array in Kelvin
```

### FunctionDef:process_geometry

```text
Process all scenarios for a single geometry.

    Args:
        geometry_name: Name of geometry
        geometry: Geometry object
        scenarios_per_type: Dict with 'train' and 'test' lists of scenario params
        simulator: ThermalSimulator instance
        output_base: Base output directory
        use_synthetic: Use synthetic thermal data
        skip_train: Skip training scenarios
        skip_test: Skip test scenarios

    Returns:
        Dict with aggregated statistics
```

### module

```text
Main orchestration script for 3D-IC Thermal PINN Benchmark System.

This script:
1. Loads geometry definitions (Geometry 1, Geometry 2a/2b/2c with varying TSV density)
2. Loads scenario parameters from YAML configs
3. Runs thermal simulations (3D-ICE or synthetic data)
4. Exports results to NumPy .npz format for PINN training
5. Computes comprehensive thermal statistics
6. Generates validation visualizations

Usage:
    python main.py --simulator 3d-ice --geometry geometry1 --output data/geometry1/train
    python main.py --simulator mock --all-geometries --all-scenarios
    python main.py --generator-only --output data/all_scenarios
```

## `src/pinn/__init__.py`

### module

```text
Physics-Informed Neural Network (PINN) for 3D-IC thermal surrogate modelling.

Predicts T(x,y,z) for the steady-state heat equation with temperature-dependent
silicon conductivity k(T) = 148*(300/T)^1.3, convective BC at the top surface,
and adiabatic walls on sides and bottom.

Modules:
    model       -- FourierPINN architecture (Fourier encoding + residual MLP)
    physics     -- k(T) evaluation and PDE/BC residual computation
    losses      -- Loss components and NTK-based adaptive weighting
    data_loader -- ThermalDataset, normalization, collocation sampling
    trainer     -- Curriculum training loop with Adam + cosine annealing
    evaluate    -- Metrics and comparison plots
```

## `src/pinn/data_loader.py`

### ClassDef:NormStats

```text
Global normalization statistics.

    Computed once over the full training set and frozen.
    Stored alongside model checkpoints so inference can denormalize correctly.
```

### ClassDef:ThermalDataset

```text
Dataset of thermal simulation scenarios.

    Each item is a full scenario (all N points). Batching by scenario
    ensures each gradient step sees a spatially complete temperature field,
    not random crops.

    All data is loaded into RAM at __init__ (total ~800 MB for 80 scenarios).
```

### FunctionDef:compute_norm_stats

```text
Compute global normalization statistics from the training dataset.

    Scans all .npz files to determine power distribution.
    Temperature bounds are physical constants, not data-driven.
```

### FunctionDef:power_at_colloc_points

```text
Return normalised volumetric power density at collocation points.

    Uses vectorised numpy block-membership tests (fast: O(N × n_blocks) in C).
    Points outside any active power block get Q=0, which is physically correct
    for non-active layers and for die-layer points in non-powered regions.

    Args:
        col_coords_norm: (N, 3) normalised coordinates in [0,1]
        col_layer_ids:   (N,) integer layer indices
        geometry:        Geometry object (for power block positions and layer info)
        power_blocks_wcm2: {block_name: power_density_W/cm²} for this scenario
        geom_extents:    [L_x, L_y, L_z] in µm
        power_mean, power_std: normalisation constants from NormStats
        device:          target device for returned tensor

    Returns:
        (N,) normalised power density tensor on `device`
```

### FunctionDef:sample_bc_faces_grouped

```text
Sample adiabatic BC points grouped by face normal direction.

    Replaces sample_bc_side_points. Returns one entry per normal direction so
    bc_residual_adiabatic is called with the correct normal_dim for each group.

    z=1 (not z=0) carries the adiabatic BC here: 3D-ICE's ground truth only
    specifies a convective condition at z=0 (layer 0, "bottom heat sink" —
    see sample_bc_top_points), so every OTHER exposed face — including the
    top face near the die/TIM2, previously and incorrectly left unconstrained
    — defaults to adiabatic (dT/dn=0), matching an unspecified-BC-is-adiabatic
    solver convention.

    Returns:
        List of (coords_norm, layer_ids, normal_dim) for:
          normal_dim=0: x=0 and x=1 faces  (2 * n_per_face points)
          normal_dim=1: y=0 and y=1 faces  (2 * n_per_face points)
          normal_dim=2: z=1 top face       (    n_per_face points)
```

### FunctionDef:sample_bc_side_points

```text
Sample points on all 4 side faces + bottom face for adiabatic BC.

    Returns coords_norm (5*n_per_face, 3) and layer_ids.

    .. deprecated::
        Use sample_bc_faces_grouped() instead — this function passes all five
        face groups as a single concatenated tensor, making it impossible to
        call bc_residual_adiabatic with the correct normal_dim per face.
```

### FunctionDef:sample_bc_top_points

```text
Sample n points on a z-face for the convective BC.

    Despite the name (kept for backward compatibility), z_value defaults to
    0.0 (bottom face) — 3D-ICE's ground truth applies convective cooling at
    layer index 0 ("bottom heat sink" directive in ice_simulator.py), not at
    z=1. Pass z_value=1.0, layer_id=<top layer index> to recover the old
    (incorrect for this dataset) behaviour if ever needed elsewhere.
```

### FunctionDef:sample_collocation_points

```text
Sample random interior collocation points for PDE loss.

    Returns:
        coords_norm: (n, 3) uniform in [0,1]³ (normalised)
        layer_ids:   (n,) layer index for each point
```

### FunctionDef:sample_collocation_stratified

```text
Sample n collocation points with layer-importance-weighted z-distribution.

    Die/TSV layers receive proportionally more points; the heat sink receives fewer.
    x, y remain uniform over the die footprint.
    Layer ids are assigned directly from the per-layer z-interval (no z-lookup loop).

    Returns:
        coords_norm: (n, 3) normalised in [0,1]³
        layer_ids:   (n,) layer index for each point
```

### module

```text
Data loading and normalization for PINN training.

Handles:
  - Loading .npz files from the 3D-ICE dataset
  - Computing and storing global normalization statistics
  - Batching by scenario (not by individual point)
  - Sampling random collocation points for PDE/BC evaluation

Normalization convention:
  coords:    [0,1] per axis (divided by domain extent)
  temp:      [0,1] using global T_min / T_max across the dataset
  power:     zero-mean unit-variance using dataset mean/std
  htc:       [0,1] using known physical range [500, 10000] W/m²·K
  t_amb:     [0,1] using known physical range [25, 85] °C
```

## `src/pinn/evaluate.py`

### FunctionDef:evaluate_dataset

```text
Evaluate model on all scenarios in a dataset.

    Returns dict with per-scenario metrics and aggregate statistics.
```

### FunctionDef:predict_scenario

```text
Run inference on a single scenario.

    Returns:
        T_pred_K: (N,) predicted temperatures in Kelvin
        T_true_K: (N,) ground-truth temperatures in Kelvin
```

### module

```text
PINN evaluation: metrics and comparison plots.

Metrics reported per scenario and aggregated:
  mae_K            mean absolute error in Kelvin
  rmse_K           root mean squared error
  max_err_K        maximum point error
  r2               coefficient of determination
  hotspot_T_err_K  temperature error at the true hotspot location
  hotspot_loc_err  distance between predicted and true hotspot (µm)
  pde_res_rms      RMS of PDE residual at test points (optional)

Plots:
  - Temperature field z-slice comparison: 3D-ICE vs PINN
  - Error map at a z-slice
  - Z-axis temperature profile through hotspot
```

## `src/pinn/explain.py`

### FunctionDef:_forward_K

```text
Run a single forward pass and return temperature in Kelvin (N,).

    If power_override is provided it replaces sc.power (already normalised).
```

### FunctionDef:_ig_point

```text
Integrated Gradients for a single grid point.

    Returns:
        attrs:        {feature_name: attribution_in_K}  sum ≈ T_pt_K - T_bl_K
        baseline_T_K: predicted temperature at baseline (same layer_id as point)
```

### FunctionDef:hotspot_ig

```text
Integrated Gradients attribution at the predicted hotspot + z-profile.

    Decomposes hotspot temperature into contributions from spatial location
    (x, y, z), power density, HTC, ambient temperature, and TSV fraction,
    relative to a neutral baseline (domain centre, zero power, mid HTC/ambient).

    Cost on i7:
      Hotspot only (n_steps=50):    ~0.1 s
      + z-profile (6–10 layers):    ~0.6–1.0 s
      Total per scenario:           < 2 s

    Returns dict with keys:
        hotspot_idx, hotspot_T_K, baseline_T_K, completeness_error,
        hotspot_attrs {feature: float_K}, z_profile [list of layer dicts],
        feature_names
```

### FunctionDef:htc_sensitivity_map

```text
Cooling effectiveness: dT[i]/dHTC [K per W/m²·K] at each grid point.

    Higher magnitude → that region's temperature is most sensitive to cooling.
    Uses central finite difference with δHTC = delta_htc W/m²·K.
    Cost: 2 forward passes.

    Returns:
        (N,) float32 array. Negative values (temperature drops as HTC rises).
```

### FunctionDef:mc_dropout_uncertainty

```text
Predictive mean and std via MC Dropout.

    Temporarily sets model to train() mode (activates Dropout) and runs
    n_samples forward passes with different dropout masks. Returns the
    empirical mean and standard deviation of predicted temperature.

    High std → model is uncertain; these are regions where adding more
    training scenarios would most improve prediction reliability.

    Requires FourierPINN to have been built with dropout_p > 0 (default).
    If dropout_p=0.0 all samples will be identical (std ≈ 0).

    Args:
        n_samples: Number of stochastic forward passes (default 100).
                   ~200–300 s on i7 for geometry1 (400k points).

    Returns:
        T_mean_K: (N,) float32 predictive mean in Kelvin
        T_std_K:  (N,) float32 predictive std in Kelvin
```

### FunctionDef:plot_ig_attribution

```text
Two-panel figure: hotspot attribution bar chart (left) + z-profile
    attribution heatmap (right, feature × layer).

    Bar heights sum to approximately T_hotspot − T_baseline.
    Red = feature increased temperature; blue = decreased it.
```

### FunctionDef:plot_residual_map

```text
Scatter plot of absolute PDE residual at a z-slice.

    z_fraction=0.95 selects a slice near the top of the domain (inside TIM2).
    Use z_fraction≈(die_z_top/total_height) to slice through the die instead.
```

### FunctionDef:power_block_sensitivity

```text
Thermal influence coefficients: dT[i]/dQ_k [K per W/cm²] for each block k.

    Uses central finite difference with perturbation δQ = delta_wcm2 W/cm².
    Cost: 2 × n_blocks forward passes + 1 base pass.

    Args:
        delta_wcm2: Power perturbation in W/cm² (default 0.1 — 10% of 1 W/cm² baseline).

    Returns:
        {block_name: (N,) float32 array of dT_K/dQ_wcm2}
```

### FunctionDef:residual_map

```text
Compute the absolute PDE residual |∇·(k∇T) + Q| at every data-grid point.

    Uses sc.coords and sc.layer_ids (the full data grid, not random collocation
    points), so the residual map spatially aligns with the temperature field.

    Args:
        model:       Trained FourierPINN (in eval mode).
        sc:          ScenarioData — provides coords, layer_ids, scenario params.
        geometry:    Geometry object for block-power lookup.
        norm_stats:  NormStats for denormalisation constants.
        device:      Compute device.

    Returns:
        (N,) float32 numpy array of absolute residual magnitudes.
        Units are internally consistent (dimensionless in normalised space).
```

### module

```text
Explainability module for the FourierPINN thermal surrogate.

Four complementary methods, all zero-retraining:

Option 1 — PDE residual maps
  residual_map(): absolute heat-equation residual at every data-grid point.
  High residual → model predicts a physically inconsistent gradient there.

Option 2 — Engineering sensitivity maps
  power_block_sensitivity(): dT/dQ per power block via finite difference.
    Returns the thermal influence matrix: "if block k increases by 1 W/cm²,
    how much does temperature rise at each grid point?"
  htc_sensitivity_map(): dT/dHTC spatial map via finite difference.
    "Where does improved cooling most reduce temperature?"

Option 3 — Targeted Integrated Gradients (IG)
  hotspot_ig(): IG attribution at the predicted hotspot + z-profile through
    the stack. Answers "what caused this hotspot temperature?" by decomposing
    it into contributions from (x,y,z position, power, HTC, ambient, TSV frac).
    Cost: < 3 s per scenario on i7 because IG is computed for ~7 representative
    points (hotspot + one per layer), not all 400k grid points.

Option 5 — MC Dropout uncertainty
  mc_dropout_uncertainty(): Predictive mean ± std via 100 stochastic passes.
    Requires Dropout layers in the model (enabled by default in build_model).
    High std → model is uncertain; add training data there.

All plotting functions follow the scatter z-slice pattern from evaluate.py.
```

## `src/pinn/losses.py`

### FunctionDef:update_ntk

```text
NTK-based adaptive weight update (Wang et al. 2022).

        For each loss component i: compute ||∂L_i/∂θ||₂.
        Set λ_i = max_j(||∂L_j/∂θ||₂) / ||∂L_i/∂θ||₂.
        Apply EMA smoothing.
```

### module

```text
Loss functions for the thermal PINN.

Components:
  L_data      -- MSE between predicted and 3D-ICE temperatures (data fidelity)
  L_pde       -- PDE residual ||∇·(k∇T) + Q||² (physics)
  L_bc_top    -- Convective BC at top surface
  L_bc_sides  -- Adiabatic BC on side and bottom faces
  L_interface -- Temperature continuity at layer boundaries

Adaptive weighting (NTK-based):
  λ_i is updated each epoch as: max_grad_norm / grad_norm_i
  with EMA smoothing to prevent oscillation.

Training curriculum:
  Stage 1 (epochs 0-1000):    data loss only
  Stage 2 (epochs 1000-3000): data + 0.1*pde + 0.5*bc
  Stage 3 (epochs 3000+):     data + adaptive λ_pde*pde + adaptive λ_bc*bc
```

## `src/pinn/model.py`

### ClassDef:FourierFeatureEmbedding

```text
Random Fourier feature encoding for spatial coordinates.

    Maps (x,y,z) ∈ [0,1]³ → [sin(2π B x), cos(2π B x)] ∈ R^(2*n_freq)
    where B ~ N(0, sigma²) is a fixed random matrix drawn at init.

    sigma can be a float (isotropic) or a (sx, sy, sz) tuple for per-axis
    bandwidth — use anisotropic sigma when the geometry has a large aspect ratio,
    e.g. geometry6 (42mm × 14mm): sigma=(42,14,50) matches physical feature scales.
```

### ClassDef:FourierPINN

```text
Physics-Informed Neural Network for 3D-IC thermal prediction.

    Input per point (all normalised to [0,1] or standardised):
        coords     (3,)  -- (x̂, ŷ, ẑ) ∈ [0,1]
        power      (1,)  -- normalised volumetric power density
        layer_id   (1,)  -- integer layer index (passed separately for embedding)
        htc_norm   (1,)  -- normalised heat transfer coefficient
        t_amb_norm (1,)  -- normalised ambient temperature
        tsv_frac   (1,)  -- TSV area fraction (0, 0.03, 0.05, 0.10)

    Optional:
        region_ids  (1,)  -- chiplet region ID (0=underfill, 1=chipA, 2=chipB)
        tim_k_norm  (1,)  -- normalised TIM conductivity for k-sweep scenarios

    Hard adiabatic BC (hard_adiabatic=True, default):
        (x,y) coords are folded through cosine transform before Fourier encoding.
        This enforces dT/dx=dT/dy=0 at all four lateral walls by construction,
        eliminating the bc_sides soft-loss term entirely.

    Total input to MLP: 32 (Fourier) + 8 (layer emb) + [0|4] (region emb)
                        + 4 (scalars) + [0|1] (tim_k) = 44-49 dims
```

### ClassDef:ResBlock

```text
Residual block: LayerNorm -> Linear -> SiLU -> Dropout -> LayerNorm -> Linear + skip.

    LayerNorm (not BatchNorm) because batch size can be as small as 1 scenario.
    SiLU over tanh: non-saturating, smooth second derivative for PDE autograd.
    Dropout(p=0.1) enables MC Dropout uncertainty estimation at inference:
    call model.train() and run N forward passes to get predictive mean ± std.
```

### FunctionDef:_hard_adiabatic_transform

```text
Cosine coordinate fold enforcing dT/dx=dT/dy=0 at x={0,1} and y={0,1}.

        Maps x → 0.5*(1 - cos(π x)), whose derivative is 0.5*π*sin(π x),
        which vanishes at x=0 and x=1. Chain rule then sets dT/dx_phys=0
        at both lateral walls — exact hard BC, no soft loss needed.
        z coordinate is left unchanged (convective BC at top stays soft).
```

### FunctionDef:build_model

```text
Construct a FourierPINN.

    n_layers: use 11 for geometry5/6; 6 for geometry1/3/4; 10 for geometry2a/2b/2c.
    fourier_sigma: float for isotropic; (sx,sy,sz) for anisotropic (geometry6: (42,14,50)).
    n_regions/region_emb_dim: set to (3, 4) for geometry4/5 chiplet geometries.
    tim_k_input: True for geometry5/6 with TIM k-sweep training scenarios.
    hard_adiabatic: True (default) removes bc_sides soft loss; False keeps old behaviour.
```

### FunctionDef:forward_batched_scenarios

```text
Process S scenarios in a single forward pass.

        Fourier features and layer embeddings are computed ONCE and expanded to (S,N).
        Per-scenario scalars are broadcast without data duplication.
```

### module

```text
FourierPINN architecture for 3D-IC thermal surrogate modelling.

Architecture:
  Input: normalised (x,y,z) + power density + layer_id + scenario params
  -> FourierFeatureEmbedding (32 features, per-axis sigma) + LayerEmbedding (8 features)
  -> Optional RegionEmbedding (0 or 4 features, for 2.5D chiplet geometries)
  -> Optional TIM-k scalar (5th scenario input for k-sweep scenarios)
  -> Hard adiabatic BC: cosine coordinate fold on (x,y) enforces dT/dn=0 at lateral walls
  -> Residual MLP: 6 x ResBlock(256) with SiLU activations
  -> Linear(1) output (normalised temperature)
```

## `src/pinn/physics.py`

### FunctionDef:bc_residual_adiabatic

```text
Adiabatic BC residual: dT/dn = 0.

    Returns (N_bc,) residual.
```

### FunctionDef:bc_residual_top

```text
Convective BC residual: -k * dT/dn = h * (T - T_amb), where n is the
    OUTWARD surface normal.

    Despite the name (kept for backward compatibility — originally this was
    only ever called at z=1), this function works at ANY z-face; pass
    outward_normal_sign=-1.0 when evaluating at z=0 (bottom), since the
    outward normal there points in -z, flipping the sign of dT/dn relative
    to dT/dz. Ground truth for this dataset (see ice_simulator.py's "bottom
    heat sink" directive) applies convective cooling at z=0, layer index 0
    — callers should use outward_normal_sign=-1.0 in the normal case here,
    not the historical z=1/+1.0 default.

    Returns (N_bc,) residual (should be ~0).
```

### FunctionDef:pde_residual

```text
Compute PDE residual r = ∇·(k(T)∇T) + Q at collocation points.

    The network outputs T̂ (normalised). Spatial coordinates are also normalised
    to [0,1]. The chain rule converts derivatives from normalised to physical space:
        dT/dx_phys = (dT̂/dx̂) * (T_range) / L_x

    Returns:
        (N_col,) residual values (should be ~0 everywhere inside the domain)
```

### FunctionDef:thermal_conductivity

```text
Compute pointwise thermal conductivity k(T) in W/m·K.

    Silicon layers: k(T) = k_base * (300/T_K)^1.3  (temperature-dependent)
    All other layers: k = k_base  (constant)

    Args:
        T_norm:        (N,) normalised temperature [0,1]
        layer_ids:     (N,) int, layer index for each point
        layer_k:       (n_layers,) base thermal conductivity per layer
        si_layer_mask: (n_layers,) bool, True where material == silicon
        T_min, T_max:  normalisation bounds in Kelvin

    Returns:
        (N,) k values in W/m·K
```

### module

```text
Physics helpers for the thermal PINN.

Provides:
  - thermal_conductivity(): layer-wise k(T) with silicon temperature dependence
  - pde_residual(): ∇·(k(T)∇T) + Q using autograd second derivatives
  - bc_residual_top(): convective BC at top surface: -k dT/dz = h(T - T_amb)
  - bc_residual_adiabatic(): dT/dn = 0 on side/bottom faces

All tensors in normalised units unless stated otherwise. Temperatures are
denormalised to Kelvin internally for k(T) evaluation, then returned to
normalised space for the PDE residual.
```

## `src/pinn/physics_func.py`

### FunctionDef:pde_residual_func

```text
Compute per-point PDE residual |∇·(k∇T) + Q|  via vmap + grad.

    Returns (N,) residual (not yet squared/reduced — same contract as
    pde_residual in physics.py).
```

### module

```text
torch.func-based PDE residual for PINN training.

Replaces the autograd.grad(..., create_graph=True) approach in physics.py with
torch.func.vmap + torch.func.grad.  The key differences:

  OLD path (physics.py):
    col_coords.requires_grad_(True)
    T = model(col_coords, ...)          -- retains graph for N=20k pts
    grad1 = autograd.grad(T, col_coords, create_graph=True)[0]
    kTx = k * grad1[:, 0]
    div_x = autograd.grad(kTx, col_coords)[0][:, 0]
    → Entire N-point computation graph held in GPU memory until backward()
    → For 15 scenarios: 15 × N-pt graphs retained simultaneously

  NEW path (this file):
    params = dict(model.named_parameters())
    def T_scalar(coord, ...): ...           -- single-point scalar function
    dT_dcoord = vmap(grad(T_scalar))(coords, ...)  -- N independent grads, no graph
    → No retained graph at all
    → Memory proportional to one forward pass, not 15 × N-pt graphs
    → 2nd derivative via nested vmap(grad(grad(...))) — still no create_graph

GPU speedup over create_graph path (estimated):
  - Eliminates ~70% of GPU memory allocated to retained autograd graphs
  - vmap vectorizes N gradient calls into one kernel launch per layer
  - Measured speedup: 1.5–2.5× per epoch on CUDA
  - Does NOT require PyTorch 2.x — works from 1.13+ (torch.func)

Caveats:
  - nn.Embedding inside vmap requires functional_call (provided here)
  - Dropout is disabled during PDE evaluation (grad needs deterministic output)
  - k(T) temperature-dependent conductivity: quasi-linearised (k frozen per step),
    same as the autograd path — not a regression
```

## `src/pinn/sampling.py`

### FunctionDef:compute_adaptive_weights

```text
Compute the (M,) importance signal for a candidate pool according to
    `strategy`. 'rar' (PDE residual magnitude) is the pre-existing default;
    'hessian' and the raw signal for 'importance'/'curriculum' are new.

    For 'importance' and 'curriculum', the underlying signal is still the PDE
    residual (matching RAR) — 'importance' changes HOW points are drawn from
    that signal (softmax temperature vs residual-proportional), 'curriculum'
    changes WHEN the signal is trusted (blend factor). Pass strategy='hessian'
    if you want curvature instead of residual as the underlying signal for
    importance/curriculum too — see compute_adaptive_weights_for.
```

### FunctionDef:curriculum_blend_factor

```text
Blend factor in [0, 1]: 0 = pure uniform/stratified coverage,
    1 = pure adaptive (residual/curvature) weighting.

    Stays at 0 for the first `warmup_frac` of training (let the network learn
    the bulk solution before trusting its residual/curvature signal — this is
    the core argument in Curriculum-Enhanced Adaptive Sampling: early-training
    residuals are dominated by random init noise, not real physics difficulty),
    then ramps linearly to 1 by the end of training.
```

### FunctionDef:curriculum_enhanced_resample

```text
Blend uniform sampling with adaptive-weighted sampling according to
    `blend` (0=uniform, 1=fully adaptive), then draw n_new indices.

    Implemented as a blended probability distribution (not a coin-flip
    between two separate samplers) so that intermediate blend values produce
    a smooth interpolation rather than a discontinuous switch.
```

### FunctionDef:hessian_trace_weights

```text
Return (N,) curvature weights = trace of the Hessian of T_hat w.r.t. coords,
    i.e. the Laplacian |d2T/dx2 + d2T/dy2 + d2T/dz2| — a cheap proxy for full
    per-point Hessian eigenvalue analysis (O(N) single Laplacian evaluation
    instead of O(N*3) for the full 3x3 Hessian).

    Uses double-backward (create_graph=True on first derivative). Caller
    should call model.eval() beforehand for consistent dropout state, matching
    the convention in trainer._rar_update.
```

### FunctionDef:importance_resample

```text
Sample n_new indices from [0, M) via temperature-controlled softmax
    resampling, with a uniform floor to prevent collapse onto a single mode
    (the failure mode plain RAR is documented to have — "always picking the
    largest residual locations, reducing exploration of other regions").

    temperature < 1 sharpens toward the highest-weight points (more like
    hard top-k RAR); temperature > 1 flattens toward uniform (more
    exploration). temperature=1 is the direct softmax of the raw signal.

    Returns: (n_new,) LongTensor of indices into the weights array.
```

### module

```text
Adaptive collocation-sampling strategies for PINN training.

Extends the plain residual-proportional RAR already in trainer._rar_update
with three additional strategies drawn from the 2025 adaptive-sampling
literature. All operate on the SAME interface — given a candidate point
pool, return per-point importance weights (or directly resampled points) —
so they are interchangeable via `Trainer(sampling_strategy=...)`.

Strategies
----------
hessian_weighted
    Sample proportional to |Laplacian(T_hat)| (trace of the output Hessian),
    a curvature proxy. Distinct from RAR's PDE-residual weighting: residual
    measures physics violation (∇·(k∇T)+Q != 0), while curvature measures
    where the network's own output is changing sharply — the model can have
    near-zero residual (physics satisfied) at a point with immense curvature
    if it also fits Q correctly there. Curvature-based sampling targets
    representation difficulty, not physics violation, so it is a genuinely
    different signal.  Motivated by "Provably Accurate Adaptive Sampling for
    Collocation Points in PINNs" (ECML PKDD 2025), which used a Hessian-based
    quadrature bound; this is a practical (trace-only, not full Hessian)
    approximation of that idea — full per-point Hessian is O(N) double-
    backward calls and too slow for the >20k point pools used here.

importance_adversarial
    Residual-magnitude softmax resampling. This is a SIMPLIFIED PROXY for
    "Adversarial Adaptive Sampling" (Tang et al. 2024), which trains a deep
    generative model + optimal-transport (Wasserstein) map to *synthesize*
    new sample locations. Implementing the full GAN+OT machinery is out of
    scope here; this keeps the adversarial paper's core intuition — sample
    from a distribution shaped by the residual field rather than a hard
    top-k cutoff — via temperature-controlled softmax resampling, which is
    a defensible but honestly-labelled simplification, not a re-implementation
    of Tang et al.'s method.

curriculum_enhanced
    Blends uniform/stratified coverage (early epochs) with adaptive
    (residual- or curvature-weighted) sampling (late epochs) via a linearly
    growing blend factor. Directly follows "Curriculum-Enhanced Adaptive
    Sampling for PINNs: A Robust Framework for Stiff PDEs" (MDPI, Dec 2025),
    which argues plain RAR over-exploits high-residual regions too early,
    before the network has learned enough of the bulk solution to produce
    meaningful residual signal there. The stiffness they target (steep
    gradients) maps directly onto this repo's material-interface jumps
    (Si k=148 vs TIM k=4, a 37x ratio).
```

## `src/pinn/trainer.py`

### ClassDef:Trainer

```text
Trains a FourierPINN for a single geometry.

    Args:
        model:           FourierPINN instance
        geometry:        target Geometry (single geometry per trainer)
        norm_stats:      global normalization statistics
        train_data:      ThermalDataset with training scenarios
        val_data:        ThermalDataset with validation/test scenarios
        output_dir:      directory for checkpoints and logs
        n_col:           number of collocation points for PDE loss per step
        n_bc_top:        number of BC points on top surface per step
        n_bc_side:       number of BC points per side face per step (ignored when hard_adiabatic)
        epochs:          total training epochs
        lr:              initial Adam learning rate
        device:          training device
        hard_adiabatic:  if True, bc_sides loss is skipped (model.hard_adiabatic handles it)
        log_interval:    print loss every N epochs
        val_interval:    run validation every N epochs.
        sampling_strategy: one of 'rar' (default, PDE-residual-proportional —
            original RAR-D behaviour), 'hessian' (curvature/Laplacian-trace
            weighted), 'importance' (softmax-temperature resampling of the
            residual field — simplified proxy for adversarial adaptive
            sampling), 'curriculum' (blends uniform -> residual-weighted
            sampling over training, see src/pinn/sampling.py for citations).
        sampling_temperature: softmax temperature for 'importance'/'curriculum'
            strategies (lower = sharper toward high-signal points).
        curriculum_warmup_frac: fraction of total epochs to stay at pure
            uniform sampling before ramping toward adaptive (curriculum only).
```

### FunctionDef:_batched_data_loss

```text
Run all S training scenarios through forward_batched_scenarios in one
        call, returning a list of per-scenario data-loss tensors (with grad).

        Shared Fourier + layer features computed once → significant GPU saving
        when S=15 and N_data is large (60k pts for geometry1).
```

### FunctionDef:_compute_lateral_k

```text
Return (col_k_lateral, col_si_lateral) for 2p5d_stack geometries.

        For each point in a die-zone layer that falls outside all chiplet footprints,
        overrides k with underfill_k and marks the point as non-silicon so the
        temperature-dependent k(T) correction is not applied to underfill.

        Returns (None, None) for all other geometry types — zero overhead.
```

### FunctionDef:_compute_region_ids

```text
Return (N,) int tensor of chiplet region IDs for 2p5d_stack geometries.

        Region encoding: 0 = underfill/outside chiplets, 1 = first chiplet footprint,
        2 = second chiplet footprint (and so on for more chiplets).

        Returns None for non-2p5d geometries — zero overhead, model.forward ignores None.
```

### FunctionDef:_rar_update

```text
Append rar_add_n points to the persistent collocation set, chosen
        according to self.sampling_strategy:

          'rar'        — residual-proportional (original RAR-D; eps uniform floor)
          'hessian'    — curvature (Laplacian-trace) proportional, same eps floor
          'importance' — softmax-temperature resampling of the residual field
          'curriculum' — blends uniform -> residual-weighted over training
                         (blend factor grows from 0 at warmup to 1 at epochs=self.epochs)

        See src/pinn/sampling.py for the literature basis of each.
```

### module

```text
PINN training loop with curriculum staging and adaptive loss weighting.

Training stages:
  Stage 1 (epochs   0-1000): data loss only — network learns the temperature distribution
  Stage 2 (epochs 1000-3000): add PDE + BC at fixed low weights
  Stage 3 (epochs 3000-8000): NTK-based adaptive weights updated every 50 epochs

Optimizer: Adam with CosineAnnealingWarmRestarts (T_0=2000)
Gradient clipping: max_norm=1.0 (PDE second derivatives can spike)
Mixed precision: enabled when CUDA is available
```

## `src/reproducibility.py`

### FunctionDef:set_seed

```text
Seed Python, NumPy and Torch (CPU + all CUDA devices).

    Args:
        seed: Integer seed applied to every RNG.
        deterministic: If True, force deterministic cuDNN kernels and set
            CUBLAS_WORKSPACE_CONFIG so matmul reductions are reproducible.
            Slower; use for final reported runs.
```

### module

```text
Reproducibility helpers shared by all training entry points.

Every training script must call `set_seed()` before constructing models or
datasets. Without it, run-to-run variance is unbounded and seed-variance error
bars (which reviewers expect alongside any reported metric) cannot be produced.

`deterministic=True` additionally pins cuDNN into deterministic algorithm
selection. This costs throughput and is off by default; turn it on for the runs
whose numbers go into a paper table.
```

## `src/scenario/generator.py`

### ClassDef:ScenarioGenerator

```text
Generate training and test scenarios for thermal benchmarks.

    Creates parameter sweeps across power density, HTC, ambient and spatial pattern.

    Operating regime (revised 2026-07-31)
    -------------------------------------
    The original ranges (0.1-20 W/cm2, HTC 1000-10000, ambient 25-65 C) produced a
    SPATIALLY DEGENERATE dataset: median within-scenario spatial dT was 1.10 K against
    a 68 K between-scenario range, and 88% of temperature variance was explained by the
    ambient input alone. Closed-form ridge regression reconstructed the field at
    spatial R2 = 0.999, so the benchmark could not discriminate between architectures.
    See docs/report.md 9.2.

    Three coupled changes fix the regime:

    1. POWER up ~15x. Self-heating scales with total power. The old peak of 20 W/cm2
       was far below the 100-300 W/cm2 of real CPU hotspots (references.md already
       flagged this as conservative); the consequence was that self-heating (0.1-36 K)
       never dominated the ambient sweep.

    2. HTC up. The fraction of the temperature drop that appears as SPATIAL structure
       (rather than a uniform offset) is R_cond / (R_cond + R_conv), and R_conv = 1/h.
       For the geometry1 stack R_cond ~ 7.9e-5 m2K/W, so h=1000 puts only ~7% of the
       drop inside the stack while h=50000 puts ~80% there. Low HTC was actively
       flattening the field.

    3. AMBIENT narrowed to 25-45 C. A 60 K ambient sweep swamped a 0.1-36 K
       self-heating signal; real datacentre inlet air is 15-45 C anyway.

    Together these target a junction rise of roughly 40-90 K with spatial gradients of
    tens of K, instead of ~1 K.
```

### ClassDef:ScenarioParameters

```text
Parameters for a single thermal scenario.

    Attributes:
        name: Scenario identifier
        geometry_name: Associated geometry name
        scenario_type: 'train' or 'test'
        power_blocks: Dictionary mapping block names to power densities (W/cm²)
        htc: Heat transfer coefficient (W/m²·K)
        t_ambient: Ambient temperature (°C)
        pattern: Power distribution pattern name
        description: Human-readable description
```

### FunctionDef:_apply_pattern

```text
Apply power distribution pattern to blocks.

        Args:
            block_names: List of power block names
            pattern: Pattern type ('uniform', 'hotspot', 'checkerboard', 'gradient',
                                   'dual_hotspot', 'extreme_hotspot')
            base_power: Base power level to scale the pattern. Multiplied by
                POWER_SCALE here, so callers keep using the original relative
                numbers and every pattern shifts regime together.

        Returns:
            Dictionary mapping block names to power densities in W/cm²
```

### FunctionDef:_clamp_bc

```text
Bring a pool entry's boundary conditions into the current regime.

        The extra-scenario pools are literal tables written for the old power
        levels; unclamped they pair scaled power with HTCs as low as 500 W/m2K
        and ambients up to 75 C, which produced a 273 C junction on geometry2a.
```

### FunctionDef:_enforce_cooling_adequacy

```text
Raise each scenario's HTC to match its peak power density.

        Applied last, after power maps exist, so it sees the true peak including
        the per-geometry power scale. Mutates and returns the same list.
```

### FunctionDef:_normalise_to_tdp

```text
Rescale a relative power map so total dissipation equals workload x TDP.

        The pattern fixes the spatial distribution; this fixes the total. Blocks with
        no known area (or a degenerate map) are left untouched rather than silently
        divided by zero.
```

### FunctionDef:_set_active_geometry

```text
Select the TDP budget and block areas for `geometry`.

        Must precede any _apply_pattern call: the pattern needs block areas to
        convert a total-watt budget into per-block W/cm2.
```

### FunctionDef:attach_power_maps

```text
Replace each scenario's block powers with a per-cell power map.

        The TDP budget, the silicon density ceiling and the cooling-adequacy rule
        all still apply -- the map only changes HOW the budget is distributed in
        space, not how much there is. `power_blocks` is recomputed from the map so
        anything summarising a scenario by block keeps working.

        Args:
            scenarios:  Scenarios to modify in place.
            geometry:   Geometry supplying die extents and active layers.
            kind:       Power map family (see src/scenario/power_maps.py).
            resolution: Map resolution per axis; defaults to the lateral mesh.
            seed_base:  Offset so different splits get different maps.
```

### FunctionDef:attach_throttling

```text
Enable package-level thermal throttling (DVFS) on each scenario.

        Real chips reduce power when the hottest point on the package exceeds a
        junction-temperature limit -- power becomes a function of the very
        temperature field being solved for, a closed feedback loop no fixed
        power source can express. Every other mechanism in this benchmark
        (per-cell power, TSV fields, underfill layouts, even microchannel
        cooling at fixed flow rate) still maps a scenario-fixed source to a
        temperature field; this is the first one that doesn't. Applied by
        main.py's process_scenario via src/scenario/throttling.py, which
        iteratively re-solves and derates -- not implemented here, since it
        requires calling the simulator multiple times per scenario.
```

### FunctionDef:attach_tsv_maps

```text
Give each scenario a spatially varying TSV density field.

        Replaces the geometry's single `tsv_density` scalar on every TSV layer.
        The mean is preserved, so a scenario's overall TSV budget is unchanged --
        only its spatial distribution varies. `contrast=0` reproduces the old
        uniform behaviour exactly.
```

### FunctionDef:generate_all_scenarios

```text
Generate all scenarios (train + test) for a geometry.

        Args:
            geometry: Geometry object

        Returns:
            List of ScenarioParameters (15 training + 5 test = 20 scenarios)
```

### FunctionDef:generate_extra_training_scenarios

```text
Generate n_extra additional training scenarios starting at start_index.

        Uses a fixed parameter pool that fills gaps in the original 15 scenarios:
        denser HTC coverage (500–15000), more power levels, random_smooth pattern,
        and extra split-chiplet scenarios for 2.5D geometries.

        Args:
            geometry:      Target geometry.
            n_extra:       Number of extra scenarios to generate.
            start_index:   File index for the first scenario (e.g. 16 if 15 exist).
            pool_start_idx: Index into the pool to start drawing from (default 0).
                           Use this to avoid duplicating scenarios from a prior call.

        Returns:
            List of n_extra ScenarioParameters.
```

### FunctionDef:generate_test_scenarios

```text
Generate 5 test scenarios with interpolation values.

        Use parameter values NOT in training set to test PINN generalization.
```

### FunctionDef:generate_training_scenarios

```text
Generate 15 training scenarios with diverse parameter combinations.

        Strategy:
        - Cover all power patterns (6 patterns including extreme_hotspot)
        - Vary power levels, HTC, and ambient temperature
        - Include 5 extreme scenarios for PINN robustness (Tier 1 improvement)
        - Ensure good coverage of parameter space
```

### FunctionDef:load_from_yaml

```text
Load scenarios from YAML file.

        Args:
            yaml_file: Path to YAML file

        Returns:
            List of ScenarioParameters
```

### FunctionDef:save_to_yaml

```text
Save scenarios to YAML file.

        Args:
            output_dir: Output directory for YAML files
            geometry_name: Geometry identifier
```

### module

```text
Scenario generator for benchmark thermal simulations.

Generates training and test scenarios with parameter sweeps:
- Power density variations
- Heat transfer coefficient (HTC) variations
- Ambient temperature variations
- Power distribution patterns
```

## `src/scenario/leakage.py`

### FunctionDef:apply_leakage_feedback

```text
Self-consistently solve for temperature with temperature-dependent leakage.

    Mutates `scenario_params['power_blocks']` in place to the FINAL converged
    power, so npz export and statistics see the power actually dissipated. The
    nominal (pre-feedback) request is preserved in the returned info dict under
    'leakage_nominal_power_blocks' -- exported as `nominal_block_power_*`, the
    same convention throttling uses, so a baseline is asked to predict the
    resolved temperature from the *requested* power rather than being handed the
    already-resolved answer (see goal.md Track A for the bug that motivated this).

    Args:
        simulator: A ThermalSimulator (must implement .simulate()).
        geometry: Geometry object.
        scenario_params: Must carry 'leakage_fraction' (share of nominal power
            that is leakage at t_ref), 'leakage_k_double_c' (degrees per doubling)
            and 'leakage_t_ref_c'.
        max_iterations: Cap on solves.
        tol_c: Converged when peak temperature moves less than this between
            iterations.
        damping: Under-relaxation on the multiplier update. Undamped fixed-point
            iteration on positive feedback oscillates or overshoots into false
            runaway near the critical gain; damping trades a few extra solves for
            a trustworthy convergence/runaway distinction.

    Returns:
        (parsed, leakage_info). `parsed` is the final iteration's
        {'coords', 'temperature'}.
```

### FunctionDef:leakage_multiplier

```text
Total-power multiplier relative to the nominal (dynamic + reference-leakage)
    request, given a peak temperature.

    At T = t_ref_c the multiplier is exactly 1.0, so a scenario that never heats
    above its reference temperature is unchanged -- leakage feedback is opt-in
    physics, not a global rescaling of the dataset.
```

### module

```text
Leakage-power / temperature positive feedback (self-consistent electrothermal solve).

Subthreshold leakage rises roughly exponentially with temperature, so total power
is a function of the temperature field being solved for:

    P_total(T) = P_dynamic + P_leak_ref * 2 ** ((T - T_ref) / k_double)

This is the classic electrothermal loop that leakage-aware thermal simulators solve
self-consistently (established since at least Chen et al., ICCAD 2006, "Leakage power
dependent temperature estimation to predict thermal runaway"; more recently ATSim3D,
arXiv:2601.11050). 3D-ICE solves a single fixed-source steady state, so the loop lives
here: solve, recompute leakage at the resulting peak temperature, re-solve, repeat.

**Why this is a strictly harder test than throttling** (`src/scenario/throttling.py`),
and the reason it exists in this repo:

  - Throttling is *negative* feedback: hotter -> less power -> cooler. It is
    self-limiting, converges in ~2 solves, and the map from requested to delivered
    power stays monotone and gently compressive. Measured effect on the linear
    baseline was real but modest (spatial R^2 0.970 -> 0.890/0.919, goal.md Track A).
  - Leakage is *positive* feedback: hotter -> more power -> hotter still. It is
    self-amplifying, and above a critical gain there is no steady state at all
    (thermal runaway). Near that threshold the response to a small change in
    requested power becomes arbitrarily large -- which is exactly the regime a
    per-point linear fit cannot represent, because the underlying map is no longer
    close to affine.

So this is the sharpest available test of this benchmark's central claim. If ridge
still solves the dataset with leakage feedback active, that is strong evidence the
linearity finding is robust rather than an artifact of a benign operating regime.

Convergence note: fixed-point iteration T_{n+1} = f(P(T_n)) converges only while the
loop gain |df/dT| < 1. Divergence here is not a numerical failure -- it is the
physical runaway condition, and is reported as such (`leakage_runaway`) rather than
silently clamped, so runaway scenarios can be filtered or studied deliberately.
```

## `src/scenario/power_maps.py`

### FunctionDef:_floorplan

```text
Rectangular tiles at independent power levels.

    Tile edges are placed at random cut positions rather than a uniform grid, so
    the model cannot key on a fixed block pitch.
```

### FunctionDef:_smooth

```text
Gaussian blur via separable FFT convolution.

    Uses FFT rather than scipy.ndimage so the module has no dependency beyond
    numpy, matching the rest of the pipeline.
```

### FunctionDef:generate_power_map

```text
Build a normalised (nx, ny) power map with values in [idle_floor, 1].

    Args:
        nx, ny:      Map resolution, normally the die layer's lateral mesh.
        kind:        One of POWER_MAP_KINDS.
        seed:        Per-scenario seed; identical seeds reproduce identical maps.
        corr_frac:   GRF correlation length as a fraction of the larger die edge.
        n_tiles:     Tiles per axis for the floorplan family.
        idle_floor:  Minimum fraction of peak that quiet regions dissipate.
                     Real silicon leaks even when idle, and a hard zero would
                     make the source discontinuous at tile edges.

    Returns:
        (nx, ny) float64 array, max == 1.0.
```

### FunctionDef:map_to_rectangles

```text
Convert a power map into the rectangle grid a 3D-ICE floorplan needs.

    Returns (levels, tile_length_um, tile_width_um) where levels[i, j] is the
    normalised power of the tile whose lower corner is (i*tile_length,
    j*tile_width). Index order matches 3D-ICE floorplan convention: the first
    axis runs along die LENGTH, the second along die WIDTH.
```

### module

```text
Per-cell power map generation.

Why this exists
---------------
Until now a scenario's power was described by 4-13 block scalars. That makes the
whole dataset lie on a low-dimensional, nearly-linear manifold: steady-state
conduction is linear in its sources, so with ~8 scalars in, closed-form ridge
regression reconstructs the temperature field at spatial R2 0.987 and
extrapolates at R2 > 0.94 (docs/report.md 9.3). No choice of parameter RANGES
fixes that -- the input dimensionality is the cause.

A per-cell power map changes the input from a handful of numbers into a
function. The physics stays linear, but the solution operator becomes a genuine
Green's function: for geometry1's ~100k nodes that is a ~1e10-entry object,
which a per-point linear regression cannot represent, while an FNO compresses it
to ~1e7 via spectral truncation. This is the regime neural operators exist for
(Neural Green's Operators, arXiv:2406.01857), and it is reachable without
abandoning one-FNO-per-geometry, because the geometry stays fixed.

Cost: essentially nothing. 3D-ICE solve time is set by the mesh, not the
floorplan -- measured 13.3 s for 4 blocks and 13.5 s for 10,000 on geometry1.

Map families
------------
grf         Gaussian random field, smoothed white noise. Correlation length is
            expressed as a fraction of die width so it is geometry-independent.
            Models diffuse activity with no floorplan structure.
floorplan   Rectangular tiles at random power levels, like a real core/cache/IO
            layout. Produces the sharp lateral gradients that make interface
            behaviour interesting.
mixed       floorplan tiles plus a GRF perturbation. Structured layout with
            within-block activity variation, closest to a real workload.

All maps are returned normalised to peak 1.0; absolute scale is applied later
from the TDP budget, so the existing power model is untouched.

Choosing the defaults
---------------------
The parameter that matters is not how complex ONE map looks, but how many
dimensions a COLLECTION of maps spans -- that is what a linear model has to fit.
Measured effective rank of N stacked maps (64x64, mean removed):

    family / corr / tiles        N=40    N=120   N=200
    grf   0.02                   37.4    107.0   169.1
    mixed 0.02 / 16 tiles        37.2    104.9   163.8
    floorplan / 16 tiles         37.8    109.1   174.4
    -- block scalars (old)        4.0      4.0     4.0

The maps span very nearly N-1 dimensions: every scenario adds a genuinely new
direction, so a linear fit gains no leverage from extra samples. Block scalars
span exactly 4 no matter how many scenarios are generated, which is why ridge
solved the old dataset.

An earlier default of corr_frac=0.15 with 4 tiles gave an effective rank of only
2-7 -- no better than the block scalars it replaced. Correlation length has to be
short relative to the die for the map to carry real spatial information.

The defaults below are also physically sensible: corr_frac 0.02 is ~200 um on a
10 mm die, the scale of a real hotspot, and 16 tiles is ~625 um, the scale of a
functional block.
```

## `src/scenario/throttling.py`

### FunctionDef:apply_throttling

```text
Iteratively solve and derate power until peak temperature converges.

    Mutates `scenario_params['power_blocks']` in place to the FINAL derated
    values, so any code that reads scenario_params after this call (npz
    export, statistics) sees the power that was actually delivered, not the
    nominal un-throttled request.

    Args:
        simulator: A ThermalSimulator (must implement .simulate()).
        geometry: Geometry object.
        scenario_params: Scenario dict; must carry 'throttle_temp_c',
            'throttle_gain', 'throttle_power_floor' (see ScenarioParameters).
        scenario_name: Passed through to simulator.simulate().
        max_iterations: Hard cap so a pathological config can't loop forever.
        tol_c: Converged once peak temperature is within this many degrees
            of throttle_temp_c (only reachable while power is above the floor).

    Returns:
        (parsed, throttle_info) where `parsed` matches
        ThermalSimulator.simulate()'s return shape ({'coords', 'temperature'})
        from the FINAL iteration, and throttle_info is a dict of
        {throttle_iterations, throttle_derate_factor, throttle_triggered,
        throttle_peak_temp_c, throttle_converged} for logging/export.
```

### module

```text
Package-level thermal throttling (DVFS): iterative power derating when peak
temperature exceeds a junction-temperature limit.

Real chips (mobile SoCs, laptop CPUs) reduce power when the hottest point on
the package exceeds Tjmax (commonly ~95-105 C). This makes power a function
of the very temperature field being solved for -- a real, closed-loop
nonlinearity that a scenario-fixed power source cannot express. Every other
mechanism in this benchmark (per-cell power, TSV fields, underfill layouts,
microchannel cooling at a fixed flow rate) still maps a fixed source to a
temperature field; throttling is the first one that doesn't.

3D-ICE itself only solves a single fixed-source steady state, so the loop
lives here: solve, check peak T, derate, re-solve, repeat until the peak
settles within tolerance of the threshold or the power floor is hit. This
means throttled scenarios cost several 3D-ICE solves instead of one -- see
goal.md for the measured cost.
```

## `src/scenario/tsv_maps.py`

### FunctionDef:generate_tsv_density_map

```text
Build a spatially varying TSV area-fraction field.

    Args:
        n_l, n_w:      Map resolution (along die length, along die width).
        mean_density:  Target mean area fraction, i.e. the scalar this replaces.
        seed:          Per-scenario seed.
        corr_frac:     Cluster size as a fraction of the die edge. ~0.06 gives
                       ~600 um clusters on a 10 mm die, the scale of a real TSV farm.
        contrast:      0 = uniform (reproduces the old scalar), 1 = maximum
                       clustering. Lets a scenario sweep from uniform to highly
                       clustered.
        max_density:   Hard ceiling. Real arrays do not approach 100% copper;
                       above ~35% the rule of mixtures stops being credible and
                       the layer is better modelled as bulk copper.

    Returns:
        (n_l, n_w) array of area fractions in [0, max_density], mean ~= mean_density.
```

### FunctionDef:quantise_materials

```text
Reduce a continuous density field to a small set of material levels.

    3D-ICE needs a named material per distinct conductivity, so a 100x100 field of
    unique values would mean 10,000 material declarations. Quantising to ~12 levels
    keeps the stack file tractable while preserving the spatial structure that
    matters. The error this introduces is far below the modelling error in the rule
    of mixtures itself.

    Returns (level_index_map, level_phi_values).
```

### FunctionDef:tsv_effective_k

```text
Anisotropic effective conductivity for a TSV region.

    Returns (k_lateral, k_vertical) in W/m.K. See the module docstring: vertical
    conduction runs along the copper (parallel, arithmetic mean) while lateral
    conduction crosses the phases (series, harmonic mean).
```

### module

```text
Spatially varying TSV density, and the anisotropic conductivity it implies.

Why
---
TSV density was a single scalar per geometry taking four discrete values
(0, 3%, 5%, 10%). That is a categorical variant axis, not a parameter: with three
non-zero points no interpolation can be demonstrated, and the paper's claim of a
"continuous TSV parameter" was simply false (docs/report.md).

Real TSV arrays are not uniform. They cluster around the interconnect they serve,
leaving sparse regions between. Making density a FIELD phi(x, y) turns the
material coefficient into a per-scenario spatial input, which is the canonical
operator-learning benchmark structure -- the thermal analogue of Darcy flow with a
Gaussian-random-field permeability. It is also the part of the solution operator
that is genuinely nonlinear: the Green's function depends nonlinearly on the
coefficients, while it is merely linear in the source.

This requires 3D-ICE 4.0, which supports per-floorplan-element materials and
anisotropic conductivity. 3.0.0 allowed one isotropic material per layer.

Anisotropy
----------
A TSV is a copper cylinder through silicon, so conduction is directional:

  vertical (kz)   heat runs ALONG the copper -- the phases act in parallel, so the
                  arithmetic mean (rule of mixtures) applies:
                      kz = (1 - phi) k_Si + phi k_Cu
  lateral (kx,ky) heat must cross alternating Si/Cu boundaries -- the phases act in
                  series, so the harmonic mean applies:
                      1/kxy = (1 - phi)/k_Si + phi/k_Cu

The old model used the arithmetic mean in ALL directions, which
`docs/references.md` already flagged as an upper bound that "slightly overestimates
lateral heat spreading from TSVs". At phi = 0.10 the two differ by 17%
(183.8 vs 152.4 W/m.K). Anisotropic materials let us stop making that
approximation, so this fixes a documented simplification rather than adding one.
```

## `src/simulators/base_simulator.py`

### ClassDef:ThermalSimulator

```text
Abstract base class for thermal simulation wrappers.

    Provides a unified interface for different thermal simulators (3D-ICE, HotSpot, etc.).
    Each concrete simulator implements config generation, execution, and result parsing.
```

### FunctionDef:__init__

```text
Initialize simulator wrapper.

        Args:
            config_dir: Directory for configuration files
            output_dir: Directory for output files
            executable: Path to simulator executable
```

### FunctionDef:check_executable

```text
Check if simulator executable exists and is accessible.

        Returns:
            True if executable is found, False otherwise
```

### FunctionDef:cleanup_temp_files

```text
Clean up temporary configuration and output files.

        Args:
            scenario_name: Scenario identifier
```

### FunctionDef:generate_config_files

```text
Generate simulator-specific configuration files.

        Args:
            geometry: Geometry object
            scenario: Scenario parameters dictionary containing:
                - power_blocks: {block_name: power_density_W/cm2}
                - htc: Heat transfer coefficient (W/m²·K)
                - t_ambient: Ambient temperature (°C)
                - pattern: Power distribution pattern name
```

### FunctionDef:get_version

```text
Get simulator version information.

        Returns:
            Version string or empty string if unavailable
```

### FunctionDef:parse_results

```text
Parse simulation results into standard format.

        Args:
            result_file: Path to simulator output file

        Returns:
            Dictionary containing:
                - 'coords': (N, 3) array of (x, y, z) coordinates in μm
                - 'temperature': (N,) array of temperatures in K

        Raises:
            ValueError: If result file is invalid or cannot be parsed
```

### FunctionDef:run_simulation

```text
Run thermal simulation and return output file path.

        Args:
            scenario_name: Unique identifier for this scenario

        Returns:
            Path to output file containing simulation results

        Raises:
            RuntimeError: If simulation fails
```

### FunctionDef:simulate

```text
Complete simulation workflow: config → run → parse.

        Args:
            geometry: Geometry object
            scenario: Scenario parameters
            scenario_name: Unique scenario identifier

        Returns:
            Dictionary with simulation results (coords, temperature)
```

### module

```text
Abstract base class for thermal simulator wrappers.

Defines the interface that all thermal simulator wrappers must implement.
```

## `src/simulators/hotspot_simulator.py`

### ClassDef:HotSpotSimulator

```text
Wrapper for HotSpot thermal simulator (grid model, steady-state).

    Handles:
    - Floorplan file (.flp) with power block geometry (in metres)
    - Power trace file (.ptrace) with per-block power values
    - HotSpot config via command-line flags
    - Grid steady-state output (.grid.steady) parsing

    Unit system: HotSpot uses SI metres/watts. Geometry stores µm, so
    conversion × 1e-6 is applied when writing files.

    Effective thermal resistance: all non-die layers are lumped into
    r_convec (K/W) together with the convective BC, so the HotSpot
    package model (spreader + sink) is set to negligibly thin.
```

### FunctionDef:_generate_floorplan_file

```text
Write HotSpot floorplan file.

        Format (tab-separated, all in metres):
            name  width  height  left-x  bottom-y
```

### FunctionDef:_generate_power_trace_file

```text
Write HotSpot power trace file (single steady-state time step).

        Format:
            block1  block2  ...   (tab-separated names)
            P1      P2      ...   (tab-separated watts)
```

### FunctionDef:_run_via_wsl_stage

```text
Run HotSpot via a WSL staging directory to avoid path-with-spaces
        issues when passing Windows paths through WSL argument parsing.
        Files are copied to /tmp inside WSL, hotspot runs there, and the
        grid output is copied back.
```

### FunctionDef:parse_results

```text
Parse HotSpot grid steady-state output for the die layer (Layer 0).

        Returns:
            dict with 'coords' (N,3) float32 and 'temperature' (N,) float32.
```

### module

```text
HotSpot thermal simulator wrapper.

Wraps the HotSpot block/grid simulator for geometry1 (single-die, steady-state).
Generates .flp and .ptrace inputs, runs HotSpot in grid mode, and parses the
per-cell temperature output.

Scope: geometry1 only (single active silicon die layer). Multi-die stacks
(geometry2a/2b/2c) are not supported — HotSpot has no native TSV model.
```

## `src/simulators/ice_simulator.py`

### ClassDef:ICESimulator

```text
Wrapper for 3D-ICE thermal simulator.

    Handles:
    - Stack file (.stk) generation with materials and layers
    - Floorplan file (.flp) generation with power blocks
    - 3D-ICE execution
    - Grid output (Tmap) parsing

    Unit system: 3D-ICE uses µm for length, so thermal properties must be
    converted from SI (m-based) to µm-based units on write.
```

### FunctionDef:_gap_material_name

```text
Material name for the region outside a footprint-carrying layer's
        DiePrint rectangles: the layer's own gap_material if it declares one
        (a real, distinct material), else the geometry's generic underfill.
        Mangled the same way _get_layer_material_name mangles an overridden
        footprint material, if the gap material's k is overridden.
```

### FunctionDef:_gap_override_key

```text
layer_k_overrides key convention for a layer's gap_material (distinct
        from the layer's own footprint material, which uses the bare layer
        name) -- e.g. 'substrate_organic__gap' overrides geometry7's organic
        substrate field independently of its 'lsi_bridge_via' footprint k.
```

### FunctionDef:_generate_floorplan_files

```text
Generate one 3D-ICE floorplan file per active layer.

        Each file (floorplan_layer{i}.flp) contains only the power blocks that
        reference layer i by name.  TSV-region blocks are excluded — they are
        passive thermal conductors in 3D-ICE (heat generated by TSVs is zero;
        their effect enters via the layer's effective thermal conductivity).

        This replaces the old single-file approach, fixing geometry2 (two active
        die layers with different block sets) and enabling geometry5 (die_zone_1
        and die_zone_2 have completely separate block sets).
```

### FunctionDef:_generate_layout_files

```text
Write one 3D-ICE 4.0 layout file per TSV layer carrying a density field,
        plus one per die layer with DiePrint footprints (silicon under the
        footprint, underfill as the layer's base material elsewhere).

        Format (bison/layout_parser.y): rectangles grouped under a material id.

            tsvmat_tsv_zone_3 :
               rectangle ( x, y, length, width ) ;
```

### FunctionDef:_get_unique_materials

```text
Extract unique materials and their properties from geometry.

        layer_k_overrides: {layer_name: k_override_W_per_mK} — per-scenario TIM
        pump-out or other material k substitutions. Overridden layers get a
        uniquely named material entry so non-overridden layers are unaffected.
        A layer's gap_material (if any) is overridden separately via the
        '{layer_name}__gap' key (see _gap_override_key) -- footprint and gap
        are two different materials on the same layer and must be sweepable
        independently.
```

### FunctionDef:_plan_sublayers

```text
Expand geometry layers into the stack elements actually emitted.

        Returns one dict per element, bottom-to-top:
            layer_idx  index of the originating geometry layer
            thickness  element thickness in um
            z_center   element centre in um (absolute)
            is_active  whether this element carries the power source
```

### FunctionDef:generate_config_files

```text
Generate 3D-ICE configuration files.

        Args:
            geometry: Geometry object
            scenario: Scenario parameters with keys:
                - power_blocks: {block_name: power_density_W/cm2}
                - htc: Heat transfer coefficient (W/m²·K)
                - t_ambient: Ambient temperature (°C)
```

### FunctionDef:parse_results

```text
Parse 3D-ICE Tmap output files.

        3D-ICE Tmap format (uniform grid, one file per stack element):
            % Thermal map for layer <inst> (please find axis information ...)
            T_00  T_01  ...  T_0m
            T_10  T_11  ...  T_1m
            ...

        Rows correspond to x-cells (NRows), columns to y-cells (NColumns).

        Args:
            result_path: Path to output directory (or single output file for
                         backward compatibility with the 4-column text format).

        Returns:
            Dictionary with:
                - 'coords': (N, 3) float32 array of (x, y, z) in µm
                - 'temperature': (N,) float32 array of temperatures in K
```

### FunctionDef:run_simulation

```text
Run 3D-ICE simulation.

        Returns:
            Path to output directory containing per-layer Tmap files.

        Raises:
            RuntimeError: If simulation fails or no output is produced.
```

### module

```text
3D-ICE thermal simulator wrapper.

Generates configuration files, runs 3D-ICE simulations, and parses results.
```

## `src/simulators/lf_simulator.py`

### ClassDef:LowFidelitySimulator

```text
Analytical 1D + 2D-Gaussian thermal simulator.

    Generates NPZ-compatible output dicts (coords, temp, power, layer, metadata)
    using the same coordinate convention as 3D-ICE output.

    Usage::

        sim = LowFidelitySimulator()
        result = sim.simulate(geometry, scenario_params)
        np.savez_compressed('lf_out.npz', **result)
```

### FunctionDef:generate_lf_dataset

```text
Batch-generate low-fidelity NPZ files for all scenarios.

    Each entry in `scenarios` must have keys: geometry_name, htc, t_ambient_celsius,
    power_blocks_wcm2, scenario_name.

    Returns list of output file paths.
```

### FunctionDef:simulate

```text
Run a low-fidelity simulation and return an NPZ-compatible dict.

        Returns keys: coords, temp, power, layer, metadata (same as 3D-ICE path).

        Physical model (correct dimensional analysis)
        ----------------------------------------------
        Per-unit-area 1D heat conduction + Gaussian-smoothed lateral variation:

          T(x,y,z) = T_amb + Q_density(x,y) [W/m²] × R_above(z) [m²K/W]

        where:
          Q_density(x,y) = 2D power flux map (W/m²) with Gaussian smoothing
          R_above(z)      = sum of h_i/k_i for layers above z  +  1/htc  [m²K/W]

        For non-active layers, Q_density = 0 and the local T reflects
        the conducted heat from all active layers above.
```

### module

```text
Low-fidelity analytical thermal simulator for multi-fidelity ARO training.

Generates temperature fields using a 1D thermal resistance ladder (for the
z-profile) combined with 2D Gaussian spreading (for lateral hotspot variation).
No 3D-ICE installation required — runs in milliseconds per scenario.

Physical model
--------------
T(x, y, z) = T_base(z) + T_spread(x, y, z)

T_base(z):  1D solution with layer-stacked thermal resistances.
            Treats the die as a 1D stack: T_bottom = T_amb + Q_total * R_total
            R_total = sum_i(dz_i / k_i / A_xy) + 1/(htc * A_xy)

T_spread(x, y, z): Superposition of 2D Gaussian "influence functions", one per
            power block.  Each block at (xc, yc) with power Q_block produces a
            Gaussian bump whose sigma is proportional to sqrt(R_th * k_eff) — a
            rough estimate of the lateral spreading length.

Accuracy vs 3D-ICE
------------------
Expect ~20-40% RMSE vs real 3D-ICE on geometry1.  This is deliberate — the LF
data provides structural correlation with the HF data without exact agreement,
which is the right regime for multi-fidelity training (ARO pre-trains on LF,
fine-tunes on HF).
```

## `src/visualization/__init__.py`

### module

```text
Visualization utilities for thermal simulations.

Provides functions to create plots of:
- Temperature fields (2D cross-sections, profiles)
- Power distribution maps
- Scenario comparisons
```

## `src/visualization/visualization.py`

### FunctionDef:create_visualization_summary

```text
Create a comprehensive set of visualization plots.

    Args:
        coords: (N, 3) coordinate array
        temperatures: (N,) temperature array
        power_density: (N,) power density array
        geometry: Geometry object
        scenario_name: Scenario identifier
        output_dir: Directory to save plots
```

### FunctionDef:plot_power_map_2d

```text
Create 2D power distribution map (die-level view).

    Args:
        coords: (N, 3) coordinate array in μm
        power_density: (N,) power density array in W/m³
        geometry: Geometry object
        output_file: Optional file path to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
```

### FunctionDef:plot_scenario_comparison

```text
Create comparison plot of multiple scenarios.

    Args:
        scenarios_data: Dict mapping scenario names to temperature arrays
        output_file: Optional file path to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
```

### FunctionDef:plot_temperature_field_2d

```text
Create 2D temperature field visualization (cross-section).

    Args:
        coords: (N, 3) coordinate array in μm
        temperatures: (N,) temperature array in K
        layer_indices: (N,) layer index array
        geometry: Geometry object
        plane: Cross-section plane ('z_mid', 'z_top', 'y_mid', 'x_mid')
        title: Plot title
        output_file: Optional file path to save plot
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
```

### FunctionDef:plot_temperature_profile

```text
Create 1D temperature profile along a direction.

    Args:
        coords: (N, 3) coordinate array in μm
        temperatures: (N,) temperature array in K
        geometry: Geometry object
        direction: Direction for profile ('x', 'y', or 'z')
        output_file: Optional file path to save plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
```

### module

```text
Visualization utilities for thermal simulation data.

Provides functions to create publication-quality plots of:
- Temperature fields (2D slices, 3D volumes)
- Power distribution maps
- Layer cross-sections
- Scenario comparisons
```

## `tests/app/test_job_queue.py`

### module

```text
JobQueue's simulator was previously hardcoded (ICESimulator constructed
directly inside _run_job), so no test could exercise the real job pipeline
-- process_scenario, NPZExporter, StatisticsCalculator -- without a real
3D-ICE/WSL install. tests/app/test_smoke.py works around this by replacing
the entire _run_job method with a hand-rolled fake, which means it never
actually exercises process_scenario or the export/stats path at all.

simulator_factory is a proper injection point: only the expensive 3D-ICE
subprocess is stubbed, everything else (coords generation, power field
regeneration, npz export, statistics) runs for real.
```

## `tests/app/test_validation.py`

### module

```text
Input validation for the app/ API.

register_geometry previously constructed a Geometry() without ever calling
.validate() -- every build_geometryN() function in geometry_builders.py calls
it explicitly, but the API endpoint didn't, so a custom geometry with
duplicate layer names, power blocks outside the die footprint, or a power
block referencing a nonexistent layer would be silently accepted (201) and
only fail confusingly later at job-submission/simulation time. submit_job
similarly accepted any power_blocks dict, so a misspelled block name silently
delivered zero power to the real blocks instead of being rejected.
```

## `tests/test_dataset_integrity.py`

### FunctionDef:test_archive_dirs_are_not_inside_live_data_root

```text
`*_old_*` archives must live outside data/3d-ice/.

    They contain known-corrupted files. Any script globbing data/3d-ice/**/*.npz
    would train on them silently.
```

### module

```text
Dataset integrity checks.

These are the tests that would have caught the three data incidents this project
has already had:

  1. 155 files of synthetic-fallback garbage produced when `--ice-executable`
     was omitted and the error was swallowed.
  2. 13 files cross-contaminated by parallel jobs sharing a temp directory.
  3. Four files with 1600-5000 C temperatures left inside `data/3d-ice/`, where a
     recursive glob would silently ingest them.

They are deliberately cheap (metadata + array statistics only) so they can run on
every commit, and they skip cleanly when no dataset is present.
```

## `tests/test_fno_crossgeom.py`

### FunctionDef:test_physical_extents_distinguish_geometries

```text
Without extent conditioning an 8x8 mm die and a 42x14 mm die resample to
    identical arrays, and the operator would be asked to learn contradictory
    mappings from the same input.
```

### module

```text
Cross-geometry FNO loading.

The point of this benchmark is several geometries with distinct physics, learned
by neural operators. FNO takes an FFT over the spatial dims, so it needs one grid
shape -- but the eight geometries produce five:

    (100, 100,  6)  geometry1, geometry3
    ( 80,  80, 10)  geometry2a/b/c
    (100,  56,  6)  geometry4
    (100,  56, 11)  geometry5
    ( 56, 168, 11)  geometry6

FNODataset(target_grid=...) resamples them onto a shared grid. These tests pin
the two properties that make that safe: the shapes really do unify, and the
geometries remain distinguishable afterwards.
```

## `tests/test_footprint_layout.py`

### module

```text
Die-footprint layout emission for geometry4/5/6 (chiplet-on-interposer).

3D-ICE previously received a single uniform silicon material for the whole
die_zone layer; the underfill gap between chiplets (k=0.7 vs Si's 148 W/m·K)
was only modelled in the PINN's PDE loss, not in the 3D-ICE ground truth --
a real train/target inconsistency (assumptions.md sect 6.1). This makes the
3D-ICE stack file carry the same heterogeneous material via a 3D-ICE 4.0
layout, closing that gap.

A previous version of this code swapped the axis convention (see
_generate_floorplan_files' documented X<->chip_length / Y<->chip_width swap)
and 3D-ICE rejected the layout with "Layout element is outside of the IC" --
caught only by running against the real executable. These tests pin the
swapped-axis contract without needing 3D-ICE installed.
```

## `tests/test_geometry7.py`

### FunctionDef:test_footprint_material_k_override_is_reflected_in_the_layout_file

```text
Regression test: overriding a FOOTPRINT-carrying layer's k (as opposed to
    a full uniform layer's, e.g. tim_top) mangles the material name in the
    stack file (_get_layer_material_name) but the .lyt layout file's rectangle
    group header used the bare unmangled name unconditionally -- 3D-ICE then
    rejected the file with 'Unknown material', a loud failure caught while
    running geometry7's material-uncertainty sweep (2026-08-18), not a silent
    one, but a real bug in the override path all the same.
```

### FunctionDef:test_passive_layer_footprint_layout_is_referenced_in_the_stack_file

```text
Regression test: before the 2026-08-17 fix, the .lyt file for a passive
    layer's DiePrint footprints was written but never referenced in stack.stk,
    so the layer stayed uniformly its own material -- the footprint silently
    had no effect on the simulated field.
```

### FunctionDef:test_solver_block_emits_numofcores

```text
The solver block explicitly states numofcores rather than relying on
    the grammar's implicit default. Default is 1 (deterministic) -- an
    earlier version of this project briefly defaulted to 8 for a wall-clock
    speedup, but multi-threaded SuperLU_MT factorization was found to be
    non-deterministic (repeat runs of the identical scenario differed by
    >1 K; single-threaded repeats were bit-for-bit identical), so it was
    reverted before being used for any real data. See ice_simulator.py's
    solver-block comment for the full correction.
```

### module

```text
geometry7 (CoWoS-L reticle-stitched pilot) structural checks, and the two
`ice_simulator.py` fixes it depends on:

1. Passive layers can carry DiePrint footprints and have them actually take
   effect (before 2026-08-17, the .lyt file was written but never referenced
   in stack.stk -- the layout silently had no effect).
2. A footprint-carrying layer can declare its own gap_material, distinct from
   the geometry's shared underfill_k-based material -- needed so the organic
   substrate field (k=0.5) isn't conflated with die-attach epoxy underfill
   (k=0.7), two different real materials that happen to be similar magnitude.

geometry7 is deliberately excluded from build_all_geometries() (it is a pilot,
not part of the standard 6-geometry benchmark dataset), so it needs its own
tests rather than picking up the ALL_GEOMS parametrization in
test_geometry_and_baselines.py.
```

## `tests/test_geometry_and_baselines.py`

### FunctionDef:_synthetic_scenarios

```text
Build scenarios whose temperature is an EXACT linear function of the block
    powers and ambient -- the regime steady-state conduction actually lives in.
    Ridge must recover this to machine precision.
```

### FunctionDef:test_all_scenarios_stay_inside_the_declared_bc_envelope

```text
Every scenario -- including those drawn from the legacy extra pools -- must
    respect MIN_HTC/MAX_HTC/MAX_AMBIENT_C. The pools are literal tables written
    for the old power levels and contain HTCs down to 500 W/m2K, which combined
    with scaled power produced physically impossible temperatures.
```

### FunctionDef:test_collect_block_keys_prefers_nominal_power_for_throttled_scenarios

```text
Regression test for a real bug: the delivered (post-throttle) power is the
    ALREADY-RESOLVED answer for a throttled scenario, not an input a model
    should get to see. Feeding ridge the delivered power made a throttled
    dataset score a *higher* R^2 (0.989) than the same geometry without
    throttling (0.970) -- the closed feedback loop was invisible to it.
    nominal_block_power_* (the pre-throttle request) is the correct feature.
```

### FunctionDef:test_cooling_capability_matches_power_density

```text
No scenario may pair a high power density with cooling that could not remove
    the heat. Power and HTC are physically coupled -- a 300 W/cm2 hotspot cannot
    be air-cooled -- and sweeping them independently manufactured 220-259 C
    junctions, which are not chips.
```

### FunctionDef:test_power_density_never_exceeds_the_silicon_ceiling

```text
Silicon cannot dissipate arbitrarily much per unit area, whatever the package
    budget. Without this ceiling, normalising to TDP let a concentrated pattern
    pour a whole 125 W budget into one 0.09 cm2 block -- 1618 W/cm2.
```

### FunctionDef:test_total_dissipation_respects_the_tdp_budget

```text
Total power must never exceed TDP x MAX_WORKLOAD_FRACTION. This is what makes
    the operating point citable instead of tuned: it is a design input, not a knob
    chosen to hit a target temperature.
```

### module

```text
Structural checks on geometry definitions, and correctness checks on the
non-neural baselines and the seeding helper.

The baseline tests matter because `scripts/baselines.py` is the yardstick every
surrogate is measured against -- if ridge is subtly wrong, every comparison built
on it is wrong too.
```

## `tests/test_geometry_aware_field.py`

### module

```text
Distance-to-nearest-power-block field: the right-sized geometry-aware
conditioning signal for this benchmark (all 6 geometries are structured
Cartesian grids, not the point-cloud/graph targets GINO/PI-GANO-style full
SDF encoders are built for). Unlike the TSV field, this is purely geometric
-- constant per geometry, no simulation data needed. See goal.md Track B.
```

## `tests/test_geometry_field_fno.py`

### module

```text
End-to-end wiring of the Track B geometry-aware field (distance to nearest
power block) through FNODataset -> FNO3d -> FNOTrainer, on real data. Both
opt-in (use_geometry_field=False by default, so existing checkpoints and
callers are unaffected) and functional (the field is nonzero and varies).
```

## `tests/test_leakage.py`

### module

```text
Leakage-power / temperature positive feedback.

Unlike throttling (negative feedback, self-limiting), leakage feedback is
self-amplifying and above a critical loop gain has no steady state at all.
These tests use a stub simulator whose peak temperature is a deterministic
linear function of total power, so the fixed point -- and the gain at which it
ceases to exist -- can be computed by hand and checked exactly.
```

## `tests/test_microchannel.py`

### FunctionDef:test_geometry_without_coolant_layer_name_raises_loudly

```text
coolant_layer_name=None (the default) means no layer can become a channel.
    Silently falling back would skip 'bottom heat sink' (cooling_mode requested
    it) without ever emitting a 'channel' element either -- an ill-posed
    problem with no heat-rejection boundary at all. Must fail loudly instead.
```

### module

```text
Microchannel liquid cooling: an opt-in `cooling_mode='microchannel_2rm'` scenario
mode that replaces the idealised `bottom heat sink` boundary condition with an
explicitly modelled 3D-ICE 4.0 `microchannel 2rm` coolant layer.

This is the one change in the benchmark that is NOT confined to the linear
conduction regime every other fix operates in (per-cell power, TSV fields,
underfill layouts) -- advection along the coolant flow direction is not linear
in the boundary data the way a fixed HTC scalar is. See goal.md.

Validated against the real 3D-ICE 4.0 binary at geometry6's ~455W core-fraction
TDP ceiling (stable, peak 167.4 C -- see goal.md for the full result and the
honest caveat that the illustrative coolant parameters used there ran hotter
than an air-cooled comparison, i.e. this proves the mechanism works, not that
the chosen parameters are a good cooler). These tests check the .stk emission
mechanics without requiring the real executable.
```

## `tests/test_nominal_power_export.py`

### FunctionDef:test_fno_dataset_uses_nominal_power_when_throttled

```text
End-to-end: FNODataset's Q_norm must be built from the nominal (higher)
    power request, not the delivered (derated) field, for a throttled
    scenario -- otherwise FNO is solving an easier problem than ridge.
```

### module

```text
Nominal (pre-throttle) per-cell power export.

Track A found that `block_power_*` npz metadata holds the DELIVERED power --
already derated for throttled scenarios -- so ridge was being fed the closed
loop's resolved answer, not the request (goal.md Track A2). That was fixed
for the ridge/metadata path via `nominal_block_power_*`
(scripts/baselines.py::collect_block_keys). FNO has the same problem one
level down: its per-cell `power` array is also the delivered field, so a
model trained on it solves an easier "resolved power -> temperature" task
instead of ridge's "requested power -> temperature" task. This tests the
fix: a `power_nominal` array recomputed from `throttle_nominal_power_blocks`
when present (identical to `power` otherwise), consumed by FNODataset in
place of `power`.
```

## `tests/test_orchestrator_exit.py`

### FunctionDef:test_unreachable_simulator_exits_nonzero

```text
A bogus 3D-ICE executable makes every scenario fail. With the synthetic
    fallback off (the default), that must surface as a non-zero exit rather than
    a cheerful summary over an empty directory.
```

### module

```text
The orchestrator must not report success over empty or partial output.

Per-scenario failures were logged and then swallowed: a run in which every
scenario raised still printed "PROCESSING COMPLETE" and returned 0. A batch
script driving this saw success while producing no files, which is how a broken
run can be mistaken for a finished dataset.
```

## `tests/test_physics.py`

### ClassDef:AnalyticField

```text
Model stub returning a prescribed closed-form normalised temperature.

    `fn` maps normalised coords (N,3) -> normalised temperature (N,), so the
    exact derivatives are known and the residual can be checked analytically.
```

### FunctionDef:test_quadratic_field_matches_analytic_divergence

```text
Manufactured solution: T_hat = c * z_hat^2, constant k, no source.

    In physical units T(z) = c * T_range * (z/L_z)^2 + T_min, so

        d2T/dz2 = 2 * c * T_range / L_z^2      [K/um^2]
        div(k grad T) = k * 2 * c * T_range / L_z^2

    The residual must equal that. This is the test that pins the chain rule:
    an extra or missing factor of T_range (or of L_z) shows up here and nowhere
    in the zero-residual tests above.
```

### module

```text
Analytic checks on the PINN physics kernel.

The PDE residual is the one piece of this codebase with no independent oracle --
if its chain rule or unit conversion is wrong, training silently optimises the
wrong objective and every downstream number is invalid. These tests pin it
against closed-form solutions on manufactured temperature fields.
```

## `tests/test_placement.py`

### module

```text
Tests for src/core/placement.py — per-scenario chiplet placement (docs/report.md §9.14-9.15).

The behaviours worth pinning down are the ones that were wrong on the first attempt:
stacked footprints must move as one chiplet, blocks must stay registered with the die they
sit on, and nothing may leave the package footprint.
```

## `tests/test_power_map_footprint_cap.py`

### module

```text
Regression guard for a real 3D-ICE 4.0 crash: a die's source layer emitted as
an IDENTIFIER-referenced layout (the Si/underfill footprint mechanism in
ice_simulator.py) combined with a full-mesh-resolution per-cell floorplan
(~5600 elements on geometry5) heap-corrupts the binary ("corrupted size vs.
prev_size", return code 6). resolution=64 was validated crash-free against
the real executable; full mesh resolution was not, on geometry5.

ScenarioGenerator.attach_power_maps must cap the map resolution to 64 whenever
the geometry carries die_footprints and the caller left resolution at its
"use full mesh" default (0). An explicit non-zero resolution request is left
alone -- the cap only substitutes for an unset default.
```

## `tests/test_power_maps.py`

### FunctionDef:test_map_collection_is_high_dimensional

```text
The whole point. A collection of N maps must span close to N dimensions --
    every scenario carrying new information -- rather than collapsing onto the
    handful of degrees of freedom that block scalars provide.
```

### FunctionDef:test_power_field_uses_the_map_not_the_blocks

```text
The exported power field must come from the map the simulator actually used.
    Rebuilding it from block means would store a coarse approximation of the
    source that produced the temperatures -- a silent train/target mismatch.
```

### module

```text
Per-cell power maps.

Block-scalar power spans exactly ~4 dimensions no matter how many scenarios are
generated, which is why closed-form ridge regression solved the old dataset
(docs/report.md 9.3). A power map makes the source a function instead, so the
solution operator becomes a genuine Green's function rather than a
low-dimensional linear map.

These tests pin the two properties that make the change worth its cost: the maps
really are high-dimensional, and swapping them in does not break the physical
constraints (TDP budget, silicon density ceiling, cooling adequacy).
```

## `tests/test_throttle_cli.py`

### module

```text
--throttle CLI wiring: ScenarioGenerator.attach_throttling itself is already
covered by tests/test_throttling.py's convergence-math tests; this checks the
flag actually reaches scenario_params and survives export, using --simulator
mock so it runs fast without a real 3D-ICE install. Mock mode never calls
apply_throttling (that only runs on the real-simulator path), so this proves
CLI plumbing, not the derate loop itself.
```

## `tests/test_throttling.py`

### module

```text
Package-level thermal throttling (DVFS): the first mechanism in this
benchmark where power is a function of the temperature field being solved
for, rather than a fixed scenario input. Real chips do this (Tjmax-triggered
power derating); every other change so far (per-cell power, TSV fields,
underfill layouts, microchannel cooling at fixed flow) keeps the source
fixed per scenario.

These tests use a stub simulator (no real 3D-ICE needed) whose peak
temperature is a deterministic function of total requested power, so the
derate/convergence math can be checked exactly.
```

## `tests/test_tsv_field_export.py`

### module

```text
The spatial TSV-density field affected the 3D-ICE ground truth (since
2026-08-05) but was never exported to .npz or exposed to any model as an
input -- only a per-geometry constant scalar (`metadata['tsv_density']`)
reached PINN/FNO/DeepONet/ARO. That made the field a hidden confounder: it
added real variance to the temperature target with no input a model could
condition on. This tests the fix: a real per-point `tsv_frac` array is now
exported, and `src/fno/model.py` accepts it as a genuine per-cell channel
(with a scalar-broadcast fallback for files/callers predating this field).
```

## `tests/test_tsv_maps.py`

### FunctionDef:test_every_tsv_geometry_declares_a_nonzero_scalar

```text
Regression test for a real bug found by auditing the regenerated dataset:
    build_geometry5/6 built their tsv_zone material with create_tsv_material(0.03)
    but never passed tsv_density= to the Geometry() constructor, so
    geometry.tsv_density silently stayed at its dataclass default of 0.0.
    ScenarioGenerator.attach_tsv_maps guards on `geometry.tsv_density <= 0` and
    no-ops when it's zero, so the spatial TSV-density field was silently never
    attached for geometry5/6 -- every scenario used one uniform TSV material
    instead, with no field variation at all, for a full regeneration cycle.
```

### FunctionDef:test_vertical_conductivity_exceeds_lateral

```text
A TSV is a copper cylinder: heat runs ALONG it (parallel, arithmetic mean) but
    must CROSS phases laterally (series, harmonic mean). Vertical must therefore
    dominate, and both must sit between the bulk values.
```

### module

```text
Spatially varying TSV density and anisotropic effective conductivity.

TSV density was one scalar with four discrete values, so it could not support an
interpolation claim. As a field it becomes a per-scenario spatial coefficient --
the part of the solution operator that is genuinely nonlinear.
```
