# Thermo — 3D-IC Thermal Surrogate Benchmark and Audit

An open benchmark of real [3D-ICE](https://www.epfl.ch/labs/esl/research/open-source-tools-datasets/3d-ice/) 4.0
solves for 3D/2.5D IC packages, plus an audit of what neural thermal surrogates actually learn on it.

- **Geometries:** 6 benchmark geometries (single die, TSV 3D stack, server die, chiplet-on-interposer, CoWoS + HBM,
  6×HBM CoWoS).
- **Surrogates:** five families (PINN, FNO/WHNO/CNO-FNO, PI-DeepONet, an autoregressive z-layer operator, few-shot
  Therm-FM), and ThermoNO.
- **Classical baselines:** a training-free layered solver and backbone-preconditioned CG.
- **Explainability:** shared XAI tooling.

## Main findings (details and every number in [`docs/report.md`](docs/report.md) §9)

1. **The fixed-placement benchmark is linearly solvable.** Closed-form ridge regression on the block-power vector
   reaches spatial R² 0.93–0.98 (v5 data, `results/v5/baselines_*.json`). Steady conduction is linear in power.
2. **Randomising the chiplet layout breaks the linear baseline** (§9.15b). Ridge becomes the worst of four baselines;
   on geometry6 its median R² is negative. Whether data is "linearly solvable" depends on the input representation:
   the same files give R² 0.95 from the per-cell power field and −0.67 from the block vector (§9.15c).
3. **The bar is a classical solver, not ridge** (§9.25, §9.30). A training-free layered DCT solver (about 10–25 ms)
   matches 3D-ICE's hotspot to 0.06–0.15 K peak error and 0–250 µm location on the chiplet geometries. Preconditioned
   with it, CG reaches the exact finite-volume solve in 20–50 iterations.
4. **A learned model beat physics by learning a simulator artefact** (§9.32).
   - 3D-ICE heats a cell that a die edge only partly covers, but gives it the gap material. The continuous placement
     offsets created such cells in the layout data, with spurious peaks up to 14 K.
   - ThermoNO's apparent 2.6–5.4× hotspot advantage over the backbone disappeared on the corrected data.
   - The layout data was regenerated on a grid-aligned placement on 2026-09-30.
   - A paired label-swap test confirms it: ThermoNO's held-out predictions reproduce about 80–95% of the artefact's
     amplitude (β 0.77–0.95, all 3 seeds, same-label nulls near 0, 5 of 5 folds). The FNO family is inconclusive: it
     shows a partial reproduction on geometry4/6 (β 0.35 and 0.26, CI above zero) but misses the pre-registered
     "learned" bar, and its seed-to-seed noise is too large for the test to resolve a one-cell edge effect (§9.32).
5. **The dataset linter's first external catch** (§9.33): the public IC-ThermBench release stores its spatial input
   channels transposed in-plane relative to the output temperature; our loader mirrored it, so our IC-ThermBench numbers
   came from mismatched pairs. Ridge-type baselines change by ≤ 0.34% when fixed. Fixed-orientation runs at 108 samples
   are in §9.31, and the full-data §9.29 numbers are not yet rerun. Draft issue reports are in `notes/` (not sent).
6. **Every data bug was caught by an independent physics re-solve, not by a metric.** These were a transposed
   temperature field, a layer-blind power field, the die-edge artefact, and others (§9.23, §9.32).

This extends the "weak baselines" critique of ML-for-PDE work
([McGreivy & Hakim, Nat. Mach. Intell. 2024](https://arxiv.org/abs/2407.07218)) to 3D-IC thermal surrogates. It adds
three mechanisms: linear solvability, representation dependence and simulator-artefact learning.

**Geometry count note (2026-08-06):** originally 8 geometries — `geometry2b`/`geometry2c`
(5%/10% TSV-density variants of `geometry2a`) were removed. Their ridge-regression
baselines were bit-identical to `geometry2a`'s (spatial R²=0.991, MAE=2.207 K, all three
to 3 decimals) and TSV density isn't yet exposed as a model input anywhere in the
pipeline, so they were three near-zero-marginal-information copies of one benchmark. TSV
density variation is still exercised as a spatial field within `geometry2a` itself via
`ScenarioGenerator.attach_tsv_maps`. See `goal.md`.

**Cross-geometry generalization scope note (2026-08-09):** "few-shot fine-tuning across
geometries" above means *interpolation among this repo's 6 trained geometries* via
`--common-grid` resampling (trilinear onto a shared grid + a single `geom_extent_norm`
scalar as conditioning), not zero-shot transfer to an unseen package shape. Current
literature's geometry-aware architectures (signed-distance-function or graph-based
geometry encoding — e.g. GINO, PI-GANO) report <3% error on genuinely unseen geometries;
this repo's mechanism is a coarser approximation and hasn't been validated with a
leave-one-geometry-out test. See `goal.md` Track B for the plan to close this gap.

---

## Project Structure

```
Thermo/
├── src/
│   ├── core/               # Geometry, material, mesh
│   ├── simulators/         # 3D-ICE, HotSpot, and low-fidelity analytical simulators
│   ├── scenario/           # Scenario parameter sweep generator
│   ├── export/             # NPZ exporter, statistics
│   ├── visualization/      # Temperature field plots
│   ├── pinn/                # FourierPINN: model, physics, losses, sampling
│   │                        # strategies, trainer, data loader, evaluate, explain
│   ├── fno/                 # FNO3d, CondFNO3d, CNOFNOHybrid, WHNO (Walsh-Hadamard),
│   │                        # physics loss, FNO+PINN hybrid, mode-importance XAI
│   ├── deeponet/             # PI-DeepONet (MLP and CNO-FNO branch variants)
│   ├── aro/                  # Autoregressive z-layer operator with RNO-style training
│   ├── hybrid/               # Layered DCT backbone solver, learned-conductance solver
│   ├── validation/           # Independent finite-volume solver (checks 3D-ICE)
│   └── main.py               # End-to-end data generation orchestrator
├── scripts/
│   ├── train_pinn.py / eval_pinn.py / explain_pinn.py
│   ├── train_fno.py / explain_fno.py
│   ├── train_deeponet.py
│   ├── train_aro.py
│   ├── finetune_therm_fm.py / explain_therm_fm.py   # few-shot cross-geometry fine-tuning
│   ├── generate_lf_data.py                            # low-fidelity dataset generation
│   └── validate_dataset.py / regenerate_failed.py / test_ice_connection.py
├── notebooks/               # Kaggle GPU training + model-comparison notebooks
├── app/                     # FastAPI web app for interactive scenario submission
├── configs/                 # 3D-ICE / HotSpot config files, scenario YAMLs
├── data/                    # .npz training data (generated separately, gitignored)
├── docs/                    # Reference documentation (see below)
└── requirements.txt
```

---

## Geometries

| Geometry | Type | Footprint | Layers | TSV density | Mesh | Points/file |
|---|---|---|---|---|---|---|
| `geometry1` | Single die | 10 × 10 mm | 6 | — | 100×100×40 | 100,000 |
| `geometry2a` | 3D TSV stack | 8 × 8 mm | 10 | 3% (field) | 80×80×72 | 89,600 |
| `geometry3` | Server die | 25 × 25 mm | 6 | — | 100×100×40 | 110,000 |
| `geometry4` | 2.5D chiplet-on-interposer | 25 × 14 mm | 6 | — | 100×56×40 | 56,000 |
| `geometry5` | CoWoS-style compute + HBM stack | 25 × 14 mm | 11 | 3% | 100×56×50 | 84,000 |
| `geometry6` | CoWoS + 6× HBM (generic, reduced scale) | 42 × 14 mm | 11 | 3% | 56×168×50 | 141,120 |

These are plausible but **reduced-scale** packages, not models of specific products: e.g. AMD MI300X has 8 HBM3 stacks and NVIDIA Rubin two near-reticle compute dies with 8 HBM4 stacks on a ~70 × 76 mm package at ~1.8–2.3 kW. Realism gaps are listed in [`docs/assumptions.md`](docs/assumptions.md).

Convective (HTC) boundary cooling is applied at `z = 0` (the `heat_sink` layer), matching 3D-ICE's ground-truth `bottom heat sink` directive. Full layer stacks, material properties, and scenario details are in [`docs/geometry_reference.md`](docs/geometry_reference.md).

**Dataset:**
- **Fixed placement:** 275 real 3D-ICE 4.0 `.npz` files across all 6 geometries (`data/3d-ice/`).
- **Layout-randomised:** 270 more files, 45 per geometry (`data/3d-ice-layout-{geometry}/`). The geometry4–6 sets
  were re-solved on grid-aligned placements on 2026-09-30 (metadata flag `placement_snapped_2026_09_30`).
- **Pilot datasets:** leakage, throttling, interface, microchannel, moving-layout and geometry7, in
  `data/3d-ice-*-pilot/` and similar directories; see `docs/report.md` §9.24.
- Every scenario has a per-cell power map.

Scenario parameters: Power is derived from a package TDP budget (30 W mobile 3D stack →
700 W six-HBM accelerator), of which the modelled blocks receive 65% — the balance
representing cache, IO and uncore. A scenario's pattern selects a workload fraction of
that budget, bounded by an absolute silicon ceiling of 300 W/cm² and, for HBM/memory
dies, 8 W/cm². Cooling must be adequate for the power density (165 W/m²·K per W/cm²),
giving HTC 2000–50,000 W/m²·K; ambient spans 25–45 °C. TSV density is a spatial field
(not a scalar) and geometry4/5/6's chiplet underfill gap is a real Si/underfill layout,
both effective as of the 2026-08-05 3D-ICE 4.0 regeneration — see `docs/assumptions.md`.

**Dataset layout.**
- `data/3d-ice/` (fixed placement) and `data/3d-ice-layout-{geometry}/` (layout-randomised) are the current datasets.
- `data/_archive*/` directories hold superseded generations. Never glob them into training.
  - `_archive_v4_pre_v5_20260924/` is from before the §9.23 transpose/power-field repair.
  - `_archive_pre_snap_20260930/` holds the geometry4–6 layout data with the die-edge artefact.
- `data/3d-ice-layout-geometry7/` is an unused, unrepaired duplicate (still transposed); do not use it.

> **Before training anything, run the baselines.**
> - `scripts/baselines.py`: ridge solves the fixed-placement data at spatial R² 0.93–0.98, because conduction is
>   linear in its sources.
> - `scripts/layout_cv.py`: 5-fold CV on the layout data; single splits were overturned twice.
> - `scripts/backbone_eval.py`: the training-free layered solver is the bar any surrogate must clear.
>
> Judge surrogates on **hotspot localisation, peak-temperature error and spatially detrended error**. Raw MAE and
> field R² are dominated by a linear component that needs no network.
>
> After any change to data generation, also validate the data with an independent physics re-solve. Energy balance
> alone could not see the die-edge artefact (`scripts/validate_3dice.py`, `warpage/scripts/validate_snap.py`).
> See [`docs/report.md`](docs/report.md) §9.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch, NumPy, SciPy, PyYAML, Matplotlib. See [`docs/installation.md`](docs/installation.md) for 3D-ICE/WSL2 setup.

### 2. Generate training data (requires 3D-ICE, WSL2 on Windows)

```bash
python src/main.py --all-geometries --simulator 3d-ice \
    --ice-executable "wsl /home/user/3d-ice/bin/3D-ICE-Emulator" \
    --output data/3d-ice --verbose
```

For development without 3D-ICE (synthetic data):
```bash
python src/main.py --all-geometries --simulator mock --output data/3d-ice-mock
```

For a fast, dependency-free low-fidelity dataset (analytical 1D-resistance + 2D-Gaussian model, used for ARO multi-fidelity pretraining):
```bash
python scripts/generate_lf_data.py --output data/lf
```

### 3. Train a model

```bash
# PINN (curriculum-staged, adaptive collocation sampling)
python scripts/train_pinn.py --geometry geometry1 --data data/3d-ice --output checkpoints

# CNO-FNO (recommended FNO variant: local conv + spectral FNO + FiLM conditioning)
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --output checkpoints/fno

# WHNO (Walsh-Hadamard basis — no Gibbs ringing at material-conductivity discontinuities)
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --model whno

# PI-DeepONet (one model across uniform-stack geometries: g1/2a/g3)
python scripts/train_deeponet.py --data data/3d-ice --output checkpoints/deeponet

# ARO (autoregressive z-layer operator)
python scripts/train_aro.py --hf-data data/3d-ice --geometries geometry1 --output checkpoints/aro

# Therm-FM (few-shot fine-tune a pretrained CNO-FNO on a new geometry)
python scripts/finetune_therm_fm.py --pretrained checkpoints/fno/geometry1_best.pt \
    --new-data data/3d-ice --geometry geometry3 --shots 10 --output checkpoints/therm_fm
```

CPU-only: add `--cpu-fast` to `train_pinn.py`/`train_fno.py` for reduced-capacity defaults. See [`docs/compute.md`](docs/compute.md) for measured/estimated training costs across all model families on CPU and 2×T4 GPU.

### 4. Use the tools built on the findings

**Interactive thermal floorplanner.** Drag chiplets around a package and watch the temperature field update.

```bash
uvicorn app.main:app --port 8000      # then open http://localhost:8000/floorplanner
```

It uses no trained model. Three tiers answer at three speeds:

| Tier | Method | Time | Peak error vs 3D-ICE (geometry4–6) |
|---|---|---|---|
| Preview, while dragging | layered DCT × tridiagonal solver | 10–25 ms | 0.06–0.25 K |
| Exact, on release | CG on the finite-volume system, preconditioned by the preview solver | 1–3 s | ≤ 0.03 K |
| Sign-off, on demand | a real 3D-ICE run | 5–60 s | reference |

- **Placement:** every position snaps to the 3D-ICE cell grid, so the die-edge artefact (§9.32) cannot occur. Overlaps
  and out-of-package positions are rejected with a reason.
- **Optimise placement:** tries a few hundred placements with the preview solver to lower the peak.
- **Export:** writes the 3D-ICE input files for the current placement.
- **Limit:** the preview is unreliable when a layer between the sources and the sink has strong in-plane conductivity
  contrast (geometry7's bridge layer). The page says so and the exact tier still applies.
- **Code:** engine in `src/solver/thermal.py`, API in `app/main.py` (`/floorplan`, `/solve`, `/optimise`,
  `/export/3dice`, `/signoff`), page in `app/static/floorplanner.*`.

**Dataset linter.** Checks `.npz` datasets for the silent bugs this project hit (§9.23, §9.32).

```bash
python scripts/lint_dataset.py data/3d-ice data/3d-ice-layout-geometry5      # exit status 1 on any error
```

| Rule | Catches |
|---|---|
| L003 orientation | transposed temperature grids |
| L005 maximum principle | the hottest node is unpowered: temperature, power and material fields disagree (die-edge artefact) |
| L006 off-grid footprint | a chiplet edge inside a 3D-ICE cell |
| L007 metadata | missing keys; `placement_dx_*` keys that name no die |
| L008 energy balance | heat out ≠ power in |
| L001/2/4/9/10/11 | bad arrays, incomplete grids, unphysical temperatures, k overrides, duplicate fields, archive paths |

The 545 benchmark files pass with no errors or warnings. The archived pre-fix layout data fails L005 in 28 / 36 / 34
of 45 files per geometry. Rules are in `src/validation/lint.py`.

**Tools.** Two standalone command-line tools package the checks behind findings 4 and 5. Neither imports from this
repository.

- **iceforge** (`tools/iceforge/`): lints 3D-ICE inputs before a solve, snaps edges to the cell grid, runs 3D-ICE
  (native, WSL or docker), and `diff` compares it with an independent finite-volume reference. On 24 archived
  real geometry4–6 solves, `diff` flags 23 / 24 pre-snap cases (DISAGREE) and passes 24 / 24 snapped ones; the one
  miss is a 0.04 K case that `iceforge check` catches statically (§9.34).
- **fieldlint** (`tools/fieldlint/`): lints steady-diffusion PDE datasets (orientation, maximum principle,
  duplicates, units, energy balance, operator residual, sample pairing). Its frozen thresholds failed on the held-out
  PDEBench Darcy set; a second round is in progress (§9.34).

```bash
pip install -e "tools/iceforge[diff]" && iceforge check model.stk      # then: iceforge snap / run / diff model.stk
pip install -e tools/fieldlint && fieldlint path/to/data --preset thermfm   # presets: ictherm, thermfm, 3dice
```

See `tools/iceforge/README.md` and `tools/fieldlint/README.md`; results in report §9.34.

### 5. Evaluate and explain

```bash
python scripts/eval_pinn.py --checkpoint checkpoints/geometry1/geometry1_best.pt \
    --data data/3d-ice --output results/geometry1 --plots

python scripts/explain_pinn.py --checkpoint checkpoints/geometry1/geometry1_best.pt \
    --data data/3d-ice --output results/explain/geometry1

python scripts/explain_fno.py --model-a checkpoints/fno/geometry1_best.pt --model-a-type cno-fno \
    --model-b checkpoints/fno/geometry1_whno_best.pt --model-b-type whno \
    --geometry geometry1 --output results/explain/fno_comparison
```

---

## Model Families

| Model | Params | Cross-geometry? | Notes |
|---|---|---|---|
| **FourierPINN** | ~807k | No | Fourier + layer embedding, 6 residual MLP blocks, hard adiabatic BC via cosine coordinate fold, optional region embedding (2.5D chiplets) and TIM-k scalar input. Four pluggable collocation-sampling strategies (`rar`, `hessian`, `importance`, `curriculum`). |
| **FNO3d / CondFNO3d / CNOFNOHybrid** | 1.6M–12.6M | No (per grid) | CNOFNOHybrid combines a CNN encoder/decoder with a FiLM-conditioned latent FNO, optionally with axial self-attention (SAU-FNO style). Physics loss uses harmonic-mean face conductivity (accurate across the large conductivity contrasts at material interfaces) plus an interface-isolated flux-continuity term. |
| **WHNO** | ~similar to FNO | No (per grid) | Walsh-Hadamard spectral basis instead of Fourier — piecewise-constant basis functions produce zero Gibbs ringing at sharp material-conductivity discontinuities (validated against a synthetic step function: 0% overshoot vs Fourier's ~8.7%). |
| **PI-DeepONet** | 538k | Yes (g1/2a/g3) | Branch (MLP or CNO-FNO spatial encoder) + Fourier-encoded trunk; scoped to uniform-stack geometries where the trunk's `(x,y,z,layer_id)` coordinate system is physically valid. |
| **ARO** | ~1M | No (per grid) | Shared 2D FiLM-FNO block applied autoregressively over z-layers (O(nx·ny) memory instead of O(nx·ny·nz)). Trained with RNO-style windowed self-rollout — the model predicts a growing window of consecutive layers using only its own prior outputs, closing the exposure-bias gap between teacher-forced training and autoregressive inference. Supports multi-fidelity pretraining on the low-fidelity analytical dataset. |
| **Therm-FM** | ~10% of CNOFNOHybrid | Few-shot | Freezes the CNN encoder, fine-tunes the FiLM generator + tail latent-FNO blocks + decoder on 5–20 shots of a new geometry. |

Compute costs, parameter counts, and architecture comparisons across all families: [`docs/compute.md`](docs/compute.md).

---

## Explainability

All XAI tooling is zero-retraining — run on any trained checkpoint without modifying the training pipeline.

| Method | Model(s) | What it shows |
|---|---|---|
| PDE residual map | PINN, FNO/WHNO | Where the heat equation is locally violated |
| Power block / HTC sensitivity | PINN | dT/dQ, dT/dHTC — thermal influence coefficients |
| Integrated Gradients | PINN | Feature attribution at the hotspot + z-profile |
| MC Dropout uncertainty | PINN | Predictive std — where the model is uncertain (also usable for sensor-placement guidance) |
| Spectral mode-importance | FNO, WHNO | Per-mode energy read directly off trained weights (no forward pass); compares Fourier's frequency-ordered vs WHNO's sequency-ordered basis on the same low-to-high-mode-index axis |
| Interface-distance-bucketed error | FNO, WHNO | Whether accuracy differences concentrate near material interfaces |
| Weight-drift analysis | Therm-FM | Diffs pretrained vs fine-tuned checkpoints by parameter group; confirms frozen layers show exactly zero drift |

---

## Key Design Decisions and Assumptions

See [`docs/assumptions.md`](docs/assumptions.md) for the complete list, including known modeling limitations (steady-state only, uniform HTC, TSV effective-medium approximation, and the 2.5D lateral-material-heterogeneity gap between the Python geometry model and the actual 3D-ICE ground truth).

---

## References

| Reference | Used for |
|---|---|
| Sridhar et al., ICCAD 2010 | 3D-ICE simulator |
| Huang et al., IEEE TVLSI 2006 | HotSpot simulator |
| Glassbrenner & Slack, Phys Rev 1964 | Silicon k(T) model |
| Raissi et al., JCP 2019 | PINN foundation |
| Tancik et al., NeurIPS 2020 | Fourier feature encoding |
| Wang et al., SIAM J. Sci. Comput. 2021 | NTK adaptive loss weights |
| Li et al., ICLR 2021 | Fourier Neural Operator |
| Sundararajan et al., ICML 2017 | Integrated Gradients |
| Yang et al. 2025 (Recurrent Neural Operators) | ARO's windowed self-rollout training |

Full citations, including geometry-specific literature cross-checks and recent (2025) adaptive-sampling / neural-operator prior art, in [`docs/references.md`](docs/references.md).

---

## Notebooks

Kaggle GPU training and model-comparison notebooks — see [`notebooks/KAGGLE_SETUP.md`](notebooks/KAGGLE_SETUP.md) for dataset upload instructions.

| Notebook | What it does |
|---|---|
| `kaggle_pinn_geometry1.ipynb` / `kaggle_pinn_geometry2.ipynb` | Standalone PINN training per geometry, with XAI (residual map, power/HTC sensitivity, integrated gradients, MC dropout uncertainty) |
| `kaggle_pinn_sampling_comparison.ipynb` | Head-to-head comparison of two collocation-sampling strategies on identical architecture/seed |
| `kaggle_sau_cnofno_vs_whno.ipynb` | CNO-FNO (axial attention) vs WHNO, with mode-importance XAI and interface-distance error analysis |
| `kaggle_therm_fm.ipynb` | Pretrain-then-few-shot-fine-tune across geometries, with weight-drift analysis |
