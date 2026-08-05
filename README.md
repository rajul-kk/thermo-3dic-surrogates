# Thermo — 3D-IC Thermal Surrogate Benchmark

A benchmark suite for neural thermal surrogates in 3D/2.5D IC packaging. Includes **8 realistic geometries** (single-die, 3%/5%/10% TSV-density 3D stacks, server die, chiplet-on-interposer, CoWoS+HBM, 6×HBM MI300X-like) with real [3D-ICE](https://www.epfl.ch/labs/esl/research/open-source-tools-datasets/3d-ice/) finite-element ground truth — not synthetic data — and implements/compares **five surrogate model families**: physics-informed neural networks (with pluggable adaptive collocation sampling), Fourier/Walsh-Hadamard/CNO neural operators, physics-informed DeepONet, an autoregressive z-layer operator, and few-shot fine-tuning across geometries. Includes zero-retraining explainability tooling for every model family.

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
| `geometry1` | Single die | 10 × 10 mm | 6 | — | 100×100×40 | 60,000 |
| `geometry2a` | 3D TSV stack | 8 × 8 mm | 10 | 3% | 80×80×72 | 64,000 |
| `geometry2b` | 3D TSV stack | 8 × 8 mm | 10 | 5% | 80×80×72 | 64,000 |
| `geometry2c` | 3D TSV stack | 8 × 8 mm | 10 | 10% | 80×80×72 | 64,000 |
| `geometry3` | Server die | 25 × 25 mm | 6 | — | 100×100×40 | 60,000 |
| `geometry4` | 2.5D chiplet-on-interposer | 25 × 14 mm | 6 | — | 100×56×40 | 33,600 |
| `geometry5` | CoWoS-style compute + HBM stack | 25 × 14 mm | 11 | 3% | 100×56×50 | 61,600 |
| `geometry6` | CoWoS + 6× HBM (MI300X-like) | 42 × 14 mm | 11 | 3% | 56×168×50 | 103,488 |

Convective (HTC) boundary cooling is applied at `z = 0` (the `heat_sink` layer), matching 3D-ICE's ground-truth `bottom heat sink` directive. Full layer stacks, material properties, and scenario details are in [`docs/geometry_reference.md`](docs/geometry_reference.md).

**Dataset**: 335 real 3D-ICE 4.0 `.npz` files across all 8 geometries, per-cell power maps
on every scenario. Power is derived from a package TDP budget (30 W mobile 3D stack →
700 W six-HBM accelerator), of which the modelled blocks receive 65% — the balance
representing cache, IO and uncore. A scenario's pattern selects a workload fraction of
that budget, bounded by an absolute silicon ceiling of 300 W/cm² and, for HBM/memory
dies, 8 W/cm². Cooling must be adequate for the power density (165 W/m²·K per W/cm²),
giving HTC 2000–50,000 W/m²·K; ambient spans 25–45 °C. TSV density is a spatial field
(not a scalar) and geometry4/5/6's chiplet underfill gap is a real Si/underfill layout,
both effective as of the 2026-08-05 3D-ICE 4.0 regeneration — see `docs/assumptions.md`.

**Dataset layout.** `data/3d-ice/` is always the current dataset — every command below
assumes it. `data/_archive/` holds superseded generations and must never be globbed into
training: `3d-ice_pre_tdp_regime/` is the spatially degenerate dataset (median ΔT 0.77 K
vs 29.05 K now), kept only for reproducing §9.1–9.2 of the report; `3d-ice_pre_4_0/` is
the last 3D-ICE 3.0.0 generation, superseded 2026-08-05.

> **Before training anything, run `scripts/baselines.py`.** Closed-form ridge regression
> solves this benchmark at spatial R² ≈ 0.99 and extrapolates at R² > 0.94, because
> steady-state conduction is linear in its sources and block-scalar power spans only ~4
> dimensions. Judge surrogates on **hotspot localisation and spatially-detrended error**,
> not raw MAE or field R² — both are dominated by a linear component that needs no network.
> Passing `--power-map` makes power a per-cell field instead, which is what breaks the
> linear model's hotspot localisation (8–1442 µm → 4547–6044 µm). See
> [`docs/report.md`](docs/report.md) §9.

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

# PI-DeepONet (one model across uniform-stack geometries: g1/2a/2b/2c/g3)
python scripts/train_deeponet.py --data data/3d-ice --output checkpoints/deeponet

# ARO (autoregressive z-layer operator)
python scripts/train_aro.py --hf-data data/3d-ice --geometries geometry1 --output checkpoints/aro

# Therm-FM (few-shot fine-tune a pretrained CNO-FNO on a new geometry)
python scripts/finetune_therm_fm.py --pretrained checkpoints/fno/geometry1_best.pt \
    --new-data data/3d-ice --geometry geometry3 --shots 10 --output checkpoints/therm_fm
```

CPU-only: add `--cpu-fast` to `train_pinn.py`/`train_fno.py` for reduced-capacity defaults. See [`docs/compute.md`](docs/compute.md) for measured/estimated training costs across all model families on CPU and 2×T4 GPU.

### 4. Evaluate and explain

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
| **PI-DeepONet** | 538k | Yes (g1/2a/2b/2c/g3) | Branch (MLP or CNO-FNO spatial encoder) + Fourier-encoded trunk; scoped to uniform-stack geometries where the trunk's `(x,y,z,layer_id)` coordinate system is physically valid. |
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
