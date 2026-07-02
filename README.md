# FourierPINN — 3D-IC Thermal Surrogate

A Physics-Informed Neural Network surrogate for steady-state thermal analysis of 3D integrated circuit package stacks. Trained on [3D-ICE](https://www.epfl.ch/labs/esl/research/open-source-tools-datasets/3d-ice/) simulation data across five benchmark geometries. Includes post-training explainability: PDE residual maps, engineering sensitivity maps, targeted Integrated Gradients, and MC Dropout uncertainty.

---

## Project Structure

```
Thermo/
├── src/
│   ├── core/               # Geometry, material, mesh
│   ├── pinn/               # FourierPINN model, trainer, data loader,
│   │                       # physics losses, evaluation, explainability
│   ├── fno/                # FNO3d model, trainer, data loader, hybrid
│   ├── simulators/         # 3D-ICE and HotSpot wrappers
│   ├── scenario/           # Scenario parameter sweep generator
│   ├── export/             # NPZ exporter, statistics
│   ├── visualization/      # Temperature field plots
│   └── main.py             # End-to-end data generation orchestrator
├── scripts/
│   ├── train_pinn.py       # PINN training CLI
│   ├── train_fno.py        # FNO training CLI
│   ├── eval_pinn.py        # Post-training evaluation
│   └── explain_pinn.py     # Explainability analysis CLI
├── notebooks/
│   ├── kaggle_pinn_geometry1.ipynb   # Kaggle GPU training (geometry1)
│   └── kaggle_pinn_geometry2.ipynb   # Kaggle GPU training (geometry2a/b/c)
├── configs/                # 3D-ICE and HotSpot config files, scenario YAMLs
├── data/                   # .npz training data (generated separately)
├── docs/
│   ├── report.md           # Paper draft
│   ├── assumptions.md      # Simplifying assumptions vs. real hardware
│   ├── references.md       # Literature cross-check and citations
│   ├── installation.md     # 3D-ICE / HotSpot setup guide
│   └── geometry_reference.md
└── requirements.txt
```

---

## Geometries

All geometries include a TIM2 layer (50 µm, k=4 W/m·K) as the topmost layer, placing the convective boundary condition at the physically correct die-to-cooler interface.

| Geometry | Type | Die size | Layers | Mesh | Training points |
|---|---|---|---|---|---|
| geometry1 | Single die | 10 × 10 mm | 6 | 100×100×40 | 400k |
| geometry2a | 3D stack | 8 × 8 mm | 10 | 80×80×72 | 461k |
| geometry2b | 3D stack | 8 × 8 mm | 10 | 80×80×72 | 461k |
| geometry2c | 3D stack | 8 × 8 mm | 10 | 80×80×72 | 461k |
| geometry3 | Server die | 25 × 25 mm | 6 | 100×100×40 | 400k |

**geometry1 / geometry3 layer stack (bottom → top):**
```
Heat sink (Cu)  5000 µm  k=400 W/m·K
TIM             100 µm   k=4 W/m·K
Spreader (Cu)   1000–2000 µm
TIM             100 µm
Die (Si)        150–200 µm  k(T)=148×(300/T)^1.3
TIM2            50 µm   ← convective BC applied here
```

**geometry2a/b/c additional layers (stacked die + TSV):**
```
Die 2 Active (Si)   50 µm
Die 2 TSV region    100 µm  k_eff = (1−φ)·k_Si + φ·k_Cu
Bonding layer       25 µm   k=50 W/m·K  (micro-bump)
Die 1 TSV region    100 µm
Die 1 Active (Si)   50 µm
```

TSV densities: 2a → 3% (k_eff=155 W/m·K), 2b → 5% (163 W/m·K), 2c → 10% (184 W/m·K).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch, NumPy, SciPy, PyYAML, Matplotlib.

### 2. Generate training data (requires 3D-ICE in WSL2)

```powershell
python src/main.py --all-geometries --simulator 3d-ice `
    --ice-executable "wsl /home/user/3d-ice/bin/3D-ICE-Emulator" `
    --output data/3d-ice --verbose
```

For development without 3D-ICE (synthetic data):
```powershell
python src/main.py --all-geometries --simulator mock --output data/3d-ice-mock
```

### 3. Train the PINN

**On CPU (i7, ~2–3 hr per geometry):**
```powershell
python scripts/train_pinn.py --geometry geometry1 `
    --data data/3d-ice --output checkpoints --cpu-fast
```

**On Kaggle P100 GPU (~1.5–2 hr per geometry):**
Upload `src/` and `data/3d-ice/` as Kaggle datasets, then run `notebooks/kaggle_pinn_geometry1.ipynb`. See `notebooks/KAGGLE_SETUP.md`.

**Full settings (GPU):**
```powershell
python scripts/train_pinn.py --geometry geometry1 `
    --data data/3d-ice --output checkpoints `
    --epochs 8000 --fourier-sigma 10.0
```

Use `--fourier-sigma 20.0` for geometry2 variants (captures TSV-scale features).

### 4. Evaluate

```powershell
python scripts/eval_pinn.py `
    --checkpoint checkpoints/geometry1/geometry1_best.pt `
    --data data/3d-ice --output results/geometry1 --plots
```

### 5. Explainability

```powershell
python scripts/explain_pinn.py `
    --checkpoint checkpoints/geometry1/geometry1_best.pt `
    --data data/3d-ice --output results/explain/geometry1
```

Outputs per test scenario: PDE residual map, per-block thermal influence map, HTC sensitivity map, Integrated Gradients at hotspot + z-profile, MC Dropout uncertainty map.

---

## Model Architecture

**FourierPINN** — 807k parameters.

| Stage | Operation | Output dim |
|---|---|---|
| Input | (x̂, ŷ, ẑ) ∈ [0,1]³ | 3 |
| Fourier encoding | B ~ N(0,σ²), [sin, cos] | 32 |
| Layer embedding | Embedding(n_layers, 8) | 8 |
| Scalar inputs | Q_norm, htc_norm, t_amb_norm, tsv_frac | 4 |
| MLP input | concat | 44 |
| Residual MLP | 6 × ResBlock(256) + Dropout(0.1) | 256 |
| Output | Linear(1) → T̂ ∈ [0,1] | 1 |

Training uses a three-stage curriculum: data-only (epochs 0–1000) → fixed physics weights (1000–3000) → NTK-adaptive weights (3000–8000). PDE residual enforces `∇·(k∇T) + Q = 0`; convective and adiabatic boundary conditions are separate loss terms.

---

## Training Flags

```
--geometry       geometry1 | geometry2a | geometry2b | geometry2c | geometry3
--fourier-sigma  10.0 (geometry1/3) | 20.0 (geometry2*)
--epochs         8000 (default) | 3000 (--cpu-fast)
--hidden-dim     256 (default) | 128 (--cpu-fast)
--n-res-blocks   6 (default)   | 4   (--cpu-fast)
--n-col          20000 (default) | 5000 (--cpu-fast)
--cpu-fast       Apply all CPU-optimised defaults (~2–3 hr per geometry on i7)
```

---

## Dataset

Each geometry: **15 training + 5 test scenarios** sweeping:
- Power density: 0.1–20 W/cm², six spatial patterns (uniform, hotspot, checkerboard, gradient, dual-hotspot, extreme-hotspot)
- HTC: 500–10,000 W/m²·K
- Ambient temperature: 25–85 °C

Ground truth from **3D-ICE Emulator** (WSL2). Data format: compressed `.npz` with arrays `coords (N,3)`, `temp (N,)`, `power (N,)`, `layer (N,)`, and a metadata dict including per-block power densities.

---

## Explainability Methods

| Method | What it shows | Cost (i7 CPU) |
|---|---|---|
| PDE residual map | Where the heat equation is violated | ~8 s/scenario |
| Power block sensitivity | dT/dQ — thermal influence coefficients | ~20 s/scenario |
| HTC sensitivity | dT/dHTC — where cooling matters most | ~4 s/scenario |
| Integrated Gradients | Feature attribution at hotspot + z-profile | ~2 s/scenario |
| MC Dropout uncertainty | Predictive std — where model is uncertain | ~200 s/scenario |

---

## Key Design Decisions and Assumptions

See [`docs/assumptions.md`](docs/assumptions.md) for the complete list. Notable items:

- **Steady-state only** — transient peak temperatures can be 10–40% higher
- **Uniform HTC** — spatially varying cooling (e.g., jet impingement) not modelled
- **Micro-bump bonding** in geometry2 — not hybrid bonding (Cu-Cu direct)
- **Arithmetic-mean TSV k** — upper bound, consistent with 3D-ICE, ~10% over actual at 10% density
- **k(T) quasi-linearised** in PDE loss — Picard iteration, valid approximation at data loss weight 1.0 vs PDE weight 0.1

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

Full citations in [`docs/references.md`](docs/references.md).
