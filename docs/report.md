# A 3D-IC Thermal Surrogate Benchmark, and What It Reveals About Surrogate Evaluation

**Rajul Kabeer**  
*Manuscript in preparation*

> **STATUS (2026-07-31) — MAJOR REFRAME IN PROGRESS.**
> The previous version of this draft claimed FourierPINN was "the first PINN for the
> multi-layer 3D-IC stack", "the first to treat TSV density as a continuous parametric
> input", and reported an *expected* MAE of < 2 K. Measurements taken on 2026-07-31 with
> `scripts/baselines.py` invalidated that framing:
>
> - **Closed-form ridge regression reaches spatial R² = 0.999 (geometry1) and 0.937
>   (geometry6)**, with spatially-detrended MAE of 0.009 K. On geometry6 its raw MAE is
>   0.616 K — already better than the < 2 K target this draft hoped a trained PINN would hit.
> - The dataset is **spatially degenerate**: across all 335 live simulations the median
>   within-scenario spatial temperature range is 1.10 K and the maximum anywhere is 6.40 K,
>   while the between-scenario mean varies by 68 K. 88% of geometry1's temperature variance
>   is explained by the ambient input alone.
> - TSV density takes **four discrete values**, not continuous ones. That claim was false.
> - Lateral heterogeneity is settled prior art (3D-ICE 4.0, arXiv:2512.05823).
>
> Sections 1–7 below are retained for reference but their novelty framing is superseded.
> Sections 8–11 are being rewritten around the abstract below. Do not submit the old framing.
>
> **Update (2026-08-01):** the full dataset has been regenerated on a physically grounded
> operating point (TDP budgets, core-fraction power, cooling coupled to power density,
> sub-layer vertical resolution). Median spatial ΔT rose 1.10 K → 10.77 K and vertical
> resolution 6–11 → 10–15 nodes. Ridge nonetheless still solves every split at
> R² > 0.94 — see §9.3, which establishes that no parameter-range choice can make this
> benchmark non-linear. A double-counted `T_range` in the PDE conduction term was also
> found and fixed (§9.5).

---

## Abstract

Thermal analysis of three-dimensional integrated circuits (3D-ICs) is a bottleneck in
early-stage design exploration, and a growing body of work applies neural surrogates —
PINNs, Fourier neural operators, DeepONets — to accelerate it. We contribute an open
benchmark of 335 3D-ICE simulations across eight package geometries, spanning single-die
mobile and server stacks, dual-die stacks at three TSV densities, and 2.5D/CoWoS-style
chiplet assemblies with up to six HBM stacks, together with the geometry definitions and
generation pipeline needed to reproduce and extend it.

Our principal finding is methodological and cautionary. Evaluating four non-neural
baselines on this data, we find that **per-point ridge regression — a closed-form solve
requiring no GPU and no training — reconstructs the spatial temperature field at R² = 0.999**,
and extrapolates to unseen power patterns, unseen power magnitudes, and unseen ambient
temperatures at R² > 0.9. This is not a defect of any particular architecture but a
consequence of the physics: steady-state conduction is linear in the volumetric sources and
in ambient temperature, and when a scenario space is parameterised by a handful of block
powers and boundary scalars, the entire solution manifold is low-dimensional and nearly
linear. We further show that raw MAE — the metric almost universally reported in this
literature — is dominated by a per-scenario scalar offset, so a model predicting a constant
per scenario can post a competitive MAE while capturing no spatial structure at all. We
therefore report **spatially-detrended error** and argue it should be standard.

We further show the result is not an artefact of an unlucky parameter range. Regenerating
the entire dataset on a physically grounded operating point — power drawn from package TDP
budgets, cooling constrained to be adequate for the power density, and the vertical
direction resolved by sub-layer discretisation — raises the median within-scenario spatial
gradient tenfold, yet ridge still solves every extrapolation split at R² > 0.94. An
intermediate version did defeat the linear model, but only by containing combinations that
cannot physically exist, such as 150 W/cm² against air-class cooling. Escaping linearity
requires changing *what varies* — per-cell power maps rather than a handful of block
scalars, variable floorplans, or transient operation — not the range over which the
existing handful varies. We release the dataset, baselines, and OOD split tooling so future
surrogate claims can be checked against a linear model before an architecture is credited.

---

## 1. Introduction

As chiplet integration and 3D stacking become mainstream packaging approaches, thermal design must keep pace with increasing power densities and more complex heat flow paths. Die stacking with TSV interconnects creates new thermal coupling mechanisms: each die heats the one above it, TSVs provide vertical thermal shortcuts, and the effective thermal resistance from junction to coolant is now distributed across multiple material interfaces. Evaluating even a single configuration requires solving the three-dimensional heat equation over a heterogeneous multi-layer domain.

Compact thermal simulators such as 3D-ICE [Sridhar et al., 2010] model this as a finite-difference RC-network on a structured Cartesian grid and can produce steady-state solutions in seconds. However, design exploration over power patterns, cooling intensities, and TSV densities requires thousands of such evaluations. Full parametric sweeps remain slow, and compact simulators have limited expressiveness — they cannot represent individual TSV columns, spatially-varying convective coefficients, or transient workload effects.

Machine learning surrogates offer a different trade-off: expensive offline training amortised over fast online inference. Prior work has applied convolutional neural networks [Zhang et al., 2025], graph neural networks [WarPGNN, 2026], and operator-learning approaches including FNO variants [Self-Attention U-Net FNO, 2025] to chip thermal prediction. Physics-informed neural networks (PINNs) add physics constraints — the heat equation and boundary conditions — as additional loss terms, which regularise training and improve generalisation beyond the training distribution. ThermPINN [Cheng et al., 2024] demonstrated PINNs for full-chip 2D thermal analysis. No prior work has applied a PINN to the 3D multi-layer package stack (die + TIM + spreader + heat sink) with TSV parametric variation, nor provided explainability analysis for any chip thermal neural surrogate.

This paper makes the following contributions:

1. An open **eight-geometry benchmark of 335 3D-ICE simulations** covering single-die mobile
   and server stacks, dual-die stacks at three TSV densities (3%, 5%, 10%), and 2.5D/CoWoS
   chiplet assemblies with up to six HBM stacks — with the full generation pipeline.
2. **A demonstration that closed-form ridge regression solves this benchmark** (spatial
   R² = 0.999), and that the result holds under extrapolation to unseen power patterns,
   power magnitudes and ambient temperatures. Any neural architecture evaluated on data of
   this kind must be compared against a linear model before its capacity is credited.
3. **Spatially-detrended error as an evaluation metric.** We show raw MAE on 3D-IC thermal
   data is dominated by a per-scenario scalar offset — 88% of variance here is explained by
   the ambient input alone — and that detrending is required to measure spatial fidelity.
4. **Evidence that the result is structural, not a parameter-range artefact.** A full
   regeneration on a physically grounded operating point raises the median spatial
   gradient from 1.10 K to 10.77 K and still leaves ridge at R² > 0.94 on every split.
   Configurations that *do* defeat the linear model turn out to be physically impossible.
   We give the conditions a non-degenerate 3D-IC thermal benchmark must satisfy.
5. Reference implementations of five surrogate families (PINN, FNO/WHNO/CNO-FNO, DeepONet,
   autoregressive z-layer operator, few-shot fine-tuning) with a shared explainability
   toolkit, released as infrastructure rather than as accuracy claims.

---

## 2. Related Work

**Compact thermal simulation.** 3D-ICE [Sridhar et al., 2010; updated 2021] is the standard open-source compact thermal solver for 3D-IC stacks, using finite differences on a structured grid. HotSpot [Huang et al., 2006] targets 2D die-level analysis with a lumped-resistance model. Both treat TSV arrays as homogenised effective-medium layers. Full FEM tools (COMSOL, Ansys) resolve individual TSV cylinders but require hours per simulation.

**Neural thermal surrogates.** CNN-based surrogates for HBM chiplet stacks [arXiv 2503.04049, 2025] achieve good accuracy on multi-layer configurations but without physics constraints, limiting generalisation. The Self-Attention U-Net FNO [arXiv 2510.15968, 2025] reports 842× speedup over FEM for 3D-IC thermal prediction using operator learning. ThermPINN [IEEE, 2024] applies PINNs to 2D VLSI full-chip thermal, demonstrating 10³–10⁴× speedup over iterative solvers.

**Physics-informed neural networks.** Raissi et al. [2019] introduced PINNs. Tancik et al. [2020] demonstrated that random Fourier feature encoding overcomes spectral bias in coordinate networks. Wang et al. [2021] introduced Neural Tangent Kernel-based adaptive loss weighting for PINNs. Cai et al. [2021] applied PINNs to 2D heat transfer problems.

**Explainability for neural surrogates.** Sensitivity analysis for PINNs is discussed in [arXiv 2301.02428]. Integrated Gradients [Sundararajan et al., 2017] provides input attribution with completeness guarantees. MC Dropout as Bayesian approximation for uncertainty quantification was proposed by Gal & Ghahramani [2016]. No prior work applies these methods to chip thermal surrogates.

---

## 3. Problem Statement

We model steady-state heat conduction in a 3D-IC package stack governed by:

$$\nabla \cdot (k(T) \nabla T) + Q = 0 \quad \text{in } \Omega$$

with a convective boundary condition at the bottom surface (the `heat_sink` layer, `z=0`), matching 3D-ICE's own boundary-condition convention:

$$-k \frac{\partial T}{\partial z}\bigg|_{z=0} = h \left(T - T_{amb}\right)$$

and adiabatic conditions on all lateral faces and the top surface (nearest the die). The domain Ω is a rectangular cuboid covering the full package stack — heat sink, spreader, TIM layers, and one or two active silicon dies. The volumetric power source Q is non-zero only in active silicon layers within designated power blocks.

The thermal conductivity of silicon is temperature-dependent: $k_{Si}(T) = 148 \cdot (300/T)^{1.3}$ W/m·K [Glassbrenner & Slack, 1964], while copper and TIM layers use constant values.

The surrogate model $\hat{T}_\theta: \mathbb{R}^7 \to \mathbb{R}$ maps per-point inputs $(x, y, z, Q, h, T_{amb}, \phi_{TSV})$ to normalised temperature, where the first three are normalised spatial coordinates, Q is normalised volumetric power density, h and $T_{amb}$ are normalised scenario-level scalars, and $\phi_{TSV} \in \{0, 0.03, 0.05, 0.10\}$ is the TSV area fraction.

---

## 4. Benchmark Geometries

Eight geometries span the design space from a mobile-class single die to server-class dies,
3D-stacked configurations with TSV arrays, and 2.5D/CoWoS chiplet assemblies. All layers are
modelled with homogenised material properties. TSV regions use the arithmetic-mean effective
conductivity $k_{eff} = (1-\phi) k_{Si} + \phi k_{Cu}$, consistent with the 3D-ICE
ground-truth simulator.

| Geometry | Type | Die size | Layers | Files | Notes |
|---|---|---|---|---|---|
| geometry1 | 2D stack | 10 × 10 mm | 6 | 45 | Mobile/desktop single die |
| geometry2a | 3D stack | 8 × 8 mm | 10 | 30 | TSV density 3% |
| geometry2b | 3D stack | 8 × 8 mm | 10 | 30 | TSV density 5% |
| geometry2c | 3D stack | 8 × 8 mm | 10 | 30 | TSV density 10% |
| geometry3 | 2D stack | 25 × 25 mm | 6 | 45 | Server-class, 8 core clusters |
| geometry4 | 2.5D stack | 25 × 14 mm | 6 | 45 | Two chiplets on interposer |
| geometry5 | 2.5D stack | 25 × 14 mm | 11 | 55 | CoWoS: compute + HBM stack |
| geometry6 | 2.5D stack | 42 × 14 mm | 11 | 55 | CoWoS: 6 HBM stacks |

Total: **335 simulations**. TSV density takes four discrete values (0, 3%, 5%, 10%) — it is a
categorical variant axis, not a continuous parameter, and interpolation in TSV space cannot
be meaningfully demonstrated from three non-zero points.

Each geometry has 6 layers (geometry1, geometry3) or 10 layers (geometry2 variants): heat sink (Cu, 5000 µm) → TIM (100 µm) → spreader (Cu) → TIM (100 µm) → active die(s) (Si, 50–200 µm) → TIM2 (50 µm). TIM2 as the topmost layer ensures the convective BC is applied at the die-to-package interface with correct contact resistance.

The z-grid uses adaptive spacing guaranteeing a minimum of 8 sample points through each active die layer, preventing the thin die layers from being dominated by the coarser heat sink discretisation.

---

## 5. Architecture: FourierPINN

The network maps a point's 7-dimensional input to normalised temperature through four stages.

**Fourier feature encoding.** Coordinates $(\hat{x}, \hat{y}, \hat{z}) \in [0,1]^3$ are projected through a fixed random matrix $\mathbf{B} \sim \mathcal{N}(0, \sigma^2)$ into sinusoidal features $[\sin(2\pi \mathbf{B}\hat{x}), \cos(2\pi \mathbf{B}\hat{x})] \in \mathbb{R}^{32}$. We use $\sigma=10$ for geometries without TSVs and $\sigma=20$ for the TSV geometries to capture the finer spatial scale (~500 µm) of TSV-enhanced heat spreading.

**Layer embedding.** The integer layer index is mapped to an 8-dimensional learnable embedding, providing the network with explicit material identity (silicon, copper, TIM) without relying on z-coordinate alone to infer material boundaries.

**Residual MLP.** The concatenated 44-dimensional input ($32 + 8 + 4$ scalars) passes through a linear projection to 256 dimensions, then six residual blocks. Each block applies LayerNorm → Linear → SiLU → Dropout(p=0.1) → LayerNorm → Linear, with a SiLU-activated skip connection. The SiLU activation provides smooth non-zero second derivatives needed for PDE autograd. LayerNorm is used rather than BatchNorm because the effective batch size is one scenario at a time. Dropout at p=0.1 serves dual purpose: mild regularisation during training, and Monte Carlo uncertainty estimation at inference.

**Output head.** A single linear layer maps to normalised temperature $\hat{T} \in [0,1]$, denormalised as $T_K = \hat{T}(T_{max} - T_{min}) + T_{min}$.

Total parameters: 807,473.

---

## 6. Training

**Dataset.** Each geometry has 25–50 training and 5 test scenarios (335 files total). Scenarios sweep power density (0.1–20 W/cm²), HTC (500–10,000 W/m²·K), and ambient temperature (25–85°C) across six spatial power patterns (uniform, hotspot, checkerboard, gradient, dual-hotspot, extreme hotspot). Ground truth temperatures are generated by the 3D-ICE Emulator running under WSL2. Note that the shipped `*_test_*` files are interpolation points inside the training sweep; extrapolation splits are generated separately by `scripts/make_ood_split.py` (§9.1).

**Normalisation.** Spatial coordinates are normalised to [0,1] by domain extents. Power density is zero-mean unit-variance standardised over the training set. HTC, ambient temperature, and TSV fraction are each normalised to [0,1] over their respective physical ranges.

**Curriculum training.** Training proceeds in three stages:

- *Stage 1 (epochs 0–900):* Data loss only, $\mathcal{L} = \mathcal{L}_{data}$. The network fits the 3D-ICE temperature field.
- *Ramp (epochs 900–1000):* PDE and BC losses linearly activated to avoid a discontinuous gradient shock.
- *Stage 2 (epochs 1000–3000):* $\mathcal{L} = \mathcal{L}_{data} + 0.1\,\mathcal{L}_{PDE} + 0.5\,\mathcal{L}_{BC}$. Physics regularisation introduced at fixed weights.
- *Stage 3 (epochs 3000–8000):* NTK-adaptive weights updated every 50 epochs. Each weight $\lambda_i$ is set proportional to $\max_j \|\nabla_\theta \mathcal{L}_j\|_2 / \|\nabla_\theta \mathcal{L}_i\|_2$, smoothed with EMA ($\alpha=0.9$).

The PDE residual uses a quasi-linearised treatment of $k(T)$: thermal conductivity is computed from the current temperature prediction but detached from the autograd graph, making the residual linear in $T$ for each gradient step (Picard iteration). This is equivalent to full nonlinear residual minimisation at convergence and avoids the ~30% compute overhead of differentiating through $k(T)$.

Collocation points for the PDE loss are sampled uniformly in the 3D domain each step, with correct volumetric power density $Q(x,y)$ assigned by spatial block membership. Adiabatic boundary conditions are enforced separately per face normal direction (x-faces with $\partial T/\partial x=0$, y-faces with $\partial T/\partial y=0$, bottom with $\partial T/\partial z=0$). The training scenario order is shuffled each epoch.

**Optimisation.** Adam with initial learning rate $3 \times 10^{-4}$ and CosineAnnealingWarmRestarts ($T_0=2000$). Gradient clipping at max-norm 1.0.

---

## 7. Explainability

We introduce three post-hoc explainability methods for chip thermal surrogates, requiring no retraining.

**PDE residual maps.** The absolute heat-equation residual $|\nabla \cdot (k\nabla\hat{T}) + Q|$ is evaluated on the data grid after training. High residual identifies regions where the model's predicted temperature gradient violates physics — useful for flagging unreliable predictions near material interfaces or TSV boundaries before committing to a design decision.

**Engineering sensitivity maps.** For each power block $k$, the spatial influence field $\partial T_i / \partial Q_k$ is computed via central finite difference (two forward passes per block, $\delta Q = 0.1$ W/cm²). This produces the thermal impedance matrix: the temperature response at each grid point to a unit power increase in each block, directly actionable for floorplan decisions. A second map, $\partial T_i / \partial h$, shows where improved cooling is most effective.

**Targeted Integrated Gradients.** Integrated Gradients [Sundararajan et al., 2017] are applied at the predicted hotspot and at one representative point per layer along the hotspot column. For each point, the attribution decomposes the predicted temperature into contributions from all seven input features relative to a neutral baseline (domain centre, zero power, mid-range HTC and ambient). The completeness property guarantees $\sum_j \text{IG}_j \approx T_{pred} - T_{baseline}$. Targeting the hotspot rather than all grid points reduces cost from ~200s to ~2s per scenario on a CPU.

**MC Dropout uncertainty.** With Dropout(p=0.1) active in each residual block during inference, $N=100$ stochastic forward passes produce a predictive distribution $T \sim \mathcal{N}(\mu, \sigma^2)$ at each grid point. High predictive standard deviation identifies regions where more training scenarios would most improve reliability.

---

## 8. Experimental Setup

**Compute.** Training is run on Kaggle P100 GPUs (16 GB HBM2, 732 GB/s bandwidth). Expected training time: 1.5–2 hours per geometry at full settings (8000 epochs). Explainability analysis runs post-training on CPU (Intel i7, no GPU required): PDE residual and sensitivity maps complete in ~25s per scenario; targeted IG in ~2s; MC Dropout uncertainty in ~200s (100 samples).

**Baselines.** Prediction accuracy is evaluated as mean absolute error (MAE), RMSE, $R^2$, and hotspot temperature error relative to 3D-ICE ground truth on the held-out test split (5 scenarios per geometry). We also report the predicted hotspot location error in µm.

**Ablations planned.** (1) Data-only training vs. curriculum PINN; (2) fixed vs. NTK-adaptive weights; (3) Fourier features vs. plain coordinates; (4) with vs. without the per-layer z-minimum grid constraint.

---

## 9. Results

### 9.1 Baselines (measured 2026-07-31, `scripts/baselines.py`)

No neural model has been trained yet; every number below is from a non-neural baseline.
`det.MAE` and `spat.R²` are computed after removing each field's mean, so they score spatial
structure only.

**In-distribution (shipped `*_test_*` split):**

| Geometry | Baseline | MAE (K) | det.MAE (K) | spatial R² |
|---|---|---|---|---|
| geometry1 | mean | 18.629 | 0.270 | 0.586 |
| geometry1 | knn (k=3) | 8.765 | 0.141 | 0.876 |
| geometry1 | **ridge** | **4.110** | **0.009** | **0.999** |
| geometry6 | mean | 10.296 | 0.244 | −3.656 |
| geometry6 | knn (k=3) | 3.797 | 0.090 | −0.621 |
| geometry6 | **ridge** | **0.616** | **0.020** | **0.937** |

Mean spatial std of the true test fields is 0.506 K (geometry1) and 0.261 K (geometry6) —
this is the entire signal a surrogate exists to predict. Ridge captures it to 0.009 K.

**Out-of-distribution (geometry1, `scripts/make_ood_split.py`), ridge only:**

| Held-out axis | MAE (K) | det.MAE (K) | spatial R² |
|---|---|---|---|
| power pattern (peaked patterns unseen) | 1.354 | 0.027 | 0.902 |
| power magnitude (upper half unseen) | 4.132 | 0.010 | 0.998 |
| ambient (upper half unseen) | 4.769 | 0.029 | 0.916 |
| **HTC (outside interquartile range)** | 3.784 | 0.020 | **−0.231** |

Linear extrapolation succeeds on three of four axes and degrades only on HTC — consistent
with temperature depending on 1/h, a reciprocal coordinate in which linear extrapolation
outside the training range is ill-posed.

The HTC result holds across geometries but with wide spread, and is strongest on the complex
CoWoS stacks (ridge spatial R², HTC OOD split):

| g1 | g2a | g3 | g4 | g5 | g6 |
|---|---|---|---|---|---|
| −0.23 | 0.44 | 0.02 | 0.72 | **−6.15** | **−3.60** |

Against R² > 0.9 on every other axis, HTC extrapolation is clearly the weak point, but it is
not uniformly fatal — ridge still handles geometry2a and geometry4. The two 11-layer chiplet
geometries (5, 6) degrade most, plausibly because more material interfaces make the
interaction between 1/h and the spatial field harder to capture with a per-point linear fit.
This is the axis on which a neural surrogate has the clearest opportunity to demonstrate
value, and the one on which it should be evaluated.

### 9.2 Dataset degeneracy

Across all 335 live simulations:

| Quantity | Value |
|---|---|
| Median within-scenario spatial ΔT | 1.10 K |
| Max within-scenario spatial ΔT (anywhere) | 6.40 K |
| Between-scenario mean-temperature range | 68.0 K |
| Ratio | 62× |
| Variance explained by ambient alone (geometry1) | 88.1% |

The cause is the power regime. Total dissipated power is 0.4–40 W/cm² summed over blocks,
giving 0.1–36 K of self-heating, while the ambient sweep alone spans 60 K. `references.md`
already noted the 20 W/cm² ceiling was conservative against the 100–300 W/cm² of real CPU
hotspots; the consequence, not previously recognised, is that the benchmark cannot
discriminate between architectures.

### 9.3 Final dataset, and why regime tuning cannot rescue the benchmark

The dataset was regenerated end to end (2026-08-01) with a physically grounded
operating point: power from package TDP budgets, only the core fraction of TDP assigned
to modelled blocks, cooling coupled to power density, and the vertical direction
resolved by sub-layer discretisation.

**Dataset (335 simulations, all real 3D-ICE):**

| Geometry | n | z-nodes | pts/file | median ΔT | max ΔT | median T | peak T |
|---|---|---|---|---|---|---|---|
| geometry1 | 45 | 10 | 100,000 | 29.05 K | 96.43 K | 78.8 °C | 158.5 °C |
| geometry2a | 30 | 14 | 89,600 | 9.26 K | 61.06 K | 54.4 °C | 92.3 °C |
| geometry2b | 30 | 14 | 89,600 | 9.21 K | 60.59 K | 54.3 °C | 91.8 °C |
| geometry2c | 30 | 14 | 89,600 | 9.09 K | 59.46 K | 54.2 °C | 90.7 °C |
| geometry3 | 45 | 11 | 110,000 | 17.74 K | 87.70 K | 62.7 °C | 134.9 °C |
| geometry4 | 45 | 10 | 56,000 | 15.76 K | 59.16 K | 69.0 °C | 121.9 °C |
| geometry5 | 55 | 15 | 84,000 | 7.35 K | 46.90 K | 58.5 °C | 113.2 °C |
| geometry6 | 55 | 15 | 141,120 | 8.96 K | 79.38 K | 63.3 °C | 133.9 °C |

Median within-scenario spatial ΔT rose from **1.10 K to 10.77 K**, vertical resolution
from 6–11 to 10–15 nodes, and only 4 of 335 scenarios exceed 125 °C (all
`extreme_hotspot` stress cases, down from 20).

**Baselines on the final data (geometry1):**

| Split | ridge det.MAE | ridge spatial R² |
|---|---|---|
| in-distribution | 0.223 K | 0.987 |
| pattern OOD | 0.961 K | 0.944 |
| power OOD | 0.357 K | 0.998 |
| HTC OOD | 0.331 K | 0.964 |

**This is the paper's central negative result, and it is now established rather than
asserted.** An intermediate dataset did drive ridge to spatial R² −16.7 on the HTC
split, which looked like a fix. It was an artefact: that version swept power and cooling
as independent variables and so contained combinations that cannot exist — 150 W/cm²
against air-class cooling, junctions at 165–273 °C. Once cooling was constrained to be
adequate for the power density, the linear model recovered to R² > 0.94 on every axis.

Rescaling cooling into the feasible band rather than clamping it to the minimum
recovered only 0.980 → 0.964, confirming the effect is structural rather than a tuning
artefact. Some correlation between power and cooling is physically obligatory, and it is
enough for a linear model.

The conclusion is therefore stronger than "this dataset happens to be easy":
**no choice of parameter ranges makes this benchmark non-linear.** Steady-state
conduction is linear in its sources and boundary values, and the scenario space is
roughly eight scalars, so the solution manifold is low-dimensional and nearly linear by
construction. Escaping that requires changing what varies — per-cell power maps instead
of a handful of block scalars, variable floorplans, or transient operation (§10) — not
changing the range over which the existing handful varies.

### 9.4 Regime fix and its effect (intermediate result, superseded by §9.3)

The degeneracy in §9.2 was a benchmark-design fault, not a property of 3D-IC thermal
problems. `src/scenario/generator.py` was revised (2026-07-31) on three coupled axes —
power scaled 15× so peak hotspots reach 300 W/cm², HTC raised to 2000–50000 W/m²·K (the
spatial fraction of the temperature drop is $R_{cond}/(R_{cond} + 1/h)$, so low HTC was
actively flattening the field), and ambient compressed to 25–45 °C so it no longer swamps
self-heating.

geometry1 was regenerated through 3D-ICE under the new regime:

| Quantity | Old regime | New regime |
|---|---|---|
| Median within-scenario spatial ΔT | 1.10 K | **9.51 K** |
| Max within-scenario spatial ΔT | 6.40 K | **80.73 K** |
| Between-scenario / spatial ratio | 62× | **6.0×** |
| Ridge spatial R² (in-distribution) | 0.999 | **0.954** |
| Ridge detrended MAE | 0.009 K | **0.232 K** |
| Ridge spatial R² (pattern OOD) | 0.902 | **0.501** |
| Ridge spatial R² (HTC OOD) | −0.23 | **−16.7** |

The benchmark now discriminates on two axes rather than none: on the power-pattern split
ridge's detrended MAE (2.70 K) exceeds the signal's own standard deviation (1.76 K), i.e.
the linear model is worse than useless there. Power extrapolation remains linear-solvable
(R² 0.923), as conduction genuinely is linear in the sources.

Only geometry1 has been regenerated; the other seven still carry the old regime and their
numbers in §9.1–9.2 should be read as describing the superseded dataset.

### 9.5 A physics bug found while building these tests

`pde_residual` applied the temperature range $(T_{max}-T_{min})$ twice — once converting
normalised gradients to physical ones, and again as a trailing factor on the divergence.
The conduction term was therefore inflated by roughly 65–100×, so the PDE loss effectively
enforced $\nabla\cdot(k\nabla T) = 0$ while ignoring the source $Q$. No trained result is
affected because none exists, but every physics-loss run before 2026-07-31 optimised the
wrong objective. It is now pinned by a manufactured-solution test that compares the residual
against the closed-form $k \cdot 2c \cdot T_{range}/L_z^2$.

### 9.6 Neural results

*None. No checkpoint has been trained. Any neural number added here must be reported
alongside the ridge baseline on the same split, using detrended metrics.*

---

## 10. Discussion

**Why a linear model wins.** Steady-state conduction with temperature-independent $k$ is a
linear map from sources and boundary data to the temperature field:
$T(x) = T_{amb} + \sum_b A_b(x) Q_b$, where the impedance $A_b$ is scenario-independent. Our
scenario space is parameterised by 4–13 block powers plus three boundary scalars, so the
whole dataset lies on a low-dimensional, nearly-linear manifold that a per-point ridge fit
recovers exactly. The only genuine nonlinearity, $k(T)$, is inactive here because
within-scenario gradients are ~1 K. Neural operators are built for high-dimensional,
nonlinear function-to-function maps; supplying them a map whose input is effectively a
handful of scalars removes the problem they exist to solve.

**Implications for the field.** Published 3D-IC thermal surrogates typically report raw MAE
or RMSE against a compact simulator, on interpolation splits, without a linear baseline. Our
results suggest such numbers can be substantially uninformative: on this dataset raw MAE is
dominated by a scalar offset that tracks the ambient input, and a closed-form solve matches
or beats the accuracy targets neural models are held to. We do not claim published results
are wrong — their datasets may be richer — but the comparison is rarely made, and it is cheap
to make. We release `scripts/baselines.py` for that purpose.

**What a discriminative benchmark needs.** From §9, four changes:
1. **Higher power density.** 100–300 W/cm² localised, so self-heating dominates the ambient
   sweep and $k(T)$ becomes active.
2. **Per-cell power maps** rather than a handful of uniform blocks, so the input is genuinely
   a function and not a short vector.
3. **Variable floorplans/geometry** across scenarios, which is where operator learning has a
   real advantage over per-point regression.
4. **Extrapolation splits by default**, especially in $1/h$ — the one axis here where the
   linear model fails.

Transient simulation would add a further nonlinear axis; the present dataset is steady-state
only.

**Threats to validity.** Ground truth is 3D-ICE, itself a compact RC-network approximation;
HotSpot cross-validation covers geometry1 only and disagrees by ~15%, and no FEM spot-check
has been performed. The geometry4/5/6 physics loss uses a heterogeneous lateral $k$ that the
ground truth does not contain (`assumptions.md` §6.1). Ridge's advantage is measured on
scenario counts of 20–50; with far more scenarios and a richer power parameterisation the
ranking could change.

**Deferred.** The originally planned discussion — TSV generalisation across geometry2a/b/c,
PDE-residual/error correlation, IG attribution plausibility, MC Dropout calibration — requires
trained models and remains open. MC Dropout is already known to be poorly calibrated here:
the predictive std measured during explainability validation was 8.6–13 K on fields whose
spatial std is ~0.3 K, i.e. roughly 30× the signal.

---

## 11. Conclusion

We release an eight-geometry, 335-simulation 3D-IC thermal benchmark with its full generation
pipeline, and report a negative result we believe is more useful than the surrogate accuracy
figures we set out to produce: on this data, closed-form ridge regression reconstructs the
spatial temperature field at R² = 0.999 and extrapolates to unseen power patterns, magnitudes
and ambient temperatures at R² > 0.9. The benchmark is discriminative only for extrapolation
in the convective coefficient.

Two practices follow. First, **report a linear baseline** — it costs milliseconds and bounds
what any architecture can claim to contribute. Second, **report spatially-detrended error**:
raw MAE on 3D-IC thermal fields is dominated by a per-scenario offset, and a constant
predictor can look competitive while capturing no spatial structure.

We also record what this benchmark would need to become discriminative — higher power
density, per-cell power maps, variable floorplans, extrapolation splits by default — and
release the baseline and OOD tooling so those conditions can be checked rather than assumed.

---

## References

1. Sridhar, A. et al. "3D-ICE: Fast Compact Transient Thermal Modeling for 3D-ICs with Inter-Tier Liquid Cooling." *ICCAD*, 2010. DOI: 10.1109/ICCAD.2010.5653749

2. Huang, W. et al. "HotSpot: A Compact Thermal Modeling Methodology for Early-Stage VLSI Design." *IEEE TVLSI*, 14(5), 2006. DOI: 10.1109/TVLSI.2006.876103

3. Glassbrenner, C. J. & Slack, G. A. "Thermal Conductivity of Silicon and Germanium from 3 K to the Melting Point." *Physical Review*, 134:A1058, 1964. DOI: 10.1103/PhysRev.134.A1058

4. Raissi, M., Perdikaris, P. & Karniadakis, G. E. "Physics-Informed Neural Networks." *Journal of Computational Physics*, 378:686–707, 2019. DOI: 10.1016/j.jcp.2018.10.045

5. Tancik, M. et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." *NeurIPS*, 2020.

6. Wang, S., Teng, Y. & Perdikaris, P. "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks." *SIAM J. Sci. Comput.*, 43(5):A3055–A3081, 2021. DOI: 10.1137/20M1318043

7. Wang, S., Sankaran, Y. & Perdikaris, P. "Respecting Causality for Training Physics-Informed Neural Networks." *CMAME*, 2024. DOI: 10.1016/j.cma.2024.116928

8. Cai, S. et al. "Physics-Informed Neural Networks for Heat Transfer Problems." *Journal of Heat Transfer*, 143(6):060801, 2021. DOI: 10.1115/1.4050542

9. Sundararajan, M., Taly, A. & Yan, Q. "Axiomatic Attribution for Deep Networks." *ICML*, 2017.

10. Gal, Y. & Ghahramani, Z. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*, 2016.

11. Zhu, K., Huang, D., Costero, L. & Atienza, D. "3D-ICE 4.0: Accurate and Efficient Thermal Modeling for 2.5D/3D Heterogeneous Chiplet Systems." arXiv:2512.05823, 2025.

12. "MFIT: Multi-FIdelity Thermal Modeling for 2.5D and 3D Multi-Chiplet Architectures." *ACM TODAES*, 2025. DOI: 10.1145/3765905

13. "Self-Attention to Operator Learning-based 3D-IC Thermal Simulation (SAU-FNO)." arXiv:2510.15968, 2025.

14. "DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design." arXiv:2504.03955, 2025.

15. "Fast Thermal-Aware Chiplet Placement Assisted by Surrogate." arXiv:2504.03808, 2025.
