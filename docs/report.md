# A 3D-IC Thermal Surrogate Benchmark, and What It Reveals About Surrogate Evaluation

**Rajul Kabeer**  
*Manuscript in preparation*

> **Revision history.** This draft originally claimed FourierPINN was "the first PINN for
> the multi-layer 3D-IC stack" and "the first to treat TSV density as a continuous
> parametric input," with an *expected* MAE of < 2 K. Measurements on 2026-07-31 with
> `scripts/baselines.py` showed closed-form ridge regression already reaches spatial
> R² = 0.999 (geometry1) with detrended MAE 0.009 K — beating the < 2 K target before any
> network was trained — on a dataset whose median within-scenario spatial range was 1.10 K
> against a 68 K between-scenario range, and whose "continuous" TSV input took four
> discrete values. The paper below is the rewrite: a benchmark-and-negative-result paper,
> not a PINN-accuracy paper. Three follow-on passes are folded into the sections below
> rather than kept as a growing changelog:
> (2026-08-01) the dataset was regenerated on a physically grounded operating point — TDP
> budgets, cooling coupled to power density, sub-layer vertical resolution — which raised
> the median spatial gradient tenfold and fixed a double-counted `T_range` term in the PDE
> residual, yet ridge still solved every split (§9.3), which is the paper's central result;
> (2026-08-05) ground truth moved to 3D-ICE 4.0, closing two data-fidelity gaps (spatial
> TSV density, real Si/underfill layouts for the chiplet geometries) without changing that
> result; (2026-08-06) geometry2b/2c were removed as near-duplicates of geometry2a (§4).

---

## Abstract

Thermal analysis of three-dimensional integrated circuits (3D-ICs) is a bottleneck in
early-stage design exploration, and a growing body of work applies neural surrogates —
PINNs, Fourier neural operators, DeepONets — to accelerate it. We contribute an open
benchmark of 275 3D-ICE simulations across six package geometries, spanning single-die
mobile and server stacks, a dual-die 3D-TSV stack, and 2.5D/CoWoS-style chiplet
assemblies with up to six HBM stacks, together with the geometry definitions and
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
requires changing *what varies*, not the range over which it varies.

We then test that prediction directly. Replacing block-scalar power with a per-cell power
field — same TDP budget, same cooling, same geometry, only the spatial distribution of the
source made function-valued — leaves the linear model's field-level R² largely intact
(0.78–0.98) but collapses its ability to locate the hotspot, from 8–1442 µm to
4547–6044 µm on a 10 mm die, i.e. no better than chance. With block power the peak sits at
one of a few fixed positions and "predicting" it is memorising a short list; with a
continuous field it must be computed. This isolates where an operator surrogate can earn
its cost, and implies such work should be judged on **hotspot localisation rather than
field R²**, since field R² is dominated by the linear component that needs no network. We
release the dataset, baselines, and OOD split tooling so future surrogate claims can be
checked against a linear model before an architecture is credited.

---

## 1. Introduction

As chiplet integration and 3D stacking become mainstream packaging approaches, thermal design must keep pace with increasing power densities and more complex heat flow paths. Die stacking with TSV interconnects creates new thermal coupling mechanisms: each die heats the one above it, TSVs provide vertical thermal shortcuts, and the effective thermal resistance from junction to coolant is now distributed across multiple material interfaces. Evaluating even a single configuration requires solving the three-dimensional heat equation over a heterogeneous multi-layer domain.

Compact thermal simulators such as 3D-ICE [Sridhar et al., 2010] model this as a finite-difference RC-network on a structured Cartesian grid and can produce steady-state solutions in seconds. However, design exploration over power patterns, cooling intensities, and TSV densities requires thousands of such evaluations. Full parametric sweeps remain slow, and compact simulators have limited expressiveness — they cannot represent individual TSV columns natively, though 3D-ICE 4.0 [Zhu et al., 2025] now supports per-element material layouts, which this benchmark's ground truth uses (§4).

Machine learning surrogates offer a different trade-off: expensive offline training amortised over fast online inference. Prior work has applied convolutional neural networks [Zhang et al., 2025], graph neural networks [WarPGNN, 2026], and operator-learning approaches including FNO variants [Self-Attention U-Net FNO, 2025] and DeepONet [DeepOHeat-v1, 2025] to chip thermal prediction. Physics-informed neural networks (PINNs) add physics constraints — the heat equation and boundary conditions — as additional loss terms, which regularise training and improve generalisation beyond the training distribution. ThermPINN [Cheng et al., 2024] demonstrated PINNs for full-chip 2D thermal analysis.

What this literature does not generally do is check a proposed architecture against a
closed-form linear baseline before crediting it with having learned the physics. We did,
on our own benchmark, and it changed what the paper is about. Steady-state conduction
with fixed thermal conductivity is linear in the volumetric power sources and boundary
scalars; when a scenario is described by a handful of block powers plus a few boundary
values (the parameterisation used, explicitly or implicitly, by most published 3D-IC
thermal datasets we are aware of), the resulting solution manifold is low-dimensional and
close to linear by construction, and a per-point ridge regression — no training, no
GPU — reconstructs it almost exactly. This is not a claim that neural surrogates cannot
help with 3D-IC thermal analysis; it is a claim that most published evaluation setups
cannot currently tell you whether they do.

This paper makes the following contributions:

1. An open **six-geometry benchmark of 275 3D-ICE simulations** covering single-die mobile
   and server stacks, a dual-die 3D-TSV stack, and 2.5D/CoWoS chiplet assemblies with up
   to six HBM stacks — with the full generation pipeline.
2. **A demonstration that closed-form ridge regression solves this benchmark** (spatial
   R² = 0.999), and that the result holds under extrapolation to unseen power patterns,
   power magnitudes and ambient temperatures. Any neural architecture evaluated on data of
   this kind must be compared against a linear model before its capacity is credited.
3. **Spatially-detrended error as an evaluation metric.** We show raw MAE on 3D-IC thermal
   data is dominated by a per-scenario scalar offset — 88% of variance here is explained by
   the ambient input alone — and that detrending is required to measure spatial fidelity.
4. **Identification of hotspot localisation as the discriminating metric.** Replacing
   block-scalar power with a per-cell field — same budget, same cooling, same geometry —
   leaves field R² largely intact (0.78–0.98) while collapsing hotspot localisation from
   8–1442 µm to 4547–6044 µm on a 10 mm die. Field R² is dominated by the linear component
   that needs no network; hotspot localisation is not, and is also the quantity thermal
   design actually cares about.
5. **Evidence that the result is structural, not a parameter-range artefact.** A full
   regeneration on a physically grounded operating point raises the median spatial
   gradient from 1.10 K to 10.77 K and still leaves ridge at R² > 0.94 on every split.
   Configurations that *do* defeat the linear model turn out to be physically impossible.
   We give the conditions a non-degenerate 3D-IC thermal benchmark must satisfy.
6. Reference implementations of five surrogate families (PINN, FNO/WHNO/CNO-FNO, DeepONet,
   autoregressive z-layer operator, few-shot fine-tuning) with a shared explainability
   toolkit, released as infrastructure rather than as accuracy claims.

---

## 2. Related Work

**Compact thermal simulation.** 3D-ICE [Sridhar et al., 2010; updated 2021] is the standard open-source compact thermal solver for 3D-IC stacks, using finite differences on a structured grid. HotSpot [Huang et al., 2006] targets 2D die-level analysis with a lumped-resistance model. 3D-ICE 4.0 [Zhu et al., 2025] adds per-element material layouts and anisotropic conductivity, which is what makes this benchmark's spatially varying TSV density and chiplet underfill layouts (§4) possible as ground truth rather than only as a PDE-loss assumption. Full FEM tools (COMSOL, Ansys) resolve individual TSV cylinders but require hours per simulation.

**Neural thermal surrogates and prior-art overlap.** CNN-based surrogates for HBM chiplet stacks [arXiv 2503.04049, 2025] achieve good accuracy on multi-layer configurations but without physics constraints, limiting generalisation. The Self-Attention U-Net FNO [arXiv 2510.15968, 2025] reports 842× speedup over FEM for 3D-IC thermal prediction using operator learning, and already validates on discontinuous thermal conductivity — the same lateral-heterogeneity mechanism §4 exercises. DeepOHeat-v1 [arXiv 2504.03955, 2025] applies DeepONet to 3D-IC thermal simulation and optimization, directly overlapping this repo's DeepONet reference implementation (§8+); we do not claim novelty for that architecture choice here. ThermPINN [IEEE, 2024] applies PINNs to 2D VLSI full-chip thermal, demonstrating 10³–10⁴× speedup over iterative solvers. MFIT [ACM TODAES, 2025] addresses multi-fidelity thermal modelling for 2.5D/3D chiplet architectures, a goal adjacent to but distinct from this paper's benchmark-and-evaluation focus.

**Physics-informed neural networks.** Raissi et al. [2019] introduced PINNs. Tancik et al. [2020] demonstrated that random Fourier feature encoding overcomes spectral bias in coordinate networks. Wang et al. [2021] introduced Neural Tangent Kernel-based adaptive loss weighting for PINNs. Cai et al. [2021] applied PINNs to 2D heat transfer problems.

**Explainability for neural surrogates.** Sensitivity analysis for PINNs is discussed in [arXiv 2301.02428]. Integrated Gradients [Sundararajan et al., 2017] provides input attribution with completeness guarantees. MC Dropout as Bayesian approximation for uncertainty quantification was proposed by Gal & Ghahramani [2016]. We are not aware of prior work applying this combination of methods to chip thermal surrogates specifically, though we make no strong claim here — see `docs/references.md` for the full prior-art audit behind this section.

**What this paper does not claim.** Lateral material heterogeneity in 3D-IC thermal modelling is settled prior art (3D-ICE 4.0 itself, and SAU-FNO's discontinuous-k validation above); we use it as a data-fidelity fix, not a contribution. The same applies to per-element TSV/underfill layouts. The contribution here is the benchmark, the linear-baseline result, and the hotspot-localisation metric (§1), not any individual architecture or geometry-modelling mechanism.

---

## 3. Problem Statement

We model steady-state heat conduction in a 3D-IC package stack governed by:

$$\nabla \cdot (k(T) \nabla T) + Q = 0 \quad \text{in } \Omega$$

with a convective boundary condition at the bottom surface (the `heat_sink` layer, `z=0`), matching 3D-ICE's own boundary-condition convention:

$$-k \frac{\partial T}{\partial z}\bigg|_{z=0} = h \left(T - T_{amb}\right)$$

and adiabatic conditions on all lateral faces and the top surface (nearest the die). The domain Ω is a rectangular cuboid covering the full package stack — heat sink, spreader, TIM layers, and one or two active silicon dies. The volumetric power source Q is non-zero only in active silicon layers within designated power blocks.

The thermal conductivity of silicon is temperature-dependent: $k_{Si}(T) = 148 \cdot (300/T)^{1.3}$ W/m·K [Glassbrenner & Slack, 1964], while copper and TIM layers use constant values.

The surrogate model $\hat{T}_\theta: \mathbb{R}^7 \to \mathbb{R}$ maps per-point inputs $(x, y, z, Q, h, T_{amb}, \phi_{TSV})$ to normalised temperature, where the first three are normalised spatial coordinates, Q is normalised volumetric power density, h and $T_{amb}$ are normalised scenario-level scalars, and $\phi_{TSV}$ is the TSV area fraction. As of the 2026-08-05 regeneration, $\phi_{TSV}$ is a spatial field within TSV-bearing layers (`src/scenario/tsv_maps.py`) rather than one scalar per layer. As of 2026-08-06, the field is exported per-point in every `.npz` file and consumed as a real per-cell input channel by the FNO reference implementations (`src/fno/model.py`); the FourierPINN (§5) and the DeepONet/ARO reference implementations still receive only the field's per-scenario mean (an improvement over the previous geometry-constant scalar, but not full per-point conditioning) — closing that remaining gap for the point-based architectures is future work.

---

## 4. Benchmark Geometries

Six geometries span the design space from a mobile-class single die to server-class dies,
a 3D-stacked configuration with TSV arrays, and 2.5D/CoWoS chiplet assemblies. All layers are
modelled with homogenised material properties. TSV regions use the arithmetic-mean effective
conductivity $k_{eff} = (1-\phi) k_{Si} + \phi k_{Cu}$, consistent with the 3D-ICE
ground-truth simulator.

| Geometry | Type | Die size | Layers | Files | Notes |
|---|---|---|---|---|---|
| geometry1 | 2D stack | 10 × 10 mm | 6 | 45 | Mobile/desktop single die |
| geometry2a | 3D stack | 8 × 8 mm | 10 | 30 | TSV density 3% (spatial field) |
| geometry3 | 2D stack | 25 × 25 mm | 6 | 45 | Server-class, 8 core clusters |
| geometry4 | 2.5D stack | 25 × 14 mm | 6 | 45 | Two chiplets on interposer |
| geometry5 | 2.5D stack | 25 × 14 mm | 11 | 55 | CoWoS: compute + HBM stack |
| geometry6 | 2.5D stack | 42 × 14 mm | 11 | 55 | CoWoS: 6 HBM stacks |

Total: **275 simulations**. Two TSV-density variants of geometry2a (geometry2b at 5%,
geometry2c at 10%) were removed 2026-08-06: their ridge-regression baselines were
bit-identical to geometry2a's to three decimal places (spatial R²=0.991, MAE=2.207 K, all
three), and no surrogate in §5/§8+ conditions on TSV density beyond a scalar mean (§3), so
the three geometries carried no distinguishable signal from each other. TSV density
variation is still exercised as a spatial field within geometry2a itself.

Each geometry has 6 layers (geometry1, geometry3) or 10 layers (geometry2a): heat sink (Cu, 5000 µm) → TIM (100 µm) → spreader (Cu) → TIM (100 µm) → active die(s) (Si, 50–200 µm) → TIM2 (50 µm). TIM2 as the topmost layer ensures the convective BC is applied at the die-to-package interface with correct contact resistance.

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

**Dataset.** Each geometry has 25–50 training and 5 test scenarios (275 files total). Power comes from a package TDP budget (30 W mobile 3D stack to 700 W six-HBM accelerator), of which the modelled blocks receive 65% — the balance representing cache, IO and uncore, which are not modelled as separate sources. A pattern's base level selects a workload fraction of that budget, capped by an absolute silicon ceiling of 300 W/cm². Cooling is required to be adequate for the resulting power density (165 W/m²·K per W/cm²), so HTC spans 2000–50000 W/m²·K, and ambient spans 25–45 °C. Ground truth is the 3D-ICE Emulator under WSL2.

Optionally (`--power-map`), block scalars are replaced by a per-cell power field at the lateral mesh resolution; see §9.4 for why this matters and what it changes.

Note that the shipped `*_test_*` files are interpolation points inside the training sweep; extrapolation splits are generated separately by `scripts/make_ood_split.py` (§9.1).

*(The retired regime — 0.1–20 W/cm² absolute, HTC 500–10,000, ambient 25–85 °C — is described in §9.2, where its degeneracy is quantified.)*

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

Across all 335 live simulations in the original (2026-07-31) regime — see §9.3 for the
current dataset's numbers, which supersede this section's:

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
resolved by sub-layer discretisation. It was regenerated again (2026-08-05) on 3D-ICE
4.0 with per-cell power maps, spatial TSV density, and real chiplet underfill layouts
active (§4, `assumptions.md` §6.1), and once more (2026-08-06) after removing
geometry2b/2c. The numbers below are current as of that last regeneration.

**Dataset (275 simulations, all real 3D-ICE 4.0):**

| Geometry | n | z-nodes | pts/file | median ΔT | max ΔT | median T | peak T |
|---|---|---|---|---|---|---|---|
| geometry1 | 45 | 10 | 100,000 | 19.14 K | 63.96 K | 50.0 °C | 125.9 °C |
| geometry2a | 30 | 14 | 89,600 | 5.02 K | 22.06 K | 46.5 °C | 69.2 °C |
| geometry3 | 45 | 11 | 110,000 | 7.38 K | 25.03 K | 42.2 °C | 79.1 °C |
| geometry4 | 45 | 10 | 56,000 | 27.95 K | 93.29 K | 48.9 °C | 156.1 °C |
| geometry5 | 55 | 15 | 84,000 | 11.46 K | 61.29 K | 54.4 °C | 126.6 °C |
| geometry6 | 55 | 15 | 141,120 | 14.08 K | 63.66 K | 55.9 °C | 124.3 °C |

Median within-scenario spatial ΔT across the current dataset is **12.30 K** (max anywhere
93.29 K), against **1.10 K** in the original degenerate regime (§9.2) — an order of
magnitude, consistent with (though not identical to) the 10.77 K reported for the
2026-08-01 intermediate regeneration; the further shift reflects the cumulative effect of
per-cell power (which caps power pointwise rather than per block, moderating some peaks
while sharpening others) and the 4.0 ground-truth change. 5 of 275 scenarios exceed
125 °C, all `extreme_hotspot` stress cases.

**Baselines on the current data (geometry1, re-measured 2026-08-06 post-4.0/post-removal):**

| Split | ridge det.MAE | ridge spatial R² |
|---|---|---|
| in-distribution | 0.407 K | 0.970 |
| pattern OOD | 0.801 K | 0.966 |
| power OOD | 1.148 K | 0.970 |
| HTC OOD | 1.101 K | 0.687 |
| ambient OOD | 1.021 K | 0.950 |

The numbers moved from the 2026-08-01 measurement (0.223–0.331 K det.MAE, R² 0.964–0.998)
but the qualitative story did not: ridge still solves every split, and HTC extrapolation
is still the comparatively weak axis (R²=0.687, versus ≥0.95 on the other three) —
consistent with §1/§10's account of why 1/h extrapolation is the harder case for a linear
model.

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

### 9.4 Per-cell power: the input dimensionality, not the parameter range

§9.3 concluded that no choice of parameter *ranges* makes this benchmark
non-linear, because the source was described by ~8 scalars. That diagnosis was
testable: replacing block scalars with a per-cell power field should change the
result, and it does.

`--power-map mixed` gives each scenario a spatially varying power field at the
lateral mesh resolution (10,000 cells for geometry1) instead of 4 block scalars.
The physics is untouched — same TDP budget, same silicon density ceiling, same
cooling rule — only the *distribution* of power varies. Effective rank of a
collection of N maps:

| | N=40 | N=120 | N=200 |
|---|---|---|---|
| per-cell maps | 37.2 | 104.9 | 163.8 |
| block scalars | 4.0 | 4.0 | 4.0 |

Simulation cost is unchanged: 3D-ICE's solve time is set by the mesh, not the
floorplan (13.3 s for 4 floorplan elements, 13.5 s for 10,000).

**Ridge on geometry1, block-scalar vs per-cell**, in both cases given the leading
20 principal components of the full power field so the linear model sees the real
source rather than block averages:

| OOD axis | block-scalar R² | per-cell R² | block hotspot err | per-cell hotspot err |
|---|---|---|---|---|
| power | 1.000 | 0.978 | 1226 µm | **4547 µm** |
| HTC | 0.999 | 0.783 | 1442 µm | **6044 µm** |
| pattern | 0.999 | 0.982 | 8 µm | **5044 µm** |

Two things change, and the second matters more than the first.

Field-level R² degrades only moderately (worst on the HTC axis, 0.999 → 0.783).
A linear model still reconstructs most of the field's variance, and we do not
claim otherwise: the bulk of a thermal field really is a smooth response to
total power and boundary conditions, which is genuinely linear.

**Hotspot localisation collapses.** Ridge locates the hotspot to within 8–1442 µm
on block-scalar data and 4547–6044 µm on per-cell data — roughly half the die
width, i.e. no better than chance. The reason is structural: with block power the
hotspot always sits at one of a few fixed block positions, so "predicting" its
location is memorising a short list. With a per-cell field the peak moves
continuously, and locating it requires actually applying the Green's function
rather than interpolating between remembered configurations.

That is the opening for an operator surrogate, and it is the engineering quantity
that matters — thermal design cares where the hot spot is, not only what the mean
field looks like. It also sharpens what any surrogate here must be judged on:
**hotspot location error, not field R²**, because field R² is dominated by the
linear part that needs no network.

### 9.5 Regime fix and its effect (intermediate result, superseded by §9.3)

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

### 9.6 A physics bug found while building these tests

`pde_residual` applied the temperature range $(T_{max}-T_{min})$ twice — once converting
normalised gradients to physical ones, and again as a trailing factor on the divergence.
The conduction term was therefore inflated by roughly 65–100×, so the PDE loss effectively
enforced $\nabla\cdot(k\nabla T) = 0$ while ignoring the source $Q$. No trained result is
affected because none exists, but every physics-loss run before 2026-07-31 optimised the
wrong objective. It is now pinned by a manufactured-solution test that compares the residual
against the closed-form $k \cdot 2c \cdot T_{range}/L_z^2$.

### 9.7 Neural results

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

**What a discriminative benchmark needs.** Four changes were identified from §9; three have
since been implemented and measured, which is how their relative importance became clear.

1. **Higher power density** — *done* (§9.3). Raised the median within-scenario spatial
   gradient from 1.10 K to 10.77 K and made $k(T)$ active (silicon conductivity now varies
   2.9% within a die, against 0.24% before). **It did not defeat the linear model**: ridge
   still solves every split at R² > 0.94. Necessary for physical realism, insufficient for
   discriminability.
2. **Per-cell power maps** — *done* (§9.4). This is the change that mattered. It leaves
   field-level R² largely intact but collapses hotspot localisation from 8–1442 µm to
   4547–6044 µm, i.e. to chance. Input dimensionality, not parameter range, was the
   binding constraint.
3. **Spatial material variation within a fixed floorplan** — *partially done* (2026-08-05).
   TSV density is now a spatial field rather than one scalar per layer, and the
   geometry4/5/6 chiplet underfill gap is now a real Si/underfill layout matching what the
   PDE loss always assumed (`assumptions.md` §6.1) rather than a train/target
   inconsistency. Both are correctness fixes to a small (~2% of field range) effect, not
   attempts to defeat linearity, and the ridge result is unchanged (§9.3). **Full variable
   floorplans across scenarios remain not done** and are still expected to matter for a
   different reason than (2): a source that moves shape, not just intensity, still keeps
   the map linear in $Q$ — what would make the operator itself scenario-dependent is
   changing the coefficients (k(x,y), boundary shape), which items 3 above only touches at
   a ~2%-of-range scale so far.
4. **Extrapolation splits by default** — *done*; `scripts/make_ood_split.py`. We no longer
   claim $1/h$ is the axis where linear models fail: that held only for an intermediate
   dataset containing physically impossible power/cooling combinations (§9.3).
5. **Advective cooling (microchannel/pin-fin)** — *mechanism built and validated
   2026-08-06, not yet integrated into the dataset*. Every fix before this one (including
   per-cell power) operates within steady-state conduction with a fixed convective boundary
   coefficient, which is exactly the regime in which the governing equation is linear in
   its sources — the reason ridge wins at all. `ICESimulator` now supports
   `cooling_mode='microchannel_2rm'` (3D-ICE 4.0's `microchannel 2rm` coolant model,
   grammar confirmed against the simulator's own test suite), which introduces advection
   along a flow direction; unlike every fix above, that is not linear in the boundary data
   the way a fixed HTC is. Validated stable against the real binary at geometry6's
   ~455 W core-fraction TDP ceiling (peak 167.4 °C). Not yet wired into `main.py`/
   `NPZExporter` — the coolant stack element produces no Tmap output, and the coords/export
   pipeline does not yet know to skip it — so no microchannel-cooled scenarios exist in the
   current 275-scenario dataset. This remains the first candidate change that could alter
   the paper's central finding rather than refine the dataset around it, once integrated
   and swept as a scenario axis.

Transient simulation would add a further nonlinear axis; the present dataset is steady-state
only.

**Threats to validity.** Ground truth is 3D-ICE, itself a compact RC-network approximation;
HotSpot cross-validation covers geometry1 only and disagrees by ~15%, and no FEM spot-check
has been performed. Ridge's advantage is measured on scenario counts of 20–50; with far more
scenarios and a richer power parameterisation the ranking could change. The spatial TSV
field is now exported per-point and consumed as a real channel by the FNO family (2026-08-06,
§3); FourierPINN, DeepONet and ARO still receive only its scenario mean, so for those three
it remains a partial confounder — real variance in the target with only a coarse summary of
its cause available as input — until per-point conditioning is added there too.

**Deferred.** The originally planned discussion — PDE-residual/error correlation, IG
attribution plausibility, MC Dropout calibration — requires trained models and remains
open. (TSV generalisation across geometry2a/b/c is no longer planned: geometry2b/2c were
removed 2026-08-06, §4.) MC Dropout is already known to be poorly calibrated here: the
predictive std measured during explainability validation was 8.6–13 K on fields whose
spatial std is ~0.3 K, i.e. roughly 30× the signal.

---

## 11. Conclusion

We release a six-geometry, 275-simulation 3D-IC thermal benchmark with its full generation
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
