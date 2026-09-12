# A 3D-IC Thermal Surrogate Benchmark, and What It Reveals About Surrogate Evaluation

**Rajul Kabeer**  
*Manuscript in preparation*

> **Revision history.** This draft originally claimed FourierPINN was "the first PINN for
> the multi-layer 3D-IC stack" and "the first to treat TSV density as a continuous
> parametric input," with an *expected* MAE of < 2 K. Measurements on 2026-07-31 with
> `scripts/baselines.py` showed closed-form ridge regression already reaches spatial
> R² = 0.999 (geometry1) with detrended MAE 0.009 K — beating the < 2 K target before any
> network was trained (that specific figure was itself later superseded by a dataset
> correction; §9.1a carries the current numbers) — on a dataset whose median within-scenario spatial range was 1.10 K
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
requiring no GPU and no training — reconstructs the spatial temperature field at
R² = 0.89–0.99 across all six geometries**, and extrapolates to unseen power patterns,
unseen power magnitudes, and unseen ambient temperatures at R² > 0.9. (An earlier draft
reported R² = 0.999 from a superseded dataset; that figure is retracted — see §9.1a for the
current measurement on every geometry.) This is not a defect of any particular architecture but a
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
field R²**, since field R² is dominated by the linear component that needs no network.

Finally, we show the problem is fixable and fix it. Randomising chiplet *placement* per
scenario — so that the thermal operator itself varies rather than only its inputs — moves
the benchmark out of the linear regime: across three re-laid-out geometries, cross-validated
over 45 scenarios each, ridge becomes the **worst** of the four baselines, and on the densest
package its median spatial R² is negative and its detrended error exceeds the signal it is
predicting. The obvious first attempt — translating each chiplet rigidly — does *not* work
despite achieving greater source movement than an external benchmark by support-overlap
measures, and we report why. We also find that **"linearly solvable" is a property of the
input representation rather than of the dataset**: the same files admit a linear fit at
R² 0.96 from the full per-cell power field and fail at R² −0.67 from the compact
block-summary vector that surrogate models are conventionally given. We release the dataset,
baselines, cross-validation and linearity-audit tooling so future surrogate claims can be
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

> **Revised 2026-09-11.** The previous list was written when ridge scored R² 0.999 and
> the project had no dataset a linear model failed on. Both have changed: the 0.999 figure
> was a stale pre-regime-fix number (§9.1a re-measures all six geometries at 0.89–0.99),
> and §9.15b now has three datasets where ridge is the *worst* model tested. Contributions
> 2 and 4 are narrowed accordingly; 7–9 are new.

1. An open **benchmark of 275 3D-ICE simulations across six package geometries** covering
   single-die mobile and server stacks, a dual-die 3D-TSV stack, and 2.5D/CoWoS chiplet
   assemblies with up to six HBM stacks — with the full generation pipeline. A further
   **310 solves** make up the placement-varied extensions: 180 layout-randomised (four
   geometries × 45, §9.15b), 90 rigid-translation (§9.15) and a 40-scenario geometry7
   pilot. **585 real 3D-ICE solves in total**; counts verified against the files on disk
   rather than quoted from an earlier draft.
2. **A demonstration that closed-form ridge regression solves the fixed-placement
   benchmark** at spatial R² 0.89–0.99 with no training and no GPU, holding under
   extrapolation to unseen power patterns, magnitudes and ambient temperatures. Ridge's
   margin over a 3-nearest-neighbour lookup is thin and sometimes negative (§9.1a), which
   sharpens rather than weakens the point: if kNN is competitive, the benchmark is not
   measuring operator learning.
3. **Spatially-detrended error as an evaluation metric.** Raw MAE on 3D-IC thermal data is
   dominated by a per-scenario scalar offset — 88% of variance here is explained by the
   ambient input alone — so detrending is required to measure spatial fidelity at all.
4. **Identification of hotspot localisation as a discriminating metric.** Field R² and
   hotspot localisation come apart: a model can hold field R² at 0.78–0.98 while locating
   the peak no better than chance, so field R² alone cannot certify a thermal surrogate.
   Two caveats are stated rather than buried. (a) The specific per-cell contrast in §9.4 was
   measured on the superseded dataset and overstates the effect; it is flagged there and
   needs re-measurement. (b) Under 5-fold CV ridge still beat both FNO configurations we
   trained on *every* hotspot metric (§9.12d). This is therefore a metric that separates
   tasks, not one on which neural operators have been shown to win.
5. **Evidence that the fixed-placement result is structural, not a parameter-range
   artefact.** Regeneration on a physically grounded operating point raises the median
   spatial gradient from 1.10 K to 10.77 K and still leaves ridge above R² 0.94 on every
   split; configurations that *do* defeat the linear model turn out to be physically
   impossible. We state the conditions a non-degenerate 3D-IC thermal benchmark must meet.
6. **A supplied missing baseline for an external benchmark.** On IC-ThermBench
   (arXiv:2608.23977) — which reports only neural models — the strongest linear baseline is
   3.4–4.4× worse than Therm-FM, so that benchmark is *not* linear-solvable; we decompose
   how much of the gap is attributable to layout conditioning versus material variation
   (§9.13, §9.14). This is the control the benchmark's own paper omits.
7. **A benchmark fix that works, with the failed attempt reported.** Randomising chiplet
   placement moves the dataset out of the linear regime: on three shelf-layout geometries
   ridge becomes the *worst* model tested, and on geometry6 its median spatial R² is
   negative and its detrended error exceeds the signal (§9.15b). Rigid translation — the
   obvious first attempt, and one that achieves better source movement than IC-ThermBench
   by support-overlap IoU — does *not* work, and we report why (§9.15).
8. **The finding that "linear-solvable" is a property of the input representation, not of
   the dataset** (§9.15c). The same 45 files score linear R² 0.962 from the full per-cell
   power field and −0.667 from the compact block-summary vector that surrogate papers
   actually consume. Any claim that a benchmark is or is not linear-solvable is
   ill-posed without naming the representation — including claims made earlier in this
   paper.
9. **A protocol for auditing surrogate benchmarks, calibrated across PDE families and
   transferred to a second discipline.** The method — fix the protocol, supply the missing
   non-neural baseline, report what survives — is packaged as `scripts/baselines.py`,
   `scripts/layout_cv.py` and `scripts/benchmark_linearity_audit.py`. The linearity
   diagnostic is calibrated against the canonical operator-learning benchmarks (PDEBench
   Darcy, Burgers, Navier–Stokes, §9.16): it correctly places nonlinear Burgers at the
   discriminative extreme, shows our *fixed* benchmark was the most linearly-solvable dataset
   measured, and shows the layout fix made ours harder than Darcy at matched sample size and
   capacity. It also found that **time-evolution PDE benchmarks omit the persistence
   baseline** — on PDEBench Navier–Stokes a linear fit scores 0.958 while assuming nothing
   moved scores 0.9997, a gap the conventional mean-field baseline misses entirely (§9.16b).
   That is a fourth instance of this paper's structural finding, in the most-used benchmark
   family in the field. The method was separately **run end-to-end on a second discipline**
   (molecular property prediction, `molprop/`, 12 dataset×split blocks on MoleculeNet). It
   reproduces there: under matched tuning budget and reported dispersion, a tuned logistic
   regression on fingerprints is **not separable** from tuned gradient-boosted trees on three
   of the four classification cells, and is the best model on BACE/random; our tuned XGBoost beats
   every published model on ESOL/random, including by a margin larger than the spread
   separating four of six published architectures. The audit also found that MoleculeNet's
   unreported Bemis-Murcko **tie-break convention** shifts results by up to 0.21 AUC in a
   dataset-dependent direction — enough to make published "scaffold split" comparisons
   unresolvable where the convention is not stated. So the thermal result is not a quirk of
   thermal data; it is a property of how surrogate benchmarks in several fields are
   evaluated. Full results and caveats in `molprop/README.md` §7.
10. Reference implementations of five surrogate families (PINN, FNO/WHNO/CNO-FNO, DeepONet,
   autoregressive z-layer operator, few-shot fine-tuning) with a shared explainability
   toolkit, released as infrastructure rather than as accuracy claims.

**Stated plainly for examiners:** contributions 2, 7 and 8 are negative or corrective
results. This paper's central claim is not that a new architecture is better; it is that the
standard evaluation setup in this domain cannot distinguish a trained neural operator from a
closed-form linear solve, that this is fixable, and that we fixed it and show the fix working.

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

### 9.1 Baselines (`scripts/baselines.py`)

`det.MAE` and `spat.R²` are computed after removing each field's mean, so they score spatial
structure only.

#### 9.1a Current measurement — all six geometries (re-measured 2026-09-09)

**Read this table, not the 2026-07-31 one below it.** §9.5 already noted that the
2026-07-31 numbers describe a superseded dataset and that only geometry1 had been
re-measured after the regime fix; §9.5's own geometry1 figures have since been superseded
too, by the 2026-08-06 dataset correction. Until now no single table in this report
carried current numbers for all six geometries. This one does. Regenerate with
`python scripts/baselines.py --geometry <geom> --data data/3d-ice` (`results/` is
gitignored, so the JSON artifacts are local-only):

| Geometry | signal σ (K) | baseline | det.MAE (K) | spatial R² | hotspot loc. err (µm) |
|---|---|---|---|---|---|
| geometry1 | 3.999 | mean | 2.308 | −0.540 | 5880 |
| geometry1 | | nearest-neighbour | 0.733 | 0.870 | 7734 |
| geometry1 | | kNN (k=3) | 0.465 | 0.962 | 7101 |
| geometry1 | | **ridge** | **0.407** | **0.970** | 7399 |
| geometry2a | 1.637 | kNN (k=3) | 0.166 | 0.985 | 5890 |
| geometry2a | | **ridge** | **0.145** | **0.986** | 5028 |
| geometry3 | 1.308 | **kNN (k=3)** | **0.212** | **0.931** | 14551 |
| geometry3 | | ridge | 0.218 | 0.925 | 18622 |
| geometry4 | 2.627 | kNN (k=3) | 0.425 | **0.903** | 16195 |
| geometry4 | | **ridge** | **0.320** | 0.891 | 8830 |
| geometry5 | 1.621 | kNN (k=3) | 0.242 | 0.947 | 16669 |
| geometry5 | | **ridge** | **0.173** | **0.959** | 13123 |
| geometry6 | 1.815 | kNN (k=3) | 0.245 | 0.954 | 24722 |
| geometry6 | | **ridge** | **0.228** | **0.956** | 31685 |

**Two things change materially versus the superseded table.**

1. **Ridge's margin is thin, not overwhelming.** On the current dataset ridge's spatial R²
   is 0.89–0.99, not 0.999, and its lead over plain kNN is small everywhere and *negative*
   on geometry3 (kNN 0.212 det.MAE / 0.931 R² vs ridge 0.218 / 0.925) and on geometry4's R²
   (kNN 0.903 vs ridge 0.891). The correct current statement is "a closed-form linear fit is
   competitive with, and usually marginally better than, k-nearest-neighbours" — not "a
   linear model solves the benchmark." The regime fix (§9.5) did what it was meant to do:
   the benchmark is no longer trivially linear-solvable.
2. **Hotspot localisation is a uniform failure, for every baseline, on every geometry.**
   Errors run 5.0–31.7 mm. geometry6's ridge error (31.7 mm) is most of that package's
   42 mm width — i.e. chance-level. No baseline localises hotspots at all. Note also that
   hotspot location is *not* degenerate in this dataset and so this is a real task being
   failed, not an artifact: geometry1 has 45 distinct hotspot cells across 45 scenarios and
   geometry6 has 54 across 55 (geometry7, by contrast, has only 6 across 40 — that one *is*
   partly degenerate, relevant when reading §9.11).

This sharpens rather than reverses §10's argument, and in the direction §9.1's own
2026-08-16 hotspot correction was already pointing: the linear baseline saturates the
metric this field usually reports (field-level R²) while **failing outright** the metric
chip-thermal work actually needs (where the hotspot is).

Two caveats on the hotspot column specifically, both resolved in §9.12c: these are
5-scenario numbers, and argmax-to-argmax distance is only a meaningful score where the
true peak is sharp (it is on geometry1, ~324 µm across; it is not on geometry6, ~9.4 mm).
§9.12c re-measures hotspot quality under 5-fold CV (45 and 55 scenarios) with metrics that
survive multi-modal fields, and those numbers should be preferred over this column.

#### 9.1b Superseded table (measured 2026-07-31, retained for history)

Kept because §9.2/§9.5 and the revision history above refer to it. These numbers describe
the pre-regime-fix, pre-2026-08-06-correction dataset and must not be quoted as current.

**In-distribution (shipped `*_test_*` split):**

| Geometry | Baseline | MAE (K) | det.MAE (K) | spatial R² | hotspot loc. err (µm) |
|---|---|---|---|---|---|
| geometry1 | mean | 18.629 | 0.270 | 0.586 | 1470.8 |
| geometry1 | nearest-neighbour | 9.119 | 0.138 | 0.466 | **0.0** |
| geometry1 | knn (k=3) | 8.765 | 0.141 | 0.876 | 2048.5 |
| geometry1 | **ridge** | **4.110** | **0.009** | **0.999** | 2080.2 |
| geometry6 | mean | 10.296 | 0.244 | −3.656 | 9454.0 |
| geometry6 | nearest-neighbour | 10.113 | 0.059 | 0.792 | **329.3** |
| geometry6 | knn (k=3) | 3.797 | 0.090 | −0.621 | 2311.3 |
| geometry6 | **ridge** | **0.616** | **0.020** | **0.937** | 1276.1 |

Mean spatial std of the true test fields is 0.506 K (geometry1) and 0.261 K (geometry6) —
this is the entire signal a surrogate exists to predict. Ridge captures it to 0.009 K.

**The hotspot column added 2026-08-16, and it complicates the story above in a way worth
stating plainly.** Earlier versions of this table reported only MAE, detrended MAE and
spatial R² — the three metrics on which ridge dominates — while the hotspot-localisation
column sat unreported in the saved baseline artifacts (`results/baselines_*.json`). On
that metric **ridge loses to plain nearest-neighbour on both geometries**, by 2080 µm vs
0 µm on geometry1 and 1276 µm vs 329 µm on geometry6. Omitting it was not deliberate, but
it was an instance of exactly the selective-metric reporting §10 criticises in the wider
literature, appearing in this paper's own headline table; it is corrected here rather than
quietly fixed.

This does not overturn the paper's finding — it sharpens it. Ridge is a field-reconstruction
specialist and a hotspot-localisation underperformer, and §9.4 shows the localisation
failure becomes total (chance-level) once power is specified per cell. The correct reading
is therefore not "a linear model solves 3D-IC thermal prediction" but the narrower and
more useful **"a linear model already saturates the metric this field usually reports,
while failing the metric it should report."** That reframing is the actual contribution,
and it applies to the baseline as much as to any neural architecture.

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

> **Numbers in this section are inconsistent with §9.1a and need re-measurement
> (flagged 2026-09-11).** The block-scalar column below reports ridge locating the hotspot
> to **8–1442 µm** on geometry1. §9.1a, measured on the corrected dataset, puts geometry1
> block-scalar ridge at **7399 µm**. A gap of that size cannot be explained by the
> in-distribution/OOD split difference; it means this table was produced on the degenerate
> pre-2026-08-06 dataset, in which the peak sat at one of a handful of fixed positions and
> locating it was, as the text below itself says, "memorising a short list".
>
> **Consequence for the claim.** The *direction* of §9.4's argument survives — per-cell
> power is a genuinely harder input than block scalars, and §9.3's diagnosis that input
> dimensionality rather than parameter range is what matters is independently supported by
> §9.15b. But the specific contrast "8–1442 µm → 4547–6044 µm" overstates the collapse,
> because on the corrected dataset the block-scalar starting point is already ~7400 µm.
> The per-cell dataset used here is no longer on disk, so this could not be re-measured;
> regenerating it with `--power-map mixed` and re-running `scripts/hotspot_eval.py` under
> k-fold CV is the outstanding task. Until then, do not quote these figures.

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

**First checkpoint, 2026-08-09.** Baseline FNO (`--model fno`, `--cpu-fast` preset:
channels=16, modes=(8,8,6), blocks=3, 296k params), 100 epochs, CPU-only, geometry1, seed
42. This run exists to validate the training/eval pipeline (data loading, loss, checkpoint
saving) end to end for the first time in this project's history, not to make an accuracy
claim — reduced capacity and few epochs relative to a real training budget. Best
validation MAE occurred at **epoch 10** (4.110 K); training loss kept falling through
epoch 100 while validation MAE climbed to 11–12 K, a clear overfit on 5 validation
scenarios at this capacity.

Evaluated on the shipped test split, same detrended metrics as §9.1:

| | MAE (K) | det.MAE (K) | spatial R² | hotspot loc. err (µm) |
|---|---|---|---|---|
| ridge (§9.1, current data) | 1.820 | 0.299 | 0.983 | 5738 |
| **FNO baseline (this run)** | **4.110** | **1.857** | **0.279** | **5685** |

Ridge wins by a wide margin here — consistent with this paper's central finding, but not
yet a fair test of it: this run is capacity- and epoch-limited by design (a CPU pipeline
smoke test), not the CondFNO/CNO-FNO/SAU-FNO configurations this repo also implements,
and training stopped essentially at its best point at epoch 10 with 90 unproductive
epochs after. A properly resourced run (GPU, full capacity, early stopping at the actual
optimum) is required before this comparison says anything about the architecture's real
ceiling on this benchmark. Hotspot localisation error (5685 µm) is comparable to ridge's
(5738 µm) even at this undertrained point, which is at least consistent with §9.4's
finding that localisation — not field reconstruction — is where an operator has room to
compete, though five test scenarios is far too few to draw a conclusion from.

**Same run, throttled data (§9.8), 2026-08-09 — now a like-for-like comparison.**
Same `--cpu-fast`, plain-FNO config, trained on a 15-scenario throttled geometry1 pilot
(6/20 scenarios triggered). `NPZExporter` now also writes a per-cell `power_nominal`
field (recomputed from `throttle_nominal_power_blocks` when throttling fired, identical
to the delivered `power` field otherwise — mirrors how `tsv_frac` is exported, §3), and
`FNODataset` builds `Q_norm` from it. FNO is therefore posed the *same* "requested power
→ resolved temperature" problem ridge is scored on, not the easier "already-resolved
power → temperature" one an earlier pass here tested by mistake.

| | det.MAE (K) | spatial R² | hotspot loc. err (µm) |
|---|---|---|---|
| ridge, nominal power (§9.8) | **0.707** | **0.919** | **0** |
| **FNO baseline (this run), nominal power** | 1.795 | 0.551 | 5139 |

FNO row reproducible via `python scripts/eval_fno_throttled.py` → `results/fno_throttled_eval.json`
(added 2026-08-16 — this eval had only been run inline in-session before and left no
artifact, a gap caught during a documentation audit).

With the asymmetry removed, FNO trails ridge by a wide margin on every metric, including
hotspot localisation — the one place per-cell power previously gave an operator a path to
win (§9.4). Still a capacity- and epoch-limited CPU smoke test (296K-parameter plain FNO,
100 epochs, 15 training scenarios), not the CondFNO/CNO-FNO/SAU-FNO configurations this
repo also implements, so this is not evidence that no operator could win on throttled
data — only that this particular undertrained run doesn't.

### 9.8 Nonlinear regimes: throttling and microchannel (pilot measurements, 2026-08-09)

Two mechanisms in this benchmark are not confined to the fixed-source linear-conduction
regime everything above operates in (§10 elaborates why that regime is why ridge wins at
all): package-level thermal throttling (power as a function of the temperature being
solved for) and microchannel liquid cooling (coolant advection along a flow direction).
Both were built and validated against the real 3D-ICE 4.0 binary in an earlier pass; this
section reports the first pilot measurements of whether they actually change the paper's
central finding.

**Microchannel** (geometry6, 12 scenarios, coolant flow rate swept 80–280 mL/min):

| | spatial R² | hotspot loc. err (µm) |
|---|---|---|
| ridge | **0.906** | 780 |
| kNN (k=3) | 0.809 | **595** |
| nearest-neighbour | 0.736 | 616 |

0.906 is the lowest spatial R² measured for any geometry in this project (every
steady-state, fixed-cooling geometry in §9.3 scores ≥0.918), and this is the first time
in this project that ridge has lost to a simpler baseline on any metric — both kNN and
plain nearest-neighbour localise the hotspot more accurately than ridge here. The pilot
is small (8 train / 4 test); this is a signal, not a settled result.

**Throttling** (geometry1, 20 scenarios, 90°C threshold, 6/20 scenarios triggered): the
first measurement here surfaced a real methodology bug rather than a finding. Ridge
initially scored spatial R² = 0.989 on the throttled data — *higher* than the same
geometry without throttling (0.970, §9.1) — because the exported `block_power_*`
metadata holds the **delivered** (already-derated) power, handing ridge the closed
loop's resolved output as an input feature rather than asking it to represent the loop.
Fixed by exporting the pre-throttle `nominal_block_power_*` value and making
`scripts/baselines.py` prefer it whenever `throttle_enabled` is set. Corrected, across
two independent pilot draws (both 15 train / 5 test, 6/20 triggered — a fresh 3D-ICE
draw with different random scenarios and derate factors each time):

| | det.MAE (K) | spatial R² |
|---|---|---|
| ridge, no throttling (§9.1, current data) | 0.407 | 0.970 |
| ridge, throttled, nominal power (draw 1, derate 0.34–0.72) | 0.831 | 0.890 |
| ridge, throttled, nominal power (draw 2, derate 0.48–0.84, current data) | **0.707** | **0.919** |
| kNN, throttled, nominal power (draw 2) | 0.526 | 0.890 |

Both draws land below the no-throttling baseline (0.970) but the exact margin moves with
the draw (0.890 vs. 0.919) — real, if modest, degradation, not yet precise enough to
quote a single number without the caveat that a 20-scenario pilot has real draw-to-draw
variance. Not a rout, but the first clean evidence in this project that a genuinely
closed-loop nonlinearity measurably erodes ridge's advantage, in contrast to every
fixed-source mechanism added before it (TSV fields, underfill layouts, per-cell power),
none of which moved the baseline outside its usual 0.92–0.99 range.

A capacity-limited plain-FNO run on draw 2, now on a like-for-like nominal-power input
(§9.7's earlier delivered-vs-nominal caveat is resolved), trails ridge by a wide margin
(det.MAE 1.795 K vs. 0.707 K, spatial R² 0.551 vs. 0.919, §9.7) — consistent with, not yet
evidence against, the central finding. Both pilots are too small and too capacity/time-
limited to license a conclusion about whether a properly-resourced operator *should* beat
ridge here — they establish that the regime is worth continuing to test, which was the
open question this pass set out to answer.

### 9.9 Interface-property uncertainty exceeds the accuracy differences being optimised (2026-08-16)

Every result above compares models to each other while holding the simulator's material
and interface properties fixed at nominal values. That is the field's standard practice,
and it embeds an assumption worth testing: that those inputs are known well enough for
sub-Kelvin model differences to be meaningful.

We tested it directly. Holding power pattern, HTC and ambient **fixed**, we varied one
interface conductivity at a time across ranges documented in `assumptions.md` — thermal
grease k = 1–8 W/m·K (ordinary literature spread, and greases also degrade in service),
TIM1 indium k = 5–80 W/m·K (the pump-out lifecycle range), and the Cu-Cu hybrid-bond
layer k = 60–400 W/m·K (what we model, versus what real hardware achieves). 30 real 3D-ICE
solves on geometry5 at two operating points, since interface resistance only matters in
proportion to the heat flux crossing it (`scripts/gen_interface_uncertainty_pilot.py`;
results in `results/interface_uncertainty_{low,high}power.json`).

| interface | k range (W/m·K) | peak-T spread, low power (ΔT≈7 K) | peak-T spread, high power (237 W/cm²) | hotspot shift |
|---|---|---|---|---|
| thermal grease (`tim_sink`) | 1–8 | 5.23 K | **28.25 K** | 504 µm |
| TIM1 indium (`tim_top`) | 5–80 | 0.52 K | 4.32 K | 496 µm |
| hybrid bonding | 60–400 | 0.01 K | 0.02 K | 248 µm |

**The measured ridge-vs-FNO detrended-MAE gap is 1.09 K (§9.7).** Thermal-grease
conductivity uncertainty alone moves peak junction temperature by **28.25 K at a realistic
high-power operating point — roughly 26× that gap** — and by 5.23 K even in the low-power
control. The effect scales with heat flux as physics requires, which is a useful internal
check that this is a real thermal-resistance effect rather than a numerical artifact. It
also moves the hotspot *location* by 248–504 µm with power held constant, perturbing
precisely the metric §9.4 identifies as the one that discriminates surrogates.

The implication is not that surrogate accuracy is worthless, but that it is being reported
without the context needed to interpret it: **on this benchmark, a model-vs-model
difference of ~1 K sits well inside the spread induced by a single unreported input
property.** A surrogate paper claiming a 50% MSE reduction over another architecture is
resolving a quantity substantially smaller than its own input uncertainty, and neither
number is usually reported alongside the other. This is a stronger and more general form
of the linear-baseline result: even where a neural operator does beat ridge, the margin
may not be the largest source of error in the pipeline.

Incidentally, this also settles a modelling judgement made on argument rather than
measurement: `assumptions.md` §2.3 flags the hybrid-bonding layer as a known ~28×
overestimate of real Cu-Cu bond resistance but rates it low priority because the absolute
error should be small. Sweeping it 60→400 W/m·K moves peak T by 0.02 K, confirming that
call empirically.

**Caveats, and a novelty boundary stated explicitly.** One geometry, one scenario per
power level, one parameter varied at a time, and 3D-ICE is itself the ground truth — this
measures *model-input sensitivity*, not validated real-hardware variation. It establishes
that this benchmark's answer is highly sensitive to an input nobody varies or reports; it
does not establish the true physical spread on real silicon, which would require the
standardized, uncertainty-aware interface-property data that [Barua, Udoy & Aziz,
arXiv:2604.03290] identify as missing from the field. Nor is the underlying *technique*
new: Monte Carlo propagation of TIM/interface-resistance uncertainty through a package
thermal model is established practice (Electronics Cooling, 2018), and a 2026 3.5D-package
study runs N=2,000-trial Monte Carlo over process variation for a related purpose
[arXiv:2606.26176]. What we add is using the resulting spread as a reference line against
neural-surrogate accuracy claims specifically — a search of the current literature found
no paper doing that comparison.

**Extended across geometries, and a self-caught error in the interaction test.** Repeating
the sweep on geometry1, geometry4 and geometry6 (98 further solves, 0 failures) at each
geometry's own high-power operating point gives a larger and more sobering picture:

| geometry | interface | k range (W/m·K) | peak-T spread |
|---|---|---|---|
| geometry1 | TIM1 (`tim_top`) | 5–80 | 40.62 K |
| **geometry4** | **die-side grease (`tim_die`)** | **1–8** | **75.98 K** |
| geometry6 | sink-side grease (`tim_sink`) | 1–8 | 12.01 K |

geometry4's 75.98 K — a physically sane, monotonic function of k, checked by hand — is now
the largest single-interface spread measured in this project, roughly 70× the reference
gap. geometry6's smaller spread (12.01 K) reflects its much larger die area spreading heat
further at the same power density, not a weaker mechanism there — sensitivity is
geometry-dependent, not a fixed property of "this benchmark."

We also ran a 3×3 joint sweep of two interfaces on geometry6 to test whether uncertainties
compound super- or sub-additively, and caught a real bug in our own analysis before
trusting the answer: the first pass identified the "both at nominal" reference point by
checking whether the two swept values were numerically equal to each other, which is never
true when the two layers' ranges don't overlap — it silently selected an arbitrary grid
corner instead, and would have reported a false **9/9 "super-additive"** finding. Corrected
by locating the true double-nominal point explicitly: **0/9 grid points are
super-additive** — the two effects sum linearly to within 0.01 K at every point tested.
That is the more useful result of the two: at this operating point, treating these
uncertainties independently is adequate, not an oversimplification. Tested at one
(lower-power) point on one geometry only; untested at higher power, where the individual
effects are much larger.

**Is the 75.98 K number worst-case-only? No — it degrades gracefully (2026-08-17).**
`scripts/gen_interface_uncertainty_pilot.py`'s new `--power median` option repeats the
geometry4 sweep at the middle-ranked-by-power `hotspot` scenario instead of the highest:

| operating point | tim_die spread | tim_sink spread |
|---|---|---|
| highest power (original) | 75.98 K | 25.65 K |
| median power | 45.55 K | 15.19 K |

Roughly 60% of the worst-case effect survives at a typical (not extreme) operating point —
still ~40× the reference architecture-comparison gap. The 1 W/m·K endpoint contributes most
of the drop between the two: at k=8 W/m·K the two operating points converge to nearly the
same peak temperature (67.04 °C median vs. 67.01 °C highest), because a well-coupled TIM
lets heat spread efficiently regardless of local block power; at k=1 W/m·K they diverge
sharply (112.6 °C vs. 143.0 °C), because a poorly-coupled TIM traps heat locally, where the
*local* block power (not the chip's total power) sets the peak. That is itself informative:
the size of this effect depends on which interface state you're in, not just how hot the
chip runs overall.

**Is the sensitivity actually linear, once expressed in the right variable? Yes
(`scripts/analyze_interface_linearity.py`, 2026-08-17).** Peak-T vs. k directly is a
mediocre linear fit (R² 0.58–0.91 across every layer/geometry tested) — but a 100 µm TIM
layer contributes a series thermal resistance R = t/k, so for fixed heat flux the physics
predicts T linear in **1/k**, not k. Refitting against 1/k confirms it: R² ≥ 0.996 on every
sweep with a spread above ~2 K (below that, the effect is inside solver-precision noise and
the fit is meaningless either way — geometry1's `tim2` and geometry6's `hybrid_bonding`
land there). This is the same finding the rest of this benchmark keeps producing, extended
to a new axis: steady-state conduction is linear in whatever variable actually enters the
heat-flow equation linearly. `scripts/baselines.py`'s ridge baseline already exploits
exactly this for the convective boundary (`feature_vector` includes `1/htc` beside `htc`,
specifically because 1/h enters T linearly); it does not yet do the same for TIM k for
*any* geometry — `feature_vector` has no interface-conductivity feature at all, even for
geometry5/6 whose own training sets already vary `tim_top`/`tim_sink` internally
(`assumptions.md` §2.2). A `1/k_tim` feature, mirroring the existing `1/htc` term, should
recover this axis for ridge as cleanly as it already recovers HTC extrapolation.

**Tested directly (`scripts/test_ridge_tim_k_feature.py`, 2026-08-17).** Leave-one-out
ridge on the 10 geometry4/`tim_die` sweep scenarios (5 k-values × {high-power,
median-power}), comparing three feature sets:

| ridge features | mean det.MAE | mean spatial R² |
|---|---|---|
| blind to k (current `baselines.py` behaviour) | 2.063 K | 0.754 |
| + raw k | 1.247 K | 0.895 |
| **+ 1/k** | **0.683 K** | **0.976** |

Monotonic in the predicted direction, and the physically-motivated feature wins outright:
adding raw k roughly halves blind ridge's error; adding 1/k instead brings it within reach
of this benchmark's ordinary in-distribution numbers (§9.1: ~0.2–1 K det.MAE, R²>0.9) even
under leave-one-out evaluation on a 10-point sample. The one hard case for every feature
set is k=1 (high-power) — R²=0.953 even with 1/k, the worst of the ten — which is the
sweep's extreme endpoint and therefore pure extrapolation under leave-one-out, not
interpolation; performance is uniformly better in the sweep's interior. Confirms both
halves of the 1/k claim: giving ridge the correct feature recovers most of the gap, and
1/k specifically (not k) is what does it.

### 9.10 Leakage/temperature positive feedback: the sharpest test tried, and the linear result holds — with one exception (2026-08-17)

Throttling (§9.8) is *negative* feedback: hotter → less power → cooler, self-limiting,
and it degraded ridge only modestly. Subthreshold leakage is *positive* feedback: hotter →
more leakage → hotter, self-amplifying, with no steady state at all above a critical loop
gain. This is structurally the sharpest test of the paper's central claim tried so far,
since the map from requested to delivered power is not even guaranteed to be well-defined,
let alone linear.

The mechanism itself is not new — self-consistent leakage-temperature solving to predict
thermal runaway dates to Chen et al. [*ICCAD* 2006], and current simulators (ATSim3D,
arXiv:2601.11050, Jan 2026) build it in directly. What we add is asking whether a linear
baseline still solves the *convergent* portion of such a dataset, which we could not find
asked elsewhere. `src/scenario/leakage.py` implements a damped fixed-point iteration on
$P_{total}(T) = P_{dynamic} + P_{leak,ref} \cdot 2^{(T-T_{ref})/k_{double}}$, referenced to
a nominal junction temperature (85°C, not ambient — an early version made exactly that
error and was caught before any real solves, since referencing to ambient spuriously
amplifies every realistically hot scenario). Wired to mirror throttling's convention
exactly: delivered power exported as `block_power_*`, the original request preserved as
`nominal_block_power_*`, so ridge is asked the same question it was for throttling.

Twenty pilot scenarios on geometry1, leakage settings swept from benign to aggressive (the
ridge-accuracy result below is measured on this original 20; the pilot was later grown to
45 scenarios specifically to test the stability-classifier question further down this
section):

- **13 converged, 6 ran away, 1 neither** (still climbing at the iteration cap) — a 30%
  runaway rate at these settings.
- **On the 13 converged scenarios (9 train / 4 test, `scripts/baselines_leakage.py`, which
  excludes runaway scenarios before fitting — scoring against a 506,854°C numerical
  divergence tests whether ridge survives garbage, not the real question), ridge still
  wins**: detrended MAE 0.248 K, spatial R² **0.972** — matching the un-throttled baseline
  (0.970, §9.1) and beating the throttled result (0.919, §9.8). Positive feedback, when it
  settles at all, does not push the problem further from linear than it already sits.
- **The genuine finding is what a regression framing doesn't even attempt to answer**: 30%
  of scenarios in this pilot have no steady-state temperature field at all. That is not an
  accuracy question a surrogate can be scored on — it is closer to a stability/
  classification problem (will this operating point converge or run away?), and nothing in
  this benchmark's current metrics addresses it.
- **What actually drives that 30%, checked directly (`scripts/analyze_leakage_convergence.py`,
  2026-08-17): not the swept leakage parameters.** Each of the 5 (`leakage_fraction`,
  `k_double_c`) settings was paired with 4 different base power/HTC/pattern scenarios, and
  every single one of the 5 settings produced mixed outcomes across its 4 repeats (e.g.
  frac=0.15/k_double=25 converges 3/4 times, runs away once) — the loop-gain parameters
  this pilot was designed to sweep do not, by themselves, predict stability. Total nominal
  requested power does, almost cleanly: every converged scenario requests ≤271 W, every
  runaway/stalled one requests ≥271 W, with exactly one pattern-driven exception at that
  boundary (a `hotspot` pattern at 271 W runs away where a `gradient` pattern at the same
  271 W converges) — consistent with the leakage multiplier being driven by *peak* local
  temperature, not total power, so a concentrated hotspot crosses the runaway threshold at
  lower total power than a spread-out one. This reframes the 30% figure: it is not really
  "30% of leakage-parameter settings are unstable," it is "runaway is governed mainly by
  how hot the *nominal* (pre-leakage) solve already runs, refined by spatial
  concentration" — closer to a one-feature classification problem than a leakage-specific
  one.
- **That reframing checks out as an actual fitted classifier, not just an eyeballed
  threshold (`scripts/classify_leakage_convergence.py`, 2026-08-17) — with a self-caught
  small-sample reversal along the way.** Logistic regression on features known *before*
  the feedback loop runs (total nominal power, HTC, ambient temperature,
  `leakage_fraction`, `k_double_c`, and an ordinal pattern-concentration score), evaluated
  with leave-one-out CV (the only honest evaluation at this sample size). At the pilot's
  original n=21, a 2-feature model (power + pattern concentration) looked like a clean
  win: 21/21 (100%) LOO accuracy vs. 20/21 for power alone and 19/21 for the full
  6-feature model. That ranking **did not survive** growing the pilot to n=45
  (`gen_leakage_pilot.py --extra-train 25`, finished 2026-08-17): re-run, the **full
  6-feature model wins, 43/45 (95.6%)**, against 40/45 (88.9%) tied between the power-only
  and power+concentration models. The apparent 2-feature win at n=21 was a small-sample
  artifact, not a real effect — a useful reminder alongside the interaction-grid bug
  earlier in this section that even a "clean-looking" cross-validated result needs
  re-checking as data grows, not just a plausible mechanism. What *does* survive the
  larger sample: the full model's standardised coefficients still rank power (−1.97) and
  HTC (−1.20) well above `leakage_fraction` (−0.15) and `k_double_c` (+0.17) — the two
  parameters this pilot was designed around remain the *weakest* predictors of whether it
  converges at all, even though no small feature subset cleanly captures the whole
  relationship on its own.

**Caveats.** One geometry, a small converged sample from one 45-scenario pilot (originally
20; the ridge-accuracy figures above are measured on the original 20 for continuity), one
damping/gain schedule — the direction is unambiguous but the exact numbers would tighten
further with more data or a second geometry.

**Net effect on the paper's central claim: it survives, with a sharper boundary.** Every
mechanism tested that keeps the problem well-posed — throttling, per-cell power, TSV
fields, underfill layouts, and now leakage feedback when convergent — leaves ridge winning
or close to it. The one place this benchmark has produced genuine nonlinearity is where
the problem stops being well-posed at all (leakage runaway here; more mildly, the
microchannel pilot's hotspot-localisation loss to nearest-neighbour, §9.8) — not in any of
the smooth closed-loop mechanisms tried.

### 9.11 geometry7: a bridge-based CoWoS-L pilot — one more mechanism, ridge still wins (2026-08-18)

Every 2.5D geometry so far (4/5/6) models CoWoS-S: one monolithic silicon interposer.
`geometry7` instead models CoWoS-L: sparse high-k bridge/via islands (a composite
material, `lsi_bridge_via`, k=60 W/m·K) in an otherwise low-k organic-substrate field
(k=0.5 W/m·K) — the first passive layer in this project with genuine internal lateral
heterogeneity rather than uniform material. Full build/complexity notes and a bridge-
coverage bug caught and fixed before this data was usable are in `docs/compute.md`.

**Ridge/kNN baseline, real 3D-ICE data (`scripts/baselines.py --geometry geometry7`,
35 train / 5 test):**

| baseline | det.MAE (K) | spatial R² | hotspot loc. err (µm) |
|---|---|---|---|
| mean | 2.874 | 0.695 | 15,142 |
| nearest-neighbour | 0.674 | 0.986 | 5,614 |
| kNN (k=3) | 0.482 | 0.992 | 5,614 |
| **ridge** | **0.462** | **0.992** | 5,838 |

Ridge wins on the two metrics that matter (det.MAE, spatial R²), by a narrow margin over
kNN — consistent with every other mechanism tested in this project. The sharp
material-conductivity discontinuity at each bridge island's edge does not break linearity
here either. Hotspot localisation is poor for every model on this geometry (5,600-5,800
µm, the worst absolute numbers recorded in this project) but ridge does not lose ground
to the simpler baselines the way it did on the microchannel pilot (§9.8) — with only 5
test scenarios this is a low-confidence read, not a settled result.

**Material-uncertainty sweep** (same methodology as §9.9, one real operating point held
fixed — the pilot's own hottest scenario, uniform/htc=5000/t_amb=45°C — varying one
material constant at a time), applied to the two constants this geometry introduces and
which are explicitly documented as engineering estimates, not citations:

| material | swept range | peak-T spread | shape |
|---|---|---|---|
| `organic_substrate` (bare field) | 0.3-0.8 W/m·K (literature range) | 47.5 K | monotonic decrease |
| `lsi_bridge_via` (bridge islands) | 20-150 W/m·K (engineering range) | **2.0 K** | flat, no meaningful sensitivity |

**A correction, not a footnote.** The first version of this sweep (generated the same day
this project briefly defaulted 3D-ICE's stack-file `numofcores` directive to 8 for a
wall-clock speedup, §"3D-ICE solve cost" in `docs/compute.md`) reported a very different
and much more dramatic story: 53.8 K spread on `organic_substrate` with a non-monotonic
dip at k=0.5, and 219.1 K spread on `lsi_bridge_via` with what looked like a sharp
plateau-then-cliff at k=20. Both numbers, and the cliff, were wrong. Re-running the exact
same sweep after that `numofcores` default was reverted (SuperLU_MT's multi-threaded
factorization was found to be non-deterministic — see `docs/compute.md` — so the earlier
run used a code path since proven unsafe) gives the clean, fully monotonic table above:
only two individual points had actually changed (`organic_substrate` k=0.5: 150.9°C →
179.1°C; `lsi_bridge_via` k=20: **397.5°C → 180.6°C**), and correcting those two collapsed
the apparent "cliff" entirely — it was a single corrupted data point, not a real threshold
effect. The lesson generalises beyond this one sweep: every dataset generated by this
project between adding that default and reverting it needs the same scrutiny before being
trusted, not just this table (checked: only the material sweep was generated in that
window; the 40-scenario pilot and its baselines above predate it and are unaffected).

With the corrected numbers, the organic-substrate result is still a real data point for
§9.9's finding — 47.5 K of spread from an unreported/uncited constant, still far larger
than the accuracy differences neural-surrogate papers are compared on. The bridge/via
result, corrected, is the opposite of what was first reported: peak-T is **flat to within
2 K across the entire 20-150 W/m·K range**, meaning this project's chosen nominal value
(60, "matching `hybrid_bonding`'s precedent, not independently derived" per `material.py`)
is not a risky choice sitting above a cliff — the geometry's other results are simply
insensitive to this particular constant across its whole plausible range. That is a
smaller, less dramatic finding than first reported, and the correct one.

**On whether any of this is a contribution**: no, and this project's own novelty framework
(§10, McGreivy & Hakim boundary) says so directly — 3D-ICE 4.0 itself is built and
published (Dec 2025, arXiv:2512.05823) specifically for heterogeneous chiplet substrate
modelling, and thermal bottlenecks from inadequate bridge/via coverage under high-power
dies are already a stated concern in the silicon-bridge/UCIe packaging literature (e.g.
ScienceDirect, Mar 2025). What is new here is narrow: applying this project's existing
baseline-comparison and interface-uncertainty methodology to one more, structurally novel
(within this benchmark) mechanism, and finding the same qualitative result — ridge wins,
and at least one unreported material constant (`organic_substrate`, not both this time)
still moves the answer more than architecture choice does.

### 9.12 A modes-vs-channels FNO sweep, and a methodology gap it exposed (2026-09-08/09)

Neural-operator scaling literature reports a real accuracy tradeoff for FNO at a fixed
parameter budget: spending that budget on more spectral modes vs. more hidden channels
is not interchangeable, and the optimal split is architecture- and problem-dependent, not
universal. This project has used the same hardcoded `channels=32, modes=(16,16,12)`
(`scripts/train_fno.py`'s default) for every FNO run on every geometry without ever
checking whether that split is a good one. `scripts/fno_modes_channels_sweep.py` was
built to check it: six `(channels, modes)` combinations at two roughly-matched parameter
tiers (~4.6-4.8M and ~6.1-6.7M params), same geometry1 data, same seed, same 50-epoch
budget, plain baseline FNO (no physics loss, matching the ridge-comparison methodology).

**Result (50 epochs, CPU):**

| config | channels | modes | params | det.MAE (K) | spatial R² |
|---|---|---|---|---|---|
| cpu-fast-preset | 16 | (8,8,6) | 395K | **0.553** | **0.949** |
| tierA-narrow-wide | 16 | (28,28,10) | 4.82M | 2.386 | 0.102 |
| tierA-mid | 20 | (22,22,10) | 4.65M | 2.318 | 0.239 |
| tierB-narrow-wide | 18 | (28,28,10) | 6.10M | 2.149 | 0.266 |
| default (project's) | 32 | (16,16,12) | 6.30M | 1.698 | 0.503 |
| tierB-wide-narrow | 44 | (12,12,8) | 6.70M | **1.667** | **0.580** |

The 395K-parameter model beat every model with 12-17x more capacity, by a wide margin.
**This is not a "smaller is better" finding.** The training-loss trajectories show why:
`cpu-fast-preset` plateaus by epoch ~20 (rL2 0.43→0.39→0.38→0.36→0.36) and is essentially
converged by epoch 50, while every larger config is still descending steeply at epoch 50
(e.g. `tierB-wide-narrow`: rL2 0.38→0.25→0.15→0.11→0.094, still falling hard, no sign of
plateau). At a fixed 50-epoch budget, a small model finishes learning while a large one is
still partway down its loss curve — a convergence-budget confound, not a capacity result.
The sweep as designed cannot cleanly answer the modes-vs-channels question, because none
of the larger configs were trained long enough to reach their achievable accuracy.

**The one signal that survives the confound**: within the large-parameter tier (all
similarly under-converged, all at the same 50 epochs), there is a consistent monotonic
trend — more channels / fewer modes converges faster than fewer channels / more modes at
matched parameter count: `tierB-wide-narrow` (44ch) > `default` (32ch) >
`tierB-narrow-wide` (18ch) > `tierA-mid` (20ch) > `tierA-narrow-wide` (16ch), on both
det.MAE and spatial R². The project's existing default sits in the middle of that
ordering, not at the front — `tierB-wide-narrow` beats it at nearly the same parameter
count and identical training budget, a free improvement at this budget if the trend holds
under full convergence (see below).

**A bigger and more consequential finding than the one the sweep was built to check.**
This project's own methodology, stated explicitly in the A3/A3b/geometry7 Kaggle
notebooks ("same budget for all models — fair comparison, not each model's own default"),
assumes a fixed epoch count is a fair comparison across architectures of different
capacity. This sweep shows that assumption is not safe: larger-capacity models need more
gradient steps to converge, so a fixed-epoch comparison systematically penalises them.
Every prior FNO-vs-CondFNO-vs-CNO-FNO comparison in this project used exactly that
fixed-epoch convention across models of different effective capacity, and none of them
checked whether the losing architecture was undertrained rather than genuinely worse.
This does not mean any specific prior result in this report is wrong — CNO-FNO's added
capacity (CNN encoder/decoder, FiLM conditioning) is modest relative to plain FNO, and the
gap to ridge everywhere in this project is large enough (10-100x on det.MAE) that a
convergence effect alone is very unlikely to close it — but it is a real, previously
unexamined confound in the comparison protocol, worth flagging for any future architecture
comparison in this project, not just this sweep.

**Correction (2026-09-09): the ridge reference used in this section was stale.** The
paragraph here originally read "even the best-converged config (0.553 K det.MAE) is ~60x
worse than ridge's 0.009 K," quoting §9.1's 2026-07-31 table without re-measuring it —
the same failure mode this report criticises elsewhere, committed inside a section written
to check someone else's assumption. §9.5 had *already* recorded that those numbers describe
a superseded dataset. Re-measured on the current data (§9.1a), geometry1 ridge scores
det.MAE **0.407 K**, spatial R² **0.970**. The real gap is therefore ~2.7x on det.MAE, not
60x. Ridge still leads on field reconstruction; the margin is much smaller than this
section originally claimed.

**Confirmatory run (200 epochs, completed 2026-09-09): the edge was mostly the
convergence-speed artifact, not a final-accuracy win.** `default` vs. `tierB-wide-narrow`
retrained at 200 epochs each (`--only default tierB-wide-narrow --epochs 200`):

| config | channels | modes | params | det.MAE (K) | spatial R² | hotspot loc. err (µm) | train time |
|---|---|---|---|---|---|---|---|
| default (project's) | 32 | (16,16,12) | 6.30M | 1.091 | **0.839** | 5083 | 99.3 min |
| tierB-wide-narrow | 44 | (12,12,8) | 6.70M | **1.087** | 0.823 | **4191** | 152.5 min |

At full convergence (both models' rL2 plateaued around 0.024-0.026, val MAE stable
5.2-5.9 K by epoch 200) the two configs are statistically indistinguishable on det.MAE
(1.091 vs 1.087, a 0.4% difference), the ranking on spatial R² actually **flips back** in
the project default's favour (0.839 vs 0.823), and only hotspot localisation still favours
`tierB-wide-narrow` (4191 vs 5083 µm). Compare this to the 50-epoch result, where
`tierB-wide-narrow` led clearly on both det.MAE (1.667 vs 1.698) and spatial R² (0.580 vs
0.503) — that gap has now mostly closed. The honest reading: `tierB-wide-narrow`'s
apparent 50-epoch advantage was substantially a convergence-speed effect (the
channel-heavy config warms up faster early in training), not a genuinely better final
accuracy at matched parameter count. It also costs 54% more wall-clock time to reach a
result that is not meaningfully better than the existing default.

**Conclusion for this project's FNO configuration**: the modes-vs-channels sweep does not
show the existing default (`channels=32, modes=(16,16,12)`) is leaving free accuracy on
the table once training is run to actual convergence — the two most different points
tested land in the same place. The genuinely load-bearing finding from this whole
exercise is not about modes vs. channels at all; it is the convergence-budget confound
itself (§9.12 above), which is a real, previously unexamined gap in this project's
fixed-epoch "fair comparison" protocol for *any* pair of architectures with different
effective capacity, and should be kept in mind for any future architecture comparison in
this project, not just this one.

#### 9.12b ⚠ RETRACTED — an apparent neural win on hotspot localisation, which did not survive cross-validation (see §9.12d)

> **Retracted 2026-09-10.** The claim below was measured on 5 test scenarios. Under 5-fold
> cross-validation on the same geometry (45 scenarios, §9.12d), it reverses completely for
> both FNO configurations tested. The subsection is kept rather than deleted because the
> project's central argument is about evaluation practice, and this is a worked example of
> the exact failure mode it criticises, produced by this project, in this project's own
> results. Read §9.12d before using anything below.

Scoring the two converged 200-epoch FNOs against the **current** ridge baseline (§9.1a)
rather than the stale one changes the reading of this experiment substantially:

| model | det.MAE (K) | spatial R² | hotspot loc. err (µm) |
|---|---|---|---|
| ridge (current, §9.1a) | **0.407** | **0.970** | 7399 |
| kNN (k=3) | 0.465 | 0.962 | 7101 |
| FNO, default (32ch, 200 ep) | 1.091 | 0.839 | 5083 |
| FNO, tierB-wide-narrow (44ch, 200 ep) | 1.087 | 0.823 | **4191** |

Both FNO configurations **beat ridge on hotspot localisation** — 4191 µm and 5083 µm
against ridge's 7399 µm, a 31–43% reduction in localisation error — while losing to it on
field reconstruction by ~2.7x on det.MAE. As far as this report records, that is the first
metric, on any geometry or regime tried in this project, on which a trained neural operator
beats the closed-form linear baseline.

Three caveats keep this from being over-claimed, and they are load-bearing:
1. **n = 5 test scenarios.** Any hotspot-localisation number on this split is a
   low-confidence read, the same caveat §9.11 carries.
2. **These FNOs are CPU-budget models**, 200 epochs against the project's normal 400–500,
   trained purely to answer a modes-vs-channels question. They were not tuned for this.
3. **Both are still bad in absolute terms.** 4–5 mm of localisation error on a 10 mm die is
   not a usable hotspot predictor. FNO is less bad than ridge here, not good.

What makes it worth recording anyway is the *direction*: it is exactly the split §9.1a and
§10 predict — the linear model owns the smooth, global, superposition-driven part of the
field (which field-level R² measures), while the neural model does relatively better at the
sharp local extremum (which is what chip-thermal work actually needs, and what the
operator-learning literature for 3D-ICs — DeepOHeat, SAU-FNO — is motivated by). It is also
a direct, if preliminary, answer to the question of whether this benchmark's headline metric
choice has been flattering the linear baseline: on this evidence, yes.

**This should be measured properly before it is claimed.** The honest next step is a run
at full epoch budget across more than five test scenarios, scoring hotspot localisation as
a first-class metric rather than a footnote column — not another CPU smoke test.

#### 9.12c Scored properly, the "neural win" is real but much narrower than 9.12b implied (2026-09-09)

Two things were built to test §9.12b rather than accept it: `scripts/hotspot_eval.py`
(k-fold CV over pooled train+test scenarios, so every scenario is scored once as a
held-out point, with hotspot metrics that survive multi-modal fields) and
`scripts/hotspot_eval_fno_cv.py` (the same folds, with an FNO trained per fold).

**First, a diagnostic that decides whether the headline metric means anything.** Whether
argmax-to-argmax distance is a meaningful score depends entirely on how sharp the true
peak is, and that turns out to be geometry-dependent:

| geometry | median spread of 100 hottest cells | ΔT across those cells | argmax distance is… |
|---|---|---|---|
| geometry1 | **324 µm** | 0.207 K | meaningful — a sharp, well-posed target |
| geometry6 | **9402 µm** | 1.646 K | largely noise — several near-equal hot regions |

So geometry1's 5-7 mm baseline errors are a genuine failure to find a hot region only
~300 µm across. geometry6's 15-31 mm "errors" are substantially an artifact of a package
with six HBM stacks and no single well-defined hotspot, and should not be quoted as a
failure of the same kind. §9.1a's hotspot column should be read with this table beside it.

**Baselines, 5-fold CV (45 scenarios on geometry1, 55 on geometry6 — not 5):**

| geometry | baseline | \|peak err\| (K) | median loc err (µm) | top-1% recall | hit ≤2 mm |
|---|---|---|---|---|---|
| geometry1 | mean | 16.875 | 6378 | 0.034 | 0.09 |
| geometry1 | nearest-neighbour | 6.311 | 6307 | 0.032 | 0.00 |
| geometry1 | kNN (k=3) | 5.306 | 6300 | 0.032 | 0.07 |
| geometry1 | **ridge** | **3.759** | **4374** | **0.093** | **0.16** |
| geometry6 | kNN (k=3) | 6.640 | 13750 | 0.370 | 0.16 |
| geometry6 | **ridge** | **3.416** | 18528 | 0.382 | 0.05 |
| geometry6 | mean | 15.864 | 15579 | **0.435** | 0.16 |

Ridge is the best *baseline* on hotspot metrics too, not just on field reconstruction — but
in absolute terms it fails the task: on geometry1 it recovers 9.3% of the top-1% hottest
cells and lands within 2 mm of the true peak in 16% of scenarios. Note also that on
geometry6 the **trivial mean-field predictor recovers more of the hot region (0.435) than
ridge does (0.382)** — predicting the training-set average finds the hot cells better than
the fitted linear model there.

**The single most useful number in this section**: on geometry1, ridge's mean absolute
*peak-temperature* error is **3.76 K**, against a detrended whole-field MAE of **0.407 K**
(§9.1a) — roughly 9x worse at the peak than across the field. Ridge's error is not
uniformly distributed; it concentrates exactly where the engineering decision is made.

**And the qualification to §9.12b.** Re-scoring the same two 200-epoch FNO checkpoints
with the fuller metric set (shipped 5-scenario split, so directly comparable to §9.12b's
numbers and carrying the same n=5 caveat):

| model | \|peak err\| (K) | median loc err (µm) | top-1% recall | hit ≤2 mm |
|---|---|---|---|---|
| kNN (k=3) | **1.777** | 7616 | 0.038 | 0.00 |
| ridge | 2.132 | 7169 | 0.032 | 0.00 |
| FNO, default | 14.860 | 4469 | 0.025 | **0.40** |
| FNO, tierB-wide-narrow | 8.060 | **3324** | **0.054** | 0.20 |

The FNOs do localise better — 3.3-4.5 mm vs ridge's 7.2 mm, and they post the only
non-zero hit rates within 2 mm — but they are **4-7x worse at predicting how hot the peak
actually is** (8.1-14.9 K vs ridge's 2.1 K). §9.12b's framing ("a neural operator beats
ridge") was drawn from the localisation column alone and is too generous. The accurate
statement is a **metric split**: the linear model knows how hot it gets but not where; the
neural model knows roughly where but not how hot. For thermal sign-off, which needs both,
neither is usable, and it is not obvious that being 15 K wrong about peak temperature is
the better failure to have.

A k-fold FNO run on identical folds (`scripts/hotspot_eval_fno_cv.py`) is what would settle
whether the localisation advantage survives at n=45; until that lands, both §9.12b and this
subsection rest on five scenarios. **It landed — see §9.12d. It does not survive.**

#### 9.12d The localisation advantage was an n=5 artifact — retraction, measured (2026-09-10)

`scripts/hotspot_eval_fno_cv.py` trains one FNO per fold on the *same* folds
`scripts/hotspot_eval.py` uses (same `kfold_indices` call, same seed), so FNO and the
baselines are scored on identical held-out scenarios. Both FNO configurations from §9.12
were run at 150 epochs, 5 folds, 45 held-out scenarios on geometry1.

**`tierB-wide-narrow` (the stronger claim — the better localiser at n=5), complete 5-fold:**

| model | \|peak err\| (K) | median loc err (µm) | loc p90 (µm) | top-1% recall | hit ≤1 mm | hit ≤2 mm |
|---|---|---|---|---|---|---|
| mean | 16.875 | 6378 | 9100 | 0.034 | 0.02 | 0.09 |
| nearest-neighbour | 6.311 | 6307 | 9915 | 0.032 | 0.00 | 0.00 |
| kNN (k=3) | 5.306 | 6300 | 9748 | 0.032 | 0.00 | 0.07 |
| **ridge** | **3.759** | **4374** | **9320** | **0.093** | **0.04** | **0.16** |
| FNO, tierB-wide-narrow | 10.539 | 6268 | 9465 | 0.065 | 0.02 | 0.07 |

**Ridge wins every column.** For comparison, the same two models at n=5 (§9.12c) had FNO
ahead on localisation (3324 µm vs ridge's 7169), on recall (0.054 vs 0.032) and on hit-rate
(0.20 vs 0.00). The `default` configuration reversed the same way (measured at n=27:
FNO 5301 µm vs ridge 4810 µm). **Both configurations, properly cross-validated, lose to
ridge on peak error, localisation and hot-cell recall.**

§9.12b's claim is therefore withdrawn in full. No neural architecture tried in this project
beats the closed-form linear baseline on any metric, on any geometry or regime tested.

**What this episode is worth keeping.** The retracted claim was not careless — it used the
project's own shipped test split, the metric already in `scripts/baselines.py`, and it was
caveated at the time ("n = 5 test scenarios", "needs a GPU-scale run before being claimed").
It was still wrong, and the direction of the error was the flattering one. Two things follow
that are more useful than the claim would have been:

1. **Five-scenario test splits can invert an architecture ranking.** This benchmark ships
   5-scenario test splits (§4), and so do the throttle/leakage/geometry7 pilots on which
   §9.7-§9.11's neural-vs-ridge comparisons rest. Every one of those comparisons carries
   this same risk and none has been cross-validated. That is a limitation of this report,
   stated here rather than left for a reader to find.
2. **A negative result about our own positive result.** §10 argues the field publishes
   architecture wins that a stronger baseline or a better evaluation would erase. This
   project produced exactly such a win and erased it internally within a day. That is a
   sharper demonstration of the argument than any external example, and it should be
   reported as one.

### 9.13 The linear baseline on IC-ThermBench: it loses, and that reframes this paper (2026-09-10)

IC-ThermBench (arXiv:2608.23977) evaluates eight baselines — U-Net, FNO, U-FNO, SAU-FNO,
DeepOHeat, Therm-FM T/B/L — and **not one is non-neural**. This section supplies the
missing row. Method and pre-registered predictions are in `docs/ic_thermbench_plan.md`,
written and committed *before* any baseline was run.

Parity is not asserted, it is enforced: their metric module is vendored verbatim
(`third_party/ic_thermbench/metrics.py`, their own docstring forbids writing a second
implementation), and our loader is checked array-exact against their own loader and
splitter (`scripts/ic_thermbench_data.py::verify_against_upstream`). Their split is
index-based and unshuffled: 10,800 train / 1,200 val / 3,000 test. λ was selected on
**their** validation split, never on test.

**Results (their metrics, their test split; RMSE is per-sample then averaged):**

| scope | baseline | RMSE ↓ | MAE | R² | MaxAE | peak-T err | Top-50 MAE | params |
|---|---|---|---|---|---|---|---|---|
| **S2** | *published best (Therm-FM)* | ***0.4427*** | — | — | — | — | — | — |
| | *published runner-up (SAU-FNO)* | *0.7028* | — | — | — | — | — | — |
| | mean field | 16.305 | 14.591 | 0.212 | 28.63 | 20.13 | 21.04 | 4 k |
| | kNN (k=3) | 4.406 | 3.502 | 0.855 | 11.80 | 3.68 | 4.53 | 0 |
| | ridge-pca | 3.430 | 2.758 | 0.961 | 9.41 | 3.51 | 4.14 | 3.1 M |
| | ridge-green | 3.131 | 2.514 | 0.962 | 8.78 | 3.10 | 3.73 | 18.9 M |
| | **ridge-per-geom** | **1.943** | 1.483 | 0.980 | 6.96 | 1.93 | 1.86 | 21.1 M |
| **S3** | *published best (Therm-FM)* | ***0.7161*** | — | — | — | — | — | — |
| | mean field | 15.942 | 14.339 | 0.092 | 27.00 | 19.27 | 20.26 | 4 k |
| | kNN (k=3) | 10.889 | 9.523 | 0.317 | 21.86 | 10.61 | 14.65 | 0 |
| | ridge-green | 4.603 | 3.785 | 0.922 | 11.89 | 4.03 | 5.37 | 35.7 M |
| | ridge-pca | 4.159 | 3.460 | 0.936 | 10.16 | 4.36 | 5.32 | 4.2 M |
| | **ridge-per-geom** | **2.488** | 1.985 | 0.972 | 7.92 | 1.91 | 2.25 | 21.0 M |
| **S4** | *published best (Therm-FM)* | ***0.9334*** | — | — | — | — | — | — |
| | mean field | 24.123 | 22.738 | 0.031 | 35.94 | 25.03 | 25.03 | 4 k |
| | kNN (k=3) | 20.654 | 19.500 | 0.177 | 32.79 | 21.50 | 23.79 | 0 |
| | ridge-green | 8.116 | 7.152 | 0.895 | 16.72 | 7.73 | 8.56 | 35.7 M |
| | ridge-pca | 6.946 | 6.090 | 0.918 | 14.04 | 6.84 | 7.49 | 4.2 M |
| | **ridge-per-geom** | **3.203** | 2.635 | 0.984 | 9.05 | 2.76 | 3.03 | 21.2 M |

**The headline: the linear baseline loses, clearly and on every scope.** The best linear
model is 4.4× / 3.5× / 3.4× worse than Therm-FM on S2 / S3 / S4, and worse than every one
of their eight neural baselines. **IC-ThermBench is not linear-solvable.** Their benchmark
discriminates, which is exactly what a benchmark is for, and it is the opposite of what
this project found on its own dataset (§9.1a).

**This reframes the paper, and the reframing is an improvement.** The claim "a closed-form
linear fit is competitive on 3D-IC thermal benchmarks" is now falsified as a general
statement — by an independent benchmark 300× larger than ours, tested with our own tooling,
in a run we predicted the wrong direction on. What survives is the *methodological* claim,
and it survives intact and stronger:

> Run the linear baseline. It costs minutes. It is **diagnostic in both directions**: on
> this project's dataset it revealed a benchmark that was near-linear-solvable, i.e. a
> benchmark-design fault (§9.2–§9.5); on IC-ThermBench it certifies the opposite. The two
> results are the same diagnostic returning opposite verdicts — and the field ran it in
> neither case.

A benchmark that *passes* the linear check has earned the neural architectures evaluated on
it. IC-ThermBench passes. Ours, before the regime fix, did not. That is a more useful thing
to be able to say than "linear models win", and it is the version we can defend.

#### 9.13a The mechanism: most of the linear model's deficit is missing layout conditioning

Conduction is exactly linear in the source for a *fixed* operator, `T = T_amb + G·Q`. So why
does a dense ridge operator lose? Measurement answers it directly: hashing the
`grid_x`/`grid_y` channels shows **the substrate geometry takes only ~20 discrete values
(20 groups on S2, 10 on S3/S4) while the power map varies essentially per sample** (1,895
distinct support patterns in 2,000 samples). There are ~10–20 *different operators*, each
driven by a continuously varying source. A single global linear map is mis-specified by
construction, and no amount of regularisation repairs that: a linear model given geometry
as a feature can only add a geometry-dependent *offset*; making `G` itself depend on
geometry requires a geometry × power interaction, which is precisely what it cannot form.

Fitting one ridge per geometry group tests this, and the effect is large:

| scope | ridge-green (global) | ridge-per-geom | error removed |
|---|---|---|---|
| S2 | 3.131 | 1.943 | **38%** |
| S3 | 4.603 | 2.488 | **46%** |
| S4 | 8.116 | 3.203 | **61%** |

(Every test sample's geometry was seen in training, so no fallbacks were used.) **A large and
growing share of what the neural operators provide on this benchmark is the ability to
condition on layout, not nonlinearity in the source.** The residual gap to Therm-FM is real
and is not explained by this, but the decomposition is, as far as we can find, not reported
anywhere — and it is available for the price of a few minutes of CPU.

#### 9.13b Prediction accounting, including one that was plainly wrong

`docs/ic_thermbench_plan.md` §4 recorded four predictions before running anything. Scoring
them honestly:

- **S2 — correct, and resolved to the branch we flagged as uncertain.** The prediction was
  "strong if the operator is shared across layouts; clearly worse if not." It is not shared
  (~20 discrete geometries, measured), and ridge is clearly worse. The uncertainty was
  genuine and the measurement settled it.
- **S3 — correct in direction.** "Clear degradation" from added per-cell conductivity:
  ridge-green 3.131 → 4.603 (+47%). Worth noting Therm-FM degraded proportionally *more*
  (0.4427 → 0.7161, +62%), so material variation is not uniquely hard for linear models.
  It is uniquely hard for *distance-based* ones: kNN collapses 4.41 → 10.89.
- **S4 — WRONG, and instructively so.** We predicted ridge would degrade *less* than the
  neural models from S3→S4, reasoning that S4's three added channels enter the linear
  solution exactly (ambient is additive; `r_convec_k_per_w` is already the `1/h` form that
  `scripts/baselines.py` hand-constructs). Actual: ridge-green degraded **+76%** (4.603 →
  8.116) against Therm-FM's **+30%**. The error in the reasoning is specific and worth
  keeping: `h` does not enter additively at all — it sets the convective boundary condition
  and therefore changes `G` itself, so its effect *multiplies* the power term. Having `1/h`
  available as a feature is useless to a model that cannot form `h × power` interactions.
  The prediction only becomes right once layout conditioning is supplied: ridge-per-geom
  degrades +29%, essentially matching Therm-FM's +30%.
- **Hotspot regime (§9.4's per-cell-power claim) — not yet separable.** Their inputs are
  per-cell power maps as predicted, and their peaks are sharp: the 100 hottest cells span
  ~4.1 cells (6.4% of the 64-cell width) on S2–S4, comparable to geometry1's well-posed
  case (§9.12c) rather than geometry6's diffuse one. So argmax localisation *would* be a
  meaningful metric here — and IC-ThermBench does not report it (their ETmax explicitly
  ignores location). That remains an additive contribution rather than a tested claim.

One prediction in four was wrong, in the flattering direction, and it was recorded in
advance where it could be scored. That is the point of recording them.

#### 9.13c S5 zero-shot: the linear operator is not merely wrong, it is unstable — and the trivial baseline nearly matches the best neural model

Their S5 protocol freezes an S4-trained model and scores it on 5,000 samples from five
unseen package systems, with preprocessing statistics unchanged. Applying our S4 fits the
same way:

| model | RMSE ↓ | MAE | R² | peak-T err |
|---|---|---|---|---|
| *published best (Therm-FM)* | ***15.51*** | — | — | — |
| *published runner-up (U-Net)* | *19.10* | — | — | — |
| **mean field (trivial)** | **29.638** | 28.905 | −0.151 | 29.57 |
| ridge-pca | 1740.5 | 1389.3 | −9037.5 | 1715.7 |
| ridge-green | 1973.5 | 1841.3 | −11668.5 | 888.8 |

Two findings, and they point in opposite directions.

**1. The linear operators do not degrade, they detonate.** ~1,700–2,000 K RMSE is not a
score, it is a numerically degenerate extrapolation — the same failure mode
`scripts/baselines.py` already guards against in its `power_pca` note (an unregularised fit
there once produced 3e12 K). It is worth stating precisely *why*, because the obvious
explanation is wrong. We checked whether S5's inputs simply lie far outside S4's range;
mostly they do not:

| channel | max abs z-score within S4 train | max abs z-score on S5 |
|---|---|---|
| `chiplet_power` | 98.8 | **61.7** |
| `local_thermal_k` | 34.9 | **31.5** |
| `grid_x` | 1.7 | 2.7 |
| `grid_y` | 1.7 | 2.3 |

The power and conductivity maps are *inside* the range S4 itself spans. Only the geometry
channels move outside, and only modestly (1.7 → 2.7). But geometry is precisely the input
that selects the operator, and the model has just ~10 discrete training values along that
axis, so a linear map extrapolating there is unconstrained: a small move in the one
direction that matters produces an unbounded output. The instability is a real property of
the fitted model, not an artefact of our transfer path, and it is a genuine limitation of
linear surrogates for structural OOD that a benchmark without non-neural baselines cannot
surface. It should be reported as instability rather than as a competitive number, and we
do not claim the specific value means anything beyond "unbounded".

Note also that λ was selected on S4's *in-distribution* validation split, exactly as their
models select checkpoints. That protocol optimises for interpolation and offers no
protection against this. We deliberately did not re-tune λ against S5, which would be
selecting on the test set.

**2. The far more interesting number is the trivial one.** On S5 zero-shot the **mean-field
predictor scores 29.64 K against Therm-FM's 15.51 K**  <!-- anchor:s5-trivial --> — the best neural model in the field's
newest benchmark is **only 1.9× better than predicting the training-set average**, on the
scope their own paper identifies as the benchmark's main distinction (a 16.6× degradation
from S4). U-Net at 19.10 K is 1.55× better than the trivial baseline.

That is exactly the observation §10 argues the field's evaluation practice hides, and it is
invisible in their Table 4 because it contains no trivial baseline to compare against. It
does not diminish IC-ThermBench — S2–S4 are genuinely discriminative (§9.13) and the S5 gap
is honestly reported as their headline finding. But "structural OOD transfer is largely
unsolved" is a much sharper statement when the reference point is a predictor with no
parameters at all, and supplying that reference point costs seconds.

### 9.14 Why our benchmark is linear-solvable and theirs is not — and why that is a criticism of ours (2026-09-11)

§9.13 leaves an obvious question: ridge is competitive on our data and loses badly on theirs,
so what is the actual structural difference? Two candidate explanations were tested. The
appealing one is wrong; the real one is unflattering to this benchmark.

**Rejected: "IC-ThermBench scores hotspots, which is where ridge is weakest."** §9.12c
established that ridge's error concentrates at the peak, so a hotspot-weighted benchmark
would disadvantage it. But their headline metric is field-wide RMSE, not a hotspot metric,
and ridge loses on *that* by 7×. Decomposing squared error by region settles it:

| dataset / model | top-1% cells hold … of signal variance | … and absorb … of squared error | concentration |
|---|---|---|---|
| IC-ThermBench S2 / ridge | 3.4% | 2.1% | 2.1× |
| this project geometry1 / ridge | 4.4% | 2.4% | **2.4×** |

Ridge's error is **no more hotspot-concentrated on their data than on ours** — marginally
less. The difficulty is spread across the whole field. This explanation is refuted.

**The actual difference: our benchmark holds the thermal operator fixed, and theirs does
not.** Three measurements, each a structural property of the datasets rather than of any
model:

| property | this project | IC-ThermBench S2 |
|---|---|---|
| distinct heat-source support patterns | **1** (every geometry: g1, g4, g6, g7) | 269 in 277 samples of one grid group |
| mean pairwise IoU of source support | **1.000** | 0.449 |
| physical cell size across samples | fixed per geometry | **varies: 0.667 / 0.875 / 0.414 mm per cell** |

Our heat sources **never move**. Across all 45 geometry1 scenarios, all 45 geometry4, all 55
geometry6 and all 40 geometry7, the power field has exactly one support pattern; only the
*amplitudes* on those fixed blocks change, alongside HTC and ambient. The medium is fixed,
the mesh is fixed, cell *i* always denotes the same physical place. That makes the map

  T(x) = T_amb + Σ_b A_b(x)·Q_b,  with A_b fixed

**exactly, and only, ridge's hypothesis class.** We did not discover that a linear model is
surprisingly competitive; we constructed a benchmark whose solution is a fixed linear
operator and then reported that fitting a fixed linear operator works.

IC-ThermBench breaks this in two independent ways. Chiplets move between samples (IoU 0.449),
so the medium — silicon islands in a substrate — is rearranged and the operator changes. And
their 64×64 array is a *normalised* grid over packages of different physical size:
`grid_x`/`grid_y` are per-sample physical coordinate maps, uniform within a sample but with
cell pitch varying roughly 2× across samples. Cell *i* is a different physical location, at a
different physical scale, in different samples. A single linear map over array indices is
comparing incommensurable domains, which is why grouping by grid hash — restoring a common
physical scale — recovers 38–61% of the error (§9.13a).

#### 9.14a The regime our benchmark occupies has been solved by classical methods since 2007

This is the part that should have been checked at the start. For a **fixed package geometry
with varying power maps**, chip-thermal EDA has a standard, validated technique: compute the
thermal impulse response (Green's function / point spread function) once by FEM, then convolve
it with any power map. It is called **power blurring**, and the "influence coefficient"
formulation — determine the matrix by applying linearly independent power vectors and solving
— is the same object our ridge baseline estimates from data.

- Kemper, Zhang, Bian & Shakouri (2007), *Ultrafast Temperature Profile Calculation in IC
  Chips* (arXiv:0709.1850): explicitly assumes a fixed package geometry, reuses one PSF across
  varying power maps, reports **hot-spot temperatures within 1 °C** and **three orders of
  magnitude** speedup over FEA.
- Ziabari, Park, … Shakouri, *Power Blurring: Fast Static and Transient Thermal Analysis
  Method for Packaged Integrated Circuits and Power Devices*, **IEEE TVLSI** (2013): within
  **2%** of a commercial FEM tool, orders-of-magnitude faster.

So the fixed-floorplan/varying-power problem was solved to ~1 °C, a thousand times faster than
FEM, **fifteen years before** the neural-operator papers this project critiques. Our §9.1
result is a rediscovery of that fact from the ML side, not a new one.

**This makes the paper's central recommendation sharper, not weaker.** "Report a linear
baseline" is vague and easy to wave away. The defensible version is specific:

> If a thermal-surrogate benchmark holds the package geometry and source locations fixed and
> varies only power amplitudes, it is the regime power blurring already solves to ~1 °C. A
> neural surrogate evaluated only there should be compared against **power blurring or an
> influence-coefficient fit**, by name — not merely against other networks. If a benchmark
> intends to test something beyond that regime, it must vary what makes the operator change:
> source *placement*, medium, or domain scale.

IC-ThermBench does vary those things, which is why it discriminates and why its neural models
earn their result. Ours did not, which is why ours did not.

#### 9.14b What this benchmark would have to change

§11 already lists "variable floorplans" among the conditions this benchmark needs; §9.14 turns
that from a suggestion into the *primary* defect, ahead of power density or per-cell power. In
priority order:

1. **Move the sources.** Randomise block placement per scenario rather than only amplitude.
   This alone converts a fixed-operator problem into a varying-operator one, and it is the
   single change most likely to make the benchmark discriminate.
2. **Vary the medium.** geometry7's bridge/organic heterogeneity (§9.11) is the right kind of
   structure, but it is currently *fixed* within the geometry — its islands never move.
3. **Vary domain scale.** Different package extents on a common normalised grid, as
   IC-ThermBench does, is a genuinely harder axis and one this project has never tested.

Until at least (1) is done, results on this dataset should be read as characterising a
fixed-operator regime, and the honest scope of §9.1's finding is: *within the regime classical
superposition methods already solve, a learned linear operator also solves it.*

### 9.15 Implementing the fix: moving the sources is not enough — layout degrees of freedom are what matter (2026-09-11)

> **Numbers superseded by §9.15b (2026-09-11).** Every figure in this section comes from a
> single 40/5 split, and 5 held-out fields turned out to be too few to measure with — one
> scenario was setting the headline R². §9.15b re-scores all of it with 5-fold CV over 45
> scenarios. The *reasoning* below stands (layout dimensionality, not support overlap, is what
> breaks a linear fit); the specific R² values do not. Cite §9.15b's table instead.

§9.14b named "move the sources" as the primary fix. It was implemented
(`src/core/placement.py`, `scripts/gen_moving_source_pilot.py`), three new pilots were
generated as real 3D-ICE solves, and **the first attempt did not work.** The second did. The
difference between them is the actual finding, and it corrects §9.14's framing.

A chiplet is translated as a rigid unit — its `DiePrint` plus the `PowerBlock`s it contains —
so in 2.5D geometries the silicon island moves with its heat and the lateral material map
changes. Per-block *positions* are now exported and consumed by `scripts/baselines.py`; a
baseline that could not see where the heat moved would be a strawman, not a fair test.
(Verified no regression: fixed-placement geometry1 scores an unchanged 0.407 / 0.970.)

**Results. `norm err` is det.MAE divided by that dataset's own signal σ, because moving the
sources roughly doubles the spatial variance and raw det.MAE is therefore not comparable
across rows:**

| dataset | layout DOF | support IoU | σ (K) | det.MAE | **norm err** | spatial R² |
|---|---|---|---|---|---|---|
| geometry1, fixed (§9.1a) | 0 (amplitudes only) | 1.000 | 4.00 | 0.407 | **0.102** | 0.970 |
| geometry1, rigid translate | 2 | 0.472 | 9.09 | 0.999 | **0.110** | 0.919 |
| geometry1, independent blocks | 8 | 0.347 | 9.28 | 2.218 | **0.239** | **0.770** |
| geometry4, fixed (§9.1a) | 0 (amplitudes only) | 1.000 | 2.63 | 0.320 | **0.122** | 0.891 |
| geometry4, rigid translate | 4 | 0.735 | 5.92 | 0.755 | **0.128** | 0.962 |

**Attempt 1 (rigid translation) failed.** Relative to the signal, ridge barely moved: +8% on
geometry1, +5% on geometry4, and geometry4's spatial R² actually *improved* (0.891 → 0.962).
This is despite geometry1's rigid-translate pilot reaching a support overlap of **IoU 0.472 —
better source movement than IC-ThermBench's 0.449**. So the sources genuinely moved as much as
theirs, and the benchmark still did not discriminate.

**Attempt 2 (independent per-block placement) worked.** Giving the layout 8 degrees of freedom
instead of 2 more than doubled ridge's relative error (0.102 → 0.239) and dropped spatial R²
from 0.970 to 0.770. The distance-based baselines collapsed outright: kNN to R² 0.611,
nearest-neighbour to **R² −0.294**, i.e. worse than predicting a constant.

**The correction to §9.14: support-overlap is the wrong diagnostic; layout dimensionality is
the right one.** IoU ordered these datasets wrongly — geometry1's rigid-translate pilot has a
*lower* IoU (0.472) than the independent-block pilot needs to break ridge, yet it left ridge
intact. The mechanism is that a rigid translation is a two-parameter family and the
temperature field is laterally smooth, so a linear model handed the displacement covers it to
first order (`T(x − d) ≈ T(x) − d·∇T`). Independent placement changes the *relative*
arrangement of the sources, so the field's structure changes rather than merely shifting, and
no first-order response in a handful of placement parameters suffices.

This also explains IC-ThermBench without appeal to its material variation: its layouts come
from a placement optimiser over ~20 chiplets, which is a high-dimensional layout space, not
translations of one arrangement. §9.14's "chiplets move" framing was right about the symptom
and wrong about the cause.

**Status of the fix.** geometry1 with independent block placement is the first configuration
in this project where a closed-form linear fit is clearly inadequate (R² 0.770, and the signal
is 9.3 K, so 2.2 K of detrended error is real). That is a benchmark worth training a neural
operator on, and it is the first one here of which that can be said. It does **not** rescue
the original six-geometry dataset, whose results remain characterisations of a fixed-operator
regime.

**Two caveats stated plainly.** (a) 45 scenarios is a small sample for an 8-dimensional layout
space; the degradation is unambiguous but the *absolute* numbers will move with more data.
(b) Independent block placement is physically loose — real floorplans are not uniform random
rectangles — so this is a benchmark-design demonstration, not a claim about realistic
floorplans. Reproducing IC-ThermBench's approach properly would mean sampling from a placement
optimiser, as ATPlace2.5D does.

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
dominated by a scalar offset that tracks the ambient input, and a closed-form solve already
lands within the low-single-digit-percent relative error that industrial thermal
sign-off literature generally treats as the bar (`docs/references.md` §9 — no single
authoritative absolute-°C figure was found in a search pass; verified sources cluster at
1–4% relative error, e.g. compact-thermal-model-vs-detailed-model agreement <1% and
CFD-vs-measurement studies ~2–4%). Two caveats keep that comparison from being a real
industrial-accuracy claim: those targets are validated against silicon or a detailed
reference model, not against another compact solver, and this benchmark's own ground
truth (3D-ICE) has never been checked against either (§10, Threats to validity); and
industrial accuracy is judged at the hotspot specifically, which is exactly where §9.4
shows ridge is weakest, not where it wins. We do not claim published results are wrong —
their datasets may be richer — but the linear-baseline comparison is rarely made, and it
is cheap to make. We release `scripts/baselines.py` for that purpose.

**This is not a new observation at the level of the field, and we do not claim it as one.**
McGreivy and Hakim [*Nature Machine Intelligence*, 2024] systematically reviewed
ML-for-PDE papers claiming to outperform standard numerical methods and found **79%
(60/76) compared against a weak baseline**, attributing the pattern to researcher degrees
of freedom, outcome-reporting bias and publication bias. Their review covers fluid-related
PDEs; what we add is (i) the chip/package thermal domain, which they do not examine, (ii)
a sharper form of the baseline claim — not that the comparison baseline was under-tuned,
but that a *closed-form linear fit with no training* saturates the usual metric, (iii) the
mechanism, namely a scenario space low-dimensional enough (~8 scalars) to lie on a
near-linear manifold, and the identification of per-cell power as the specific change that
breaks it (§9.4), and (iv) the metric argument of §9.1/§9.4. Their finding that negative
results are systematically under-reported is also the reason this paper exists in the form
it does.

This gap is not hypothetical even in work published after this project began. SAU-FNO
[arXiv:2510.15968, Oct 2025] — a self-attention/U-Net/FNO hybrid for 3D-IC thermal
prediction, and close prior art to this repo's own `CNOFNOHybrid`+axial-attention
architecture (§8.2) — reports 842× speedup and >50% MSE reduction over COMSOL/MTA without
stating a linear or closed-form baseline comparison anywhere. We do not know whether a
ridge fit would close that gap on their dataset; the point is that the question wasn't
asked, in a paper specifically about this problem class, three months before this
writing.

**Where this connects to the field's own stated priorities.** A recent survey of
multiscale 3D-IC thermal modeling [Barua, Udoy & Aziz, arXiv:2604.03290, 2026] argues the
field's "main challenge is no longer raw prediction speed, but robustness under
[distribution] shift," and calls for models to be "more uncertainty aware, more
explainable." Two results in this paper answer that directly, not by design but in
hindsight: the leave-one-geometry-out generalization test (§10, "Cross-geometry
generalization") found a lightweight geometry-aware conditioning mechanism does *not*
reliably improve zero-shot robustness across held-out geometries once tested across
multiple seeds — a negative result on exactly the robustness question the survey
identifies as unresolved, not just a benchmark number. And the MC-Dropout uncertainty
calibration check (§10, "Deferred") found predictive std 30× the actual spatial signal —
direct evidence that "uncertainty aware" is not yet true of at least one standard
technique on this problem class, consistent with the same survey's separate observation
that UQ is not yet propagated through multiscale/reduced-order model chains. Separately,
the survey flags incomplete multiphysics coupling (thermal-mechanical-electrical) as a
structural gap; this paper's throttling (§9.8, a genuine electro-thermal feedback loop)
and microchannel (fluid-thermal advection) mechanisms are concrete, already-measured
steps in that direction, not previously framed that way in this document.

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
   ~455 W core-fraction TDP ceiling (peak 167.4 °C), and the `process_scenario`→
   `NPZExporter` export path was separately confirmed to work correctly on a
   microchannel-cooled scenario end to end (`ICESimulator` already omits the coolant
   element's Tmap request, and the real-simulator path sources coordinates from what
   3D-ICE actually returned, not from the geometry's static layer list, so no mismatch
   occurs). What's missing is CLI wiring, not a pipeline fix — none of the 6 registered
   geometries have `coolant_layer_name` set, so there is no way to select a
   microchannel-cooled run through `main.py` yet — so no microchannel-cooled scenarios
   exist in the current 275-scenario dataset. This remains the first candidate change
   that could alter the paper's central finding rather than refine the dataset around
   it, once a generation script exists and it is swept as a scenario axis.
6. **Thermal throttling (DVFS)** — *mechanism built and validated 2026-08-06, not yet
   integrated into the dataset*. `src/scenario/throttling.py` wraps 3D-ICE in an outer
   solve-derate-resolve loop: power is reduced when peak temperature exceeds a
   junction-temperature limit, then the scenario is re-solved, converging once the peak
   settles under the threshold or a power floor is hit. This makes power a function of the
   temperature field being solved for — a genuine closed feedback loop that a
   scenario-fixed source cannot express, distinct from every fix above including advective
   cooling (which still solves a single fixed-source steady state, just with a different
   boundary condition). Validated on geometry3 against the real binary at two overshoot
   levels; both converged in 2 solves, cheaper than the 3–5 solves budgeted before
   implementation, because steady-state conduction's linearity in power means one
   proportional correction from the first solve's overshoot lands close to the target. Not
   yet wired into `main.py`'s CLI path, so no throttled scenarios exist in the current
   dataset either.

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

**Cross-geometry generalization is scoped narrower than "few-shot fine-tuning across
geometries" (§1, item 6) might suggest.** The mechanism (`--common-grid` trilinear
resampling onto a shared grid plus a single `geom_extent_norm` scalar as conditioning) is
what current operator-learning literature would call brute-force grid alignment, not
geometry-aware encoding — contrast with SDF- or graph-based geometry conditioning (e.g.
GINO, PI-GANO), which report <3% error on genuinely unseen shapes.

A leave-one-geometry-out test — train on 5 geometries, test zero-shot on the 6th, with
vs. without a per-cell distance-to-nearest-power-block field
(`generate_distance_to_power_block_field`, `src/core/mesh.py`) as a 6th FNO input channel
— was run twice: first as a single run (2026-08-09, geometry4 held out, seed 42), then
repeated across 3 seeds × 2 held-out geometries (geometry4, geometry1; 2026-08-10) to
check whether the first result was a single-run artifact. It was.

The single run gave a superficially clean-looking result: MAE, detrended MAE, and
hotspot location error all improved modestly with the field (3.78→3.10 K, 1.24→1.14 K),
spatial R² got worse (0.326→0.258). The multi-seed/multi-holdout follow-up shows why that
shouldn't have been trusted on its own:

| holdout | metric | baseline (mean±std, n=3) | geom-field (mean±std, n=3) | winner |
|---|---|---|---|---|
| geometry4 | spatial R² | 0.299±0.159 | 0.397±0.139 | field |
| geometry4 | hotspot loc. err | 5991±1377 | 4871±220 | field |
| geometry1 | spatial R² | 0.464±0.018 | 0.378±0.081 | **base** |
| geometry1 | hotspot loc. err | 4875±692 | 5856±866 | **base** |

The field helps on geometry4 (all 4 metrics) but hurts on geometry1 on the two metrics
that matter most for this paper's framing (spatial R², hotspot localisation) — and within
geometry4 alone, the three seeds don't even agree on direction: seed 42 (the original
single-run result) shows spatial R² *dropping* 0.326→0.258, while seeds 43 and 44 show it
*rising* 0.479→0.586 and 0.092→0.346 respectively. The original single-seed run happened
to land on the one case out of three where the field looked worst on that metric —
exactly the kind of single-run artifact multi-seed validation exists to catch.
**Conclusion: the geometry-aware field is not a validated generalisation mechanism at
this scale** (205 training scenarios, 60 epochs, 198K-parameter model, one grid
resolution). Its effect is geometry-dependent and seed-noisy rather than a consistent win
or loss, which is itself informative: claims that a lightweight per-cell geometric feature
"helps zero-shot generalization" on this benchmark need either substantially more
resourcing (more scenarios/epochs/holdouts) or a fundamentally different conditioning
mechanism (SDF/graph-based, as GINO/PI-GANO use) before they're defensible — not another
single-run experiment on this architecture. What's confirmed regardless: the *current
default* mechanism supports interpolation among the 6 trained geometries, not validated
zero-shot transfer to an unseen package shape.

**Deferred.** The originally planned discussion — PDE-residual/error correlation, IG
attribution plausibility, MC Dropout calibration — requires trained models and remains
open. (TSV generalisation across geometry2a/b/c is no longer planned: geometry2b/2c were
removed 2026-08-06, §4.) MC Dropout is already known to be poorly calibrated here: the
predictive std measured during explainability validation was 8.6–13 K on fields whose
spatial std is ~0.3 K, i.e. roughly 30× the signal.

---

### 9.15b Under cross-validation the layout fix is stronger than §9.15 said, and §9.15's numbers were not reliable (2026-09-11)

§9.15's table is a **single 45-scenario dataset split 40/5**. Five held-out fields is too few:
inspecting per-scenario spatial correlation showed ridge scoring r ≈ +0.9 on four test
scenarios and r ≈ −0.25 on one, so a single scenario was setting the headline R². Every
layout dataset was therefore re-scored with 5-fold CV over all 45 scenarios
(`scripts/layout_cv.py`, n = 45 held-out fields per model, identical folds for every model).

**`norm err` is det.MAE / that dataset's own σ. `R² < 0` counts held-out scenarios where the
model is worse than that field's own spatial mean — i.e. outright failures:**

| dataset (5-fold CV, n=45) | σ (K) | model | det.MAE | norm err | R² mean | R² median | R² < 0 |
|---|---|---|---|---|---|---|---|
| geometry1, independent blocks | 7.21 | mean | 3.852 | 0.534 | −0.306 | 0.447 | 16/45 |
| | | nn | 3.118 | 0.432 | 0.283 | 0.503 | 8/45 |
| | | **knn** | 2.925 | 0.406 | **0.465** | **0.654** | 5/45 |
| | | ridge | **2.281** | **0.316** | 0.203 | 0.468 | 5/45 |
| geometry4, shelf | 4.83 | mean | 2.925 | 0.606 | −0.389 | 0.345 | 12/45 |
| | | nn | 1.923 | 0.399 | 0.337 | 0.607 | 8/45 |
| | | **knn** | 1.999 | 0.414 | **0.401** | **0.637** | 7/45 |
| | | ridge | 3.014 | 0.625 | **−0.667** | 0.423 | 12/45 |
| geometry5, shelf | 3.09 | mean | 2.033 | 0.658 | 0.164 | 0.403 | 10/45 |
| | | **nn** | 1.545 | 0.500 | **0.496** | 0.628 | 3/45 |
| | | knn | **1.482** | **0.479** | 0.337 | **0.748** | 3/45 |
| | | ridge | 2.254 | 0.729 | **−2.305** | 0.302 | 17/45 |
| geometry6, shelf | 3.62 | mean | 2.213 | 0.612 | 0.338 | 0.350 | 8/45 |
| | | nn | 2.118 | 0.586 | 0.364 | **0.552** | 8/45 |
| | | **knn** | **2.006** | **0.555** | **0.449** | 0.524 | 5/45 |
| | | ridge | 3.742 | **1.035** | **−2.866** | **−0.303** | **26/45** |

**Three findings, one of which is a correction.**

**(1) On all three shelf-layout geometries ridge is now the worst model, not the best.** On
geometry4, geometry5 and geometry6 ridge is last by spatial R² on *both* mean and median, and
it is the only model whose mean R² is strongly negative. It fails outright — worse than the
field's own mean — on 12/45, 17/45 and **26/45** held-out scenarios, against 3–7/45 for the
distance-based baselines. **geometry6 is the decisive case:** ridge's *median* R² is negative
(−0.303), i.e. on more than half of held-out scenarios it is beaten by that scenario's own
spatial mean, and its normalised detrended error is **1.035 — the error exceeds the signal it
is trying to predict.** This is the first configuration in this project where the closed-form
linear fit is not merely adequate-but-uninteresting; it is beaten outright by a
3-nearest-neighbour lookup.

**(2) Ridge's failure mode is confident misplacement, not shrinkage.** The obvious explanation
for a low R² with a competitive det.MAE is that ridge regresses toward a flat field, which R²
punishes and MAE forgives. That was measured and it is **wrong**: ridge's predicted spatial
amplitude is the *highest* of all four baselines (amplitude ratio 0.80–0.93 of true σ, versus
0.50–0.66 for the mean and kNN predictors). Ridge commits to full-amplitude structure in the
wrong place. Its mean spatial correlation is the lowest of the four (+0.585 on geometry5 vs
+0.866 for kNN), and its hotspot-location error is the largest (10.3 mm vs 6.1 mm for kNN on
the single-split run). Under-committing models earn modest positive R² by predicting little;
ridge earns a negative one by predicting a lot, incorrectly. For a hotspot-localisation task
this is the worse failure of the two, and it is invisible in a raw-MAE table.

**(3) The correction: §9.15's geometry1 number was optimistic.** §9.15 reported
independent-block geometry1 at spatial R² 0.770 from the single 40/5 split. Under CV the same
dataset and model gives **R² mean 0.203, median 0.468** — the fix is *more* damaging to ridge
than §9.15 claimed, not less. But the direction of the error is not the point; the point is
that a 5-scenario split was not a measurement. §9.15's table should be read as superseded by
this one wherever the two disagree. The same applies to the single-split shelf figures quoted
while this work was in progress (geometry4 R² 0.245, geometry5 R² 0.182): both sit between the
CV mean and CV median, which is what one expects of a single draw from a heavy-tailed
distribution, and neither should be cited.

**What survives, and what it costs.** Ridge remains the best model on geometry1 by det.MAE
(norm err 0.316) even though kNN beats it on R², so "layout randomisation breaks ridge" is
true of the shelf geometries and only partly true of independent-block geometry1. The honest
summary is that **layout randomisation moves this benchmark from one where no model can beat a
closed-form solve to one where a trivial nonparametric baseline already does** — which is
exactly the precondition for a neural operator experiment to be meaningful, and which the
original six-geometry dataset never satisfied. Whether an FNO beats kNN here is now an open
and worthwhile question; it was not one before.

**Caveat that still stands.** All three datasets are 45 real 3D-ICE solves over an 8–14
dimensional layout space, which is sparse. CV uses every scenario as held-out exactly once,
which removes the single-split lottery but not the sparsity. The heavy negative R² tail (mean
far below median for every model) is itself a symptom of that sparsity, and it is why both
statistics are reported rather than one.

### 9.15c "Linear-solvable" is a property of the representation, not of the dataset (2026-09-11)

Adding the shelf datasets to the cross-benchmark linearity audit
(`scripts/benchmark_linearity_audit.py`) produced a direct contradiction with §9.15b, and
resolving it is more informative than either result alone.

| dataset | audit: eff. DOF | audit: linear spatial R² | audit verdict | §9.15b ridge R² (CV) |
|---|---|---|---|---|
| geometry4, shelf | 7.6 | **0.962** | LINEAR-SOLVABLE | **−0.667** |
| geometry5, shelf | 3.2 | 0.418 | discriminative | −2.305 |
| geometry6, shelf | 2.5 | 0.536 | discriminative | −2.866 |

On geometry4 the two disagree completely: the audit says a linear fit reaches R² 0.962, and
the baselines say ridge scores −0.667 on the same 45 files.

**The two are measuring different input representations, and both are correct.**

- The **audit** feeds the *full per-cell power field* (56,000 cells for geometry4) and fits a
  linear map from that field to the temperature field.
- **`scripts/baselines.py`** feeds the compact vector every surrogate paper actually uses:
  per-block mean powers, per-block positions, HTC, ambient, TSV density — 8 block features
  plus boundary scalars for geometry4.

Steady-state conduction is exactly linear in the per-cell source distribution, so a linear map
from the *field* is the physically correct hypothesis class and should do well — which is what
the audit measures. It stops doing well only insofar as moving a chiplet also moves *silicon*,
changing the operator itself. That is why the audit's three shelf rows order as they do:
geometry4 moves two chiplets on one tier and the material map barely changes (R² 0.962), while
geometry5 and geometry6 move many chiplets including stacked HBM, changing the material map
substantially (R² 0.42–0.54).

The compact representation is a different matter. Block-mean power plus a block centroid
throws away *where within the block* the heat is and how the moved silicon reshapes the
conduction path, and no linear function of those ~10 scalars recovers the field once the
layout varies. That is what §9.15b measures, and it is the representation that matters for
the benchmark's stated purpose, because it is the one surrogate models are given.

**Three consequences.**

1. **§9.14's framing needs one more correction.** The question "is this benchmark
   linear-solvable?" has no answer without naming the input representation. The original
   six-geometry dataset was linear-solvable in *both* representations, which is why it was
   uninformative. The shelf datasets are linear-solvable in the field representation (mostly)
   and firmly not in the compact one.
2. **This is a testable prediction for a neural operator.** An FNO consumes the full power
   *field*, not the compact vector — the same representation in which the audit says
   geometry4 is still 96% linearly explained. So the prediction is that an FNO should do well
   on geometry4-shelf and struggle on geometry5/6-shelf, and that its margin over
   `baselines.py` ridge will be largest on geometry4 (where ridge's representation is
   impoverished but the task is not hard) rather than on geometry6 (where the task itself is
   hard). We have not run this. It is the sharpest experiment this benchmark now supports.
3. **The DOF diagnostic did not survive contact with these datasets.** §9.15 proposed layout
   dimensionality as the thing that predicts linear-solvability. The audit's effective-DOF
   estimate is *lowest* for geometry5/6 (2.5–3.2) — the two datasets that are hardest — and
   highest for geometry4 (7.6), the easiest. Effective DOF is computed from the input PCA
   spectrum, so it measures variability of the *power field*, which is not what layout
   randomisation primarily changes. It should not be read as a difficulty score.

**Caveat on the audit numbers, which applies to §9.13's table too.** The audit scores on a
single held-out 20% split. For the IC-ThermBench scopes that is 400+ fields and is fine; for
our 45-scenario datasets it is ~9 fields, which is precisely the sample size §9.15b showed to
be unreliable. The audit's rows for our own datasets are therefore screening indicators, not
measurements, and where they conflict with a cross-validated number the cross-validated number
wins. The geometry4 contradiction above is real and representational, not a sampling artefact
— the gap is far too large for that — but the specific value 0.962 should not be quoted to
three figures.

### 9.16 Calibrating the diagnostic against canonical PDE benchmarks (2026-09-12)

Until now the linearity audit had been run on thirteen datasets, **all of them thermal** — ten
of ours plus IC-ThermBench. That supports "a diagnostic for 3D-IC thermal benchmarks" and not
the broader claim contribution 9 makes. This section adds the canonical operator-learning
benchmarks from the physics literature as external calibration.

**Data.** PDEBench [Takamoto et al., NeurIPS 2022 D&B; DaRUS doi:10.18419/darus-2986]. Its
files are 1.3 GB (Darcy) to 10 GB (Navier–Stokes) each, far beyond what this probe needs, so
`scripts/pde_benchmark_data.py` reads only the leading N samples over HTTP range requests and
caches them (Darcy 1,000 samples = 66 MB, ~50 s). No full download is required to reproduce.

#### 9.16a The ladder

Detrended spatial R² of the best linear fit. **This is the only cross-benchmark-comparable
column** — see §9.16c.

| benchmark | equation | n | pca k | linear spatial R² | verdict |
|---|---|---|---|---|---|
| pdebench/diffusion-reaction | diffusion-reaction, 2-species | 300 | 2 | **0.0002** | discriminative |
| pdebench/burgers (t=20) | Burgers, nonlinear | 400 | 16 | **−0.077** | discriminative |
| ours/geometry5-shelf | heat, layout varied | 45 | 4 | 0.418 | discriminative |
| ours/geometry6-shelf | heat, layout varied | 45 | 4 | 0.536 | discriminative |
| pdebench/darcy-beta1.0 | Darcy, coefficient varied | 45 | 4 | **0.816** | discriminative |
| pdebench/darcy-beta1.0 | " | 1000 | 32 | 0.852 | discriminative |
| ours/geometry1-layout | heat, layout varied | 45 | 8 | 0.863 | discriminative |
| ours/geometry4-shelf | heat, layout varied | 45 | 8 | 0.962 | LINEAR-SOLVABLE |
| ours/geometry1-fixed | heat, fixed layout | 45 | 8 | 0.988 | LINEAR-SOLVABLE |
| **pdebench/shallow-water** | shallow water, 1-parameter | 300 | 64 | **0.9999** | **LINEAR-SOLVABLE** |

**Three things this establishes.**

1. **The diagnostic is validated on external ground truth.** It places Burgers — nonlinear
   through the $u\,\partial_x u$ term — at the maximally discriminative end (R² −0.077, i.e. a
   linear probe explains *none* of the spatial structure), with nothing tuned to make that
   happen.
2. **Our original fixed-placement benchmark sat at the solved extreme** (0.988), further from
   Burgers than any external benchmark measured.
3. **The layout fix moved our benchmark past Darcy in difficulty.** geometry5/6-shelf
   (0.42–0.54) are *less* linearly explained than PDEBench Darcy (0.816). This holds **at
   matched sample size and matched model capacity**: Darcy re-run at n=45 with pca_k=4, the
   exact regime our shelf datasets run in, still scores 0.816. It is therefore not an artifact
   of our datasets being small. (Darcy being comparatively easy for linear methods is
   consistent with existing criticism of it as an operator-learning benchmark.)

#### 9.16b Navier–Stokes: the missing baseline is persistence, not the mean

The first Navier–Stokes run returned **R² 0.9577, verdict LINEAR-SOLVABLE** — an absurd result
for the canonical nonlinear PDE, and it was ours, not the benchmark's. Predicting
$v(t+\Delta)$ from $v(t)$ with a 2-step gap out of 1000 is nearly the identity map: input and
output correlate at **0.994**.

| baseline on `pdebench/navier-stokes` (stride 2) | detrended R² |
|---|---|
| mean field | **−0.357** |
| **persistence — "the field is unchanged"** | **0.9997** |
| linear probe | 0.9577 |

**The linear fit is worse than not modelling at all**, by 0.042. And critically, **the
mean-field baseline does not catch this** — it scores −0.357, making the task look hard. A
benchmark can pass a trivial-baseline check and still be trivial, if the trivial baseline
checked is the wrong one.

Sweeping the time gap shows persistence dominates throughout. **All rows below use a constant
300 samples**; an earlier version of this table did not, and is corrected here — see the note.

| time stride | linear R² | persistence R² | linear − persistence |
|---|---|---|---|
| 2 | 0.9354 | **0.9999** | −0.065 |
| 50 | 0.6459 | **0.9370** | −0.291 |
| 200 | −0.0504 | **0.2139** | −0.264 |

A linear operator never beats persistence on this data at any gap — the correct outcome for a
nonlinear chaotic system, and the opposite of what the original verdict said.

> **Correction (2026-09-12).** The first version of this sweep stepped the pair start time
> *by* the stride, so longer gaps yielded fewer windows: 300 / 76 / **16** samples at strides
> 2 / 50 / 200. The stride-200 row therefore had ~12 training samples, and its linear score
> was depressed by sample starvation rather than by the time gap — it read −0.276 then and
> reads **−0.0504** at matched n. Two consequences. (a) The numbers above replace the earlier
> ones. (b) The claim that the gap to persistence *widens* with stride was wrong: at matched n
> it is −0.065, −0.291, −0.264, widening to stride 50 and then flat. What survives unchanged
> is the finding itself — persistence beats the linear fit at every gap — because persistence
> involves no fitting and so was never affected by the sample count. The loader now uses
> overlapping windows to hold n constant.

**Burgers behaves oppositely, which is the control that makes the Navier–Stokes reading
sound.** On `pdebench/burgers-nu0.01` persistence scores **−3.686**, far *worse* than the
linear probe's −0.077. The two nonlinear benchmarks fail a linear fit for different reasons,
and separating them requires both baselines:

| benchmark | linear R² | persistence R² | reading |
|---|---|---|---|
| burgers (t=20) | −0.077 | **−3.686** | field changes a lot; nothing cheap works — genuinely hard |
| navier–stokes (stride 2) | 0.958 | **0.9997** | field barely changes; the task does not test dynamics |

A benchmark is only informative when *both* trivial baselines fail. Reporting one without the
other is what lets a near-identity task look like a solved operator-learning problem.

**A false positive this gate caught in our own tool.** The persistence baseline was initially
computed whenever input and output shapes matched, which fired on IC-ThermBench — a
power → temperature map on a shared 64×64 grid — and reported a meaningless persistence R² of
−46.2, comparing watts to kelvin. Persistence is now computed only for benchmarks explicitly
declared as time-evolution (`TIME_EVOLUTION` in the audit), not inferred from shape. No
verdict was affected, because −46.2 fails both the `> 0.95` triviality test and the
`best_r2 > persistence` requirement, but the reported figure was nonsense.

**This is the same structural finding as the rest of the paper, in a fourth place.** Static
field prediction omits the mean/ridge baseline (§9.1); IC-ThermBench omits any non-neural
baseline (§9.13); **time-evolution PDE benchmarks omit persistence.** Neural-operator papers
routinely report next-step error on Navier–Stokes without one, and a reader cannot tell from
the paper whether a reported score beats assuming nothing moved.

The audit now computes persistence automatically whenever input and output share a shape,
**requires a linear fit to beat it** before returning LINEAR-SOLVABLE, and reports
`TRIVIAL(persist)` for tasks persistence alone solves.

**Caveat on our Navier–Stokes setup, stated plainly.** These samples are not i.i.d.: they are
time pairs from only 4 trajectories, consecutive pairs overlap (the target of one is the input
of the next), and fields are spatially subsampled 4×. The split is contiguous, so the test set
falls almost entirely in one held-out trajectory, but the effective sample size is far below
300. The persistence result is robust to this — it involves no fitting — but the linear-probe
R² should be read as optimistic.

#### 9.16c Two metric traps found by running these benchmarks

**(a) `rel L2` is not comparable across benchmarks, and the audit previously invited that
comparison.** It normalises by the field *including its mean*. Measured spatial-structure to
field-magnitude ratios: thermal **0.010** (≈325 K fields carrying ≈3 K of structure) versus
Darcy **0.560**. A mean-only predictor therefore scores rel L2 ≈ 0.01 on thermal and ≈ 0.56 on
Darcy. By rel L2 Darcy looks 5× *harder* than our shelf data; by detrended R² it is *easier*.
Detrended spatial R² is the comparable quantity and is what the verdict thresholds on, so no
published verdict was affected — but this is §9.1's offset-domination problem reappearing
inside our own tooling, and the table now carries an explicit warning.

**(b) A near-uniform target makes detrended R² meaningless, and we hit it.** Mapping Burgers
to its *final* timestep gave R² **−747.7**. The tell was that the mean-field predictor scored
**−209.9** — a mean predictor cannot be that bad, so the metric had broken, not the model.
Cause: viscous dissipation flattens the solution, leaving 41/400 targets with spatial std
< 1e-3, and detrended R² then divides by ≈0. Measured structure decay and moved the output to
t=20, where 60% of initial structure remains and no field is degenerate. The degenerate task
is kept registered as `pdebench/burgers-final` with a loud warning, since
`third_party/ic_thermbench/README.md` independently warns of the same failure mode.

**(c) `--max-samples` silently changed the test set, not just the sample count.** The cap took
the leading N samples (`X[:N]`) while `audit()` splits contiguously, so on an *ordered* dataset
both the training set and the test slice moved with N. IC-ThermBench ships an index-based
unshuffled split, and S2 read 0.7034 at N=2000 against 0.7812 at N=3000 — while S4 moved the
other way (0.5757 → 0.5683), which is the signature of slice idiosyncrasy rather than a
sample-size trend. Replacing head-truncation with a fixed-seed permutation cut the S2 spread
from **0.078 to 0.024**:

| cap | head-truncation | seeded permutation |
|---|---|---|
| N=2000 | 0.7034 | 0.6881 |
| N=3000 | 0.7812 | 0.7120 |
| spread | **0.078** | **0.024** |

The residual 0.024 is a genuine training-size effect and is expected; the removed 0.054 was an
artifact of which slice happened to land in test. **No published number changes**: §9.13's
"3.4–4.4× worse than Therm-FM" comes from `scripts/ic_thermbench_baselines.py` on the
benchmark's own official splits, and the only audit table in this report (§9.15c) covers
45-scenario datasets that were never truncated. It is fixed because the audit is released as a
reusable artifact under contribution 9, and a `--max-samples` flag that quietly redraws the
test set is a trap for anyone who picks it up.

**(d) Pooled R² is dominated by high-magnitude samples when their scale varies.** PDEBench
Darcy at beta=0.01 returns linear R² **−85.2** — and the *mean-field* baseline returns
**−67.6**. Neither is degeneracy (no near-zero-variance fields) nor a bug: per-sample solution
magnitude spans **545×** there against 14× at beta=1.0, and spatial R² is pooled over test
samples, so a handful of large fields dominate both sums of squares and the score stops
describing typical behaviour. The audit now reports `magnitude_spread` and warns above 50×.
beta=1.0 is the representative Darcy row and the one used in §9.16a.

This is the third distinct way detrended R² can mislead, all three found by running external
benchmarks rather than by inspection: near-zero-variance targets (b), cross-benchmark offset
domination (a), and magnitude spread (d). Each needs its own guard, and each produces a number
that reads like a finding if unchecked.

#### 9.16e PDEBench 2D shallow water is a one-parameter family, and a linear fit solves it

Added 2026-09-12 after extending the audit to two further PDE families. This is the strongest
external result in this paper, and it is the defect §9.14 diagnoses in *our* benchmark, found
in a published one.

| benchmark | eff. DOF | linear spatial R² | persistence R² | verdict |
|---|---|---|---|---|
| **pdebench/shallow-water** | 4.5 | **0.9999** | −0.227 | **LINEAR-SOLVABLE** |
| pdebench/diffusion-reaction | 238.1 | **0.0002** | −538.1 | discriminative |

The two new families bracket every other row in the audit: shallow water is **more
linearly-solvable than our own fixed-placement geometry1** (0.9875), which this paper already
condemns as degenerate; diffusion-reaction is the hardest dataset measured.

**Why shallow water is solvable — measured, across all 1000 trajectories.** Its initial
conditions are not a rich family:

- every initial field takes exactly **two values**, 1.0 and 2.0 (a binary dam);
- the dam centre is **exactly (63.5, 63.5) in all 1000 samples** — measured min = max, standard
  deviation **0.000000**;
- **only the radius varies**, over 7.74–18.02 (81 distinct areas).

So the entire benchmark is a **one-parameter family**: a radially symmetric dam break at a
fixed centre with a fixed height ratio, differing only in radius. A linear function of that
single radius explains **78.5%** of the target field's variance on its own. The target needs
**3 principal components for 95%** of variance and 5 for 99%.

**Verified independently, not just by the audit.** Written from scratch (PCA-ridge, no shared
code), at n=1000, with a **random** split rather than the audit's contiguous one, to rule out
trajectory ordering:

| pca_k | detrended R² (n=1000, random split) |
|---|---|
| 2 | 0.9823 |
| 3 | 0.9913 |
| 8 | 0.9988 |
| 64 | **0.99996** |

Two principal components already reach 0.982, and results are stable across λ spanning six
orders of magnitude — this is a low-dimensional solution manifold, not an overfit. The n=300
and n=1000 results agree to three decimal places.

**Why this matters more than our own negative result.** §9.2–§9.3 argue that a scenario space
parameterised by a handful of scalars produces a near-linear solution manifold, and that this
makes a benchmark unable to distinguish architectures. Our fixed-placement benchmark had ~8
such scalars. **PDEBench 2D shallow water has one.** It is used as an operator-learning
benchmark, and a closed-form linear fit with three components reproduces its solutions at
R² 0.991 with no training and no GPU.

**Scope, stated carefully.** (a) This is the fixed-horizon task $h(t{=}0) 	o h(t{=}20)$, the
standard FNO-style operator-learning setup; PDEBench also supports autoregressive rollout over
all 101 steps, which is a different and harder task we have not audited. (b) The finding is
about *this data file* (`2D_rdb_NA_NA.h5`, the only shallow-water file in the release), not
about the shallow-water equations, which are perfectly capable of rich behaviour under a
broader initial-condition family. (c) It is not a criticism of PDEBench as a data release,
which documents its generation process; it is a criticism of using this file to claim an
architecture has learned an operator. (d) Persistence scores −0.227 here, so this is a
different defect from the Navier–Stokes one in §9.16b — the task genuinely evolves, the
*scenario family* is what is impoverished.

**Diffusion-reaction is the control.** Same file format, same loader, same task construction,
effective DOF 238 versus 4.5, and a linear fit explains **0.02%** of the structure. The
diagnostic separates the two by three orders of magnitude in DOF, which is what a calibrated
instrument should do.

#### 9.16d What this does and does not license

It licenses: *the diagnostic has been calibrated across three PDE families (Darcy, Burgers,
Navier–Stokes) plus two thermal benchmark suites, and correctly orders them.* It does **not**
license any claim that these benchmarks are defective — Darcy and Burgers behave exactly as
their physics predicts. The defect found is in *reporting practice* on time-evolution tasks,
where persistence is absent, and that is stated as a practice recommendation rather than a
criticism of PDEBench, which is a data release and does not itself claim baselines.

## 11. Conclusion

> **Rewritten 2026-09-10.** The previous conclusion claimed ridge "reconstructs the spatial
> temperature field at R² = 0.999" and "extrapolates ... at R² > 0.9", and framed the
> dataset release as a contribution. All three were stale or unsupportable: the R² figures
> came from §9.1b's superseded pre-regime-fix dataset (§9.5 had already recorded that the
> regime fix moved them), and IC-ThermBench (arXiv:2608.23977, Aug 2026) is now an open
> 2.5D/3D-IC thermal benchmark with 50,000 samples against this repo's 275. Corrected below
> rather than quietly adjusted.

We release a six-geometry, 275-simulation 3D-IC thermal dataset with its full generation
pipeline. This is no longer a novel contribution on its own — IC-ThermBench (arXiv:2608.23977)
is larger, has a unified evaluation pipeline, and defines five generalization scopes — so the
dataset should be read as the substrate for the finding below, not as the finding.

> **Second revision, 2026-09-10 (same day).** After the rewrite below was written, we ran
> this project's own baseline tooling against IC-ThermBench (§9.13). **The linear baseline
> loses there, clearly, on every scope.** That falsifies the strong form of this paper's
> claim, so the framing below is narrowed accordingly: the contribution is a *diagnostic*
> and what it reveals, not an assertion that linear models suffice for this domain.

The finding is a negative result we believe is more useful than the surrogate accuracy figures
we set out to produce. Measured on the current dataset (§9.1a, 2026-09-09), closed-form ridge
regression reconstructs the spatial temperature field at **spatial R² 0.89–0.99** across the
six geometries — beating or matching every other baseline, and beating plain kNN only
narrowly, with kNN actually ahead on geometry3. No neural architecture tried in this project
has beaten it on that metric. Out-of-distribution behaviour is materially worse than
in-distribution (§9.5) but has not been re-measured since the 2026-08-06 dataset correction
and should not be quoted numerically until it is.

**The sharper result, and the one we would lead with now (§9.12c).** Ridge's error is not
uniformly distributed over the field. On geometry1 its mean absolute *peak-temperature* error
is **3.76 K** against a whole-field detrended MAE of **0.407 K** — roughly 9× worse exactly
where thermal sign-off decisions are made. Under 5-fold cross-validation it recovers 9.3% of
the 100 hottest cells and places the peak within 2 mm in 16% of scenarios; on geometry6 the
*trivial mean-field predictor* recovers more of the hot region than ridge does. So the honest
statement is not "a linear model solves 3D-IC thermal prediction" but:

> **A linear model saturates the metric this field usually reports, while failing the metric
> the application actually needs.**

That reframing is the contribution, and it is not a criticism of ridge alone — every model
tried here, neural included, fails hotspot localisation in absolute terms.

Four practices follow. First, **report a linear baseline** — it costs milliseconds and bounds
what any architecture can claim to contribute. IC-ThermBench's eight baselines, published this
year for exactly this domain, contain no non-neural model at all, so this remains unaddressed.
Second, **report spatially-detrended error**: raw MAE on 3D-IC thermal fields is dominated by
a per-scenario offset, and a constant predictor can look competitive while capturing no spatial
structure. Third, **report hotspot metrics separately from field metrics, and state whether
the peak is well-posed** — argmax-based localisation is meaningful on geometry1 (100 hottest
cells span 324 µm) and close to meaningless on geometry6 (9402 µm, several near-equal peaks).
IC-ThermBench's Tmax-Err and Top-50 MAE are convergent evidence for the second half of this;
localisation distance remains unreported even there.

Fourth, and learned the hard way, **cross-validate before reporting an architecture ranking
from a small test split**. This project measured an apparent neural win over ridge on hotspot
localisation from a 5-scenario split, caveated it, and had it reverse completely under 5-fold
cross-validation on the same geometry — for both configurations tested (§9.12d). Five-scenario
test splits are what this benchmark and its pilots ship, and they are enough to invert a
ranking in the flattering direction.

**What the linear check is actually for (§9.13).** Running this project's own baseline against
IC-ThermBench settles the scope of the claim. There, the linear baseline **loses** — 3.4–4.4×
worse than Therm-FM on S2–S4, worse than all eight of their neural baselines. So "a linear fit
is competitive on 3D-IC thermal benchmarks" is false as a general statement, and we say so.
The diagnostic is what generalises, and it is diagnostic **in both directions**: on our dataset
it exposed a benchmark that was near-linear-solvable, i.e. a design fault we then fixed
(§9.5); on IC-ThermBench it certifies a benchmark that genuinely discriminates. A benchmark
that passes this check has earned the architectures evaluated on it. The field ran it in
neither case, and it costs minutes.

Two things that only became visible by running it. **(a)** Most of the linear model's deficit
on IC-ThermBench is *missing layout conditioning*, not nonlinearity in the source: the
substrate geometry takes ~10–20 discrete values while the power map varies per sample, and
fitting one operator per geometry removes 38–61% of the error (§9.13a). **(b)** On their
structural-OOD scope, the best neural model is only **1.9× better than predicting the training
mean** (15.51 K vs 29.64 K), while the linear operator is unstable rather than merely
inaccurate (§9.13c). Neither is visible in a results table containing no trivial baseline.

**We then fixed our own benchmark, and report the attempt that failed (§9.15, §9.15b).**
Recording what a benchmark *would* need is cheap; doing it is the test of the diagnosis. We
randomised chiplet placement per scenario so the thermal operator varies rather than only its
inputs. The first attempt — translating each chiplet rigidly — **did not work**, even though
it moved the sources more than IC-ThermBench does by support-overlap IoU (0.472 vs 0.449);
ridge's relative error rose by 5–8% and on geometry4 its R² *improved*. The second attempt,
placing chiplets independently, did. Cross-validated over 45 scenarios on each of three
re-laid-out geometries, **ridge becomes the worst of the four baselines**, losing to a
3-nearest-neighbour lookup; on the densest package (geometry6) its median spatial R² is
**negative** and its detrended error **exceeds the signal** (norm err 1.035, 26/45 scenarios
worse than the field's own mean). That is the first configuration in this project on which
training a neural operator is a question worth asking.

**And the diagnosis needed one more correction (§9.15c).** Asking whether a benchmark is
"linearly solvable" is ill-posed without naming the input representation. The same 45
geometry4 files admit a linear fit at R² 0.962 from the full per-cell power field and fail at
R² −0.667 from the compact block-summary vector that surrogate models are conventionally
given. Conduction *is* exactly linear in the per-cell source, so the field representation is
the physically correct hypothesis class; it degrades only insofar as moving a chiplet also
moves silicon and changes the operator. The compact vector discards precisely that. This
corrects §9.14 and §9.15, both of which asserted linear-solvability as a property of a
dataset, and it supplies a falsifiable prediction we have not yet tested: an FNO consumes the
field, so it should win by the largest margin on geometry4-shelf — where ridge's
representation is impoverished but the task is not hard — rather than on geometry6-shelf,
where the task itself is hard.

A methodological point applies to both results and cost us two retractions. **A 40/5 split is
not a measurement.** Every single-split figure from the layout work (geometry1 R² 0.770,
geometry4 0.245, geometry5 0.182) was overturned by 5-fold cross-validation over the same 45
scenarios, and the distribution of per-scenario R² is heavy-tailed enough that mean and median
must both be reported. `scripts/layout_cv.py` exists so this is a default rather than an
afterthought.

We release the baseline, hotspot, cross-validation and linearity-audit tooling
(`scripts/baselines.py`, `scripts/hotspot_eval.py`, `scripts/layout_cv.py`,
`scripts/benchmark_linearity_audit.py`) so these conditions can be checked rather than
assumed, together with the 180 layout-randomised 3D-ICE solves the fix produced.

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

16. Barua, B. P., Udoy, M. R. I. & Aziz, A. "A Review of Multiscale Thermal Modeling in Heterogeneous 3D ICs." arXiv:2604.03290, 2026.

17. McGreivy, N. & Hakim, A. "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations." *Nature Machine Intelligence*, 2024. DOI: 10.1038/s42256-024-00897-5. arXiv:2407.07218.

18. Chen, T. et al. "Leakage power dependent temperature estimation to predict thermal runaway." *ICCAD*, 2006.

19. "ATSim3D: Towards Accurate Thermal Simulator for Heterogeneous 3D-IC Systems Considering Nonlinear Leakage and Conductivity." arXiv:2601.11050, 2026.

20. Zhu, K., Huang, D., Costero, L. & Atienza, D. "3D-ICE 4.0: Accurate and efficient thermal modeling for 2.5D/3D heterogeneous chiplet systems." arXiv:2512.05823, 2025. (The simulator this project runs on — cited here specifically because it establishes that heterogeneous-substrate thermal modeling, the capability geometry7 depends on, is the tool's own stated purpose, not something this project extended.)

21. "Design and verification of silicon bridge in 2.5D advanced package based on universal chiplet interconnect express (UCIe)." *Microelectronics Reliability* (ScienceDirect), 2025.
