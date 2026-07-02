# FourierPINN: A Physics-Informed Neural Surrogate for 3D-IC Thermal Analysis with Uncertainty Quantification and Explainability

**Rajul Kabeer**  
*Manuscript in preparation*

---

## Abstract

Thermal analysis of three-dimensional integrated circuits (3D-ICs) is a computationally intensive bottleneck in early-stage design exploration. Compact simulators such as 3D-ICE provide fast steady-state solutions but still require seconds-to-minutes per configuration, making parametric studies over power maps and cooling conditions expensive. We present FourierPINN, a physics-informed neural network surrogate trained on 3D-ICE simulation data across five benchmark geometries spanning single-die, stacked-die with through-silicon via (TSV) arrays, and server-class configurations. The model encodes spatial coordinates via random Fourier features to overcome spectral bias, conditions on layer identity through a learned embedding, and is trained using a curriculum that transitions from supervised data-fitting through progressive physics regularisation with Neural Tangent Kernel-adaptive loss weighting. To our knowledge this is the first PINN surrogate specifically designed for the multi-layer 3D-IC thermal stack, the first to treat TSV density as a continuous parametric input, and the first to apply targeted explainability methods — PDE residual mapping, engineering sensitivity maps, integrated gradients, and Monte Carlo Dropout uncertainty — to a chip-level thermal surrogate. Training data, geometry definitions, and trained checkpoints will be released openly.

---

## 1. Introduction

As chiplet integration and 3D stacking become mainstream packaging approaches, thermal design must keep pace with increasing power densities and more complex heat flow paths. Die stacking with TSV interconnects creates new thermal coupling mechanisms: each die heats the one above it, TSVs provide vertical thermal shortcuts, and the effective thermal resistance from junction to coolant is now distributed across multiple material interfaces. Evaluating even a single configuration requires solving the three-dimensional heat equation over a heterogeneous multi-layer domain.

Compact thermal simulators such as 3D-ICE [Sridhar et al., 2010] model this as a finite-difference RC-network on a structured Cartesian grid and can produce steady-state solutions in seconds. However, design exploration over power patterns, cooling intensities, and TSV densities requires thousands of such evaluations. Full parametric sweeps remain slow, and compact simulators have limited expressiveness — they cannot represent individual TSV columns, spatially-varying convective coefficients, or transient workload effects.

Machine learning surrogates offer a different trade-off: expensive offline training amortised over fast online inference. Prior work has applied convolutional neural networks [Zhang et al., 2025], graph neural networks [WarPGNN, 2026], and operator-learning approaches including FNO variants [Self-Attention U-Net FNO, 2025] to chip thermal prediction. Physics-informed neural networks (PINNs) add physics constraints — the heat equation and boundary conditions — as additional loss terms, which regularise training and improve generalisation beyond the training distribution. ThermPINN [Cheng et al., 2024] demonstrated PINNs for full-chip 2D thermal analysis. No prior work has applied a PINN to the 3D multi-layer package stack (die + TIM + spreader + heat sink) with TSV parametric variation, nor provided explainability analysis for any chip thermal neural surrogate.

This paper makes the following contributions:

1. A five-geometry benchmark dataset of 100 3D-ICE simulations covering a single-die mobile stack, three stacked-die TSV variants (3%, 5%, 10% density), and a server-class large die.
2. FourierPINN: a physics-informed surrogate with Fourier feature encoding, learned layer embedding, and curriculum training using NTK-adaptive loss weights.
3. Monte Carlo Dropout uncertainty quantification integrated at training time.
4. Three post-hoc explainability methods — PDE residual maps, engineering sensitivity maps (thermal influence coefficients), and targeted Integrated Gradients — applied to a chip thermal surrogate for the first time.

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

Five geometries span the design space from a mobile-class single die to a server-class large die and 3D-stacked configurations with TSV arrays. All layers are modelled with homogenised material properties. TSV regions use the arithmetic-mean effective conductivity $k_{eff} = (1-\phi) k_{Si} + \phi k_{Cu}$, consistent with the 3D-ICE ground-truth simulator.

| Geometry | Type | Die size | Layers | Mesh | Notes |
|---|---|---|---|---|---|
| geometry1 | 2D stack | 10 × 10 mm | 6 | 100×100×40 | Mobile/desktop single die |
| geometry2a | 3D stack | 8 × 8 mm | 10 | 80×80×72 | TSV density 3% |
| geometry2b | 3D stack | 8 × 8 mm | 10 | 80×80×72 | TSV density 5% |
| geometry2c | 3D stack | 8 × 8 mm | 10 | 80×80×72 | TSV density 10% |
| geometry3 | 2D stack | 25 × 25 mm | 6 | 100×100×40 | Server-class, 8 core clusters |

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

**Dataset.** Each geometry has 15 training and 5 test scenarios. Scenarios sweep power density (0.1–20 W/cm²), HTC (500–10,000 W/m²·K), and ambient temperature (25–85°C) across six spatial power patterns (uniform, hotspot, checkerboard, gradient, dual-hotspot, extreme hotspot). Ground truth temperatures are generated by the 3D-ICE Emulator running under WSL2.

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

*Results pending training completion. Expected: mean MAE < 2 K on test scenarios for all geometries; hotspot temperature error < 3 K; hotspot location error < 500 µm.*

---

## 10. Discussion

*To be completed after results.*

Key discussion points anticipated:
- Whether TSV density generalisation holds across geometry2a/b/c from a single model
- PDE residual map correlation with prediction error — does high residual predict high error?
- Integrated Gradients validity: does attribution profile match physical intuition (power dominates at hotspot; z-position dominates in heat sink)?
- MC Dropout calibration: is predictive std a reliable proxy for actual error?

---

## 11. Conclusion

*To be completed after results.*

We have presented FourierPINN, a physics-informed thermal surrogate for 3D-IC package stacks with integrated explainability. The combination of accurate surrogate modelling with physics-residual maps, thermal influence coefficients, targeted attribution, and uncertainty quantification addresses a gap between existing ML thermal tools (accurate but opaque) and engineering design needs (fast, trustworthy, interpretable).

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
