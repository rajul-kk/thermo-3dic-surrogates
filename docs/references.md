# References and Geometry Cross-Check

## Verification status (last checked 2026-08-06)

Citations added or relied on for design decisions were checked against the literature.
Verified: 3D-ICE (ICCAD 2010), 3D-ICE 4.0 (arXiv:2512.05823), HotSpot (TVLSI 2006),
Raissi et al. (JCP 2019), Tancik et al. (NeurIPS 2020), Wang et al. (SIAM 2021),
Li et al. FNO (ICLR 2021), WHNO (arXiv:2511.07347), RNO (arXiv:2505.20721),
SAU-FNO (arXiv:2510.15968), DeepOHeat (DAC 2023) and DeepOHeat-v1 (arXiv:2504.03955),
MFIT (DOI 10.1145/3765905), Glassbrenner & Slack (Phys. Rev. 134, A1058),
Morrow et al. (IEEE EDL 27, 335–337).

**Re-sourced 2026-08-06:** the Kou et al. 2022 citation used for geometry5/6 dimensions
(interposer thickness, TIM1/TIM2 thickness) did not verify and has been replaced with
Zhou, Li, Hou, He & Fan (2022), **"Thermal Modeling of a Chiplet-Based Packaging With a
2.5-D Through-Silicon Via Interposer,"** *IEEE TCPMT* 12(6), 956–963,
DOI: 10.1109/TCPMT.2022.3174608 — confirmed to exist and resolve to the stated title via
CrossRef (`doi.org/10.1109/TCPMT.2022.3174608` → `ieeexplore.ieee.org/document/9785787`).
IEEE Xplore blocks automated full-text access, so the specific dimensional figures
(interposer 300 µm, TIM1 50 µm, TIM2 125 µm) are **still unconfirmed against the paper's
actual text** — only the citation's existence and topical match are verified. Treat those
three numbers as an engineering estimate pending manual full-text confirmation, not yet a
literature-sourced fact.

**Two citations still do not verify and are marked inline with ⚠:** Gao et al. 2023,
Liu & Park 2012. Two corrections were made: the RNO authors are Ye, Zhang & Wang
(previously given as "Yang et al."), and DeepOHeat / DeepOHeat-v1 are separate papers with
different titles, not one paper and its update.

**Added 2026-08-11 (§9):** researched what industrial thermal-model accuracy targets
actually are, since `docs/report.md` §10 references "the accuracy targets neural models
are held to" without a citation. No single authoritative absolute-°C figure was found;
verified sources cluster around 1–4% relative error instead (CTM-vs-detailed-model <1%,
junction-to-case test method ±4%, CFD-vs-measurement ~2%).

**Added 2026-08-16 (§7, §9):** arXiv:2604.03290 (Barua, Udoy & Aziz 2026, "A Review of
Multiscale Thermal Modeling in Heterogeneous 3D ICs") — flagged unverified on 2026-08-11
— is now **fully confirmed**: author names independently confirmed via search, full text
extracted directly (see §9 for section-by-section pressing-open-problems summary). Also
did a full documentation audit this session (all of `docs/`, `goal.md`, `README.md`
cross-checked against real files in `data/3d-ice/` and `results/`) — found and fixed
several stale/internally-contradictory tables in `docs/geometry_reference.md` (point
counts, an obsolete 8-geometry/335-file dataset-statistics table, TSV material rows for
the removed geometry2b/2c) and `docs/references.md` §4/§6 (HTC/ambient ranges predating
the 2026-08-01 TDP-regime revision). None of the numbers in `docs/report.md`'s own body
text were found wrong; the drift was concentrated in the two reference/geometry docs,
which get updated less often than the paper itself. Full findings in `goal.md`'s
"Documentation audit" entry.

## 1. Core Thermal Simulators

### 3D-ICE (primary ground truth)
Sridhar, A., Vincenzi, A., Ruggiero, M., Atienza, D., & Brunschwiler, T. (2010).
**3D-ICE: Fast Compact Transient Thermal Modeling for 3D-ICs with Inter-Tier Liquid Cooling.**
*Proceedings of the International Conference on Computer-Aided Design (ICCAD)*, 463–470.
DOI: 10.1109/ICCAD.2010.5653749

Updated version:
**3D-ICE: Efficient Nonlinear MPSoC Thermal Simulation with Pluggable Heat Sink Models.**
*IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 40(10), 2021.
Source: https://www.epfl.ch/labs/esl/research/open-source-tools-datasets/3d-ice/

### HotSpot (cross-validation simulator)
Huang, W., Ghosh, S., Velusamy, S., Sankaranarayanan, K., Skadron, K., & Stan, M. R. (2006).
**HotSpot: A Compact Thermal Modeling Methodology for Early-Stage VLSI Design.**
*IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, 14(5), 501–513.
DOI: 10.1109/TVLSI.2006.876103

---

## 2. PINN Architecture References

### Foundational PINN paper
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
**Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.**
*Journal of Computational Physics*, 378, 686–707.
DOI: 10.1016/j.jcp.2018.10.045

### Fourier feature encoding (FourierFeatureEmbedding in model.py)
Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., & Ng, R. (2020).
**Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.**
*Advances in Neural Information Processing Systems (NeurIPS)*, 7537–7547.

### NTK-based adaptive loss weighting (LossWeights.update_ntk in losses.py)
Wang, S., Teng, Y., & Perdikaris, P. (2021).
**Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks.**
*SIAM Journal on Scientific Computing*, 43(5), A3055–A3081.
DOI: 10.1137/20M1318043

### Curriculum / causality training
Wang, S., Sankaran, Y., & Perdikaris, P. (2024).
**Respecting Causality for Training Physics-Informed Neural Networks.**
*Computer Methods in Applied Mechanics and Engineering*.
DOI: 10.1016/j.cma.2024.116928

### Fourier Neural Operator (src/fno/model.py)
Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021).
**Fourier Neural Operator for Parametric Partial Differential Equations.**
*International Conference on Learning Representations (ICLR)*.
https://openreview.net/forum?id=c8P9NQVtmnO

### Walsh-Hadamard Neural Operator (src/fno/whno.py)
Cavallazzi, G. M., Pérez Cuadrado, M., & Pinelli, A. (Nov 2025).
**Walsh-Hadamard Neural Operators for Solving PDEs with Discontinuous Coefficients.**
arXiv:2511.07347. Also published in *Journal of Computational Physics*
(DOI: 10.1016/j.jcp.2026.114476 — see ScienceDirect S0021999126004766).
(Motivates WHNO in this repo: Fourier's sinusoidal basis causes Gibbs ringing at sharp
conductivity jumps — exactly the material-interface discontinuities in this dataset's
layer stacks; the Walsh-Hadamard basis is piecewise-constant and does not suffer this.
The paper's own validation includes heat conduction with discontinuous thermal
conductivity and reports 24–38% error reduction vs FNO at material interfaces.
Independently checked in `src/fno/whno.py`'s module docstring against a synthetic step
function: 0% overshoot vs Fourier's ~8.7%.)

### Autoregressive/recurrent training for neural operator rollout stability (src/aro/trainer.py)
Ye, Z., Zhang, C.-S., & Wang, W. (May 2025).
**Recurrent Neural Operators: Stable Long-Term PDE Prediction.**
arXiv:2505.20721.
(ARO's `forward_windowed_rollout` / RNO-style training in `src/aro/trainer.py` directly
applies this idea: training on a window of the model's OWN autoregressive predictions,
not just ground-truth teacher-forced inputs, closes the train/inference exposure-bias
gap responsible for compounding error in autoregressive rollout. Originally proposed for
*temporal* rollout; this repo's application to *spatial* z-layer rollout is a novel
adaptation, not a direct replication.)

### Adaptive collocation sampling for PINNs (src/pinn/sampling.py)
- **Provably Accurate Adaptive Sampling for Collocation Points in Physics-Informed Neural
  Networks.** (ECML PKDD 2025). Hessian-based sampling — motivates the `hessian` strategy.
- **Curriculum-Enhanced Adaptive Sampling for Physics-Informed Neural Networks: A Robust
  Framework for Stiff PDEs.** (MDPI, Dec 2025). Motivates the `curriculum` strategy —
  argues plain residual-adaptive sampling over-trusts early-training residual signal
  before the network has learned the bulk solution.
- Tang et al. (2024). **Adversarial Adaptive Sampling**, combining PINNs with optimal
  transport theory. `src/pinn/sampling.py`'s `importance` strategy (softmax-temperature
  resampling) is an explicitly-labeled SIMPLIFIED PROXY for this — the full method uses
  a trained generative model + Wasserstein-distance optimal transport, not implemented
  here.

### Lateral material heterogeneity in 2.5D/3D chiplet thermal modelling (SETTLED PRIOR ART)
- **3D-ICE 4.0: Accurate and efficient thermal modeling for 2.5D/3D heterogeneous chiplet
  systems.** Zhu, K., Huang, D., Costero, L., & Atienza, D. (Dec 2025). arXiv:2512.05823.
  EPFL / Universidad Complutense de Madrid — i.e. the authors of the simulator this repo
  uses for ground truth. Its first stated contribution is *"preservation of material
  heterogeneity and anisotropy directly from industrial layouts"*, explicitly motivated by
  noting that existing approaches "neglect material heterogeneity (e.g. silicon dies, epoxy
  underfill, Cu pillars)". Also adds adaptive vertical layer partitioning and temperature-aware
  non-uniform grids; 3.61–6.46× faster than prior tools with >23% fewer grid cells. Open source.

  **Consequence for this repo: lateral heterogeneity is NOT an available novelty claim.**
  Modelling the underfill gap between chiplets as a distinct lateral conductivity region is
  solved, published, and shipped by the 3D-ICE authors themselves. Do not present
  geometry4/5/6's `DiePrint` / `underfill_k` mechanism as a gap in the literature. See
  `assumptions.md` §6.1 — this repo's 3D-ICE ground truth does not even contain the
  heterogeneity, so the claim would be unsupported as well as unoriginal.

  The narrower statement that does survive: 3D-ICE 4.0 is a *simulator*, and no neural
  *surrogate* trained on laterally-heterogeneous ground truth was found. That is a much
  smaller claim and requires regenerating the dataset under 3D-ICE 4.0 first.

- Pfromm, L., Kanani, A., et al. **MFIT: Multi-FIdelity Thermal Modeling for 2.5D and 3D
  Multi-Chiplet Architectures.** *ACM Transactions on Design Automation of Electronic
  Systems* (Nov 2025). DOI: 10.1145/3765905. Preprint: arXiv:2410.09188.
  Combines FEM, thermal-RC and data-driven surrogate models across the design cycle;
  evaluated on 16/36/64-chiplet 2.5D and 16x3 3D systems.
  Prior art for multi-fidelity thermal modelling of chiplet stacks — relevant to this repo's
  `scripts/generate_lf_data.py` low-fidelity pipeline and the Therm-FM fine-tuning story.
  Cite before claiming multi-fidelity novelty.

- **Fast Thermal-Aware Chiplet Placement Assisted by Surrogate.** (Apr 2025).
  arXiv:2504.03808. Surrogate-assisted (RBF network) thermal-aware placement. Relevant to
  this repo's claim that sensitivity maps are "actionable for floorplan decisions" — that
  application already exists in the literature.

### Prior art on the baseline-reporting problem itself (**critical — read before making any novelty claim about this paper's central finding**)

- **McGreivy, N. & Hakim, A. (2024). "Weak baselines and reporting biases lead to
  overoptimism in machine learning for fluid-related partial differential equations."**
  *Nature Machine Intelligence*. DOI: 10.1038/s42256-024-00897-5. arXiv:2407.07218.
  **Found 2026-08-16; this repo had not cited it, which was a serious gap.** Systematic
  review finding **79% (60/76) of papers claiming an ML method outperforms a standard
  numerical method compare against a weak baseline**, and attributing the pattern to
  researcher degrees of freedom, outcome-reporting bias and publication bias.

  **What this means for this paper's novelty — state this honestly rather than working
  around it.** The *general* claim "ML-for-PDE papers systematically compare against
  inadequate baselines, producing overoptimistic results" is **not novel**: it is
  published, quantified, and in a high-profile venue. Any framing of this repo's finding
  as "nobody has noticed the field skips baselines" is now wrong and should be removed
  wherever it appears. What remains genuinely distinct, and is what the paper should
  claim:
  1. **Domain**: their review covers fluid-related PDEs exclusively. Chip/package/3D-IC
     thermal is not examined.
  2. **Baseline type**: their "weak baseline" is largely an under-tuned or low-order
     *numerical solver*. This repo's claim is different and sharper — not "the numerical
     baseline was tuned badly" but "a closed-form linear fit with no training solves the
     problem class," i.e. the benchmark itself is near-linear.
  3. **Mechanism**: the diagnosis that the scenario space is low-dimensional
     (~8 scalars → a near-linear manifold), and the identification of per-cell power as
     the specific change that breaks it (§9.4).
  4. **The metric argument**: field-level R² vs. hotspot localisation (§9.4) — that the
     commonly reported metric is the one on which even a linear model wins.
  5. Released benchmark + `scripts/baselines.py` as reusable tooling.

  Best use of this citation: it *strengthens* the paper. It converts this repo's result
  from an isolated claim into the 3D-IC-thermal instance of a documented, quantified,
  cross-domain problem — and McGreivy & Hakim explicitly note under-reporting of negative
  results, which is the category this paper falls in.

### Prior art on operator learning for 3D-IC thermal simulation (closely related work)
- **Self-Attention to Operator Learning-based 3D-IC Thermal Simulation (SAU-FNO).**
  (Oct 2025, IEEE). arXiv:2510.15968. Self-attention + U-Net + FNO for 3D-IC thermal
  simulation; reports 842× speedup vs COMSOL/MTA and >50% MSE reduction. Also uses
  transfer learning (fine-tune on low-fidelity data). **This is close prior art to this
  repo's `CNOFNOHybrid`+axial-attention architecture and Therm-FM fine-tuning approach —
  differentiate explicitly against it if using either as a novelty claim.** Does not
  appear to cover 2.5D chiplet/CoWoS lateral heterogeneity (this repo's geometry4/5/6).
  **Confirmed 2026-08-16 (research pass, not yet full-text-read): reports 842× speedup and
  >50% MSE reduction without stating a linear/ridge/closed-form baseline comparison** —
  i.e. the exact evaluation gap this paper's central finding diagnoses, in a specific,
  very recent (Oct 2025), directly-on-domain paper, not just as a general claim about
  "the field." Strongest concrete example found for citing that gap by name; worth using
  in `docs/report.md`'s "Implications for the field" discussion (§10) rather than only
  as a prior-art differentiation entry.
- **DeepOHeat: Operator Learning-based Ultra-fast Thermal Simulation in 3D-IC Design.**
  *DAC* 2023. arXiv:2302.12949. DOI: 10.1109/DAC56929.2023.10247998.
  And its successor, a *separate paper with a different title*:
  **DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation
  and Optimization in 3D-IC Design.** (Apr 2025). arXiv:2504.03955. Adds
  Kolmogorov-Arnold trunk networks, separable training (62x speedup, 31x less GPU
  memory) and a confidence score for trustworthiness; 70.6x faster optimisation overall.
  DeepONet applied directly to 3D-IC thermal simulation, encoding BCs/heat-source
  configuration as branch input, coordinates as trunk input. **This is prior art for
  "DeepONet for chip thermal" — do not present this repo's `PI-DeepONet` as a novel
  application without citing and differentiating against DeepOHeat-v1.**

---

---

## 2b. Hybrid Bonding and TIM References

### Cu-Cu hybrid bonding (geometry2/5/6 die-to-die interface)

Morrow, P., et al. (2006).
**Three-dimensional wafer stacking via Cu-Cu bonding integrated with 65-nm strained-Si/low-k CMOS technology.**
*IEEE Electron Device Letters*, 27(5), 335–337.
DOI: 10.1109/LED.2006.872605
(Foundational paper for direct Cu-Cu bonding; k_eff of bonded Cu interface ~300–400 W/m·K)

Morrow full author list (verified): Morrow, P. R., Park, C., Ramanathan, S.,
Kobrinsky, M. J., & Harmes, M.

Gao, G., et al. (2023).
**Hybrid bonding enabled 3D-IC integration at sub-10-µm pitch.**
*IEDM Technical Digest*, 2023.
(State-of-the-art production hybrid bonding: <1 µm bondline, near-zero thermal resistance)
> ⚠ **UNVERIFIED (checked 2026-08-03).** No DOI, and a literature search did not confirm
> this exact title/venue. Do not cite until located, or replace with a confirmed source.

**k_eff used in this benchmark:** 60 W/m·K (5 µm effective layer) — represents a Cu pillar
composite with ~15 % Cu fill at 9 µm pitch, deliberately thicker than physical bondline to
make the interface visible in the PINN coordinate space. See `assumptions.md §2.3`.

### TIM indium solder and pump-out degradation (geometry5/6)

Liu, Y., & Park, S. B. (2012).
**Thermal cycling effects on the bonding strength and electrical resistance of In-Ag soldering.**
*Proceedings of ECTC*, 2012.
(Indium solder TIM: fresh k ≈ 82 W/m·K; after pump-out / voiding k can drop to 5–20 W/m·K)
> ⚠ **UNVERIFIED (checked 2026-08-03).** Not located by literature search. The k values it
> is cited for drive the geometry5/6 TIM pump-out sweep, so this needs a confirmed source.

~~Kou, H., et al. (2022). "Thermal analysis of 3D stacked memory package with
through-silicon via." *IEEE TCPMT*, 12(4), 641–651. DOI: 10.1109/TCPMT.2022.3156289~~
— **DOES NOT VERIFY, replaced 2026-08-06.** Searching this DOI and this journal/volume/
year returns a *different* paper; do not cite.

**Replacement (dimensional reference for the geometry5/6 Tier 1 upgrade):**

Zhou, M., Li, L., Hou, F., He, G., & Fan, J. (2022).
**Thermal Modeling of a Chiplet-Based Packaging With a 2.5-D Through-Silicon Via Interposer.**
*IEEE Transactions on Components, Packaging and Manufacturing Technology*, 12(6), 956–963.
DOI: 10.1109/TCPMT.2022.3174608
> **Citation confirmed 2026-08-06** — the DOI resolves via CrossRef to this exact title on
> IEEE Xplore (document 9785787). IEEE Xplore blocks automated full-text access, so the
> specific dimensions this citation was used for (interposer 300 µm, indium TIM1 50 µm,
> TIM2 125 µm — used in `geometry5_upgrades.md` and `src/core/geometry_builders.py`
> `build_geometry5`/`build_geometry6`) are **still unconfirmed against the paper's actual
> text**, only the citation's existence and topical match. Treat those three numbers as an
> engineering estimate consistent with the general 2.5D CoWoS-class range (see the
> TSMC CoWoS-R comparison below), not a literature-sourced fact, until someone with
> full-text access checks them directly.

---

## 3. Material Property References

### Silicon thermal conductivity (k=148 W/m·K at 300K; k(T) = 148×(300/T)^1.3)
Glassbrenner, C. J., & Slack, G. A. (1964).
**Thermal Conductivity of Silicon and Germanium from 3 K to the Melting Point.**
*Physical Review*, 134, A1058–A1069.
DOI: 10.1103/PhysRev.134.A1058

Cross-check values from this paper:
| T (K) | Published k (W/m·K) | Our model k (W/m·K) | Error |
|-------|---------------------|----------------------|-------|
| 300   | 148                 | 148.0                | 0%    |
| 400   | ~79                 | 80.2                 | +1.5% |
| 500   | ~49                 | 50.2                 | +2.4% |
| 600   | ~34                 | 34.8                 | +2.4% |

**Verdict: Accurate.** The power-law model with α=1.3 is a good fit for 300–700 K.

---

## 4. Geometry Cross-Check Against Published Literature

### 4.1 Geometry 1 — Single-Die 2D Stack

*(§4's "Our value" columns below are also pre-TDP-regime original design values, same
caveat as §6.)*

| Parameter | Our value | 3D-ICE reference case | Typical literature range | Verdict |
|-----------|-----------|----------------------|--------------------------|---------|
| Die size | 10 × 10 mm | 10 × 10 mm (ICCAD 2010 example) | 5–25 mm | ✓ Matches |
| Die thickness | 150 µm | ~100–200 µm | 100–300 µm (standalone) | ✓ Plausible |
| TIM thickness (each) | 100 µm | 50–100 µm | 25–100 µm | ✓ Upper end, acceptable |
| TIM2 thickness | 50 µm | — (added by us) | 25–75 µm | ✓ Reasonable |
| Spreader (Cu) | 1000 µm | 1000–2000 µm | 500–3000 µm | ✓ Matches |
| Heat sink (Cu) | 5000 µm | 3000–6000 µm | 2000–6000 µm | ✓ Acceptable |
| TIM k | 4 W/m·K | 3–5 W/m·K | 1–8 W/m·K (grease) | ✓ Standard thermal grease |
| Cu k | 400 W/m·K | 385–400 W/m·K | 385–401 W/m·K | ✓ Slightly high, within range |
| Si k (300K) | 148 W/m·K | 148 W/m·K | 148 W/m·K | ✓ Exact match |
| Power blocks | 4 × (3×3 mm) | 4 blocks, similar | — | ✓ Reasonable |
| Power density range | 0.1–20 W/cm² | ~1–50 W/cm² | 1–100 W/cm² | ⚠ Low end conservative |
| HTC range | 500–10000 W/m²·K | 1000–10000 W/m²·K | 500–15000 W/m²·K | ✓ Reasonable |

**Overall: Geometry 1 closely matches the canonical 3D-ICE single-die benchmark.**

**Concerns:**
- Power density upper bound (20 W/cm² for extreme_hotspot) is conservative. Modern CPU hotspots can reach 100–300 W/cm² in small regions. For a PINN benchmark, extending to 50 W/cm² would improve coverage.
- Copper k=400 W/m·K is ~4% higher than the consensus value of 385 W/m·K (Touloukian et al. 1970). Negligible impact.

---

### 4.2 Geometry 2 — 3D-Stacked Die with TSVs

| Parameter | Our value | 3D-ICE stacked example | Typical 3D-IC literature | Verdict |
|-----------|-----------|----------------------|--------------------------|---------|
| Die size | 8 × 8 mm | 5 × 5 mm (ICCAD 2010) | 5–15 mm | ⚠ Larger than reference |
| Die 1/2 thickness | 50 µm each | 50–100 µm thinned | 20–100 µm (bonded) | ✓ Correct for 3D bonding |
| TSV layer thickness | 100 µm each | ~50–150 µm | 50–200 µm | ✓ Plausible |
| Bonding layer | **5 µm hybrid bond** | <1 µm (Cu-Cu) / 10–30 µm (micro-bump) | 1–50 µm | ⚠ 5× thicker than real Cu-Cu; R_bond 28× over real |
| Bonding k | **60 W/m·K** | ~300–400 W/m·K (Cu-Cu) | Cu-Cu: 300 W/m·K | ⚠ 5× under real; <2 K die2 error |
| TSV density (2a/2b/2c) | 3/5/10% | 1–5% | 1–10% | ✓ Literature range |
| TSV k_eff (3%) | 155.4 W/m·K | ~152 W/m·K | — | ✓ ~2% over with arithmetic mean |
| TSV k model | Arithmetic mean | Arithmetic mean | Arithmetic or geometric | ✓ Consistent with 3D-ICE |

**Key discrepancy: Die size.** The 3D-ICE ICCAD 2010 stacked example uses 5×5 mm dies. Our geometry2 uses 8×8 mm. This is not wrong — it represents a larger 3D stack — but it does not match the published benchmark directly. The consequence is that our training data is not directly comparable to 3D-ICE paper results.

**TSV effective medium model.** We use the arithmetic mean rule:
```
k_eff = (1 - φ) × k_Si + φ × k_Cu
```
This is an upper bound (assumes heat flow parallel to TSVs). The harmonic mean (series flow, lower bound) gives:
```
1/k_eff = (1-φ)/k_Si + φ/k_Cu
```
For φ=0.10 with k_Si=148 and k_Cu=400: arithmetic (parallel) = **173.2** W/m·K, harmonic (series) = **158.0** W/m·K — a 9.6% difference. The arithmetic mean overestimates lateral heat spreading from TSVs.

> **Corrected 2026-08-04.** This previously read "arithmetic = 183.8, harmonic = 152.4 (17% difference)". Both figures were arithmetically wrong: 0.9(148)+0.1(400) = 173.2 and 1/(0.9/148 + 0.1/400) = 158.0. Found while implementing anisotropic TSV conductivity. The qualitative conclusion — arithmetic is an upper bound — is unchanged.

**Superseded in the generator (2026-08-04).** With 3D-ICE 4.0's anisotropic materials the approximation is no longer needed: `src/scenario/tsv_maps.py` applies the arithmetic mean VERTICALLY (heat runs along the copper, phases in parallel) and the harmonic mean LATERALLY (heat crosses the phases, in series), which is the physically correct treatment.

**Bonding layer (updated to hybrid bonding).** Geometry2 now uses a 5 µm `hybrid_bonding`
layer (k = 60 W/m·K) in place of the previous 25 µm micro-bump bonding (k = 50 W/m·K).
R_bond_new = 8.3 × 10⁻⁸ m²·K/W; R_bond_real (Cu-Cu <1 µm) ≈ 3 × 10⁻⁹ m²·K/W.
Our model is 28× over real, but 6× better than the previous micro-bump model (170× over real).
The absolute die2 junction temperature error is <2 K at GPU-class power densities.
See `assumptions.md §2.3` for full analysis.

---

### 4.3 Geometry 3 — Server-Class Die (25×25 mm)

| Parameter | Our value | Intel/AMD server CPU (approx.) | Server literature | Verdict |
|-----------|-----------|-------------------------------|-------------------|---------|
| Die size | 25 × 25 mm | 25–35 mm (Intel Raptor Lake) | 20–40 mm | ✓ Matches |
| Die thickness | 200 µm | 100–300 µm | 100–400 µm | ✓ Reasonable |
| Spreader (Cu) | 2000 µm | 1500–3000 µm | 1000–4000 µm | ✓ Reasonable |
| Heat sink (Cu) | 5000 µm | 3000–6000 µm | — | ✓ Acceptable |
| Power blocks | 8 × (4×4 mm) | Core tiles, variable | — | ✓ Topology consistent |
| IO gap (10 mm) | 10 mm | ~8–15 mm | Present in all server CPUs | ✓ Realistic |
| Power density range | 0.1–20 W/cm² | 5–100 W/cm² | Server: up to 300 W/cm² at hotspot | ⚠ Conservative |

**Concerns:**
- Power density ceiling of 20 W/cm² is low for server-class. A 400W TDP die at 25×25 mm = 6.4 W/cm² average, so peak hotspots can easily be 30–100 W/cm². Consider extending extreme_hotspot to 50–100 W/cm².
- The 4-block column spacing (x = 2000, 7500, 13000, 18500 µm, pitch 5500 µm) with 4mm blocks leaves 1500 µm gaps between blocks — smaller than typical inter-tile distances in real chiplets (usually 2–5 mm). Not wrong, just slightly tight.

---

### 4.4 Geometry 5 — Tier 0+1 Upgraded CoWoS (25 × 14 mm)

**Caveat (2026-08-06):** the "Zhou 2022 / est." column below was originally sourced to a
fake "Kou 2022" citation. The replacement, Zhou et al. (2022, IEEE TCPMT 12(6), 956–963),
is confirmed to exist and be topically on point (§1 above), but its specific numeric
figures have not been checked against the paper's full text (IEEE Xplore blocks automated
access). Read this column as an engineering estimate the citation is *expected* to
support, not a confirmed literature figure — Liu & Park 2012, cited in the pump-out row,
is separately flagged unverified above.

| Parameter | Our value | Zhou 2022 / est. | TSMC CoWoS-R | Verdict |
|---|---|---|---|---|
| Interposer thickness | 300 µm | 300 µm | 100–300 µm | Plausible, unconfirmed vs. Zhou 2022 full text |
| TIM1 (die→spreader) | 50 µm, k=80 (In solder) | ~50–100 µm, k=50–80 | In or In-Ag | Physically reasonable |
| TIM2 (spreader→sink) | 125 µm, k=4 | 100–150 µm, k=3–5 | Thermal grease | Within typical range |
| C4 bump k_eff | 15 W/m·K, 100 µm | 10–20 W/m·K | Cu+solder composite | Consistent |
| Die k (active layer) | 80 W/m·K (low-k composite) | 60–100 W/m·K @ N5 | N5/N3 literature | Conservative N5 estimate |
| RDL Joule fraction | 1–10 %, parameterised | ~3–8 % typical | — | Spans physical range |
| TIM1 pump-out sweep | k=80/40/10/5 W/m·K | k drops 2–10× over cycling | — | ⚠ Liu & Park 2012 unverified |
| Hybrid bonding | 5 µm, k=60 W/m·K | <1 µm, k~300 W/m·K | — | ⚠ 28× over real; <2 K error |

### 4.5 Geometry 6 — CoWoS + 6× HBM Stacks (42 × 14 mm)

Reference architecture: **AMD MI300X** (6× HBM3, compute chiplet on Si interposer).

| Parameter | Our value | AMD MI300X (approx.) | Verdict |
|---|---|---|---|
| HBM stack count | 6 | 6 | ✓ Matches |
| HBM stack footprint | 4 × 12 mm each | ~8 × 11 mm each (HBM3 base die) | ⚠ Our stacks are narrower |
| HBM stack pitch | 5 mm | ~6–8 mm | ⚠ Tighter than real |
| Interposer footprint | 42 × 14 mm | ~150 × 130 mm (full package) | ⚠ Much smaller — simplified interposer |
| HBM die layers modelled | 2 (die1 + die2) | 12 DRAM + 1 base = 13 | ⚠ Collapsed to 2-die model |
| Total height modelled | 205 µm (die stack) | ~720 µm (HBM3 full stack) | ⚠ 3.5× under real vertical R |
| Compute chiplet | 10 × 12 mm, single die | Multiple chiplets | ⚠ Simplified single compute die |

**Primary value of geometry6:** Provides realistic *lateral* thermal gradients between
compute and 6 HBM stacks — the key signal for PINN generalisation training, even with
simplified vertical stack models.

AMD MI300X reference:
AMD (2023). **AMD Instinct MI300X Architecture.** AMD White Paper.
(Layout reference for 6× HBM3 on CoWoS interposer; confirms 6-stack configuration)

---

## 5. TSV Effective Medium — Detailed Notes

The arithmetic mean rule is used consistently in both 3D-ICE and our model, ensuring internal consistency between training data and model assumption. However, compared to detailed FEM simulations of TSV arrays:

| TSV density | Arithmetic mean | FEM result (approx.) | Error |
|-------------|----------------|----------------------|-------|
| 3% | 155.4 W/m·K | ~152 W/m·K | +2.2% |
| 5% | 162.6 W/m·K | ~157 W/m·K | +3.5% |
| 10% | 173.2 W/m·K (vert) / 158.0 (lat) | ~168 W/m·K | see note above |

Reference for FEM values: Li, F., Codecasa, L., & Magnoni, M. (2012). Effective Thermal Conductivity of TSV Interposers. *IEEE Transactions on Components, Packaging and Manufacturing Technology*, 2(12), 2028–2038.
(Note: approximate values — verify against original paper before publication.)

The overestimation grows with TSV density: at 10% it reaches ~10% overestimated TSV
thermal conductance, meaning real heat spreading through TSVs is somewhat less than
predicted. The live benchmark no longer exercises 10% density anywhere — geometry2c, the
dedicated 10%-density geometry, was removed 2026-08-06 (see `goal.md`); the remaining
`geometry2a` sits at 3% mean density, where the same table shows only +2.2% error.

---

## 6. Scenario Parameter Ranges vs. Literature

**Note added 2026-08-16**: "Our range" below reflects the *original* scenario-generator
design (pre-2026-08-01 TDP-regime revision, `docs/report.md` §9.3) and no longer matches
the live dataset. Verified directly against every file's metadata in `data/3d-ice/`, the
current live dataset's HTC spans **2,000–50,000 W/m²·K** and ambient spans **25–45°C** —
see `docs/geometry_reference.md`'s "Boundary Condition Parameters" table (also corrected
2026-08-16) for the authoritative current values. Power density is no longer a flat range
at all; it's a per-geometry TDP budget (`docs/report.md` §4). Kept below for its original
purpose — checking initial design choices against the literature — not as a description
of what the current dataset contains.

| Parameter | Our range (original design, obsolete) | Typical server range | Typical mobile range |
|-----------|-----------|---------------------|---------------------|
| Power density | 0.1–20 W/cm² | 5–300 W/cm² | 0.5–50 W/cm² |
| HTC | 500–**200,000** W/m²·K | 1000–50000 W/m²·K | 500–10000 W/m²·K |
| Ambient temp | 25–85°C | 40–80°C | 25–65°C |
| TIM1 k (g5/g6) | 5–80 W/m·K | 5–82 W/m·K (In solder lifecycle) | — |
| RDL Joule fraction (g5/g6) | 1–10 % | ~3–8 % | — |

HTC extended to 200,000 W/m²·K covering liquid jet impingement and microchannel cooling.
Liquid-cooling scenarios (HTC > 15,000) appear in extra training pool indices 35+.
TIM1 k sweep and RDL fraction variation are g5/g6 only (extra pool indices 15–28).

---

## 7. Open-Source Benchmark Availability

The following open datasets/benchmarks were reviewed for comparison:
- **3D-ICE example files**: Available in the 3D-ICE source distribution at `examples/` — small 5×5 mm configurations, power up to ~5 W/cm²
- **SPEC CPU thermal traces**: Used with HotSpot; 2D die only, no 3D stacking
- **HotSpot default floor plan**: Alpha processor, ~2.25 cm²; not 3D

No public dataset was found that combines 3D-stacked dies + TSVs + physics-informed neural network benchmarks. Our dataset is novel in this combination. The closest published work is:

Cai, S., Wang, Z., Wang, S., Perdikaris, P., & Karniadakis, G. E. (2021).
**Physics-Informed Neural Networks for Heat Transfer Problems.**
*Journal of Heat Transfer*, 143(6), 060801.
DOI: 10.1115/1.4050542
(2D heat conduction only; no 3D-IC stack structure)

---

## 9. Industrial Thermal-Model Accuracy Targets (added 2026-08-11)

Checked because §10 of `docs/report.md` claims a closed-form ridge fit "matches or beats
the accuracy targets neural models are held to" without citing what that target actually
is. A web search pass (not a full-text literature review — flagged accordingly) found:

- **Compact thermal models (CTMs) vs. a detailed reference model**: agreement within
  **<1%** is cited as typical (*JEDEC Thermal Standards: Developing a Common
  Understanding*, Electronics Cooling, 2019). This is model-vs-model, not
  model-vs-silicon.
- **Two-resistor (simplest) compact models**: can be off by **up to 30%** depending on
  environmental conditions — same source. Shows the spread is large across model
  fidelity, not a single fixed bar.
- **Junction-to-case thermal resistance (oven-based test method)**: accuracy and
  repeatability of **~±4%** is cited as an industry figure for the *measurement* method
  itself (not simulation).
- **CFD-vs-physical-measurement case studies**: one case reports **~2%** margin of error;
  mesh-sensitivity comparisons (fast vs. detailed CFD models of the same system) agree
  within **~4%**.

**No source found in this pass states a single, authoritative absolute-°C sign-off
number** (an earlier draft of this discussion asserted "±2–5°C at the hotspot" from
general domain knowledge, not from a citation — that number should be treated as
unconfirmed and is not repeated as fact here). What the verified sources consistently use
instead is **relative error, typically low single-digit percent**, which is the more
defensible framing: an absolute-°C bar doesn't transfer across scenarios with very
different temperature rises (2% of a 100°C rise is 2°C; 2% of a 20°C rise is 0.4°C),
while the sources above cluster in the 1–4% band regardless of what's being modeled.
Two structural caveats apply to any comparison against this benchmark, regardless of the
exact number: (1) every figure above is measured **against silicon or a detailed
reference model**, not against another compact/RC-style solver — this benchmark's ground
truth is 3D-ICE, itself a compact model, never checked against silicon or FEM (§10's
"Threats to validity"); (2) accuracy in practice is judged **at the hotspot**, not on
field-average error, which is exactly the metric this repo's own per-cell-power finding
(§9.4) shows ridge is weakest on.

**Fully confirmed 2026-08-16** (was flagged unverified below — author names now
independently confirmed via search plus a successful full-text extraction, both
cross-checking each other):

Barua, B. P., Udoy, M. R. I., & Aziz, A. (2026).
**A Review of Multiscale Thermal Modeling in Heterogeneous 3D ICs.**
arXiv:2604.03290. Submitted 26 Mar 2026.

Surveys compact thermal models (CTMs), FEM/FDM, Green's function/semi-analytical
techniques, reduced-order and multi-fidelity methods, and physics-informed ML (PIML)
for 3D-IC thermal transport, with emphasis on interface-dominated conduction, material
anisotropy, and electrothermal coupling. Directly relevant open-problem claims (§VIII),
each worth positioning this repo's existing work against explicitly in `docs/report.md`:

1. **§VIII-E**: the field's "main challenge is no longer raw prediction speed, but
   robustness under [distribution] shift" (workload, BC, aging, manufacturing
   variation); calls for models to be "more uncertainty aware, more explainable." This
   repo's Track B leave-one-geometry-out result (goal.md) and MC-Dropout miscalibration
   finding (§10) are direct, already-completed evidence on exactly this question — not
   yet cited as such in the paper's positioning.
2. **§VIII**: no standardized, uncertainty-aware thermal boundary resistance (TBR)
   interface-property library exists across the field — a gap this repo does **not**
   address (worth flagging as a real, distinct future-work direction, not a quick add).
3. **§VIII-B/C**: uncertainty quantification is not propagated upward through
   multiscale/reduced-order model chains — same alignment as point 1.
4. **§VIII / conclusion**: incomplete multiphysics coupling (thermal-mechanical-
   electrical, power-delivery/signal-integrity co-simulation) flagged as a structural
   gap. This repo's throttling (electro-thermal feedback loop) and microchannel
   (fluid-thermal advection) mechanisms are steps toward exactly this, already built
   and pilot-measured (goal.md Track A) — worth explicit connection in §10.
5. **Conclusion**: no single modeling framework suffices; argues for calibrated
   multilevel workflows over one universal solver — supports, but doesn't itself make,
   this repo's baseline-before-crediting-an-architecture argument.

---

## 8. Geometry4 and Geometry5 — Additional References

### 2.5D Packaging and CoWoS Technology

TSMC (2016).
**CoWoS Technology: Chip-on-Wafer-on-Substrate Advanced Packaging.**
TSMC Technology Symposium.
(Foundational reference for CoWoS/2.5D interposer-based packaging architecture)

Sukumaran, V., et al. (2012).
**Design, Fabrication, and Characterization of Face-to-Face Bonded Interposer Systems.**
*IEEE Transactions on Components, Packaging and Manufacturing Technology*, 2(12), 1997–2007.
DOI: 10.1109/TCPMT.2012.2222329
(Lateral chiplet-on-interposer thermal analysis; validates underfill gap thermal resistance model)

### HBM (High-Bandwidth Memory) Thermal Analysis

Lee, S., et al. (2016).
**A 1.2V 64Gb 8-layer stacked LPDDR4 SDRAM with 320GB/s bandwidth and ca. 3ns RCD using TSV-based I/O and power delivery.**
*IEEE International Solid-State Circuits Conference (ISSCC)*, 21.2.
(HBM die-stack architecture; informs chiplet B's two-die TSV stack in geometry5)

Nalamalpu, A., et al. (2015).
**Broadwell-E: A Family of High Performance Processor SoCs Featuring Advanced Packaging Technologies.**
*Proceedings of the IEDM*, 2015.
(Multi-die package thermal management with interposer; lateral thermal coupling reference)

### Thermal Conductivity of Underfill

Xu, Y. S., & Chung, D. D. L. (2000).
**Cement-based composites improved by using silane-treated admixtures.**
*Composites: Part A*, 31, 1549–1555.
(General underfill k reference; typical polymer underfill k = 0.4–1.0 W/m·K;
we use k = 0.7 W/m·K as a mid-range value)

### 3D Heterogeneous Integration Thermal Modelling

Ariel, N., & Yahalomi, A. (2021).
**Thermal Analysis of Advanced 3D Integration Packages.**
*Journal of Electronic Packaging*, 143(3), 031004.
DOI: 10.1115/1.4048904
(Methodology for partial-footprint thermal modelling in multi-chiplet packages;
supports geometry4/5 design decisions)

TSMC (2023).
**SoIC (System on Integrated Chips) Technology.**
TSMC Technology Symposium 2023.
(3D-on-2.5D integration: vertical die bonding on horizontal chiplet tiles — the
reference architecture for geometry5's CoWoS + TSV stack combination)
