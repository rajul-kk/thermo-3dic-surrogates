# Simplifying Assumptions vs. Real Hardware

Each assumption is classified:
- **Simplifying** — a known, deliberate trade-off that is internally consistent and physically reasonable
- **Wrong** — does not match any real hardware configuration; should be fixed before publication

---

## 1. Thermal Physics

### 1.1 Steady-state only — **Simplifying**
**What we do:** Solve the steady-state heat equation `∇·(k∇T) + Q = 0`.  
**Real hardware:** Transistor switching generates nanosecond-scale thermal pulses. Die-level thermal time constants are 0.1–10 ms. Package-level time constants are seconds.  
**Impact:** Peak transient temperatures at local hotspots can be 10–40% higher than steady-state. The surrogate is valid for workload-average thermal analysis, not for timing-correlated analyses or dynamic power management.

### 1.2 Temperature-dependent k for silicon only — **Simplifying**
**What we do:** `k_Si(T) = 148 × (300/T)^1.3` (Glassbrenner & Slack 1964). Copper, TIM, and bonding layers use constant k.  
**Real hardware:** Copper k drops ~5% from 300–600 K (401 → ~385 W/m·K). TIM k is weakly temperature-dependent. Both effects are small compared to silicon's 4× drop.  
**Impact:** <5% error in spreader/sink temperatures; negligible for design purposes.

### 1.3 Uniform convective HTC across top surface — **Simplifying**
**What we do:** A single scalar h [W/m²·K] is broadcast over the entire top surface in the BC: `-k·dT/dz = h·(T − T_amb)`.  
**Real hardware:** Jet impingement centre h can be 3–5× higher than edges. Microchannel coolers have spatially varying h along flow direction.  
**Impact:** ±5–15 K spatial error at surface for high-performance cooling (h > 5000 W/m²·K). Internal die temperatures are less affected because the die/spreader/TIM stack attenuates surface non-uniformity.

### 1.4 Adiabatic side and top faces — **Simplifying**
**What we do:** `dT/dn = 0` on all four lateral side faces and the top face (`z=1`, nearest the die). Convective (HTC) cooling is applied at the **bottom** face (`z=0`, the `heat_sink` layer) — this matches 3D-ICE's own `bottom heat sink` boundary-condition directive.

> **Correction note:** an earlier version of the PINN's physics-loss code (`trainer.py`'s `top_layer_id`) enforced the convective boundary condition at the *top* face instead of the bottom, the opposite end of the stack from where 3D-ICE actually cools. This was a real bug in the PINN's own PDE/BC loss target, not a documentation error — the 3D-ICE ground truth data was always correct. Fixed by moving the convective term to `z=0` (with the correct outward-normal sign) and adding an explicit adiabatic term at `z=1` (previously left entirely unconstrained under the default `hard_adiabatic=True` setting, since the cosine-fold hard BC only covers the lateral x/y walls, never z). Any PINN checkpoint trained before this fix was fit against the wrong physics-loss target; its data fit is unaffected but its PDE/BC-loss diagnostics should not be trusted.

**Real hardware:** PCB conduction provides ~0.5–5 W/K lateral path; the die-side (top) face is not literally adiabatic in real hardware — it's simplified to be so because no lid/ambient contact is modeled there.
**Impact:** Small for large dies (heat spreader dominates). For geometry2 (8×8 mm), lateral heat loss is ~2–5% of total; the adiabatic assumption slightly over-predicts temperature.

### 1.5 No radiation — **Simplifying**
**What we do:** No radiative boundary condition.  
**Real hardware:** At 400 K surface, radiation ≈ 1–3 W total — negligible vs. convective cooling of tens to hundreds of watts.  
**Impact:** <1% error for all geometries in this benchmark.

---

## 2. Material Properties

### 2.1 Copper k = 400 W/m·K — **Simplifying**
**What we do:** Use k = 400 W/m·K for all copper layers (spreader, heat sink).  
**Real hardware:** Electrolytic copper: k = 385–395 W/m·K. Electroplated copper (used in spreaders): k = 350–385 W/m·K due to grain structure.  
**Impact:** ~4% overestimate of spreader conductance; <2 K error at die junction. Acceptable for a benchmark.

### 2.2 TIM2 k = 4 W/m·K (constant, geometry1/2/3/4) — **Simplifying**
**What we do:** Single TIM value representing thermal grease for die-to-spreader interface.  
**Geometry5/6:** TIM1 (die-to-spreader, `tim_top`) is indium solder at k = 80 W/m·K — physically accurate for high-performance 3D packages. TIM2 (`tim_sink`, spreader-to-sink) remains 4 W/m·K (grease).  
**Real hardware:** Thermal greases: 3–8 W/m·K; indium foil: 82 W/m·K. TIM k is pressure- and temperature-dependent.  
**Impact:** Constant TIM k ignores pump-out degradation (k can drop 2–10× over thermal cycles). Geometry5/6 models this explicitly via `layer_k_overrides['tim_top']` with k ∈ {80, 40, 10, 5} W/m·K in training scenarios.

### 2.3 Hybrid bonding layer k = 60 W/m·K, 5 µm thick — **Simplifying**
**What we do:** 5 µm explicit layer between die_zone_1 and tsv_zone in g2/g5/g6 with k_eff = 60 W/m·K (Cu pillar composite at 9 µm pitch, ~15 % fill fraction).  
**Real hardware (Cu-Cu hybrid bonding):** Bondline < 1 µm; k ≈ 300–400 W/m·K. Thermal resistance is essentially zero (R ≈ 1 µm/300 = 3 × 10⁻⁹ m²·K/W per unit area).  
**Our model:** 5 µm / 60 W/m·K = 8.3 × 10⁻⁸ m²·K/W — ~28× larger than reality.  
**Previous model (micro-bump):** 25 µm / 50 W/m·K = 5 × 10⁻⁷ m²·K/W — ~170× larger than reality.  
**Impact:** Our hybrid bonding layer reduces the previous error by 6× but still overestimates die-to-die resistance by ~28×. The absolute temperature error at the die2 junction is <2 K for GPU-class power densities (dominated by TIM and C4 resistance), so this is a low-priority accuracy limitation. It is documented as a deliberate modelling choice: the 5 µm layer makes the interface visible in the PINN's coordinate space and NPZ data.

### 2.4 TSV effective k from arithmetic mean — **Simplifying**
**What we do:** `k_eff = (1−φ)·k_Si + φ·k_Cu` (parallel/upper-bound rule).  
**Real hardware:** Actual k_eff is closer to geometric mean or a Bruggeman effective-medium result. The arithmetic mean overestimates by 2% at φ=3%, 9% at φ=10%.  
**Impact:** Geometry2c (10% TSV) overestimates TSV thermal conductance by ~10%. TSV layers appear as slightly better thermal vias than they are. This is the same approximation 3D-ICE uses, so training data and PINN assumption are internally consistent.

---

## 3. Geometry

### 3.1 Geometry1 die size (10×10 mm) — **Simplifying**
**What we do:** 10 mm × 10 mm die, 150 µm thick.  
**Real hardware:** Mobile SoC dies: 5–12 mm × 5–12 mm. Desktop CPU dies: 10–25 mm. GPU compute dies: 20–35 mm.  
**Status:** Matches the canonical 3D-ICE benchmark configuration. Representative of mid-range mobile/desktop dies.

### 3.2 Geometry2 die size (8×8 mm) vs. 3D-ICE reference (5×5 mm) — **Simplifying**
**What we do:** 8 mm × 8 mm dies, 50 µm thick each.  
**3D-ICE published example:** 5 mm × 5 mm at similar thicknesses.  
**Real 3D-stacked dies:** HBM DRAM dies: 5.5 × 7.5 mm. Logic dies in CoWoS: 10–25 mm. Wide I/O DRAM: ~12 × 8 mm.  
**Impact:** Our geometry2 is physically plausible for a logic-die 3D stack but does not match the 3D-ICE benchmark directly. Results are not directly comparable to published 3D-ICE accuracy metrics.

### 3.3 TSV region modelled as uniform-k layer (whole footprint) — **Simplifying**
**What we do:** The `die1_tsv` / `die2_tsv` layers apply the homogenised k_eff uniformly across the full 8×8 mm footprint.  
**Real hardware:** TSVs occupy only the TSV array region (our power blocks `tsv_array_d1/d2`, 2×2 mm). The rest of the die in those layers is pure silicon.  
**Impact:** The non-TSV-array area of the TSV layer is over-enhanced by k_eff − k_Si = 7–36 W/m·K. Heat spreading in the "untouched silicon" region is overestimated. This matches how 3D-ICE models TSVs, so training data and model are consistent, but both differ from detailed per-via simulations.

### 3.4 Geometry3 power density ceiling (20 W/cm²) — **Simplifying**
**What we do:** Maximum power block density = 20 W/cm² (extreme_hotspot scenario).  
**Real hardware:** Server CPU TDP = 300–500 W; chip area = 600–900 mm²; average = 3–8 W/cm². Peak local hotspot (L2 cache array, ALU cluster): 30–100 W/cm². High-end GPU local peaks: up to 300 W/cm².  
**Impact:** The model is not trained on the hotspot regime relevant for thermal emergency prediction. Extending extreme_hotspot to 50–100 W/cm² would improve utility for server workloads.

### 3.5 Heat sink modelled as a solid copper block — **Simplifying**
**What we do:** Heat sink = solid copper layer, 5000 µm thick, with convective BC on the top face.  
**Real hardware:** Finned air-cooled heat sinks have effective k much lower than bulk copper in the fin direction. Liquid-cooled cold plates have anisotropic k (high in-plane, low through-plane before convection). Vapour chambers: extremely high effective k but with heat spreading built into the phase-change structure.  
**Impact:** Our model overestimates heat sink thermal mass and assumes isotropic thermal conductivity. For a benchmark focused on the die and package stack (not the cooling hardware itself), this is acceptable. The h value captures the net effect of the cooling system.

---

## 4. PINN Architecture

### 4.1 Single global Fourier feature matrix B (not per-layer) — **Simplifying**
**What we do:** One fixed random matrix B ~ N(0, σ²) maps all (x,y,z) coordinates to Fourier features.  
**Ideal:** Separate frequency scales for in-plane (die hotspots, ~100 µm scale) and through-plane (layer interfaces, 25–100 µm layer thicknesses) directions.  
**Impact:** σ=10 is tuned for geometry1 die-level features. The heat sink (5000 µm thick) and TIM layers (50–100 µm) are at very different spatial scales; a single σ is a compromise.

### 4.2 Layer embedding encodes material identity — **Simplifying**
**What we do:** Integer layer index → 8D learnable embedding. The network must learn material properties (k, ρCp) from the combination of embedding + training data.  
**Ideal:** Directly provide k and ρCp as additional input features.  
**Impact:** Harder to generalise to unseen material properties. Fine for this fixed-geometry benchmark.

### 4.3 Interface loss not implemented — **Simplifying**
**What we do:** Temperature continuity and heat-flux continuity at layer boundaries are enforced implicitly through data fitting and the PDE residual, not via an explicit interface condition.  
**Ideal:** Enforce `-k₁·dT/dz|below = k₂·dT/dz|above` at each layer boundary.  
**Impact:** The MLP is C∞ continuous so temperature continuity is automatic. Heat-flux continuity depends on the PDE residual being well-satisfied at the interface, which is harder near the die/TIM2 boundary where k jumps 37×. Small discontinuities in dT/dz predictions may persist near this boundary.

### 4.5 Quasi-linearised k(T) in PDE residual — **Simplifying**
**What we do:** `thermal_conductivity` is called with `T_hat.detach()` so k is a frozen coefficient during each backward pass. This is a Picard iteration: k updates implicitly as T_hat improves over epochs.  
**Full nonlinear alternative:** Remove `.detach()` so autograd differentiates through `k(T)`, computing the extra term `(dk/dT)·(∂T/∂x)²` in the divergence. Costs ~30% more memory/time for every PDE backward pass.  
**Why simplification is acceptable:** PDE weight = 0.1 (regularisation, not the primary signal). At convergence both approaches reach the same solution. The linearisation error is smaller than the target MAE (<2 K) and the computational cost saving is ~1 hr per geometry on CPU.

### 4.4 Scenario-level scalar conditioning (htc, t_amb, tsv_frac) — **Simplifying**
**What we do:** HTC and ambient temperature are single scalars broadcast over the whole top surface.  
**Real hardware:** Spatially varying HTC field.  
**Impact:** See §1.3. The PINN cannot represent spatially non-uniform cooling regardless of training data quality.

### 2.5 RDL Joule heating fraction is scenario-variable — **Simplifying**
**What we do:** RDL blocks (`chipA_rdl`, `hbm1_rdl`, etc.) receive `rdl_joule_fraction × base_power`, where fraction ∈ {1, 3, 5, 8, 10 %} is varied across scenarios. This teaches the PINN that RDL heating is an independent variable.  
**Real hardware:** RDL current density and track resistivity determine Joule heating; the fraction depends on layout, routing density, and current load. Typical values: 2–8 % of die power for dense HBM RDL; up to 15 % at high current.  
**Impact:** The parameterisation spans a physically plausible range. The exact mapping from chip-level current to Joule fraction is not modelled (that would require IR drop simulation). Treated as a bounded uncertainty input to the PINN.

---

## 6. Geometry4, 5, and 6 — 2.5D/3D-on-2.5D Specific Assumptions

### 6.1 Uniform k in die_zone for 3D-ICE (geometry4 and geometry5) — **Simplifying**

**What we do:** The 3D-ICE `.stk` file assigns a single material (silicon, k = 148 W/m·K)
to the entire `die_zone` / `die_zone_1` layer across the full interposer footprint. The
underfill gap (k = 0.7 W/m·K) between chiplets is not separately modelled in 3D-ICE.

**Real hardware:** The underfill gap has ~200× lower conductivity than Si. Lateral heat
spreading through the gap is negligible, so the main error is that 3D-ICE slightly
overestimates thermal coupling between chiplets.

**Impact:** The PINN's PDE loss uses the correct heterogeneous k via `_compute_lateral_k()`
(underfill where no DiePrint footprint). The training data source (3D-ICE) uses uniform Si.
The model learns primarily from data and uses the PDE as regularisation (weight λ_pde = 0.1),
so the inconsistency produces a small systematic error in the predicted lateral coupling.

**Residual error estimate:** 5–15 K overcoupling in the underfill gap region. For the
chiplet die interiors (the regions that matter most), the error is negligible because
heat flows predominantly vertically through the layer stack.

**This is a train/target inconsistency, not merely a simplification.** The PDE residual is
computed against a heterogeneous k that the ground-truth data does not contain, so the
physics loss actively pulls predictions away from the data it is trained on in the gap
region. The λ_pde = 0.1 weighting bounds the damage but does not remove it. Any result
reported on geometry4/5/6 should either use λ_pde = 0 or disclose this.

**Proper fix (not yet applied): upgrade the ground truth to 3D-ICE 4.0.**
3D-ICE 4.0 (arXiv:2512.05823, Dec 2025 — by the original 3D-ICE authors) natively preserves
material heterogeneity and anisotropy from layouts, which is exactly the missing capability.
Regenerating geometry4/5/6 under it would make data and physics loss consistent, and is far
cheaper than hand-segmenting `.stk` layers in `ice_simulator.py`. Cost: a full re-run of the
geometry4/5/6 scenarios (~155 simulations).

**Do not treat lateral heterogeneity as a novelty claim.** It is settled prior art —
see `references.md` §"Lateral material heterogeneity in 2.5D/3D chiplet thermal modelling".

### 6.2 Single-die TSV model for chiplet B (geometry5) — **Simplifying**

**What we do:** Chiplet B's TSV zone uses a spatially uniform effective conductivity
k_TSV = (1−φ)·k_Si + φ·k_Cu (rule of mixtures with φ = 3% TSV fill fraction).

**Real hardware:** Actual HBM TSV arrays are not uniformly distributed — they are
clustered near the die centre, creating non-uniform lateral k in the bonding layer.
The effective medium assumption (uniform k_TSV) averages this variation.

**Impact:** ~10% error in the peak temperature of chiplet B's die2 (top die). For the
primary research goal (demonstrating PINN generalisation across geometry types), this
approximation is consistent with the treatment of TSVs in geometry2.

### 6.3 No die-height mismatch effect — **Simplifying**

**What we do:** Both chiplet A (1 die, 50 µm active) and chiplet B (2 dies + TSV + hybrid
bonding, ~210 µm total) share the same z-layer structure. The "empty" z-levels above
chiplet A are filled with underfill (k = 0.7 W/m·K) in the PINN's lateral k map.

**Impact:** Small for steady-state. Underfill above chiplet A is a near-adiabatic column,
so very little heat flows through it. The dominant error is that the convective BC is
applied uniformly at z_top — for chiplet A this means the BC is applied above 155 µm of
underfill (~222 K/W effective extra resistance per cm²), slightly over-cooling the column
that should be in contact with the TIM. For GPU-class power densities the induced error
is <3 K.

### 6.4 Geometry6 HBM stack count and footprint — **Simplifying**

**What we do:** Six identical HBM stacks at 5 mm pitch on a 42 × 14 mm interposer.
Each stack is 4 × 12 mm.

**Real hardware (AMD MI300X):** Six HBM3 stacks, each comprising 12 DRAM dies +
1 base die = 13 layers, total height ~720 µm. Stacks have variable pitch (not uniform).

**Impact:** The multi-layer HBM stack is collapsed to a two-die model (die_zone_1 +
tsv_zone + die_zone_2 = 205 µm). This underestimates the vertical thermal resistance
of the HBM stack by ~3.5×. Peak HBM junction temperature is underpredicted; the
interposer temperature gradients (the primary PINN learning target) are largely
unaffected since they are dominated by lateral power variations, not vertical stack
resistance.

---

## Summary Table

| Assumption | Category | Max error | Priority to fix |
|---|---|---|---|
| Hybrid bonding k_eff = 60 W/m·K, 5 µm (g2/g5/g6) | Simplifying | ~28× over real; <2 K die2 error | Low (abs. error small) |
| RDL Joule fraction parameterised not physics-derived | Simplifying | Bounded 1–10 % uncertainty | Low |
| TIM pump-out k-sweep (g5/g6 only) | Simplifying | Covers 80→5 W/m·K range | None — fully implemented |
| Uniform HTC top surface | Simplifying | ±5–15 K at surface | Medium for real cooling comparison |
| Arithmetic mean TSV k | Simplifying | ~9% at 10% TSV | Low (consistent with 3D-ICE) |
| Constant k for Cu/TIM2 | Simplifying | <2 K | Low |
| Solid copper heat sink | Simplifying | Small for die-focus | Low |
| No transient | Simplifying | 10–40% peak transient | Medium for power management use cases |
| Power density ceiling 20 W/cm² | Simplifying | N/A (extrapolation) | Medium for server geometry3 |
| Geometry2 die size vs. 3D-ICE ref | Simplifying | N/A | Low (plausible, not published ref) |
| Adiabatic side/top | Simplifying | 2–5% temperature | Low |
| HBM/memory-stack power cap (g5/g6) | Fixed (was a bug) | Max scenario T dropped 134°C→94.8°C for geometry6 | None — resolved; see §7 |
| No radiation | Simplifying | <1% | None |
| Uniform k in die_zone (g4/5/6) | Simplifying | 5–15 K in underfill gap | Low (data-dominant PINN) |
| Two-die HBM model (vs. 12-die real HBM3) — geometry6 | Simplifying | ~3.5× underestimate of HBM vertical R | Low (interposer gradients unaffected) |
| No die-height mismatch (g5/g6) | Simplifying | <3 K | Low |

---

## 7. Fixed Issues (Historical — kept for traceability)

### 7.1 HBM/memory-stack power density was uncapped — **Fixed**

**What was wrong:** The scenario generator's power patterns (`uniform`, `split_chiplet_b_hot`, etc.) assigned HBM/memory-stack dies (geometry5's `chipB_d1*`/`chipB_d2*`, geometry6's `hbm{n}_d1`/`hbm{n}_d2`) the **same power-density range as compute logic** — up to 20 W/cm² in extreme scenarios. Real HBM3 dies dissipate roughly 0.1–0.5 W/cm² under typical-to-heavy load; the uncapped scenarios pushed HBM dies well past their ~95–105°C junction-temperature reliability spec (geometry6 recorded a peak of **134°C** across the training set). This directly contradicted geometry6's own docstring claim of representing an "MI300X-like" (real-chip) configuration.

**Fix:** `src/scenario/generator.py::_apply_pattern` now caps any block matching `hbm*`, `chipB_d1*`, or `chipB_d2*` to 2.0 W/cm² regardless of pattern, applied as a post-process step after every power-pattern branch. geometry5 and geometry6's full 55-scenario datasets were regenerated via 3D-ICE after the fix. **Result:** geometry6's max training-scenario temperature dropped from 134°C to 94.8°C, now inside the realistic HBM3 envelope.

**Original (pre-fix) data:** backed up to `data/3d-ice_backup_pre_hbm_cap/` if a before/after comparison is needed.

### 7.2 PINN convective BC was enforced at the wrong z-face — **Fixed**

See §1.4 above for the full description. Summary: the PINN's own physics-loss code assumed convective cooling at `z=1` (top, near the die); 3D-ICE's real ground truth cools at `z=0` (bottom, `heat_sink`). The training data was never wrong — only the PINN's PDE/BC-loss target was. Fixed in `src/pinn/trainer.py`, `src/pinn/physics.py`, and `src/pinn/data_loader.py`; no dataset regeneration was needed since the ground truth was always correct.
