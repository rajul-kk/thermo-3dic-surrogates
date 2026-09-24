# Summary: A Cautionary Benchmark for 3D-IC Thermal Surrogates

**Status note.** This is a condensed, submission-oriented summary of the project. It states
only the *current, corrected* claims and their final numbers. It intentionally omits the
retraction history, superseded measurements, and process narrative that `docs/report.md`
preserves in full — every claim below is traceable to a numbered section there
(`§9.x`), and that document is the source of record if this one and `report.md` ever
disagree. Full bibliography: `docs/references.md`. Molecular-property replication detail:
`molprop/README.md`.

> **Validation, repair and regeneration, 2026-09-23/24 (report §9.23–9.24).** An independent FV
> solver confirmed 3D-ICE is correct and found faults in *this project's* pipeline. All are now
> fixed:
> - transposed temperature labels;
> - a layer-blind power field;
> - whole-die power maps that put ~30% of the power in underfill;
> - an unconverged grid (single-node thick dies; a geometry4/5 mesh axis swap).
>
> The fixed dataset and four layout datasets were regenerated, and 3D-ICE matches the FV solver
> to ≤1.14% and grid convergence to ≤1.45%. Every number below is re-measured on the v5 data.
> The 306 pilot solves behind report §9.8–9.11 were regenerated too, and every pilot conclusion
> survives. **Still open:** neural-model comparisons need GPU re-runs.

---

## Abstract

Neural surrogates for 3D-IC thermal simulation (PINNs, Fourier neural operators, DeepONets)
are an active research area, but the field rarely reports a non-neural baseline. We show that
on the standard evaluation setup — a fixed package floorplan with only power amplitude
varying — **closed-form ridge regression reconstructs the temperature field at spatial
R² 0.93–0.98** with no training and no GPU, across six 3D-IC package geometries and 275 real
3D-ICE simulations, and that this is a consequence of the physics (steady-state conduction is
linear in the volumetric sources), not a weak dataset. We then show the benchmark is fixable — randomising chiplet
*placement* moves three of four re-laid-out geometries out of the linear regime, with ridge
becoming the *worst* baseline tested — and that "linearly solvable" turns out to be a property
of the *input representation*, not the dataset: the same 45 files score R² 0.95 from the
full power field and −0.67 from the compact vector surrogate papers conventionally use. We
supply the missing linear baseline for an external benchmark (IC-ThermBench), where it
*loses* — confirming the diagnostic works in both directions. Finally, on layout-varying data,
a linear fit on the full power field localises hotspots up to 8.75× more accurately than
ridge on the compact vector — seed-stable on all three sharp-peaked geometries, a new
measurement — and split conformal prediction gives valid peak-temperature intervals where
MC Dropout did not. We release the dataset, 675 3D-ICE solves in total, and the
baseline/audit tooling used throughout.

---

## 1. What this paper is, stated plainly

This is a **negative-and-corrective-results paper with one small constructive measurement**,
not a new-architecture paper. Two of its ten original contributions are explicitly negative
(ridge beats the field's baselines; a benchmark-design idea turns out not to be novel), one is
a diagnostic method, and one (§7 below) is a genuinely new but narrow measurement. No claim
here should be read as "neural surrogates don't work" — IC-ThermBench (§4) is a
counter-example the paper itself supplies.

**Venue fit (checked 2026-09-22).** NeurIPS 2026's "Evaluations & Datasets" track explicitly
solicits auditing and stress-testing of existing benchmarks and states submissions "need not
beat a baseline" — contributions 2/7/9 and 6 map onto that call. Re-check the call before
submitting.

---

## 2. Data

**675 real 3D-ICE solves**, verified against files on disk (2026-09-23):

- **275 fixed-placement scenarios** across 6 package geometries — single-die mobile/server
  stacks, a dual-die 3D-TSV stack, and 2.5D/CoWoS chiplet assemblies with up to 6 HBM stacks.
- **400 placement-varied solves**: 270 layout-randomised (6 geometries × 45), 90
  rigid-translation, and a 40-scenario CoWoS-L pilot (geometry7).

Full geometry specs: `docs/geometry_reference.md`. Generation pipeline and known modelling
assumptions: `docs/assumptions.md`.

---

## 3. Method

Four non-neural baselines (mean, nearest-neighbour, k-NN, ridge) are fit per-point against
five neural architectures implemented in this repo (PINN, FNO/CNO-FNO/WHNO, PI-DeepONet, an
autoregressive z-layer operator, few-shot fine-tuning). All comparisons use:

- **Spatially-detrended error** (per-scenario mean removed) — raw MAE on this data is
  dominated by a per-scenario offset (88% of variance from ambient temperature alone) and is
  not a meaningful spatial-fidelity metric.
- **k-fold cross-validation**, not single train/test splits — a 40/5 split inverted an
  architecture ranking in this project once (§9.12b→§9.12d) and is not used for any reported
  number below.
- **Hotspot metrics reported separately from field metrics**, gated by a peak-sharpness
  diagnostic (`scripts/hotspot_eval.py`): argmax-distance is only meaningful where the true
  peak is well-posed (checked per geometry, not assumed).

Tooling released: `scripts/baselines.py`, `scripts/hotspot_eval.py`, `scripts/layout_cv.py`,
`scripts/benchmark_linearity_audit.py`.

---

## 4. Results

### 4.1 The fixed-placement benchmark is linear-solvable (§9.1a)

| geometry | ridge spatial R² | kNN spatial R² |
|---|---|---|
| geometry1 | **0.966** | 0.896 |
| geometry2a | **0.982** | 0.961 |
| geometry3 | **0.945** | 0.908 |
| geometry4 | **0.959** | 0.929 |
| geometry5 | 0.975 | **0.981** |
| geometry6 | 0.933 | **0.950** |

Ridge's margin over 3-NN is thin, and on two geometries it is *negative*: kNN wins. If a
distance-weighted lookup is competitive, the benchmark is not measuring operator learning.
This holds under extrapolation to unseen power patterns, magnitudes, and ambient
temperatures, and is not an artifact of an unrealistic operating point: regenerating on
physically-grounded power/cooling budgets raised the median spatial gradient 10× and ridge
still solves every split at R² > 0.94 (§9.3).

**Mechanism (§9.14a):** this is the exact regime classical EDA "power blurring" methods
(Kemper et al. 2007; Ziabari et al., IEEE TVLSI 2013) already solve to ~1°C, three orders of
magnitude faster than FEA — a fixed geometry with a fixed thermal impulse response, convolved
against a varying power map. The finding is a rediscovery of that fact from the ML side, not
a new one; it is presented as such.

### 4.2 Hotspot localisation discriminates only once the layout varies (§9.12, §9.24)

Field R² and hotspot localisation are different tasks. On fixed placement with physically placed
power, though, the hotspot sits in one of a few block regions, and ridge finds it. Its median
error is 0.6–1.0 mm on geometry1/2a and 3.5–4.0 mm on geometry3/4, with top-1% recall
0.28–0.68. It is still the model with the lowest peak-*temperature* error (2.8–4.3 K).

An earlier version of this summary called localisation a universal failure on fixed placement.
That measured the pre-v5 data, whose per-cell power maps spread heat over the whole die, and it
is withdrawn. The metric earns its place on layout-varying data, where the hotspot can be
anywhere (§4.4, §4.7). An earlier apparent neural win on this metric came from a 5-scenario
split and reversed under 5-fold CV (§9.12b→d, retracted in place).

### 4.3 The diagnostic validates in the other direction: IC-ThermBench is not linear-solvable (§9.13)

Running this project's own baseline tooling against an external benchmark (IC-ThermBench,
arXiv:2608.23977, 50,000 samples, no non-neural baseline in its own paper): the linear
baseline **loses**, 3.4–4.4× worse than Therm-FM across every reported scope, worse than all
eight of that paper's neural baselines. This is the control the diagnostic is *for* — it
correctly certifies a benchmark that discriminates, not just one that doesn't. Decomposed:
most of the linear model's deficit there is missing layout conditioning rather than
nonlinearity (fitting one operator per discrete substrate geometry recovers 38–61% of the
error, §9.13a), and on the hardest (structural-OOD) scope the best neural model is only 1.9×
better than the training mean (§9.13c) — neither visible without a trivial baseline in the
table.

### 4.4 The benchmark is fixable, and the first fix attempted did not work (§9.15, §9.15b)

Randomising chiplet placement per scenario — so the thermal operator itself varies, not just
its input — moves the benchmark out of the linear regime. Cross-validated over 45 scenarios
per geometry:

| geometry (shelf) | ridge R² (mean) | ridge R² (median) | worst by mean? | worst by median? |
|---|---|---|---|---|
| geometry4 | −0.663 | 0.412 | **yes** | no — the trivial mean-field baseline is worse (0.344) |
| geometry5 | −2.268 | 0.298 | **yes** | **yes** |
| geometry6 | −2.866 | −0.303 | **yes** | **yes** |

Ridge is worst by mean R² on all three; by median only on two — geometry4's heavy-tailed
distribution gives the trivial mean-field baseline a worse *typical* scenario despite a less
catastrophic *average* one. On geometry6, ridge's detrended error *exceeds the signal it
predicts* (norm err 1.035, 26/45 scenarios worse than the field's own mean). The first attempt tried — rigid translation of
each chiplet — **does not work** despite moving sources further than an external benchmark by
support-overlap IoU (0.472 vs 0.449); the reported reason is that support overlap, not layout
degrees of freedom, is the wrong thing to optimise (§9.15).

**Novelty scope, corrected 2026-09-18 (§9.16i):** layout randomisation as a benchmark fix is
not itself novel — HSLD (arXiv:2103.11177, 2021) and ChipTherm (50 placement-varying package
families) both predate this project by years. What survives is the *measured effect on this
specific benchmark* and the *failed-attempt report*, neither of which either prior work
supplies.

### 4.5 "Linear-solvable" is a property of the representation, not the dataset (§9.15c)

The same 45 geometry4-shelf files score linear R² **0.95** from the full per-cell power field
and **−0.663** from the compact block-summary vector surrogate papers conventionally consume.
Conduction is exactly linear in the per-cell source, so the field representation is the
physically correct hypothesis class. On repaired data this also holds on geometry5/6
(field R² 0.94, 0.89 against compact −2.27, −2.87); it degrades only insofar as moving a chiplet also moves
silicon. The compact vector discards exactly that. This is a benchmarking-flavoured
restatement of the classical affine-vs-non-affine parametrisation distinction from
reduced-basis/model-order-reduction theory (Kolmogorov n-width decay), not a new mechanism —
checked and stated as such (§9.16g).

**The predicted follow-on experiment was run (§9.15d).** An FNO consumes the field
representation, so it should inherit the field's cross-geometry difficulty gap
(geometry4−geometry6 ≈ +0.427 pre-repair; +0.263 on repaired data), not the compact vector's
(≈ +2.192). *(The FNO runs in this paragraph were trained on transposed targets (§9.23) and
are unreliable until re-run.)* Testing both plain FNO
and the strongest architecture this repo implements (CNO-FNO with axial attention and a
physics-informed loss): the prediction held, weakly (gap +0.125 to +0.178) — but the more
robust finding is that **even the strongest architecture still loses to closed-form linear
regression on its own input**, on both geometries, at 5–10× the GPU cost of a plain FNO.

### 4.6 A methodology transfer: the same audit run on canonical PDE benchmarks and a second discipline (§9.16, §9.16b/h)

The linearity diagnostic, calibrated against PDEBench (Darcy, Burgers, Navier–Stokes, shallow
water), correctly places nonlinear Burgers at the discriminative extreme and confirms this
project's fixed-placement dataset was the most linearly-solvable dataset measured, ahead of
Darcy. It also found that time-evolution PDE benchmarks generally omit a **persistence
baseline** (assume nothing moved): on PDEBench Navier–Stokes at a short time gap, a linear fit
scores 0.958 while persistence scores 0.9997.

**Narrowed after further testing (§9.16h).** Re-run across 26 independently generated
simulation files rather than one, the result **reverses** — a linear fit beats persistence at
every horizon tested, confirmed stable at that scale. The honest conclusion is not a claim
about PDE-surrogate evaluation practice generally; it is that this project's own
cost-saving probe (reading a leading slice of one file via HTTP range request, rather than
the ~1200 trajectories such benchmarks are normally pooled across) gave a measurement that
inverts once tested properly. Recorded as a caution about lightweight benchmark-auditing
tooling, not a finding about PDEBench.

**Transferred to molecular property prediction** (`molprop/`, MoleculeNet, 12 dataset×split
blocks): a tuned logistic regression on fingerprints is not separable from tuned
gradient-boosted trees on 0 of 4 random-split classification cells (separably worse on 2 of 4
scaffold-split cells, extended to 8 cells in `molprop/README.md` §7.6). MoleculeNet's
unreported Bemis-Murcko tie-break convention shifts results by up to 0.21 AUC. **Both
headline claims were independently published in 2024** (MOLTOP, arXiv:2407.12136) before this
audit was run — recorded transparently in `molprop/README.md` §8. What survives: the
measurement itself, a matched-budget control, and a PRC-AUC ranking-flip demonstration MOLTOP
does not report.

### 4.7 New measurement: hotspot localisation on layout-varying data (§9.17)

No prior work checks all three of {layout-varying data, hotspot localisation as a metric,
non-neural baseline} together — ChipTherm has the first and third but not the second; the
closest hotspot-position paper (arXiv:2503.04049) has the second but a fixed geometry and no
non-neural baseline; HSLD (2021) has the first but neither of the others. This project
measured it on six layout-randomised geometries, gated by a peak-sharpness diagnostic
applied *before* any model was fit. Ratios are ridge ÷ field error (>1 = field better), mean over
four seeds, v5 data:

| geometry | peak sharp enough for distance? | loc ratio (seed range) | recall ratio | shelf ÷ fixed (loc) |
|---|---|---|---|---|
| geometry1-shelf | yes | **8.75×** (7.62–9.62) | 5.86× | 2.89× |
| geometry2a-shelf | yes | **2.73×** (2.27–3.77) | 5.98× | 2.04× |
| geometry3-shelf | yes | 1.39× (1.20–1.69) | 6.05× | **0.41× (reverses)** |
| geometry4-shelf | no (diffuse) | 1.89× (1.30–2.36) | 3.99× | 1.93× |
| geometry5-shelf | no | 1.68× (1.46–1.91) | 5.54× | — |
| geometry6-shelf | no | 2.10× (1.66–2.43) | 7.01× | — |

The field model localises better than ridge on all six layout-randomised geometries, and all
four seeds agree on every one. On fixed placement it is better on three of four, ties on
geometry4 (0.98×), and is seed-unstable on geometry3. Layout randomisation amplifies the recall
advantage on all four geometries with a fixed counterpart, and the distance advantage on three
of four.

The counterweight is part of the claim: ridge has the lower peak-*temperature* error on every
dataset (e.g. 4.60 K vs 13.53 K on geometry2a-shelf). The representation that finds *where* the
peak is costs accuracy in *how hot* it is. That split holds between two *linear* models, so it
cannot be attributed to network capacity.

**Novelty scope.** The mechanism (the compact vector discards sub-block spatial detail) is not
new. What is new is the measurement: the magnitudes and their geometry-dependence.

### 4.8 Calibrated peak-temperature intervals (§9.20)

The project's MC Dropout uncertainty was ~30× too wide. Split conformal prediction, nested
inside the 5-fold CV with a held-out calibration set, meets its guarantee on every
geometry/model tested:

| data | model | empirical coverage at 80% / 90% target | half-width (K) |
|---|---|---|---|
| geometry1 fixed | ridge | 93.3% / 93.3% | 8.8 |
| geometry6 fixed | ridge | 87.3% / 90.9% | 8.0 / 10.1 |
| geometry1-shelf | ridge | 93.3% / 93.3% | 25.9 |
| geometry1-shelf | linear (field) | 93.3% / 93.3% | 39.5 |

The intervals are honest but wide: conformal guarantees coverage, not narrowness, and the
width reflects the point models' peak-temperature error. **Novelty scope:** split conformal for thermal neural operators already exists
(arXiv:2606.09923, field intervals). This is an application check, not a new method.

---

## 5. Threats to validity

- **Numerical correctness is independently checked; physical fidelity is not** (report
  §9.23–9.24). An independent finite-volume solver (`src/validation/fv_solver.py`) matches the
  exact 1D solution to 1e-8 K. On 18 paired cases it reproduces 3D-ICE to ≤1.14% of the
  temperature rise, and grid refinement changes the dataset's answers by ≤1.45%. Every saved
  solve balances energy to <4e-5. That establishes that the benchmark's ground truth is the
  converged solution of the stated conduction problem. It does not establish that the stated
  problem matches silicon: there is still no comparison against measurement or against a
  detailed FEM model with resolved microstructure (TSVs, bumps), and the modelling
  simplifications in `docs/assumptions.md` stand.
- **Sample sizes are small** (45 scenarios per layout-varying geometry) for the
  high-dimensional layout spaces involved; absolute numbers should be read as directional,
  not asymptotic.
- **The packages are reduced-scale, not models of current products.** The largest (geometry7,
  868 mm², 1 kW) is about a sixth of a Rubin-class package's area at roughly half its power,
  with 2-die HBM stacks instead of 12–16-high HBM4. The linearity findings do not depend on
  scale; absolute temperatures do (`docs/assumptions.md`).
- **geometry7 has no layout-varying data**: its chiplets exceed the die width for 1D
  shelf-packing (`docs/compute.md`). The 45 files in `data/3d-ice-layout-geometry7/` are
  nominal-placement duplicates and are not used.
- **Every quantitative novelty claim in this paper has been through at least one adversarial
  prior-art check** (§9.14a, §9.16f/g, §9.16i) and several were narrowed or withdrawn as a
  result; this summary states only what survived.

---

## 6. Conclusion

The standard 3D-IC thermal-surrogate evaluation setup — fixed floorplan, varying power
amplitude only — does not distinguish a trained neural operator from a closed-form linear
solve. This is fixable (randomise placement), and the fix works, measurably. The field's own
external benchmark (IC-ThermBench) shows the diagnostic is not vacuous: benchmarks that vary
enough do discriminate, and neural methods earn their result there. What this project adds
beyond the diagnostic is narrow but real: a specific, seed-verified measurement of how much a
linear model's *representation* — not its architecture — determines whether it can find a
hotspot, on data no prior published benchmark evaluates this way.

**What to cite this for:** report a non-neural baseline before crediting an architecture;
report spatially-detrended error; report hotspot metrics separately from field metrics, gated
by a peak-sharpness check; and if a benchmark holds geometry and source locations fixed,
compare against power blurring or an influence-coefficient fit by name, not only against other
networks.
