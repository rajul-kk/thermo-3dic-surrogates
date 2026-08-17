# Compute Cost Reference — 3D-IC Thermal Surrogate Benchmark

All times on **2× NVIDIA T4** (16 GB each, 65 TFLOPS FP16, 320 GB/s).
CPU benchmarks measured on this machine; GPU estimates derived from measured
CPU epoch times × workload-specific T4/CPU speedup factors.

**Unit note, added 2026-08-16**: "Xk pts" figures below (e.g. geometry6's "470k pts") use
the *declared* mesh resolution (nx×ny×nz, e.g. 56×168×50=470,400 for geometry6), not the
point count 3D-ICE actually exports per file (geometry6 = 141,120, per
`docs/geometry_reference.md`'s corrected Overview table — 3D-ICE's z-grid is adaptive, so
the two differ for every geometry). FNO by default trains on the real per-file grid shape
(confirmed from training logs: `Grid from data: (100, 100, 10) (geometry declares (100,
100, 40))` for geometry1), so cost estimates keyed to the declared mesh may be
conservative overestimates rather than the true per-sample compute cost. Not corrected
throughout this file — flagging the discrepancy rather than re-deriving every estimate,
since these are cost *estimates* for future runs, not settled experimental results.

---

## GPU Specifications

| GPU | VRAM | FP32 | FP16 (Tensor Cores) | Memory BW | Notes |
|---|---|---|---|---|---|
| T4 | 16 GB | 8.1 TFLOPS | 65 TFLOPS | 320 GB/s | Turing, good for FP16 inference |
| 2× T4 | 32 GB | 16.2 TFLOPS | 130 TFLOPS | 640 GB/s (sum) | DDP or parallel |

**DDP efficiency** (2 GPUs vs 1): 70% (1.7× speedup). Gradient sync
overhead 5–10%, FFT bandwidth-bound ops don't halve perfectly.

**T4/CPU speedup factors used:**
- PINN (FP32, autograd-heavy, small tensors, Python loop): **45×**
- FNO (FP16 mixed, FFT-dominated, large tensors): **25×** (bandwidth-bound)
- CNO-FNO (FP16 mixed, Conv3d + small-latent FNO): **60×** (Conv3d im2col maps perfectly to tensor cores; GroupNorm is the CPU bottleneck, fast on GPU)
- PI-DeepONet (FP32 trunk autograd, small MLP): **30×**

---

## Measured CPU Epoch Times

| Model | Geometry | Grid | CPU time/epoch | Measured |
|---|---|---|---|---|
| PINN (807k params, 20k col, FP32) | geometry1 | 100×100×40 | **89.0 s** | Yes |
| PINN | geometry2a | 80×80×72 | **96.8 s** | Yes |
| FNO baseline (12.6M params, FP16) | geometry1 | 100×100×40 | **32.3 s** | Yes |
| FNO baseline | geometry2a | 80×80×72 | **36.0 s** | Yes |
| FNO baseline | geometry4 | 100×56×40 | **20.7 s** | Yes |
| FNO baseline | geometry5 | 100×56×50 | **24.0 s** | Estimated (+2s for 8→11 layers) |
| FNO baseline | geometry6 | 56×168×50 | **58.0 s** | Estimated (470k pts, 2.7× g5) |
| PI-FNO (FD residual) | geometry1 | 100×100×40 | **32.3 s** | Yes (no overhead vs baseline) |
| CNO-FNO (1.6M params) — data only | geometry1 | 100×100×40 | **62.1 s** | Yes |
| CNO-FNO + PI | geometry1 | 100×100×40 | **60.8 s** | Yes |
| PI-DeepONet (500k params) — PI step | all 7 | N/A | **27 s/epoch** (105 scenarios, 27 steps) | Yes |

> **CNO-FNO is slower than baseline FNO on CPU** (62s vs 32s) because full-resolution
> Conv3d + GroupNorm on the 100×100×40 grid are CPU-memory-bandwidth-bound and poorly
> accelerated by MKL.  On T4, Conv3d im2col GEMMs run at 65 TFLOPS FP16 (tensor cores)
> giving **60× CPU→GPU speedup** vs FNO's 25× (FFT is bandwidth-bound). CNO-FNO is
> therefore **faster than baseline FNO on GPU** despite being slower on CPU.

---

## Per-Model, Per-Geometry Training Time — Full Dataset, All 6 Geometries (T4)

Real train/test scenario counts per geometry (`data/3d-ice/`, verified 2026-08-17):

| Geometry | Train | Test | Points/file |
|---|---|---|---|
| geometry1 | 40 | 5 | 100,000 |
| geometry2a | 25 | 5 | 89,600 |
| geometry3 | 40 | 5 | 110,000 |
| geometry4 | 40 | 5 | 56,000 |
| geometry5 | 50 | 5 | 84,000 |
| geometry6 | 50 | 5 | 141,120 |

Only geometry1/geometry2a/geometry4 CPU epoch times are directly measured (see table
above); geometry5/geometry6 were already estimated from point-count scaling.
**geometry3 is newly estimated here** the same way: interpolating FNO's per-scenario CPU
cost (0.808 s/scenario at geometry1's 100k pts) linearly against geometry3's 110,000 pts
gives ~0.889 s/scenario × 40 train = **35.5 s/epoch (estimated, not measured)**.

The FNO g1↔g2a ratio (1.782×) is then reused as a shared per-geometry scaling factor for
PINN and CNO-FNO, since architecture-specific per-scenario cost tracks grid/layer
complexity similarly across models — validated against the one place two models were both
independently measured on the same geometry pair: applying the FNO g1→g2a ratio to PINN's
measured g1 time predicts g2a at 99.1 s vs. the actual measured 96.8 s (2.4% error).

**2× T4 DDP** below = single-T4 total ÷ 1.7 (the 70%-efficiency figure from the GPU
Specifications section above). Totals in the last row assume all 6 geometries trained
**sequentially** on one path (single-T4 column) — pairing two geometries per run across
both GPUs (as the existing All-7-Geometries section does) roughly halves that sum instead.

### PINN baseline (8,000 epochs, 45× T4 speedup)

| Geometry | CPU epoch (s) | T4 epoch (s) | Single-T4 total | 2× T4 DDP total | Basis |
|---|---|---|---|---|---|
| geometry1 | 89.0 | 1.98 | **4.4 h** | **2.6 h** | Measured |
| geometry2a | 96.8 | 2.15 | **4.8 h** | **2.8 h** | Measured |
| geometry3 | 97.8 | 2.17 | **4.8 h** | **2.8 h** | Estimated |
| geometry4 | 57.0 | 1.27 | **2.8 h** | **1.7 h** | Estimated |
| geometry5 | 66.1 | 1.47 | **3.3 h** | **1.9 h** | Estimated |
| geometry6 | 159.8 | 3.55 | **7.9 h** | **4.6 h** | Estimated |
| **All 6, sequential** | — | — | **~28.0 h** | **~16.5 h** | — |

### FNO baseline (500 epochs, 25× T4 speedup)

| Geometry | CPU epoch (s) | T4 epoch (s) | Single-T4 total | 2× T4 DDP total | Basis |
|---|---|---|---|---|---|
| geometry1 | 32.3 | 1.29 | **10.8 min** | **6.3 min** | Measured |
| geometry2a | 36.0 | 1.44 | **12.0 min** | **7.1 min** | Measured |
| geometry3 | 35.5 | 1.42 | **11.8 min** | **7.0 min** | Estimated |
| geometry4 | 20.7 | 0.83 | **6.9 min** | **4.1 min** | Measured |
| geometry5 | 24.0 | 0.96 | **8.0 min** | **4.7 min** | Estimated |
| geometry6 | 58.0 | 2.32 | **19.3 min** | **11.4 min** | Estimated |
| **All 6, sequential** | — | — | **~68.8 min** | **~40.5 min** | — |

### CNO-FNO + PI + FiLM (400 epochs, 60× T4 speedup, the recommended default)

| Geometry | CPU epoch (s) | T4 epoch (s) | Single-T4 total | 2× T4 DDP total | Basis |
|---|---|---|---|---|---|
| geometry1 | 62.1 | 1.04 | **6.9 min** | **4.1 min** | Measured |
| geometry2a | 69.2 | 1.15 | **7.7 min** | **4.5 min** | Estimated |
| geometry3 | 68.2 | 1.14 | **7.6 min** | **4.5 min** | Estimated |
| geometry4 | 39.8 | 0.66 | **4.4 min** | **2.6 min** | Estimated |
| geometry5 | 46.1 | 0.77 | **5.1 min** | **3.0 min** | Estimated |
| geometry6 | 111.5 | 1.86 | **12.4 min** | **7.3 min** | Estimated |
| **All 6, sequential** | — | — | **~44.1 min** | **~26.0 min** | — |

**Reading this table:** on a single T4, the full 6-geometry sweep costs ~28 GPU-hours for
PINN baseline, ~69 GPU-minutes for FNO baseline, or ~44 GPU-minutes for the recommended
CNO-FNO+PI+FiLM — the same ~19× PINN-vs-FNO and ~1.6× FNO-vs-CNO-FNO ratios already noted
in the geometry1-only table above hold in aggregate. Only the geometry1/2a/4 rows carry a
"Measured" CPU basis; geometry3/5/6 rows are extrapolated from point-count and per-geometry
scaling ratios and carry the same ±50% uncertainty as the rest of this file.

---

## Single-Geometry Training Time: 2× T4

DDP across both T4s. One geometry trained per run.

| Model | Variant / flags | Epochs | T4 epoch time | **2× T4 DDP time** | Notes |
|---|---|---|---|---|---|
| **PINN** (baseline) | — | 8,000 | ~2.0 s | **~2.6 h** | FP32, 2nd-order autograd |
| **PINN + upgrades** | `--compile --func-pde` | 8,000 | ~0.5 s | **~37 min** | compile 2.5×, func-pde 1.5×, batched 2× |
| **FNO baseline** | — | 500 | ~1.3 s | **~8 min** | FP16, data-only |
| **PI-FNO** | `--pi-weight 0.1` | 400 | ~1.4 s | **~7 min** | FD residual adds ~5%; fewer epochs |
| **CondFNO (FiLM)** | `--film` | 500 | ~1.4 s | **~8 min** | FiLM generator adds <5% |
| **PI-FNO + FiLM** | `--film --pi-weight 0.1` | 400 | ~1.5 s | **~8 min** | |
| **CNO-FNO + PI + FiLM** | `--best` | 400 | ~1.0 s | **~6 min** | 60× GPU speedup (tensor cores); 1.6M vs 12.6M params |
| **PI-DeepONet** *(1 geom only)* | `--pde-weight 0.1` | 1,000 | ~0.13 s | **~1 min** | 15 scenarios, 4 steps/epoch; underuses architecture |

> **Key ratio:** PINN (baseline) is **~19× slower** than best FNO per geometry;
> PINN with upgrades is **~4× slower** than `--best` CNO-FNO.

---

## All-7-Geometries Training Time: 2× T4

**Naming note (2026-08-18):** this section's title/body predate the 2026-08-06 dataset
correction (8 geometries -> 6: `g1/2a/2b/2c/g3/g4/g5/g6` -> `g1/2a/g3/g4/g5/g6`) and were
never updated to match -- "7" and "8" here refer to the OLD geometry count, not the new
`geometry7` CoWoS-L pilot added this session (see "GPU Cost: geometry7" section below,
after the A3/A3b pilots). Not fixed here since correcting the numbers themselves would
need re-deriving each figure; flagging the stale count so it isn't confused with the new
geometry7 pilot.

PINN and FNO run 8 separate models (2 in parallel, one per GPU, pairing by time).
PI-DeepONet trains one model on all 5 uniform-stack geometries simultaneously via DDP.
Geometry6 (470k pts) adds ~2.7× overhead vs. geometry5 for FNO/CNO runs.

| Model | Strategy | **Wall time (2× T4)** | Total GPU-hours | Notes |
|---|---|---|---|---|
| **PINN** (baseline) | 2× parallel | **~24 h** | 52 GPU-h | 4.4 h/geom avg × 8 / 1.5 (parallelism) |
| **PINN + upgrades** | 2× parallel (DDP per geom) | **~5 h** | 10 GPU-h | 37 min/geom × 8; g6 ~90 min (470k pts) |
| **FNO baseline** | 2× parallel | **~55 min** | 1.8 GPU-h | g6 adds ~15 min extra (470k pts, 500 epochs) |
| **PI-FNO** | 2× parallel | **~50 min** | 1.7 GPU-h | |
| **CondFNO (FiLM)** | 2× parallel | **~55 min** | 1.8 GPU-h | |
| **PI-FNO + FiLM** | 2× parallel | **~52 min** | 1.7 GPU-h | |
| **CNO-FNO + PI + FiLM** | 2× parallel | **~38 min** | 1.3 GPU-h | g6 single run ~16 min (tensor-core Conv3d scales well) |
| **PI-DeepONet** | DDP, 1 joint run | **~9 min** | 0.3 GPU-h | 5 stack geometries (g1/2a/2b/2c/g3); g4/5/6 use per-geom CNO-FNO |
| **GINO** *(theoretical)* | DDP, 1 joint run | **~2 h** | 4 GPU-h | GNN encoder/decoder overhead; one model all geoms |

> **PI-DeepONet all-7 total (12 min) is faster than CNO-FNO for a single geometry (~6 min × 4 rounds = 28 min).**
> CNO-FNO wins per-geometry accuracy; DeepONet wins cross-geometry deployment speed.
> CNO-FNO is slower than FNO on CPU but faster on GPU — use `--best` only with CUDA.

---

## GPU Cost: A3/A3b Pilot Architecture Comparisons (throttled & leakage, geometry1)

`kaggle_a3_throttled_arch_comparison.ipynb` and `kaggle_a3b_leakage_arch_comparison.ipynb`
(added 2026-08-16/17, `goal.md` Tracks A3/D) train **FNO, CondFNO (FiLM), CNO-FNO+PI+FiLM,
and CNO-FNO+attention (SAU-FNO style)** — same `channels=32, blocks=4, epochs=400, batch=4`
config for all four, on two small geometry1 pilot sets:

| Pilot | Train scenarios | Test scenarios | Notes |
|---|---|---|---|
| Throttled (A3) | 15 | 5 | 6/20 total scenarios triggered derating |
| Leakage (A3b) | 9 (converged only) | 4 (converged only) | 6/20 scenarios diverged (runaway), excluded from both splits |

Both pilots use geometry1's grid (100,000 pts/file, same as the full-dataset FNO benchmark
above), just far fewer scenarios — 15 or 9 train vs. ~40 for the full geometry1 training set
the 32.3s/epoch CPU number was measured on. Scaling that measurement by train-scenario count
(batch=4 in both cases, so cost is ~linear in scenario count) gives the CPU baseline for each
pilot; T4 times use the same speedup factors as the rest of this file (FNO/CondFNO 25×,
CNO-FNO 60×). CondFNO uses the 34s/epoch (32.3s + <5% FiLM overhead) baseline; CNO-FNO+attention
is estimated at 1.3× plain CNO-FNO's cost (axial self-attention on the small 8×8×5 latent grid,
not the full 3D grid — cheap, but not free).

| Model | Throttle pilot CPU/epoch (est.) | Throttle **1×T4**, 400 ep | Leakage pilot CPU/epoch (est.) | Leakage **1×T4**, 400 ep |
|---|---|---|---|---|
| FNO | ~12.1 s | **~3.2 min** | ~7.3 s | **~1.9 min** |
| CondFNO (FiLM) | ~12.8 s | **~3.4 min** | ~7.7 s | **~2.0 min** |
| CNO-FNO+PI+FiLM | ~23.3 s | **~2.6 min** | ~14.0 s | **~1.6 min** |
| CNO-FNO+attention | ~30.3 s | **~3.4 min** | ~18.2 s | **~2.0 min** |

**Why 2× T4 DDP is not recommended for these pilots** (unlike the full-dataset table above):
at batch=4, 15 train scenarios is only ~4 batches/epoch, and 9 converged leakage scenarios
is only ~2-3 — splitting either across 2 GPUs leaves ~1-2 batches/GPU/epoch. DDP's gradient
sync happens once per batch regardless of batch size, so at this scale the fixed per-step
communication overhead (NCCL all-reduce latency, ~tens of ms) is comparable to or larger than
the compute the tiny batch actually does. The 70%-efficiency/1.7× figure quoted earlier in
this file was derived from full-dataset runs with far more batches/epoch to amortize that
overhead over; it does not apply here. Expect 2×T4 DDP on these pilots to be **flat or slower
than single-T4 wall-clock**, not faster — run them on 1× T4 and use the second GPU for a
different model in parallel instead (e.g. FNO on GPU 0, CNO-FNO on GPU 1, ~7-8 min total for
all four models across both pilots rather than ~11 min sequential on one GPU).

All four models × both pilots fit comfortably inside a single Kaggle session (well under the
30 GPU-hr/week free-tier budget) even run sequentially on 1× T4, matching the notebooks' own
"under 30 min total for all four" estimate. These per-epoch estimates are **not yet measured**
— extrapolated from the existing FNO/CNO-FNO CPU benchmarks and speedup factors above, carrying
the same ±50% uncertainty noted at the end of this file; the notebooks' own summary JSON output
(`a3_..._summary.json` / `a3b_..._summary.json`) will supersede this table once actually run.

---

## GPU Cost: geometry7 (CoWoS-L reticle-stitched pilot, added 2026-08-18)

`geometry7` (`src/core/geometry_builders.py`) is a new pilot geometry, NOT part of the
standard 6-geometry dataset -- see `build_geometry7`'s docstring for the full design
rationale. Structurally it is the largest and most complex geometry in this project by
every axis measured:

| geometry | area (mm²) | layers | power blocks | die footprints | declared pts | actual pts (measured) | TDP (W) |
|---|---|---|---|---|---|---|---|
| geometry1 | 100 | 6 | 4 | 0 | 400,000 | 100,000 | 125 |
| geometry2a | 64 | 10 | 6 | 0 | 460,800 | 89,600 | 30 |
| geometry3 | 625 | 6 | 8 | 0 | 400,000 | 110,000 | 250 |
| geometry4 | 350 | 6 | 8 | 2 | 224,000 | 56,000 | 200 |
| geometry5 | 350 | 11 | 15 | 4 | 280,000 | 84,000 | 400 |
| geometry6 | 588 | 11 | 29 | 19 | 470,400 | 141,120 | 700 |
| **geometry7** | **868** | **11** | **46** | **37** | **694,400** | **208,320** | **1000** |

Actual point count was measured directly (one real 3D-ICE solve, 2026-08-18), not
extrapolated: 208,320 points, giving a real per-file grid of `(56, 248, 15)` via the same
`n_points / (nx × ny)` method `scripts/train_fno.py` uses to derive z-depth from data
(§ "Overview" table's method in `docs/geometry_reference.md`). The declared/actual ratio
(0.30) matches geometry6's (0.30) almost exactly, suggesting the adaptive z-grid's
behavior is a property of the layer stack (same 11 layers, same thicknesses, same
`MAX_SUBLAYER_UM` subdivision rule) rather than of lateral extent -- consistent with
geometry7 reusing geometry5/6's exact layer stack and changing only the interposer layer's
lateral material structure and overall footprint size.

**Why it's the most complex geometry**: geometry6's 6-HBM CoWoS-S design already had 11
layers and 588 mm²; geometry7 adds a 2nd compute die, 2 I/O dies, 2 more HBM4 stacks (8
total), and replaces the uniform-silicon interposer with a sparse-bridge structure (9
silicon islands in an organic-substrate field) that no other geometry has -- a passive
layer with real internal lateral heterogeneity, as opposed to every other passive layer in
this project, which is uniform material end to end.

**Real generation cost (measured)**: all 40 scenarios (15 base train + 5 test + 20
extra-train, matching the standard per-geometry train/test recipe) solved successfully,
0 failures, in **45.6 minutes total (68.4 s/scenario average)** -- close to the ~60-70s
estimated from the first few scenarios before the full run completed.

**A real finding surfaced by the data, not a solver bug: `split_chiplet_a_hot` scenarios
produce physically-implausible-for-silicon peak temperatures.** Peak temperature across
the 40 scenarios: median 134.0°C, mean 200.0°C, 17/40 exceed 150°C, 5/40 exceed 300°C
(worst: 1125.3°C). Every one of the extreme outliers (1125.3, 786.2, 572.0, 561.4,
368.2°C) is the `split_chiplet_a_hot` pattern -- the existing generator pattern that
concentrates 5-10x power onto chipA while chipB/HBM idle at 10%. The mechanism is
directly traceable, not mysterious: `chipA`'s die_zone_1 footprint (x=1-10mm) has almost
no coverage from the `bridge_ab` LSI island (which only starts at x=9.5mm), so nearly all
of chipA's heat must cross the full 300µm organic-substrate field (k=0.5 W/m·K) vertically
to reach the heat-sink boundary below. Back-of-envelope confirms it exactly: for
`geometry7_train_034` (chipA at 180.86 W/cm² under `split_chiplet_a_hot`), ΔT = q·(t/k) =
1,808,570 W/m² × (300e-6/0.5) m²·K/W = **1085 K**, against a simulated rise of 1100 K
(825°C - (-25°C)... i.e. 1125.3°C - 25°C ambient = 1100.3 K) -- within 1.5% of the
first-order series-resistance estimate, confirming this is the real mechanism, not a
numerical artifact.

This is arguably a genuine (if now over-illustrated) demonstration of CoWoS-L's real
packaging risk cited in the literature search behind this geometry ("every step up the
interposer scale introduces non-linear increases in ... thermal management difficulty") --
but it also means **this specific 40-scenario pilot is not yet a well-behaved training
set as generated**: a real package would place bridge/thermal-via coverage under every
compute die, not just at the reticle seam and HBM edges, precisely to avoid this. The
current `bridge_ab`/`bridge_hbm{1..8}` placement only models signal-routing bridges: it
omits power-delivery thermal vias real designs would also route under each compute die.
**Flagged as a modelling limitation to fix before this pilot is used for baseline/FNO
work**, not silently absorbed into the dataset -- the honest fix is broader bridge
coverage under chipA/chipB (or excluding `split_chiplet_a_hot`/`split_chiplet_b_hot` from
this geometry's pattern set until that's done), not re-normalising the results after the
fact.

**FNO/CNO-FNO/PINN training cost, extrapolated** (same point-count-scaling methodology as
the geometry3/5/6 estimates in the table above, cross-validated here via two independent
anchors -- scaling from geometry1's measured 0.808 s/scenario directly by point-count
ratio gives 1.68 s/scenario; fitting a line through geometry1's measured and geometry6's
already-estimated per-scenario costs and evaluating at geometry7's point count gives 1.735
s/scenario; both land within 3% of each other, so 1.7 s/scenario is used below):

| model | epochs | CPU epoch (35 train scenarios) | T4 epoch | single-T4 total | 2×T4 DDP total |
|---|---|---|---|---|---|
| FNO baseline | 500 | ~59.5 s | ~2.38 s | **~19.8 min** | **~11.7 min** |
| CNO-FNO+PI+FiLM | 400 | ~114.5 s | ~1.91 s | **~12.7 min** | **~7.5 min** |
| PINN baseline | 8,000 | ~163.8 s | ~3.64 s | **~8.1 h** | **~4.8 h** |

**The counterintuitive result**: despite geometry7 having 1.48× geometry6's area and 47%
more points per file, its FNO/CNO-FNO training wall-clock is nearly IDENTICAL to
geometry6's (§ "Per-Model, Per-Geometry" table above: geometry6 FNO ~19.3 min single-T4 vs
geometry7's ~19.8 min; CNO-FNO 12.4 min vs 12.7 min). The pilot's smaller training-scenario
count (35, vs geometry6's full 50) almost exactly offsets the larger per-scenario grid
cost -- a reminder that **training wall-clock is driven by (scenario count) × (per-scenario
cost)**, not by geometry size alone, and a larger geometry with fewer scenarios can cost
about the same as a smaller geometry with more. All figures here carry the same ±50%
uncertainty as the rest of this file, plus the additional uncertainty of extrapolating
across a wider size gap than the geometry3/5 estimates did.

---

## Inference Cost (per scenario, single T4, batch=1)

| Model | Forward time | Memory | Notes |
|---|---|---|---|
| **PINN** | 5–15 ms | ~300 MB | Point-wise MLP on 60k pts |
| **FNO baseline** | 8–12 ms | ~1–2 GB | FFT on 100×100×40 grid |
| **PI-FNO** | 8–12 ms | ~1–2 GB | FD residual not computed at inference |
| **CondFNO (FiLM)** | 9–13 ms | ~1–2 GB | Tiny FiLM generator overhead |
| **PI-FNO + FiLM** | 9–13 ms | ~1–2 GB | |
| **CNO-FNO + PI + FiLM** | 10–18 ms | ~700 MB | CNN encode (fast on GPU) + latent FNO (tiny FFT) + CNN decode |
| **PI-DeepONet** | 5–15 ms | ~200 MB | MLP branch + MLP trunk at N query points; no FFT |
| **GINO** *(theoretical)* | 15–25 ms | ~1–2 GB | GNN encode + latent FNO + GNN decode |

> **CNO-FNO uses less GPU memory than baseline FNO** despite having CNN layers:
> the FNO activation tensors dominate memory, and the latent FNO is 64× smaller.

---

## Model Parameters

| Model | Parameters | Notes |
|---|---|---|
| FourierPINN | 807k | 6 ResBlocks × hidden=256 |
| FourierPINN (cpu-fast) | 140k | 4 blocks × hidden=128 |
| FNO baseline | 12.6M | hidden=32, modes=(16,16,12), 4 blocks |
| CondFNO (FiLM) | 12.8M | +FiLMGenerator ~200k |
| **CNO-FNO + PI + FiLM** | **1.6M** (g1), 2.4M (g2) | Latent FNO modes (8,8,5); 7.8× fewer params than FNO |
| **WHNO** | **12.6M** | Same channels/modes config as FNO baseline (Walsh-Hadamard basis is a drop-in swap for the spectral conv, so equal param count by construction) |
| PI-DeepONet | 538k | Branch 538→256×4→128 + Trunk 132→256×4→128 |
| **ARO** | **~1.06M** | n_layers=11 (geometry5/6 default), hidden=32, 4 FiLM-FNO blocks, weight-tied across all z-layers |
| GINO (theoretical) | ~8–12M | FNO latent + GNN encoder/decoder |

> **WHNO and ARO training-time benchmarks are not yet measured at full config** (only
> smoke-tested at reduced channels/epochs to validate correctness — see their respective
> notebook/script validation runs). Rough estimates below are architecture-based, not
> measured; treat with the same ±50% caveat the GPU extrapolations below already carry,
> but with lower confidence since even the CPU baseline hasn't been measured at scale.
> - **WHNO**: expect CPU/GPU cost roughly on par with FNO baseline — the Walsh-Hadamard
>   transform is the same O(N log N) complexity class as FFT, with an added grid-padding-
>   to-next-power-of-2 step (e.g. 100→128, 40→64) that FNO doesn't need, adding modest
>   overhead proportional to the padding ratio.
> - **ARO**: the autoregressive z-layer loop means training cost scales with `n_layers`
>   sequential 2D-FNO forward/backward passes per sample, rather than one 3D pass — likely
>   slower per-epoch than CNO-FNO at equal channel width despite far fewer total parameters,
>   though each individual 2D step is far cheaper than a full 3D op. RNO-style windowed
>   self-rollout training (see `src/aro/trainer.py`) adds a second forward/backward pass per
>   step during pretraining, roughly doubling per-epoch cost during that phase only.

---

## Architecture Comparison

| Model | Cross-geom? | Sharp interfaces | Global spreading | Param count | CLI |
|---|---|---|---|---|---|
| PINN | No | Good (PDE) | Good (MLP) | 807k | `train_pinn.py` |
| FNO baseline | No (per grid) | Poor | Excellent | 12.6M | `--model fno` |
| CondFNO (FiLM) | No | Poor | Excellent + BC adapt | 12.8M | `--model cond-fno` |
| **CNO-FNO+PI+FiLM** | No | **Excellent** | **Excellent** | **1.6M** | *(default)* |
| **WHNO** | No (per grid) | **Excellent** (validated: 0% Gibbs overshoot vs FNO's ~8.7% on a synthetic step function) | Good | 12.6M | `--model whno` |
| PI-DeepONet | **Yes (g1/2a/2b/2c/g3)** | Medium | Medium | 538k | `train_deeponet.py` |
| **ARO** | No (per grid, weight-tied across z) | Good (autoregressive z-conditioning) | Good | ~1.06M | `train_aro.py` |
| CNO-FNO+PI+FiLM (g6) | No (per grid) | **Excellent** | **Excellent** | ~2.4M | `--geometry geometry6` |
| GINO | **Yes** | Excellent | Excellent | ~10M | Not implemented |

---

## Recommended Training Commands

```bash
# CNO-FNO (default model, physics on, recommended):
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice

# Higher capacity for publication runs:
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --channels 64

# Baseline FNO (ablation / fast iteration):
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --model fno

# FiLM-only (no CNO, with physics):
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --model cond-fno --physics

# All 6 geometries in parallel on 2 T4s (run simultaneously on separate GPUs):
CUDA_VISIBLE_DEVICES=0 python scripts/train_fno.py --geometry geometry1 geometry3 \
    --data data/3d-ice --name g1g3
CUDA_VISIBLE_DEVICES=1 python scripts/train_fno.py --geometry geometry2a geometry4 \
    --data data/3d-ice --name g2a-g4
# geometry6 is large (470k pts) — run solo or paired with a small geometry:
CUDA_VISIBLE_DEVICES=0 python scripts/train_fno.py --geometry geometry6 --data data/3d-ice
CUDA_VISIBLE_DEVICES=1 python scripts/train_fno.py --geometry geometry5 --data data/3d-ice

# Cross-geometry surrogate (5 uniform-stack geometries, one run):
python scripts/train_deeponet.py --data data/3d-ice \
    --output checkpoints/deeponet --pde-weight 0.1 --epochs 1000
# geometry4/5 (2.5D chiplets with lateral k variation): use per-geometry CNO-FNO
# python scripts/train_fno.py --geometry geometry4 --data data/3d-ice
# python scripts/train_fno.py --geometry geometry5 --data data/3d-ice

# PINN with GPU efficiency upgrades:
python scripts/train_pinn.py --geometry geometry1 --data data/3d-ice \
    --output checkpoints/pinn --compile --func-pde

# PINN or FNO on CPU (no GPU):
python scripts/train_pinn.py --geometry geometry1 --data data/3d-ice --cpu-fast
python scripts/train_fno.py  --geometry geometry1 --data data/3d-ice --cpu-fast
```

---

## Notes on Estimation Methodology

CPU epoch times were measured directly on this machine (Windows 11, PyTorch CPU).
GPU times were extrapolated using T4/CPU speedup factors validated against the
FNO/PINN literature:

- FNO: Li et al. 2021 report 36h on V100 for NS-3D (1000 samples, 500 epochs).
  Scaling to 15 samples + T4 gives 8–12 min/geometry, consistent with our extrapolation.
- PINN: Raissi et al. 2019; modern thermal PINN benchmarks report 2–6h on V100
  for comparable parameter counts. Our 2.6h on 2× T4 is consistent.
- CNO-FNO: derived from latent grid reduction (64× FFT savings) and CNN overhead.
  Will be confirmed once CUDA environment is available.

All GPU estimates carry ±50% uncertainty. Relative ordering is reliable;
absolute times should be validated with a CUDA run.
