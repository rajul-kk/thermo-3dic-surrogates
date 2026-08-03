# Geometry Reference — 3D-ICE Thermal PINN Benchmark

This document describes the benchmark geometries, their physical layer stacks,
mesh discretisation, power block layouts, and the training/test scenario sets used
to generate the dataset in `data/3d-ice/`.

---

## Overview

| Name | Type | Die footprint | Layers | TSV density | Mesh (x × y × z) | Points/file |
|---|---|---|---|---|---|---|
| `geometry1` | Single die | 10 × 10 mm | 6 | — | 100 × 100 × 40 | 60,000 |
| `geometry2a` | Dual die + hybrid bond | 8 × 8 mm | 10 | 3 % | 80 × 80 × 72 | 46,080 |
| `geometry2b` | Dual die + hybrid bond | 8 × 8 mm | 10 | 5 % | 80 × 80 × 72 | 46,080 |
| `geometry2c` | Dual die + hybrid bond | 8 × 8 mm | 10 | 10 % | 80 × 80 × 72 | 46,080 |
| `geometry3` | Server die | 25 × 25 mm | 6 | — | 100 × 100 × 40 | 60,000 |
| `geometry4` | 2.5D chiplet | 25 × 14 mm | 6 | — | 100 × 56 × 40 | 33,600 |
| `geometry5` | 3D-on-2.5D (CoWoS, Tier 0+1) | 25 × 14 mm | 11 | 3 % (chiplet B) | 100 × 56 × 50 | 44,800 |
| `geometry6` | CoWoS + 6× HBM (MI300X-class) | 42 × 14 mm | 11 | 3 % (each HBM) | 56 × 168 × 50 | 470,400 |

All coordinates in µm. z = 0 is the bottom face of the heat sink (coolant side).

---

## Coordinate System

```
z (up, µm)
│
│  ← die (active, heat-generating)
│  ← TIM
│  ← spreader
│  ← TIM
│  ← heat sink
│
0 ─────── x (µm)
         ╲
          y (µm)
```

- **x, y** span the die footprint; (0, 0) is the bottom-left corner.
- **z** increases from heat sink bottom to die top.
- Convective cooling is applied at z = 0 (bottom of heat sink).

---

## Geometry 1 — Single Die (2D Cross-Sectional Stack)

### Layer Stack

| Index | Name | Material | Thickness (µm) | z range (µm) | k (W/m·K) | Active |
|---|---|---|---|---|---|---|
| 0 | `heat_sink` | Copper | 5000 | 0 – 5000 | 400 | No |
| 1 | `tim_bottom` | TIM | 100 | 5000 – 5100 | 4 | No |
| 2 | `spreader` | Copper | 1000 | 5100 – 6100 | 400 | No |
| 3 | `tim_top` | TIM | 100 | 6100 – 6200 | 4 | No |
| 4 | `die` | Silicon | 150 | 6200 – 6350 | 148 | **Yes** |
| 5 | `tim2` | TIM | 50 | 6350 – 6400 | 4 | No |

Total height: **6400 µm**

### Power Blocks

Four 3 mm × 3 mm blocks in the four corners of the 10 × 10 mm die (1 mm margins).

| Block | x (µm) | y (µm) | w × h (mm) | Area (cm²) |
|---|---|---|---|---|
| `block1` | 1000 | 1000 | 3 × 3 | 0.090 |
| `block2` | 6000 | 1000 | 3 × 3 | 0.090 |
| `block3` | 1000 | 6000 | 3 × 3 | 0.090 |
| `block4` | 6000 | 6000 | 3 × 3 | 0.090 |

### Mesh

100 × 100 × 40 = **400,000 mesh points** (100 µm/cell in x and y).
3D-ICE returns 6 layers × 100 × 100 = **60,000 points** in the NPZ `coords` array.

---

## Geometry 2a / 2b / 2c — Dual-Die 3D Stack with Hybrid Bonding

### Physical Description

Two-die 3D-IC stack with **Cu-Cu hybrid bonding** (as of this benchmark release).
Die 1 (bottom) and Die 2 (top) are bonded through a 5 µm hybrid bonding interface
(k_eff = 60 W/m·K, 9 µm Cu-pillar pitch). Each die has a TSV region providing
through-silicon thermal paths. The three variants differ in TSV density (3/5/10 %).

Previous releases used micro-bump bonding (25 µm, k = 50 W/m·K). The upgrade
reduces die-to-die interface resistance by ~6× and better represents production
3D-IC stacks as of 2023+ (Intel Foveros, TSMC SoIC).

### Layer Stack (identical structure for 2a / 2b / 2c)

| Index | Name | Material | Thickness (µm) | z range (µm) | k (W/m·K) | Active |
|---|---|---|---|---|---|---|
| 0 | `heat_sink` | Copper | 5000 | 0 – 5000 | 400 | No |
| 1 | `tim_sink` | TIM | 100 | 5000 – 5100 | 4 | No |
| 2 | `spreader` | Copper | 1000 | 5100 – 6100 | 400 | No |
| 3 | `tim_die2` | TIM | 100 | 6100 – 6200 | 4 | No |
| 4 | `die2_active` | Silicon | 50 | 6200 – 6250 | 148 | **Yes** |
| 5 | `die2_tsv` | Si + TSV | 100 | 6250 – 6350 | k_tsv | No |
| 6 | `hybrid_bonding` | Cu-Cu hybrid bond | 5 | 6350 – 6355 | 60 | No |
| 7 | `die1_tsv` | Si + TSV | 100 | 6355 – 6455 | k_tsv | No |
| 8 | `die1_active` | Silicon | 50 | 6455 – 6505 | 148 | **Yes** |
| 9 | `tim2` | TIM | 50 | 6505 – 6555 | 4 | No |

Total height: **6555 µm**

### TSV Layer Thermal Properties

| Variant | TSV density φ | k_tsv (W/m·K) | ρCp_tsv (J/m³·K) |
|---|---|---|---|
| geometry2a | 3 % | 155.6 | 1.68 × 10⁶ |
| geometry2b | 5 % | 160.6 | 1.70 × 10⁶ |
| geometry2c | 10 % | 173.2 | 1.76 × 10⁶ |

Formula: `k_eff = (1 − φ) × k_Si + φ × k_Cu` (arithmetic mean, upper bound).

### Power Blocks

Six blocks per geometry: two compute cores and one TSV array per die.

| Block | Die | x (µm) | y (µm) | w × h (µm) | TSV |
|---|---|---|---|---|---|
| `core1_d1` | Die 1 | 1000 | 1000 | 2500 × 2500 | No |
| `core2_d1` | Die 1 | 4500 | 1000 | 2500 × 2500 | No |
| `tsv_array_d1` | Die 1 | 3000 | 3500 | 2000 × 2000 | **Yes** |
| `core1_d2` | Die 2 | 1000 | 1000 | 2500 × 2500 | No |
| `core2_d2` | Die 2 | 4500 | 5500 | 2500 × 2500 | No |
| `tsv_array_d2` | Die 2 | 3000 | 3500 | 2000 × 2000 | **Yes** |

### Mesh

80 × 80 × 72 = **460,800 mesh points** (100 µm/cell in x and y).
3D-ICE returns 10 layers × 80 × 80 = **64,000 points** in the NPZ `coords` array.

---

## Geometry 3 — Server-Class Single Die (25 × 25 mm)

### Layer Stack

| Index | Name | Material | Thickness (µm) | z range (µm) | k (W/m·K) | Active |
|---|---|---|---|---|---|---|
| 0 | `heat_sink` | Copper | 5000 | 0 – 5000 | 400 | No |
| 1 | `tim_bottom` | TIM | 100 | 5000 – 5100 | 4 | No |
| 2 | `spreader` | Copper | 2000 | 5100 – 7100 | 400 | No |
| 3 | `tim_top` | TIM | 100 | 7100 – 7200 | 4 | No |
| 4 | `die` | Silicon | 200 | 7200 – 7400 | 148 | **Yes** |
| 5 | `tim2` | TIM | 50 | 7400 – 7450 | 4 | No |

Total height: **7450 µm**. 8 × (4 × 4 mm) blocks in two rows of 4 with 10 mm IO gap.

Mesh: 100 × 100 × 40 = **400,000 pts** → 3D-ICE returns **60,000 pts/file**.

---

## Geometry 4 — 2.5D Chiplet Assembly (25 × 14 mm)

### Layer Stack

| Index | Name | Material | Thickness (µm) | z range (µm) | k (W/m·K) | Active |
|---|---|---|---|---|---|---|
| 0 | `heat_sink` | Copper | 5000 | 0 – 5000 | 400 | No |
| 1 | `tim_sink` | TIM | 100 | 5000 – 5100 | 4 | No |
| 2 | `spreader` | Copper | 1000 | 5100 – 6100 | 400 | No |
| 3 | `tim_die` | TIM | 100 | 6100 – 6200 | 4 | No |
| 4 | `die_zone` | Si / underfill | 150 | 6200 – 6350 | 148 / 0.7 | **Yes** |
| 5 | `interposer` | Silicon | 100 | 6350 – 6450 | 148 | No |

Total height: **6450 µm**

ChipA (compute, 10 × 12 mm at x=2mm, y=1mm): 4 × (5 × 6 mm) blocks in `die_zone`.
ChipB (IO, 8 × 12 mm at x=15mm, y=1mm): 4 × (4 × 6 mm) blocks in `die_zone`.
Gap x ∈ [12, 15 mm]: underfill k = 0.7 W/m·K.

Mesh: 100 × 56 × 40 = **224,000 pts** → 3D-ICE returns **33,600 pts/file**.

---

## Geometry 5 — 3D-on-2.5D, Tier 0+1 Upgraded (CoWoS, 25 × 14 mm)

### Physical Description

CoWoS-style assembly: Chiplet A (compute, 10 × 12 mm) + Chiplet B (HBM-style, 8 × 12 mm,
two-die TSV stack) on a shared 25 × 14 mm silicon interposer.

**Tier 0 upgrades applied:**
- RDL self-heating layer (`rdl_layer`, 5 µm, active) with parameterised Joule fraction (1–10 %)
- Die k reduced to 80 W/m·K (`silicon_low_k`, N5/N3 node Cu/low-k composite)

**Tier 1 upgrades applied:**
- C4 bump array (`c4_bumps`, 100 µm, k_eff = 15 W/m·K) between RDL and die_zone_1
- TIM1 indium solder (`tim_top`, 50 µm, k = 80 W/m·K)  — **TIM pump-out sweep: k = 80/40/10/5**
- TIM2 thickness 125 µm (`tim_sink`)
- Interposer 300 µm bulk Si (Kou 2022 published dims)

**Hybrid bonding added:** 5 µm Cu-Cu layer between `die_zone_1` and `tsv_zone`.

### Layer Stack

| Index | Name | Material | Thickness (µm) | z range (µm) | k (W/m·K) | Active |
|---|---|---|---|---|---|---|
| 0 | `heat_sink` | Copper | 5000 | 0 – 5000 | 400 | No |
| 1 | `tim_sink` | TIM grease | 125 | 5000 – 5125 | 4 (TIM2) | No |
| 2 | `spreader` | Copper | 1000 | 5125 – 6125 | 400 | No |
| 3 | `tim_top` | Indium solder | 50 | 6125 – 6175 | **80** (TIM1, swept 5–80) | No |
| 4 | `interposer` | Silicon | 300 | 6175 – 6475 | 148 | No |
| 5 | `rdl_layer` | Si low-k | 5 | 6475 – 6480 | 80 | **Yes** |
| 6 | `c4_bumps` | C4 array | 100 | 6480 – 6580 | 15 | No |
| 7 | `die_zone_1` | Si low-k | 50 | 6580 – 6630 | 80 | **Yes** |
| 8 | `hybrid_bonding` | Cu-Cu bond | 5 | 6630 – 6635 | 60 | No |
| 9 | `tsv_zone` | Si+TSV (3 %) | 100 | 6635 – 6735 | 155.6 (B) / 0.7 (gap) | No |
| 10 | `die_zone_2` | Si low-k | 50 | 6735 – 6785 | 80 (B) / 0.7 (gap) | **Yes** |

Total height: **6785 µm** (11 layers)

### Power Blocks

| Block(s) | Layer | Description |
|---|---|---|
| `chipA_rdl`, `chipB_rdl` | `rdl_layer` | RDL Joule heating — power = `rdl_joule_fraction` × base |
| `chipA_c1..c4` | `die_zone_1` | Compute chiplet A, 2 × 2 grid of 5 × 6 mm |
| `chipB_d1_c1..c4` | `die_zone_1` | HBM chiplet B die1, 2 × 2 grid of 4 × 6 mm |
| `chipB_tsv` | `tsv_zone` | Passive TSV region (is_tsv_region=True) |
| `chipB_d2_c1..c4` | `die_zone_2` | HBM chiplet B die2, 2 × 2 grid of 4 × 6 mm |

### Scenario-Variable Parameters

| Parameter | Scope | Values |
|---|---|---|
| `rdl_joule_fraction` | All g5 extra scenarios | 0.01, 0.03, 0.05 (default), 0.08, 0.10 |
| `layer_k_overrides['tim_top']` | TIM pump-out scenarios | 80.0 (fresh), 40.0, 10.0, 5.0 W/m·K |

### Mesh

100 × 56 × 50 = **280,000 mesh pts** (250 µm/cell in x and y).
3D-ICE returns 11 layers × 100 × 56 = **61,600 pts/file** (previously 44,800 with old 8-layer stack).

### Per-Layer Floorplan Files

Active layers: `rdl_layer` (layer 5), `die_zone_1` (layer 7), `die_zone_2` (layer 10).
Three `.flp` files written per scenario.

---

## Geometry 6 — CoWoS + 6 × HBM Stacks (MI300X-class, 42 × 14 mm)

### Physical Description

Extends geometry5 to six HBM stacks side-by-side on the interposer, matching the
layout of AMD MI300X (6 × HBM3). Each HBM stack is an independent two-die TSV stack
(same `die_zone_1 / hybrid_bonding / tsv_zone / die_zone_2` structure as chiplet B
in geometry5). Chiplet A (compute) is unchanged at 10 × 12 mm.

Footprint: **42 × 14 mm**
```
chipA   : x = 1 mm,  10 × 12 mm
hbm1    : x = 12 mm,  4 × 12 mm
hbm2    : x = 17 mm,  4 × 12 mm
hbm3    : x = 22 mm,  4 × 12 mm
hbm4    : x = 27 mm,  4 × 12 mm
hbm5    : x = 32 mm,  4 × 12 mm
hbm6    : x = 37 mm,  4 × 12 mm  (right margin 1 mm → 42 mm total)
```

### Layer Stack

Identical 11-layer structure to geometry5. Total height: **6785 µm**.

### Power Blocks

| Block(s) | Layer | Description |
|---|---|---|
| `chipA_rdl` | `rdl_layer` | Compute RDL Joule heating |
| `hbm1_rdl .. hbm6_rdl` | `rdl_layer` | HBM RDL Joule heating (6 blocks) |
| `chipA_c1..c4` | `die_zone_1` | Compute die, 2 × 2 grid of 5 × 6 mm |
| `hbm1_d1 .. hbm6_d1` | `die_zone_1` | HBM bottom die, 4 × 12 mm each |
| `hbm1_tsv .. hbm6_tsv` | `tsv_zone` | Passive TSV (is_tsv_region=True) |
| `hbm1_d2 .. hbm6_d2` | `die_zone_2` | HBM top die, 4 × 12 mm each |

Total active power blocks: **23** (excluding 6 TSV passive regions).

### Mesh

56 × 168 × 50 = **470,400 mesh pts** (250 µm/cell in x and y).
3D-ICE returns 11 layers × 56 × 168 = **103,488 pts/file**.

---

## Material Library

All properties at 300 K unless noted.

| Material | k (W/m·K) | ρCp (J/m³·K) | Used in |
|---|---|---|---|
| Silicon | 148 | 1.63 × 10⁶ | g1/2/3/4 die, interposer |
| Copper | 400 | 3.55 × 10⁶ | Heat sink, spreader |
| TIM (thermal grease) | 4 | 4.00 × 10⁶ | TIM layers (all geometries) |
| TIM indium solder | **80** | 1.70 × 10⁶ | `tim_top` in g5/g6 (TIM1) |
| Silicon low-k | **80** | 1.63 × 10⁶ | Die active layers in g5/g6 (N5/N3 node) |
| C4 bump array | **15** | 1.70 × 10⁶ | `c4_bumps` in g5/g6 |
| Hybrid bonding | **60** | 3.30 × 10⁶ | Die-to-die interface in g2/g5/g6 |
| Si + TSV 3 % | 155.6 | 1.68 × 10⁶ | TSV layers (g2a, g5, g6 HBM stacks) |
| Si + TSV 5 % | 160.6 | 1.70 × 10⁶ | TSV layers (g2b) |
| Si + TSV 10 % | 173.2 | 1.76 × 10⁶ | TSV layers (g2c) |

**Temperature-dependent silicon k:** `k(T) = 148 × (300/T)^1.3` W/m·K
(Glassbrenner & Slack, 1964). Used in PINN physics loss; 3D-ICE uses room-temperature
constant per scenario.

---

## Scenario Set

### Dataset Summary

| Geometry | Train | Test | Total | Key scenario features |
|---|---|---|---|---|
| `geometry1` | 40 | 5 | 45 | +25 extra: HTC/power sweep, `random_smooth` |
| `geometry2a` | 25 | 5 | 30 | +10 extra; hybrid bonding interface |
| `geometry2b` | 25 | 5 | 30 | +10 extra; hybrid bonding |
| `geometry2c` | 25 | 5 | 30 | +10 extra; hybrid bonding |
| `geometry3` | 40 | 5 | 45 | +25 extra: server TDP sweep |
| `geometry4` | 30 | 5 | 35 | +15 extra: split-chiplet lateral coupling |
| `geometry5` | 50 | 5 | 55 | +35 extra incl. TIM k-sweep (k=80/40/10/5) + RDL fraction (1–10%) |
| `geometry6` | 50 | 5 | 55 | +35 extra: same as g5 + 6-HBM split-chiplet patterns |
| **Total** | **280** | **40** | **320** | **320 `.npz` files** |

### Boundary Condition Parameters

| Parameter | Original (001–015) | Extended (016+) | Test |
|---|---|---|---|
| Peak power density | 0.1 – 20.0 W/cm² | 0.3 – 10.0 W/cm² | 0.3, 1.5, 3.0 W/cm² |
| HTC | 500 – 10,000 W/m²·K | 500 – 200,000 W/m²·K | 3000, 7500 W/m²·K |
| Ambient temperature | 25 – 85 °C | 25 – 75 °C | 35, 55 °C |
| TIM1 k (g5/g6 only) | 80 W/m·K (fresh) | 5, 10, 40, 80 W/m·K | 80 W/m·K |
| RDL Joule fraction (g5/g6) | 5 % | 1, 3, 5, 8, 10 % | 5 % |

### Power Distribution Patterns

| Pattern | Description | Block values |
|---|---|---|
| `uniform` | All blocks equal | All = base |
| `hotspot` | One block hot | Block 0 = base; others = 0.1 × base |
| `checkerboard` | Alternating | Even = base; odd = 0.5 × base |
| `gradient` | Linear ramp | 0.5 × base → base |
| `dual_hotspot` | Two blocks hot | Blocks 0,1 = base; others = 0.2 × base |
| `extreme_hotspot` | Single extreme | Block 0 = base; others = 0.01 × base |
| `random_smooth` | Random multiplier | Each block × U(0.3, 2.0) |
| `split_chiplet_a_hot` | ChipA hot | chipA* = base; all others = 0.1 × base |
| `split_chiplet_b_hot` | ChipB/HBM hot | chipB* or hbm* = base; chipA = 0.1 × base |

RDL blocks always receive `rdl_joule_fraction × base_power` regardless of pattern.

**Operating regime (revised 2026-08-01).** Power is no longer an absolute W/cm² sweep. A
pattern's `base_power` now selects a *workload fraction* of the package TDP budget
(`TDP_BY_GEOMETRY_W`: 30 W mobile 3D stack → 700 W six-HBM accelerator), of which the
modelled blocks receive `CORE_FRACTION_OF_TDP` = 0.65 — the remainder representing cache,
IO and uncore, which are not modelled as separate sources. Two absolute ceilings then
apply: `MAX_LOGIC_POWER_DENSITY_WCM2` = 300 W/cm² (silicon limit) and an 8.0 W/cm² cap on
HBM/memory dies (`hbm*`, `chipB_d1*`, `chipB_d2*`), which genuinely run an order of
magnitude below logic. Finally, cooling is required to be adequate for the resulting power
density (`HTC_PER_WCM2` = 165 W/m²·K per W/cm²), because a 300 W/cm² hotspot cannot be
air-cooled.

The old regime (0.1–20 W/cm² absolute, HTC 1000–10000, ambient 25–65 °C) produced a
spatially degenerate dataset — median within-scenario ΔT of 1.10 K against a 68 K
between-scenario range — which closed-form ridge regression reconstructed at spatial
R² = 0.999. The revised regime raises the median spatial ΔT to 10.77 K. See
[`docs/report.md`](report.md) §9.3 and [`docs/assumptions.md`](assumptions.md) §3.4.

### Test Scenarios (identical across all geometries)

| # | Pattern | Peak power | HTC | T_amb | Purpose |
|---|---|---|---|---|---|
| `_test_001` | uniform | 1.5 W/cm² | 7500 | 35 °C | Interpolation baseline |
| `_test_002` | hotspot | 3.0 W/cm² | 3000 | 55 °C | Hotspot + poor cooling |
| `_test_003` | checkerboard | 1.5 W/cm² | 7500 | 35 °C | Pattern interpolation |
| `_test_004` | uniform | 0.3 W/cm² | 3000 | 35 °C | Low power interpolation |
| `_test_005` | gradient | 3.0 W/cm² | 7500 | 55 °C | Gradient + high temp |

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total files | 320 (280 train + 40 test, 8 geometries) |
| Points per file — g1/g3 | 60,000 (6 layers × 100 × 100) |
| Points per file — g2a/2b/2c | 64,000 (10 layers × 80 × 80) |
| Points per file — g4 | 33,600 (6 layers × 100 × 56) |
| Points per file — g5 | 61,600 (11 layers × 100 × 56) |
| Points per file — g6 | 103,488 (11 layers × 56 × 168) |
| Validation status | All files real 3D-ICE data; no synthetic fallback |

---

## NPZ File Format

```python
data = np.load('geometry5_train_001.npz', allow_pickle=True)
data['coords']    # (N, 3) float32 — (x, y, z) in µm
data['temp']      # (N,)   float32 — temperature in Kelvin
data['power']     # (N,)   float32 — volumetric power density in W/m³
data['layer']     # (N,)   int32   — layer index (0 = heat sink, top = die)
data['metadata']  # (1,)   object  — dict with scenario parameters + geometry info
```

Key metadata fields: `scenario_name`, `geometry`, `htc`, `t_ambient_celsius`,
`pattern`, `mesh_resolution`, `tsv_density`, `rdl_joule_fraction`,
`layer_k_overrides`, `layer_N_name/k/thickness_um/z_bottom_um/z_top_um`.

---

## Data Generation Commands

```bash
# Geometry 2a/2b/2c (hybrid bonding updated):
python src/main.py --simulator 3d-ice --geometry geometry2a --extra-train 10 \
  --ice-executable "wsl /home/rajul/3d-ice/bin/3D-ICE-Emulator" --output data/3d-ice

# Geometry 5 (Tier 0+1 + hybrid bond + TIM sweep + RDL fraction, 55 scenarios):
python src/main.py --simulator 3d-ice --geometry geometry5 --extra-train 35 \
  --ice-executable "wsl /home/rajul/3d-ice/bin/3D-ICE-Emulator" --output data/3d-ice

# Geometry 6 (6 HBM stacks, 42x14mm, 55 scenarios):
python src/main.py --simulator 3d-ice --geometry geometry6 --extra-train 35 \
  --ice-executable "wsl /home/rajul/3d-ice/bin/3D-ICE-Emulator" --output data/3d-ice
```

Skip-if-exists logic is active: any interrupted run can be safely restarted with the
same command — completed files are skipped automatically.
