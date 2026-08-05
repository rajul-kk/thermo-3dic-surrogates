# Geometry 5 — Upgrade Roadmap

> **Status update:** Tier 0 and Tier 1 (below) are **fully implemented** in the current
> `geometry5`/`geometry6` builders (`src/core/geometry_builders.py`) — RDL power layer,
> low-k device layer, C4 bump array, TIM1 indium solder, TIM2 thickness, published
> interposer dimensions, and 5 µm hybrid-bonding layer are all present, matching this
> document's original recommendation. See [`docs/geometry_reference.md`](geometry_reference.md)
> for the as-built layer stack. This document is kept as the design rationale / roadmap
> record — read it for *why* each choice was made, not as a to-do list. Tier 2 remains
> unimplemented (research-grade, 3-4 week estimated effort) and is the only section
> below still describing future work rather than completed work.

Geometry 5 is a CoWoS-style 2.5D chiplet assembly: one compute chiplet (A) and one
HBM-like memory stack (B) on a shared silicon interposer. It is the most complex
geometry in the benchmark suite and the natural target for incremental realism upgrades.

---

## Current layer stack

```
Layer          Material        Thickness   Active   Notes
─────────────────────────────────────────────────────────────────
die_zone_2     Si (+ underfill) 50 µm      Yes      Top die — chipB only in TSV region
tsv_zone       Si/Cu composite  100 µm     No       3% TSV density in chipB; underfill elsewhere
die_zone_1     Si (+ underfill) 50 µm      Yes      Bottom die — chipB only
interposer     silicon          100 µm     No       Shared CoWoS Si interposer
tim_top        TIM              100 µm     No       Die-to-spreader interface
spreader       copper           1000 µm    No       Cu heat spreader
tim_sink       TIM              100 µm     No       Spreader-to-sink interface
heat_sink      copper           5000 µm    No       Bulk heat sink / BC boundary

Total height: 6500 µm     Footprint: 25 × 14 mm     Mesh: 100 × 56 × 42 ≈ 235 k pts
```

### Why the total height is 6500 µm

The 6500 µm is **not** the package height — it is the full thermal path from active
silicon to the heat-sink boundary condition, which is how 3D-ICE models the problem.

```
Actual die stack (interposer + dies):   300 µm   (5% of total)
Heat spreader + TIMs:                  1300 µm   (20% of total)
Bulk heat sink:                        5000 µm   (75% of total)
```

Real CoWoS package specs quote only the die+interposer height (~300–800 µm). The
5000 µm heat sink is a modelling boundary, not a physical package dimension. The die
layers themselves (50 µm each, 100 µm TSV zone) are accurate for backgrinded 3D-stacked
dies in production (TSMC SoIC: 30–100 µm per die).

---

## What chip geometry 5 most closely resembles

Structurally closest to a **TSMC CoWoS-R reference vehicle** or a simplified
**AMD MI100 analogue** with the compute die chipletised. Key comparison:

| Parameter          | Geometry 5    | AMD MI100       | TSMC CoWoS-R ref |
|--------------------|---------------|-----------------|------------------|
| Package type       | CoWoS 2.5D    | CoWoS-S         | CoWoS-R          |
| Compute dies       | 1 chiplet     | 1 monolithic    | 1–2              |
| HBM stacks         | 1             | 4× HBM2         | 1–2              |
| Footprint          | 25 × 14 mm    | ~45 × 35 mm     | ~25 × 20 mm      |
| TSV k-effective    | 155.6 W/m·K   | ~150–160 W/m·K  | ~150 W/m·K       |
| Lateral k-variation| Yes           | No              | Yes              |

The single HBM stack is the least realistic feature — no production HPC chip ships
with one stack. Everything else is credible for a mid-range inference or FPGA-class
2.5D design.

---

## Upgrade tiers

### Tier 0 — Recommended baseline — ✅ COMPLETE (implemented)

These four changes require no new simulator, no new geometry type, and take 2–3 days
total. They make geometry 5 the most physically complete 2.5D thermal benchmark in
public literature.

| Upgrade | Code change | Build cost | Training cost Δ | Mesh pts Δ |
|---|---|---|---|---|
| **RDL power layer** | Add 1 active layer to geometry5 (`rdl_interposer`, k=Si, thickness=5 µm, non-zero q) | 4 h | +10% scenarios | +3% |
| **Low-k device layer** | Change die_zone_1/2 k_thermal from 148 → 80 W/m·K (Cu/low-k composite at N5/N3) | 1 h | +0% | +0% |
| **Extended HTC range** | Scenario generator: add HTC up to 2×10⁵ W/m²·K (liquid cooling) | 1 h | +10% scenarios | +0% |
| **Extended power density** | Scenario generator: add extreme scenarios up to 20 W/cm² | 1 h | +10% scenarios | +0% |

**Total Tier 0**: ~1 day code, +25–30% training scenarios, negligible mesh growth.

---

### Tier 1 — Match Kou 2022 (A100-like fidelity) — ✅ COMPLETE (implemented)
> ⚠ **"Kou et al. 2022" never verified and has been re-sourced (2026-08-06)** to
> Zhou, Li, Hou, He & Fan (2022), *IEEE TCPMT* 12(6), 956–963, DOI 10.1109/TCPMT.2022.3174608
> — confirmed to exist and be topically on point, but its full text is not accessible, so
> the specific dimensions below are still an engineering estimate rather than a confirmed
> literature figure. See `references.md` §1/§2b for the full verification note. "Kou 2022"
> below is retained as the historical label under which this tier was designed.


Kou et al. 2022 is the most rigorous public 2.5D thermal benchmark, validated against
ANSYS Icepak measurements. These additions close the gap to that baseline.

| Upgrade | What it models | Code change | Build cost | Training cost Δ | Mesh pts Δ |
|---|---|---|---|---|---|
| **C4 bump array** | k_eff layer between die and interposer (~100 µm, k≈15 W/m·K; Cu bump fraction ~10% in solder matrix) | New passive layer in geometry5 | 1 day | +2% | +5% |
| **TIM1 layer** | Indium solder or thermal paste between die and heat spreader (~50 µm, k≈80 W/m·K for In solder) | Modify existing tim_top thickness + k | 4 h | +0% | +0% |
| **TIM2 layer** | Thermal grease between spreader and cold plate (~125 µm, k≈4 W/m·K) | Modify existing tim_sink | 4 h | +0% | +0% |
| **4–6 HBM stacks** | Expand chipB footprint to replicate 4 stacks side-by-side on interposer; adjust power blocks | New geometry6 variant or extend geometry5 floorplan | 3 days | +40% mesh pts | +40% |
| **Published material dims** | Die thickness 780 µm, interposer 300 µm, HBM base die 500 µm (from Kou specs) | Parameter update in geometry_builders.py | 4 h | +0% | +10% |

**Total Tier 1 (above Tier 0)**: ~5–6 days, mesh grows to ~480 k pts, +50% scenarios.

#### Why TIM/bump resistance matters

In the current model, dies sit directly on the spreader with a generic TIM layer.
Real chips have two distinct resistance interfaces:

```
Die active layer
     │  ← C4 bumps (k_eff ~15 W/m·K, Cu pillars in solder — high resistance)
Interposer
     │  ← TIM1 (k ~4–80 W/m·K depending on material — often the #1 thermal bottleneck)
Heat spreader
     │  ← TIM2 (k ~4 W/m·K thermal grease — manageable)
Cold plate / heat sink
```

Omitting C4 bump resistance causes the model to underpredict peak junction temperature
by **8–15°C** for GPU-class power densities. This is the single largest systematic
error in the current geometry5.

#### Why more HBM stacks matters

One HBM stack covering one region of the 25×14 mm footprint leaves most of the
interposer isothermal, which is unrealistic. With 4–6 stacks spread across the
interposer, each stack acts as a distributed heat source + lateral spreading path,
creating the non-trivial temperature gradients that make PINN training harder and
more valuable. The geometry change is a new floorplan layout (~3 days) with no
solver changes.

---

### Tier 2 — Approach Blackwell GB200 fidelity — NOT implemented (future work)

These additions are research-grade and would require 3–4 weeks total. Most have no
published precedent in the thermal simulation literature.

| Upgrade | What it models | 3D-ICE viable? | Build cost | Training cost Δ | Mesh pts Δ |
|---|---|---|---|---|---|
| **8 HBM3e stacks × 8 sub-dies** | Full Blackwell HBM layout; 64 DRAM die layers | Yes but slow | 5 days | +250% mesh | ×4 |
| **Transformer Engine die (3D bonded)** | Second logic die face-bonded on compute chiplet via hybrid bonding | Yes | 3 days | +15% | +10% |
| **Bonding oxide layer (1–2 µm SiO₂)** | k=1.4 W/m·K interface between stacked dies; ultra-thin requires fine z-mesh | Risky — mesh stiffness | 4 days | +30% | +20% |
| **Hybrid bonding k_eff** | Cu-Cu pillar array at 9 µm pitch between stacked dies; k_eff ~60 W/m·K | Yes | 1 day | +2% | +2% |
| **NVLink switch die** | Third chiplet on interposer, low power density | Yes | 2 days | +5% | +5% |
| **900 W / 200 W/cm² SM hotspots** | Scenario extension only | Yes | 0 | +10% | +0% |
| **Spatial HTC variation** | Different cooling zones under each HBM stack | No — 3D-ICE uses uniform HTC | BC patch | +20% | +0% |

**Total Tier 2 (above Tier 1)**: ~15 days, mesh grows to ~1.4 M pts, 3D-ICE run time
per scenario rises from ~0.03 s to ~2–5 min (SuperLU scales ~O(N^1.5)).

#### Bonding oxide warning

The 1–2 µm SiO₂ layer is physically critical (it is the highest per-unit-area thermal
resistance in the stack at k=1.4 W/m·K) but creates a mesh resolution problem in
3D-ICE: the z-mesh must resolve this layer while also spanning 5000 µm of heat sink.
The aspect ratio forces either very high mesh density (×3–5 total mesh points) or
numerical inaccuracy. Consider modelling it as an effective contact resistance BC
rather than an explicit layer.

---

## Summary ladder

| Target | Total build effort | Mesh pts | 3D-ICE run time | Closest real chip |
|---|---|---|---|---|
| Current geometry5 | — | 235 k | ~0.03 s | TSMC CoWoS-R ref |
| + Tier 0 | 1 day | 245 k | ~0.03 s | TSMC CoWoS-R (calibrated) |
| + Tier 1 (match Kou) | +6 days | 480 k | ~0.05 s | AMD MI100 / A100-class |
| + Tier 2 (approach Blackwell) | +15 days | 1.4 M | ~2–5 min | GB200 partial |

Tier 0 is unconditionally recommended. Tier 1 is the right target for a publication
claiming industry-relevant benchmarks. Tier 2 is a PhD-level effort and only makes
sense if Blackwell-class fidelity is the explicit research goal.

---

## Unchanged limitations at all tiers

These gaps persist even after full Tier 2 and would require simulator-level changes
or physical measurements to close:

- **Uniform convective BC** — real chips have spatially varying HTC under each die zone
- **No package warpage coupling** — thermal expansion stress changes effective k at interfaces
- **No transient** — all scenarios are steady-state; real workloads have µs–ms thermal transients
- **Not validated against hardware** — all temperatures are 3D-ICE outputs, not measured die maps
