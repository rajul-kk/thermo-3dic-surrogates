# 3D-ICE Thermal PINN Benchmark System - Full Project Analysis

## Context

This analysis covers the complete state of the 3D-ICE Thermal PINN Benchmark System at `d:\Work\3D-ICE Thermal-modelling\Thermo`. The project is a data generation pipeline that produces Physics-Informed Neural Network (PINN) training datasets from 3D integrated circuit thermal simulations. Five development phases are reported as complete, but the actual benchmark data has not yet been generated, and several runtime bugs prevent the CLI from working correctly beyond synthetic (mock) mode.

---

## What This Project Does

**In one sentence**: Generate 3D thermal simulation datasets to train a neural network surrogate that predicts chip temperature fields 100x faster than any simulator.

**The chain**:
1. Define 3D IC package geometries (silicon dies, copper spreaders, TIM layers, TSV interconnects)
2. Generate 80 parameterized thermal scenarios (power maps, cooling conditions, ambient temperatures)
3. Run thermal simulations (3D-ICE compact solver) to get T(x,y,z) ground truth
4. Export (coords, temperature, power, layer) arrays as `.npz` files
5. Train a PINN: `f(x,y,z) -> T` with the heat equation embedded in the loss function
6. Inference: PINN predicts full 3D temperature fields in <1 ms vs 20 seconds for 3D-ICE

**Geometries - are they 2D or 3D?** All four geometries produce fully **3D coordinate arrays** (x, y, z). "Geometry 1" is called "2D cross-sectional" because it has one silicon die (no multi-die stacking), but its mesh is 100x100x40 = 400,000 3D points. Geometries 2a/2b/2c are genuinely 3D multi-die stacks with vertical TSV interconnects through 9 material layers.

---

## System Architecture

```
main.py (CLI orchestration)
    +-- src/core/           -> geometry, material, mesh (infrastructure)
    +-- src/scenario/       -> 80-scenario parameter sweep generation
    +-- src/simulators/     -> 3D-ICE wrapper + abstract base
    +-- src/export/         -> .npz export + thermal statistics
    +-- src/visualization/  -> matplotlib plots
```

**Output**: 80 compressed `.npz` files (~600-800 MB total) + JSON stats + PNG plots

---

## HIGH-LEVEL SYSTEM FEATURES

| Feature | Status | Notes |
|---|---|---|
| Multi-geometry IC stack definitions | Complete | 4 geometries (1 x 2D, 3 x 3D-TSV) |
| Temperature-dependent material k(T) | Complete | Silicon k(T) = 148x(300/T)^1.3 |
| Adaptive mesh generation | Complete | Z-density proportional to layer thickness |
| 80-scenario parameter space | Complete | 4 geom x (15 train + 5 test) |
| Synthetic (mock) thermal solver | Complete | Gaussian + gradient model |
| 3D-ICE simulator wrapper | Class exists | NOT wired into main.py (bug) |
| HotSpot simulator wrapper | Missing | CLI option exists, no implementation |
| NPZ export for PINN training | Complete | coords/temp/power/layer/metadata |
| Thermal statistics (25+ metrics) | Complete | Hotspot, gradients, layer-wise |
| 2D visualization (z-slice, power map) | Complete | x/y profiles raise NotImplementedError |
| CLI orchestration | Partial | 3d-ice and hotspot modes broken |
| Benchmark dataset (80 .npz files) | Not generated | Awaiting pipeline execution |
| PINN/FNO training code | Not started | Out of scope for this pipeline |

---

## UNIT-LEVEL FEATURES

### `src/core/geometry.py` - Layer / PowerBlock / Geometry dataclasses
- Stack assembly with z-coordinate assignment
- Spatial lookup: `get_layer_at_z()`, `get_power_block_at_xy()`
- TSV layer and multi-die support
- **Status**: Complete, no known issues

### `src/core/geometry_builders.py` - 4 benchmark geometries
- `build_geometry1()` - 2D cross-section, 5 layers, 100x100x40 mesh
- `build_geometry2a/2b/2c()` - 3D stacks, 9 layers, 3%/5%/10% TSV, 80x80x50 mesh
- **Status**: Complete, no known issues

### `src/core/material.py` - Material property database
- 11 materials: Si, Cu, Al, TIM variants, TSV-enhanced Si, bonding
- TSV effective k via parallel resistor model
- **Status**: Complete, no known issues

### `src/core/mesh.py` - Cartesian mesh generation
- Adaptive z-spacing, layer index labeling, power density field generation
- RegularGridInterpolator for field remapping
- **Status**: Complete, no known issues

### `src/scenario/generator.py` - 80-scenario suite
- 6 power patterns: uniform, hotspot, checkerboard, gradient, dual_hotspot, extreme_hotspot
- Tier 1 extreme scenarios: 10 W/cm2 peak, HTC=500, T_ambient=85C
- YAML export/import
- **Status**: Complete, no known issues

### `src/simulators/base_simulator.py` - Abstract simulator interface
- ABC with `generate_config_files()`, `run_simulation()`, `parse_results()`
- **Status**: Complete, no known issues

### `src/simulators/ice_simulator.py` - 3D-ICE wrapper
- Generates `.stk` and `.flp` config files
- Calls `3D-ICE-Emulator` subprocess, parses text grid output
- `_generate_power_trace()` is a placeholder stub (transient only)
- **Status**: Class is complete; NOT instantiated in main.py (see bugs)

### `src/export/npz_exporter.py` - PINN data format
- Exports (coords, temp, power, layer, metadata) in compressed `.npz`
- Batch export + dataset summary functions
- **Status**: Complete, tested with real output in `test_output/`

### `src/export/statistics.py` - Thermal analysis
- Temperature stats, hotspot location, layer-wise breakdown, gradient analysis
- JSON export per scenario
- **Status**: Complete, tested

### `src/visualization/visualization.py`
- `plot_temperature_field_2d()` - z-slice heatmaps: working
- `plot_power_map_2d()` - die-level power distribution: working
- `plot_temperature_profile()` - x/y directions raise `NotImplementedError`: broken
- **Status**: Partially complete

### `src/utils/__init__.py` - Utility module
- **Empty placeholder** - no implementations
- **Status**: Stub only

---

## DELIVERABLES STATUS

| Deliverable | Location | Status |
|---|---|---|
| Core geometry infrastructure | `src/core/` | Done |
| 4 benchmark geometry definitions | `src/core/geometry_builders.py` | Done |
| 80-scenario YAML configs | `configs/scenarios/*.yaml` | Done |
| 3D-ICE simulator wrapper | `src/simulators/ice_simulator.py` | Class done, not wired |
| NPZ export pipeline | `src/export/npz_exporter.py` | Done |
| Thermal statistics | `src/export/statistics.py` | Done |
| CLI orchestration | `src/main.py` | Mock works; real simulators broken |
| Benchmark dataset (80 .npz files) | `data/` | Not yet generated |
| Installation guide | `docs/installation.md` | Done |
| README + literature references | `README.md`, `resources.txt` | Done |
| Test suite | `test_*.py` | 3 test files; status unverified |
| PINN/FNO training code | N/A | Not in scope yet |

---

## BUGS - BLOCKING CORRECT RUNNING

### BUG 1: 3D-ICE mode never wired into main.py (Critical)
**File**: `src/main.py`
**Problem**: When `--simulator 3d-ice` is passed, main.py initializes `simulator = None` and always falls back to `generate_synthetic_temperature()`. The `ICESimulator` class is never instantiated.
**Fix needed**: In `main()`, branch on `args.simulator` to create an `ICESimulator(config_dir, output_dir, executable)` instance, and in `process_scenario()`, call `simulator.simulate()` instead of the synthetic fallback when simulator is not None.

### BUG 2: HotSpot simulator option crashes (Critical)
**File**: `src/main.py`
**Problem**: `--simulator hotspot` is a valid CLI choice, but no HotSpot wrapper class exists.
**Fix needed**: Remove `hotspot` from CLI choices until the wrapper is implemented (Phase 6).

### BUG 3: Visualization x/y profile crash (Minor)
**File**: `src/visualization/visualization.py`
**Problem**: `plot_temperature_profile(direction='x')` and `direction='y'` raise `NotImplementedError`.
**Fix needed**: Add `if direction != 'z': return` guard.

---

## FEASIBILITY STUDY

### TIER A - LEAST CONCERNING (solid, tested, no blockers)

| Component | Why it's fine |
|---|---|
| Core geometry/material/mesh | Validated properties; adaptive mesh proven in test output |
| Scenario generation | 80 scenarios YAML-exported; deterministic; Phase 5 tests passed |
| NPZ export | Real output file exists: `test_output/geometry1_train_001_test.npz` |
| Statistics calculator | 25+ metrics tested; JSON output verified |
| Python dependency stack | NumPy, SciPy, pandas, matplotlib, PyYAML all installed, above minimum |
| CLI argument parsing | All 8 arguments verified in Phase 5 test suite |
| Mock (synthetic) simulator | Runs today in 3-8s per scenario; no external dependencies |

**Bottom line**: The full pipeline runs today with `pip install seaborn pytest` and one command - in mock mode.

### TIER B - MODERATELY CONCERNING (solvable, clear fixes)

| Component | Concern | Mitigation |
|---|---|---|
| ICESimulator wiring | Dead code path - never called | ~20-line fix in main.py |
| Visualization x/y profiles | NotImplementedError crash | 2-line guard fix |
| seaborn / pytest missing | Two packages absent | `pip install seaborn pytest pytest-cov` |
| HotSpot wrapper | Zero implementation behind CLI option | Mirror ICESimulator structure |
| Synthetic data accuracy | Mock temperatures are physically rough; wrong for PINN training | Replace with 3D-ICE before Phase 7 |

### TIER C - MOST CONCERNING (real unknowns, risk of delay)

| Component | Why it's a real concern |
|---|---|
| 3D-ICE on Windows | Linux-native C codebase; requires WSL2 + Ubuntu + GCC. Build failures (flex/bison, library paths) block the real simulator entirely. |
| 3D-ICE subprocess parsing | `ICESimulator.parse_results()` reads floating-point text grids. Coordinate ordering, whitespace, and locale decimal separators can cause silent misreads. Path never end-to-end tested. |
| PINN compute cost | ARCHITECTURE_PLAN states 2-4 hours per geometry on i7 CPU. 400k points x 15 scenarios x 5000 epochs is heavy. GPU strongly recommended for FNO (1-2 days CPU otherwise). |
| HotSpot TSV approximation | No native TSV support; effective-k homogeneous layer workaround has unknown error. Could be 20-30% temperature error in die layers for geometry2. |
| FNO data sufficiency | 60 training scenarios is below the 200-500 recommended. FNO requires Phase 8 expansion. |
| COMSOL automation | LiveLink for Python requires licensed install + setup. Practical only for 5-10 cases; more becomes a bottleneck. |

**Overall verdict**: The data pipeline is 90% done and risks are localized. The PINN phase is where risk concentrates. Start PINN on geometry1 alone (CPU-feasible) before scaling.

---

## SIMULATOR STRATEGY

### Tool Comparison

| Attribute | 3D-ICE | HotSpot 7.0 | COMSOL |
|---|---|---|---|
| License | Free, open-source | Free, open-source | Commercial |
| Platform | Linux/macOS (WSL on Windows) | Linux/macOS (WSL on Windows) | Windows/Linux/macOS native |
| Runtime per scenario | 5-30 s | 1-10 s | Minutes to hours |
| Fidelity vs FEM | ~5-15% error | ~10-20% error | Ground truth |
| TSV support | Native | None (workaround only) | Full 3D geometry |
| 3D multi-die stacking | Native | Limited | Full |
| Batch/scripting fit | Excellent | Excellent | Complex (Java/MATLAB/Python API) |
| PINN data generation | Excellent | Good (2D only) | Poor (too slow) |
| Transient support | Yes | Yes | Yes |
| Radiation BC | No | No | Yes |
| Windows native | No (needs WSL) | No (needs WSL) | Yes |

### Recommended: 3D-ICE (primary) + HotSpot 7.0 (cross-validation) + COMSOL (spot-checks)

**3D-ICE** generates all 80 training scenarios. Fast, free, TSV-native, already wrapped.

**HotSpot 7.0** cross-validates geometry1 (2D case) independently. Same floorplan, different compact model. Difference between tools reveals systematic error.

**COMSOL** provides 5-8 high-fidelity FEM reference cases for publication-grade accuracy bounds. Not used for training data. Specific reference scenarios:
1. Geometry1 scenario 14 - extreme hotspot (20 W/cm2), highest gradient in 2D case
2. Geometry1 scenario 12 - HTC=500, poor cooling, worst-case thermal resistance
3. Geometry2a scenario 11 - high power at 3% TSV density
4. Geometry2c scenario 11 - high power at 10% TSV density
5. Geometry2b scenario 13 - high ambient (85C), mid-TSV case

---

## IMPLEMENTATION PLAN - PHASES 6 THROUGH 9

**Where we are**: Phases 1-5 produced the complete data generation pipeline. Code is built and tested in mock mode. No real simulation data exists. Two simulator integrations are broken. Ultimate goal: trained, validated PINN surrogate predicting T(x,y,z) across 3D-IC geometries faster than any simulator.

**How many phases ahead**: 4 phases (6-9). Phases 6-8 are the core development track. Phase 9 is the research frontier (FNO + publication), contingent on Phase 7 succeeding.

```
Phase 6          Phase 7          Phase 8          Phase 9
Pipeline    -->  PINN         --> Multi-tool    --> FNO +
Hardening        Training         Validation        Publication
& Real           & Eval           + COMSOL          (advanced)
Dataset          geometry1        Reference
                 -> all geom
```

| Phase | Name | Duration | Key Deliverable | Blocked if skipped |
|---|---|---|---|---|
| 6 | Pipeline Hardening | 3 weeks | 80 real .npz from 3D-ICE | Phase 7 cannot start |
| 7 | PINN Training | 5 weeks | 4 trained PINN models, MAE < 3 K | Phase 9 cannot start |
| 8 | Multi-Tool Validation | 3 weeks | COMSOL accuracy bounds, 200+ scenarios | Publication lacks rigour |
| 9 | FNO + Publication | 8 weeks | FNO operator, cross-geom benchmarks, paper | Optional research phase |

**Critical path**: P6-T1 (bug fix) -> P6-T5 (WSL + 3D-ICE) -> P6-T7 (full dataset) -> P7-T3/T4 (PINN arch + loss) -> P7-T6 (geometry1 PINN) -> P7-T7 (all geometries) -> P8-T1 (COMSOL) -> P8-T4 (200+ scenarios) -> P9.

---

## PHASE 6 - Pipeline Hardening & Real Dataset Generation
**Duration**: ~3 weeks | **Deliverable**: 80 real `.npz` files from 3D-ICE; HotSpot cross-check on geometry1; COMSOL reference cases started

### Objective
Fix the broken simulator paths, prove the end-to-end pipeline works with real physics, and produce the training dataset Phase 7 depends on. Phase 7 cannot start without this.

### Tasks

**P6-T1 - Fix pipeline bugs** (Day 1-2, unblocks all downstream work)

| Task | File | Change |
|---|---|---|
| Wire ICESimulator into main.py: branch on `args.simulator`, instantiate `ICESimulator`, call `simulator.simulate()` in `process_scenario()` | `src/main.py` | ~20 lines |
| Fix visualization NotImplementedError: add `if direction != 'z': return` guard | `src/visualization/visualization.py` | 2 lines |
| Remove `hotspot` from `--simulator` choices; add NotImplementedError if selected | `src/main.py` | 3 lines |
| Add `--ice-executable` CLI argument for WSL binary path | `src/main.py` | 5 lines |

**P6-T2 - Install Python packages** (Day 1)
```bash
pip install seaborn pytest pytest-cov mph
```
`mph` is the COMSOL LiveLink Python client - install now for P6-T10.

**P6-T3 - Run test suite baseline** (Day 2)
```bash
pytest test_geometries.py test_scenarios.py test_phase4.py -v --tb=short
```
Record which tests pass. Fix any failures before continuing. This is the regression baseline.

**P6-T4 - Generate and validate mock dataset** (Day 3)
```bash
python src/main.py --all-geometries --simulator mock --output data/mock --verbose
```
Verify 80 `.npz` files, correct shapes, temperature range 298-500 K, power fields within 5% of scenario budget.
Write `scripts/validate_dataset.py` to automate these checks.

**P6-T5 - Install 3D-ICE via WSL2** (Days 4-6)
- Enable WSL2 + install Ubuntu 22.04 from Microsoft Store
- Inside WSL: `sudo apt install gcc make bison flex libsuperlu-dev`
- Clone and compile: `git clone https://github.com/esl-epfl/3d-ice && cd 3d-ice && make`
- Verify: `./bin/3D-ICE-Emulator` prints version string

**P6-T6 - End-to-end 3D-ICE smoke test on geometry1** (Day 7)
```bash
python src/main.py --geometry geometry1 --simulator 3d-ice \
  --ice-executable "wsl /home/user/3d-ice/bin/3D-ICE-Emulator" \
  --output data/ice_test --verbose
```
Check `ICESimulator.parse_results()` carefully - text grid parsing is the highest-risk step.

*Validation gate V6-A*: mock vs 3D-ICE on geometry1 test scenarios: peak T within +/-40 K; same die layer as hotspot; R_th within factor of 2.

**P6-T7 - Generate full 80-scenario 3D-ICE dataset** (Days 8-10)
```bash
python src/main.py --all-geometries --simulator 3d-ice \
  --ice-executable "wsl ..." --output data/3d-ice --verbose
```
Expected runtime: 20-40 minutes. Primary training dataset for Phase 7.

*Validation gate V6-B*: all 80 files present; no NaN/Inf; TSV effect visible (geometry2c peak T < geometry2a at same scenario); extreme scenarios below 600 K.

**P6-T8 - Install HotSpot 7.0 via WSL2** (Days 6-7, parallel with P6-T5)
Download HotSpot 7.0 from UVA repository; compile; verify `./hotspot` runs on a test floorplan.

**P6-T9 - Implement HotSpot simulator wrapper** (Days 8-11)
Create `src/simulators/hotspot_simulator.py` subclassing `ThermalSimulator`:
- `generate_config_files()`: write `.flp` floorplan and HotSpot config matching geometry1 layer stack
- `run_simulation()`: call `hotspot` subprocess
- `parse_results()`: parse HotSpot grid temperature output to (N,3) coords + (N,) temps
- Scope: geometry1 only, steady-state only
- Add `hotspot` back to `--simulator` CLI choices; wire instantiation in `main()`

*Validation gate V6-C*: 3D-ICE vs HotSpot on all 20 geometry1 scenarios: peak T within 10%; same hotspot block identified; Si die layer T within 5%.

**P6-T10 - COMSOL geometry1 reference model** (Days 12-15, conditional on license)
If COMSOL is available:
- Build geometry1 in COMSOL: 5-layer stack, 10x10 mm floorplan, convective BC at top
- Material properties from `src/core/material.py`: Si k=148, Cu k=400, TIM k=4 W/m.K
- Run 5 reference scenarios (scenarios 11, 12, and 3 test scenarios)
- Export temperature field as CSV (x,y,z,T); convert to `.npz` with `fidelity='fem'` metadata

*Validation gate V6-D*: 3D-ICE vs COMSOL: peak die T within +/-10%; TIM gradient within +/-20%. If error exceeds 15%, document bias and consider correction factor before PINN training.

### Phase 6 Exit Criteria
- [ ] 80 `.npz` files from 3D-ICE in `data/3d-ice/`
- [ ] `scripts/validate_dataset.py` passes on all 80 files
- [ ] HotSpot geometry1 cross-check within 10% of 3D-ICE
- [ ] COMSOL reference cases completed or formally deferred
- [ ] Test suite still passes (no regressions from bug fixes)

---

## PHASE 7 - PINN Implementation & Training
**Duration**: ~5 weeks | **Deliverable**: 4 trained PINN models (one per geometry), evaluated on 20 held-out test scenarios

### Objective
Implement, train, and evaluate a Physics-Informed Neural Network predicting T(x,y,z). Start with geometry1 to prove the approach, then scale. This is the primary AI deliverable of the project.

### Tasks

**P7-T1 - Install ML stack** (Day 1)
```bash
pip install torch deepxde tensorboard
# GPU (if available): pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**P7-T2 - Data loader for `.npz` files** (Days 2-3)
Create `src/pinn/data_loader.py`:
- `ThermalDataset(torch.utils.data.Dataset)`: loads `.npz`, returns (coords, temp, power, layer) tensors
- Normalisation: coords -> [0,1] per axis; temp -> zero-mean unit-variance per scenario
- Train/test split from `train/` and `test/` directories
- DataLoader with batch size 4096

**P7-T3 - PINN architecture** (Days 3-4)
Create `src/pinn/model.py`:
```
Input:  (x, y, z) - normalised coordinates
  -> Dense(256, tanh)
  -> Dense(256, tanh)
  -> Dense(256, tanh)
  -> Dense(256, tanh)
  -> Dense(1)
Output: T_hat(x,y,z) - normalised temperature
```
Use `tanh` activations (smooth second derivatives needed for PDE loss). ~260k parameters.

**P7-T4 - Loss function implementation** (Days 4-6)
Create `src/pinn/losses.py`:

```
L_data = MSE(T_pred, T_3DICE)                              [data fidelity]
L_pde  = MSE(div(k * grad(T)) + q, 0)                     [heat equation]
L_bc   = MSE(-k * dT/dz - HTC*(T - T_ambient), 0) at top  [convective BC]
L      = L_data + lambda_pde * L_pde + lambda_bc * L_bc
```

Start with lambda_pde=0.1, lambda_bc=1.0. Use `torch.autograd.grad` for spatial derivatives.
k is looked up per layer from `MaterialLibrary`; q is the power density array.

**P7-T5 - Training loop** (Days 6-8)
Create `src/pinn/trainer.py`:
- Optimizer: Adam, lr=1e-3, cosine annealing schedule
- Epochs: 5000 with early stopping (patience=500)
- Collocation points: 10,000 randomly sampled per epoch for PDE loss
- TensorBoard logging: total loss, L_data, L_pde, L_bc, val MAE
- Save best checkpoint: `checkpoints/geometry1_best.pt`

Expected runtime: 2-4 hours CPU (i7), 20-40 min GPU.

**P7-T6 - Train and evaluate on geometry1** (Days 8-12)
```bash
python src/pinn/train.py --geometry geometry1 \
  --data data/3d-ice/geometry1 --epochs 5000 --output checkpoints/
```
Evaluate on 5 held-out test scenarios:
- MAE on temperature: target < 2 K
- Max error at hotspot: target < 5 K
- Inference time for 400k points: target < 100 ms

*Validation gate V7-A*: If MAE > 5 K on geometry1, debug before scaling. Check L_data drops first in training; verify coordinate normalisation consistency; check k(T) per-layer lookup.

**P7-T7 - Scale to all 4 geometries** (Days 13-20)
Train separate PINN for geometry2a, 2b, 2c. Each gets its own checkpoint. Geometry2 PDE loss must use layer-specific k including TSV effective conductivity.

*Validation gate V7-B*: All 4 PINNs achieve MAE < 3 K on test scenarios.

**P7-T8 - Inference speed benchmark** (Day 21)
Compare for 400k-point prediction: 3D-ICE (15-30 s) vs PINN (target < 100 ms CPU, < 10 ms GPU). Target speedup: >100x.

**P7-T9 - Visualise PINN predictions** (Days 21-22)
Add `src/pinn/visualise.py`:
- Side-by-side z-slice: 3D-ICE vs PINN temperature field
- Error map: |T_PINN - T_3DICE| per point
- Z-axis profile: predicted vs ground truth with layer boundaries

### Phase 7 Exit Criteria
- [ ] 4 trained PINN checkpoints (one per geometry)
- [ ] All 4 geometries: MAE < 3 K on held-out test scenarios
- [ ] Inference > 100x faster than 3D-ICE on CPU
- [ ] TensorBoard training curves saved for all 4 runs
- [ ] Side-by-side comparison plots for 5 test scenarios per geometry

---

## PHASE 8 - Multi-Tool Validation & Data Hardening
**Duration**: ~3 weeks | **Deliverable**: COMSOL-validated accuracy bounds; multi-fidelity dataset; 200+ scenarios for FNO readiness

### Objective
Establish rigorous accuracy bounds on the full pipeline (3D-ICE -> PINN) using COMSOL as ground truth. Expand the dataset. This phase produces the evidence needed for publication.

### Tasks

**P8-T1 - COMSOL geometry2 reference models** (Days 1-8)
Build COMSOL models for geometry2a/2b/2c: 9-layer TSV stack; TSV region as effective-k block from `MaterialLibrary.compute_tsv_effective_k()`. Run 4 additional reference scenarios. Automate via `mph` library if license allows; manual export otherwise. Export as `.npz` with `fidelity='fem'` metadata.

**P8-T2 - 3D-ICE vs COMSOL accuracy report** (Days 8-10)
Create `scripts/compare_fidelity.py`:
- Interpolate both to common grid (COMSOL unstructured vs 3D-ICE Cartesian)
- Report: peak T error, RMS error, TIM gradient error, hotspot location error
- Publication-quality comparison figures

*Validation gate V8-A*: 3D-ICE within +/-10% of COMSOL peak die T for all 5-8 reference cases.

**P8-T3 - Multi-fidelity dataset construction** (Days 10-12)
- Add `fidelity='compact'` field to all 3D-ICE/HotSpot `.npz` metadata
- Combine into `data/multifidelity/`
- Update `scripts/validate_dataset.py` for both fidelity levels
- Document which scenarios have FEM ground truth

**P8-T4 - Expand scenario suite to 200+ for FNO readiness** (Days 12-18)
- Add Latin Hypercube Sampling (LHS) to `src/scenario/generator.py`: `--lhs-sampling N` flag
- Generate 50 LHS training scenarios per geometry (200 total) via 3D-ICE
- Additional storage: ~600 MB (total ~1.2 GB)

**P8-T5 - Radiation BC implementation** (Days 15-18, parallel)
In `src/simulators/ice_simulator.py`:
- Add linearized radiation: `h_effective = h_conv + 4*epsilon*sigma*T_avg^3`
- Add `emissivity` field to `ScenarioParameters` (default 0.8)
- Document magnitude: at T=400 K, h_rad ~14 W/m2.K (small vs HTC=5000 but measurable)

**P8-T6 - Dataset versioning** (Days 18-20)
- Add `dataset_version`, `generator_git_hash`, `simulator_version` to `.npz` metadata
- Tag git repo: `v1.0-dataset` after full 200-scenario generation
- Write `DATASET_README.md` documenting all fields, units, coordinate conventions

### Phase 8 Exit Criteria
- [ ] COMSOL reference cases for all 4 geometries completed
- [ ] 3D-ICE vs COMSOL accuracy report: < 10% error on peak T
- [ ] Multi-fidelity dataset in `data/multifidelity/`
- [ ] 200+ scenario dataset generated and validated
- [ ] Git tag `v1.0-dataset` applied

---

## PHASE 9 - FNO, Cross-Geometry Generalization & Publication
**Duration**: ~8 weeks | **Deliverable**: Trained FNO operator; cross-geometry benchmarks; preprint

> **Entry condition**: Phase 7 complete with all 4 PINNs validated; Phase 8 200+ scenario dataset available.

### Tasks

**P9-T1 - Install FNO library** (Day 1)
```bash
pip install neuraloperator einops
```

**P9-T2 - FNO architecture** (Days 2-6)
Create `src/fno/model.py`:
```
Input: (x, y, z, q(x,y,z), phi_tsv) -- coordinate + power map + TSV density
  -> Lifting layer: Dense(128)
  -> FNO Block 1: FFT -> spectral MLP -> IFFT + residual
  -> FNO Block 2, 3, 4 (same structure)
  -> Projection: Dense(1)
Output: T_hat(x,y,z)
```

**P9-T3 - Multi-geometry FNO training** (Days 7-20)
Train single FNO on all 200+ scenarios across all 4 geometries. Augment with intermediate TSV densities (2%, 7%, 12%) using interpolated effective-k. Expected: 1-2 days CPU, 4-8 hours GPU.

**P9-T4 - Cross-geometry generalization study** (Days 20-25)
- Train on geometry1 + geometry2a; evaluate zero-shot on geometry2b + geometry2c
- Quantify TSV density sensitivity: PINN/FNO error vs TSV density (3%, 5%, 10%)
- Compare PINN (geometry-specific) vs FNO (cross-geometry) on accuracy and data efficiency

**P9-T5 - Hybrid PINN-FNO correction** (Days 25-35, experimental)
```
T_final = T_FNO + PINN_correction(x, y, z, T_FNO)
```
Assess whether hybridisation improves accuracy over either model alone.

**P9-T6 - Publication preparation** (Days 30-56)
- Benchmark table: 3D-ICE / HotSpot / PINN / FNO / COMSOL (accuracy, speed, data cost)
- Cross-geometry generalization curves
- COMSOL as ground truth for error figures
- Target venue: Journal of Electronic Packaging, IEEE TCAS, or arXiv thermal ML benchmark

### Phase 9 Exit Criteria
- [ ] FNO trained and evaluated on 200+ scenario dataset
- [ ] Cross-geometry generalization: FNO on unseen geometry within 5% MAE
- [ ] Full benchmark comparison table completed
- [ ] Paper draft submitted

---

## PHASE 6.5 - Geometry & Data-Pipeline Bug Fixes (Pre-PINN)
**Duration**: ~2 days | **Do this before starting Phase 7 PINN training**

The geometry and mesh code has several bugs that would silently corrupt PINN training data. Fix these first.

### Bug Fixes (Critical, blocking PINN correctness)

**Fix 1 — Die2 y-coordinate copy-paste error** (`src/core/geometry_builders.py`)

In `build_geometry2a/2b/2c()`, the two power blocks on Die 2 both have `y=4500.0`. The lower block `core1_d2` should be `y=1000.0`.

```python
# WRONG (both blocks share y=4500):
core1_d2 = PowerBlock(name='core1_die2', x=500.0, y=4500.0, ...)
core2_d2 = PowerBlock(name='core2_die2', x=4500.0, y=4500.0, ...)

# CORRECT:
core1_d2 = PowerBlock(name='core1_die2', x=500.0, y=1000.0, ...)   # lower half
core2_d2 = PowerBlock(name='core2_die2', x=4500.0, y=4500.0, ...)  # upper half
```

**Fix 2 — TSV k_eff docstring / model mismatch** (`src/core/material.py`)

The docstring says "series resistors" but the code uses arithmetic mean (parallel). Arithmetic mean is the correct upper bound for TSV-enhanced Si. Fix the docstring, not the formula.

Also: remove the duplicate pre-computed TSV property constants (lines ~100-122) that diverge from `create_tsv_material()`. Always use `create_tsv_material(tsv_density)` as the single source of truth.

**Fix 3 — `get_layer_at_z()` excludes z_top** (`src/core/geometry.py`)

Condition is `z_bottom <= z < z_top`. Points at exact z_top (top surface) return None, breaking the convective BC. Fix to include the top boundary:

```python
# Last layer only — include its top surface
if i == len(self.layers) - 1:
    return layer if layer.z_bottom <= z <= layer.z_top else None
return layer if layer.z_bottom <= z < layer.z_top else None
```

**Fix 4 — Mesh `np.unique()` silently drops z-levels** (`src/core/mesh.py`)

`generate_adaptive_z_coords()` passes through `np.unique()` which collapses duplicate z-values at layer boundaries. If two thin layers happen to produce the same float coordinate, points are dropped silently. Fix: use `np.unique()` only at the very end, after all layers are concatenated, and assert the output count equals the requested `nz`.

**Fix 5 — Silent layer-0 assignment for boundary points** (`src/core/mesh.py`)

`max(0, layer_idx)` at the point-labeling step assigns z<0 points (numerical noise below die bottom) to layer 0 instead of raising an error. Fix: assert `layer_idx >= 0` and log a warning with the offending coordinate count; clamp only if count < 10 (clearly floating-point noise).

### Enhancement: Store Normalization Constants in NPZ

Before PINN training, `npz_exporter.py` should store the normalization stats it uses, so the PINN can denormalize predictions at inference time without out-of-band metadata files.

Add to `export_scenario()`:
```python
norm_stats = {
    'coords_min': coords.min(axis=0).tolist(),   # [x_min, y_min, z_min] µm
    'coords_max': coords.max(axis=0).tolist(),
    'temp_min': float(temperature_field.min()),   # K
    'temp_max': float(temperature_field.max()),
    'power_mean': float(power_density.mean()),    # W/m³
    'power_std':  float(power_density.std()),
}
metadata.update({f'norm_{k}': v for k, v in norm_stats.items()})
```

### Phase 6.5 Exit Criteria
- [ ] Die2 power blocks have distinct y-coordinates (verify with unit test)
- [ ] `material.py` has single TSV property source (`create_tsv_material()`)
- [ ] `get_layer_at_z()` returns correct layer for z == z_top
- [ ] Mesh z-level count assertion passes for all 4 geometries
- [ ] NPZ files contain `norm_*` metadata keys
- [ ] Re-generate geometry2a/2b/2c 3D-ICE dataset after Die2 fix (old data is wrong)

---

## PHASE 7 (REVISED) - PINN Architecture & Training
**Duration**: ~5 weeks | **Deliverable**: 4 trained PINN models, geometry1 MAE < 1.5 K, all geoms < 3 K

**No PINN code currently exists.** The entire `src/pinn/` module must be built from scratch. This section replaces the earlier Phase 7 sketch with a fully optimized design.

---

### 7.1 — Problem Formulation

**PDE**: Steady-state heat conduction with temperature-dependent conductivity:
```
∇·(k(T,x) ∇T) + Q(x) = 0   inside Ω
-k(T) ∂T/∂n = h(T - T_∞)   on Γ_top (convective cooling)
∂T/∂n = 0                   on Γ_sides, Γ_bottom (adiabatic walls)
```

**Input features** per point:
- `(x̂, ŷ, ẑ)` — normalised coordinates [0,1]
- `Q̂(x)` — normalised volumetric power density
- `layer_id` — integer 0..N_layers-1 (embedded, not one-hot)
- `htc_norm` — normalised heat transfer coefficient (scenario-level broadcast)
- `t_amb_norm` — normalised ambient temperature (scenario-level broadcast)
- `tsv_density` — scalar fraction 0/0.03/0.05/0.10 (geometry-level broadcast)

**Output**: `T̂(x)` — normalised temperature, denormalized via stored norm constants.

---

### 7.2 — Architecture: Multi-Scale Thermal PINN

**File**: `src/pinn/model.py`

#### Input Encoding (critical for accuracy)

Raw (x̂,ŷ,ẑ) ∈ [0,1] gives the network low-frequency bias. Fourier feature encoding removes this:

```python
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, n_freq=16, sigma=10.0):
        # B ~ N(0, sigma²), shape (3, n_freq)
        # output: [sin(2π B x), cos(2π B x)]  →  2*n_freq features per point
```

Use `sigma=10.0` for geometry1 (fine features ~1 mm in 10 mm domain). Increase to `sigma=20.0` for geometry2 (finer TSV features).

Additionally, a **layer-aware positional embedding** handles the sharp material discontinuities at layer boundaries. Encode the layer index through a learned `nn.Embedding(n_layers, 8)` table and concatenate with the Fourier features.

Full input to MLP:
```
fourier_feats (32) + layer_emb (8) + power_feat (1) + htc_norm (1) + t_amb_norm (1) + tsv_density (1)
= 44 features
```

#### Network Body

```
Input (44) → Linear(256) → SiLU
           → [Residual Block × 6]
           → Linear(256) → SiLU → Linear(1)
```

**Residual block**:
```python
class ResBlock(nn.Module):
    # Linear(256→256) + SiLU + Linear(256→256) + skip connection
    # LayerNorm before each linear (not BatchNorm — batch size varies)
```

**Why SiLU not tanh**: SiLU (Swish) is smooth and non-saturating. Tanh saturates → vanishing gradients in deep networks. PDE loss requires second derivatives; SiLU's second derivative is non-zero almost everywhere. Tanh second derivative → 0 in saturation.

**Why residual not plain MLP**: 6-layer residual ≈ plain 12-layer in expressivity but trains 3x faster (gradient highway). Temperature fields have both long-range smooth variation (HTC, ambient) and sharp local features (hotspot, TSV boundaries). Residuals capture both.

Total parameters: ~400k. Fits comfortably in 2 GB RAM; inference on 400k points < 50 ms CPU.

---

### 7.3 — Loss Function Design

**File**: `src/pinn/losses.py`

#### Component losses

```python
L_data = MSE(T_pred, T_3DICE)                                       # data fidelity

L_pde  = MSE(divergence(k(T,x) * gradient(T)) + Q, 0)              # heat equation
         # k(T,x) = k_layer[layer_id] * (silicon_ratio * (300/T)^1.3 + metal_ratio)
         # computed at collocation points (no labels needed)

L_bc_top = MSE(-k(T) * dT/dz + h * (T - T_amb), 0)                # convective BC at z_top
L_bc_sides = MSE(dT/dn, 0)                                          # adiabatic sides+bottom

L_interface = MSE(T_above - T_below, 0) at layer boundaries         # temperature continuity
              (heat flux continuity is implied by L_pde if k is discontinuous)

L_total = L_data + λ_pde * L_pde + λ_bc * L_bc_top
          + λ_sides * L_bc_sides + λ_intf * L_interface
```

#### Adaptive loss weighting (critical)

Fixed λ values require manual tuning that breaks across geometries. Use **NTK-based adaptive weights** (Wang et al. 2022):

```python
# Each epoch, compute gradient norms of each loss component
# λ_i ← max_loss_grad_norm / loss_i_grad_norm
# Apply EMA smoothing: λ_i ← 0.9 * λ_i_prev + 0.1 * λ_i_new
```

Alternative (simpler): **SoftAdapt** — normalize λ_i by the running mean of each loss.

Start training with `λ_pde=0, λ_bc=0` (data-only) for 1000 epochs, then ramp PDE and BC losses. This prevents the PDE loss from dominating before the network has learned the rough temperature field shape.

---

### 7.4 — Training Strategy

**File**: `src/pinn/trainer.py`

#### Curriculum training

```
Stage 1 (epochs 0-1000):    L = L_data only
Stage 2 (epochs 1000-3000): L = L_data + 0.1*L_pde + 0.5*L_bc  (warm-up PDE)
Stage 3 (epochs 3000-8000): L = L_data + λ_pde*L_pde + λ_bc*L_bc  (adaptive λ)
```

Rationale: if PDE loss fires cold, it pushes the network toward unphysical solutions before it's learned the data distribution. The staged approach guarantees L_data is small before L_pde is enforced.

#### Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2000, T_mult=2, eta_min=1e-6
)
```

Cosine annealing with warm restarts helps escape sharp minima that form when PDE loss activates.

#### Collocation points

- **Data points**: all N points from each `.npz` file in the batch
- **PDE collocation**: 20,000 additional random points per batch from `Uniform(Ω)` (no simulator labels needed)
- **BC points**: 2,000 points on z_top surface, 1,000 on each side face

Total per backward pass: ~30,000 points. Fine on GPU; may need to reduce to 10,000 on CPU.

#### Batch construction

Don't batch individual points — batch entire **scenarios**. Each training step processes 1-4 full scenarios. This ensures each batch contains a spatially complete field with realistic temperature gradients, not random crops.

```python
class ScenarioBatchSampler:
    # Each epoch: shuffle scenario list, yield groups of batch_size scenarios
    # Within each scenario: load all N points, sample collocation points on-the-fly
```

#### Mixed precision

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = compute_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

~2x speedup on GPU with no accuracy loss. Keep model weights in float32, compute in float16.

#### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

PDE second-derivative computation can produce large gradient spikes. Clip at norm=1.0.

---

### 7.5 — Data Preprocessing

**File**: `src/pinn/data_loader.py`

#### Normalization strategy

Per-dataset global normalization (not per-scenario) so the PINN learns a single consistent mapping:

```
x̂ = x / die_length_µm                       # [0, 1]
ŷ = y / die_width_µm                         # [0, 1]
ẑ = z / total_height_µm                      # [0, 1]
T̂ = (T - T_min_global) / (T_max_global - T_min_global)  # [0, 1]
Q̂ = (Q - Q_mean) / Q_std                    # zero-mean unit-variance
htc_norm = (htc - htc_min) / (htc_max - htc_min)
t_amb_norm = (t_amb - t_amb_min) / (t_amb_max - t_amb_min)
```

Global statistics (compute once over full training set, save to `data/norm_stats.json`):
```
T_min = 298.0 K (lower bound across all 80 scenarios)
T_max = 550.0 K (conservative upper bound for extreme scenarios)
htc range: [500, 10000] W/m²·K
t_amb range: [25, 85] °C
```

These constants must be frozen before training starts and stored alongside model checkpoints. The PINN denormalizes at inference: `T = T̂ * (T_max - T_min) + T_min`.

#### Dataset split

- **geometry1**: 15 train scenarios, 5 test scenarios (predefined in YAML)
- **geometry2a/2b/2c**: same split
- Do NOT shuffle train/test — the test scenarios were specifically chosen to be interpolation points in parameter space

#### ThermalDataset class

```python
class ThermalDataset(torch.utils.data.Dataset):
    def __init__(self, npz_files: List[Path], norm_stats: dict):
        # Load all .npz files into RAM on __init__ (total ~800 MB, fits in 16 GB)
        # Store as torch tensors on CPU, move to device in __getitem__
    
    def __getitem__(self, idx):
        # Returns: coords, temp, power, layer_ids, scenario_params
        # All already normalized
    
    def get_collocation_points(self, n: int, geometry: Geometry):
        # Sample n random points from Uniform(Ω)
        # Returns: coords (n,3) normalized
```

---

### 7.6 — Physics Implementation

**File**: `src/pinn/physics.py`

#### k(T) evaluation

```python
def thermal_conductivity(T_norm: Tensor, layer_ids: Tensor,
                         layer_k_values: Tensor, T_min: float, T_max: float) -> Tensor:
    T_K = T_norm * (T_max - T_min) + T_min   # denormalize to Kelvin
    k_base = layer_k_values[layer_ids]        # shape (N,)
    # For Si layers: k(T) = k_base * (300 / T_K) ** 1.3
    # For non-Si: k = k_base (constant)
    is_si = (layer_ids == SI_LAYER_IDX)       # boolean mask
    k_si = k_base * (300.0 / T_K) ** 1.3
    return torch.where(is_si, k_si, k_base)
```

This requires autograd through the `(300/T)^1.3` term. The Jacobian ∂k/∂T is non-zero and non-trivial — the PDE becomes quasi-linear, not linear. This is correct physics and the network can learn it, but it means the PDE residual has a T-dependent coefficient.

#### PDE residual

Use `torch.autograd.grad` with `create_graph=True` to compute second-order spatial derivatives:

```python
def pde_residual(model, x_col: Tensor, scenario_params: dict) -> Tensor:
    x_col.requires_grad_(True)
    T_hat = model(x_col, scenario_params)
    
    # First derivatives
    dT = torch.autograd.grad(T_hat.sum(), x_col, create_graph=True)[0]
    dTdx, dTdy, dTdz = dT[:, 0], dT[:, 1], dT[:, 2]
    
    k = thermal_conductivity(T_hat, ...)
    
    # k * grad(T) components
    kTx = k * dTdx
    kTy = k * dTdy
    kTz = k * dTdz
    
    # Divergence: d(kTx)/dx + d(kTy)/dy + d(kTz)/dz
    div_kT = (autograd_grad(kTx, x_col)[:, 0]
            + autograd_grad(kTy, x_col)[:, 1]
            + autograd_grad(kTz, x_col)[:, 2])
    
    Q = get_power_density(x_col, scenario_params)
    return div_kT + Q   # should be 0
```

Note: this computes 6 autograd passes per PDE evaluation. Expensive but unavoidable for correct second-order PDE. On GPU: ~3x slower than data loss alone. Acceptable given 8000 epochs.

---

### 7.7 — Evaluation & Metrics

**File**: `src/pinn/evaluate.py`

```python
metrics = {
    'mae_K':     mean absolute error in Kelvin (primary metric, target < 2 K)
    'rmse_K':    root mean squared error
    'max_err_K': max point error (target < 5 K)
    'hotspot_loc_err_um': distance between predicted and true hotspot location
    'hotspot_T_err_K':    temperature error at true hotspot location
    'pde_residual_rms':   RMS of PDE residual at test points (physics consistency)
    'r2':                 coefficient of determination (target > 0.999)
}
```

Report per-geometry and per-scenario. Plot: true vs predicted T scatter, z-slice heatmaps (3D-ICE vs PINN), error maps.

---

### 7.8 — File Structure

```
src/pinn/
    __init__.py
    model.py          # FourierPINN, ResBlock, FourierFeatureEmbedding, LayerEmbedding
    losses.py         # pde_residual(), bc_loss(), interface_loss(), adaptive_weights()
    physics.py        # thermal_conductivity(), get_power_density()
    data_loader.py    # ThermalDataset, ScenarioBatchSampler, load_norm_stats()
    trainer.py        # Trainer class (curriculum stages, optimizer, logging)
    evaluate.py       # compute_metrics(), plot_comparison(), plot_error_map()

scripts/
    train_pinn.py     # CLI entry point: --geometry, --data, --epochs, --output
    eval_pinn.py      # load checkpoint, run on test set, export metrics JSON
```

---

### 7.9 — Dependencies to Add

```
torch>=2.1.0
torchvision       # for mixed precision utilities
tensorboard>=2.14
```

Add to `requirements.txt`. GPU is strongly recommended (CUDA 12.x). On CPU: ~6-10 hours per geometry. On GPU (RTX 3070 or better): ~30-60 minutes per geometry.

---

### 7.10 — Implementation Order

1. **Phase 6.5 fixes first** — no PINN training until Die2 bug and mesh normalization are fixed
2. `data_loader.py` + `norm_stats.json` computation — verify all 80 .npz load correctly
3. `model.py` — FourierPINN architecture, unit test forward pass (shapes, no NaN)
4. `physics.py` — k(T) and PDE residual, unit test with known analytical solution
5. `losses.py` — all 4 loss components; test each component in isolation
6. `trainer.py` — curriculum stages, data-only run first (Stage 1 only, 1000 epochs)
7. Full training on geometry1 (Stage 1→3)
8. Scale to geometry2a/2b/2c

---

### Phase 7 Exit Criteria
- [ ] `src/pinn/` module with all files above
- [ ] geometry1 test MAE < 1.5 K (tighter than original 2 K target given k(T) physics)
- [ ] geometry2a/2b/2c test MAE < 3 K each
- [ ] PDE residual RMS < 5% of temperature range at test collocation points
- [ ] Hotspot location error < 500 µm for all test scenarios
- [ ] Inference on 400k points: < 100 ms CPU, < 10 ms GPU
- [ ] TensorBoard training curves saved for all 4 geometries
- [ ] Norm stats frozen and saved alongside model checkpoints

---

## QUICK-START COMMANDS (Phase 6, Day 1)

```bash
# 1. Install missing packages
pip install seaborn pytest pytest-cov mph

# 2. Run test suite baseline
cd "d:\Work\3D-ICE Thermal-modelling\Thermo"
pytest test_geometries.py test_scenarios.py test_phase4.py -v

# 3. Generate mock dataset (no external tools needed)
python src/main.py --all-geometries --simulator mock --output data/mock --verbose

# 4. Inspect a generated file
python -c "
import numpy as np, pathlib
files = list(pathlib.Path('data/mock').rglob('*.npz'))
print(f'{len(files)} files')
d = np.load(files[0], allow_pickle=True)
print('keys:', list(d.keys()))
print('coords shape:', d['coords'].shape)
print('temp range (K):', round(float(d['temp'].min()),1), '-', round(float(d['temp'].max()),1))
"
```

---

## GEOMETRY UPGRADE PLAN

**Purpose**: Evaluate whether the current 4-geometry mesh is adequate for PINN/FNO training and what targeted upgrades would improve accuracy or representational coverage.  
**Audit source**: Static analysis of `src/core/geometry_builders.py`, `src/core/mesh.py`, and geometry2 layer stack definitions.

---

### Current Geometry Snapshot

| Geometry | Grid | Points | Layers | Die thickness | Heat sink thickness | TSV |
|---|---|---|---|---|---|---|
| geometry1 | 100×100×40 | 400 k | 5 | 100 µm die | 1000 µm Cu lid | None |
| geometry2a | 80×80×50 | 320 k | 9 | 50 µm × 2 dies | 5000 µm Cu sink | 3% |
| geometry2b | 80×80×50 | 320 k | 9 | 50 µm × 2 dies | 5000 µm Cu sink | 5% |
| geometry2c | 80×80×50 | 320 k | 9 | 50 µm × 2 dies | 5000 µm Cu sink | 10% |

---

### ISSUE 1 — Die layers critically under-resolved in geometry2 (HIGH PRIORITY)

**Problem**: The adaptive z-mesh in `generate_adaptive_z_coords()` allocates z-points proportionally to layer thickness. With a 5000 µm heat sink dominating the 50-layer budget, each 50 µm die layer receives `max(2, floor(50/total_height * nz))` ≈ 2 points. The temperature gradient is steepest in the die layer — 2 points cannot represent it.

**Consequence for PINN**: The training data has nearly uniform temperature within each die layer in z. The PINN learns a flat profile there, then fails to generalise to the true gradient when evaluated at arbitrary collocation points.

**Consequence for FNO**: The 80×80×50 grid is sufficient in x/y but the z-allocation is physically wrong — die layers share a single interior point.

**Fix: per-layer minimum z-density**

In `src/core/mesh.py`, `generate_adaptive_z_coords()`:

```python
MIN_POINTS_PER_LAYER = {
    'die':       8,   # steep gradient, hotspot prediction accuracy
    'tsv':       6,   # TSV conductivity discontinuity
    'tim':       4,   # thin, high-gradient
    'spreader':  3,
    'sink':      2,   # large, nearly isothermal
}
```

Allocate `min_pts` per layer first, distribute remaining `nz - sum(min_pts)` proportionally. This requires increasing `nz` for geometry2 from 50 to ~70–80 to preserve a sensible total.

**Tradeoffs**:

| Aspect | Before | After (nz=72) |
|---|---|---|
| geometry2 grid | 80×80×50 = 320 k pts | 80×80×72 = 461 k pts |
| FNO memory per batch | ~540 MB | ~778 MB (still fits 8 GB GPU) |
| PINN training data fidelity | Die layer 2-point (misleading) | Die layer 8-point (accurate) |
| 3D-ICE re-generation needed | No (3D-ICE output is point cloud) | No |
| Training data re-reshape | Yes — npz reshape must match new nz | Simple grid change |
| Complexity change | None | Low — single dict in mesh.py |

**Recommendation**: Implement. Accuracy gain is high; cost is low. **The current 2-point die resolution is the most critical accuracy bug in the dataset.**

---

### ISSUE 2 — Power block overlap in geometry2 (MEDIUM PRIORITY)

**Problem**: In `build_geometry2a/2b/2c()`, `tsv_array_d1` occupies y ∈ [3000, 5000] µm and `core1_d1` occupies y ∈ [1000, 3500] µm. These overlap by 500 µm at y ∈ [3000, 3500]. The power density field `generate_power_density_field()` uses last-match wins — whichever block is defined later overwrites the overlap region.

**Consequence**: TSV array silently masks part of core1_d1. The affected die region has lower power than intended. This subtly shifts the hotspot location, affects training data accuracy, and would confuse TSV-density ablation studies (the 500 µm gap is similar in size to TSV effects between 3% and 5% density).

**Fix**: In `geometry_builders.py`, shift `tsv_array_d1` to y ∈ [3500, 5500]:
```python
PowerBlock(name='tsv_array_d1', x=0.0, y=3500.0,  # was 3000.0
           width=10000.0, height=2000.0, ...)
```
Or separate the core into two non-overlapping sub-blocks with a TSV gap:
```python
# core1_d1 split: lower (y ∈ [1000,3000]) + upper (y ∈ [3500,6000])
```

**Tradeoffs**:

| Aspect | Impact |
|---|---|
| Physical accuracy | Removes unintended power masking; hotspot location now correct |
| Dataset compatibility | All geometry2 .npz files become stale and must be re-generated |
| Implementation complexity | Trivial — coordinate change in one function |
| Validation | Verify: after fix, sum of power_density field ≈ intended total chip power |

**Recommendation**: Fix before any publication benchmark. The overlap is an unintentional geometry error, not a physical modelling choice.

---

### ISSUE 3 — No geometry validation layer (MEDIUM PRIORITY)

**Problem**: `build_geometry2a()` and siblings have no assertions to catch:
- Power block x/y coordinates outside die footprint
- Layer z-ranges with gaps or overlaps after `assemble_stack()`
- Duplicate layer names
- Blocks referencing non-existent layer names

If a builder call has a typo (like the Die2 y=4500 bug already fixed), it runs silently.

**Fix**: Add a `validate()` method to `Geometry` in `src/core/geometry.py`:

```python
def validate(self) -> None:
    """Raise ValueError on any structural inconsistency."""
    # 1. Check z-range continuity: each layer.z_bottom == prev.z_top
    # 2. Check all power blocks are within [0, width] x [0, height]
    # 3. Check no duplicate layer names
    # 4. Check all power block layer_name references exist in self.layers
    # 5. Check no power block overlaps exceed 10% of smallest block area (warning)
```

Call `geometry.validate()` at end of each `build_geometry*()` function.

**Tradeoffs**:

| Aspect | Impact |
|---|---|
| Bug prevention | Catches coordinate errors at build time, not silently in output |
| Complexity | ~50 lines in geometry.py; straightforward assertions |
| Runtime cost | Negligible (called once at startup) |
| Future-proofing | New geometries validated automatically |

**Recommendation**: Implement alongside Issue 2 fix. Cheap insurance.

---

### ISSUE 4 — Unused material definitions (LOW PRIORITY)

**Problem**: `src/core/material.py` defines `substrate` (FR4-like, k=0.3 W/m·K) and `underfill` (epoxy, k=0.8 W/m·K) materials, but neither is used in any geometry builder. This is harmless but signals either missing layers or dead code.

**Context**: For 2.5D/3D-IC packaging at the package level (not just the die stack), a `substrate` layer under the heat spreader would add a high-thermal-resistance path. This is physically present in real packages.

**Option A — Remove unused materials** (minimal):
Delete `substrate` and `underfill` from `MaterialLibrary._materials`. Reduces confusion.

**Option B — Add substrate layer to geometry1** (structural upgrade):
Add a 200 µm FR4 substrate layer below the Si die in geometry1. This changes the temperature field (substrate adds ~20 K/W resistance), so all geometry1 PINN training data must be re-generated.

**Option C — Add underfill to geometry2 inter-die bonding** (structural upgrade):
The current geometry2 inter-die bond is a single 5 µm `bonding` layer (k=1.5 W/m·K). Replacing with `underfill` (k=0.8 W/m·K) reduces effective conductivity by ~47% — potentially significant for die-to-die heat transfer.

**Tradeoffs**:

| Option | Accuracy gain | Data re-gen | Complexity |
|---|---|---|---|
| A — Remove | None (cleanup only) | No | Trivial |
| B — Substrate layer | +5–15 K realistic offset | Full geometry1 re-gen | Low |
| C — Underfill bonding | +3–8 K inter-die delta | Full geometry2 re-gen | Low |

**Recommendation**: Option A now (clean up dead code). Options B/C before any publication submission, since the missing substrate is the physically largest omission.

---

### ISSUE 5 — Uniform HTC across entire top surface (LOW PRIORITY)

**Problem**: The convective boundary condition uses a single HTC value for the entire top surface. Real packages have:
- Higher HTC in centre (impinging jet or fin alignment)
- Lower HTC at corners (boundary layer thicker)
- Thermal interface material resistance between die and heat sink (not currently modelled)

**Option A — Spatially varying HTC field** (structural upgrade):
Add a `htc_field(x, y)` callable to `ScenarioParameters` that maps each (x,y) point to a local HTC. Default: constant (backward compatible). Gaussian-peaked variants for jet impingement.

**Option B — Add TIM resistance as explicit layer** (structural upgrade):
Insert a 50 µm TIM2 layer (k=3 W/m·K) between the top die surface and the heat sink in the geometry stack. This is already the most common omission flagged in compact thermal modelling literature.

**Option C — Defer** (no change):
Uniform HTC is standard in compact models. The PINN/FNO will learn correct temperatures for training data as generated. Validation against COMSOL (where TIM and HTC distribution can be set precisely) will bound the resulting error.

**Tradeoffs**:

| Option | Physical realism | Training complexity | PINN BC complexity |
|---|---|---|---|
| A — Spatial HTC | High (jet impingement) | +1 input channel to FNO | BC integral becomes field convolution |
| B — TIM layer | Medium (bulk effect) | No change to FNO | No change to BC |
| C — Defer | Low | None | None |

**Recommendation**: Option B (TIM layer) is the highest realism-to-cost ratio upgrade. Option A is significant work that adds an input channel to the FNO and complicates the PINN BC — defer to publication phase. Option C acceptable for initial benchmarks.

---

### ISSUE 6 — TSV density is discrete, not continuous (MEDIUM PRIORITY, FNO-specific)

**Problem**: The three geometry2 variants provide TSV densities at {3%, 5%, 10%} only. The FNO is expected to generalise across TSV density as a continuous parameter (`tsv_frac` input channel), but trained only on 3 discrete values. Interpolation to 7% TSV is untested. Extrapolation beyond 10% is unsupported.

**Fix**: Generate 3–5 additional intermediate TSV scenarios using `create_tsv_material(tsv_density)` for density ∈ {1%, 2%, 7%, 12%, 15%}. These require new geometry builder entries or a parameterised `build_geometry2(tsv_density)` factory. Each new density needs 20 3D-ICE scenarios (~20 min simulation time).

**Tradeoffs**:

| Aspect | 3-point training | 8-point training |
|---|---|---|
| FNO TSV interpolation error | ~10–20% at midpoints (estimated) | ~2–5% at midpoints |
| 3D-ICE simulation time | Already done | +5 densities × 20 scenarios ≈ 2–3 hours |
| Storage | Already done | +~200 MB |
| Grid compatibility | geometry2 all share 80×80×50 | Same — parameterised builder |
| Code change | None | Parameterised `build_geometry2(tsv_density)` factory |

**Recommendation**: Implement the parameterised factory and generate intermediate densities before FNO training. The 3-point training set is likely insufficient for continuous generalisation claims.

---

### ISSUE 7 — No geometry3 (large die / high-power) (LOW PRIORITY, RESEARCH)

**Problem**: The 4 geometries all share a ~10×10 mm die footprint and are thermally similar in structure. There is no high-power server-class geometry (e.g., 30×30 mm, 300 W TDP) or memory-on-logic heterogeneous geometry (different die sizes bonded face-to-face). Including one would improve the universality claim of the FNO operator.

**Tradeoffs**:

| Aspect | Impact |
|---|---|
| Research novelty | Significantly increases publication scope |
| Implementation | New geometry builder, new scenario set, different mesh |
| 3D-ICE simulation | 20 additional scenarios (~1 hour) |
| FNO training | Requires matching grid resolution or a geometry-agnostic operator |
| Priority | Phase 9+ (research extension) |

**Recommendation**: Defer to Phase 9. Not needed for initial PINN validation benchmarks.

---

### Summary Table — Geometry Upgrade Decisions

| Issue | Priority | Accuracy impact | Complexity | Data re-gen? | Decision |
|---|---|---|---|---|---|
| 1 — Die z-resolution (nz=72) | **Critical** | High — fixes 2-pt die gradient error | Low | No (reshape only) | **Do now** |
| 2 — Power block overlap | **High** | Medium — fixes hotspot location | Trivial | Yes, geometry2 only | **Do before benchmark** |
| 3 — Geometry validation | Medium | Bug prevention | Low | No | **Do alongside #2** |
| 4A — Remove unused materials | Low | Cleanup only | Trivial | No | **Do now** |
| 4B — TIM layer (geometry1) | Low | +5–15 K realism | Low | Yes, geometry1 | Defer to Phase 8 |
| 5 — Spatial HTC | Low | High (jet model) | High | Yes | Defer to Phase 9 |
| 5B — TIM resistance layer | Low | Medium | Low | Yes | Defer to Phase 8 |
| 6 — TSV intermediate densities | Medium | FNO generalisation | Low | Yes (new runs) | **Do before FNO training** |
| 7 — Geometry3 (server class) | Low | Research breadth | Medium | Yes | Defer to Phase 9 |

### Immediate Action Items (before PINN/FNO training)

1. **Fix die z-resolution** — modify `generate_adaptive_z_coords()` with per-layer minimum density dict; set geometry2 `nz=72` (8 pts/die × 2 dies + headroom)
2. **Remove unused materials** — delete `substrate` and `underfill` from `MaterialLibrary._materials`
3. **Fix power block overlap** — shift `tsv_array_d1.y` from 3000 to 3500 in all three geometry2 builders
4. **Add `geometry.validate()`** — call from each builder function
5. **Re-generate geometry2 3D-ICE dataset** — 20 scenarios × 3 geometries ≈ 1 hour (power block overlap fix makes old data physically incorrect)
6. **Add parameterised `build_geometry2(tsv_density)` factory** — generate 2–3 intermediate TSV densities before FNO training
