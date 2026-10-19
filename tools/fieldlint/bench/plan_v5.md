# fieldlint round 5 (v5): validation plan (DRAFT for user review; not frozen)

Written 2026-10-15. Status: **PLAN ONLY.** No rule code was modified, no held-out candidate was downloaded, no rule was run on any new
data, nothing is committed. What *was* done: (a) read-only diagnostics on ALREADY-SEEN data (v3/v4 tuning and held-out files, v4 held-out
results; one CPU process at a time, small Python scripts kept in the session scratchpad, not in the repo); (b) web checks of candidate
datasets (pages and APIs only, no data files fetched). Provenance labels used below: **[VERIFIED]** = page or API fetched on 2026-10-15
and the stated fact read there; **[UNVERIFIED]** = found by search or inferred, not confirmed; **[MEMORY]** = from my background knowledge
or the project memory, not checked this session.

Everything measured in v4 (held-out tables, C1-C4) becomes **tuning evidence for v5**. v5 must therefore use only files nobody has run a
rule on.

---------------------------------------------------------------------------------------------------------------------------------

## 1. Diagnosis of the v4 C3 failure (F005)

### 1.1 Exactly which check fired

Source: `src/fieldlint/rules.py` lines 514-588 (`f005_range`), v4 result files `bench/results/v4_heldout/{pdegym_helmholtz,pdegym_wave_layer}.json`.

Both warnings are the **same single check**: the *below-ambient* check (`rules.py` lines 547-548 and 574-576), message
"N sample(s) have u below the ambient level although the source is non-negative (a passive problem cannot cool below ambient)".
Condition per sample: `ambient is not None and source.min() >= 0 and u.min() < ambient - max(1e-3*ptp(u), 1e-6)`.
No other F005 check fired (units are `''`, so the K/C range checks are off; k > 0; no constant fields).

| control | what the preset declares | why the premise holds *by mapping* | what the data really are | samples firing (400 / 256 checked) |
|---|---|---|---|---|
| PDEgym Helmholtz (steady) | `ambient: 0.0`, `source_uniform: 1.0` | an **invented** uniform source of 1 (the dataset has **no** source term; forcing is the nonzero Dirichlet value b) makes `source.min() >= 0` true on every sample | u in [-4.71, 4.59], oscillating about 0: 37% of cells below 0 (median), per-sample undershoot 0.9-1.5 x the positive height | 400 / 400 (100/100 in my 100-sample re-check) |
| PDEgym Wave-Layer, `time_dependent: true` | `ambient: 0.0`, `source = u(t0)`, `u = u(T)` | `u(t0)` is a sum of Gaussians, `source.min()` in [0, 2.6e-13] >= 0, so the premise holds | wave snapshot, u in [-0.31, 0.39], 57% of cells below 0 (median) | 256 / 256 (100/100 re-check) |

**Why the time-dependent flag did not help:** `scope_skip` (rules.py 148-171) is called by `operator_skip`, which F002/F003/F008/F009 use. `f005_range` never calls it, so the
`time_dependent` declaration, `operator: advective|other` and the advection probe have **no effect on F005**. This is a design omission of v4 ("F005 left unconditional", listed in
`bench_results_v4_heldout.md` as POST HOC candidate 1), not a tuning accident.

**Why earlier controls did not fire (important, it was luck):** Wave-Gauss (v3), CNO wave, Allen-Cahn, Navier-Stokes, DiffusionPDE Helmholtz and CNO discontinuous transport all have `ambient: 0.0` and
oscillating or negative-lobed u, but each has a source with at least one negative cell (sign-changing sources, or float32 noise down to -6e-8 in Wave-Gauss' `u0`), so `source.min() >= 0` was false on 100/100
of the samples. Wave-Layer's `u0` is exactly >= 0, which is the only reason it fires. CNO discontinuous transport escapes only through the `1e-3*ptp` tolerance (its undershoot is -6.4e-4 against a tolerance of about 1e-3). So v3/v4
were one sample-level coincidence from the same failure on four other controls. The fix must not rely on "the source happens to be negative somewhere".

### 1.2 What the check is supposed to catch (do not lose it)

On seen data the below-ambient check legitimately fires on exactly three sets (all adjudicated TRUE or diagnostic):

* `cno_darcy` (stored output is <= 0 everywhere, opposite sign of its documented equation; every sample; 94% of cells below ambient). Material-benign sign convention, protocol-adjudicated TRUE in v3.
* `dpde_darcy_denorm` (a few tiny negative values after de-standardisation; undershoot about 0.1-0.2% of range; adjudicated TRUE in v3).
* `dpde_darcy_asreleased` (diagnostic; standardised u, offset by the mean; also has an F005 k <= 0 error).

Faulted copies also use F005 through the K-range checks (`unit_slip`) and, on 3D-ICE/IC-ThermBench, through per-sample ambient on `shift_index`; those are untouched by the proposal below.

### 1.3 Proposed fix (physics-motivated; two parts; tuned only on seen data)

**Part A: scope gate for the below-ambient check (declaration based).**
`f005_range` evaluates `scope_skip(ds, opt)` (time-dependent, `operator: advective|other`, effectively 1D) once. If out of scope, the **below-ambient check only** is not applied and F005 reports an info line
"below-ambient check not applied: out of scope (<reason>)". The integrity checks (constant field, K/C range, k <= 0, zero source) stay unconditional because they are properties of the file, not of the physics.
Physics: a wave or transport snapshot is not a passive relaxation to ambient, so "cannot cool below ambient" is not a statement about it. This is the same reason the other four rules already decline.
Fixes Wave-Layer with the declaration (and, as a by-product, every flagged time-dependent control). Does not help steady Helmholtz.

**Part B: weak-minimum-principle (WMP) probe for the below-ambient check (automatic, no declaration).**
Physics: if `-div(k grad u) + c(u - ambient) = s` with `s >= 0`, `k > 0`, `c >= 0`, the weak minimum principle says `min u` over the domain is attained on the boundary **or** is >= ambient. A constant offset of u does not
change the first statement. So for a sample with `source >= 0` that is *not* sign-definite (u.max > ambient + tol), define

`V = ( min over the in-plane boundary ring of u - min u ) / ptp(u)`   (per z-layer, take the max over layers)

V > 0 means an interior undershoot below the boundary minimum: impossible for a passive diffusion solution with a non-negative source, and typical for Helmholtz/wave/oscillatory data. A sample counts as **oscillatory**
when V >= `osc_v` = **0.05**. If >= 50% of the checked samples are oscillatory, the below-ambient warning is replaced by an info line "u violates the weak minimum principle (median V = ...): not a passive
diffusion solution; the steady-diffusion rules do not apply" and the F005 summary says so. Isolated oscillatory samples (< 50%) still warn, so a corrupted fraction of an otherwise-diffusion dataset is not hidden.
Sign-definite fields (u.max <= ambient + tol, e.g. `cno_darcy`) and constant offsets (V = 0, e.g. `dpde_darcy_asreleased`) are untouched: they still warn.

**Tuning evidence collected now (seen data only, first 400 samples, non-negative source, u.max > ambient; script in the session scratchpad, read-only):**

| group | datasets | V (per sample) |
|---|---|---|
| true diffusion data, all seen families (IC-ThermBench S2-S5, 8 Therm-FM sets, 6 3D-ICE geometries, PDEBench Darcy x5, FNO Darcy x2, PDEgym Poisson-Gauss, ELADO A1/B1, UCSD Poisson, DiffusionPDE Darcy denorm/as-released, ConvDiff, CNO transport, ELADO Helmholtz E1/C1 where applicable) | 35 sets | **max V = 0.0139** (HS_SC_refine1); 29 of 35 have max V = 0.000 |
| Wave-Layer (control) | 256 samples | min 0.092, median 0.298, max 0.482 |
| PDEgym Helmholtz (control) | 400 samples | min 0.525, median 0.632, max 0.751 |

Gap between the largest diffusion value (0.0139) and the smallest control value (0.092): 6.6x. `osc_v` = 0.05 sits at 3.6x above the former and 1.8x below the latter (geometric centre 0.036; 0.05 is a round value
inside the gap, not a fitted point). My first instinct (0.10) was *wrong* because the smallest Wave-Layer sample is 0.092; this is why the table matters. Fraction rule 50%: Wave-Layer has >= 100% of samples >= 0.05 and Helmholtz 100%;
all diffusion data 0%.

**Ruled out by the same diagnostic (do not re-propose without new evidence):**
* *Interior-local-minimum count* (a diffusion field with non-negative source has none): fails on seen data. Multi-layer thermal sets (Therm-FM IND_8C/IND_32C 40/40 samples, 3D-ICE geometry 5-6, IC-ThermBench S5) have many strict interior minima per z-layer because of vertical coupling, and PDEgym Helmholtz has them in only 16/40 samples.
* *Sign-straddle / lobe fraction alone* ("d" = depth below ambient / height above): `dpde_darcy_asreleased` (a true offset fault) has d = 0.48-1.45, indistinguishable from Helmholtz (0.15-1.5) and Wave-Layer (0.24-1.6).
* *Connected-component count of the negative region*: 1 for DiffusionPDE Helmholtz, 2-4 for PDEgym Helmholtz, 1 for the as-released Darcy; no usable gap.

**What each part buys, honestly:**
* A alone repairs the pre-registered declared-scope C3 (Wave-Layer flagged) and removes an inconsistency; it cannot repair steady Helmholtz.
* B alone repairs both v4 failures on seen data without a declaration, but B is a **second scope inference** (like the advection probe), so it can also swallow a genuine in-scope fault: a corrupted dataset in which a large fraction of samples
  have interior undershoot >= 5% of range (heavy noise, a failed solve) would be reported as "not diffusion" rather than "below ambient". Today F002/F003/F008/F009 see such data too, so the loss is bounded.
* ELADO C1 / E1 (Helmholtz stored as |u|) never triggered F005 and still do not; they are declined by the R^2 gate of F008/F009, not by this fix.
* **F005 is not a general out-of-scope detector.** An unflagged Wave-Layer (C3-strict) still draws F002 (error) and F003; part B does not touch those. Extending the WMP probe into `operator_skip` (so F002/F003 would also decline when V says "not diffusion") is the obvious v5b, but it needs its own tuning evidence on F002/F003/F008/F009 detection (every seen faulted run must keep its status) and I do not recommend bundling it. Decision for the user (section 8).

**Tuning check required before any freeze (Phase A, seen data only):** re-run `run_v4_design.py`-style tuning over all 31 in-scope datasets x 15 conditions plus the 18+ controls (the v4 tables), and require: (i) no rule's status changes on any run except F005 on Wave-Layer (both flagged and unflagged) and PDEgym Helmholtz, which must become info; (ii) F005 still flags `cno_darcy` (all 13 fault copies), `dpde_darcy_denorm`, `dpde_darcy_asreleased` and every `unit_slip` / `shift_index` detection exactly as before; (iii) zero new clean findings. Add unit tests (synthetic Helmholtz box mode, synthetic offset diffusion, synthetic sign-flipped diffusion, synthetic wave snapshot with `time_dependent`). Then freeze `frozen_thresholds_v5.json` with the code SHA-256 (the v2/v3/v4 files stay untouched).

**Also noted in the diagnosis (not a rule change, a protocol change):** the invented `source_uniform: 1.0` for steady Helmholtz-type controls is the reason the premise holds. v4 disclosed it ("a reading, not a documented source") and
the verification pass found the data card for PDEgym Helmholtz inconsistent with the stored data (equation residual about 0.95 relative; `a` in [1, 2], not [0, 1]; so the v4 prediction "u > 0" was wrong). For v5 every control card must be written from the **stored data's
dimensions and the generator code**, not only from the paper, and the mapping of a source-less PDE (Helmholtz) must be decided once, before any rule runs: either keep the "user's plausible reading" (uniform 1; harder, realistic) or leave `source` unset (F005's premise cannot hold; easier). I recommend the harder reading as binding and the unset-source run as a reported diagnostic.

---------------------------------------------------------------------------------------------------------------------------------

## 2. Fresh held-out data

Already used (do not reuse as held-out): everything in `bench/datasets_v3.md`/`datasets_v4.md` and the three tuning families (IC-ThermBench S2-S5 first 400 samples, Therm-FM 8 steady sets, project 3D-ICE geometry1-6 fixed solves, PDEBench Darcy beta 0.01/0.1/1/10/100, FNO Darcy 128 and native 421, PDEgym Poisson-Gauss/Helmholtz/Wave-Gauss/Wave-Layer, CNO Poisson/Darcy/Wave/Allen-Cahn/NS/transports, DiffusionPDE Poisson/Darcy/Helmholtz, ELADO A1_test/B1_test/E1_test/C1_test, UCSD Poisson4/ConvDiff1, ConDiff x5). Same-generator-different-file sets are *low independence* and are labelled so.

**Honest summary first.** The web checks found **no public chip/heat dataset that is verified, downloadable, gridded, and paired (source/conductivity -> temperature)** apart from what the project already used. The thermal candidates below are either unverified, unstructured-mesh, time-dependent, or the project's own data. That is itself a finding for the thesis (section 7).

### 2.1 Thermal / heat-conduction application candidates (goal (a): at least two)

| id | candidate | role | URL | licence | size / format | status |
|---|---|---|---|---|---|---|
| T1 | **SISSAmathLab/thermal-conduction** (Hugging Face): steady-state and unsteady snapshots of heat conduction on piecewise-conductivity "stripe" geometries, 2,320 rows (1,160 geometry + 1,160 snapshots), splits like `3_stripes.star_basic.1_adaptive_refined` | in scope (if the steady field and a source/BC mapping can be read) | https://huggingface.co/datasets/SISSAmathLab/thermal-conduction | **not stated** (README empty) [VERIFIED: README empty] | 2.36 GB parquet; geometry: `points, edge_index, constraints_values, constraints_mask, conductivity, cells`; snapshots: `unsteady, times, steady` | **[VERIFIED]** exists and sizes (HF API 2026-10-15). **[UNVERIFIED]** equation, source, BCs, units; **unstructured adaptive mesh**: needs a pre-registered rasterisation to a grid, so the field is *processed*, not native (F008/F009 will probably skip, F003 maybe). Licence unknown: ask the authors before publishing results that name it |
| T2 | **Edalatifar et al., "A dataset for conduction heat transfer and deep learning"** (Mendeley Data v2, 2022): image-based Laplace/Dirichlet conduction, FVM | in scope (if fields can be mapped) | https://data.mendeley.com/datasets/rw9yk3c559 | CC BY 4.0 [VERIFIED] | the Mendeley files API lists **only** `ReadDataset.py` (666 B) and `ReadDataset.m`; the named `HeatTransferPhenomena_35_58.npz/.zip` is **not listed**, size/grid/fields/sample count unknown | **[UNVERIFIED]**: may not be downloadable via the API; resolve by a manual page visit. Low priority |
| T3 | **Heat-2D** benchmark (simple_chip, multi_chip, forced_convection; inputs: conductivity, heat source, convection coefficient; output: temperature; used in arXiv 2606.09923) | in scope, the best thematic match (chip-like 2D steady conduction with Robin cooling) | no repository found. Possible pointer: `github.com/physai/physai` (404 on fetch) | unknown | unknown | **[UNVERIFIED]**: the name and the three variants exist in a paper and on a leaderboard page; hosting not found. Ask the corresponding author (Michael Chin) for the data or the generator; do not count it until a file with a hash exists |
| T4 | **IC-ThermBench S1 "source suite"** (Google Drive folder) | in scope if it is not Therm-FM re-packaged | https://github.com/Day333/ThermalBench -> Drive folder | S1 "retains its original licence"; S2-S5 CC BY 4.0 [VERIFIED] | size not stated | **[UNVERIFIED]** overlap with the Therm-FM sets already seen. S2-S5 are 15,000 / 15,000 / 15,000 / 5,000 samples (the "50,000-sample extension" in the abstract is S2-S5 and is **not fresh**: only samples 0-399 per scope were linted, but the files are seen, so samples 400+ are *same-file* evidence only) |
| T5 | **Project's own layout-varied 3D-ICE solves** (`data/3d-ice-layout-geometry1..7`, `3d-ice-interface-*`, `3d-ice-microchannel-pilot`, `3d-ice-geometry7-*`, `3d-ice-throttle/leakage-pilot`): real chip stacks with real geometry, layout randomisation (the regime where ridge fails) | in scope, real thermal application data | local | project data | 4-43 MB per directory, tens of samples each | **[MEMORY/ls]** directories exist; per-set N is about 40-55 (< the 200-sample minimum of the v3/v4 protocols); `moving-geometry4` was already linted by the earlier floorplanner/linter work (exclude); the repo's `test_realdata.py` mentions layout/microchannel directories (check it did not tune on them). Not independent of the authors, never seen by fieldlint (to be confirmed by grep of `bench/` and `logs/`). **Known ground truth**: the project's memory records a 2026-09-23 transposed-temperature bug in older saved data; the linter should flag exactly those, which gives a free positive control on real thermal data |
| T6 | **Generated thermal sets with an independent solver** (fallback if T1-T3 fail): 2-3 chip-stack sets (power-map layouts, TIM/spreader conductivity, convective top boundary) from an FEM code that is not 3D-ICE/HotSpot (e.g. FEniCS or scikit-fem), generator + seed + parameters frozen in the protocol *before* generation | in scope | n/a | n/a | about 50 MB per set | not a public dataset; strongest control over ground truth (can inject faults before and after the solve), weakest on independence. Needs a user decision |
| T7 | 4TU "Square cavity coupled heat transfer dataset" (COMSOL, dual-cavity turbine blade, PINN study) | **time-dependent thermal control** (200 x 420 x 1000 values, a time axis) | https://data.4tu.nl/datasets/66b5bd53-ef82-41da-bc45-104e8de0c4cf | CC BY 4.0 [VERIFIED] | 3.27 GB `.rar`, CSV | **[VERIFIED]** exists; fields/steadiness **[UNVERIFIED]**; not a source-to-temperature pairing, so only usable as a thermal-flavoured control with a mapping decision |
| T8 | UCR VSCLAB "commercial thermal map dataset" (measured thermal maps of 8 CPUs/GPU/TPU; pickles) | **rejected**: measured maps, no PDE pairing; full set "on request"; no licence | https://github.com/sheldonucr/commercial_thermal_map_dataset | not stated | samples only | [VERIFIED] |
| T9 | GRIFDIR Zenodo `10.5281/zenodo.20753300` (Gaussian-blob conductivity fields 32x32 and irregular domains) | probably **not usable** (inputs; solution fields not confirmed) | https://doi.org/10.5281/zenodo.20753300 | CC BY 4.0 [VERIFIED] | 529 MB zip | [VERIFIED] exists; contents **[UNVERIFIED]** |

Recommendation: plan v5 around **T5 (own layout-varied solves, real thermal)** and **T1 (SISSA thermal conduction, processed mesh)** as the two thermal application sets, with T3 as an upgrade if the authors answer, and T6 as the fallback that keeps goal (a) alive. If only T5 is
usable, the thesis claim becomes "one real thermal application family" and must say so.

### 2.2 Fresh advective control (goal (b))

| id | candidate | URL | licence | size / format | status |
|---|---|---|---|---|---|
| A1 | **FlowBench**, 2D lid-driven cavity families "Navier Stokes + Heat Transfer" (constant / variable Reynolds number), 128/256/512 squared, `X`,`Y` npz with a temperature field | https://huggingface.co/datasets/BGLab/FlowBench | **CC-BY-NC-4.0** [VERIFIED] (non-commercial; fine for a thesis, state it) | total repo 1.72 TB; the LDC subset size is not stated, so fetch **one** `.npz` by range request, cap at 3 GB | **[VERIFIED]** repo, licence, grid sizes. **[UNVERIFIED]** which channel is temperature, steady vs transient, subset size. Thermally relevant (heat advected by flow) and genuinely advection dominated |
| A2 | **Fourier Neural Mappings advection-diffusion** (Huang, Nelsen, Trautner; CaltechDATA) | https://data.caltech.edu/records/r5ga1-55d06 | **CC0** [VERIFIED] | `advection_diffusion.zip` 670.3 MB [VERIFIED]; train/test for KLE dimension 2/20/1000 (GitHub) | **[VERIFIED]** exists. **[UNVERIFIED]**: steady? full fields or only finite-dimensional observables (the paper is about parameter-to-observable maps; likely the latter, in which case it has no u field and is unusable) |
| A3 | **The Well `rayleigh_benard`** (buoyancy-driven thermal convection; CC-BY-4.0 per the Well's Hugging Face page) | https://huggingface.co/datasets/polymathic-ai/rayleigh_benard | CC-BY-4.0 [VERIFIED for helmholtz_staircase page; assumed same for the collection: **[UNVERIFIED]**] | 35 validation files of 1.08 GB each [VERIFIED]; take one | time-dependent: a **flagged** thermal control, not a test of the probe (probe is only run when not flagged) |
| A4 | AirfRANS (steady RANS around airfoils) | https://airfrans.readthedocs.io | **[MEMORY]** | **[MEMORY]** about 9 GB, unstructured | **[UNVERIFIED]**; backup only; mapping to (u, source, k) is awkward |
| A5 | **Self-generated advection-diffusion set** with a frozen generator (like T6) | n/a | n/a | n/a | fallback; independent of availability, not independent of the authors |
| A6 | CNO Euler-airfoil (steady hyperbolic) already unpacked in `data/_external/cno_rpb` but never registered | local | CC-BY-4.0 [v3 card] | seen file list but never linted | mapping unclear; optional extra |

Recommendation: A1 (one fetched file) as the primary fresh advective control; A5 as the contingency if A1's temperature channel cannot be mapped; the probe's pre-registered prediction is stated in section 3. Keep A2 off the list until the shape of the stored outputs is confirmed by looking at the README inside the zip (header only).

### 2.3 Fresh Helmholtz / wave controls (goal (c))

| id | candidate | role | URL | licence | size / format | status |
|---|---|---|---|---|---|---|
| H1 | **The Well `helmholtz_staircase`**: time-harmonic Helmholtz with a point source in a staircase waveguide, real + imaginary pressure plus a mask, 1024 x 256, 16 omega values | steady Helmholtz control with a real (non-uniform, localised) source | https://huggingface.co/datasets/polymathic-ai/helmholtz_staircase | CC-BY-4.0 [VERIFIED] | repo 54.8 GB, but the **test split is 16 files x 335.5 MB** (HF API); take 2 files (about 0.7 GB), HDF5 | **[VERIFIED]** exists and file sizes. **[UNVERIFIED]**: array names, where the source position is stored, 1024 x 256 grid is non-square (check `min_inplane_cells` and the adapters) |
| H2 | **The Well `acoustic_scattering_*`** (wave in discontinuous / inclusion / maze media) | time-dependent wave control (flagged) | https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_inclusions | CC-BY-4.0 (collection, **[UNVERIFIED]** per set) | validation chunks are **8.11 GB each** [VERIFIED] -> over the per-file budget; use a ranged read of the first trajectories like v3/v4 did for Wave-Gauss/Layer | needs a conversion script (new code, outside the hash) |
| H3 | **WaveBench** (Zenodo 8015145; Helmholtz and time-dependent wave sets per its paper) | steady Helmholtz + time-dependent controls | https://zenodo.org/records/8015145 | CC BY 4.0 [VERIFIED] | **one 75.6 GB zip** [VERIFIED]; content listing **[UNVERIFIED]** | **over budget**; only usable if single members can be range-read from the zip (Zenodo supports ranges, as in v4). Flagged: do not download whole |
| H4 | OpenFWI velocity-map -> seismic-data sets | wave | [MEMORY] | [MEMORY] | tens of GB | mismatched grids for (u, source, k); skip |
| H5 | **DiffusionPDE-normalized Helmholtz**, PDEgym Helmholtz, ELADO C1/E1 | already used | | | | not fresh |

Recommendation: H1 (two test files) as the steady Helmholtz control and H2 (ranged read, first 256 trajectories of one file) as the flagged wave control. Plus A3 for a flagged thermal-convection control.

### 2.4 Fresh in-scope non-thermal sets (to keep the number of independent generator families honest)
Fresh files of already-seen families are cheap but low independence: ELADO `A1_tail` (1.96 GB), `A1_os_test` (3.98 GB, out-of-spectrum variant of A1), `C1_tail` (0.49 GB, a Helmholtz control), `D1_test` (0.98 GB; PDE not stated in the arXiv abstract, which only says "Poisson and Helmholtz with non-constant coefficients" **[VERIFIED abstract]**); ConDiff other covariances/variances and 64 x 64 grids; PDEBench Darcy beta 0.5/5. Each can be added as a labelled "same family, fresh file" row and must be excluded from the family-level count.
A genuinely new generator family is the scarce resource: SISSA thermal conduction (T1) and Heat-2D (T3) are the only candidates found; say so.

### 2.5 Download budget
Core plan (all verified or cheap): FlowBench one file <= 3 GB, Helmholtz-staircase 2 files 0.7 GB, Wave range reads about 0.4 GB, Rayleigh-Benard 1 file 1.1 GB, SISSA thermal-conduction 2.4 GB, own 3D-ICE (local, 0.2 GB), plus optional ELADO A1_tail 2 GB, C1_tail 0.5 GB, D1_test 1 GB. Total about **9-12 GB**, below the 20 GB ceiling. **Flagged over budget:** WaveBench full zip (75.6 GB), Helmholtz-staircase repo (54.8 GB), acoustic_scattering single file (8.1 GB), FlowBench repo (1.72 TB). None of these is to be downloaded whole.

---------------------------------------------------------------------------------------------------------------------------------

## 3. Pre-registered criteria

### 3.1 Keep C1-C4 unchanged in definition and threshold

| id | criterion (as v4 section 10) | keep? and why |
|---|---|---|
| C1 | clean-adjusted any-rule detection of whole-dataset transpose / flip_x / flip_y over in-scope applicable trials >= 80% (report the exact one-sided Clopper-Pearson lower bound) | **keep.** Changing a threshold that passed twice (v3 0.86, v4 1.00) would look like fitting. Add what the bound can bear: all-hit bounds are 0.05^(1/n): n = 27 -> 0.89, n = 36 -> 0.92; one miss at n = 27 -> 0.80 |
| C2 | clean false positives after adjudication <= floor(N/10) (0 for N < 10) | **keep**, unchanged; adjudication routes (a)/(b)/(c) and P1-P3 carried over |
| C3 | 0 false alarms among F002, F003, F005, F008, F009 on binding controls (flagged for time-dependent, as-is for steady) | **keep the definition**; the two v4 failures are fixed in code (section 1), not by relaxing the criterion. **Additions (new controls, not new thresholds):** the fresh advective control (A1) is a binding control and, with the unchanged rule, must show either "declined as advective" or no F002/F003/F008/F009 warning; fresh steady Helmholtz (H1) and flagged waves (H2, A3) are binding |
| C4 | F008 applicable on >= 50% of native in-scope sets | **keep.** Restate the native list per dataset card before any run; processed sets (T1) are non-native by construction |

Additional reporting (not criteria, but part of the verdict text): per-family and per-dataset (cluster) detection, because the 3 faults on one dataset are not independent. Dataset-cluster lower bound for k of k clean datasets: k = 9 -> 0.72, k = 12 -> 0.78, k = 16 -> 0.83. State that the pooled trial CI is the optimistic one. Also report the C3 false-alarm rate with its upper bound: 0 of 6 controls -> upper 95% bound 0.39; 0 of 10 -> 0.26. Binary pass at "0 of 6" is a statement about these six files, not a rate.

### 3.2 New criteria (need fault injection; implemented as new fault kinds in a new `run_v5_*.py`, fault code outside the hashed rules)

**Prediction audit first (records, not criteria).** Reading `rules.py` suggests two structural limits that v5 should state in advance rather than discover:
* F004 compares up to `max(max_samples, 2000)` samples (so all pairs for N <= 2000) and exact-hashes, then a max-abs-difference test with `dup_rtol = 1e-5` of range. It detects *copies*, not perturbed copies. v4: cross-split duplicates 9/9, no clean hits.
* F009 scores `pair_samples = 32` **evenly spaced** samples. For N = 1000 neighbours are about 31 apart, so an off-by-one pairing fault (`u_i` replaced by `u_{i+1}`) has its correct partner *outside* the 32 sampled columns; every R^2 in the row is then near 0, the applicability gate reads "operator model does not apply", and F009 **skips**. v4 confirms: F009 detected `shift_index` on 0/9 (only F003 on 1/9). F009 has real detection power only on small N (the 45-55-sample 3D-ICE sets) or on a contiguous block. So v5 must test F009 in two arms (below). A rule change (sample a contiguous block of M samples) is a candidate v5 modification that I do **not** include in the base plan; it would be tuned on seen data exactly as part A/B and its effect tested in the tuning stage. Decision in section 8.

**C5 (F004, duplicate detection), per in-scope dataset with N >= 200:**
 * Faults (new, exact-copy): `dup_x1` (1% of samples are copies of earlier train samples placed in the test region), `dup_x5` (5%), `dup_x10` (the existing `cross_dup`), `dup_within5` (5% copies inside the train split). Minimum 2 duplicated samples at 1%.
 * Faults (perturbed copy, characterisation only): `dupnoise_1e-7` (relative to ptp; below `dup_rtol`, must be detected), `dupnoise_1e-4` and `dupnoise_1e-2` (above `dup_rtol`; a documented non-detection, reported, **not** a criterion).
 * Pass thresholds: **>= 95%** exact-copy detection over all (dataset, dose) cells with a one-sided 95% lower bound reported (deterministic hash: 100% expected, the bound only measures how many datasets there are: 12 datasets x 4 faults -> 0.94); `dupnoise_1e-7` detection >= 90%; clean copy: F004 warning or error on 0 in-scope sets after adjudication (P2). F004's own cross-split rule needs split labels, so splits are assigned positionally as before. A clean F004 hit that is a true duplicate in the released file is a discovery, reported separately.
 * CI: Clopper-Pearson exact, per cell and pooled; clustered by dataset in the secondary report.
 * Why these numbers: v2-v4 pooled `cross_dup` detection was 18/18, 20/20 and 9/9; the 95% bar leaves room for one miss in 20 and still catches a broken hash path.

**C6 (F009, pairing detection), two arms, on in-scope sets where F008 is applicable on the clean copy (the only place F009 can run):**
 * Arm A (as shipped, N up to 1000, 32 evenly spaced rows): faults `shift_index` (u_i := u_{i+1}), `shift_index_5` (u_i := u_{i+5}), `swap_blocks20` (20% of positions permuted among themselves), `shuffle_all`. **Pre-registered prediction: ~0% detection by construction** (partner outside the sampled columns). Reported, non-binding, because it is a limitation and not a measurement of the idea.
 * Arm B (user-usable): the same lint on a head block of M = 64 consecutive samples (`--head 64`, the documented way to lint a block), faults `shift_index`, `shift_index_5`, `swap_blocks20`, with `shuffle_all` reported only (a full shuffle puts the partner outside the block, so skips are expected and counted as misses, as in v2).
 * Binding thresholds (Arm B only): **clean-adjusted `shift_index` detection >= 70%** over F008-applicable sets (tuning: 12 of 12 applicable sets detected `shift_index`, 8 of 20 skipped; the 70% allows for held-out generators shifting some sets below the gate), **0 F009 findings on clean copies** (tuning 0/18, v4 0/9), and the **skip rate reported**. Lower bounds: with 10 applicable sets all hit, 0.74.
 * Why: F009 is the only rule meant for pairing, and the tuning table shows `shift_index` is otherwise caught by chance (F003 on 12/18, F008 4/18).

**C7 (advection probe, specificity and held-out behaviour; not a pass/fail of the paper but a pre-registered requirement of the claim "probe tested"):**
 * Specificity: the probe must trigger on **0** in-scope clean copies and on **0** whole-dataset transpose/flip faulted copies of in-scope sets (a trigger there is a false decline; v4: 0 of 31 tuning sets and 0 of 9 held-out sets x 15 conditions).
 * Sensitivity: on the fresh advective control A1 (and A5 if used): reported, with the pre-registered prediction "declined as advective (`out of scope: advective`) with median R^2 gain >= 0.20 and margin >= 0.05" and the opposite outcome counted as a probe failure, **not** hidden. With a single advective family the result is "1 of 1" or "0 of 1" and is stated as such. It counts in C3 as a binding control.

### 3.3 Verdict rule
Unchanged for the four original criteria (PAPER-WORTHY / PARTIAL / NOT). C5-C7 are reported with their own pass/fail and do not change the mechanical verdict of C1-C4 (adding failing conditions to a rule that already failed twice would only move the target). The user may choose to promote C5/C6 into the verdict: decision in section 8.

---------------------------------------------------------------------------------------------------------------------------------

## 4. Procedure

**Phase A: v5 design on seen data only (no new data).**
1. Implement part A (and, if approved, part B and the optional F009 contiguous-block change) on a branch; unit tests first (synthetic Helmholtz mode, offset diffusion, sign-flipped diffusion, time-dependent wave).
2. Tuning stage: all seen datasets x conditions (`run_v4_design.py` pattern, new `run_v5_design.py`), before/after table; acceptance conditions in section 1.3 plus "no detection lost" for F002/F003/F008/F009. Tuning results: `bench/results/v5_tuning/`.
3. Write `frozen_thresholds_v5.json` (options, code SHA-256 of `rules.py`, `adapters.py`, `model.py`, `runner.py`, every preset JSON in `src/fieldlint/presets/`, `bench_code_sha256` of `run_bench.py` and the new runner), exactly as v4. The v2/v3/v4 frozen files stay untouched (v2-v4 hash checks will fail on v5 code; their stored results stand).

**Phase B: acquisition and data cards (no rule runs at all).**
4. Download the section-2 candidates into `data/_external_v5/` (one new empty directory per archive; run converters with `python -I`, scripts live outside the data directory); record sha256 and server md5/LFS ids.
5. Write `datasets_v5.md` (card per dataset: equation, BC, units, native/processed from generator code or paper, **sign convention from code**, mapping, role in {inscope, diagnostic, control}, and per-dataset predictions). Cards state **array shapes and value ranges that were read** and nothing else. Improvement over v4: **no F001 smoke test**; loadability is checked with a script that only calls h5py/numpy shape/dtype reads. If a card needs a statistic (as the Darcy min/max in v4), it is written into the card with a reason.
6. Stage mapping configs (`presets_v5_staging/*.json`, passed by path; not in `src/`). Decide the native list, the Helmholtz-source mapping and N before any run. Dataset size rule: first N <= 1000 samples in file order; minimum 200 (own 3D-ICE sets are pooled into one registered run per geometry or dropped if < 200; state which).

**Phase C: freeze, run once.**
7. Write `protocol_v5.md` (carry over v4 sections 5-8 and 10-11; add C5-C7, new fault kinds, the family-count rule). Freeze commit contains protocol, datasets cards, runner, report script, and precedes every held-out run (the user commits; I do not).
8. `run_v5_heldout.py` (copy of `run_v4_heldout.py`) refuses to re-run a name whose result file exists; at most **2 CPU jobs**; hash check of `frozen_thresholds_v5.json` at every start; the result file records git HEAD, uncommitted changes, SHA-256s, library versions, argv. Order: controls first (cheap), then in-scope, then fault-injection stages (C5, C6). Results are not inspected between runs except for crashes.
9. `report_v5.py` (mechanical tables for C1-C7, per-family and per-dataset clusters, CI by Clopper-Pearson). C2 and adjudication by hand.
10. Post-hoc analysis (anything not in the protocol) labelled POST HOC, results in `bench/results/v5_heldout_posthoc/`, cannot alter a criterion number.

**Runtime estimate** (from the v4 log: 13 runs, started 23:53, last start 00:57 + about 3 min, so about **1 h 10 min wall at 2 jobs**, of which 128 x 128 sets about 3.5 min and the 512 x 512 ELADO B1 the longest): v5 has about 10-12 in-scope runs with 14 + about 10 new conditions each (F004 doses and F009 arms add roughly 1.5x) plus 6-8 controls: **about 2-3 h wall at 2 jobs**, plus Phase A tuning (about 1-1.5 h) and downloads/conversions (hours, network bound; v4's Zenodo took about 40-60 KB/s single connection, ranged parallel downloads fixed it).

---------------------------------------------------------------------------------------------------------------------------------

## 5. Risks

1. **Fix overfit / second scope inference.** Part B is a second scope guess like the advection probe; its threshold rests on 35 diffusion sets (all one-sided families: Poisson/Darcy/thermal) against 2 oscillatory controls (1 steady, 1 wave). The gap is wide (6.6x) but there are only two oscillatory families behind it, and the fresh Helmholtz (H1) and wave (H2) are the first real test. If H1 straddles in a different way (point source, Neumann, mask) the V statistic may behave differently: the card must record that as a prediction, not a hope.
2. **Another mapping accident.** A new control whose invented source has a negative cell silently passes F005 while still failing F002 or F003: the v3 luck in section 1.1. Cards must say which rule the control is expected to hit.
3. **Scarcity of thermal application data.** T1 is unstructured/processed with an unknown licence and equation; T2/T3 are unverified; T5 is the authors' own, N < 200 per set and partly linted already. The thesis claim has to be sized to what is actually held out (section 7).
4. **Advection probe: one family.** Even a pass on A1 is "2 of 2 advective families". Its margins were chosen on one family; a fail on A1 is more informative than a pass.
5. **Rule-change exposure.** Every v5 code change except part A is optional; each increases the number of hypotheses. Keep the change list minimal and tuned only on seen data.
6. **Contamination of the held-out set by my own look.** The V-statistic thresholds were computed on seen data only; no held-out file was opened. The Phase-B card reading (README/paper/code) is allowed by the v4 precedent but must not include any statistic.
7. **Licences.** FlowBench CC-BY-NC-4.0; SISSA thermal-conduction no licence; Helmholtz-staircase CC-BY-4.0; IC-ThermBench S1 original licence. State licences in the data card and do not redistribute converted files for NC/unlicensed sets.
8. **C3 is brittle by construction** (one alarm on any control fails the verdict). A graded reading is available (decision below), but changing it after seeing the v4 failure is itself a choice the reader should see before data are run.
9. **Fault-injection design is partly ours.** C5/C6 thresholds come from tuning performance; hold them to what the tuning tables support (done above), nothing more.

## 6. What a pass or fail would let us claim (thesis wording, sized to evidence)

* **C1-C4 all pass on v5 (with at least one thermal application family and one fresh advective control):** "On N held-out datasets from G generator families, including T thermal application sets, fieldlint v5, frozen before the data were run, detects orientation faults with a clean-adjusted rate of x% (exact one-sided 95% lower bound y), is silent on clean in-scope data, declines out-of-scope steady and time-dependent controls including a thermal-convection and a Helmholtz control, and applies its operator model on the native-grid sets." Scope sentences that must stay: steady diffusion-type fields, generic Poisson/Darcy families plus T thermal sets, advection tested on k families, a single-author rule set.
* **C3 fails again:** "A second pre-registered round again failed the no-false-alarm criterion on out-of-scope controls; scope must be declared by the user." Report which rules and why; that is the second iteration of the same lesson and strengthens the negative-result paper.
* **C5 passes:** "exact-duplicate leakage across splits is detected at doses down to 1% (bound x); perturbed duplicates above 1e-5 relative are not detected (documented)." **C5 fails** only if the hash path is broken; unlikely.
* **C6 Arm A at about 0%, Arm B passes:** "pairing faults are detectable by F009 only on a contiguous block of size 64 of F008-applicable data, not on a 1000-sample dataset subsampled evenly." That is an honest, useful limit. If Arm B fails, F009 should be described as unvalidated.
* **C7:** a probe claim ("the advection probe declines advective data and does not decline diffusion data") may be made only if A1 (or A5) was run held-out and the specificity count is 0; otherwise the README keeps "no held-out test".
* **Not claimable under any outcome:** that fieldlint certifies a dataset correct; generalisation across "PDE datasets in general"; any statement about 3D-IC data not in the held-out set (the 6 tuning geometries are tuning).

## 7. What the thesis can say today about thermal evidence (regardless of v5)
Fixed-placement thermal evidence (IC-ThermBench S2-S5, 8 Therm-FM sets, 6 3D-ICE geometries) is tuning data. Held-out evidence on thermal data currently stands at zero sets. Even if v5 passes, the thermal claim is limited to the one or two thermal families that are actually held out.

## 8. Decisions needed from the user

1. **Fix scope:** approve part A only; A + B (recommended, evidence above); or A + B + the v5b generalisation of the WMP probe into the scope gate of F002/F003/F008/F009 (bigger, separate tuning stage).
2. **F009 contiguous-block sampling:** change the rule (tuned on seen data) or leave F009 as is and document "needs N <= 64 or `--head 64`".
3. **Helmholtz-type source mapping:** invented uniform source = 1 (harder, recommended, binding) or no source (easier); the other is a diagnostic.
4. **Thermal application data:** which of T1 (processed mesh, unknown licence), T3 (contact Heat-2D authors), T5 (own layout-varied solves, small N, partly seen by earlier linter) and T6 (independent-solver generated) may be used; whether a generated set counts as "held-out" in the thesis. Minimum N per set (200) vs the own sets (about 40-55 each: pool, relax, or drop).
5. **Advective control:** A1 FlowBench (CC-BY-NC) and/or A5 generated; accept that the advection probe will be a "n = 1 or 2" claim.
6. **C3 reading:** keep binary (0 alarms) as in v3/v4, or add a pre-registered graded verdict (e.g. PARTIAL if the only alarms are on a single rule across controls) before data are run.
7. **Promote C5 / C6 into the mechanical verdict or keep them as reported criteria.**
8. **Download budget confirmation (about 9-12 GB) and permission to range-read inside large files (Well wave sets, WaveBench) rather than skipping them.**
9. **Whether the earlier linter work also ran on `layout-geometry*` and the pilots** (I could not confirm; if yes, drop T5).

## Appendix: facts verified or not (2026-10-15)

Verified by page/API fetch: ELADO Zenodo record 20738464 file list (A1_os_test 3.98 GB, A1_os_train 3.98, A1_train 3.92, A1_tail 1.96, B1_train 3.92, B1_ramp 7.84, C1_train 0.97, C1_tail 0.49, D1_ramp 1.95, D1_test 0.98, E1_train 0.96 GB); The Well helmholtz_staircase licence CC-BY-4.0, 54.8 GB, test files 335,544,320 B each; acoustic_scattering_inclusions valid chunks 8,111,783,936 B each; rayleigh_benard valid files 1,082,130,432 B each; WaveBench Zenodo 8015145 single zip 75,625,669,111 B CC BY 4.0; SISSAmathLab/thermal-conduction 2.36 GB parquet, no licence text; CaltechDATA r5ga1-55d06 CC0, advection_diffusion.zip 670.3 MB; FlowBench CC-BY-NC-4.0 1.72 TB; GRIFDIR Zenodo record CC BY 4.0, 529 MB; 4TU coupled heat transfer CC BY 4.0, 3.27 GB; Mendeley rw9yk3c559 CC BY 4.0, only reader scripts listed; IC-ThermBench repo scopes (S2-S5 CC BY 4.0, 64 x 64 x 1, 4.6 GB Google Drive file; S1 on Google Drive, original licence). The ELADO arXiv abstract mentions only Poisson and Helmholtz with non-constant coefficients.
Not verified: array layouts of every new candidate, Heat-2D hosting, Mendeley npz availability, FlowBench temperature channel and steadiness, advection_diffusion output shape, licences of The Well collection members other than helmholtz_staircase, AirfRANS and OpenFWI details (memory only).
