# fieldlint v3: data cards (part of the pre-registration; DRAFT, not yet frozen)

Written 2026-10-09 in Phase 1 from documentation, file headers and key names only (see `protocol_v3.md` section 2 for exactly what was looked at).
Nothing here is a measurement. Every "Prediction" is derived from documentation and from the rule definitions in `README.md`
(`fieldlint --list-rules`), and is recorded before any rule other than F001 was run on these data. Paths are under `data/_external/`
(`data/` is git-ignored, checked with `git check-ignore`); the file lists with sizes and sha256 are in section 4.

Legend. **Native**: the stored grid is the grid the PDE was solved on (or the values were evaluated analytically on it), with evidence;
**processed**: resampled, interpolated between grids, or normalised after the solve. **u, source, k** are fieldlint's names:
`u` solution, `source` right-hand side, `k` coefficient (conductivity). Preset = `src/fieldlint/presets/<name>.json` (mapping only).
"Run name" is the name used by `run_v3.py`.

## 1. Registry and acquisition status

| # | dataset | role | obtained | samples used | grid | native? | run name |
|---|---|---|---|---|---|---|---|
| 1 | PDEgym Poisson-Gauss | inscope | yes, 2.6 GB | first 1000 of 20000 | 128 x 128 | native (documented) | `pdegym_poisson_gauss` |
| 2a | CNO RPB Poisson | inscope | yes (zip 2.4 GB) | first 1000 of 2048 | 64 x 64 | native (analytic solution sampled) | `cno_poisson` |
| 2b | CNO RPB Darcy | inscope | yes | all 512 | 64 x 64 | processed / undocumented (FNO scripts interpolate between grids) | `cno_darcy` |
| 3a | UCSD / LLNL MTNN Poisson (`Poisson4.npz`) | inscope | yes (tar 2.6 GB, after repeated restarts) | first 1000 of 10000 | 32 x 32 | native (FEM vertex grid) | `ucsd_poisson` |
| 3b | UCSD / LLNL MTNN convection-diffusion (`ConvDiff1.npz`) | control (advection) | yes, same tar | first 1000 of 10000 | 32 x 32 | native | `ucsd_convdiff` |
| 3c | UCSD / LLNL MTNN Darcy files | not run | yes (same tar), but the files hold permeability `U` and total flux `Q` only, no solution field | | | | |
| 4a | DiffusionPDE-normalized Poisson | inscope (denormalised) + diagnostic (as released) | yes, one shard | first 1000 of 1852 in the shard (50000 in the set) | 128 x 128 | solver grid (FD) but standardised and mollifier-processed | `dpde_poisson_denorm`, `dpde_poisson_asreleased` |
| 4b | DiffusionPDE-normalized Darcy | inscope (denormalised) + diagnostic (as released) | yes, one shard | first 1000 | 128 x 128 | processed (FNO scripts; standardised) | `dpde_darcy_denorm`, `dpde_darcy_asreleased` |
| 5 | lu-group deeponet-fno Darcy (rectangular, triangular) | not run | **no: data are on a Yale SharePoint that answers 403 to a non-interactive client; not bypassed** | | | the rectangular case is Li et al.'s `piececonst` set, which v2 already used | |
| 6 | ELADO Poisson A1_test | inscope | yes (3.9 GB zstd, 4.2 GB h5) | all 1000 | 512 x 512 | native (FEM vertex grid) | `elado_a1_test` |
| 7 | HSL / TFRD heat-source-layout | not run | **no: full data are on Baidu netdisk (account needed); the GitHub repos hold 10-16 tiny samples per configuration, far fewer than 200; the generator exists but generating it was excluded by the brief** | | | | |
| C1 | CNO RPB Wave | control | yes | 1000 (t = 5 groups) | 64 x 64 | analytic | `cno_wave` |
| C2 | CNO RPB Allen-Cahn | control | yes | all 512 | 64 x 64 | | `cno_allen_cahn` |
| C3 | CNO RPB Navier-Stokes (shear layer) | control | yes | 1000 | 64 x 64 | | `cno_navier_stokes` |
| C4 | CNO RPB smooth transport | control | yes | 1000 | 64 x 64 | analytic | `cno_cont_transport` |
| C5 | CNO RPB discontinuous transport | control | yes | 1000 | 64 x 64 | analytic | `cno_disc_transport` |
| C6 | PDEgym Wave-Gauss | control | yes, subset by HTTP range requests (256 of 3504 trajectories of `solution_0.nc`; full chunk would be 3.7 GB) | 256 | 128 x 128 x 16 t | FD at 128^2 | `pdegym_wave_gauss` |
| C7 | ELADO Helmholtz E1_test | control | yes | all 1000 | 256 x 256 | native (FEM) | `elado_helmholtz_e1` |
| C8 | DiffusionPDE-normalized Helmholtz | control | yes, one shard | first 1000 | 128 x 128 | standardised | `dpde_helmholtz` |

Total on disk for v3: about 23 GB (limit was 30 GB). The CNO Euler airfoil file was unpacked but is **not** registered (steady hyperbolic flow, not in the
requested control list and the README gives no scope flag that could declare it).

## 2. Cards

### 2.1 PDEgym Poisson-Gauss  (`pdegym_poisson_gauss`, in scope)

* **Field / application**: ML-for-PDE foundation-model benchmark (ETH CAMLab); diffusion of a source made of Gaussians, a generic steady diffusion problem.
* **Source, licence, paper**: Hugging Face `camlab-ethz/Poisson-Gauss`, `Poisson-Gauss.nc` (NetCDF4/HDF5), commit f728b1f5, CC-BY-NC-4.0 (README YAML). Herde, Raonić et al., "Poseidon: Efficient Foundation Models for PDEs", NeurIPS 2024, arXiv 2405.19101.
* **Files**: section 4 (sha256 equals the Hugging Face LFS id `4ab5952e...`).
* **Equation and BC as documented**: -Laplace(u) = f in (0,1)^2, homogeneous Dirichlet; f = sum_{i=1}^N exp(-((x-mu_x,i)^2+(y-mu_y,i)^2)/(2 sigma_i^2)), N ~ Geom(0.4), mu ~ U[0,1], sigma ~ U[0.025,0.1]; 20000 solutions with FEniCS FEM (Poseidon App. B.2.14). k = 1 (not stored).
* **Native or processed**: *native per documentation*: App. B.2.14 and App. D.4 state the solves are at 128^2 resolution with FEniCS (solver run times quoted "for a resolution of 128^2"). Not independently verified (how the FEM vertex values were mapped to the 128-grid is not documented). Values are stored raw: float32, `source` in [0, 2.37] and `solution` in [0, 0.0269] over the first 200 samples, not normalised (Poseidon normalises inside its loader).
* **Array mapping**: `solution` -> u, `source` -> source, k absent (uniform 1), layout (sample, x, y) = `NHW`, no dirichlet mask, splits none (the dataset's own 19640/120/240 split is not used because the first 1000 samples are all training samples; the bench assigns positional splits as in v2). Preset `v3_poisson_gauss`.
* **Role and reason**: in scope: documented steady Poisson with non-negative source and zero Dirichlet, source varies per sample, solved at the stored resolution.
* **Prediction (from documentation)**:
  F001 ok. F002 applicable and ok (source >= 0; the Gaussian tails make the source mask cover most of the grid, so the rule is weakly discriminating). F003 applicable (source varies), expected ok. F004 ok. F005 ok (u >= 0, as the documented equation requires). F006 info, expected near-linear is NOT expected (the map f -> u is linear, so a ridge fit will do well: expect high held-out R^2, info only). F007 skipped (no flux). F008/F009 applicable: a P1 FEM Laplacian on a regular mesh is the 5-point stencil, source smooth over >= 3 cells, so the operator fit should explain most of the variance; expect clean median R^2 well above 0.3 and detection of transpose / flips. F008 uses source only (k uniform).
  Faults: transpose / flips detected by F003 and F008; rolls by F008 (if R^2 is high).

### 2.2 CNO Representative PDE Benchmarks, Poisson  (`cno_poisson`, in scope)

* **Field**: operator-learning benchmark (ETH CAMLab, "RPB"); analytic multiscale Poisson problem.
* **Source, licence, paper**: Zenodo record 10406879, DOI 10.5281/zenodo.10406879, `data.zip` (md5 268a51a9c90f4ee01cd0c1495664ec25 verified), CC-BY-4.0. Raonić, Molinaro, De Ryck, Rohner, Bartolucci, Alaifari, Mishra, de Bézenac, "Convolutional Neural Operators for robust and accurate learning of PDEs", NeurIPS 2023, arXiv 2302.01178. Loader: `CNO2d_classic/Problems/CNOBenchmarks.py` in github.com/camlab-ethz/ConvolutionalNeuralOperator (read, not run).
* **Array semantics (from the loader)**: per-sample HDF5 groups `Sample_<i>` with datasets `input` and `output`, each 64 x 64 (reshaped (1,64,64)); file-level constants `min_inp`, `max_inp`, `min_out`, `max_out` used by the loader to min-max normalise **at load time**; the stored arrays are raw (checked: `input` in [-4.16, 3.54] with `min_inp` -4.16). Grid helper in the loader: `torch.meshgrid(linspace(0,1,s), linspace(0,1,s))` (axis 0 = x).
* **Equation and BC as documented**: -Laplace(u) = f in D = (0,1)^2, u|dD = 0 (eq. 4.1); f = (pi/K^2) sum_{i,j=1}^K a_ij (i^2+j^2)^(-r) sin(pi i x) sin(pi j y), K = 16, a_ij ~ U[-1,1]; exact solution u = (1/(pi K^2)) sum a_ij (i^2+j^2)^(r-1) sin sin (SM C.3.1). Sampled at 64 x 64.
* **Native or processed**: native in the sense that there is no discretisation: the analytic solution is evaluated on the stored grid (no resampling, no stored normalisation). Whether the grid includes the boundary nodes (`linspace(0,1,64)` in the loader suggests yes) is only inferred.
* **Mapping**: `input` -> source, `output` -> u, k absent (1), `NHW`. Preset `v3_cno_poisson`. Subset: first 1000 of 2048 `Sample_*` groups in numeric order (the file holds 2048; the loader uses 1024 + 128 + 256 of them).
* **Role**: in scope: documented steady Poisson, Dirichlet 0.
* **Prediction**: F001 ok. **F002 skipped** (the source changes sign, i.e. it has sinks). F003 applicable, expected ok (u is a smoothed version of f, so the smoothed-source correlation is positive). F004 ok. F005 ok. F008/F009: applicable if the stencil error of mode numbers up to 16 on a 64-grid (up to about 10% in the symbol) leaves R^2 above 0.3; I expect it to (R^2 of order 0.9) but the node/cell convention of the grid is unknown, so this is the least certain F008 prediction among the Poisson sets. Faults: F003 detects transpose / flips; F008 likewise if applicable.

### 2.3 CNO RPB Darcy  (`cno_darcy`, in scope)

* **Source**: same Zenodo record and loader as 2.2, `Darcy_64x64_IN.h5` (512 samples, groups `sample_<i>` lower case, plus `grid` (64,64,2), `min_inp` 3, `max_inp` 12, `min_out` -0.0230, `max_out` 0.0).
* **Equation and BC as documented**: -div(a grad u) = f in D, u|dD = 0, f = 1, a = psi(GP) with psi = 12 on the positive and 3 on the negative part, squared-exponential GP with length scale 0.1 (in-distribution), "the experimental setup is the same as the one presented in [FNO]" (CNO paper eq. 4.7-4.8, SM C.3.6, 64 x 64).
* **Native or processed**: **processed / undocumented**. The paper gives the resolution (64 x 64) and says the set-up equals the FNO one; the FNO generating script (`solve_gwf.m`, Li et al.) interpolates the coefficient from cell centres to nodes with splines, solves with finite differences, and interpolates the solution back with splines (read in github.com/lu-group/deeponet-fno `data/darcy_rectangular_pwc/solve_gwf.m`). That interpolation means the stored arrays are not the solver's own discrete solution. The `grid` array is nodal (`linspace(0,1,64)`). This is evidence for "processed", not proof (the CNO team's own generation script is not in the repository).
* **Observed sign convention (min/max only)**: the stored `output` is <= 0 everywhere (min -0.0230, max 0.0), while the documented equation with f = 1 gives u >= 0. Either the output is stored as -u or the PDE has a sign flip; the loader takes it as is. This was the only observation in Phase 1 that was surprising.
* **Mapping**: `input` -> k (3 / 12), `output` -> u as stored (NOT sign-corrected), uniform source 1.0, ambient 0, `NHW`. Preset `v3_cno_darcy`.
* **Role**: in scope: documented steady Darcy.
* **Prediction**: F001 ok. F002, F003 skipped (uniform source). F004 ok. **F005 flags u below the ambient 0 with a non-negative source**; under protocol section 7 (P1) that finding is a TRUE fault of the release (u <= 0 contradicts u >= 0), so it is not a false positive. F008/F009: expected **skipped** (processed grid, same family as the resampled FNO Darcy file of v2 that explained R^2 = 0.03 and the native 421 file that explained 0.31); if applicable it is a surprise. Since the source is uniform, a skip of F008 means that transposed / flipped k is invisible to every rule, so any-rule detection of whole-dataset transpose / flips is expected to be near 0 by the true detection logic. Caveat for scoring: v2 scores a rule as detecting a fault when it is flagged on the faulted copy, without subtracting a flag that is already present on the clean copy. Here F005 is expected to fire on every copy (clean and faulted), so the v2 any-rule count would credit it with detecting every fault. The protocol therefore adds one scoring rule (section 6): a fault is credited to a rule only if that rule is not already flagged on the clean copy of the same dataset (the v2 tuning tables are unaffected because no clean flag existed except the F004 duplicates of IC-ThermBench S2).

### 2.4 DiffusionPDE-normalized, Poisson (`dpde_poisson_denorm` in scope; `dpde_poisson_asreleased` diagnostic)

* **Field**: generative inverse-problem benchmarks (diffusion-model priors for PDEs).
* **Source, licence, paper**: Hugging Face `jcy20/DiffusionPDE-normalized`, `poisson_hf/` (Arrow shard 0 of 27: 1852 rows; the set has 50000), commit 3ededc4f, CC-BY-NC-SA-4.0. Processing: Huang et al., "FunDPS" arXiv 2505.17004 (App. D, E), scripts in github.com/neuraloperator/FunDPS `utils/`. Original data: Huang, Yang, Lin et al., "DiffusionPDE: Generative PDE-solving under partial observation", NeurIPS 2024, arXiv 2406.17763 (App. B.1).
* **Normalisation (documented)**: the dataset README says "we normalized the datasets to zero mean and 0.5 standard deviation to follow EDM's practice". `poisson_hf/metadata.json` gives per-channel `mean` [0, 9.672e-06], `std` [0.2919494, 0.00417478], `min`, `max`, `shape` [2, 128, 128], `num_samples` 50000. Checked in the sibling Darcy set: stored channel 0 is exactly -0.5 / +0.5 for a = 3 / 12, which equals (x - 7.5)/4.5 * 0.5 with the metadata mean and std, so the inverse is x_raw = x * (2 std) + mean. The stored array is `data` (N, 2, 128, 128) float64 (channels: 0 = a, 1 = u by the DiffusionPDE convention, channel order is not stated in the Hugging Face card; the paper says only "concatenated on the channel dimension").
* **Equation and BC as documented**: u = 0 on the boundary; Poisson written as nabla^2 u + k^2 u = a with k = 0, i.e. **nabla^2 u = a** (note the sign: -Laplace(u) = -a); a = mu a Gaussian random field; second-order finite differences; no-slip boundary "enforced by multiplying by the mollifier sin(pi x1) sin(pi x2)"; "both a and u have resolutions of 128 x 128" (nodes vs cells not stated).
* **Native or processed**: the arrays are standardised (processed). The variant `..._denorm` undoes the documented standardisation with an affine map in the preset (the `affine` field of the frozen adapter), giving the arrays on the solver's grid. The mollifier is a possible post-solve modification of u; whether it multiplies a, u or the boundary only is not stated.
* **Mapping**: channel 1 -> u, channel 0 -> source, k absent, layout `NCHW` with channel index. `v3_dpde_poisson_denorm`: u affine [2 std, mean] = [0.00834956, 9.672e-06], source affine [0.5838988, 0] . `v3_dpde_poisson` (as released): no affine.
* **Role**: `..._denorm`: in scope (documented steady Poisson); `..._asreleased`: diagnostic. Reason for two variants, decided before any run: a rule premise (an absolute level / sign convention) is violated by the standardisation itself; running both lets the report say what a user who loads the release as is would see, without counting that against the rules.
* **Prediction**: F001 ok. F002 skipped (sign-changing source: sinks). F003 applicable; **risk**: with nabla^2 u = a, u is *anti*-correlated with the smoothed a, so the stored orientation may score worse than a transposed or flipped one on the plain correlation (the Therm-FM HS_SC pattern); the band-passed condition may protect it. A false alarm here is a pre-registered possibility, to be adjudicated by section 7 (it would be a false positive unless documentation confirms a transposed storage, which it does not). F004 ok. F005: ok for `..._denorm`; for as released, level checks are meaningless (zero-mean u) and may produce info/warning. F008/F009: applicable only if the mollifier did not alter u; either outcome is registered as consistent with the documentation. Faults: F003 detects when applicable.

### 2.5 DiffusionPDE-normalized, Darcy (`dpde_darcy_denorm` in scope; `dpde_darcy_asreleased` diagnostic)

* **Source**: as 2.4, `darcy_hf/` (shard 0 of 27, 1852 rows; first 1000 used). `metadata.json`: `mean` [7.5, 0.00569201936], `std` [4.5, 0.00379030361], `min` [3.0, -0.287...], `max` [12.0, 0.1177...].
* **Equation and BC as documented**: -div(a grad u) = q, q = 1, u = 0 on the boundary, a = 12 where mu >= 0 else 3, mu ~ N(0, (-Laplace + 9I)^-2), FEM with the FNO released scripts; 128 x 128 (DiffusionPDE Sec. 4.1, App. B.1).
* **Native or processed**: processed (same FNO scripts as 2.3 plus standardisation). Stored channel 0 equals +-0.5 exactly (min/max check), so the standardisation is the documented one.
* **Mapping**: channel 0 -> k, channel 1 -> u, uniform source 1.0; `..._denorm`: k affine [9.0, 7.5] (a returns to {3, 12}), u affine [0.00758060722, 0.00569201936]; as released: no affine (k in {-0.5, +0.5}). Presets `v3_dpde_darcy_denorm`, `v3_dpde_darcy`.
* **Prediction**: F002/F003 skipped (uniform source). `..._asreleased`: **F005 flags k <= 0 (error)**, expected by construction (diagnostic role, not counted). `..._denorm`: F005 ok. F008/F009: expected skipped (FNO-script lineage, as 2.3); any-rule detection of transposed / flipped k then near 0.
* Note: the metadata `min` of u is negative (-0.287) although the first 1000 rows reach only about 0 after the inverse map; not investigated (a statistic over the 50000-set that Phase 1 did not examine).

### 2.6 ELADO Poisson, A1_test  (`elado_a1_test`, in scope)

* **Field**: neural-operator failure-mode benchmark for elliptic PDEs (Poisson with non-constant coefficient).
* **Source, licence, paper**: Zenodo record 20738464 ("ELADO Operator Learning Dataset Part 1"), CC-BY-4.0, `A1_test.h5.zst` (md5 8f5fcf49666b7bfa50a57e53948e930b verified). Ehebrecht, Scharle, Atzmueller, "ELADO: Elliptic PDE Assessment Datasets for Operator Learning", arXiv 2606.20771 (2026-06-18). Generator scripts: `scripts.tar.zst` in the same record (`poisson_dirichlet.py`, dolfinx; read, not run).
* **Equation and BC as documented**: -div(a grad u) = 1 on the unit square, u = 0 on the boundary; a in [a_min 0.01, a_max 1.0], A1: sigma_s0 = 0.04, sigma_jump = 0.06, tau = 0, stationary; mesh 511 x 511 cells, coefficient resolution 512, P1 elements (paper Sec. 3, Eq. 9; script: CG1 coefficient and solution, `create_unit_square(N_cells, N_cells)` with `N_cells = input_field.shape[0] - 1`).
* **Native or processed**: **native**: the stored arrays are 512 x 512 float64, the vertex grid of the 511-cell mesh (the paper trains on a 4x subsample, which is not applied to the released files; script `mesh2array` returns the vertex values reshaped to the square). No normalisation is stated and none is visible in the key set. The `.h5` is the byte-for-byte zstd-decompressed file.
* **Array mapping**: HDF5 keys `x` (1000, 512, 512) = coefficient a, `y` (1000, 512, 512) = u (names inferred from the generator and ranges: x in [0.024, 0.98], y in [0, 0.19] over the first 3 samples, no further statistics); `NHW`; x -> k, y -> u, uniform source 1.0. Preset `v3_elado_poisson`. The Zenodo file has no split labels.
* **Role**: in scope: documented steady diffusion, k varies, source uniform, native grid.
* **Prediction**: F001 ok. F002/F003 skipped (uniform source). F004 ok. F005 ok (u >= 0, a > 0). F008/F009 applicable: P1 FEM with a nodal coefficient on a regular triangulation is a 5-point-like stencil with an arithmetic-type face mean, and the fitted face mean should find it; expect clean median R^2 near 1. If so, F008 detects transposed / flipped / rolled k (k is the only varying input). This is the sharpest test of "F008 works on native FEM data" in the set.

### 2.7 UCSD / LLNL MTNN steady diffusion  (`ucsd_poisson`, in scope; `ucsd_convdiff`, control)

* **Field**: DOE national-laboratory data for multilevel training of neural networks (generic steady heat conduction on the unit square).
* **Source, licence, paper**: UC San Diego Library Digital Collections, DOI 10.6075/J0HM58MK, object bb1852369g, "Partial Differential Equations (PDE) and Solutions: Darcy Flow, Convection-Diffusion, Poisson Problems", Ponce, Li, Mao, Fairbanks (2022), CC-BY-4.0. File `bb1852369g_2_1.tar` (2 612 940 800 bytes, matches the server's Content-Length); members used: `data/poisson_data/Poisson4.npz` and `data/convdiff_data/ConvDiff1.npz`, extracted unchanged by `convert_v3.py ucsd:...` (a byte copy of the tar member). The README refers to the MTNN repository (github.com/LLNL/MTNN) but contains no equation or boundary-condition text.
* **Arrays (README + key names)**: `Kappa` (10000, 32, 32) heat conductivity, `F` (10000, 32, 32) heat source, `U` (10000, 32, 32) numerical solution, `X`, `Y` (32, 32) mesh coordinates in [0, 1] (X varies along the last axis, Y along the first), `FluxX`, `FluxY` (10000, 32, 32) heat flux; float64. Min/max over the whole Poisson file: Kappa in [0.1, 2.1], F in [0.36, 31.98] (positive), U in [0, 2.72].
* **Equation and BC as documented**: "steady-state ... pure diffusion problems ... on the 2-D unit square ... discretized by finite element methods (FEniCS/DOLFIN)"; inputs = diffusion coefficients, mesh point coordinates and heat source. **No boundary condition and no explicit equation are documented**; the conservative reading is -div(Kappa grad U) = F. U has minimum 0.0, which suggests homogeneous Dirichlet values on at least part of the boundary, but that is an inference, so the preset declares no ambient and no Dirichlet mask.
* **Native or processed**: **native** by documentation (FEM vertex values at the mesh coordinates `X`, `Y`, a 32 x 32 vertex grid of the unit square). Nothing in the README suggests rescaling.
* **Mapping**: `Kappa` -> k, `F` -> source, `U` -> u, layout `NHW`; first 1000 samples (`head 1000`). Presets `v3_ucsd_poisson`, `v3_ucsd_convdiff`. Splits: none. The grid is small (32 x 32).
* **Role**: Poisson: in scope. ConvDiff1: control: the README says "(convection-) diffusion"; the convective velocity is neither stored nor documented, and fieldlint's README excludes advective problems. It shares `Kappa` and `F` with the Poisson file (identical min/max), so only `U` differs.
* **Prediction (Poisson)**: F001 ok. F002 applicable (source > 0 everywhere), expected ok (the source mask is the whole grid). F003 applicable (source varies, non-negative), expected ok. F004 ok. F005: no ambient declared, so the below-ambient check cannot fire; expected ok. F008/F009 applicable: FEM on a regular mesh with nodal Kappa and nodal F; both source and k vary, so the operator fit has the most information of any set here; expect a high clean median R^2. Detection of transpose / flips by F003 and F008, roll faults by F008 (a 1-3 cell roll on a 32-point axis is a large shift, so roll detection should be easier here than on 128+ grids).
* **Prediction (ConvDiff1, control)**: steady, so no time flag exists. If advection enters the stored U, the symmetric 5-point operator fits worse than for Poisson. Possible outcomes: F008 skipped by the gate (declined) or F008 applicable at a lower R^2 (no alarm on clean data either way). F002 may flag if the maximum of U is advected away from the source: a false alarm is possible and is pre-registered as a risk.

### 2.8 Controls

All controls are **out of scope by documentation**; the mapping is "a user's most plausible reading": source = the input field, u = the output field (as in v2's control stage). Each time-dependent control is run **twice**, with the README's scope declaration (`time_dependent: true`, the binding run for C3) and without it (reported as C3-strict).

| control | file | equation (as documented) | mapping | prediction |
|---|---|---|---|---|
| `cno_wave` | `WaveData_64x64_IN.h5`, groups `*_t_5` | wave equation u_tt = c^2 Laplace u, c = 0.1, exact sine-series solution (CNO SM C.3.2), u0 -> u(t=5) | input -> source, output -> u | flagged run: F002/F003/F008/F009 skipped "out of scope: time-dependent". Unflagged: F002 skipped (sign-changing source); F003 / F008 uncertain (u(5) is not a diffusion of u0), F008 expected skipped |
| `cno_allen_cahn` | `AllenCahn_64x64_IN.h5` | Allen-Cahn, nonlinear reaction-diffusion, u0 -> u(T) with u in about [-1, 1] | same | flagged: skipped. Unflagged: F002 skipped (sign-changing), F005 may warn on levels |
| `cno_navier_stokes` | `NavierStokes_64x64_IN.h5` | incompressible NS shear layer, periodic | same | as above |
| `cno_cont_transport` | `ContTranslation_64x64_IN.h5` | linear transport, velocity (0.2, -0.2), smooth u0 (values in [0, 1]) | same | flagged: skipped. Unflagged: the source is non-negative, so F002 and F003 are applicable and a displaced bump can trigger them: a false alarm is *possible* without the flag |
| `cno_disc_transport` | `DiscTranslation_64x64_IN.h5` | transport of a discontinuous u0 | same | as above |
| `pdegym_wave_gauss` | first 256 trajectories of `camlab-ethz/Wave-Gauss` `solution_0.nc` (N,16,128,128) float32 and `c_0.nc` (N,128,128), CC-BY-NC-4.0; Poseidon App. B.2.10 | u_tt - c(x)^2 Laplace u = 0, absorbing boundaries, FD at 128^2, T = 1; u0 sum of Gaussians; c = c0 + sum of 4 Gaussians (1500-2500 base) | solution[:,0] -> source, solution[:,15] -> u, c -> k (the file has 16 time points, the README says 15) | flagged: skipped. Unflagged: F002/F003 applicable (u0 >= 0) and may alarm |
| `elado_helmholtz_e1` | `E1_test.h5` | Helmholtz with a wavenumber field a in [16, 30] and a fixed centred Gaussian source (not stored), u = 0 on the boundary, P2 elements, 256 x 256; stored y = |u| (paper Eq. 10) | x -> k, y -> u, uniform source 1 (same preset logic as the Poisson sibling) | steady, so no time flag exists. F002/F003 skipped (uniform source). F005 ok (|u| >= 0). F008/F009 are the open question: the operator model -div(a grad u) is the wrong operator for this field, so a decline (skipped by the R^2 gate) is expected and a flag would be a false alarm |
| `ucsd_convdiff` | `ConvDiff1.npz` (MTNN), first 1000 | steady convection-diffusion on the unit square (convective velocity undocumented); see 2.7 | Kappa -> k, F -> source, U -> u (as the Poisson sibling) | steady, no time flag; see 2.7 |
| `dpde_helmholtz` | `helmholtz_hf` shard 0, as released (standardised) | nabla^2 u + k^2 u = a, k = 1, u = 0 on the boundary, a GRF; FD 128^2 (DiffusionPDE) | channel 0 -> source, channel 1 -> u | F002 skipped (sign-changing); F003: same anti-correlation risk as 2.4 if the convention is the same; F008 expected skipped |

The CNO Euler airfoil file was not registered (see section 1).

## 3. Datasets considered and their status

**UCSD/LLNL MTNN** (DOI 10.6075/J0HM58MK): obtained; see card 2.7. The landing page was reachable (HTTP 200, not bot-blocked); the direct file link `/dc/object/bb1852369g/_2_1.tar/download` works without a login but does not support HTTP range requests and the server stalled long transfers, so the 2.6 GB tar needed a resume-by-skip downloader and several restarts. The Darcy-flow files in the same tar (`match_pde_data_*`, `train_data_32`, `test_data_32`, `coord_fix_*`) hold the permeability grid `U` and total outflow flux `Q` (README), not a solution field, so they cannot be mapped to (u, source, k) and are not used.

**lu-group/deeponet-fno** (Lu et al., CMAME 393, 114778, 2022; CC-BY-NC-SA-4.0). Data links are on a Yale SharePoint folder; an anonymous request is answered by a redirect to the OneDrive web UI and HTTP 403 ("you must first browse to the web site and select the option to login automatically"). Not downloaded; no bypass attempted. The rectangular piecewise-constant case is the same Li et al. data as `fno_darcy`/`fno_darcy_native` of v2 anyway.

**ELADO**: public, Zenodo 20738464 (CC-BY-4.0): A1/A1-os/B1/C1/D1/E1 (train, test, tail, ramp), 0.5-7.8 GB each, h5 + zstd. Only `A1_test` (Poisson, in scope) and `E1_test` (Helmholtz, control) were taken.

**HSL / TFRD (heat-source layout temperature fields)**: Chen et al., "A Machine Learning Surrogate Modeling Benchmark for Temperature Field Reconstruction of Heat-Source Systems" (TFRD; arXiv 2108.08298, Sci. China Inf. Sci. 2022) and Chen et al. HSL-TFP benchmark (arXiv 2103.11177, code github.com/idrl-lab/supervised_layout_benchmark). The data are announced on Baidu netdisk (links and passwords in the repo READMEs; an account is needed), so they were not downloaded. The repos contain only a handful of samples (10 train + 8 test per configuration for the HSL repo; 16 train + 6 test per boundary-condition case for shendu-sw/TFR-HSS-Benchmark), below the 200-sample minimum. The generator (shendu-sw/recon-data-generator, FEniCS, LGPL-3.0) exists; generating the data ourselves was excluded by the brief. **Not obtained.**

**PDEgym Helmholtz** (`camlab-ethz/Helmholtz`): the repository lists no files; not used. DiffusionPDE-normalized `helmholtz_hf` is used as a control instead.

## 4. Files, sizes and sha256

The sha256 values were computed after download (and equal the host's value where the host publishes one: Hugging Face LFS ids for `Poisson-Gauss.nc` and the three Arrow shards; Zenodo md5 for `data.zip`, `A1_test.h5.zst`, `E1_test.h5.zst`). URLs:

* `https://huggingface.co/datasets/camlab-ethz/Poisson-Gauss/resolve/main/Poisson-Gauss.nc`
* `https://zenodo.org/api/records/10406879/files/data.zip/content` (record 10406879)
* `https://huggingface.co/datasets/jcy20/DiffusionPDE-normalized/resolve/main/{poisson_hf,darcy_hf,helmholtz_hf}/data-00000-of-00027.arrow` (+ `metadata.json`, `README.md`)
* `https://zenodo.org/api/records/20738464/files/{A1_test.h5.zst,E1_test.h5.zst,scripts.tar.zst}/content` (record 20738464)
* `https://huggingface.co/datasets/camlab-ethz/Wave-Gauss/resolve/main/{solution_0.nc,c_0.nc}`: read by HTTP range requests, sources not stored; Hugging Face LFS ids: see `wave_gauss/converted/wave_gauss_first256.MANIFEST.json` (`sources_lfs`)
* `https://library.ucsd.edu/dc/object/bb1852369g/_2_1.tar/download`, `.../_1_1.md/download`

The `converted/` files are produced by `bench/convert_v3.py` (lossless; each has a `*.MANIFEST.json` with the source and output hashes). `extracted/` is `data.zip` unzipped.


**1. PDEgym Poisson-Gauss (`poisson_gauss/`)**

| file (under `data/_external/`) | bytes | sha256 |
|---|---|---|
| `poisson_gauss/Poisson-Gauss.nc` | 2623698923 | `4ab5952ef83493725987f6ca958b1ea5cdb0ce1aac39e2903b60793360057b2a` |
| `poisson_gauss/README.md` | 591 | `e1ec91cd963a373547f8d5f05699650e0fef4859be05edcee126e477db89f486` |
| `poisson_gauss/converted/poisson_gauss_first1000.npz` | 131074006 | `7ce8c8a37c1f153297c14371d96b5224a87d11c28804d9767b9bb2c64ff13403` |

**2. CNO Representative PDE Benchmarks (`cno_rpb/`)**

| file (under `data/_external/`) | bytes | sha256 |
|---|---|---|
| `cno_rpb/data.zip` | 2379635051 | `3d74dd473e3256bc1e6704cb7aa696c635f4497eda5e62be2f62deb8bba887cb` |
| `cno_rpb/extracted/data/PoissonData_64x64_IN.h5` | 137685064 | `4bdea48f8e0a2cc30ea2a9916014a306833fad000846add2465786a473f77f3a` |
| `cno_rpb/extracted/data/Darcy_64x64_IN.h5` | 34457976 | `e460c2d997046f3a514866299f97be042eb9d247910603cc88007acef50d7a91` |
| `cno_rpb/extracted/data/WaveData_64x64_IN.h5` | 1344373944 | `51816a9ee85c5611d2600c8d17da3ef5d8dfbbb027b2b6c2b9861f443f3d9824` |
| `cno_rpb/extracted/data/AllenCahn_64x64_IN.h5` | 34426408 | `077e3119006432a46f573deebfeb0f81727c9ce65f0fa82f5930be0b71fdefd6` |
| `cno_rpb/extracted/data/NavierStokes_64x64_IN.h5` | 35286656 | `c8c8fd787950bdf91d6b438603549a54283e789ff0ec342660e19205b57b63f2` |
| `cno_rpb/extracted/data/ContTranslation_64x64_IN.h5` | 68841088 | `1c11687184b365e895c3095a96f9677b3e9a40de376cb75a9011a48dca61fa76` |
| `cno_rpb/extracted/data/DiscTranslation_64x64_IN.h5` | 68841088 | `449621e5a97a4b2fcb5180b042ed2a49f328c731beab5f3e7d2e1370c7e759ca` |
| `cno_rpb/converted/PoissonData_64x64_IN_first1000.npz` | 65538660 | `acf3da86f41116380aae46826326598b963b0848c64f29770039ab4bf4e66dd0` |
| `cno_rpb/converted/Darcy_64x64_IN_first512.npz` | 33589034 | `5527ef8d320d8cc61a20850be1c56392e7c7a65b5b3ca5424397e2e6bfeceecb` |
| `cno_rpb/converted/WaveData_64x64_IN_t_5_first1000.npz` | 65538596 | `2f4fc309cb41dde6da4c236ff90ea2dbcbab41ecad52f898256ab58cbb331d38` |
| `cno_rpb/converted/AllenCahn_64x64_IN_first512.npz` | 33556764 | `e102133997b837c30bd5579bfbf53fbdcbbf280a30754fa44b55673814956398` |
| `cno_rpb/converted/NavierStokes_64x64_IN_first1000.npz` | 32768508 | `a02bdfe97cbaea5b91174968016bcb16a27dc1139109c6523e5bb5a1c732152d` |
| `cno_rpb/converted/ContTranslation_64x64_IN_first1000.npz` | 65536508 | `dc3b9440d35829b32153be44d30cdb23d4f8fdb97c09c654c90af02ea4dae8ea` |
| `cno_rpb/converted/DiscTranslation_64x64_IN_first1000.npz` | 65536508 | `a91108aea0b724850a9553568b10dcbce8e83d08e904af37f5adebd824684a7c` |

**3. DiffusionPDE-normalized (`diffusionpde_norm/`)**

| file (under `data/_external/`) | bytes | sha256 |
|---|---|---|
| `diffusionpde_norm/README.md` | 529 | `65fd4ea3c5b1a3eb633e1c92c546646109a3800f9029da9ba62a3430d4358efa` |
| `diffusionpde_norm/poisson_hf/data-00000-of-00027.arrow` | 487418216 | `94f576312ac5c65b9ab83fb716b0115adc4f66fd1ae8875c9af70fe255ab25c4` |
| `diffusionpde_norm/poisson_hf/metadata.json` | 347 | `940d5e1fa59284c369d5d73641a254bac9926abe5ea517316c1bf29e641a6a07` |
| `diffusionpde_norm/darcy_hf/data-00000-of-00027.arrow` | 487418216 | `40af41c32fc6acfe831f9708afcd07bf3b43857a9520b381035661eccdf2a913` |
| `diffusionpde_norm/darcy_hf/metadata.json` | 329 | `d49393837609d0e06b4a81a7fd22bb8b8ca47fc6e10cb92ac3b28d0712821bd7` |
| `diffusionpde_norm/helmholtz_hf/data-00000-of-00027.arrow` | 487418216 | `2dd8960c77fd1cc9571e2036592bf541b2b9ecd3d8431959f28833e5a88afc3b` |
| `diffusionpde_norm/helmholtz_hf/metadata.json` | 349 | `b86d20a133be3c96f748ce5f59468c836570507ae1ad8a8f1d9887634af0767e` |
| `diffusionpde_norm/converted/poisson_hf_first1000.npz` | 262144262 | `fc36589fd5ab5a040c868b86bbe8085c179133cd5f320b565196b161fb86af9c` |
| `diffusionpde_norm/converted/darcy_hf_first1000.npz` | 262144262 | `556b762774b76582e33664402672d712d391645aadd5c645c33e5736749587c3` |
| `diffusionpde_norm/converted/helmholtz_hf_first1000.npz` | 262144262 | `7405a570f6c38a6c0e8175641da83fbe2a7b32359479c4c96dbababe93ba33ec` |

**4. ELADO (`elado/`)**

| file (under `data/_external/`) | bytes | sha256 |
|---|---|---|
| `elado/A1_test.h5.zst` | 3918944584 | `282769284b7b4ce049f99523ec57bc3552d769af30b12496d115166e727c0209` |
| `elado/E1_test.h5.zst` | 964872916 | `c9309ba9ab838e77f475322585e64021a960155a8cba8608b7e6814880c1d466` |
| `elado/scripts.tar.zst` | 2496 | `07593df20c689fbeca57fd57bc5be171d7195c2d4d6f77a8df2659efdfac4ffa` |
| `elado/converted/A1_test.h5` | 4194306048 | `52d00c9baedf6695eb70b5ebd95716e377eab4ef89331ab58db7dfb9b30366e1` |
| `elado/converted/E1_test.h5` | 1048628096 | `0d0469157e4f6b9b35e8cd21d34eca2a4e8e7382d540f7eeb55dacfeaf71a2b5` |

**5. PDEgym Wave-Gauss subset (`wave_gauss/`)**

| file (under `data/_external/`) | bytes | sha256 |
|---|---|---|
| `wave_gauss/converted/wave_gauss_first256.npz` | 285213176 | `58a604423193bc77a532db31aadc7e3c5ba4d58c0d57f25b2043952ee6598009` |


**6. UCSD/LLNL MTNN (`ucsd_pde/`)**

| file (under `data/_external/`) | bytes | sha256 |
|---|---|---|
| `ucsd_pde/bb1852369g_1_1.md` | 2774 | `85cfafa9a88c1a1b66d82e83ac5dc209d6aead2bb77a3536072b9a0e06c4f68b` |
| `ucsd_pde/bb1852369g_2_1.tar` | 2612940800 | `8030c06a8544ee58ca7b3666e1e0f22129d83a2751ae51bc240168e3b8b68233` |
| `ucsd_pde/converted/Poisson4.npz` | 409618068 | `116365fd76614d772dd38fae18e3527e39234e784d62b2d942ac417d44629d6c` |
| `ucsd_pde/converted/ConvDiff1.npz` | 409618068 | `1a174c6768c4a202b2c85027a15c14dd883e405a423a664294cf42afc18d2b2e` |

