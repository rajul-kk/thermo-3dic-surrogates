# fieldlint v4: data cards for the held-out acquisition (pre-registration part; Phase B, acquisition only)

Written 2026-10-09 during acquisition, from documentation, file headers, key names, shapes and dtypes only. **No fieldlint rule other than the
loader smoke test (see section 5) was run on these data, and no correlation, R^2 or other statistic was computed.** Where a min/max was read, it
is stated in the card and was needed only to settle a sign or mapping question. Every "Prediction" is derived from documentation and from the
rule definitions in `README.md` (`fieldlint --list-rules`: F001 finite values and shapes, F002 source support, F003 source/solution smoothing
correlation, F004 duplicates, F005 level/ambient, F006 linear fit info, F007 flux, F008/F009 operator fit and transpose/flip/roll detection).
Nothing about these data was shared with the v4 design work. Paths are under `data/_external_v4/` (git-ignored).

Format as in `datasets_v3.md`. **Native**: the stored grid is the grid the PDE was solved on; **processed**: resampled/interpolated/normalised
after the solve. Presets are staged in `tools/fieldlint/bench/presets_v4_staging/` (mapping only; to be moved into `src/fieldlint/presets/`
at the v4 freeze). Converter: `tools/fieldlint/bench/convert_v4.py` (a copy of `convert_v3.py`, extended; v3 is unchanged).

**Freshness.** None of these files was opened by any fieldlint version. "Same generator, fresh files" marks files whose generator family was
used in tuning or in an earlier held-out round (PDEBench Darcy, ELADO) but whose file was not.

## 1. Registry

| # | dataset | role | status | samples | grid | native? | preset (staging) |
|---|---|---|---|---|---|---|---|
| 1a | ConDiff `poisson_grid128` test | inscope | obtained (52 MB) | 200 (all) | 128 x 128 | native (cell-centred FV) | `v4_condiff_poisson` |
| 1b | ConDiff `gaussian0.1_grid128` test | inscope | obtained | 200 | 128 x 128 | native; k_cell derived | `v4_condiff` (primary), `v4_condiff_nok` (diagnostic) |
| 1c | ConDiff `gaussian1.0_grid128` test | inscope | obtained | 200 | 128 x 128 | as 1b | same |
| 1d | ConDiff `cubic0.4_grid128` test | inscope | obtained | 200 | 128 x 128 | as 1b | same |
| 1e | ConDiff `exponential2.0_grid128` test | inscope | obtained | 200 | 128 x 128 | as 1b | same |
| 2 | ELADO Poisson `B1_test` | inscope (same generator, fresh file; different coefficient spectrum) | obtained (3.9 GB zst, md5 verified; 4.2 GB h5) | 1000 | 512 x 512 | native (FEM vertex grid) | `v4_elado_poisson` |
| 3a | PDEBench Darcy beta 0.1 | inscope (same generator as tuning, fresh file) | obtained (1.3 GB, md5 verified) | first 1000 of 10000 | 128 x 128 | cell-centred, solved as a time-evolution to steady state | `v4_pdebench_darcy` |
| 3b | PDEBench Darcy beta 10.0 | as 3a | obtained (1.3 GB, md5 verified) | first 1000 | 128 x 128 | as 3a | same |
| 3c | PDEBench Darcy beta 100.0 | as 3a | obtained (1.3 GB, md5 verified) | first 1000 | 128 x 128 | as 3a | same |
| C1 | PDEgym Helmholtz (steady) | control (steady, non-diffusion) | obtained (5.2 GB, sha256 = HF LFS id) | first 1000 of 19675 | 128 x 128 | native FD | `v4_pdegym_helmholtz` |
| C2 | ELADO Helmholtz `C1_test` | control (steady, non-diffusion) | obtained (0.97 GB zst, md5 verified; 1.05 GB h5) | 1000 | 256 x 256 | native (FEM) | `v4_elado_helmholtz` |
| C3 | PDEgym Wave-Layer | control (time-dependent) | obtained by HTTP range requests (256 of 3504 trajectories) | 256 | 128 x 128 x 21 t | FD at 128^2 | `v4_pdegym_wave_layer` |

Total new data on disk: see section 4 (limit about 25 GB).

## 2. Cards

### 2.1 ConDiff, the five 128 x 128 test files  (in scope)

* **Field / application**: ML-for-PDE benchmark of steady diffusion with random coefficients (composite materials, heat transfer, Darcy flow); CC-BY-4.0.
* **Source, paper, code**: Hugging Face `condiff/ConDiff` (files `<name>/<name>_test.h5`); Trifonov, Rudikov, Iliev, Laevsky, Oseledets, Muravleva, "ConDiff: A Challenging Dataset for Neural Solvers of Partial Differential Equations", arXiv 2406.04709; generator `condiff.py` in github.com/condiff-dataset/ConDiff (main; fetched 2026-10-09). The test files hold 200 realisations each (paper Sec. 2: "1000 training and 200 test realizations"); all 200 are used, which meets the 200-sample minimum exactly.
* **Equation and BC as documented**: paper eq. (1), `-div(k(x) grad u(x)) = f(x)` on the unit square, `u = 0` on the boundary; `k = exp(GRF)` with cubic / exponential / Gaussian covariance, correlation length 0.05, variance in {0.1, 0.4, 1.0, 2.0}; `f ~ N(0, 1)` per cell; Poisson subset: `k = 1`.
* **Native or processed**: **native**. The paper says "cell-centered second-order finite volume method. The coefficients are in the center of cells, the values are in the nodes"; the code (`condiff.py` lines 121-160, `fd_mtx2`) uses n = 128 cells per side, a coefficient array on (n+1)^2 = 129 x 129 points, and solves `A x = b` directly on the 128 x 128 cells with `spsolve` (line 29). No resampling, no normalisation. Stored raw: `k` (200,129,129), `rhs` (200,16384), `x` (200,16384), all float64.
* **Sign convention (read from the code before any rule was run)**: `fd_mtx2` (lines 145-159) builds a symmetric positive-definite matrix: the diagonal `ac` is the sum of four positive corner coefficients and the off-diagonals are `-al, -ad, -au, -ar`, scaled by `(n+1)^2`. `diffusion()` lines 26-29: `coef = np.exp(k); A = fd_mtx2(coef); b = rng.normal(0, 1, grid*grid); x = splinalg.spsolve(A, b)`, and `rhs` is saved as `b`, `x` as `x` (lines 32-33, 64, 66). The same holds for Poisson (lines 83-85, all-ones coefficient). So the stored pair satisfies `A x = rhs` with `A` = discretisation of `-div(k grad .)`: **source = +rhs, u = +x, no negation**. Fieldlint's convention `-div(k grad u) = s` agrees. (Non-negative u is NOT implied: f is sign-changing.)
* **Index map / layout hazard (code)**: `x`, `rhs` are flat vectors made with `flatten("F")` (lines 151-155), i.e. flat index `p = i + n*j` for cell (i, j), where `i` is the first index of the coefficient array and cell (i, j) touches nodes `a[i:i+2, j:j+2]` (lines 130-148). A plain C-order reshape of `x` to (128, 128) therefore yields the **transpose** of the (i, j) field. `convert_v4.py condiff:<name>` applies the documented Fortran-order index map (`x.reshape(N,n,n).transpose(0,2,1)`; every value is copied unchanged; a round-trip assertion is in the code), so that `u`, `source` and `k_cell` share the generator's axes. This is a re-indexing by the documented map, not a repair of a defect; the file as released is flat and carries no axis order.
* **Two k mappings (both pre-registered; the choice is made here, from documentation)**:
  * **PRIMARY, `v4_condiff`**: `k` := `k_cell`, the mean of the four corner nodes of the stored coefficient, i.e. `ac/4` of `fd_mtx2` (lines 145-148: `ac[i,j] = a[i,j] + a[i,j+1] + a[i+1,j] + a[i+1,j+1]`), the solver's own cell coefficient. **Derived, not lossless** (the stored 129 x 129 `k` cannot share a grid with `u`, and F001 would flag the mismatch). The face coefficients the solver actually uses are the means of two nodes (lines 130-140), not the mean of two `k_cell` values, so even this mapping is only approximately the solver's operator.
  * **SECONDARY diagnostic, `v4_condiff_nok`**: `k` omitted (`u` and `source` only) although `k` varies per sample. Not counted in criteria.
  * Poisson (1a): `k = 1`, only one variant (`v4_condiff_poisson`).
* **Role**: in scope (documented steady diffusion, native grid). The five files cover: no coefficient (1a), smooth low contrast (Gaussian 0.1), smooth high contrast (Gaussian 1.0), a compact-support covariance (cubic 0.4) and a rough field at the highest contrast (exponential 2.0, contrast up to 1e5).
* **Prediction (documentation only)**:
  F001 ok (primary and Poisson; `k_cell` shares the 128 x 128 grid). F002 skipped (source is sign-changing: white-noise N(0,1)). F003 applicable; expected ok but with low discriminating power: the smoothed source is positively related to `u` through a positive-definite operator, yet the source is white noise, so the smoothing correlation per sample may be weak (uncertain). F004 ok. F005 ok (no non-negative-source premise; ambient 0 with `u` of both signs is not a violation of anything documented; a flag would be a false alarm). F007 skipped.
  **F008/F009 (primary, `v4_condiff`)**: applicable. The operator is a 5-point cell-centred finite-volume stencil on exactly the stored grid; the face coefficients are node means, which differ from means of `k_cell`, so the fit is exact only up to that approximation. Expect clean median R^2 near 1 for Poisson (exact stencil, k = 1) and for `gaussian0.1`; registered as **uncertain** for `gaussian1.0`, `cubic0.4` and especially `exponential2.0` (rough field, high contrast, node-mean versus cell-mean mismatch): there a decline by the R^2 gate is possible, and it is not a false negative. Detection: transposed/flipped/rolled `k` or source should be detected by F008 where it is applicable and by F003 when it is not weak.
  **Secondary (`k` omitted)**: F008/F009 expected **skipped or poor** (the operator needs `k`, which varies); if F008 is applicable at a low R^2 it is registered as a surprise. No fault in `k` can be injected.
  Poisson (1a): F003 and F008/F009 applicable, F008 near 1.

### 2.2 ELADO Poisson, B1_test  (`v4_elado_poisson`, in scope; same generator as A1, fresh file)

* **Source, licence, paper**: Zenodo record 20738464 (CC-BY-4.0), `B1_test.h5.zst` (3 923 178 230 bytes, md5 `41b88a5479a6336a30a2eee370168a47` from the Zenodo API); Ehebrecht, Scharle, Atzmueller, arXiv 2606.20771, Sec. 5.2 (Poisson, "identical" to Sec. 5.1 with sigma_s0 = 0.02, sigma_jump = 0.0267, a_min/a_max unchanged; B1-test has 1000 samples); generator `poisson_dirichlet.py` in `scripts.tar.zst` (read, not run).
* **Equation and BC as documented**: `-div(a grad u) = 1`, `u = 0` on the boundary, P1 finite elements on the (N_cells+1)^2 vertex grid, 512 x 512 (script: `A_form = inner(a*grad(u), grad(v))*dx`, `L_form = f*v*dx` with `f = 1`, so the operator is positive definite and the source is +1). Arrays `x` (coefficient a) and `y` (u), as for A1 (confirmed on the decompressed file: `x`, `y` (1000, 512, 512) float64).
* **Native or processed**: native, as A1 (card 2.6 of `datasets_v3.md`); the `.h5` is the byte-for-byte zstd decompression.
* **Sign convention**: from the script as above: u >= 0 for a > 0, source +1, no negation. Same preset logic as `v3_elado_poisson`.
* **Prediction**: as v3 card 2.6 for A1_test: F001 ok; F002/F003 skipped (uniform source); F004 ok; F005 ok (u >= 0); F008/F009 applicable with clean median R^2 near 1 (P1 FEM with a nodal coefficient: an arithmetic-type face mean; the rougher coefficient spectrum of B1, sigma_s0 0.02, may lower it slightly; a decline would be a surprise for a native FEM file). Detection of transposed/flipped/rolled `a` by F008.

### 2.3 PDEBench 2D Darcy, beta 0.1 / 10.0 / 100.0  (`v4_pdebench_darcy`, in scope; same generator as tuning, fresh files)

* **Source, licence, paper**: DaRUS DOI 10.18419/DARUS-2986 (CC-BY), files `2D/DarcyFlow/2D_DarcyFlow_beta<b>_Train.hdf5` (1 310 724 488 bytes each; MD5s from the DaRUS API: beta 0.1 `294f9a03a4aa16b0e386469ca8b471be`, beta 10.0 `a7f23cf8011fc211b180828af39b7d1a`, beta 100.0 `7c09f3b1bb097737d3fd52ffb0c6f1c8`); Takamoto et al., PDEBench, arXiv 2210.07182, App. D.4 and Table of dataset sizes ("10 000 samples, 128 x 128 spatial cells, 1 time step, 2 variables").
* **Equation and BC as documented**: `-div(a(x) grad u(x)) = beta` on (0,1)^2, `u = 0` on the boundary (App. D.4 eq. 13-14; the paper's second condition is written on "(0,1)^2" but is the Dirichlet condition), `f` constant = beta; the solution is obtained "by solving a temporal evolution equation ... until reaching a steady state" with the same numerics as the 1D diffusion-reaction case (finite volume, per the paper).
* **Native or processed**: cell-centred 128 x 128 grid, the grid of the solver (documented); the steady state is reached by time stepping, so the stored `u` is converged to the steady discrete solution up to the stopping tolerance (not documented). Same arrangement as the v1/v2 beta 1.0 / 0.01 files (HDF5 keys `nu` (10000,128,128) and `tensor` (10000,1,128,128)).
* **Sign convention**: the repository (github.com/pdebench/PDEBench) holds no Darcy data-generation code (the only file with "Darcy" in the data_gen tree is `run_DarcyFlow2D.sh`, which calls the reaction-diffusion script; the PDEBench `models/config` files only read the data). Decision from the paper: `-div(a grad u) = beta` with beta > 0 and `u = 0` boundary, i.e. u >= 0; no negation. Mapping-only check (first 50 samples of each converted file, min/max only): `tensor` (u) is strictly positive in all three files (smallest 1.9e-5 / 1.9e-3 / 1.9e-2, largest 0.43 / 7.6 / 72), `nu` takes values in [0.1, 1.0]; so u >= 0 as documented, with scale proportional to beta.
* **Mapping**: `nu` -> k, `tensor` -> u, `source_uniform` read from the file name (the v1/v2 `darcy` preset logic); npz packaging as `data/pde-bench/darcy_beta*_n1000.npz` (first 1000 samples, flattened to 16384).
* **Prediction (from documentation and the v2 record that this generator was matched by an arithmetic face mean)**: F001 ok; F002/F003 skipped (uniform source); F004 ok; F005 ok (u >= 0); F008/F009 applicable with clean median R^2 near 1 for beta 0.1 and 100 as for beta 1.0 / 0.01 (the arrangement is the same, only the source scale changes, which does not enter an R^2); detection of transposed/flipped/rolled `k`. A decline would be a surprise. Because these files share the generator with the tuning data they test file-to-file stability, not generalisation to a new generator.

### 2.4 PDEgym Helmholtz  (`v4_pdegym_helmholtz`, control, steady non-diffusion)

* **Source, licence, paper**: Hugging Face `camlab-ethz/Helmholtz`, `Helmholtz.h5` (5 196 371 856 bytes, sha256 = HF LFS id `4e8317aabb02c88729fb0065aa1d509e2c792ee8975c60b690f748fb3fdd6afe`, verified), CC-BY-NC-4.0; Herde, Raonic et al., Poseidon, arXiv 2405.19101, App. B.2.15. (`datasets_v3.md` recorded this repository as listing no files; it now lists `Helmholtz.h5`.)
* **Equation and BC as documented**: eq. (69), `-Laplace(u) - omega^2 a(x, y) u = 0`, `omega = 5/2`, `u = b` on the boundary of (0,1)^2 with `b ~ U[0.25, 0.5]`; `a` a normalised sum of 2-7 Gaussians in [0,1]; steady; finite differences at 128^2 ("similar to DeVito"); 19675 solutions. **There is no source term.**
* **Arrays (README + file)**: 19675 groups `Sample_<i>` each with `a` (128,128), `u` (128,128) float64 and a scalar `bc` float64. Axes as stored; first 1000 in numeric order stacked by `convert_v4.py helmholtz_pdegym`.
* **Mapping (a user's most plausible reading, as v3 `elado_helmholtz_e1`)**: `a` -> k, `u` -> u, uniform source 1, ambient 0.
* **Sign convention**: equation (69) has the diffusion operator with the usual sign and a zero right-hand side; the Dirichlet value b > 0 supplies the forcing. The mapping's source of 1 does not exist in the data; it is a reading, not a documented source.
* **Prediction**: F001 ok. F002/F003 skipped (uniform source). F004 ok. F005 ok (omega^2 a <= 6.25 is below the smallest Dirichlet eigenvalue 2 pi^2 of the Laplacian, so the operator is positive definite and u > 0 with b > 0; ambient 0 is not violated). F008/F009: the diffusion operator with `a` as the *diffusion* coefficient is the wrong operator (here `a` is a reaction coefficient and the data contain no source), so **expected skipped by the R^2 gate**; an applicable F008 would be a false alarm for a steady control.

### 2.5 ELADO Helmholtz, C1_test  (`v4_elado_helmholtz`, control, steady non-diffusion)

* **Source**: Zenodo 20738464, `C1_test.h5.zst` (969 753 283 bytes, md5 `f351b1a25f74e8f3520bab4cc09f5173`); family C1, paper Sec. 5.3, Eq. (10): Helmholtz with a_i = 1, complex u, coefficient (wavenumber) field with a_min = 6, a_max = 14, 255 x 255 mesh (coefficient axis 256), `u = 0` on the boundary, a Gaussian excitation not stored; quantity of interest |u| (storage not stated in the paper; the generator `helmholtz_dirichlet.py` returns real and imaginary parts, and the released `y` is expected to be |u| by analogy with E1 in v3).
* **Mapping**: `x` -> k, `y` -> u, uniform source 1 (as `v3_elado_helmholtz`; the two ELADO Helmholtz files share the preset logic). D1 (PDE unstated in the paper) and A1_tail (same family as v3's A1_test) were not taken.
* **Prediction**: steady, so no time flag exists. F002/F003 skipped (uniform source). F005 ok if y = |u| >= 0. F008/F009: the diffusion operator is wrong for Helmholtz, so **skipped by the R^2 gate expected**; applicable would be a false alarm.

### 2.6 PDEgym Wave-Layer  (`v4_pdegym_wave_layer`, control, time-dependent)

* **Source**: Hugging Face `camlab-ethz/Wave-Layer` (CC-BY-NC-4.0): `solution_0.nc` (first 256 of 3504 trajectories), `c_0.nc` (first 256 rows), read by HTTP range requests (LFS ids in the manifest); Poseidon App. B.2.11. Wave equation `u_tt - c(x)^2 Laplace u = 0`, absorbing boundaries, FD at 128^2, T = 1, `u0` a sum of Gaussians, `c` piecewise constant in 3-6 vertical layers, `c_i ~ U[2000, 5000]`. Stored `solution` (256, 21, 128, 128) float32 (Poseidon says 21 snapshots, the dataset README says 15), `c` (256, 128, 128).
* **Mapping**: as v3 Wave-Gauss: `solution[:,0]` -> source, `solution[:,20]` -> u, `c` -> k.
* **Prediction**: run twice as in v3 (with and without the README's `time_dependent` scope declaration). Flagged: F002/F003/F008/F009 skipped "out of scope: time-dependent". Unflagged: F002/F003 applicable (u0 >= 0) and may raise false alarms; F008 expected skipped.

### 2.7 Considered, not obtained (see section 3)

## 3. Datasets considered and their status

* **UCSD/LLNL MTNN**: the archive already held in `data/_external/ucsd_pde` has only two diffusion-type files (`Poisson4.npz`, `ConvDiff1.npz`), both used in v3, and Darcy files without a solution field. No Poisson/ConvDiff variants exist beyond them (member list of the tar checked). The unused samples 1001-10000 of those two files are not fresh files and were not taken. **Nothing fresh.**
* **PDEgym**: the only steady elliptic sets in the camlab-ethz organisation are Poisson-Gauss (used in v3) and Helmholtz (taken here); all others (CE-*, NS-*, Wave-*, ACE, FNS-KF, SE-AF, GCE-RT) are time-dependent. So no new steady diffusion set exists there.
* **ELADO**: Zenodo 20738464 holds A1/A1-os/B1/C1/D1/E1 (train, test, tail, ramp, outliers). Taken: B1_test (Poisson) and C1_test (Helmholtz control). Not taken: A1_tail, A1_train, A1_os_* (same family and equation as v3's A1_test), D1 (PDE not stated in the paper), B1_train/B1_ramp (B1_ramp is 7.8 GB). The C1 file also carries six scalar arrays `m_o*_p*_nt_zz*` (1000,) next to `x` and `y`; their meaning is undocumented and they are not mapped. B1 and C1 keys otherwise as A1/E1: `x`, `y` (1000, 512, 512) and (1000, 256, 256), float64.
* **PDEBench diffusion-reaction (13.2 GB) and shallow water (6.6 GB)** as time-dependent controls: not taken. The DaRUS storage honours range requests, but opening the large HDF5 through fsspec did not complete while other downloads were running, and Wave-Layer was already cheap. Wave-Layer is the one time-dependent control.
* **Others considered by a brief search**: TFRD/HSL heat-source data (Baidu netdisk, excluded in v3); Multiphysics Bench (arXiv 2505.17575, not examined for steady scalar diffusion data); the "Localized PCA-Net" 2D Poisson set (no public release found).


Junk files left in place (not used, not deleted, per instruction): `elado/A1_tail.h5.zst` and `elado/B1_test.h5.zst` (truncated fragments of a failed first single-connection download; A1_tail was dropped as a plan item, B1_test was re-downloaded as `B1_test.h5.zst.part2`), and `pdebench_darcy/2D_DarcyFlow_beta0.1_Train.hdf5` (3 MB fragment; the full file is the `.part2`). Their hashes are not recorded and they must not be read. The `*.part2` / `.state` pairs are the ranged-download output and resume state of `scratchpad/pdl.py` (a throwaway script, not committed). Zenodo throttles a single connection to about 40-60 KB/s; 96 parallel ranged connections gave about 0.3-0.7 MB/s per file, so B1_test (3.9 GB) completed, well inside the 90-minute criterion. The `.hdf5.part2` of beta 100 was first finished one chunk short because the planned size was copied from the beta 0.1 file (DaRUS size 1 310 724 760, not 1 310 724 488); the last chunk was re-fetched and the MD5 then matched.


## 4. Files, sizes and sha256

Total of the registered files below (sources, downloads and converted outputs): 21.13 GB (the data directory is about 20 GB including the junk fragments). The first table lists every file with size and sha256; converted outputs also have `*.MANIFEST.json` files next to them with the source and output hashes.

| file (under `data/_external_v4/`) | bytes | sha256 | URL / how produced | note |
|---|---|---|---|---|
| `condiff/poisson_grid128/poisson_grid128_test.h5` | 52430848 | `31ee6e8a8521bf0095cb4063cbdd48a865ec0392542a7eb6e804553a3d961fdd` | https://huggingface.co/datasets/condiff/ConDiff/resolve/main/poisson_grid128/poisson_grid128_test.h5 | LFS-id match: True |
| `condiff/converted/poisson_grid128_test.npz` | 52429300 | `d57029cbdff7e36a3759d02ce038cbcb87a7bf53a8e6555d7bd56c866846f1ed` | (convert_v4.py condiff:poisson_grid128) |  |
| `condiff/gaussian0.1_grid128/gaussian0.1_grid128_test.h5` | 79056448 | `65bec4a50eeb0b8a7b90839c0c7145cbe2dbeb33b1fd30d476c21f7d05b98616` | https://huggingface.co/datasets/condiff/ConDiff/resolve/main/gaussian0.1_grid128/gaussian0.1_grid128_test.h5 | LFS-id match: True |
| `condiff/converted/gaussian0.1_grid128_test.npz` | 105269788 | `b5252e9c84b9cd1d91972459e2e239fc1d9244e96fac2942e9ff142ebb92e2bf` | (convert_v4.py condiff:gaussian0.1_grid128) |  |
| `condiff/gaussian1.0_grid128/gaussian1.0_grid128_test.h5` | 79056448 | `65f852580b24f9be573a3cdbda6b98c31d8f5d1d78091910cbd554c89cb02e82` | https://huggingface.co/datasets/condiff/ConDiff/resolve/main/gaussian1.0_grid128/gaussian1.0_grid128_test.h5 | LFS-id match: True |
| `condiff/converted/gaussian1.0_grid128_test.npz` | 105269788 | `86851a91e476504da546b7164c30df9a25935c17f92c00c89ea841a8b9ad8a0f` | (convert_v4.py condiff:gaussian1.0_grid128) |  |
| `condiff/cubic0.4_grid128/cubic0.4_grid128_test.h5` | 79056448 | `0996d4feba7d559d04b966f09a888918c27fcfbd43e68ff2934d5c36cb0d6428` | https://huggingface.co/datasets/condiff/ConDiff/resolve/main/cubic0.4_grid128/cubic0.4_grid128_test.h5 | LFS-id match: True |
| `condiff/converted/cubic0.4_grid128_test.npz` | 105269788 | `c57866f7f28a304005563cc0be3b5854f123b7bedb82bf32b8b856772882a446` | (convert_v4.py condiff:cubic0.4_grid128) |  |
| `condiff/exponential2.0_grid128/exponential2.0_grid128_test.h5` | 79056448 | `604382c22dc56590f2765cc0c2884719664e3ceb742a3fcfbce148fe5203a8e1` | https://huggingface.co/datasets/condiff/ConDiff/resolve/main/exponential2.0_grid128/exponential2.0_grid128_test.h5 | LFS-id match: True |
| `condiff/converted/exponential2.0_grid128_test.npz` | 105269788 | `c3386dc43ccaceac84c031987b3ece0e2f0c51a6347e912b71138b461a5d3572` | (convert_v4.py condiff:exponential2.0_grid128) |  |
| `helmholtz_pdegym/Helmholtz.h5` | 5196371856 | `4e8317aabb02c88729fb0065aa1d509e2c792ee8975c60b690f748fb3fdd6afe` | https://huggingface.co/datasets/camlab-ethz/Helmholtz/resolve/main/Helmholtz.h5 | LFS-id match: True |
| `helmholtz_pdegym/converted/helmholtz_first1000.npz` | 262152726 | `277c56977008d08369adf3c964123d02364e43b7d010b558c59fd72585a71663` | (convert_v4.py helmholtz_pdegym) |  |
| `wave_layer/converted/wave_layer_first256.npz` | 369099256 | `29272b56bc4969e2cd7e39828d82d61f8c6ac142aebc1334c963f1cbb44d9132` | (convert_v4.py wave_layer; sources by range request, LFS ids in manifest) |  |
| `pdebench_darcy/2D_DarcyFlow_beta0.1_Train.hdf5.part2` | 1310724488 | `74bbdf8b5db9cfd36168b003114673261337fd0422fa492ab1cb0c79f5d157e8` | https://darus.uni-stuttgart.de/api/access/datafile/133218 | MD5 verified against DaRUS |
| `pdebench_darcy/converted/darcy_beta0.1_n1000.npz` | 131072490 | `91b50d05e0f5264e958d5f86d6b792131895865e209fa7975fdfb09e0376a872` | (convert_v4.py darcy:0.1) |  |
| `pdebench_darcy/2D_DarcyFlow_beta10.0_Train.hdf5.part2` | 1310724488 | `d8e34530cb756618f389a765443e49ca8e1b69f1597f761bd7ff374ac92bca75` | https://darus.uni-stuttgart.de/api/access/datafile/133220 | MD5 verified against DaRUS |
| `pdebench_darcy/converted/darcy_beta10.0_n1000.npz` | 131072490 | `6584add567311235ec7e626249fe0e48ef27349d976e50680bae032a3e5a38b9` | (convert_v4.py darcy:10.0) |  |
| `pdebench_darcy/2D_DarcyFlow_beta100.0_Train.hdf5.part2` | 1310724760 | `2785a332c35bc532de23add63d6525c1d6f360b432ef0452ffe622bbbcdfa8f5` | https://darus.uni-stuttgart.de/api/access/datafile/133221 | MD5 verified against DaRUS |
| `pdebench_darcy/converted/darcy_beta100.0_n1000.npz` | 131072490 | `54817ac04b3b6f39f667614b77850614fc3e98f9fa7f9eb1039dbbbedfb20328` | (convert_v4.py darcy:100.0) |  |
| `elado/B1_test.h5.zst.part2` | 3923178230 | `c2bb5046f29f2f9545856524180dd38f0eef84088387bafe599a07737f0f5f5b` | https://zenodo.org/api/records/20738464/files/B1_test.h5.zst/content | MD5 verified against Zenodo; ranged download |
| `elado/converted/B1_test.h5` | 4194306048 | `1d0034634d76101ecafbbb608f11e7d822c0ac70eb67be9ade774aa147e09be1` | (convert_v4.py elado:B1_test) |  |
| `elado/C1_test.h5.zst.part2` | 969753283 | `e7fd487dfa9459c4af01dd1528988addcf0f38767cbe5eab084c232f70ef33d5` | https://zenodo.org/api/records/20738464/files/C1_test.h5.zst/content | MD5 verified against Zenodo; ranged download |
| `elado/converted/C1_test.h5` | 1048628096 | `3a51c9b79257598c6da202c85480c8580c4d043554268fc5338c56282daa14a8` | (convert_v4.py elado:C1_test) |  |

The Hugging Face LFS ids of the ConDiff files and of `Helmholtz.h5` equal the sha256 values above; the Wave-Layer sources are range-read, their LFS ids are in the manifest. The ELADO Zenodo records and DaRUS give MD5 only, which matched. The ConDiff paper (arXiv 2406.04709), Poseidon (arXiv 2405.19101), PDEBench (arXiv 2210.07182) and ELADO (arXiv 2606.20771) were read from the arXiv PDFs/HTML; the ConDiff code from raw.githubusercontent.com/condiff-dataset/ConDiff/main/condiff.py.

## 5. Smoke test

Run after the freeze commit `bf27d68` ("fieldlint v4: ... frozen v4 thresholds") existed (it adds `frozen_thresholds_v4.json`): each registered set was loaded through the staged preset with `--rules F001 --max-samples 20` and the working-tree CLI. All 13 runs (ConDiff Poisson, four ConDiff primary, ConDiff diagnostic `nok`, ELADO B1 and C1, PDEBench Darcy 0.1/10/100, PDEgym Helmholtz, Wave-Layer) report `F001 integrity ok  20 samples checked; no problems`, with the expected sample counts (200, 200, 1000, 1000, 1000, 1000, 256). No other rule was run, and no output other than F001's was looked at. The presets declare no `operator:` key (the v4 optional declaration), leaving the automatic detection to the v4 runner.
