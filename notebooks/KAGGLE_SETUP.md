# Kaggle Setup Guide

> **v5 data, 2026-09-24: read this first.** Every neural result before this date is void
> (`docs/report.md` §9.23–9.24): the data was transposed, and all three neural loaders scrambled
> or zeroed their inputs. For `kaggle_geometry4_vs_geometry6_fno.ipynb`:
> 1. **Remove** any old `geometry4_shelf` / `geometry6_shelf` datasets from the notebook. File
>    names overlap with v5, and the notebook searches every attached dataset.
> 2. Upload `data/kaggle_v5/geometry4_shelf_v5.zip` and `geometry6_shelf_v5.zip` as new datasets
>    (built locally; `data/` is gitignored). Zips for geometry1/2a/3/5 are there too.
> 3. Settings -> Internet -> On (clones the fixed `src/` from `main`); Accelerator -> GPU T4.
> 4. The physics-informed loss is **off** (`PDE_WEIGHT = FLUX_WEIGHT = 0`). Its
>    finite-difference residual assumes uniform z, and 3D-ICE's z-nodes are not uniform.
> 5. Expected grids: geometry4 (56, 100, 10), geometry6 (56, 168, 15), axes (length, width, z).

> **`kaggle_geometry4_vs_geometry6_fno.ipynb` does not need the src dataset at all.** The
> repo is public and tracks `src/`, so that notebook clones it (set **Settings -> Internet ->
> On**) and falls back to an attached dataset if internet is off. The `.npz` data still has to
> be uploaded, because `data/` is gitignored and is therefore not in the repo.
>
> **Slugs do not need to match for `kaggle_geometry4_vs_geometry6_fno.ipynb`.** That notebook
> locates `src/` and the `.npz` files by searching the attached datasets for their *contents*,
> because Kaggle silently rewrites slugs (case, hyphens, an appended suffix). If it cannot find
> them it prints every attached dataset and what each contains. The older notebooks still use
> hard-coded slugs.

## Step 1: Upload source code as a Kaggle dataset

1. Zip the `src/` directory from this project
2. Create a new Kaggle dataset named **thermo-pinn-src**
3. Upload the zip. The dataset root must contain the `src/` folder
   (verify: `/kaggle/input/thermo-pinn-src/src/core/geometry.py` should exist)

## Step 2: Upload training data as a Kaggle dataset

1. Zip the contents of `data/3d-ice/` (the `.npz` files)
2. Create a new Kaggle dataset named **3dice-thermal-data**
3. Upload the zip. `.npz` files should be directly inside the dataset root

For geometry1 notebook you only need `geometry1_train_*.npz` and `geometry1_test_*.npz`.
For geometry2 notebook you need `geometry2a_*` files (geometry2b/2c removed 2026-08-06,
see `goal.md`).

## Step 3: Create Kaggle notebooks

1. Go to kaggle.com → Code → New Notebook
2. Upload one of the notebooks below
3. Under **Data** (right panel), add both datasets:
   - `thermo-pinn-src`
   - `3dice-thermal-data`
4. Enable **GPU accelerator** (Settings → Accelerator → GPU T4 x1)
5. Run all cells

### Available notebooks

| Notebook | What it trains | Data needed |
|---|---|---|
| `kaggle_pinn_geometry1.ipynb` | Single-geometry PINN, includes XAI (residual map, power/HTC sensitivity, IG, MC dropout uncertainty) | geometry1 files |
| `kaggle_pinn_geometry2.ipynb` | Single-geometry PINN (geometry2a), includes XAI (residual map, power/HTC sensitivity, IG, MC dropout uncertainty) | geometry2a files |
| `kaggle_pinn_sampling_comparison.ipynb` | Two PINNs, same architecture/seed, different collocation-sampling strategy (`rar` vs `curriculum`), includes collocation-point evolution visualization | geometry1 files (default; any single geometry works) |
| `kaggle_sau_cnofno_vs_whno.ipynb` | CNO-FNO (axial attention) vs WHNO on the same geometry, includes spectral mode-importance XAI and interface-distance-bucketed error comparison | geometry1 files (default; `geometry6` recommended for a stronger interface-discontinuity story, much slower) |
| `kaggle_therm_fm.ipynb` | Pretrain CNO-FNO on `geometry1`, few-shot fine-tune on `geometry3` (same mesh shape, required by CNOFNOHybrid's fixed-grid architecture), includes fine-tuned-vs-scratch shots comparison and weight-drift analysis | geometry1 AND geometry3 files (both needed) |
| `kaggle_a3_throttled_arch_comparison.ipynb` | FNO vs CondFNO vs CNO-FNO vs CNO-FNO+attention (SAU-FNO), all four on the **throttled** geometry1 pilot (`goal.md` Track A3). Tests whether GPU-scale capacity changes the CPU-smoke-test result where FNO trailed ridge badly. Evaluates with the same detrended metrics `scripts/baselines.py` uses, printed against the ridge/kNN reference scores already on record. | throttled geometry1 `.npz` files from `data/3d-ice-throttle-pilot/geometry1/`, not the standard `3dice-thermal-data` dataset, upload separately (slug suggestion: `3dice-throttle-pilot-data`) |
| `kaggle_geometry4_vs_geometry6_fno.ipynb` | **FNO vs ridge vs linear-on-field on the shelf-layout datasets, 5-fold CV on the folds `docs/report.md` §9.15b uses.** Tests §9.15c's falsifiable prediction that FNO accuracy tracks the *field* representation's difficulty ordering (0.94 geometry4 / 0.51 geometry6, gap 0.43) rather than the compact representation's (−0.67 / −2.87, gap 2.19). Recomputes both baselines in-notebook so the comparison cannot drift from a stale number, and prints a sanity line against the report. | geometry4 shelf files from `data/3d-ice-layout-geometry4/geometry4/` AND geometry6 shelf files from `data/3d-ice-layout-geometry6/geometry6/`, uploaded as two datasets (slug suggestions: `3dice-layout-geometry4`, `3dice-layout-geometry6`) |
| `kaggle_a3b_leakage_arch_comparison.ipynb` | Same four architectures, on the **leakage-feedback** geometry1 pilot (`goal.md` Track D), positive electrothermal feedback, structurally sharper than throttling's negative feedback. Automatically filters to the converged subset only (runaway scenarios have no physically meaningful steady state, some diverge past 500,000°C) before training or scoring. No FNO of any capacity has been run on this data yet; the CPU reference (ridge det.MAE 0.248 K, R² 0.972 on the converged split) is ridge/kNN only. | leakage geometry1 `.npz` files from `data/3d-ice-leakage-pilot/geometry1/`, upload separately (slug suggestion: `3dice-leakage-pilot-data`) |
| `kaggle_geometry7_fno_arch_comparison.ipynb` | Same four architectures, on the **geometry7 CoWoS-L bridge/via pilot** (`docs/report.md` §9.11), the first geometry in this project with a genuine sharp lateral material discontinuity in a passive layer (k=60 vs k=0.5 W/m·K). No FNO of any capacity has been run on this data yet; only ridge/kNN (ridge det.MAE 0.462 K, spatial R² 0.992) are on record. | geometry7 pilot `.npz` files from `data/3d-ice-geometry7-pilot/geometry7/` (35 train, 5 test), upload separately (slug suggestion: `3dice-geometry7-pilot-data`) |

The three comparison notebooks (`kaggle_pinn_sampling_comparison`, `kaggle_sau_cnofno_vs_whno`,
`kaggle_therm_fm`) follow the same Kaggle dataset setup as above: no additional datasets
needed beyond `thermo-pinn-src` and `3dice-thermal-data`, as long as the data for whichever
geometry/geometries the notebook uses is included in the uploaded `3dice-thermal-data` dataset.

## Step 4: Save outputs

Checkpoints are saved to `/kaggle/working/checkpoints/`. After training:
- Download `geometry1_best.pt` (or `geometry2a_best.pt` etc.)
- Copy to `checkpoints/<geom_name>/` locally for inference

## GPU settings recommendation

| Notebook | GPU | Expected time |
|----------|-----|---------------|
| geometry1 | T4 x1 | ~2-4 hr |
| geometry2a | T4 x1 | ~3-5 hr |

Kaggle free tier: 30 GPU-hours/week. Train geometry1 first to validate the pipeline.
