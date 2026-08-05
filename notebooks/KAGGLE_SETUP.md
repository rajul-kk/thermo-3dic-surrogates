# Kaggle Setup Guide

## Step 1 — Upload source code as a Kaggle dataset

1. Zip the `src/` directory from this project
2. Create a new Kaggle dataset named **thermo-pinn-src**
3. Upload the zip — the dataset root must contain the `src/` folder  
   (verify: `/kaggle/input/thermo-pinn-src/src/core/geometry.py` should exist)

## Step 2 — Upload training data as a Kaggle dataset

1. Zip the contents of `data/3d-ice/` (the `.npz` files)
2. Create a new Kaggle dataset named **3dice-thermal-data**
3. Upload the zip — `.npz` files should be directly inside the dataset root

For geometry1 notebook you only need `geometry1_train_*.npz` and `geometry1_test_*.npz`.  
For geometry2 notebook you need `geometry2a_*` files (geometry2b/2c removed 2026-08-06,
see `goal.md`).

## Step 3 — Create Kaggle notebooks

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
| `kaggle_pinn_geometry1.ipynb` | Single-geometry PINN — includes XAI (residual map, power/HTC sensitivity, IG, MC dropout uncertainty) | geometry1 files |
| `kaggle_pinn_geometry2.ipynb` | Single-geometry PINN (geometry2a) — includes XAI (residual map, power/HTC sensitivity, IG, MC dropout uncertainty) | geometry2a files |
| `kaggle_pinn_sampling_comparison.ipynb` | Two PINNs, same architecture/seed, different collocation-sampling strategy (`rar` vs `curriculum`) — includes collocation-point evolution visualization | geometry1 files (default; any single geometry works) |
| `kaggle_sau_cnofno_vs_whno.ipynb` | CNO-FNO (axial attention) vs WHNO on the same geometry — includes spectral mode-importance XAI and interface-distance-bucketed error comparison | geometry1 files (default; `geometry6` recommended for a stronger interface-discontinuity story, much slower) |
| `kaggle_therm_fm.ipynb` | Pretrain CNO-FNO on `geometry1`, few-shot fine-tune on `geometry3` (same mesh shape, required by CNOFNOHybrid's fixed-grid architecture) — includes fine-tuned-vs-scratch shots comparison and weight-drift analysis | geometry1 AND geometry3 files (both needed) |

The three comparison notebooks (`kaggle_pinn_sampling_comparison`, `kaggle_sau_cnofno_vs_whno`,
`kaggle_therm_fm`) follow the same Kaggle dataset setup as above — no additional datasets
needed beyond `thermo-pinn-src` and `3dice-thermal-data`, as long as the data for whichever
geometry/geometries the notebook uses is included in the uploaded `3dice-thermal-data` dataset.

## Step 4 — Save outputs

Checkpoints are saved to `/kaggle/working/checkpoints/`. After training:
- Download `geometry1_best.pt` (or `geometry2a_best.pt` etc.)
- Copy to `checkpoints/<geom_name>/` locally for inference

## GPU settings recommendation

| Notebook | GPU | Expected time |
|----------|-----|---------------|
| geometry1 | T4 x1 | ~2–4 hr |
| geometry2a | T4 x1 | ~3–5 hr |

Kaggle free tier: 30 GPU-hours/week. Train geometry1 first to validate the pipeline.
