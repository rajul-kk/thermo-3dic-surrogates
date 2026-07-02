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
For geometry2 notebook you need `geometry2a_*`, `geometry2b_*`, `geometry2c_*` files.

## Step 3 — Create Kaggle notebooks

1. Go to kaggle.com → Code → New Notebook
2. Upload `kaggle_pinn_geometry1.ipynb` or `kaggle_pinn_geometry2.ipynb`
3. Under **Data** (right panel), add both datasets:
   - `thermo-pinn-src`
   - `3dice-thermal-data`
4. Enable **GPU accelerator** (Settings → Accelerator → GPU T4 x1)
5. Run all cells

## Step 4 — Save outputs

Checkpoints are saved to `/kaggle/working/checkpoints/`. After training:
- Download `geometry1_best.pt` (or `geometry2a_best.pt` etc.)
- Copy to `checkpoints/<geom_name>/` locally for inference

## GPU settings recommendation

| Notebook | GPU | Expected time |
|----------|-----|---------------|
| geometry1 | T4 x1 | ~2–4 hr |
| geometry2 (one variant) | T4 x1 | ~3–5 hr |
| geometry2 (all 3 variants, TRAIN_ALL_VARIANTS=True) | T4 x1 | ~10–15 hr |

Kaggle free tier: 30 GPU-hours/week. Train geometry1 first to validate the pipeline.

## Switching geometry2 variants

In `kaggle_pinn_geometry2.ipynb`, change the config cell:

```python
GEOM_NAME = 'geometry2a'  # change to 'geometry2b' or 'geometry2c'
```

Or set `TRAIN_ALL_VARIANTS = True` to train all three in sequence (uses ~12 hr GPU).
