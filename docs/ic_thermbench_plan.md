# Plan: adding the missing non-neural baseline to IC-ThermBench

**Status:** scoped 2026-09-10. Data downloaded and verified (4.6 GB, `data/ic-thermbench/`, gitignored). Channel semantics measured (§2). Loader and metric parity **done and verified exactly** (§6 steps 2-3). Next: run the three baselines (§5).
**Owner question this answers:** what contribution is left after IC-ThermBench, and can this
repo make it without GPU compute?

---

## 1. Why this experiment

IC-ThermBench (arXiv:2608.23977, Aug 2026, `github.com/Day333/ThermalBench`) is an open
benchmark for 2.5D/3D-IC thermal learning: ~82,000 samples, five progressive generalization
scopes, a unified pipeline, MIT code / CC BY 4.0 data. It removed "we release a benchmark"
from this repo's available novelty claims (`docs/references.md`, §2).

It also has a gap this repo is unusually well placed to fill. Its **eight baselines are
U-Net, FNO, U-FNO, SAU-FNO, DeepOHeat and Therm-FM T/B/L — every one a deep network.** There
is no ridge, no kNN, no nearest-neighbour, nothing closed-form. A benchmark built explicitly
to make thermal-surrogate comparison fair and reproducible still does not check whether a
linear fit already solves its tasks.

This repo has that tooling built and validated (`scripts/baselines.py`,
`scripts/hotspot_eval.py`), and — more importantly — has a **prior, documented, falsifiable
prediction** about what should happen. §9.4 found that per-cell power specification is the
single change that breaks the linear baseline's hotspot localisation. IC-ThermBench's input
is a per-cell power map. So this is a test of an existing claim on independent data at ~300×
the sample count, not a fishing expedition.

**Framing, to be stated up front in any writeup:** we are not competing with their models.
We are supplying the row their table is missing, and reporting what it implies about the
metrics.

---

## 2. What the data actually is (verified from their repo, 2026-09-10)

| scope | CLI token | P | input channels (C-axis order) | samples |
|---|---|---|---|---|
| S2 | `level2` | 3 | `chiplet_power`, `grid_x`, `grid_y` | 15,000 |
| S3 | `level3` | 4 | S2 + `local_thermal_k` | 15,000 |
| S4 | `level4` | 7 | S3 + `ambient_K`, `h_w_m2k`, `r_convec_k_per_w` | 15,000 |
| S5 | `level5` | 7 | same schema, unseen Cases 11-15 | 5,000 |

Tensor contract: `input.mat["data"]` is `(B, P, Z, Y, X)` float32, `output.mat["data"]` is
`(B, Z, Y, X)` float32 in kelvin; the loader transposes to `(B, X, Y, Z, P)` → `(B, X, Y, Z)`.
**X = Y = 64, Z = 1**, so 4,096 output cells per sample. Data are HDF5-backed `.mat`.

### Channel semantics — measured from the downloaded data, 2026-09-10

Shapes confirmed exactly as documented: `input (B, P, 1, 64, 64)`, `output (B, 1, 64, 64)`,
with B = 15,000 / 15,000 / 15,000 / 5,000 for S2/S3/S4/S5. Inspecting the first 200 samples
of `level4` separates the genuinely spatial channels from the per-sample scalars:

| channel | spatial std (within sample) | std across samples | range | verdict |
|---|---|---|---|---|
| `chiplet_power` | 60.5 | 40.4 | 0 – 300 | **per-cell map** |
| `grid_x` | 12.9 | 5.4 | 0 – 59 | **per-cell, and varies per sample** |
| `grid_y` | 12.8 | 6.2 | 0 – 61 | **per-cell, and varies per sample** |
| `local_thermal_k` | 48.3 | 9.5 | 0.205 – 100 | **per-cell map** |
| `ambient_K` | 1.9e-05 | 8.8 | 298.3 – 328.1 | scalar per sample |
| `h_w_m2k` | 4.8e-04 | 5212 | 507.6 – 19,660 | scalar per sample |
| `r_convec_k_per_w` | 1.8e-09 | 0.0298 | 9.19e-04 – 0.185 | scalar per sample |

Three consequences, two of which correct assumptions made when this plan was first written:

1. **`grid_x`/`grid_y` are not degenerate.** They were flagged in §7 as probably constant
   across samples and therefore useless to a linear model. They are not: they vary both
   spatially and sample-to-sample, and appear to encode each case's grid geometry. This is
   how layout identity reaches the model, and a linear baseline can use it. Risk withdrawn.
2. **`ambient_K` enters exactly additively**, so ridge gets that dimension essentially for
   free — it is the `T_amb` term in `T = T_amb + G·Q`.
3. **`r_convec_k_per_w` is already supplied in the `1/h` form.** The benchmark hands every
   model the convective *resistance* directly. This is precisely the term
   `scripts/baselines.py::feature_vector` constructs by hand (`1/htc`) to keep ridge fair,
   and here it does not even need constructing.

**This sharpens the S4 prediction in §4 below.** The naive reading is that S4 is harder than
S3 because it varies more. The physics says the opposite for a *linear* model: S4's three
added channels are the additive-ambient term and the convective resistance, both of which
enter the linear solution exactly, whereas S3's addition (`local_thermal_k`) changes the
operator itself and cannot be represented linearly. **Ridge may well degrade less from S3→S4
than the neural models do**, which would be a distinctive and testable signature.

### Deterministic split

Split is index-based and never shuffled, so it is reproducible from the `.mat` files alone
without running their code:

```
test  = last 20% of all samples          -> 3,000  (300/case)
train = first 90% of the leading 80%     -> 10,800 (1,080/case)
val   = last 10% of the leading 80%      -> 1,200  (120/case)
```

Samples are stored case-interleaved so each segment stays case-balanced. **Do not restack by
case.** S5 is evaluation-only (zero-shot on all 5,000; few-shot uses first 500/case as the
adaptation pool, last 500/case as the common holdout).

Simulator is HotSpot, grid-based detailed-3D flow. S2 layouts come from ATPlace2.5D
(Cases 1-10); S3-S5 extend the same case conventions.

## 3. Published numbers we are adding a row to

| track | best (Therm-FM) | runner-up | best peak ΔT |
|---|---|---|---|
| S2 · layout | **0.4427** K RMSE | SAU-FNO 0.7028 | 0.2610 |
| S3 · + material | **0.7161** | U-FNO 0.8016 | 0.3111 |
| S4 · + boundary | **0.9334** | SAU-FNO 1.2158 | 0.4076 |
| S5 zero-shot | 15.51 | U-Net 19.10 | — |
| S5 10-shot | 2.73 | U-FNO 3.59 | — |

Their metric set: `rmse`, `mean_absolute_error`, `r2`, `max_absolute_error`, plus Top-50 MAE.
Denormalisation happens on CPU in float64 before metrics (`exp/exp_operator.py`).

---

## 4. Predictions, stated before running anything

These follow from the physics and from this repo's own prior results. Writing them down now
is the point: a prediction recorded after seeing the answer is worth nothing.

| scope | what varies | prediction | reasoning |
|---|---|---|---|
| **S2** | layout only; material and BC fixed | **Uncertain, and informative either way.** Strong if the operator is shared across layouts; clearly worse if not. | With fixed material and BC, `T = T_amb + G·Q` with `G` a fixed linear operator. If the 10 layouts change only *where power is injected*, one `G` suffices and ridge should be competitive. If they change the substrate/chiplet geometry, `G` differs per case and a single linear map cannot be right. |
| **S3** | + per-cell `local_thermal_k` | **Clear degradation.** | Conductivity variation changes the operator itself. `T` is linear in `Q` but *not* in `k`, so a single linear map in the inputs cannot represent it. This is §9.9/§9.11's territory. |
| **S4** | + `ambient_K`, `h`, `r_convec` | **Smaller degradation than S3→S4 shows for the neural models** (sharpened after inspecting the channels, see §2). | All three added channels are per-sample scalars that enter the linear solution *exactly*: ambient is the additive `T_amb` term, and `r_convec_k_per_w` is the convective resistance already in `1/h` form. Unlike S3's material variation, none of them changes the operator. |
| **S5** | unseen Cases 11-15 | **Failure**, comparable in kind to the neural models' 16.6× degradation. | Extrapolation to an unseen operator. Nothing in the training data constrains it. |
| **hotspot metrics, all scopes** | — | **Ridge localises poorly**, per §9.4. | Per-cell power is the regime §9.4 identified as breaking linear-baseline localisation. This is the specific prior claim under test. |

If S2→S4 degradation tracks "how far the scope departs from linearity in the inputs", that
is a mechanistic account of IC-ThermBench's own headline difficulty gradient which their
paper does not currently offer.

---

## 5. Baselines to run

Three, reported together with parameter counts so nobody has to guess whether the strong one
is over-parameterised:

1. **Learned Green's function (the physically-correct linear model).** A dense ridge operator
   from the 4,096-cell power map to the 4,096-cell temperature field, plus scalar features.
   ~16.8M parameters. Conduction genuinely *is* linear in the source, so this is the correct
   model class, not a strawman. This is the number that matters.
2. **PCA-restricted ridge.** Power field projected onto its leading components
   (`scripts/baselines.py::power_pca_features`, built for exactly this), then ridge. Far
   fewer parameters; guards against "your baseline only wins because it is huge".
3. **kNN / nearest-neighbour / mean.** Free, and the mean-field predictor has already
   embarrassed ridge once on hotspot recall (§9.12c, geometry6), so it is not a formality.

Cost, estimated from the tensor shapes: `AᵀA` is 4100² × 10,800 ≈ 1.8e11 FLOPs (~2 min with
BLAS); Cholesky plus 4,096 right-hand sides is comparable. **Minutes per scope on CPU, peak
RAM 1-2 GB, no GPU.**

---

## 6. Work plan

1. **Download and unpack** S2-S5 (~4.6 GB, Google Drive) into `data/ic-thermbench/`.
   `data/` is gitignored, so nothing here gets committed. *(started 2026-09-10)*
2. **Loader** for their tensor contract, reproducing the index split exactly. Verify against
   their `data_provider/data_loader.py` rather than reimplementing from the docs.
3. ~~**Metric parity harness.**~~ **DONE 2026-09-10, by a stronger route than planned.**
   The original plan was to reimplement their metrics and validate by reproducing a
   published number from a released checkpoint (a 44 GB download). Two things made that
   unnecessary:
   - Their `utils/metrics.py` is pure numpy, self-contained, MIT-licensed, and its own
     docstring says *"This is the benchmark's only metric implementation... Do not write a
     second 'equivalent' implementation."* It is now vendored verbatim at
     `third_party/ic_thermbench/metrics.py` (md5 recorded, licence included). Parity is by
     construction rather than by test, which is strictly stronger.
   - `scripts/ic_thermbench_data.py::verify_against_upstream` checks our loaded arrays
     against **their own loader and splitter** and requires exact equality. Run on S2:
     all six arrays (train/val/test × x/y) match exactly, with counts 10,800 / 1,200 /
     3,000 as documented.

   The checkpoint route would only have tested *their model's* correctness, which was never
   in question. Note also that normalisation does not enter our path at all: their metrics
   are computed after denormalisation on raw kelvin fields, our loader yields exactly those
   same kelvin arrays, and our baselines predict in kelvin directly.

   **Sanity check passed**: the trivial mean-field predictor scores 16.31 K RMSE on S2
   (against Therm-FM's 0.4427 K), confirming the benchmark is not solvable by a constant
   and that nothing is leaking through the split.
4. **Run the three baselines** on S2/S3/S4 (train/test), S5 zero-shot, and S5 few-shot at
   the same K values they report.
5. **Hotspot metrics** via `scripts/hotspot_eval.py`, including the peak well-posedness
   diagnostic (§9.12c) — their ETmax deliberately ignores location, so localisation distance
   is additive to what they report.
6. **Write up** into `docs/report.md`, comparing against §4's predictions and stating which
   were wrong.

Estimated effort: ~1 day, dominated by steps 2-3.

---

## 7. Risks

- **Step 3 is the load-bearing one.** If we cannot reproduce one of their published numbers
  from a released checkpoint, every baseline number we produce is uninterpretable and the
  experiment should stop there rather than be reported with a caveat.
- ~~**`grid_x` / `grid_y` are probably constant across samples.**~~ **Checked 2026-09-10 and
  withdrawn** — both vary spatially *and* across samples, and carry the per-case grid
  geometry (§2). They are usable features, not degenerate ones.
- **Fairness of the dense operator.** 16.8M parameters against FNO's few million. Mitigated
  by reporting baseline 2 alongside, and by reporting parameter counts openly.
- **Google Drive quota limits** can block the 4.6 GB download; may need a manual browser
  fetch.
- **We may simply lose.** Therm-FM at 0.4427 K RMSE on S2 is a strong number. If ridge is
  clearly beaten across all scopes, that is a real and publishable result too — it would be
  the first benchmark in this project's experience where the linear baseline does *not*
  suffice, and it would sharpen rather than weaken the central argument, which is about
  *checking*, not about linear models always winning.

---

## 8. What this is and is not

**Is:** a small, cheap, falsifiable test of a documented prior claim (§9.4) on independent
data, filling a verified gap in the field's current standard benchmark.

**Is not:** a competing benchmark, a new architecture, or a claim that linear models solve
3D-IC thermal prediction. §9.12d already retracted this project's one apparent neural win;
the same standard applies in the other direction.
