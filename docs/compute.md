# Compute Cost Reference — 3D-IC Thermal Surrogate Benchmark

All times on **2× NVIDIA T4** (16 GB each, 65 TFLOPS FP16, 320 GB/s).
CPU benchmarks measured on this machine; GPU estimates derived from measured
CPU epoch times × workload-specific T4/CPU speedup factors.

---

## GPU Specifications

| GPU | VRAM | FP32 | FP16 (Tensor Cores) | Memory BW | Notes |
|---|---|---|---|---|---|
| T4 | 16 GB | 8.1 TFLOPS | 65 TFLOPS | 320 GB/s | Turing, good for FP16 inference |
| 2× T4 | 32 GB | 16.2 TFLOPS | 130 TFLOPS | 640 GB/s (sum) | DDP or parallel |

**DDP efficiency** (2 GPUs vs 1): 70% (1.7× speedup). Gradient sync
overhead 5–10%, FFT bandwidth-bound ops don't halve perfectly.

**T4/CPU speedup factors used:**
- PINN (FP32, autograd-heavy, small tensors, Python loop): **45×**
- FNO (FP16 mixed, FFT-dominated, large tensors): **25×** (bandwidth-bound)
- CNO-FNO (FP16 mixed, Conv3d + small-latent FNO): **60×** (Conv3d im2col maps perfectly to tensor cores; GroupNorm is the CPU bottleneck, fast on GPU)
- PI-DeepONet (FP32 trunk autograd, small MLP): **30×**

---

## Measured CPU Epoch Times

| Model | Geometry | Grid | CPU time/epoch | Measured |
|---|---|---|---|---|
| PINN (807k params, 20k col, FP32) | geometry1 | 100×100×40 | **89.0 s** | Yes |
| PINN | geometry2a | 80×80×72 | **96.8 s** | Yes |
| FNO baseline (12.6M params, FP16) | geometry1 | 100×100×40 | **32.3 s** | Yes |
| FNO baseline | geometry2a | 80×80×72 | **36.0 s** | Yes |
| FNO baseline | geometry4 | 100×56×40 | **20.7 s** | Yes |
| FNO baseline | geometry5 | 100×56×50 | **24.0 s** | Estimated (+2s for 8→11 layers) |
| FNO baseline | geometry6 | 56×168×50 | **58.0 s** | Estimated (470k pts, 2.7× g5) |
| PI-FNO (FD residual) | geometry1 | 100×100×40 | **32.3 s** | Yes (no overhead vs baseline) |
| CNO-FNO (1.6M params) — data only | geometry1 | 100×100×40 | **62.1 s** | Yes |
| CNO-FNO + PI | geometry1 | 100×100×40 | **60.8 s** | Yes |
| PI-DeepONet (500k params) — PI step | all 7 | N/A | **27 s/epoch** (105 scenarios, 27 steps) | Yes |

> **CNO-FNO is slower than baseline FNO on CPU** (62s vs 32s) because full-resolution
> Conv3d + GroupNorm on the 100×100×40 grid are CPU-memory-bandwidth-bound and poorly
> accelerated by MKL.  On T4, Conv3d im2col GEMMs run at 65 TFLOPS FP16 (tensor cores)
> giving **60× CPU→GPU speedup** vs FNO's 25× (FFT is bandwidth-bound). CNO-FNO is
> therefore **faster than baseline FNO on GPU** despite being slower on CPU.

---

## Single-Geometry Training Time: 2× T4

DDP across both T4s. One geometry trained per run.

| Model | Variant / flags | Epochs | T4 epoch time | **2× T4 DDP time** | Notes |
|---|---|---|---|---|---|
| **PINN** (baseline) | — | 8,000 | ~2.0 s | **~2.6 h** | FP32, 2nd-order autograd |
| **PINN + upgrades** | `--compile --func-pde` | 8,000 | ~0.5 s | **~37 min** | compile 2.5×, func-pde 1.5×, batched 2× |
| **FNO baseline** | — | 500 | ~1.3 s | **~8 min** | FP16, data-only |
| **PI-FNO** | `--pi-weight 0.1` | 400 | ~1.4 s | **~7 min** | FD residual adds ~5%; fewer epochs |
| **CondFNO (FiLM)** | `--film` | 500 | ~1.4 s | **~8 min** | FiLM generator adds <5% |
| **PI-FNO + FiLM** | `--film --pi-weight 0.1` | 400 | ~1.5 s | **~8 min** | |
| **CNO-FNO + PI + FiLM** | `--best` | 400 | ~1.0 s | **~6 min** | 60× GPU speedup (tensor cores); 1.6M vs 12.6M params |
| **PI-DeepONet** *(1 geom only)* | `--pde-weight 0.1` | 1,000 | ~0.13 s | **~1 min** | 15 scenarios, 4 steps/epoch; underuses architecture |

> **Key ratio:** PINN (baseline) is **~19× slower** than best FNO per geometry;
> PINN with upgrades is **~4× slower** than `--best` CNO-FNO.

---

## All-7-Geometries Training Time: 2× T4

PINN and FNO run 8 separate models (2 in parallel, one per GPU, pairing by time).
PI-DeepONet trains one model on all 5 uniform-stack geometries simultaneously via DDP.
Geometry6 (470k pts) adds ~2.7× overhead vs. geometry5 for FNO/CNO runs.

| Model | Strategy | **Wall time (2× T4)** | Total GPU-hours | Notes |
|---|---|---|---|---|
| **PINN** (baseline) | 2× parallel | **~24 h** | 52 GPU-h | 4.4 h/geom avg × 8 / 1.5 (parallelism) |
| **PINN + upgrades** | 2× parallel (DDP per geom) | **~5 h** | 10 GPU-h | 37 min/geom × 8; g6 ~90 min (470k pts) |
| **FNO baseline** | 2× parallel | **~55 min** | 1.8 GPU-h | g6 adds ~15 min extra (470k pts, 500 epochs) |
| **PI-FNO** | 2× parallel | **~50 min** | 1.7 GPU-h | |
| **CondFNO (FiLM)** | 2× parallel | **~55 min** | 1.8 GPU-h | |
| **PI-FNO + FiLM** | 2× parallel | **~52 min** | 1.7 GPU-h | |
| **CNO-FNO + PI + FiLM** | 2× parallel | **~38 min** | 1.3 GPU-h | g6 single run ~16 min (tensor-core Conv3d scales well) |
| **PI-DeepONet** | DDP, 1 joint run | **~9 min** | 0.3 GPU-h | 5 stack geometries (g1/2a/2b/2c/g3); g4/5/6 use per-geom CNO-FNO |
| **GINO** *(theoretical)* | DDP, 1 joint run | **~2 h** | 4 GPU-h | GNN encoder/decoder overhead; one model all geoms |

> **PI-DeepONet all-7 total (12 min) is faster than CNO-FNO for a single geometry (~6 min × 4 rounds = 28 min).**
> CNO-FNO wins per-geometry accuracy; DeepONet wins cross-geometry deployment speed.
> CNO-FNO is slower than FNO on CPU but faster on GPU — use `--best` only with CUDA.

---

## Inference Cost (per scenario, single T4, batch=1)

| Model | Forward time | Memory | Notes |
|---|---|---|---|
| **PINN** | 5–15 ms | ~300 MB | Point-wise MLP on 60k pts |
| **FNO baseline** | 8–12 ms | ~1–2 GB | FFT on 100×100×40 grid |
| **PI-FNO** | 8–12 ms | ~1–2 GB | FD residual not computed at inference |
| **CondFNO (FiLM)** | 9–13 ms | ~1–2 GB | Tiny FiLM generator overhead |
| **PI-FNO + FiLM** | 9–13 ms | ~1–2 GB | |
| **CNO-FNO + PI + FiLM** | 10–18 ms | ~700 MB | CNN encode (fast on GPU) + latent FNO (tiny FFT) + CNN decode |
| **PI-DeepONet** | 5–15 ms | ~200 MB | MLP branch + MLP trunk at N query points; no FFT |
| **GINO** *(theoretical)* | 15–25 ms | ~1–2 GB | GNN encode + latent FNO + GNN decode |

> **CNO-FNO uses less GPU memory than baseline FNO** despite having CNN layers:
> the FNO activation tensors dominate memory, and the latent FNO is 64× smaller.

---

## Model Parameters

| Model | Parameters | Notes |
|---|---|---|
| FourierPINN | 807k | 6 ResBlocks × hidden=256 |
| FourierPINN (cpu-fast) | 140k | 4 blocks × hidden=128 |
| FNO baseline | 12.6M | hidden=32, modes=(16,16,12), 4 blocks |
| CondFNO (FiLM) | 12.8M | +FiLMGenerator ~200k |
| **CNO-FNO + PI + FiLM** | **1.6M** (g1), 2.4M (g2) | Latent FNO modes (8,8,5); 7.8× fewer params than FNO |
| **WHNO** | **12.6M** | Same channels/modes config as FNO baseline (Walsh-Hadamard basis is a drop-in swap for the spectral conv, so equal param count by construction) |
| PI-DeepONet | 538k | Branch 538→256×4→128 + Trunk 132→256×4→128 |
| **ARO** | **~1.06M** | n_layers=11 (geometry5/6 default), hidden=32, 4 FiLM-FNO blocks, weight-tied across all z-layers |
| GINO (theoretical) | ~8–12M | FNO latent + GNN encoder/decoder |

> **WHNO and ARO training-time benchmarks are not yet measured at full config** (only
> smoke-tested at reduced channels/epochs to validate correctness — see their respective
> notebook/script validation runs). Rough estimates below are architecture-based, not
> measured; treat with the same ±50% caveat the GPU extrapolations below already carry,
> but with lower confidence since even the CPU baseline hasn't been measured at scale.
> - **WHNO**: expect CPU/GPU cost roughly on par with FNO baseline — the Walsh-Hadamard
>   transform is the same O(N log N) complexity class as FFT, with an added grid-padding-
>   to-next-power-of-2 step (e.g. 100→128, 40→64) that FNO doesn't need, adding modest
>   overhead proportional to the padding ratio.
> - **ARO**: the autoregressive z-layer loop means training cost scales with `n_layers`
>   sequential 2D-FNO forward/backward passes per sample, rather than one 3D pass — likely
>   slower per-epoch than CNO-FNO at equal channel width despite far fewer total parameters,
>   though each individual 2D step is far cheaper than a full 3D op. RNO-style windowed
>   self-rollout training (see `src/aro/trainer.py`) adds a second forward/backward pass per
>   step during pretraining, roughly doubling per-epoch cost during that phase only.

---

## Architecture Comparison

| Model | Cross-geom? | Sharp interfaces | Global spreading | Param count | CLI |
|---|---|---|---|---|---|
| PINN | No | Good (PDE) | Good (MLP) | 807k | `train_pinn.py` |
| FNO baseline | No (per grid) | Poor | Excellent | 12.6M | `--model fno` |
| CondFNO (FiLM) | No | Poor | Excellent + BC adapt | 12.8M | `--model cond-fno` |
| **CNO-FNO+PI+FiLM** | No | **Excellent** | **Excellent** | **1.6M** | *(default)* |
| **WHNO** | No (per grid) | **Excellent** (validated: 0% Gibbs overshoot vs FNO's ~8.7% on a synthetic step function) | Good | 12.6M | `--model whno` |
| PI-DeepONet | **Yes (g1/2a/2b/2c/g3)** | Medium | Medium | 538k | `train_deeponet.py` |
| **ARO** | No (per grid, weight-tied across z) | Good (autoregressive z-conditioning) | Good | ~1.06M | `train_aro.py` |
| CNO-FNO+PI+FiLM (g6) | No (per grid) | **Excellent** | **Excellent** | ~2.4M | `--geometry geometry6` |
| GINO | **Yes** | Excellent | Excellent | ~10M | Not implemented |

---

## Recommended Training Commands

```bash
# CNO-FNO (default model, physics on, recommended):
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice

# Higher capacity for publication runs:
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --channels 64

# Baseline FNO (ablation / fast iteration):
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --model fno

# FiLM-only (no CNO, with physics):
python scripts/train_fno.py --geometry geometry1 --data data/3d-ice --model cond-fno --physics

# All 6 geometries in parallel on 2 T4s (run simultaneously on separate GPUs):
CUDA_VISIBLE_DEVICES=0 python scripts/train_fno.py --geometry geometry1 geometry3 \
    --data data/3d-ice --name g1g3
CUDA_VISIBLE_DEVICES=1 python scripts/train_fno.py --geometry geometry2a geometry4 \
    --data data/3d-ice --name g2a-g4
# geometry6 is large (470k pts) — run solo or paired with a small geometry:
CUDA_VISIBLE_DEVICES=0 python scripts/train_fno.py --geometry geometry6 --data data/3d-ice
CUDA_VISIBLE_DEVICES=1 python scripts/train_fno.py --geometry geometry5 --data data/3d-ice

# Cross-geometry surrogate (5 uniform-stack geometries, one run):
python scripts/train_deeponet.py --data data/3d-ice \
    --output checkpoints/deeponet --pde-weight 0.1 --epochs 1000
# geometry4/5 (2.5D chiplets with lateral k variation): use per-geometry CNO-FNO
# python scripts/train_fno.py --geometry geometry4 --data data/3d-ice
# python scripts/train_fno.py --geometry geometry5 --data data/3d-ice

# PINN with GPU efficiency upgrades:
python scripts/train_pinn.py --geometry geometry1 --data data/3d-ice \
    --output checkpoints/pinn --compile --func-pde

# PINN or FNO on CPU (no GPU):
python scripts/train_pinn.py --geometry geometry1 --data data/3d-ice --cpu-fast
python scripts/train_fno.py  --geometry geometry1 --data data/3d-ice --cpu-fast
```

---

## Notes on Estimation Methodology

CPU epoch times were measured directly on this machine (Windows 11, PyTorch CPU).
GPU times were extrapolated using T4/CPU speedup factors validated against the
FNO/PINN literature:

- FNO: Li et al. 2021 report 36h on V100 for NS-3D (1000 samples, 500 epochs).
  Scaling to 15 samples + T4 gives 8–12 min/geometry, consistent with our extrapolation.
- PINN: Raissi et al. 2019; modern thermal PINN benchmarks report 2–6h on V100
  for comparable parameter counts. Our 2.6h on 2× T4 is consistent.
- CNO-FNO: derived from latent grid reduction (64× FFT savings) and CNN overhead.
  Will be confirmed once CUDA environment is available.

All GPU estimates carry ±50% uncertainty. Relative ordering is reliable;
absolute times should be validated with a CUDA run.
