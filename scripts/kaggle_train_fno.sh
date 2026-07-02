#!/usr/bin/env bash
# kaggle_train_fno.sh — Train all 8 geometries across 2 T4 GPUs on Kaggle.
#
# Strategy: geometry-level parallelism.
# Each geometry trains independently, so we dispatch 4 geometries to GPU:0
# and 4 to GPU:1 as sequential processes on each GPU.
# Total wall time ≈ ceil(8/2) * per_geometry_time ≈ 4 × 6 min = ~24 min
# (vs 48 min sequential on one GPU).
#
# Usage:
#   bash scripts/kaggle_train_fno.sh --data /kaggle/input/thermal-benchmark/data
#   bash scripts/kaggle_train_fno.sh --data /kaggle/input/thermal-benchmark/data \
#       --channels 48 --blocks 6 --attention
#   bash scripts/kaggle_train_fno.sh --data /kaggle/input/thermal-benchmark/data \
#       --channels 64 --attention --patience 5
#
# Logs: checkpoints/fno/logs/geometry*.log
# Checkpoints: checkpoints/fno/<geometry_name>/<geometry_name>_best.pt

set -euo pipefail

# ── Parse this script's args ─────────────────────────────────────────────────
DATA=""
OUTPUT="checkpoints/fno"
FNO_EXTRA_ARGS=""  # forwarded to train_fno.py unchanged

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)    DATA="$2";    shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        *)  FNO_EXTRA_ARGS="$FNO_EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$DATA" ]]; then
    echo "ERROR: --data <path> is required" >&2
    exit 1
fi

mkdir -p "$OUTPUT/logs"

# ── Check GPU count ───────────────────────────────────────────────────────────
N_GPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "GPUs detected: $N_GPU"

GPU0_GEOMS="geometry1 geometry2a geometry3 geometry4"
GPU1_GEOMS="geometry2b geometry2c geometry5 geometry6"

train_on_gpu() {
    local gpu_id="$1"
    local geoms="$2"
    for geom in $geoms; do
        echo "[GPU $gpu_id] Starting $geom at $(date '+%H:%M:%S')"
        CUDA_VISIBLE_DEVICES=$gpu_id python scripts/train_fno.py \
            --geometry "$geom" \
            --data "$DATA" \
            --output "$OUTPUT" \
            --compile \
            $FNO_EXTRA_ARGS \
            > "$OUTPUT/logs/${geom}.log" 2>&1
        echo "[GPU $gpu_id] Finished $geom at $(date '+%H:%M:%S')"
    done
}

if [[ "$N_GPU" -ge 2 ]]; then
    echo "Running 2-GPU parallel: GPU0=$GPU0_GEOMS | GPU1=$GPU1_GEOMS"
    train_on_gpu 0 "$GPU0_GEOMS" &
    PID0=$!
    train_on_gpu 1 "$GPU1_GEOMS" &
    PID1=$!
    wait $PID0 && wait $PID1
else
    echo "Only $N_GPU GPU(s) found — running all 8 geometries sequentially on GPU 0"
    train_on_gpu 0 "$GPU0_GEOMS $GPU1_GEOMS"
fi

echo "All geometries done. Checkpoints in $OUTPUT/"
ls -lh "$OUTPUT"/*/geometry*_best.pt 2>/dev/null || echo "(no checkpoints found — check logs)"
