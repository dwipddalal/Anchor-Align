#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VARIANT="${VARIANT:-baseline}"
GPUS="${GPUS:-0,1,2,3}"
STEPS="${STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

case "$VARIANT" in
  baseline)
    CONFIG="$REPO_ROOT/configs/mug_baseline.yaml"
    ;;
  mse01_alv7)
    CONFIG="$REPO_ROOT/configs/mug_mse01_alv7.yaml"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT (expected baseline or mse01_alv7)" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
DETECTED_NUM_GPUS="${#GPU_IDS[@]}"
NUM_GPUS="${NUM_GPUS:-$DETECTED_NUM_GPUS}"
if [[ "$NUM_GPUS" -ne "$DETECTED_NUM_GPUS" ]]; then
  echo "NUM_GPUS=$NUM_GPUS does not match GPUS=$GPUS ($DETECTED_NUM_GPUS IDs)" >&2
  exit 2
fi

export BASE_VLM="${BASE_VLM:-Qwen/Qwen2.5-VL-3B-Instruct}"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
export EEF_XYZ_DELTA_CACHE="${EEF_XYZ_DELTA_CACHE:-$REPO_ROOT/data/eef_deltas_cache_mug.pkl}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-starvla-realworld}"

STARVLA_TMP_ROOT="${STARVLA_TMP_ROOT:-/tmp/starvla-realworld-${USER:-user}}"
mkdir -p "$STARVLA_TMP_ROOT/triton" "$OUTPUT_ROOT"
export TMPDIR="$STARVLA_TMP_ROOT"
export TRITON_CACHE_DIR="$STARVLA_TMP_ROOT/triton"

cd "$REPO_ROOT"
python scripts/preflight.py --variant "$VARIANT" --num-gpus "$NUM_GPUS"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

echo "Starting xArm7 mug training: variant=$VARIANT gpus=$GPUS steps=$STEPS save_interval=$SAVE_INTERVAL"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml \
  --num_processes "$NUM_GPUS" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --trainer.max_train_steps "$STEPS" \
  --trainer.save_interval "$SAVE_INTERVAL" \
  "$@"
