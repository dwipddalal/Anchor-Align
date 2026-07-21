#!/usr/bin/env bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=180g
#SBATCH --gpu-bind=none
#SBATCH -J calvin_eval_batched
#SBATCH -t 2:00:00
#SBATCH -o slurm-logs/slurm-%x-%j.out
#SBATCH -e slurm-logs/slurm-%x-%j.err

# CALVIN-ABC batched eval (single-GPU, B parallel CalvinEnv subprocesses).
#
# Usage:
#   sbatch --export=ALL,CHECKPOINT=<path>,RUN_NOTE=<note>[,BATCH=8] slurm/slurm_calvin_eval_batched.slurm
#
# Resource sizing (GH200, B parallel envs share node):
#   --cpus-per-task=24  -> ~3 cpus per env worker (B=8) for PyBullet
#   --mem=64g           -> ~8g/env actual; 64g for proc workers + model + sim
#   --gpus-per-node=1   -> 1 GH200 (model is small enough for B=8 batched fwd)
#
# This is the PRODUCTION batched eval. Not parallel evals across nodes — that
# would be wasteful for 0.5B model. Single GPU + B=8 envs is the right shape.

set -euo pipefail

CONDA_SH="${CONDA_SH:-/path/to/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vla-adapter-pt27}"
REPO_DIR="${REPO_DIR:?REPO_DIR must be set to the Anchor-Align repo root}"

: "${CHECKPOINT:?ERROR: CHECKPOINT not set.}"
[ -d "$CHECKPOINT" ] || { echo "ERROR: checkpoint dir not found: $CHECKPOINT"; exit 2; }

CHECKPOINT_NAME=$(basename "$CHECKPOINT")
RUN_NOTE="${RUN_NOTE:-$CHECKPOINT_NAME}"
BATCH="${BATCH:-8}"
USE_WANDB="${USE_WANDB:-False}"
OUTPUT_DIR="${REPO_DIR}/experiments/logs/calvin/batched_${RUN_NOTE}"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  CALVIN-ABC BATCHED eval (B=$BATCH)"
echo "  CHECKPOINT : $CHECKPOINT"
echo "  RUN_NOTE   : $RUN_NOTE"
echo "  OUTPUT_DIR : $OUTPUT_DIR"
echo "  Job: ${SLURM_JOB_NAME:-} (ID: ${SLURM_JOB_ID:-?}) on $(hostname)"
echo "================================================================"

module load cuda-compat/12.8 || true
export LD_LIBRARY_PATH=/path/to/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/CALVIN/calvin/calvin_models:${REPO_DIR}/CALVIN/calvin/calvin_env:${PYTHONPATH:-}"
export CALVIN_ROOT="${REPO_DIR}/calvin"
unset DISPLAY

cd "$REPO_DIR"

echo "Python: $(python -V)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import calvin_agent, calvin_env; print('calvin OK')"
nvidia-smi || exit 2

python vla-scripts/evaluate_calvin_batched.py \
  --pretrained_checkpoint "$CHECKPOINT" \
  --calvin_path "calvin" \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --use_proprio True \
  --num_open_loop_steps 8 \
  --batch_size $BATCH \
  --num_sequences 1000 \
  --use_hi3_ensemble ${USE_HI3:-False} \
  --use_wandb $USE_WANDB \
  --run_id_note "$RUN_NOTE" \
  --local_log_dir "${REPO_DIR}/experiments/logs" \
  --log_dir "$OUTPUT_DIR" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo ""
echo "[$(date '+%F %T')] CALVIN batched eval complete."
echo "Output: $OUTPUT_DIR"
