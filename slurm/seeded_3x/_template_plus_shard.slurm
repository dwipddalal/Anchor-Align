#!/usr/bin/env bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700g
#SBATCH --gpu-bind=none
#SBATCH -t 02:30:00
#SBATCH -o slurm-logs/slurm-%x-%j.out
#SBATCH -e slurm-logs/slurm-%x-%j.err

# 3-seed Plus eval, single shard. Env vars: CHECKPOINT, CFG_TAG, SEED, START_TASK, END_TASK, SHARD_ID
set -euo pipefail
: "${CHECKPOINT:?ERROR: CHECKPOINT not set}"
: "${CFG_TAG:?ERROR: CFG_TAG not set}"
: "${SEED:?ERROR: SEED not set}"
: "${START_TASK:?ERROR: START_TASK not set}"
: "${END_TASK:?ERROR: END_TASK not set}"
: "${SHARD_ID:?ERROR: SHARD_ID not set}"
[ -d "$CHECKPOINT" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }

source "${CONDA_SH:-/sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh}"
conda activate "vla-adapter"
nvidia-smi || exit 2

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 MUJOCO_GL=egl
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

cd "${REPO_DIR:?REPO_DIR must be set to the Anchor-Align repo root}"
mkdir -p experiments/logs/3seed/libero_plus

LOG="experiments/logs/3seed/libero_plus/${CFG_TAG}_seed${SEED}_shard${SHARD_ID}--$(date +%Y%m%d_%H%M%S).log"
echo "=== [$(date '+%F %T')] Plus | cfg=${CFG_TAG} seed=${SEED} shard=${SHARD_ID} tasks=[${START_TASK},${END_TASK}] ===" | tee "$LOG"

python experiments/robot/libero_plus/run_libero_plus_eval_batched.py \
  --pretrained_checkpoint "$CHECKPOINT" \
  --base_suite_name libero_spatial \
  --perturbation_category all \
  --start_task_id $START_TASK --end_task_id $END_TASK \
  --batch_size 48 \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --use_l1_regression True --center_crop True --num_open_loop_steps 8 \
  --use_wandb False --seed $SEED \
  --run_id_note "3seed_${CFG_TAG}_seed${SEED}_shard${SHARD_ID}" \
  2>&1 | tee -a "$LOG"

echo "=== [$(date '+%F %T')] Done. Log: $LOG ==="
