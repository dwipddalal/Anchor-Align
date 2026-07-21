#!/usr/bin/env bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32g
#SBATCH --gpu-bind=none
#SBATCH -J calvin_eval
#SBATCH -t 8:00:00
#SBATCH -o slurm-logs/slurm-%x-%j.out
#SBATCH -e slurm-logs/slurm-%x-%j.err

# CALVIN-ABC evaluation runner.
#
# Usage:
#   sbatch --export=ALL,CHECKPOINT=<path>,RUN_NOTE=<note> slurm/slurm_calvin_eval.slurm
#
# Required env-var inputs (set via sbatch --export=...):
#   CHECKPOINT  -- path to the *_chkpt directory produced by training.
#                  Must contain config.json, model.safetensors, action_head--*_checkpoint.pt,
#                  proprio_projector--*_checkpoint.pt, lora_adapter/.
#
# Optional env vars:
#   RUN_NOTE       -- appended to log dir name; default = basename of CHECKPOINT
#   CALVIN_PATH    -- override the calvin repo path (default = "calvin" symlink at repo root)
#   USE_WANDB      -- "True" to push results to wandb (default False — local logs only)
#
# Output directory: experiments/logs/calvin/<RUN_NOTE>/
#
# Notes:
# - Uses the `calvin` conda env (created via CALVIN/scripts/setup_env.sh).
# - PYTHONPATH includes the repo root so the calvin env can import the local prismatic package.
# - CALVIN_ROOT="calvin" is the symlink at the repo root (-> CALVIN/calvin).

set -euo pipefail

CONDA_SH="${CONDA_SH:-/sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vla-adapter-pt27}"  # torch 2.7 + calvin pkgs (see setup/envs/)
REPO_DIR="${REPO_DIR:?REPO_DIR must be set to the Anchor-Align repo root}"

: "${CHECKPOINT:?ERROR: CHECKPOINT not set. Use sbatch --export=ALL,CHECKPOINT=<path>,...}"

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint directory not found: $CHECKPOINT"
    exit 2
fi

CHECKPOINT_NAME=$(basename "$CHECKPOINT")
RUN_NOTE="${RUN_NOTE:-$CHECKPOINT_NAME}"
CALVIN_PATH_LOCAL="${CALVIN_PATH:-calvin}"
USE_WANDB="${USE_WANDB:-False}"
OUTPUT_DIR="${REPO_DIR}/experiments/logs/calvin/${RUN_NOTE}"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  CALVIN-ABC eval"
echo "  CHECKPOINT     : $CHECKPOINT"
echo "  RUN_NOTE       : $RUN_NOTE"
echo "  CALVIN_PATH    : $CALVIN_PATH_LOCAL"
echo "  USE_WANDB      : $USE_WANDB"
echo "  OUTPUT_DIR     : $OUTPUT_DIR"
echo "  Job: ${SLURM_JOB_NAME:-} (ID: ${SLURM_JOB_ID:-unknown}) on $(hostname)"
echo "================================================================"

module load cuda-compat/12.8 || true
export LD_LIBRARY_PATH=/sw/user/cudatoolkits/installs/cuda-12.6.1/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}

source "$CONDA_SH"
conda activate "$CONDA_ENV"

# Make the local prismatic + calvin_agent + calvin_env packages importable.
# vla-adapter-pt27 has all the prismatic deps + torch 2.7; the CALVIN packages
# are imported directly from CALVIN/calvin/calvin_{models,env}/ via PYTHONPATH.
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/CALVIN/calvin/calvin_models:${REPO_DIR}/CALVIN/calvin/calvin_env:${PYTHONPATH:-}"

# CALVIN_ROOT must point to the cloned calvin repo. The script reads
# os.environ['CALVIN_ROOT'] which it sets to "calvin" by default; we override
# with an absolute path to be safe (works regardless of cwd).
export CALVIN_ROOT="${REPO_DIR}/calvin"

# Headless EGL rendering for PyBullet (no X server). The calvin_env wrapper
# already handles EGL device selection; we just make sure no DISPLAY is set.
unset DISPLAY

# Wandb config (only used when USE_WANDB=True)
export WANDB_API_KEY=3ebbd9beeff577ba7467cc740546814eaaf929fe
export WANDB_ENTITY="lapis-4"
export WANDB_PROJECT="vla-adapter-calvin-eval"

cd "$REPO_DIR"

echo "Python: $(python -V)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import calvin_agent, calvin_env; print(f'calvin_agent at {calvin_agent.__file__}')"
python -c "import prismatic; print(f'prismatic at {prismatic.__file__}')"
nvidia-smi || exit 2

# Eval flags mirror training:
#   --use_l1_regression True
#   --num_images_in_input 2
#   --use_proprio True
#   --num_open_loop_steps 8 (matches NUM_ACTIONS_CHUNK=8 for CALVIN)
python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint "$CHECKPOINT" \
  --calvin_path "$CALVIN_PATH_LOCAL" \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --use_proprio True \
  --num_open_loop_steps 8 \
  --use_wandb $USE_WANDB \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  --run_id_note "$RUN_NOTE" \
  --local_log_dir "${REPO_DIR}/experiments/logs" \
  --log_dir "$OUTPUT_DIR" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo ""
echo "[$(date '+%F %T')] CALVIN eval complete."
echo "Output: $OUTPUT_DIR"
