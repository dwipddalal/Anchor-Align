#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${CKPT_ANCHOR:?ERROR: CKPT_ANCHOR not set (path to anchor-align checkpoint)}"
CKPT_BASELINE="${CKPT_BASELINE:-}"  # optional: set to also run the action-only baseline comparison arm
[ -d "$CKPT_ANCHOR" ] || { echo "ERROR: anchor checkpoint missing: $CKPT_ANCHOR"; exit 1; }
if [ -n "$CKPT_BASELINE" ] && [ ! -d "$CKPT_BASELINE" ]; then
  echo "ERROR: baseline checkpoint missing: $CKPT_BASELINE"; exit 1
fi

SEEDS=(21)
TPL_DIR="slurm/seeded_3x"
TPL_STD="$TPL_DIR/_template_spatial_std.slurm"
TPL_PRO="$TPL_DIR/_template_pro.slurm"
TPL_PLUS="$TPL_DIR/_template_plus_shard.slurm"

PLUS_SHARDS=(
  "0:0:600"
  "1:601:1200"
  "2:1201:1800"
  "3:1801:2401"
)

JOB_IDS=()

submit_one() {
  local jobname="$1" tpl="$2"; shift 2
  local exports="$*"
  local jid
  jid=$(sbatch --parsable --job-name="$jobname" --export="ALL,$exports" "$tpl") || { echo "FAILED: $jobname"; return 1; }
  echo "  $jid  $jobname"
  JOB_IDS+=("$jid")
}

CONFIGS=(anchor)
if [ -n "$CKPT_BASELINE" ]; then CONFIGS+=(baseline); fi

for cfg in "${CONFIGS[@]}"; do
  if [ "$cfg" = "anchor" ]; then
    CKPT="$CKPT_ANCHOR"; TAG="anc_v7kl01_10k"
  else
    CKPT="$CKPT_BASELINE"; TAG="bl_action_10k"
  fi
  echo "=== [$cfg] $TAG ==="
  for SEED in "${SEEDS[@]}"; do
    submit_one "3s_${TAG}_std_s${SEED}" "$TPL_STD" \
      "CHECKPOINT=$CKPT,CFG_TAG=$TAG,SEED=$SEED"

    for PERT in lan object swap; do
      submit_one "3s_${TAG}_${PERT}_s${SEED}" "$TPL_PRO" \
        "CHECKPOINT=$CKPT,CFG_TAG=$TAG,SEED=$SEED,PERT=$PERT"
    done

    for shard_spec in "${PLUS_SHARDS[@]}"; do
      IFS=':' read -r SHARD_ID START_TASK END_TASK <<< "$shard_spec"
      submit_one "3s_${TAG}_plus_s${SEED}_sh${SHARD_ID}" "$TPL_PLUS" \
        "CHECKPOINT=$CKPT,CFG_TAG=$TAG,SEED=$SEED,SHARD_ID=$SHARD_ID,START_TASK=$START_TASK,END_TASK=$END_TASK"
    done
  done
done

echo
echo "=== Submitted ${#JOB_IDS[@]} jobs ==="
printf '%s\n' "${JOB_IDS[@]}" > "$TPL_DIR/submitted_jobids.txt"
echo "Job IDs saved to $TPL_DIR/submitted_jobids.txt"
