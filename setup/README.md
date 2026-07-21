# Setup — installation and SLURM configuration

This directory contains helpers for setting up the repo on a fresh machine.

## Files

- [`install_libero_variants.sh`](install_libero_variants.sh) — clone + pip-install one of the three LIBERO variants (Standard / PRO / Plus)
- [`envs/`](envs/) — conda environment snapshots (`vla-adapter-*-requirements.txt`) for exact-version reproduction

## Running SLURM jobs on your own cluster

All SLURM templates in `slurm/` are **parameterized via env vars** so they run on any cluster without editing. Set these once per session before submitting jobs:

```bash
export REPO_DIR="$(pwd)"                            # required — path to this repo
export HF_HOME="$HOME/.cache/huggingface"           # optional — HF cache root
export LIBERO_PRO_ROOT="../libero-variants/LIBERO-PRO"     # required for PRO evals
export LIBERO_PLUS_ROOT="../libero-variants/LIBERO-plus"   # required for Plus evals
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"  # optional — conda activation script for your cluster
```

Any variable marked "required" will cause the SLURM script to bail out with a
clear error if unset — no silent failures on someone else's cluster.

The `#SBATCH --account=YOUR_ACCOUNT` and `-p <partition>` header lines are the
one thing you must adapt: pass your own allocation with
`sbatch -A <account> -p <partition> <script>` (command-line flags override the
in-file directives) or edit the headers once.

## Per-benchmark job submission

### Reproduce the 3-seed LIBERO Spatial campaign

```bash
export REPO_DIR="$(pwd)"
export LIBERO_PRO_ROOT="../libero-variants/LIBERO-PRO"

# Download the Spatial checkpoint (once)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Dwipz/Anchor-Align',
    allow_patterns='libero-spatial/*',
    local_dir='./checkpoints',
)
"

# Optional: also export CKPT_BASELINE (an action-only baseline ckpt) to run the
# comparison arm; the paper's baseline is documented in results/3seed_registry.md.

CKPT_ANCHOR="$(pwd)/checkpoints/libero-spatial" \
bash slurm/seeded_3x/submit_all.sh
```

By default `submit_all.sh` runs a single seed (seed=21) as a smoke test — edit
the `SEEDS=(21)` array to `SEEDS=(7 21 42)` for the full 3-seed campaign.

### Run a one-shot eval

Eval jobs are plain python commands (see the README's Evaluation section and
`REPRODUCE.md`); wrap them in your cluster's batch system as needed. Example:
LIBERO-Goal PRO eval on the goal checkpoint from HF.

```bash
export LIBERO_PRO_ROOT="../libero-variants/LIBERO-PRO"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero_pro/run_libero_pro_eval.py \
  --pretrained_checkpoint Dwipz/Anchor-Align/libero-goal \
  --base_suite_name libero_goal \
  --perturbation_type lan \
  --use_proprio True --num_images_in_input 2 --use_pro_version True
```

The loader accepts `org/repo/subfolder` HF specs directly and auto-downloads
on first use.

## SLURM output logs

All templates now write output to `slurm-logs/slurm-<jobname>-<jobid>.out` in
the current working directory (was previously an absolute Delta path). Make
sure `slurm-logs/` exists before submitting:

```bash
mkdir -p slurm-logs
```

Or add `#SBATCH --chdir=<something>` if your cluster requires the output path
to be pre-existing.
