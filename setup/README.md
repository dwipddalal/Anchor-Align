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
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"  # required — conda activation script for your cluster
# Optional site-specific CUDA configuration:
export CUDA_COMPAT_MODULE="cuda-compat/12.8"
export CUPTI_LIB_DIR="/path/to/cuda/extras/CUPTI/lib64"
```

Any variable marked "required" will cause the SLURM script to bail out with a
clear error if unset — no silent failures on someone else's cluster.

The templates retain an `#SBATCH --account=YOUR_ACCOUNT` placeholder and do
not hard-code a partition. Pass both values for your cluster with
`sbatch -A <account> -p <partition> <script>`, or edit the account placeholder.

## Run a one-shot evaluation

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

All templates write output to `slurm-logs/slurm-<jobname>-<jobid>.out` in the
current working directory. Make sure `slurm-logs/` exists before submitting:

```bash
mkdir -p slurm-logs
```

Or add `#SBATCH --chdir=<something>` if your cluster requires the output path
to be pre-existing.
