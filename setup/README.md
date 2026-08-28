# Setup — installation and SLURM configuration

This directory contains helpers for setting up the repo on a fresh machine.

## Files

- [`install_libero_variants.sh`](install_libero_variants.sh) — clone + pip-install one of the three LIBERO variants (Standard / PRO / Plus)
- [`envs/`](envs/) — conda environment snapshots (`vla-adapter-*-requirements.txt`) for exact-version reproduction

## Adapting the SLURM examples

The CALVIN files in `slurm/` are starting points, not drop-in jobs for every
cluster. They contain no credentials, private filesystem paths, account, or
partition. Runtime paths are supplied through environment variables, while
resource and time requests should be reviewed for your scheduler.

Set the required runtime paths before submitting:

```bash
export REPO_DIR="$(pwd)"                 # required — path to this repo
export CONDA_SH="/path/to/conda.sh"       # required — conda activation script
# export HF_HOME="/path/to/hf-cache"      # optional — Hugging Face cache root
# export CUDA_COMPAT_MODULE="module-name" # optional — site-specific module
# export CUPTI_LIB_DIR="/path/to/CUPTI/lib64" # optional — site-specific library path
```

Any variable marked "required" will cause the SLURM script to bail out with a
clear error if unset — no silent failures on someone else's cluster.

The examples do not choose an account or partition. If your cluster requires
them, pass them at submission time:

```bash
sbatch -A <account> -p <partition> <script>
```

The checked-in CPU, memory, GPU, and wall-time requests are example values and
can also be overridden or edited for the target cluster. Separately,
`setup_pt27_env.sh` targets PyTorch 2.7 with CUDA 12.8 on aarch64/GH200; adapt
its package versions for other architectures or CUDA installations.

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
