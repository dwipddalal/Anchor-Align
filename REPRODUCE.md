# Reproducing Anchor-Align Numbers

This document tells you exactly how to reproduce the four release checkpoints' numbers and what tolerance you should expect on a different machine. The goal is verification, not bit-exactness.

---

## TL;DR

1. `snapshot_download` a checkpoint from [🤗 Dwipz/Anchor-Align](https://huggingface.co/Dwipz/Anchor-Align).
2. Run the appropriate eval script from this repo.
3. Pipe the resulting log/JSON through [`scripts/verify_reproduction.py`](scripts/verify_reproduction.py) — it compares against `results/3seed_registry.json` and prints ✅ / ⚠️ / ❌ based on tolerance.

If the verification passes, your setup reproduces our numbers within statistical variance. If it doesn't, the script tells you exactly which metric drifted and by how much.

---

## Reference environment

Our reported numbers were produced on:

| | |
|---|---|
| GPU | NVIDIA GH200 120GB (aarch64) |
| CUDA | 12.4 (driver 550.163.01) |
| cuDNN | via PyTorch 2.7 wheel |
| PyTorch | 2.7 (aarch64 wheel; 2.2 works too — see `pyproject.toml`) |
| Python | 3.10.16 |
| LIBERO | commit at repo tip of [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| LIBERO-PRO fork | needed on `PYTHONPATH` for PRO evals |
| LIBERO-Plus fork | needed for Plus evals |
| Determinism flags set | `torch.backends.cudnn.deterministic=True`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"` |

**Cross-machine variance you should expect on a different setup:**

| Change | Typical drift | Notes |
|---|---|---|
| Same GH200, same PyTorch, same seed | ±0.5 pp | This is our within-seed floor |
| Same GH200, different seed | ±1.0 pp | Our 3-seed campaign shows σ ≈ 0.3–0.6 |
| Different GPU (H100, A100, RTX 4090) | ±1.5–2.5 pp | Fused-kernel numerics differ by GPU family |
| Different CUDA/PyTorch minor version | ±1.0 pp | Sometimes a cuDNN algorithm swap |
| **Total realistic budget** | **±2 pp** | Use this as your pass/fail threshold |

The `verify_reproduction.py` script uses a **±2.0 pp warning band** and a **±3.0 pp failure band** by default. Both are configurable via `--warn-tol` and `--fail-tol`.

---

## Two ways to point at a checkpoint

The eval scripts (via `experiments/robot/openvla_utils.py`) accept three checkpoint spec formats. Pick whichever is convenient:

| Spec | Meaning | Notes |
|---|---|---|
| `./checkpoints/libero-spatial` | Local directory | Fastest if you already downloaded |
| `Dwipz/Anchor-Align/libero-spatial` | HF Hub repo + subfolder | Auto-downloads only that subfolder into the HF cache on first use |
| `Dwipz/Anchor-Align` | HF Hub repo (top-level) | Errors out with a helpful list of available subfolders |

You do **not** need to run `snapshot_download` explicitly — passing the `org/repo/subfolder` form to `--pretrained_checkpoint` does it automatically and caches for reuse.

## Per-benchmark reproduction recipe

### LIBERO Spatial (headline result — the 3-seed campaign)

**Expected numbers** (3-seed mean, from `results/3seed_registry.md`):

| Metric | Mean ± stderr | Range |
|---|---|---|
| Standard SR | 97.87 ± 0.27 | 97.6% – 98.4% |
| PRO lan | 97.00 ± 0.31 | 96.6% – 97.6% |
| PRO object | 96.07 ± 0.35 | 95.4% – 96.6% |
| PRO swap | 22.53 ± 0.24 | 22.2% – 23.0% |
| Plus overall | 90.30 ± 0.16 | 90.0% – 90.5% |

**Command (one seed):**

```bash
# 1. Download the Spatial checkpoint
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Dwipz/Anchor-Align',
    allow_patterns='libero-spatial/*',
    local_dir='./checkpoints',
)
"

# 2. Standard SR (single seed)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_batched.py \
  --pretrained_checkpoint ./checkpoints/libero-spatial \
  --task_suite_name libero_spatial \
  --num_parallel_envs 50 --save_videos False \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --use_l1_regression True --center_crop True --num_open_loop_steps 8 \
  --use_wandb False --seed 7 --use_seed_in_env True \
  --run_id_note repro_spatial_std_seed7 \
  2>&1 | tee /tmp/repro_spatial_std_seed7.log

# 3. Verify
python scripts/verify_reproduction.py \
  --config anc_v7kl01_10k --benchmark spatial_std \
  --log /tmp/repro_spatial_std_seed7.log
```

**Command (full 3-seed campaign):**

```bash
# Set the env-var defaults (or export them in your shell profile)
export REPO_DIR="$(pwd)"
export LIBERO_PRO_ROOT="../libero-variants/LIBERO-PRO"   # see setup/install_libero_variants.sh

# Submit the anchor-align arm. CKPT_BASELINE is optional: set it only if you
# also have an action-only baseline checkpoint to run the comparison arm.
CKPT_ANCHOR="$(pwd)/checkpoints/libero-spatial" \
bash slurm/seeded_3x/submit_all.sh
```

Edit `SEEDS=(21)` to `SEEDS=(7 21 42)` in `submit_all.sh` first to run all three seeds; default is a single-seed smoke test. See [`setup/README.md`](setup/README.md) for the full env-var reference.

---

### LIBERO Object

**Expected numbers** (seed=7, from `MODEL_CARD.md`):

| Metric | Value |
|---|---|
| PRO lan | 100.0% (500/500) |
| PRO object | 89.6% (448/500) |
| PRO swap | 0.0% (0/500) |
| Plus overall | 83.6% (2104/2518) |

**Command:**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dwipz/Anchor-Align', allow_patterns='libero-object/*', local_dir='./checkpoints')"

# PRO lan
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero_pro/run_libero_pro_eval.py \
  --pretrained_checkpoint ./checkpoints/libero-object \
  --base_suite_name libero_object --perturbation_type lan \
  --use_proprio True --num_images_in_input 2 --use_pro_version True \
  --num_parallel_envs 50 --seed 7 \
  2>&1 | tee /tmp/repro_object_pro_lan.log

python scripts/verify_reproduction.py \
  --config object_v7kl015_2.5k --benchmark libero_pro --perturbation lan \
  --log /tmp/repro_object_pro_lan.log
```

(Repeat with `--perturbation_type object` and `--perturbation_type swap` for the other two. Plus is analogous — see below.)

---

### LIBERO Goal

**Expected numbers** (seed=7):

| Metric | Value |
|---|---|
| Standard SR | 97.8% (489/500) |
| PRO lan | 96.4% (482/500) |
| PRO object | 76.6% (383/500) |
| PRO swap | 2.2% (11/500) |
| Plus overall | 72.8% (1885/2591) |

**Command:**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dwipz/Anchor-Align', allow_patterns='libero-goal/*', local_dir='./checkpoints')"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_batched.py \
  --pretrained_checkpoint ./checkpoints/libero-goal \
  --task_suite_name libero_goal \
  --num_parallel_envs 50 --seed 7 \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --use_l1_regression True --center_crop True --num_open_loop_steps 8 \
  --use_pro_version True --use_wandb False \
  2>&1 | tee /tmp/repro_goal_std.log

python scripts/verify_reproduction.py \
  --config goal_v7kl01_25k --benchmark spatial_std \
  --log /tmp/repro_goal_std.log
```

---

### LIBERO-10 (Long)

**Expected numbers** (seed=7):

| Metric | Value |
|---|---|
| Standard SR | 90.4% (452/500) |
| PRO lan | 89.8% (449/500) |
| PRO object | 39.6% (198/500) |
| PRO swap | 0.6% (3/500) |
| Plus overall | 69.2% (1744/2519) |

**Command:**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dwipz/Anchor-Align', allow_patterns='libero-long/*', local_dir='./checkpoints')"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_batched.py \
  --pretrained_checkpoint ./checkpoints/libero-long \
  --task_suite_name libero_10 \
  --num_parallel_envs 50 --seed 7 \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --use_l1_regression True --center_crop True --num_open_loop_steps 8 \
  --use_pro_version True --use_wandb False \
  2>&1 | tee /tmp/repro_l10_std.log

python scripts/verify_reproduction.py \
  --config l10_v7kl015_45k --benchmark spatial_std \
  --log /tmp/repro_l10_std.log
```

---

## Inference flags used everywhere

Every eval command above uses the same inference flags. These are the paper flags — do not change them if you want reproducible numbers:

```
--use_proprio         True
--num_images_in_input 2
--use_film            False
--use_l1_regression   True
--use_pro_version     True
--center_crop         True
--num_open_loop_steps 8
```

For LIBERO Spatial + PRO, add `--use_seed_in_env True` if you want the seed to actually vary env initial states across runs. Without that flag, PRO/Standard rollouts are seed-independent (only CUDA non-determinism varies). LIBERO-Plus always uses `cfg.seed` in its env-seed formula, so no extra flag is needed.

---

## What a passing verification looks like

```
$ python scripts/verify_reproduction.py --config anc_v7kl01_10k --benchmark spatial_std --log /tmp/repro_spatial_std_seed7.log

Reproduction check
──────────────────
config:     anc_v7kl01_10k  (Anchor-Align (v7+KL=0.1))
benchmark:  spatial_std
paper:      97.87% ± 0.27 (3-seed mean, seeds 7/21/42)
yours:      97.60% (488/500) — seed=7
delta:      -0.27 pp
tolerance:  warn ±2.0 pp / fail ±3.0 pp

✅ PASS — within statistical variance of the reported number.
```

## What a failing verification looks like

```
⚠️ WARN — delta -2.4 pp is outside the ±2.0 pp warning band.
   This is common if you're on a very different GPU (e.g. RTX 4090 vs GH200).
   If your delta is < ±3.0 pp you're still within statistical bounds.
   Open a GitHub issue with your GPU + CUDA + PyTorch version if you'd like
   us to investigate.

❌ FAIL — delta -4.1 pp is outside the ±3.0 pp failure band.
   Something is likely wrong with your setup:
     - wrong ckpt directory?
     - flag values changed (see "Inference flags used everywhere" in REPRODUCE.md)?
     - LIBERO-PRO / LIBERO-Plus repos not on PYTHONPATH?
     - VLM backbone weights mismatch?
```

---

## When to open an issue

- Any FAIL result. Include your GPU model, CUDA/cuDNN/PyTorch versions, and the eval log.
- Any WARN result if the delta is consistently negative across multiple benchmarks (that suggests a systematic issue, not just kernel noise).
