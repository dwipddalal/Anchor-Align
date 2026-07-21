# Evaluating Anchor-Align Checkpoints

This document shows how to run the released checkpoints on the standard
LIBERO, LIBERO-PRO, and LIBERO-Plus evaluation entry points. Reported results
are listed in the main [`README.md`](README.md) and the paper.

## Quick start

1. Download a checkpoint from [Dwipz/Anchor-Align](https://huggingface.co/Dwipz/Anchor-Align).
2. Run the matching evaluation command below.
3. Compare the resulting success rate with the corresponding README or paper table.

## Reference environment

| Component | Version |
|---|---|
| GPU | NVIDIA GH200 120GB (aarch64) |
| CUDA | 12.4 (driver 550.163.01) |
| cuDNN | Bundled with the PyTorch wheel |
| PyTorch | 2.7; 2.2 is also supported by `pyproject.toml` |
| Python | 3.10.16 |
| LIBERO | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |

LIBERO-PRO and LIBERO-Plus must be installed separately and available on
`PYTHONPATH`; see [`setup/install_libero_variants.sh`](setup/install_libero_variants.sh).

## Checkpoint paths

The evaluation scripts accept either a local checkpoint directory or a
Hugging Face repository/subfolder spec:

| Example | Meaning |
|---|---|
| `./checkpoints/libero-spatial` | Local checkpoint directory |
| `Dwipz/Anchor-Align/libero-spatial` | Hugging Face repository and subfolder |
| `Dwipz/Anchor-Align` | Repository root; the loader lists available subfolders |

Passing the `org/repo/subfolder` form downloads and caches that checkpoint on
first use.

## LIBERO Spatial

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Dwipz/Anchor-Align',
    allow_patterns=['config.json', 'libero-spatial/*'],
    local_dir='./checkpoints',
)
"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_batched.py \
  --pretrained_checkpoint ./checkpoints/libero-spatial \
  --task_suite_name libero_spatial \
  --num_parallel_envs 50 --save_videos False \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --use_l1_regression True --center_crop True --num_open_loop_steps 8 \
  --use_wandb False
```

## LIBERO Object

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dwipz/Anchor-Align', allow_patterns=['config.json', 'libero-object/*'], local_dir='./checkpoints')"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero_pro/run_libero_pro_eval.py \
  --pretrained_checkpoint ./checkpoints/libero-object \
  --base_suite_name libero_object \
  --perturbation_type lan \
  --num_parallel_envs 50 \
  --use_proprio True --num_images_in_input 2 --use_pro_version True \
  --use_film False --use_l1_regression True --center_crop True \
  --num_open_loop_steps 8 --use_wandb False
```

Repeat with `--perturbation_type object` or `swap` for the other LIBERO-PRO
conditions.

## LIBERO Goal

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dwipz/Anchor-Align', allow_patterns=['config.json', 'libero-goal/*'], local_dir='./checkpoints')"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_batched.py \
  --pretrained_checkpoint ./checkpoints/libero-goal \
  --task_suite_name libero_goal \
  --num_parallel_envs 50 \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --use_l1_regression True --center_crop True --num_open_loop_steps 8 \
  --use_pro_version True --use_wandb False
```

## LIBERO-10 (Long)

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dwipz/Anchor-Align', allow_patterns=['config.json', 'libero-long/*'], local_dir='./checkpoints')"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_batched.py \
  --pretrained_checkpoint ./checkpoints/libero-long \
  --task_suite_name libero_10 \
  --num_parallel_envs 50 \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --use_l1_regression True --center_crop True --num_open_loop_steps 8 \
  --use_pro_version True --use_wandb False
```

LIBERO-Plus uses
[`experiments/robot/libero_plus/run_libero_plus_eval_batched.py`](experiments/robot/libero_plus/run_libero_plus_eval_batched.py)
with the same checkpoint and inference settings.

## Inference settings

The reported evaluations use:

```text
--use_proprio         True
--num_images_in_input 2
--use_film            False
--use_l1_regression   True
--use_pro_version     True
--center_crop         True
--num_open_loop_steps 8
```

If results differ substantially from the paper, open a GitHub issue with the
checkpoint name, benchmark, GPU, CUDA and PyTorch versions, command, and log.
