# Real-world xArm7 mug training

This repository is a portable snapshot of the StarVLA code used to train the
real-world task:

> Pick up the green mug and place it on the plate.

It supports the original QwenGR00T baseline and the completed MSE-anchor +
alignment-v7 recipe. Both use two xArm7 camera streams, 8-D absolute
joint-position actions (7 arm joints and one gripper value), 8-D state, 448 x
448 images, and a 64-step action horizon.

This is offline training code. The robot does not need to be connected during
training. Live xArm control, camera capture, safety interlocks, and rollout
evaluation are separate deployment components and are not included here.

## Repository contents

- `starVLA/`: the complete Python source snapshot used by the mug runs.
- `deployment/`: StarVLA image utilities imported by QwenGR00T.
- `examples/RealWorldXArm7/train_files/data_registry/`: xArm7 LeRobot registry.
- `configs/`: portable baseline and MSE/alignment training configurations.
- `scripts/train_mug.sh`: preflight plus multi-GPU launch entrypoint.
- `scripts/precompute_eef_deltas.py`: optional FK-cache generator for alignment-v7.
- `SOURCE_PROVENANCE.md`: upstream commit and local-change provenance.

Dataset recordings, Qwen weights, EEF caches, outputs, and checkpoints are
external artifacts and are intentionally excluded from Git.

## Tested software stack

The completed run used Python 3.10, CUDA 12.4, PyTorch 2.6.0+cu124,
torchvision 0.21.0+cu124, Transformers 4.57.0, Accelerate 1.5.2, and DeepSpeed
0.16.9. Start from a CUDA 12.4 machine with a CUDA toolkit available to
DeepSpeed JIT compilation.

```bash
conda create -n starvla-xarm python=3.10 -y
conda activate starvla-xarm

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

Set `CUDA_HOME` if the toolkit is not discoverable automatically. It must point
to a CUDA toolkit directory, not merely the Python environment.

## Required artifacts

`DATA_ROOT` must contain a LeRobot v3 dataset named
`mug_xarm7_lerobot` with this shape:

```text
${DATA_ROOT}/mug_xarm7_lerobot/
  data/chunk-*/file-*.parquet
  videos/observation.images.cam_high/chunk-*/file-*.mp4
  videos/observation.images.cam_wrist/chunk-*/file-*.mp4
  meta/episodes/chunk-*/file-*.parquet
  meta/info.json
  meta/modality.json
  meta/stats.json
  meta/stats_gr00t.json
  meta/tasks.parquet
```

The expected dataset has 8-D `action` and `observation.state` arrays. The
modality map splits each into seven joints plus one gripper value. Its task
metadata must contain the instruction shown above. The original run used 51
episodes and 28,774 frames at 30 FPS, but the loader can train on another
number of episodes if the schema and robot semantics match.

`BASE_VLM` may be an absolute local path or the Hugging Face model id
`Qwen/Qwen2.5-VL-3B-Instruct`. A local path is preferable for cluster jobs.

The `mse01_alv7` variant additionally requires `EEF_XYZ_DELTA_CACHE`, a pickle
mapping each episode's first-state fingerprint to a `(frames, 3)` float32 array
of Cartesian EEF deltas. Treat pickle files as trusted local artifacts.

## Validate before training

From this repository root:

```bash
export DATA_ROOT=/path/to/starvla-data
export BASE_VLM=/path/to/Qwen2.5-VL-3B-Instruct

GPUS=0,1,2,3 PREFLIGHT_ONLY=1 \
  bash scripts/train_mug.sh
```

For MSE/alignment validation, also set the cache:

```bash
export EEF_XYZ_DELTA_CACHE=/path/to/eef_deltas_cache_mug.pkl
VARIANT=mse01_alv7 GPUS=0,1,2,3 PREFLIGHT_ONLY=1 \
  bash scripts/train_mug.sh
```

The preflight checks source files, Python packages, CUDA visibility, model
location, dataset metadata, both camera streams, and the optional EEF cache.

## Train

Baseline:

```bash
DATA_ROOT=/path/to/starvla-data \
BASE_VLM=/path/to/Qwen2.5-VL-3B-Instruct \
GPUS=0,1,2,3 \
bash scripts/train_mug.sh
```

MSE-anchor + alignment-v7:

```bash
DATA_ROOT=/path/to/starvla-data \
BASE_VLM=/path/to/Qwen2.5-VL-3B-Instruct \
EEF_XYZ_DELTA_CACHE=/path/to/eef_deltas_cache_mug.pkl \
VARIANT=mse01_alv7 GPUS=0,1,2,3 \
bash scripts/train_mug.sh
```

Defaults reproduce the recipe: four GPUs, batch 4 per GPU, gradient
accumulation 4, global batch 64, 80,000 optimizer updates, cosine learning-rate
schedule, and checkpoints every 10,000 updates. Override launch values with
environment variables:

```bash
STEPS=100 SAVE_INTERVAL=50 GPUS=0 NUM_GPUS=1 \
WANDB_MODE=disabled bash scripts/train_mug.sh
```

The MSE variant logs `loss/mse_raw`, `loss/mse_weighted`, and
`hparams/mse_loss_weight` alongside the total and alignment losses.

Outputs default to `./outputs/<run_id>`. Set `OUTPUT_ROOT` to place them on a
larger filesystem. W&B is disabled by default. For online logging, authenticate
outside this repository and set `WANDB_MODE=online`, `WANDB_ENTITY`, and
optionally `WANDB_PROJECT`.

## Generate the alignment cache

The generator reads joint states directly from the LeRobot parquet files and
uses Pinocchio FK. Supply an xArm7 URDF containing `link_eef` (or select another
frame with `--eef-frame`). The original joint recordings are in degrees.

```bash
pip install pin

python scripts/precompute_eef_deltas.py \
  --dataset-dir /path/to/starvla-data/mug_xarm7_lerobot \
  --urdf /path/to/xarm7.urdf \
  --output /path/to/eef_deltas_cache_mug.pkl \
  --joint-unit degrees
```

Run the MSE/alignment preflight after generation; it verifies that the cache is
non-empty and that every entry has `(frames, 3)` shape. Dataset loading reports
the number of fingerprints matched to trajectories. Do not start a full run if
that report is less than the dataset episode count.
