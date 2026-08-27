#!/usr/bin/env python3
"""Validate the local artifacts and environment before launching mug training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "accelerate": "accelerate",
    "albumentations": "albumentations",
    "av": "av",
    "cv2": "opencv-python-headless",
    "deepspeed": "deepspeed",
    "einops": "einops",
    "numpydantic": "numpydantic",
    "omegaconf": "omegaconf",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
    "pytorch3d": "pipablepytorch3d",
    "qwen_vl_utils": "qwen-vl-utils",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "wandb": "wandb",
}

EXPECTED_TASK = "Pick up the green mug and place it on the plate."


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"ERROR: {message}")


def require_file(errors: list[str], path: Path) -> None:
    if not path.is_file():
        add_error(errors, f"missing file: {path}")


def validate_dataset(errors: list[str], dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        add_error(errors, f"mug dataset directory not found: {dataset_dir}")
        return

    required_meta = [
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "modality.json",
        dataset_dir / "meta" / "stats.json",
        dataset_dir / "meta" / "stats_gr00t.json",
        dataset_dir / "meta" / "tasks.parquet",
    ]
    for path in required_meta:
        require_file(errors, path)

    info_path = dataset_dir / "meta" / "info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text())
            features = info.get("features", {})
            action_shape = features.get("action", {}).get("shape")
            state_shape = features.get("observation.state", {}).get("shape")
            if info.get("codebase_version") != "v3.0":
                add_error(errors, "dataset meta/info.json must use LeRobot codebase_version v3.0")
            if action_shape != [8] or state_shape != [8]:
                add_error(errors, f"expected 8-D action and state, got action={action_shape}, state={state_shape}")
            if int(info.get("total_episodes", 0)) < 1:
                add_error(errors, "dataset has no episodes")
        except (OSError, ValueError, TypeError) as exc:
            add_error(errors, f"could not parse {info_path}: {exc}")

    modality_path = dataset_dir / "meta" / "modality.json"
    if modality_path.is_file():
        try:
            modality = json.loads(modality_path.read_text())
            video_keys = set(modality.get("video", {}))
            if video_keys != {"cam_high", "cam_wrist"}:
                add_error(errors, f"expected cam_high and cam_wrist modalities, got {sorted(video_keys)}")
        except (OSError, ValueError, TypeError) as exc:
            add_error(errors, f"could not parse {modality_path}: {exc}")

    parquets = list((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    episode_parquets = list((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    high_videos = list((dataset_dir / "videos" / "observation.images.cam_high").glob("chunk-*/file-*.mp4"))
    wrist_videos = list((dataset_dir / "videos" / "observation.images.cam_wrist").glob("chunk-*/file-*.mp4"))
    if not parquets:
        add_error(errors, f"no trajectory parquet files under {dataset_dir / 'data'}")
    if not episode_parquets:
        add_error(errors, f"no episode metadata parquet files under {dataset_dir / 'meta' / 'episodes'}")
    if not high_videos:
        add_error(errors, "no cam_high MP4 files found")
    if not wrist_videos:
        add_error(errors, "no cam_wrist MP4 files found")

    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if tasks_path.is_file() and importlib.util.find_spec("pandas") is not None:
        try:
            import pandas as pd

            tasks = pd.read_parquet(tasks_path)
            task_names = {str(value) for value in tasks.index.tolist()}
            if EXPECTED_TASK not in task_names:
                add_error(errors, f"expected task {EXPECTED_TASK!r}, found {sorted(task_names)}")
        except Exception as exc:
            add_error(errors, f"could not validate task metadata {tasks_path}: {exc}")

    print(
        f"Dataset: {dataset_dir} "
        f"({len(parquets)} data chunks, {len(episode_parquets)} episode-metadata chunks, "
        f"{len(high_videos)} high-camera videos, {len(wrist_videos)} wrist-camera videos)"
    )


def validate_base_model(errors: list[str], value: str, repo_root: Path) -> None:
    candidate = Path(value).expanduser()
    explicitly_local = candidate.is_absolute() or value.startswith(".")
    if not candidate.is_absolute():
        candidate = repo_root / candidate

    if explicitly_local or candidate.exists():
        if not candidate.is_dir():
            add_error(errors, f"local BASE_VLM directory not found: {candidate}")
        else:
            require_file(errors, candidate / "config.json")
            print(f"Base model: local directory {candidate}")
        return

    if "/" not in value:
        add_error(errors, f"BASE_VLM is neither a local directory nor a Hugging Face repo id: {value}")
    else:
        print(f"Base model: Hugging Face repo {value} (download/cache may require network access)")


def validate_cache(errors: list[str], cache_path: Path) -> None:
    if not cache_path.is_file():
        add_error(errors, f"alignment-v7 cache not found: {cache_path}")
        return
    try:
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        if not isinstance(cache, dict) or not cache:
            add_error(errors, f"alignment-v7 cache must be a non-empty dict: {cache_path}")
            return
        invalid = [key for key, value in cache.items() if getattr(value, "ndim", None) != 2 or value.shape[1] != 3]
        if invalid:
            add_error(errors, f"alignment-v7 cache has malformed entries (first key: {invalid[0]})")
        print(f"EEF cache: {cache_path} ({len(cache)} episodes)")
    except Exception as exc:
        add_error(errors, f"could not read alignment-v7 cache {cache_path}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "mse01_alv7"), default="baseline")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("DATA_ROOT", repo_root / "data")).expanduser().resolve()
    dataset_dir = data_root / "mug_xarm7_lerobot"
    base_vlm = os.environ.get("BASE_VLM", "Qwen/Qwen2.5-VL-3B-Instruct")
    output_root = Path(os.environ.get("OUTPUT_ROOT", repo_root / "outputs")).expanduser().resolve()
    cache_path = Path(
        os.environ.get("EEF_XYZ_DELTA_CACHE", repo_root / "data" / "eef_deltas_cache_mug.pkl")
    ).expanduser().resolve()
    errors: list[str] = []

    print(f"Repository: {repo_root}")
    print(f"Variant: {args.variant}")
    print(f"Output root: {output_root}")

    config_name = "mug_baseline.yaml" if args.variant == "baseline" else "mug_mse01_alv7.yaml"
    for relative in (
        f"configs/{config_name}",
        "starVLA/training/train_starvla.py",
        "starVLA/model/framework/VLM4A/QwenGR00T.py",
        "starVLA/dataloader/gr00t_lerobot/datasets.py",
        "starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml",
        "starVLA/config/deepseeds/ds_config_ga4.yaml",
        "deployment/model_server/tools/image_tools.py",
        "examples/RealWorldXArm7/train_files/data_registry/data_config.py",
    ):
        require_file(errors, repo_root / relative)

    missing_modules = [package for module, package in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]
    if missing_modules:
        add_error(errors, "missing Python packages: " + ", ".join(sorted(missing_modules)))

    validate_dataset(errors, dataset_dir)
    validate_base_model(errors, base_vlm, repo_root)
    if args.variant == "mse01_alv7":
        validate_cache(errors, cache_path)

    if importlib.util.find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            add_error(errors, "PyTorch cannot access CUDA")
        elif torch.cuda.device_count() < args.num_gpus:
            add_error(errors, f"requested {args.num_gpus} GPUs but PyTorch sees {torch.cuda.device_count()}")
        else:
            print(f"CUDA: torch {torch.__version__}, {torch.cuda.device_count()} visible GPU(s)")

    if errors:
        print(f"Preflight failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
