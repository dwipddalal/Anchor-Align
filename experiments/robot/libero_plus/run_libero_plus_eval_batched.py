import torch
torch.backends.cuda.enable_cudnn_sdp(False)  # E12: avoid NaN on GH200 bf16
"""
run_libero_plus_eval_batched.py

Evaluates a trained policy on the LIBERO-Plus benchmark using parallel
environments + batched model inference for ~3-5x speedup over the sequential
run_libero_plus_eval.py.

Key differences from the sequential version:
  1. Runs N heterogeneous tasks in parallel via SubprocVectorEnv
     (each env gets its own BDDL file)
  2. Batches GPU inference across all envs needing requery, grouping by
     tokenized prompt length (required by the model architecture)
  3. Pre-computes task metadata (BDDL paths, descriptions, token lengths)
     to create efficient batches

Architecture constraint: In prismatic/extern/hf/modeling_prismatic.py,
NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1 is a scalar applied uniformly
across the batch. All items in a single forward pass must have identical
input_ids length. We handle this by grouping tasks by token length.

Usage:
    python experiments/robot/libero_plus/run_libero_plus_eval_batched.py \
        --pretrained_checkpoint <path> \
        --base_suite_name libero_spatial \
        --batch_size 32 \
        --save_videos False
"""

import json
import logging
import math
import multiprocessing
import os
import re
import sys
import yaml
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure spawn method for SubprocVectorEnv
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# ── Set LIBERO_CONFIG_PATH BEFORE importing libero ──
LIBERO_PLUS_ROOT = os.environ.get("LIBERO_PLUS_ROOT", "../libero-variants/LIBERO-plus")
LIBERO_PLUS_PKG = os.path.join(LIBERO_PLUS_ROOT, "libero", "libero")
# Per-process config dir to avoid race conditions when multiple jobs share a node
LIBERO_PLUS_CONFIG_DIR = os.path.join(
    LIBERO_PLUS_ROOT, f".libero_config_{os.getpid()}"
)

os.makedirs(LIBERO_PLUS_CONFIG_DIR, exist_ok=True)
_config_path = os.path.join(LIBERO_PLUS_CONFIG_DIR, "config.yaml")
_config = {
    "benchmark_root": LIBERO_PLUS_PKG,
    "bddl_files": os.path.join(LIBERO_PLUS_PKG, "bddl_files"),
    "init_states": os.path.join(LIBERO_PLUS_PKG, "init_files"),
    "datasets": os.path.join(LIBERO_PLUS_ROOT, "libero", "datasets"),
    "assets": os.path.join(LIBERO_PLUS_PKG, "assets"),
}
with open(_config_path, "w") as f:
    yaml.dump(_config, f)
os.environ["LIBERO_CONFIG_PATH"] = LIBERO_PLUS_CONFIG_DIR

import atexit
def _cleanup_config():
    import shutil
    try:
        shutil.rmtree(LIBERO_PLUS_CONFIG_DIR, ignore_errors=True)
    except Exception:
        pass
atexit.register(_cleanup_config)

# Make LIBERO-Plus's `libero` package override standard LIBERO
sys.path.insert(0, LIBERO_PLUS_ROOT)

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv

import wandb

# Append so interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    get_vla_action_batch,
    get_vla_action_batch_multi_prompt,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# ── Reuse constants and helpers from sequential LIBERO-Plus eval ──

PERTURBATION_CATEGORIES = [
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
]
DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    # ── Model ──────────────────────────────────────────────────────────────────
    model_family:           str              = "openvla"
    pretrained_checkpoint:  Union[str, Path] = ""
    use_l1_regression:      bool             = True
    use_minivlm:            bool             = True
    num_diffusion_steps:    int              = 50
    use_film:               bool             = False
    num_images_in_input:    int              = 2
    use_proprio:            bool             = True
    center_crop:            bool             = True
    num_open_loop_steps:    int              = 8
    unnorm_key:             Union[str, Path] = ""
    load_in_8bit:           bool             = False
    load_in_4bit:           bool             = False

    # ── LIBERO-Plus environment ────────────────────────────────────────────────
    base_suite_name:        str  = "libero_spatial"
    num_steps_wait:         int  = 10
    num_trials_per_task:    int  = 1
    env_img_res:            int  = 256

    # ── Batched evaluation ─────────────────────────────────────────────────────
    batch_size:             int  = 32            # Tasks per batch (tune for GPU memory)
    save_videos:            bool = False

    # ── Task range (for splitting across GPUs) ─────────────────────────────────
    start_task_id:          int  = 0
    end_task_id:            int  = -1

    # ── Filtering by perturbation category ─────────────────────────────────────
    perturbation_category:  str  = "all"

    # ── Logging ────────────────────────────────────────────────────────────────
    run_id_note:            Optional[str] = None
    local_log_dir:          str           = "./experiments/logs/libero_plus"
    use_wandb:              bool          = False
    wandb_entity:           str           = "your-wandb-entity"
    wandb_project:          str           = "your-wandb-project"
    seed:                   int           = 7
    save_version:           str           = "vla-adapter"
    use_pro_version:        bool          = True
    phase:                  str           = "Inference"
    # fmt: on


# ── Helper functions (reused from sequential version) ──────────────────────


def get_clean_task_description(task, task_classification: Dict) -> str:
    """Get the clean language instruction for a LIBERO-Plus task."""
    task_name = task.name
    cls_info = task_classification.get(task_name, {})
    category = cls_info.get("category", "")

    if category == "Language Instructions" or "_language_" in task_name:
        return task.language

    base_name = task_name
    base_name = re.sub(r'_view_[\d_]+_initstate_\d+$', '', base_name)
    base_name = re.sub(r'_view_[\d_]+$', '', base_name)
    base_name = re.sub(r'_initstate_\d+$', '', base_name)
    base_name = re.sub(r'_table_\d+$', '', base_name)
    base_name = re.sub(r'_tb_\d+$', '', base_name)
    base_name = re.sub(r'_light_.*$', '', base_name)
    base_name = re.sub(r'_add_\d+$', '', base_name)
    base_name = re.sub(r'_level\d+$', '', base_name)

    clean_language = base_name.replace("_", " ")
    return clean_language


def load_task_classification(suite_name: str) -> Dict:
    """Load task_classification.json and build lookup maps."""
    cls_path = os.path.join(
        LIBERO_PLUS_PKG, "benchmark", "task_classification.json"
    )
    with open(cls_path, "r") as f:
        all_cls = json.load(f)

    if suite_name not in all_cls:
        logger.warning(f"Suite {suite_name} not in task_classification.json. "
                       f"Available: {list(all_cls.keys())}")
        return {}

    lookup = {}
    for entry in all_cls[suite_name]:
        lookup[entry["name"]] = {
            "category": entry.get("category", "Unknown"),
            "difficulty_level": entry.get("difficulty_level"),
        }
    return lookup


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "center_crop must be True when trained with image_aug!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit)
    assert cfg.base_suite_name in [s.value for s in TaskSuite], \
        f"Invalid base_suite_name: {cfg.base_suite_name}"
    assert cfg.batch_size >= 1, "batch_size must be >= 1"
    cfg.save_version = f"libero_plus/{cfg.base_suite_name}"


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    key = cfg.base_suite_name
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    assert key in model.norm_stats, f"Unnorm key '{key}' not found in norm_stats!"
    cfg.unnorm_key = key


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-libero_plus_batched-{cfg.base_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.perturbation_category != "all":
        cat_short = cfg.perturbation_category.replace(" ", "_").lower()
        run_id += f"--{cat_short}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to: {local_log_filepath}")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id


def log_message(msg: str, log_file=None):
    logger.info(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def prepare_obs_from_vec(vec_obs, env_idx, resize_size):
    """Extract policy-ready observation from SubprocVectorEnv output."""
    obs_i = vec_obs[env_idx]
    img = obs_i["agentview_image"][::-1, ::-1]
    wrist_img = obs_i["robot0_eye_in_hand_image"][::-1, ::-1]
    observation = {
        "full_image": resize_image_for_policy(img, resize_size),
        "wrist_image": resize_image_for_policy(wrist_img, resize_size),
        "state": np.concatenate((
            obs_i["robot0_eef_pos"],
            quat2axisangle(obs_i["robot0_eef_quat"]),
            obs_i["robot0_gripper_qpos"],
        )),
    }
    return observation, img.copy()


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def _make_env_fn(env_args, env_seed=0):
    """Returns an env constructor (avoids lambda closure bug with SubprocVectorEnv)."""
    def _init():
        # Seed subprocess RNG BEFORE env construction so robosuite init is deterministic
        import random as _random
        np.random.seed(env_seed)
        _random.seed(env_seed)
        env = OffScreenRenderEnv(**env_args)
        env.seed(env_seed)
        return env
    return _init


def print_results_breakdown(results: List[dict], task_classification: Dict, log_file=None):
    """Print detailed breakdown of results by category and difficulty level."""
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r["success"] is None)
    evaluated = total - skipped
    sr = successes / evaluated * 100 if evaluated > 0 else 0

    log_message(f"\n{'='*70}", log_file)
    log_message(f"LIBERO-PLUS RESULTS SUMMARY (BATCHED)", log_file)
    log_message(f"{'='*70}", log_file)
    log_message(f"Total tasks: {total} | Evaluated: {evaluated} | Skipped: {skipped}", log_file)
    log_message(f"Successes: {successes}/{evaluated} = {sr:.1f}%", log_file)

    cat_results = {}
    for r in results:
        if r["success"] is None:
            continue
        cls_info = task_classification.get(r["task_name"], {})
        category = cls_info.get("category", "Unknown")
        if category not in cat_results:
            cat_results[category] = {"total": 0, "success": 0}
        cat_results[category]["total"] += 1
        if r["success"]:
            cat_results[category]["success"] += 1

    log_message(f"\n{'─'*70}", log_file)
    log_message(f"  {'Category':<25} {'Success':>8} {'Total':>8} {'Rate':>8}", log_file)
    log_message(f"{'─'*70}", log_file)
    for cat in sorted(cat_results.keys()):
        s = cat_results[cat]["success"]
        t = cat_results[cat]["total"]
        rate = s / t * 100 if t > 0 else 0
        log_message(f"  {cat:<25} {s:>8} {t:>8} {rate:>7.1f}%", log_file)

    level_results = {}
    for r in results:
        if r["success"] is None:
            continue
        cls_info = task_classification.get(r["task_name"], {})
        level = cls_info.get("difficulty_level")
        level_key = f"L{level}" if level is not None else "Unknown"
        if level_key not in level_results:
            level_results[level_key] = {"total": 0, "success": 0}
        level_results[level_key]["total"] += 1
        if r["success"]:
            level_results[level_key]["success"] += 1

    log_message(f"\n{'─'*70}", log_file)
    log_message(f"  {'Difficulty':<25} {'Success':>8} {'Total':>8} {'Rate':>8}", log_file)
    log_message(f"{'─'*70}", log_file)
    for lvl in sorted(level_results.keys()):
        s = level_results[lvl]["success"]
        t = level_results[lvl]["total"]
        rate = s / t * 100 if t > 0 else 0
        log_message(f"  {lvl:<25} {s:>8} {t:>8} {rate:>7.1f}%", log_file)

    # Cross-tabulation
    log_message(f"\n{'─'*70}", log_file)
    log_message(f"CROSS-TABULATION: Category x Difficulty (success rate %)", log_file)
    log_message(f"{'─'*70}", log_file)
    cross = {}
    for r in results:
        if r["success"] is None:
            continue
        cls_info = task_classification.get(r["task_name"], {})
        cat = cls_info.get("category", "Unknown")
        lvl = cls_info.get("difficulty_level")
        lvl_key = f"L{lvl}" if lvl is not None else "?"
        key = (cat, lvl_key)
        if key not in cross:
            cross[key] = {"total": 0, "success": 0}
        cross[key]["total"] += 1
        if r["success"]:
            cross[key]["success"] += 1

    all_levels = sorted(set(k[1] for k in cross.keys()))
    header = f"  {'Category':<25}" + "".join(f"{l:>8}" for l in all_levels) + f"{'Total':>8}"
    log_message(header, log_file)
    for cat in sorted(set(k[0] for k in cross.keys())):
        row = f"  {cat:<25}"
        cat_total_s, cat_total_t = 0, 0
        for lvl in all_levels:
            d = cross.get((cat, lvl), {"total": 0, "success": 0})
            rate = d["success"] / d["total"] * 100 if d["total"] > 0 else 0
            row += f"{rate:>7.1f}%"
            cat_total_s += d["success"]
            cat_total_t += d["total"]
        cat_rate = cat_total_s / cat_total_t * 100 if cat_total_t > 0 else 0
        row += f"{cat_rate:>7.1f}%"
        log_message(row, log_file)

    log_message(f"{'='*70}\n", log_file)

    return {
        "overall_sr": sr,
        "per_category": {
            cat: cat_results[cat]["success"] / cat_results[cat]["total"] * 100
            if cat_results[cat]["total"] > 0 else 0
            for cat in cat_results
        },
        "per_level": {
            lvl: level_results[lvl]["success"] / level_results[lvl]["total"] * 100
            if level_results[lvl]["total"] > 0 else 0
            for lvl in level_results
        },
    }


# ── Batched evaluation core ───────────────────────────────────────────────


def precompute_task_metadata(
    task_suite,
    task_ids: List[int],
    task_classification: Dict,
    processor,
    cfg: GenerateConfig,
    log_file=None,
) -> List[Tuple]:
    """Pre-compute metadata for all tasks: description, BDDL path, init state, token length.

    Returns list of tuples:
        (task_id, task, init_state, description, bddl_file_path, token_length)
    """
    from PIL import Image as _PILImage
    dummy_img = _PILImage.new("RGB", (224, 224))

    metadata = []
    skipped = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)

        # Load init states
        try:
            initial_states = task_suite.get_task_init_states(task_id)
        except Exception as e:
            log_message(f"  Task {task_id} ({task.name}): SKIP - cannot load init states: {e}", log_file)
            skipped += 1
            metadata.append((task_id, task, None, None, None, None))
            continue

        init_state = initial_states[0]

        # Get BDDL file path — robust: try config, fallback to package dir
        try:
            bddl_base = get_libero_path("bddl_files")
        except (TypeError, KeyError, FileNotFoundError, OSError):
            # Fallback: derive from LIBERO-plus package location
            import libero as _libero_pkg
            bddl_base = os.path.join(os.path.dirname(os.path.abspath(_libero_pkg.__file__)), "bddl_files")
        bddl_file_path = os.path.join(
            bddl_base, task.problem_folder, task.bddl_file
        )

        # Get clean task description
        description = get_clean_task_description(task, task_classification)

        # Build prompt and tokenize to get token length
        if not cfg.use_minivlm:
            prompt = f"In: What action should the robot take to {description.lower()}?\nOut:"
        else:
            prompt = (
                f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                f"<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {description.lower()}?"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
        tok = processor(prompt, dummy_img)
        token_length = tok["input_ids"].shape[-1]

        metadata.append((task_id, task, init_state, description, bddl_file_path, token_length))

    if skipped > 0:
        log_message(f"Pre-compute: {skipped} tasks skipped (no init states), "
                    f"{len(metadata) - skipped} tasks ready", log_file)

    return metadata


def create_batches(
    task_metadata: List[Tuple],
    batch_size: int,
) -> List[List[Tuple]]:
    """Group tasks by token length, then chunk into batches of at most batch_size.

    Returns flat list of batches. Each batch contains tasks with the same token length.
    """
    # Filter out skipped tasks (init_state is None)
    valid = [m for m in task_metadata if m[2] is not None]

    # Group by token length
    groups = defaultdict(list)
    for m in valid:
        token_length = m[5]
        groups[token_length].append(m)

    # Within each group, chunk into batches
    batches = []
    for tlen in sorted(groups.keys()):
        group = groups[tlen]
        for i in range(0, len(group), batch_size):
            batches.append(group[i:i + batch_size])

    return batches


def run_batch_of_tasks(
    cfg: GenerateConfig,
    batch_tasks: List[Tuple],
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    log_file=None,
    use_subprocess: bool = True,
) -> List[dict]:
    """Run a batch of heterogeneous LIBERO-Plus tasks in parallel.

    Each task in batch_tasks is:
        (task_id, task, init_state, description, bddl_file_path, token_length)

    All tasks in a batch have the same token_length (enforced by create_batches).

    Returns list of result dicts: [{task_id, task_name, success}, ...]
    """
    N = len(batch_tasks)
    max_steps = TASK_MAX_STEPS[TaskSuite(cfg.base_suite_name)]

    # Create heterogeneous env functions (each task has its own BDDL file)
    env_fns = []
    for env_idx, (task_id, task, init_state, desc, bddl_path, tlen) in enumerate(batch_tasks):
        env_args = {
            "bddl_file_name": bddl_path,
            "camera_heights": cfg.env_img_res,
            "camera_widths": cfg.env_img_res,
        }
        env_fns.append(_make_env_fn(env_args, env_seed=cfg.seed + task_id))

    # Create vectorized environment (with retry for frame buffer contention)
    import time as _time
    vec_env = None
    for attempt in range(5):
        try:
            if N > 1 and use_subprocess:
                vec_env = SubprocVectorEnv(env_fns)
            else:
                vec_env = DummyVectorEnv(env_fns)
            break
        except Exception as e:
            if attempt < 4:
                log_message(
                    f"  Env creation attempt {attempt+1}/5 failed ({type(e).__name__}), "
                    f"retrying in 5s...", log_file,
                )
                _time.sleep(5)
            else:
                raise
    assert vec_env is not None

    # Reset and set initial states
    vec_env.reset()
    init_states = [m[2] for m in batch_tasks]
    obs = vec_env.set_init_state(init_states)

    # Wait for objects to stabilize
    dummy = np.tile(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64), (N, 1))
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = vec_env.step(dummy)

    # Episode loop
    action_queues = [deque(maxlen=cfg.num_open_loop_steps) for _ in range(N)]
    dones = [False] * N
    successes = [False] * N
    task_labels = [m[3] for m in batch_tasks]  # per-env task descriptions

    for t in range(max_steps):
        # Identify envs needing requery
        need_requery = [i for i in range(N)
                        if len(action_queues[i]) == 0 and not dones[i]]

        if need_requery:
            obs_list = []
            requery_labels = []
            for i in need_requery:
                ob, _ = prepare_obs_from_vec(obs, i, resize_size)
                obs_list.append(ob)
                requery_labels.append(task_labels[i])

            # Choose batching strategy based on label diversity
            if len(set(requery_labels)) == 1:
                # All same label → use efficient single-prompt batch
                actions_list = get_vla_action_batch(
                    cfg, model, processor, obs_list, requery_labels[0],
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )
            else:
                # Mixed labels (all same token length within this batch) → multi-prompt
                actions_list = get_vla_action_batch_multi_prompt(
                    cfg, model, processor, obs_list, requery_labels,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )

            for idx, env_i in enumerate(need_requery):
                action_queues[env_i].extend(actions_list[idx])

        # Step all envs
        actions = np.zeros((N, 7))
        for i in range(N):
            if not dones[i] and action_queues[i]:
                actions[i] = process_action(action_queues[i].popleft(), cfg.model_family)
            else:
                actions[i] = np.array([0, 0, 0, 0, 0, 0, -1])

        obs, _, done_flags, _ = vec_env.step(actions)

        for i in range(N):
            if done_flags[i] and not dones[i]:
                dones[i] = True
                successes[i] = True

        if all(dones):
            break

    vec_env.close()

    # Build result dicts
    results = []
    for i, (task_id, task, init_state, desc, bddl_path, tlen) in enumerate(batch_tasks):
        results.append({
            "task_id": task_id,
            "task_name": task.name,
            "success": successes[i],
        })

    return results


def run_batch_with_fallback(
    cfg: GenerateConfig,
    batch_tasks: List[Tuple],
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    log_file=None,
) -> List[dict]:
    """Run a batch with adaptive parallelism fallback.

    Tries the full batch first. On SubprocVectorEnv crash, halves the sub-batch
    size and retries. Keeps halving until success or falls back to sequential
    (DummyVectorEnv, batch=1).
    """
    parallel = len(batch_tasks)
    remaining = list(batch_tasks)
    all_results = []

    while remaining:
        chunk = remaining[:parallel]
        try:
            use_sub = parallel > 1
            chunk_results = run_batch_of_tasks(
                cfg, chunk, model, resize_size, processor,
                action_head, proprio_projector, log_file, use_subprocess=use_sub,
            )
            all_results.extend(chunk_results)
            remaining = remaining[parallel:]
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as e:
            log_message(
                f"  Crashed with {parallel} parallel envs ({type(e).__name__})",
                log_file,
            )
            if parallel > 1:
                parallel = max(1, parallel // 2)
                log_message(f"  Retrying with {parallel} parallel envs...", log_file)
            else:
                # Even sequential failed — count this task as failure
                task_id, task, _, _, _, _ = remaining[0]
                log_message(
                    f"  Task {task_id} ({task.name}) failed even sequentially, "
                    f"counting as failure.",
                    log_file,
                )
                all_results.append({
                    "task_id": task_id,
                    "task_name": task.name,
                    "success": False,
                })
                remaining = remaining[1:]

    return all_results


# ── Main entry point ──────────────────────────────────────────────────────


@draccus.wrap()
def eval_libero_plus_batched(cfg: GenerateConfig) -> float:
    """Main function: batched evaluation on LIBERO-Plus benchmark."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Load task classification
    task_classification = load_task_classification(cfg.base_suite_name)
    log_message(f"Loaded task classification: {len(task_classification)} entries for {cfg.base_suite_name}")

    # Initialize model
    model, action_head, proprio_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize benchmark suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.base_suite_name]()
    num_tasks = task_suite.n_tasks

    # Determine task range
    start_task = cfg.start_task_id
    end_task = num_tasks - 1 if cfg.end_task_id < 0 else min(cfg.end_task_id, num_tasks - 1)

    # Filter by perturbation category if specified
    if cfg.perturbation_category != "all":
        # Normalize: "background_textures" → "background textures" for comparison
        query_cat = cfg.perturbation_category.replace("_", " ").strip().lower()
        filtered_task_ids = []
        for tid in range(start_task, end_task + 1):
            task = task_suite.get_task(tid)
            cls_info = task_classification.get(task.name, {})
            entry_cat = cls_info.get("category", "").strip().lower()
            if entry_cat == query_cat:
                filtered_task_ids.append(tid)
        task_ids = filtered_task_ids
        log_message(
            f"Filtered to {len(task_ids)} tasks with category '{cfg.perturbation_category}'",
            log_file,
        )
    else:
        task_ids = list(range(start_task, end_task + 1))

    log_message(f"Suite: {cfg.base_suite_name} (LIBERO-Plus, batched)", log_file)
    log_message(f"Total tasks in suite: {num_tasks}", log_file)
    log_message(f"Evaluating tasks {start_task}–{end_task} ({len(task_ids)} tasks)", log_file)
    log_message(f"Batch size: {cfg.batch_size}", log_file)
    log_message(f"Checkpoint: {cfg.pretrained_checkpoint}", log_file)

    # Pre-compute task metadata (descriptions, BDDL paths, token lengths)
    log_message("Pre-computing task metadata...", log_file)
    task_metadata = precompute_task_metadata(
        task_suite, task_ids, task_classification, processor, cfg, log_file
    )

    # Create batches grouped by token length
    batches = create_batches(task_metadata, cfg.batch_size)
    log_message(f"Created {len(batches)} batches (batch_size={cfg.batch_size})", log_file)

    # Log token length distribution
    tlen_counts = defaultdict(int)
    for m in task_metadata:
        if m[2] is not None:
            tlen_counts[m[5]] += 1
    for tlen in sorted(tlen_counts.keys()):
        log_message(f"  Token length {tlen}: {tlen_counts[tlen]} tasks", log_file)

    # Run evaluation
    all_results = []
    total_successes = 0
    total_evaluated = 0

    # Add skipped tasks to results first
    for m in task_metadata:
        if m[2] is None:
            all_results.append({
                "task_id": m[0],
                "task_name": m[1].name,
                "success": None,
            })

    for batch_idx, batch in enumerate(tqdm.tqdm(batches, desc="LIBERO-Plus batched eval")):
        # Re-seed before each batch for reproducibility (deterministic per task_id)
        batch_seed = cfg.seed + batch[0][0]  # cfg.seed + first task_id in batch
        set_seed_everywhere(batch_seed)

        batch_results = run_batch_with_fallback(
            cfg, batch, model, resize_size, processor,
            action_head, proprio_projector, log_file,
        )

        all_results.extend(batch_results)

        for r in batch_results:
            if r["success"] is not None:
                total_evaluated += 1
                if r["success"]:
                    total_successes += 1

        # Periodic progress
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(batches):
            sr = total_successes / total_evaluated * 100 if total_evaluated > 0 else 0
            log_message(
                f"  Batch {batch_idx+1}/{len(batches)} | "
                f"{total_successes}/{total_evaluated} = {sr:.1f}%",
                log_file,
            )

    # Print detailed breakdown
    summary = print_results_breakdown(all_results, task_classification, log_file)

    # Save results JSON
    results_json_path = local_log_filepath.replace(".txt", "_results.json")
    with open(results_json_path, "w") as f:
        json.dump({
            "config": {
                "checkpoint": str(cfg.pretrained_checkpoint),
                "base_suite_name": cfg.base_suite_name,
                "perturbation_category": cfg.perturbation_category,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
            },
            "summary": summary,
            "per_task_results": all_results,
        }, f, indent=2)
    log_message(f"Results JSON saved to: {results_json_path}", log_file)

    # Log to wandb
    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": summary["overall_sr"],
            "num_tasks/evaluated": total_evaluated,
        })
        for cat, sr in summary["per_category"].items():
            wandb.log({f"success_rate/{cat}": sr})
        for lvl, sr in summary["per_level"].items():
            wandb.log({f"success_rate/{lvl}": sr})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return summary["overall_sr"]


if __name__ == "__main__":
    eval_libero_plus_batched()
