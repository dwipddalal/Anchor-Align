"""
run_libero_plus_eval.py

Evaluates a trained policy on the LIBERO-Plus benchmark (10,030 tasks across
7 perturbation dimensions and 5 difficulty levels).

Key differences from LIBERO / LIBERO-PRO evaluation:
  1. sys.path insert for the LIBERO-Plus package
  2. LIBERO_CONFIG_PATH set to a LIBERO-Plus–specific config directory
  3. num_trials_per_task=1 (each perturbation variant IS its own task)
  4. Post-hoc analysis using task_classification.json (per-category, per-level)
  5. base_suite_name selects which suite (libero_spatial, libero_object, etc.)
  6. unnorm_key uses base_suite_name (checkpoint trained on base suite)

Reference: https://github.com/sylvestf/LIBERO-plus
Paper: "LIBERO-Plus: In-depth Robustness Analysis of VLA Models" (arXiv:2510.13626)
"""

import json
import logging
import math
import multiprocessing
import os
import re
import sys
import yaml
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

# Ensure spawn method for SubprocVectorEnv
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# ── Change 1: Set LIBERO_CONFIG_PATH BEFORE importing libero ──
LIBERO_PLUS_ROOT = os.environ.get("LIBERO_PLUS_ROOT", "../libero-variants/LIBERO-plus")
LIBERO_PLUS_PKG = os.path.join(LIBERO_PLUS_ROOT, "libero", "libero")
LIBERO_PLUS_CONFIG_DIR = os.path.join(LIBERO_PLUS_ROOT, ".libero_config")

# Create config directory and config.yaml for LIBERO-Plus
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

# ── Change 2: Make LIBERO-Plus's `libero` package override standard LIBERO ──
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


def get_clean_task_description(task, task_classification: Dict) -> str:
    """Get the clean language instruction for a LIBERO-Plus task.

    For non-language perturbations (background texture, camera, light, etc.),
    the task.language attribute is derived from the filename and includes
    perturbation suffixes (e.g., "table 1", "view 0 0 100 0 0 initstate 218").
    These suffixes are NOT real language and would confuse the model.

    For language perturbation tasks, task.language IS the correct perturbed
    instruction (extracted from the BDDL file by the benchmark code).

    This function strips perturbation suffixes to return the clean base instruction.
    """
    task_name = task.name
    cls_info = task_classification.get(task_name, {})
    category = cls_info.get("category", "")

    # For language perturbation tasks, use the benchmark-provided language
    # (already extracted from BDDL by grab_language_from_filename)
    if category == "Language Instructions" or "_language_" in task_name:
        return task.language

    # For other tasks, extract the base task name by stripping perturbation suffixes
    # and convert to clean language
    base_name = task_name

    # Strip suffixes in order of specificity
    # _view_X_Y_Z_W_V_initstate_M (camera viewpoints / robot init states)
    base_name = re.sub(r'_view_[\d_]+_initstate_\d+$', '', base_name)
    # _view_X_Y_Z_W_V (camera viewpoints without initstate)
    base_name = re.sub(r'_view_[\d_]+$', '', base_name)
    # _initstate_N (robot initial states)
    base_name = re.sub(r'_initstate_\d+$', '', base_name)
    # _table_N (background textures)
    base_name = re.sub(r'_table_\d+$', '', base_name)
    # _tb_N (background textures variant)
    base_name = re.sub(r'_tb_\d+$', '', base_name)
    # _light_* (light conditions)
    base_name = re.sub(r'_light_.*$', '', base_name)
    # _add_N (objects layout - confounding objects)
    base_name = re.sub(r'_add_\d+$', '', base_name)
    # _levelN (objects layout - difficulty levels)
    base_name = re.sub(r'_level\d+$', '', base_name)

    # Convert base task name to natural language
    clean_language = base_name.replace("_", " ")
    return clean_language


def get_bddl_language(task) -> Optional[str]:
    """Extract the language instruction from a BDDL file's (:language ...) field."""
    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    try:
        with open(bddl_file_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("(:language"):
                    lang = stripped[len("(:language"):].rstrip(")").strip()
                    return lang
    except Exception:
        pass
    return None


# ── Perturbation categories in LIBERO-Plus ──
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


# Max steps per base suite (same as standard LIBERO)
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
    base_suite_name:        str  = "libero_spatial"     # libero_spatial, libero_object, libero_goal, libero_10
    num_steps_wait:         int  = 10                   # Steps to wait for objects to stabilize
    num_trials_per_task:    int  = 1                    # ── Change 3: 1 trial per task (each task IS a perturbation)
    env_img_res:            int  = 256

    # ── Parallel evaluation ────────────────────────────────────────────────────
    num_parallel_envs:      int  = 10                   # Parallel envs for multi-trial tasks
    save_videos:            bool = False

    # ── Task range (for splitting across GPUs) ─────────────────────────────────
    start_task_id:          int  = 0
    end_task_id:            int  = -1                   # -1 = all tasks

    # ── Filtering by perturbation category ─────────────────────────────────────
    perturbation_category:  str  = "all"                # "all" or specific category name

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


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "center_crop must be True when trained with image_aug!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit)
    assert cfg.base_suite_name in [s.value for s in TaskSuite], \
        f"Invalid base_suite_name: {cfg.base_suite_name}. Must be one of {[s.value for s in TaskSuite]}"
    assert cfg.num_parallel_envs >= 1
    cfg.save_version = f"libero_plus/{cfg.base_suite_name}"


def load_task_classification(suite_name: str) -> Dict:
    """Load task_classification.json and build lookup maps for per-category/level analysis."""
    cls_path = os.path.join(
        LIBERO_PLUS_PKG, "benchmark", "task_classification.json"
    )
    with open(cls_path, "r") as f:
        all_cls = json.load(f)

    if suite_name not in all_cls:
        logger.warning(f"Suite {suite_name} not in task_classification.json. "
                       f"Available: {list(all_cls.keys())}")
        return {}

    # Build lookup: task_name -> {category, difficulty_level}
    lookup = {}
    for entry in all_cls[suite_name]:
        lookup[entry["name"]] = {
            "category": entry.get("category", "Unknown"),
            "difficulty_level": entry.get("difficulty_level"),
        }
    return lookup


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
    """Use base_suite_name for normalization stats (checkpoint was trained on base suite)."""
    key = cfg.base_suite_name
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    assert key in model.norm_stats, f"Unnorm key '{key}' not found in norm_stats!"
    cfg.unnorm_key = key


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-libero_plus-{cfg.base_suite_name}-{cfg.model_family}-{DATE_TIME}"
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


def prepare_obs_from_env(obs, resize_size):
    """Extract policy-ready observation from a single environment."""
    img = obs["agentview_image"][::-1, ::-1]
    wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    observation = {
        "full_image": resize_image_for_policy(img, resize_size),
        "wrist_image": resize_image_for_policy(wrist_img, resize_size),
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
    }
    return observation, img.copy()


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_single_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    task_classification: Dict,
    processor=None,
    action_head=None,
    proprio_projector=None,
    log_file=None,
):
    """Run evaluation for a single LIBERO-Plus task (1 trial).

    Returns:
        success (bool): Whether the task was completed successfully.
        task_name (str): Name of the task.
    """
    task = task_suite.get_task(task_id)
    task_name = task.name

    # Get initial states (LIBERO-Plus handles path resolution internally)
    try:
        initial_states = task_suite.get_task_init_states(task_id)
    except Exception as e:
        log_message(f"  Task {task_id} ({task_name}): SKIP - cannot load init states: {e}", log_file)
        return None, task_name

    # Get task description and BDDL file path
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    # Get clean language instruction (strips perturbation suffixes)
    task_description = get_clean_task_description(task, task_classification)

    max_steps = TASK_MAX_STEPS[TaskSuite(cfg.base_suite_name)]

    # Create environment
    try:
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": cfg.env_img_res,
            "camera_widths": cfg.env_img_res,
        }
        # Seed RNG before env construction for deterministic robosuite init
        np.random.seed(cfg.seed + task_id)
        env = OffScreenRenderEnv(**env_args)
        env.seed(cfg.seed + task_id)
    except Exception as e:
        log_message(f"  Task {task_id} ({task_name}): SKIP - env creation failed: {e}", log_file)
        return None, task_name

    try:
        # Reset and set initial state
        env.reset()
        init_state = initial_states[0]  # Use first init state
        obs = env.set_init_state(init_state)

        # Wait for objects to stabilize
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(dummy_action)

        # Run episode
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        replay_images = [] if cfg.save_videos else None
        success = False

        for t in range(max_steps):
            # Requery model if action queue is empty
            if len(action_queue) == 0:
                ob, img = prepare_obs_from_env(obs, resize_size)
                if cfg.save_videos:
                    replay_images.append(img)

                # Single observation -> batch of 1
                actions_list = get_vla_action_batch(
                    cfg, model, processor, [ob], task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )
                action_queue.extend(actions_list[0])
            else:
                if cfg.save_videos:
                    _, img = prepare_obs_from_env(obs, resize_size)
                    replay_images.append(img)

            # Execute action
            action = process_action(action_queue.popleft(), cfg.model_family)
            obs, reward, done, info = env.step(action)

            if done:
                success = True
                break

        # Save video if requested
        if cfg.save_videos and replay_images:
            save_rollout_video(
                replay_images, task_id, success=success,
                task_description=task_description, log_file=log_file,
                save_version=cfg.save_version,
            )

    except Exception as e:
        log_message(f"  Task {task_id} ({task_name}): ERROR during rollout: {e}", log_file)
        success = False
    finally:
        env.close()

    return success, task_name


def print_results_breakdown(
    results: List[dict],
    task_classification: Dict,
    log_file=None,
):
    """Print detailed breakdown of results by category and difficulty level."""

    # Overall
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r["success"] is None)
    evaluated = total - skipped
    sr = successes / evaluated * 100 if evaluated > 0 else 0

    log_message(f"\n{'='*70}", log_file)
    log_message(f"LIBERO-PLUS RESULTS SUMMARY", log_file)
    log_message(f"{'='*70}", log_file)
    log_message(f"Total tasks: {total} | Evaluated: {evaluated} | Skipped: {skipped}", log_file)
    log_message(f"Successes: {successes}/{evaluated} = {sr:.1f}%", log_file)

    # Per-category breakdown
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

    # Per-difficulty breakdown
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

    # Cross-tabulation: category x difficulty
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


@draccus.wrap()
def eval_libero_plus(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO-Plus benchmark."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Load task classification for post-hoc analysis
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
        filtered_task_ids = []
        for tid in range(start_task, end_task + 1):
            task = task_suite.get_task(tid)
            cls_info = task_classification.get(task.name, {})
            if cls_info.get("category") == cfg.perturbation_category:
                filtered_task_ids.append(tid)
        task_ids = filtered_task_ids
        log_message(
            f"Filtered to {len(task_ids)} tasks with category '{cfg.perturbation_category}'",
            log_file,
        )
    else:
        task_ids = list(range(start_task, end_task + 1))

    log_message(f"Suite: {cfg.base_suite_name} (LIBERO-Plus)", log_file)
    log_message(f"Total tasks in suite: {num_tasks}", log_file)
    log_message(f"Evaluating tasks {start_task}–{end_task} ({len(task_ids)} tasks)", log_file)
    log_message(f"Num trials per task: {cfg.num_trials_per_task}", log_file)
    log_message(f"Checkpoint: {cfg.pretrained_checkpoint}", log_file)

    # Run evaluation
    results = []
    total_successes = 0
    total_evaluated = 0

    for i, task_id in enumerate(tqdm.tqdm(task_ids, desc="LIBERO-Plus eval")):
        set_seed_everywhere(cfg.seed + task_id)
        success, task_name = run_single_task(
            cfg, task_suite, task_id, model, resize_size,
            task_classification, processor, action_head, proprio_projector, log_file,
        )

        results.append({
            "task_id": task_id,
            "task_name": task_name,
            "success": success,
        })

        if success is not None:
            total_evaluated += 1
            if success:
                total_successes += 1

        # Periodic progress
        if (i + 1) % 50 == 0 or (i + 1) == len(task_ids):
            sr = total_successes / total_evaluated * 100 if total_evaluated > 0 else 0
            log_message(
                f"  Progress: {i+1}/{len(task_ids)} tasks | "
                f"{total_successes}/{total_evaluated} = {sr:.1f}%",
                log_file,
            )

    # Print detailed breakdown
    summary = print_results_breakdown(results, task_classification, log_file)

    # Save results JSON
    results_json_path = local_log_filepath.replace(".txt", "_results.json")
    with open(results_json_path, "w") as f:
        json.dump({
            "config": {
                "checkpoint": str(cfg.pretrained_checkpoint),
                "base_suite_name": cfg.base_suite_name,
                "perturbation_category": cfg.perturbation_category,
                "seed": cfg.seed,
            },
            "summary": summary,
            "per_task_results": results,
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
    eval_libero_plus()
