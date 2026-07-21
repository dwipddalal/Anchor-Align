import torch
torch.backends.cuda.enable_cudnn_sdp(False)  # E12: avoid NaN on GH200 bf16
"""
run_libero_pro_eval.py

Evaluates a trained policy on LIBERO-PRO perturbation benchmarks
(language, swap, object, task perturbations).

Supports parallel environment evaluation via SubprocVectorEnv + batched
model inference for ~5-10x speedup over sequential evaluation.

Fork of experiments/robot/libero/run_libero_eval.py with targeted changes:
  1. sys.path insert for LIBERO-PRO package
  2. perturbation_type + base_suite_name config fields
  3. check_unnorm_key uses base_suite_name (training key)
  4. max_steps lookup uses base_suite_name
  5. BDDL language extraction for language perturbation
  6. Rollout paths organized by perturbation type
  7. Parallel env evaluation with SubprocVectorEnv + batched inference
"""

import json
import logging
import math
import multiprocessing
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

# Ensure spawn method for LIBERO's SubprocVectorEnv
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# ── Change 1: Make LIBERO-PRO's `libero` package override standard LIBERO ──
sys.path.insert(0, os.environ.get("LIBERO_PRO_ROOT", "../libero-variants/LIBERO-PRO"))

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv

import wandb

# Append current directory so that interpreter can find experiments.robot
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


# Define task suite constants (base suites only — perturbation variants are derived)
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

# Valid perturbation types
VALID_PERTURBATION_TYPES = ("lan", "swap", "object", "task")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO-PRO environment-specific parameters
    #################################################################################################################
    # ── Change 2: perturbation_type + base_suite_name ──
    base_suite_name: str = "libero_spatial"          # Base task suite (libero_spatial, libero_object, etc.)
    perturbation_type: str = "lan"                   # Perturbation type: lan, swap, object, task
    task_suite_name: str = ""                        # Derived: {base_suite_name}_{perturbation_type} (set in validate_config)
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Parallel evaluation parameters
    #################################################################################################################
    num_parallel_envs: int = 10                      # Number of parallel environments per task (1 = sequential)
    save_videos: bool = True                         # Whether to save rollout videos (disable for speed)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs/libero_pro"  # ── Change 6: separate log dir ──

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    use_seed_in_env: bool = False                    # if True, mix cfg.seed into env_seed for true multi-seed variance

    # Task-parallel evaluation: run a subset of tasks (for multi-GPU parallelism)
    start_task_id: int = 0                           # First task ID to evaluate (inclusive)
    end_task_id: int = -1                            # Last task ID to evaluate (inclusive, -1 = all)

    # fmt: on
    save_version: str = "vla-adapter"                # version of
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # ── Change 2: Validate base suite and perturbation type, derive task_suite_name ──
    assert cfg.base_suite_name in [suite.value for suite in TaskSuite], (
        f"Invalid base suite: {cfg.base_suite_name}"
    )
    assert cfg.perturbation_type in VALID_PERTURBATION_TYPES, (
        f"Invalid perturbation type: {cfg.perturbation_type}. Must be one of {VALID_PERTURBATION_TYPES}"
    )
    cfg.task_suite_name = f"{cfg.base_suite_name}_{cfg.perturbation_type}"

    # ── Change 6: Organize rollouts by perturbation type ──
    if cfg.save_version == "vla-adapter":  # Only override default
        cfg.save_version = f"libero_pro/{cfg.perturbation_type}"

    assert cfg.num_parallel_envs >= 1, "num_parallel_envs must be >= 1"



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key.

    Uses base_suite_name (e.g. 'libero_spatial') since the checkpoint was trained
    on the base suite, not a perturbation variant.
    """
    # ── Change 3: Use base_suite_name for normalization stats lookup ──
    unnorm_key = cfg.base_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()



def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None



def prepare_observation_from_vec(vec_obs, env_idx, resize_size):
    """Extract and prepare observation for env_idx from vectorized observation.

    Args:
        vec_obs: Vectorized observation from SubprocVectorEnv/DummyVectorEnv.
                 Stacked as np.array of dicts: vec_obs[env_idx]["key"].
        env_idx: Index of the environment to extract observation for.
        resize_size: Target image size for the policy.

    Returns:
        observation: Dict with 'full_image', 'wrist_image', 'state' for the policy.
        img: Original (rotated) agentview image for replay video.
    """
    obs_i = vec_obs[env_idx]

    # Extract and rotate images (180 degrees to match train preprocessing)
    img = obs_i["agentview_image"][::-1, ::-1]
    wrist_img = obs_i["robot0_eye_in_hand_image"][::-1, ::-1]

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate((
            obs_i["robot0_eef_pos"],
            quat2axisangle(obs_i["robot0_eef_quat"]),
            obs_i["robot0_gripper_qpos"],
        )),
    }

    return observation, img.copy()



def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action



def get_bddl_language(task):
    """Extract the perturbed language instruction from a BDDL file's (:language ...) field.

    For language perturbation, task.language is derived from the filename and may NOT
    contain the actual perturbed instruction. The BDDL file's (:language ...) field
    holds the true perturbed text.
    """
    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    with open(bddl_file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("(:language"):
                # Extract text between "(:language " and closing ")"
                lang = stripped[len("(:language"):].rstrip(")").strip()
                return lang
    return None


def _make_env_fn(env_args, env_seed=0):
    """Factory that returns a proper env constructor (avoids lambda closure bug)."""
    def _init():
        # Seed subprocess RNG BEFORE env construction so robosuite init is deterministic
        import random as _random
        np.random.seed(env_seed)
        _random.seed(env_seed)
        env = OffScreenRenderEnv(**env_args)
        env.seed(env_seed)
        return env
    return _init


def run_task_parallel(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None,
):
    """Run evaluation for a single task using parallel environments + batched inference.

    Spawns up to cfg.num_parallel_envs environments per round, runs them in lockstep,
    and uses batched model forward passes for all environments that need requerying.
    Falls back to DummyVectorEnv (sequential equivalent) when num_parallel_envs=1.
    """
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Get task description from a temporary single env
    temp_env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    # Get BDDL file path for vectorized env creation
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    temp_env.close()
    del temp_env

    # ── Change 5: For language perturbation, extract instruction from BDDL file ──
    if cfg.perturbation_type == "lan":
        bddl_lang = get_bddl_language(task)
        if bddl_lang is not None:
            log_message(
                f"[lan] Overriding task.language with BDDL instruction:\n"
                f"  filename-derived: {task_description}\n"
                f"  BDDL (:language): {bddl_lang}",
                log_file,
            )
            task_description = bddl_lang
        else:
            log_message(
                f"[lan] WARNING: Could not parse (:language) from BDDL file for task {task_id}; "
                f"falling back to filename-derived: {task_description}",
                log_file,
            )

    # Determine valid episodes (filter out failed expert demos for custom initial states)
    valid_episodes = []
    for episode_idx in range(cfg.num_trials_per_task):
        if cfg.initial_states_path == "DEFAULT":
            valid_episodes.append((episode_idx, initial_states[episode_idx]))
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
            valid_episodes.append((episode_idx, initial_state))

    if not valid_episodes:
        log_message(f"No valid episodes for task {task_id}, skipping.", log_file)
        return total_episodes, total_successes

    N = min(cfg.num_parallel_envs, len(valid_episodes))
    num_rounds = math.ceil(len(valid_episodes) / N)
    max_steps = TASK_MAX_STEPS[TaskSuite(cfg.base_suite_name)]

    log_message(f"\nTask {task_id}: {task_description}", log_file)
    log_message(f"  {len(valid_episodes)} episodes, {N} parallel envs, {num_rounds} rounds", log_file)

    task_episodes, task_successes = 0, 0

    def _run_one_round(round_episodes, use_subprocess=True):
        """Run one round of episodes. Returns list of (episode_idx, success) results.

        If use_subprocess=True, uses SubprocVectorEnv for parallelism.
        If False, uses DummyVectorEnv (in-process, no pipes = no EOFError crashes).
        """
        batch_size = len(round_episodes)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": cfg.env_img_res,
            "camera_widths": cfg.env_img_res,
        }
        if not cfg.use_seed_in_env:
            env_fns = [_make_env_fn(env_args, env_seed=task_id * 1000 + actual_ep_idx)
                       for actual_ep_idx, _ in round_episodes]
        else:
            env_fns = [_make_env_fn(env_args, env_seed=cfg.seed * 100000 + task_id * 1000 + actual_ep_idx)
                       for actual_ep_idx, _ in round_episodes]
        if batch_size > 1 and use_subprocess:
            env = SubprocVectorEnv(env_fns)
        else:
            env = DummyVectorEnv(env_fns)

        env.reset()
        batch_init_states = [ep[1] for ep in round_episodes]
        obs = env.set_init_state(batch_init_states)

        dummy_actions = np.tile(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64), (batch_size, 1))
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(dummy_actions)

        action_queues = [deque(maxlen=cfg.num_open_loop_steps) for _ in range(batch_size)]
        dones = [False] * batch_size
        successes = [False] * batch_size
        replay_images = [[] for _ in range(batch_size)] if cfg.save_videos else None

        for t in range(max_steps):
            need_requery = [i for i in range(batch_size) if len(action_queues[i]) == 0 and not dones[i]]

            if need_requery:
                obs_list = []
                for i in need_requery:
                    ob, img = prepare_observation_from_vec(obs, i, resize_size)
                    obs_list.append(ob)
                    if cfg.save_videos:
                        replay_images[i].append(img)

                actions_list = get_vla_action_batch(
                    cfg, model, processor, obs_list, task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )
                for idx, env_i in enumerate(need_requery):
                    action_queues[env_i].extend(actions_list[idx])

            if cfg.save_videos:
                for i in range(batch_size):
                    if i not in need_requery and not dones[i]:
                        _, img = prepare_observation_from_vec(obs, i, resize_size)
                        replay_images[i].append(img)

            actions = np.zeros((batch_size, 7))
            for i in range(batch_size):
                if not dones[i] and len(action_queues[i]) > 0:
                    actions[i] = process_action(action_queues[i].popleft(), cfg.model_family)
                else:
                    actions[i] = np.array([0, 0, 0, 0, 0, 0, -1])

            obs, rewards, done_flags, infos = env.step(actions)

            for i in range(batch_size):
                if done_flags[i] and not dones[i]:
                    dones[i] = True
                    successes[i] = True

            if all(dones):
                break

        env.close()
        return successes, replay_images

    def _run_episodes_adaptive(episodes, max_parallel, round_label, log_file):
        """Run a batch of episodes with adaptive parallelism.

        Tries max_parallel first. On any crash, halves parallelism and retries
        the failed chunk. Keeps halving until it works or falls back to sequential
        (DummyVectorEnv, batch_size=1).
        """
        parallel = max_parallel
        remaining = list(episodes)
        all_successes = []
        all_replay_images = [[] for _ in episodes] if cfg.save_videos else None
        ep_offset = 0

        while remaining:
            chunk = remaining[:parallel]
            try:
                use_sub = parallel > 1
                successes, replay_imgs = _run_one_round(chunk, use_subprocess=use_sub)
                # Success — record results and advance
                all_successes.extend(successes)
                if cfg.save_videos and replay_imgs:
                    for i, img in enumerate(replay_imgs):
                        all_replay_images[ep_offset + i] = img
                ep_offset += len(chunk)
                remaining = remaining[parallel:]
            except Exception as e:
                log_message(
                    f"  {round_label}: crashed with {parallel} parallel envs "
                    f"({type(e).__name__}). ",
                    log_file,
                )
                if parallel > 1:
                    parallel = max(1, parallel // 2)
                    log_message(f"  Retrying with {parallel} parallel envs...", log_file)
                else:
                    # Even sequential failed — count this episode as failure
                    log_message(
                        f"  Episode also failed sequentially, counting as failure.",
                        log_file,
                    )
                    all_successes.append(False)
                    ep_offset += 1
                    remaining = remaining[1:]

        return all_successes, all_replay_images

    for round_idx in range(num_rounds):
        ep_start = round_idx * N
        ep_end = min(ep_start + N, len(valid_episodes))
        round_episodes = valid_episodes[ep_start:ep_end]

        successes, replay_images = _run_episodes_adaptive(
            round_episodes, len(round_episodes), f"Round {round_idx+1}", log_file
        )

        # Record results and save videos
        for i in range(len(round_episodes)):
            task_episodes += 1
            total_episodes += 1
            if successes[i]:
                task_successes += 1
                total_successes += 1

            if cfg.save_videos and replay_images is not None:
                save_rollout_video(
                    replay_images[i], total_episodes, success=successes[i],
                    task_description=task_description, log_file=log_file,
                    save_version=save_version,
                )

            log_message(f"  Episode {total_episodes}: success={successes[i]}", log_file)

        log_message(
            f"  Round {round_idx+1}/{num_rounds} done. "
            f"Running total: {total_successes}/{total_episodes} "
            f"({total_successes / total_episodes * 100:.1f}%)",
            log_file,
        )

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes



@draccus.wrap()
def eval_libero_pro(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO-PRO perturbation benchmarks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO-PRO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Base suite: {cfg.base_suite_name}", log_file)
    log_message(f"Perturbation type: {cfg.perturbation_type}", log_file)
    log_message(f"Parallel envs: {cfg.num_parallel_envs}", log_file)
    log_message(f"Save videos: {cfg.save_videos}", log_file)

    # Determine task range for parallel evaluation
    start_task = cfg.start_task_id
    end_task = num_tasks - 1 if cfg.end_task_id < 0 else min(cfg.end_task_id, num_tasks - 1)
    task_ids = list(range(start_task, end_task + 1))
    log_message(f"Evaluating tasks {start_task} to {end_task} ({len(task_ids)} tasks)", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        set_seed_everywhere(cfg.seed + task_id)
        total_episodes, total_successes = run_task_parallel(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            total_episodes,
            total_successes,
            log_file,
            cfg.save_version,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero_pro()
