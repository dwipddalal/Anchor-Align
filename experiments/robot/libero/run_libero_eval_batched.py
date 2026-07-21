import torch
torch.backends.cuda.enable_cudnn_sdp(False)  # E12: avoid NaN on GH200 bf16
"""
run_libero_eval_batched.py

Evaluates a trained policy in LIBERO using parallel environments + batched
model inference for ~5-10x speedup over the sequential run_libero_eval.py.

Key difference from run_libero_eval.py:
  - num_parallel_envs episodes run simultaneously via SubprocVectorEnv
  - All envs that need requerying are batched into a single GPU forward pass
  - save_videos defaults to False for maximum speed

Usage:
    python experiments/robot/libero/run_libero_eval_batched.py \
        --pretrained_checkpoint <path> \
        --task_suite_name libero_object \
        --num_parallel_envs 16 \
        --save_videos False
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

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv

import wandb

# Ensure spawn method for SubprocVectorEnv
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

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
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT  = "libero_object"
    LIBERO_GOAL    = "libero_goal"
    LIBERO_10      = "libero_10"
    LIBERO_90      = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT:  280,
    TaskSuite.LIBERO_GOAL:    300,
    TaskSuite.LIBERO_10:      520,
    TaskSuite.LIBERO_90:      400,
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

    # ── LIBERO env ─────────────────────────────────────────────────────────────
    task_suite_name:        str  = "libero_spatial"
    num_steps_wait:         int  = 10
    num_trials_per_task:    int  = 50
    initial_states_path:    str  = "DEFAULT"
    env_img_res:            int  = 256

    # ── Batched eval ──────────────────────────────────────────────────────────
    num_parallel_envs:      int  = 16        # episodes per round (tune for your GPU)
    save_videos:            bool = False     # set True only for debugging (slow)

    # ── Task range (for splitting across multiple runs if needed) ──────────────
    start_task_id:          int  = 0
    end_task_id:            int  = -1        # -1 = all tasks

    # ── Negation / language probe ─────────────────────────────────────────────
    negation_prefix:        str  = ""        # e.g. "do not " — prepended to every task description

    # ── Logging ────────────────────────────────────────────────────────────────
    run_id_note:            Optional[str] = None
    local_log_dir:          str           = "./experiments/logs"
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
    assert cfg.task_suite_name in [s.value for s in TaskSuite], \
        f"Invalid task suite: {cfg.task_suite_name}"
    assert cfg.num_parallel_envs >= 1


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
    key = cfg.task_suite_name
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    assert key in model.norm_stats, f"Unnorm key '{key}' not found in norm_stats!"
    cfg.unnorm_key = key


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
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
    img       = obs_i["agentview_image"][::-1, ::-1]      # rotate 180° (matches train)
    wrist_img = obs_i["robot0_eye_in_hand_image"][::-1, ::-1]
    observation = {
        "full_image":  resize_image_for_policy(img, resize_size),
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


def run_task_batched(
    cfg, task_suite, task_id, model, resize_size,
    processor, action_head, proprio_projector,
    total_episodes, total_successes, log_file, save_version,
):
    """Run one task using N parallel envs + batched inference."""
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Get task description + bddl path via temp env
    temp_env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    temp_env.close()
    del temp_env

    # ── Optional negation / language probe ────────────────────────────────────
    if cfg.negation_prefix:
        # Normalize: ensure exactly one space between prefix and original prompt
        prefix = cfg.negation_prefix.rstrip() + " "
        original_description = task_description
        task_description = prefix + task_description
    else:
        original_description = task_description

    valid_episodes = [(i, initial_states[i]) for i in range(cfg.num_trials_per_task)]
    N = min(cfg.num_parallel_envs, len(valid_episodes))
    num_rounds = math.ceil(len(valid_episodes) / N)
    max_steps = TASK_MAX_STEPS[TaskSuite(cfg.task_suite_name)]

    if cfg.negation_prefix:
        log_message(f"\nTask {task_id}: original='{original_description}'  ->  negated='{task_description}'", log_file)
    else:
        log_message(f"\nTask {task_id}: {task_description}", log_file)
    log_message(f"  {len(valid_episodes)} episodes | {N} parallel envs | {num_rounds} rounds", log_file)

    task_episodes, task_successes = 0, 0

    def _run_one_round(round_episodes, use_subprocess=True):
        """Run one round of episodes. Returns (successes_list, replay_images).

        If use_subprocess=True, uses SubprocVectorEnv for parallelism.
        If False, uses DummyVectorEnv (in-process, no pipes = no EOFError crashes).
        """
        batch_size = len(round_episodes)
        env_args = {
            "bddl_file_name":  task_bddl_file,
            "camera_heights":  cfg.env_img_res,
            "camera_widths":   cfg.env_img_res,
        }
        env_fns = [_make_env_fn(env_args, env_seed=task_id * 1000 + actual_ep_idx)
                   for actual_ep_idx, _ in round_episodes]
        if batch_size > 1 and use_subprocess:
            vec_env = SubprocVectorEnv(env_fns)
        else:
            vec_env = DummyVectorEnv(env_fns)

        vec_env.reset()
        obs = vec_env.set_init_state([ep[1] for ep in round_episodes])

        dummy = np.tile(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64), (batch_size, 1))
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = vec_env.step(dummy)

        action_queues = [deque(maxlen=cfg.num_open_loop_steps) for _ in range(batch_size)]
        dones         = [False] * batch_size
        successes     = [False] * batch_size
        replay_images = [[] for _ in range(batch_size)] if cfg.save_videos else None

        for t in range(max_steps):
            need_requery = [i for i in range(batch_size)
                            if len(action_queues[i]) == 0 and not dones[i]]

            if need_requery:
                obs_list = []
                for i in need_requery:
                    ob, img = prepare_obs_from_vec(obs, i, resize_size)
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
                        _, img = prepare_obs_from_vec(obs, i, resize_size)
                        replay_images[i].append(img)

            actions = np.zeros((batch_size, 7))
            for i in range(batch_size):
                if not dones[i] and action_queues[i]:
                    actions[i] = process_action(action_queues[i].popleft(), cfg.model_family)
                else:
                    actions[i] = np.array([0, 0, 0, 0, 0, 0, -1])

            obs, _, done_flags, _ = vec_env.step(actions)

            for i in range(batch_size):
                if done_flags[i] and not dones[i]:
                    dones[i] = True
                    successes[i] = True

            if all(dones):
                break

        vec_env.close()
        return successes, replay_images

    for round_idx in range(num_rounds):
        ep_start = round_idx * N
        ep_end   = min(ep_start + N, len(valid_episodes))
        round_episodes = valid_episodes[ep_start:ep_end]

        try:
            successes, replay_images = _run_one_round(round_episodes, use_subprocess=True)
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            log_message(
                f"  Round {round_idx+1}: SubprocVectorEnv crashed ({type(e).__name__}), "
                f"retrying {len(round_episodes)} episodes sequentially...",
                log_file,
            )
            successes = []
            replay_images = [[] for _ in round_episodes] if cfg.save_videos else None
            for ep_i, ep in enumerate(round_episodes):
                try:
                    ep_success, ep_imgs = _run_one_round([ep], use_subprocess=False)
                    successes.append(ep_success[0])
                    if cfg.save_videos and ep_imgs:
                        replay_images[ep_i] = ep_imgs[0]
                except Exception as ep_e:
                    log_message(
                        f"  Episode {ep_i} also failed sequentially ({type(ep_e).__name__}: {ep_e}), "
                        f"counting as failure.",
                        log_file,
                    )
                    successes.append(False)

        # Record results
        for i in range(len(round_episodes)):
            task_episodes   += 1
            total_episodes  += 1
            if successes[i]:
                task_successes  += 1
                total_successes += 1
            if cfg.save_videos and replay_images:
                save_rollout_video(
                    replay_images[i], total_episodes,
                    success=successes[i], task_description=task_description,
                    log_file=log_file, save_version=save_version,
                )
            log_message(f"  ep {total_episodes}: success={successes[i]}", log_file)

        log_message(
            f"  Round {round_idx+1}/{num_rounds}: "
            f"{total_successes}/{total_episodes} "
            f"({total_successes/total_episodes*100:.1f}%)",
            log_file,
        )

    task_sr = task_successes / task_episodes if task_episodes > 0 else 0
    log_message(f"Task {task_id} SR: {task_sr:.4f} ({task_successes}/{task_episodes})", log_file)

    if cfg.use_wandb:
        wandb.log({f"success_rate/{task_description}": task_sr,
                   f"num_episodes/{task_description}": task_episodes})

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    start_task = cfg.start_task_id
    end_task   = num_tasks - 1 if cfg.end_task_id < 0 else min(cfg.end_task_id, num_tasks - 1)
    task_ids   = list(range(start_task, end_task + 1))

    log_message(f"Suite: {cfg.task_suite_name} | Tasks {start_task}–{end_task} | "
                f"num_parallel_envs={cfg.num_parallel_envs}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        set_seed_everywhere(cfg.seed + task_id)
        total_episodes, total_successes = run_task_batched(
            cfg, task_suite, task_id, model, resize_size,
            processor, action_head, proprio_projector,
            total_episodes, total_successes, log_file, cfg.save_version,
        )

    final_sr = total_successes / total_episodes if total_episodes > 0 else 0
    log_message(f"\n{'='*60}", log_file)
    log_message(f"FINAL: {total_successes}/{total_episodes} = {final_sr*100:.1f}%", log_file)
    log_message(f"{'='*60}", log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_sr, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()
    return final_sr


if __name__ == "__main__":
    eval_libero()
