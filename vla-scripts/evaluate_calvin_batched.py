"""Batched CALVIN-ABC eval. ~B× faster than evaluate_calvin.py on a single GPU.

Architecture:
    Parent process:
        - Loads model on GPU once
        - Orchestrates B sequence runners (each = one chain of 5 subtasks)
        - Every requery, batches B observations into a single model.predict_action
          call and distributes the resulting B action chunks back to workers
    B subprocess CalvinEnv workers:
        - Each runs an independent PyBullet env (own pb.connect)
        - Receives commands (reset / step / get_info) over multiprocessing.Pipe
        - Runs lockstep with sibling workers; the slowest finishing a subtask
          gates batch progress to the next subtask

Usage (drop-in replacement for evaluate_calvin.py):
    python vla-scripts/evaluate_calvin_batched.py \
        --pretrained_checkpoint <ckpt> \
        --calvin_path calvin \
        --use_l1_regression True \
        --num_images_in_input 2 \
        --use_proprio True \
        --num_open_loop_steps 8 \
        --batch_size 8 \
        --num_sequences 1000

The eval is seeded via cfg.seed (same convention as evaluate_calvin.py).
Set debug=True to also write success/fail videos (slow; off by default).
"""
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import multiprocessing as mp

import draccus
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from termcolor import colored

# --- Path setup mirroring evaluate_calvin.py ---
CALVIN_ROOT = os.environ.get("CALVIN_ROOT", "calvin")
sys.path.append(f"{CALVIN_ROOT}/calvin_models")
sys.path.append(f"{CALVIN_ROOT}/calvin_env")

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

# Import model loading + setup helpers from the canonical eval. We reuse them
# verbatim — only the eval loop differs.
from evaluate_calvin import (
    GenerateConfig as _BaseConfig,
    print_and_save,
    count_success,
    process_action,
)
import vla_evaluation
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    get_vla,
)
from experiments.robot.robot_utils import set_seed_everywhere


# =============================================================================
# Subprocess CalvinEnv worker
# =============================================================================
def _env_worker_loop(
    dataset_path: str,
    observation_space: dict,
    cmd_pipe,  # bidirectional Connection
    base_seed: int,  # worker's deterministic baseline RNG seed
):
    """Run inside the subprocess: hold a CalvinEnv and serve RPC commands.

    Commands (sent from parent via cmd_pipe):
        ('reset', robot_obs, scene_obs, sequence_seed) -> ('ok', obs)
            sequence_seed: int — re-seeds python/numpy/torch RNG before reset
            so every sequence's trajectory is reproducible regardless of B.
        ('step', action_list)           -> ('ok', obs, reward, done, info)
        ('get_obs',)                    -> ('ok', obs)
        ('get_info',)                   -> ('ok', info)
        ('shutdown',)                   -> exits cleanly
    """
    import random as _random

    # Worker-baseline RNG (in case any code calls random.* outside of a sequence)
    _random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    # Defer heavyweight imports until inside the subprocess (PyBullet + EGL).
    # Each subprocess gets its own pb.connect() => no client conflicts.
    sys.path.append(f"{CALVIN_ROOT}/calvin_models")
    sys.path.append(f"{CALVIN_ROOT}/calvin_env")
    from calvin_env_wrapper import CalvinEnvWrapperRaw

    val_folder = Path(dataset_path) / "validation"
    env = CalvinEnvWrapperRaw(val_folder, observation_space, "cpu")

    try:
        while True:
            msg = cmd_pipe.recv()
            cmd = msg[0]
            if cmd == "reset":
                _, robot_obs, scene_obs, sequence_seed = msg
                # Per-sequence re-seed: ensures sequence i always produces the
                # same trajectory regardless of which worker handles it or B.
                import random as _random
                _random.seed(sequence_seed)
                np.random.seed(sequence_seed)
                torch.manual_seed(sequence_seed)
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                cmd_pipe.send(("ok", env.get_obs()))
            elif cmd == "step":
                _, action = msg
                obs, reward, done, info = env.step(action)
                cmd_pipe.send(("ok", obs, reward, done, info))
            elif cmd == "get_obs":
                cmd_pipe.send(("ok", env.get_obs()))
            elif cmd == "get_info":
                cmd_pipe.send(("ok", env.get_info()))
            elif cmd == "shutdown":
                break
            else:
                cmd_pipe.send(("err", f"unknown cmd {cmd}"))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            cmd_pipe.close()
        except Exception:
            pass


class EnvWorker:
    """Parent-side handle to one subprocess CalvinEnv worker."""

    def __init__(self, dataset_path: str, observation_space: dict, base_seed: int):
        ctx = mp.get_context("spawn")
        parent_pipe, child_pipe = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(
            target=_env_worker_loop,
            args=(dataset_path, observation_space, child_pipe, base_seed),
            daemon=True,
        )
        self.proc.start()
        child_pipe.close()
        self.pipe = parent_pipe

    def send(self, *msg):
        self.pipe.send(msg)

    def recv(self):
        return self.pipe.recv()

    def call(self, *msg):
        self.send(*msg)
        return self.recv()

    def shutdown(self):
        try:
            self.pipe.send(("shutdown",))
        except Exception:
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()


# =============================================================================
# Per-sequence state (parent-side bookkeeping for one env's chain of subtasks)
# =============================================================================
@dataclass
class SequenceState:
    sequence_idx: int                 # global sequence id (0..num_sequences-1)
    initial_state: dict               # CALVIN initial scene/robot state
    chain: list                       # list[str] of 5 subtask names
    succ_count: int = 0               # final score = consecutive successes (0..5)
    aborted: bool = False             # set when a subtask fails (chain stops)
    cur_subtask_idx: int = 0          # current subtask within `chain`
    # Per-step transient state (reset every subtask):
    action_queue: deque = field(default_factory=lambda: deque(maxlen=8))
    obs: dict = None                  # latest observation
    start_info: dict = None           # info captured at subtask start (for oracle)
    subtask_done: bool = False        # current subtask ended (success or oracle's "stop")
    subtask_succeeded: bool = False
    steps_in_subtask: int = 0
    # HiFI-3 ensemble: stores up to 3 action chunks (each 8x7 list of tensors)
    # captured at offsets 0, 1, 2 within a macro-step. None if not yet computed.
    action_buffers: list = field(default_factory=lambda: [None, None, None])


# =============================================================================
# Batched model forward
# =============================================================================
def _batched_predict_action(eva, obs_list, lang_list, processor, OPENVLA_IMAGE_SIZE=224):
    """Run model.predict_action on a batch of (obs, lang).

    Mirrors vla_evaluation.DualSystemCalvinEvaluation.step() but stacks B inputs
    into one forward pass. Returns: list[B] of action chunks (each (8, 7)).
    """
    from vla_evaluation import (
        check_image_format, resize_image_for_policy, center_crop_image,
        normalize_proprio,
    )

    B = len(obs_list)
    all_pixel_values = []
    all_input_ids = []
    all_attention_mask = []
    all_proprio = []

    for obs, instruction in zip(obs_list, lang_list):
        # Static (primary) and gripper (wrist) images
        img_static = obs["rgb_obs"]["rgb_static"]
        img_gripper = obs["rgb_obs"]["rgb_gripper"]
        check_image_format(img_static)
        check_image_format(img_gripper)
        if img_static.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            img_static = resize_image_for_policy(img_static, OPENVLA_IMAGE_SIZE)
        if img_gripper.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            img_gripper = resize_image_for_policy(img_gripper, OPENVLA_IMAGE_SIZE)
        pil_static = center_crop_image(Image.fromarray(img_static).convert("RGB"))
        pil_gripper = center_crop_image(Image.fromarray(img_gripper).convert("RGB"))

        prompt = (
            f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
            f"You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nWhat action should the robot take to "
            f"{instruction.lower()}?<|im_end|>\n<|im_start|>assistant\n"
        )

        inputs = processor(prompt, pil_static).to(eva.OFT.device, dtype=torch.bfloat16)
        wrist_inputs = processor(prompt, [pil_gripper]).to(eva.OFT.device, dtype=torch.bfloat16)
        # Concatenate primary + wrist along channel dim, matching single-env step()
        cat_pix = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

        all_pixel_values.append(cat_pix)
        all_input_ids.append(inputs["input_ids"])
        all_attention_mask.append(inputs["attention_mask"])

        proprio = np.concatenate([obs["robot_obs"][:7], obs["robot_obs"][-1:]])
        _ckey = next((k for k in ("calvin_abc", "calvin_abc_rlds", "calvin")
                      if k in eva.OFT.norm_stats), "calvin_abc_rlds")
        proprio = normalize_proprio(proprio, eva.OFT.norm_stats[_ckey]["proprio"])
        all_proprio.append(proprio)

    # Pad input_ids to common length. Use RIGHT-PADDING — action prediction here
    # is *not* autoregressive text generation. The model expects the prompt's
    # tokens at the same RoPE positions they had during training (i.e. starting
    # at position 0). Left-padding shifts the prompt to higher positions, which
    # breaks RoPE-conditioned heads (especially anchor's align_dir_proj that
    # reads off specific token positions). Right-padding keeps prompt at
    # positions [0..prompt_len_i-1] as in the single-env path.
    maxlen = max(x.shape[1] for x in all_input_ids)
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(processor.tokenizer, "eos_token_id", 0) or 0
    padded_ids = []
    padded_mask = []
    for ids, mask in zip(all_input_ids, all_attention_mask):
        pad = maxlen - ids.shape[1]
        if pad > 0:
            ids = torch.nn.functional.pad(ids, (0, pad), value=pad_id)   # pad on RIGHT
            mask = torch.nn.functional.pad(mask, (0, pad), value=0)       # mask on RIGHT
        padded_ids.append(ids)
        padded_mask.append(mask)

    pixel_values = torch.cat(all_pixel_values, dim=0)
    input_ids = torch.cat(padded_ids, dim=0)
    attention_mask = torch.cat(padded_mask, dim=0)
    proprio_batch = np.stack(all_proprio, axis=0)

    _ckey = next((k for k in ("calvin_abc", "calvin_abc_rlds", "calvin")
                  if k in eva.OFT.norm_stats), "calvin_abc_rlds")

    with torch.no_grad():
        action, _ = eva.OFT.predict_action(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            unnorm_key=_ckey,
            do_sample=False,
            proprio=proprio_batch,
            proprio_projector=eva.proprio_projector,
            action_head=eva.action_head,
            noisy_action_projector=eva.noisy_action_projector,
            use_film=False,
        )

    # action shape can be (B, K, 7) or flat (B*K, 7). Normalize to (B, K, 7).
    if action.ndim == 2:
        K = action.shape[0] // B
        action = action.reshape(B, K, 7)
    # gripper-action sign convention (matches single-env step())
    action[..., -1] = 1 - action[..., -1]
    K = action.shape[1]
    out = []
    for i in range(B):
        out.append([action[i, j] for j in range(min(K, 8))])
    return out


# =============================================================================
# HiFI-3 batched ensemble per-subtask runner
# =============================================================================
def _step_and_check_hi3(active, k, states, workers, eva, processor,
                       val_annotations, task_oracle):
    """Run one subtask's rollout in batched HiFI-3 ensemble mode.

    Mirrors `vla-scripts/evaluate_calvin.py:rollout_hi3`:
        outer for macro in range(80):
            3 batched model calls (at offsets 0, 1, 2 within macro-step)
            ~10 env steps per macro using 3-way action averaging
        Each env independently tracks subtask_done; when oracle declares the
        subtask solved we stop stepping that env (other envs continue).
    """
    def _batched_call(envs):
        """Run one batched model call for the given list of env indices.
        Returns: dict {env_idx -> 8-element list of action tensors}."""
        if not envs:
            return {}
        obs_list = [states[j].obs for j in envs]
        lang_list = [val_annotations[states[j].chain[k]][0] for j in envs]
        chunks = _batched_predict_action(eva, obs_list, lang_list, processor)
        return {j: chunks[idx] for idx, j in enumerate(envs)}

    def _step_envs(envs, action_fn):
        """Send 'step' to each env using action_fn(j) -> tensor/array. Then
        recv results, update obs, check subtask completion via task_oracle."""
        if not envs:
            return
        for j in envs:
            a = action_fn(j)
            a = a.float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
            a = process_action(a, "openvla")
            workers[j].send("step", a.tolist())
        for j in envs:
            ack = workers[j].recv()
            assert ack[0] == "ok"
            obs_j, _, _, info_j = ack[1], ack[2], ack[3], ack[4]
            states[j].obs = obs_j
            states[j].steps_in_subtask += 1
            subtask_name = states[j].chain[k]
            completed = task_oracle.get_task_info_for_set(
                states[j].start_info, info_j, {subtask_name},
            )
            if len(completed) > 0:
                states[j].subtask_done = True
                states[j].subtask_succeeded = True

    def _alive():
        return [j for j in active if not states[j].subtask_done]

    # rollout_hi3 outer macro loop = 80 iterations × ~10 env steps each
    for macro in range(80):
        # === Model call 0 (compute chunk0) ===
        envs = _alive()
        if not envs:
            return
        c0 = _batched_call(envs)
        for j in envs:
            states[j].action_buffers[0] = c0[j]
        # Env step using chunk0[0]
        _step_envs(_alive(), lambda j: states[j].action_buffers[0][0])

        # === Model call 1 (compute chunk1, after env stepped once) ===
        envs = _alive()
        if not envs:
            continue
        c1 = _batched_call(envs)
        for j in envs:
            states[j].action_buffers[1] = c1[j]
        # Env step using avg(c0[1], c1[0])
        _step_envs(_alive(), lambda j: (states[j].action_buffers[0][1]
                                        + states[j].action_buffers[1][0]) / 2)

        # === Model call 2 (compute chunk2, after env stepped twice) ===
        envs = _alive()
        if not envs:
            continue
        c2 = _batched_call(envs)
        for j in envs:
            states[j].action_buffers[2] = c2[j]
        # Env step using 3-way avg(c0[2], c1[1], c2[0])
        _step_envs(_alive(), lambda j: (states[j].action_buffers[0][2]
                                        + states[j].action_buffers[1][1]
                                        + states[j].action_buffers[2][0]) / 3)

        # === Inner loop t=2..6 (replicates the original code's range(2,7)
        # which duplicates t=2 and reuses the same averaging formula). ===
        # Note: t=2 here re-applies the same 3-way avg as the line above; this
        # matches rollout_hi3's behavior (the duplicate is in upstream code).
        for t in range(2, 7):
            envs = _alive()
            if not envs:
                break
            _step_envs(envs, lambda j, t=t: (states[j].action_buffers[0][t]
                                              + states[j].action_buffers[1][t-1]
                                              + states[j].action_buffers[2][t-2]) / 3)

        # === Step 7: avg(c1[7], c2[6]) ===
        envs = _alive()
        if not envs:
            continue
        _step_envs(envs, lambda j: (states[j].action_buffers[1][7]
                                    + states[j].action_buffers[2][6]) / 2)

        # === Step 8: c2[7] ===
        envs = _alive()
        if not envs:
            continue
        _step_envs(envs, lambda j: states[j].action_buffers[2][7])


# =============================================================================
# Batched eval loop
# =============================================================================
@dataclass
class BatchedConfig(_BaseConfig):
    batch_size: int = 8                # Parallel envs (== subprocess workers)
    num_sequences: int = 1000
    save_videos: bool = False          # Skip MoviePy/libx264 writes for speed
    use_hi3_ensemble: bool = False     # If True, use rollout_hi3-style 3-way temporal
                                       # ensemble (3 model calls / 8 env steps + action
                                       # averaging across overlapping chunks). Matches
                                       # vla-scripts/evaluate_calvin.py:rollout_hi3.
                                       # If False, use simple queue rollout (1 call / 8 steps).


def evaluate_batched(cfg: BatchedConfig):
    # ----- Maximum reproducibility for a single seed -----
    # 1. Master seed for python/numpy/torch/cuda (covers parent-process state).
    seed_everything(cfg.seed, workers=True)
    # 2. cuDNN deterministic mode (matters for the model forward pass).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 3. Deterministic algorithm selection in PyTorch (warn-only so unsupported
    #    bf16 ops don't crash; in practice the regression head is bf16-safe).
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    # 4. cuBLAS deterministic workspace — required by use_deterministic_algorithms
    #    on CUDA >= 10.2 to avoid silently nondeterministic gemm.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print(f"[batched] seed={cfg.seed} (cudnn.deterministic, deterministic_algorithms warn_only)", flush=True)

    # --- Model + Eva (single GPU instance, shared across B envs) ---
    cfg.unnorm_key = cfg.unnorm_key or "calvin_abc"
    cfg.model_family = "openvla"
    cfg.pretrained_checkpoint = str(cfg.pretrained_checkpoint)

    print(f"[batched] loading model from {cfg.pretrained_checkpoint}", flush=True)
    model = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression or cfg.use_diffusion else None
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None

    eva = vla_evaluation.DualSystemCalvinEvaluation(
        model, proprio_projector, noisy_action_projector, action_head,
        processor, use_x0_prediction=cfg.use_x0_prediction,
    )

    # --- Task oracle + lang annotations ---
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # --- Sequences to evaluate (deterministic given cfg.seed) ---
    eval_sequences = get_sequences(cfg.num_sequences)
    print(f"[batched] {len(eval_sequences)} sequences, batch_size={cfg.batch_size}", flush=True)

    # --- Spawn B env workers ---
    obs_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    dataset_path = os.path.join(CALVIN_ROOT, "dataset/task_ABC_D")

    print(f"[batched] spawning {cfg.batch_size} env workers...", flush=True)
    t_spawn = time.time()
    # Each worker gets a deterministic baseline seed = cfg.seed + 1000 * worker_idx.
    # Per-sequence seed (cfg.seed + sequence_idx) is sent in each 'reset' command.
    workers = [
        EnvWorker(dataset_path, obs_space, base_seed=cfg.seed + 1000 * w)
        for w in range(cfg.batch_size)
    ]
    print(f"[batched] workers ready in {time.time()-t_spawn:.1f}s", flush=True)

    # --- Output dir ---
    log_dir = Path(getattr(cfg, "log_dir", None) or cfg.local_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    sr_path = log_dir / "success_rate.txt"
    result_path = log_dir / "result.txt"
    open(sr_path, "w").close()

    ep_len = 360
    EP_LEN = ep_len
    OPEN_LOOP = cfg.num_open_loop_steps

    all_results = []
    pbar = tqdm.tqdm(total=len(eval_sequences), desc="CALVIN-batched")

    # --- Main loop: process sequences in chunks of B ---
    for batch_start in range(0, len(eval_sequences), cfg.batch_size):
        batch_seqs = eval_sequences[batch_start:batch_start + cfg.batch_size]
        B = len(batch_seqs)
        states = []
        for j, (initial_state, chain) in enumerate(batch_seqs):
            states.append(SequenceState(
                sequence_idx=batch_start + j,
                initial_state=initial_state,
                chain=list(chain),
            ))

        # Reset all B envs to their initial states (parallel).
        # Each sequence uses its own deterministic seed (cfg.seed + sequence_idx)
        # so identical sequence_idx always gives the same trajectory regardless
        # of B or which worker handles it.
        for j in range(B):
            robot_obs, scene_obs = get_env_state_for_initial_condition(states[j].initial_state)
            seq_seed = cfg.seed + states[j].sequence_idx
            workers[j].send("reset", robot_obs, scene_obs, seq_seed)
        for j in range(B):
            ack = workers[j].recv()
            assert ack[0] == "ok"
            states[j].obs = ack[1]

        # Process the chain subtask-by-subtask. All B sequences advance together,
        # gated by the slowest. An env that fails subtask k aborts (no further
        # subtasks). An env that succeeds early waits idle until all peers finish.
        for k in range(5):
            # Collect indices still in the running (chain not aborted yet)
            active = [j for j in range(B) if not states[j].aborted and states[j].cur_subtask_idx == k]
            if not active:
                break

            # Reset model state once per subtask (matches model.reset() in single-env rollout)
            eva.reset()

            for j in active:
                states[j].action_queue.clear()
                states[j].action_buffers = [None, None, None]
                states[j].subtask_done = False
                states[j].subtask_succeeded = False
                states[j].steps_in_subtask = 0
                # start_info captured *before* any step in this subtask — task_oracle
                # uses it as the "before" snapshot to detect the subtask completing.
                workers[j].send("get_info")
            for j in active:
                ack = workers[j].recv()
                states[j].start_info = ack[1]

            if not cfg.use_hi3_ensemble:
                # ---- ORIGINAL queue rollout (1 model call per 8 env steps) ----
                # Per-step inner loop
                for step in range(EP_LEN):
                    # Identify envs that need a model requery (queue empty AND not done)
                    requery = [j for j in active
                               if not states[j].subtask_done and len(states[j].action_queue) == 0]
                    if requery:
                        obs_list = [states[j].obs for j in requery]
                        lang_list = [val_annotations[states[j].chain[k]][0] for j in requery]
                        chunks = _batched_predict_action(eva, obs_list, lang_list, processor)
                        for idx, j in enumerate(requery):
                            states[j].action_queue.extend(chunks[idx])

                    # Send step commands in parallel
                    stepping = [j for j in active if not states[j].subtask_done]
                    if not stepping:
                        break
                    for j in stepping:
                        a = states[j].action_queue.popleft()
                        a = a.float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                        # Match single-env rollout: binarize gripper to {-1,+1} and
                        # invert sign for openvla (the env's apply_action asserts
                        # gripper_action ∈ {-1, 1}). process_action does both.
                        a = process_action(a, "openvla")
                        workers[j].send("step", a.tolist())

                    # Receive results & oracle-check
                    for j in stepping:
                        ack = workers[j].recv()
                        assert ack[0] == "ok"
                        obs_j, _, _, info_j = ack[1], ack[2], ack[3], ack[4]
                        states[j].obs = obs_j
                        states[j].steps_in_subtask += 1
                        subtask_name = states[j].chain[k]
                        completed = task_oracle.get_task_info_for_set(
                            states[j].start_info, info_j, {subtask_name},
                        )
                        if len(completed) > 0:
                            states[j].subtask_done = True
                            states[j].subtask_succeeded = True
            else:
                # ---- HiFI-3 batched ensemble (mirrors evaluate_calvin.py:rollout_hi3) ----
                # 3 model calls per macro-step + action averaging across overlapping chunks.
                # 80 outer macro-steps × ~10 env steps = up to ~800 env steps per subtask.
                _step_and_check_hi3(
                    active, k, states, workers, eva, processor, val_annotations,
                    task_oracle,
                )

            # Subtask k loop ended — apply success/fail to the chains
            for j in active:
                if states[j].subtask_succeeded:
                    states[j].succ_count += 1
                    states[j].cur_subtask_idx += 1
                else:
                    states[j].aborted = True

        # Batch done — record and dump partial summary
        for j in range(B):
            all_results.append(states[j].succ_count)

        success_list = count_success(all_results)
        with open(sr_path, "a") as f:
            line = f"{batch_start + B}/{len(eval_sequences)}: " + \
                   " | ".join(f"{sr:.3f}" for sr in success_list) + "\n"
            f.write(line)
        pbar.update(B)
        pbar.set_description(
            " ".join([f"{i+1}/5: {v*100:.1f}%" for i, v in enumerate(success_list)])
        )

    pbar.close()

    # --- Final print + save ---
    print("\n[batched] FINAL:")
    success_list = count_success(all_results)
    for i, v in enumerate(success_list):
        print(f"  {i+1}/5 : {v*100:.2f}%")
    avg_len = sum(success_list)
    print(f"  avg_len: {avg_len:.3f}")

    # Build a sequences-like object so print_and_save doesn't choke
    fake_sequences = [(s.initial_state, s.chain) for s in
                      [SequenceState(i, *eval_sequences[i], []) for i in range(len(eval_sequences))][:0]]
    # Just dump our own JSON and skip print_and_save's tqdm dependency
    with open(log_dir / "results.json", "w") as f:
        json.dump({
            "num_sequences": len(eval_sequences),
            "results": all_results,
            "success_rate_per_length": success_list,
            "avg_len": avg_len,
            "checkpoint": cfg.pretrained_checkpoint,
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
        }, f, indent=2)
    print(f"[batched] wrote {log_dir / 'results.json'}")

    # Shutdown workers
    for w in workers:
        w.shutdown()


@draccus.wrap()
def main(cfg: BatchedConfig):
    set_seed_everywhere(cfg.seed)
    evaluate_batched(cfg)


if __name__ == "__main__":
    # spawn is required for CUDA-safe forking (parent has CUDA initialized)
    mp.set_start_method("spawn", force=True)
    main()
