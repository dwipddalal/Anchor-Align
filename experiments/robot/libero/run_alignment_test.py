"""Alignment test: LP_ACC, AP_ACC_top1, LP↔AP alignment from LIBERO training data.

Three pathways per frame:
  - AP: standard policy forward → action_head → first-action Δxyz → top-1 axis label
  - LP_trained (v7 only): pre_action_hidden → align_dir_proj → frozen lm_head → argmax over the 6 direction-word token IDs
  - LP_gen: prompt with the MolmoAct-style direction question + same 2-image stack → generate 1 word

GT label set is computed from the dataset's ground-truth Δxyz with δ=0.2 multi-axis tolerance
(matches `accuracy_metric._gt_label_set_from_vec3`).

Frames are sampled directly from the LIBERO RLDS training pipeline.

Usage:
  python run_alignment_test.py --ckpt <path> --dataset libero_spatial_no_noops --num-frames 5 --phase verify
  python run_alignment_test.py --ckpt <path> --dataset libero_spatial_no_noops --num-frames 1500 --phase full --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIRECTION_WORDS = ["forward", "backward", "left", "right", "up", "down"]

# The alignment head predicts a motion-direction word. The libero-spatial/object/goal checkpoints
# use a transposed X/Y direction-label convention relative to the LIBERO/robosuite frame, so their
# alignment head emits "right" for forward motion, "forward" for left, and so on. Passing
# --remap-axes applies a fixed 4-cycle on the horizontal words (up/down untouched) to map those
# predictions back to LIBERO labels for scoring.
#
# WHEN TO PASS --remap-axes (per released Dwipz/Anchor-Align checkpoint):
#   libero-spatial : YES   (transposed X/Y convention)
#   libero-object  : YES   (transposed X/Y convention)
#   libero-goal    : YES   (transposed X/Y convention)
#   libero-long    : NO    (LIBERO-frame convention; remapping would introduce a transpose)
# For any other checkpoint, pass --remap-axes iff its alignment head uses the transposed
# convention; the remap is a 4-cycle, so applying it to a LIBERO-frame checkpoint corrupts labels.
AXIS_SWAP_REMAP = {
    "right": "forward",
    "left": "backward",
    "forward": "left",
    "backward": "right",
    "up": "up",
    "down": "down",
}


def remap_axis_swap(word: Optional[str]) -> Optional[str]:
    if word is None:
        return None
    return AXIS_SWAP_REMAP.get(word, word)


# ---------- GT label helpers (mirrors accuracy_metric.py) ----------

def gt_label_set_from_vec3(vec, delta: float = 0.2, eps: float = 1e-8) -> Set[str]:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    ax, ay, az = abs(x), abs(y), abs(z)
    m = max(ax, ay, az)
    if m < eps:
        return set()
    t = (1.0 - delta) * m
    out: Set[str] = set()
    if ax >= t:
        out.add("forward" if x >= 0 else "backward")
    if ay >= t:
        out.add("left" if y >= 0 else "right")
    if az >= t:
        out.add("up" if z >= 0 else "down")
    return out


def ap_top1_label_from_vec3(vec, eps: float = 1e-8) -> Optional[str]:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    ax, ay, az = abs(x), abs(y), abs(z)
    m = max(ax, ay, az)
    if m < eps:
        return None
    if ax >= ay and ax >= az:
        return "forward" if x >= 0 else "backward"
    if ay >= az:
        return "left" if y >= 0 else "right"
    return "up" if z >= 0 else "down"


def normalize_direction_label(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip().lower()
    for ch in [".", ",", ";", ":", "(", ")", "[", "]", "{", "}", "?", "!"]:
        t = t.replace(ch, " ")
    t = " ".join(t.split())
    syn = {
        "left": "left", "right": "right",
        "forward": "forward", "forwards": "forward", "front": "forward",
        "back": "backward", "backward": "backward", "backwards": "backward",
        "up": "up", "upward": "up", "upwards": "up",
        "down": "down", "downward": "down", "downwards": "down",
    }
    last_dir, last_pos = None, -1
    for tok in t.split():
        for d in syn:
            pos = tok.rfind(d)
            if pos > last_pos:
                last_pos, last_dir = pos, syn[d]
    if last_dir is not None:
        return last_dir
    for w, n in syn.items():
        if w in t:
            return n
    return None


# ---------- Eval config (mimics finetune cfg fields used by openvla_utils) ----------

class _Cfg:
    def __init__(self, ckpt: str, unnorm_key: str):
        self.pretrained_checkpoint = ckpt
        self.unnorm_key = unnorm_key
        self.num_images_in_input = 2
        self.use_proprio = True
        self.use_l1_regression = True
        self.use_film = False
        self.use_diffusion = False
        self.lora_rank = 64
        self.center_crop = True
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.num_open_loop_steps = 1
        self.use_minivlm = False
        self.save_version = ""
        self.use_pro_version = True  # matches run_libero_eval_batched.py default


# ---------- Checkpoint helpers ----------

def find_ckpt_file(ckpt_dir: str, prefix: str) -> Optional[str]:
    for f in os.listdir(ckpt_dir):
        if f.startswith(prefix) and f.endswith(".pt"):
            return os.path.join(ckpt_dir, f)
    return None


def load_align_dir_proj(ckpt_dir: str, llm_dim: int) -> Optional[torch.nn.Linear]:
    path = find_ckpt_file(ckpt_dir, "align_dir_proj--")
    if path is None:
        return None
    proj = torch.nn.Linear(llm_dim, llm_dim).to(torch.bfloat16).to(DEVICE)
    sd = torch.load(path, weights_only=True, map_location=DEVICE)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    proj.load_state_dict(sd)
    proj.eval()
    return proj


def unwrap_lm_head(lm_head: torch.nn.Module) -> torch.nn.Module:
    """Peel SwitchableLMHead and PEFT base_layer wrappers to find the real Linear."""
    if hasattr(lm_head, "real_lm_head"):
        lm_head = lm_head.real_lm_head
    if hasattr(lm_head, "base_layer"):
        lm_head = lm_head.base_layer
    return lm_head


def get_lm_head_weight(vla) -> Optional[torch.Tensor]:
    """Return the lm_head weight tensor (vocab, hidden), or None if Identity."""
    head = unwrap_lm_head(vla.language_model.lm_head)
    if hasattr(head, "weight"):
        return head.weight
    return None


# ---------- LIBERO RLDS frame iterator ----------

def make_rlds_iterator(dataset_name: str, data_root: str = "data/libero"):
    """Yield raw RLDS samples without going through training-style action tokenization.

    Each yielded dict has:
      - 'primary': np.ndarray (H,W,3) uint8
      - 'wrist':   np.ndarray (H,W,3) uint8
      - 'lang':    str
      - 'actions': np.ndarray (chunk, action_dim) — first row is current action (Δxyz, ...)
      - 'proprio': np.ndarray (proprio_dim,) optional
    """
    from prismatic.vla.datasets import RLDSDataset

    class _PassthroughTransform:
        """Skip action tokenization — just return raw images, lang, actions, proprio."""
        def __call__(self, batch):
            obs = batch["observation"]
            primary = np.asarray(obs["image_primary"][0])
            wrist = None
            for k in obs.keys():
                if "wrist" in k:
                    wrist = np.asarray(obs[k][0])
                    break
            lang = batch["task"]["language_instruction"].decode().lower()
            actions = np.asarray(batch["action"])
            proprio = np.asarray(obs["proprio"]) if "proprio" in obs else None
            return {
                "primary": primary,
                "wrist": wrist,
                "lang": lang,
                "actions": actions,
                "proprio": proprio,
            }

    # Build a minimal RLDSDataset
    # We need an action_tokenizer + base_tokenizer + image_transform for RLDSBatchTransform's signature,
    # but PassthroughTransform overrides __call__. The constructor of RLDSDataset only uses
    # batch_transform via __iter__. So we can pass our PassthroughTransform directly.
    ds = RLDSDataset(
        data_root_dir=Path(REPO_ROOT) / data_root,
        data_mix=dataset_name,
        batch_transform=_PassthroughTransform(),
        resize_resolution=(224, 224),
        shuffle_buffer_size=1000,  # small to keep startup fast
        train=True,
        image_aug=False,
        seed=42,
    )
    return iter(ds)


# ---------- Per-frame inference ----------

def build_inputs(processor, primary: np.ndarray, wrist: np.ndarray, lang: str, num_images: int = 2):
    """Replicate get_vla_action prompt + 2-image input construction."""
    from experiments.robot.openvla_utils import prepare_images_for_vla

    cfg_stub = type("S", (), {"center_crop": True, "num_images_in_input": num_images})()
    images_np = [primary] + ([wrist] if wrist is not None else [])
    images = prepare_images_for_vla(images_np, cfg_stub)  # returns List[PIL.Image]

    prompt = f"In: What action should the robot take to {lang.lower()}?\nOut:"
    inputs = processor(prompt, images[0]).to(DEVICE, dtype=torch.bfloat16)
    if len(images) > 1:
        wrist_inputs_list = [processor(prompt, im).to(DEVICE, dtype=torch.bfloat16) for im in images[1:]]
        all_pixels = [inputs["pixel_values"]] + [wi["pixel_values"] for wi in wrist_inputs_list]
        inputs["pixel_values"] = torch.cat(all_pixels, dim=1)
    return inputs


def _resolve_unnorm_key(vla, requested: str) -> str:
    """Return `requested` if present in the model's norm_stats, else the only available key.

    OOD eval (e.g., spatial ckpt on object frames) needs to use the ckpt's own
    training-dataset stats since action unnormalization is linear and direction
    sign is preserved.
    """
    keys = list(vla.norm_stats.keys()) if hasattr(vla, "norm_stats") else []
    if requested in keys:
        return requested
    if len(keys) >= 1:
        return keys[0]
    return requested  # let the original error surface


@torch.inference_mode()
def run_ap_with_hidden(vla, action_head, proprio_projector, inputs, proprio_vec, unnorm_key, capture_hidden: bool):
    """Runs the standard predict_action and (if requested) intercepts the hidden states.

    Returns (action_chunk, last_hidden_at_pre_action).
    pre_action position = NUM_PATCHES + NUM_PROMPT_TOKENS - 1
    """
    pre_action_hidden_holder = {}

    if capture_hidden:
        # Wrap the inner Qwen2Model.forward to capture the last layer's hidden states
        inner_lm = vla.language_model.model
        orig_fwd = inner_lm.forward

        def patched_fwd(*args, **kwargs):
            kwargs["output_hidden_states"] = True
            out = orig_fwd(*args, **kwargs)
            # out.hidden_states is a tuple per layer; take the last
            pre_action_hidden_holder["last_hidden"] = out.hidden_states[-1]
            return out

        inner_lm.forward = patched_fwd

    try:
        action, _ = vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            proprio=proprio_vec,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=False,
        )
    finally:
        if capture_hidden:
            inner_lm.forward = orig_fwd

    last_hidden = pre_action_hidden_holder.get("last_hidden")
    if last_hidden is not None:
        # Pre-action position
        # NUM_PATCHES = patches_per_image * num_images
        num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        # NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
        num_prompt_tokens = inputs["input_ids"].shape[-1] - 1
        pre_action_pos = num_patches + num_prompt_tokens - 1
        # last_hidden has shape (B, seq_len, D); clone+detach so the tensor is usable
        # outside this inference_mode context
        pre_action_hidden = last_hidden[:, pre_action_pos, :].detach().clone()
    else:
        pre_action_hidden = None

    return action, pre_action_hidden


@torch.no_grad()
def lp_trained_word(pre_action_hidden: torch.Tensor, align_dir_proj, lm_head_weight: torch.Tensor, dir_token_ids: Dict[str, int]) -> Tuple[str, Dict[str, float]]:
    """Apply align_dir_proj → frozen lm_head, restrict to 6 direction tokens, argmax."""
    proj = align_dir_proj(pre_action_hidden)  # (B, D)
    logits = F.linear(proj.float(), lm_head_weight.float())  # (B, vocab)
    ids = [dir_token_ids[w] for w in DIRECTION_WORDS]
    sub = logits[0, ids]  # (6,)
    probs = F.softmax(sub, dim=0).tolist()
    word = DIRECTION_WORDS[int(sub.argmax().item())]
    return word, {w: float(p) for w, p in zip(DIRECTION_WORDS, probs)}


@torch.inference_mode()
def lp_generate_word(vla, processor, primary, wrist, lang, dir_token_ids: Dict[str, int], lm_head_weight: torch.Tensor) -> str:
    """Vision-conditioned LP via manually-built multimodal forward.

    Build [vision_patches; LP_prompt_text] embeddings, run through language_model
    (no action queries inserted), read logits at last text position via the provided
    lm_head_weight, restrict to the 6 direction-token IDs, argmax → word.
    """
    from experiments.robot.openvla_utils import prepare_images_for_vla

    images_np = [primary] + ([wrist] if wrist is not None else [])
    cfg_stub = type("S", (), {"center_crop": True, "num_images_in_input": len(images_np)})()
    images_pil = prepare_images_for_vla(images_np, cfg_stub)

    lp_prompt = (
        f"Answer in one word from the following options: left, right, up, down, "
        f"forward, backward. In which immediate direction should the end effector "
        f"move to take it one step closer to accomplishing the task: {lang}? "
        f"You need to talk, one word."
    )
    inputs = processor(lp_prompt, images_pil[0]).to(DEVICE, dtype=torch.bfloat16)
    if len(images_pil) > 1:
        more = [processor(lp_prompt, im).to(DEVICE, dtype=torch.bfloat16) for im in images_pil[1:]]
        all_pixels = [inputs["pixel_values"]] + [m["pixel_values"] for m in more]
        inputs["pixel_values"] = torch.cat(all_pixels, dim=1)

    # Manually build [vision_patches; text_embeds] and forward through inner LM only.
    text_embeds = vla.get_input_embeddings()(inputs["input_ids"])  # (1, T, D)
    # Process vision (mirror modeling_prismatic._process_vision_features without FiLM/labels)
    proj_patches = vla._process_vision_features(inputs["pixel_values"], language_embeddings=text_embeds, use_film=False)
    combined = torch.cat([proj_patches, text_embeds], dim=1)  # (1, P+T, D)
    attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=combined.device)

    # Call inner Qwen2Model directly (no lm_head, since outer one might be Identity)
    out = vla.language_model.model(
        input_ids=None,
        attention_mask=attn_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=combined,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    last_h = out.last_hidden_state[0, -1, :]  # (D,)
    logits = F.linear(last_h.float().unsqueeze(0), lm_head_weight.float())  # (1, vocab)
    ids = [dir_token_ids[w] for w in DIRECTION_WORDS]
    sub = logits[0, ids]
    word = DIRECTION_WORDS[int(sub.argmax().item())]
    return word


# ---------- Verification ----------

def run_verify(ckpt: str, dataset: str):
    from experiments.robot.openvla_utils import (
        get_action_head, get_processor, get_proprio_projector, get_vla,
    )

    print(f"\n=== VERIFY {ckpt}\n=== dataset {dataset}")
    cfg = _Cfg(ckpt, dataset)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    llm_dim = vla.llm_dim

    # 1. lm_head inspection
    lm_head = vla.language_model.lm_head
    print(f"lm_head outer type: {type(lm_head).__name__}")
    head = unwrap_lm_head(lm_head)
    print(f"lm_head unwrapped type: {type(head).__name__}")
    if hasattr(head, "weight"):
        w = head.weight
        print(f"  weight shape: {tuple(w.shape)}, dtype: {w.dtype}")
        print(f"  mean abs: {w.float().abs().mean().item():.6f}")
    else:
        print("  -> no weight (Identity?). Free-form generation NOT possible without restoring base lm_head.")

    # 2. Prompt token lengths
    sample_tasks = [
        "pick up the alphabet soup and place it in the basket",
        "open the top drawer of the cabinet",
        "put the bowl on top of the cabinet",
    ]
    print("Prompt token lengths:")
    for t in sample_tasks:
        prompt = f"In: What action should the robot take to {t.lower()}?\nOut:"
        ids = processor.tokenizer(prompt, add_special_tokens=True).input_ids
        print(f"  len={len(ids)}, NUM_PROMPT_TOKENS={len(ids)-1}  '{t[:50]}'")

    # 3. Direction tokens
    print("Direction word token IDs:")
    dir_token_ids = {}
    for w in DIRECTION_WORDS:
        ids = processor.tokenizer.encode(w, add_special_tokens=False)
        dir_token_ids[w] = ids[0]
        print(f"  {w:9s} ids={ids} (len={len(ids)})")

    # 4. align_dir_proj inspection
    has_align = find_ckpt_file(ckpt, "align_dir_proj--") is not None
    print(f"align_dir_proj present: {has_align}")
    if has_align:
        proj = load_align_dir_proj(ckpt, llm_dim)
        print(f"  loaded: in={proj.in_features}, out={proj.out_features}, weight mean abs: {proj.weight.float().abs().mean().item():.6f}")

    # 5. One-frame sanity: pre_action_hidden + LP_trained
    print("\n--- one-frame sanity ---")
    action_head = get_action_head(cfg, llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim=8)

    it = make_rlds_iterator(dataset)
    sample = next(it)
    print(f"Got sample: lang='{sample['lang'][:60]}', actions shape={sample['actions'].shape}")
    gt = sample['actions'][0, :3]
    print(f"GT first action [:3] = {gt}, label_set={gt_label_set_from_vec3(gt)}")

    inputs = build_inputs(processor, sample["primary"], sample["wrist"], sample["lang"], num_images=2)
    proprio_vec = sample["proprio"][:8] if sample["proprio"] is not None else None

    lm_head_weight = get_lm_head_weight(vla)
    capture = has_align
    unnorm_key = _resolve_unnorm_key(vla, dataset)
    print(f"resolved unnorm_key: {unnorm_key} (requested: {dataset})")
    action, pre_h = run_ap_with_hidden(vla, action_head, proprio_projector, inputs, proprio_vec, unnorm_key, capture_hidden=capture)
    print(f"AP first action [:3] = {action[0, :3]}, ap_label = {ap_top1_label_from_vec3(action[0, :3])}")

    if has_align and lm_head_weight is not None and pre_h is not None:
        proj = load_align_dir_proj(ckpt, llm_dim)
        word, probs = lp_trained_word(pre_h, proj, lm_head_weight, dir_token_ids)
        print(f"LP_trained word: {word}")
        print("LP_trained probs:")
        for w, p in probs.items():
            print(f"  {w:9s} {p:.4f}")

    if lm_head_weight is not None:
        try:
            word = lp_generate_word(vla, processor, sample["primary"], sample["wrist"], sample["lang"], dir_token_ids, lm_head_weight)
            print(f"LP_gen word: {word}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"LP_gen failed: {e}")

    # Cleanup
    del vla, processor, action_head, proprio_projector
    if has_align:
        del proj
    torch.cuda.empty_cache()


# ---------- Full evaluation ----------

@torch.inference_mode()
def run_full(ckpt: str, dataset: str, num_frames: int, out_path: Optional[str] = None, remap_axes: bool = False, gt_l2_threshold: float = 0.0):
    from experiments.robot.openvla_utils import (
        get_action_head, get_processor, get_proprio_projector, get_vla,
    )

    print(f"\n=== FULL EVAL ckpt={ckpt}\n=== dataset={dataset}, num_frames={num_frames}")
    t_start = time.time()
    cfg = _Cfg(ckpt, dataset)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    llm_dim = vla.llm_dim
    action_head = get_action_head(cfg, llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim=8)
    align_dir_proj = load_align_dir_proj(ckpt, llm_dim)
    has_align = align_dir_proj is not None
    lm_head_weight = get_lm_head_weight(vla)

    dir_token_ids = {w: processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in DIRECTION_WORDS}
    unnorm_key = _resolve_unnorm_key(vla, dataset)
    print(f"has_align={has_align}, lm_head_weight is None={lm_head_weight is None}, unnorm_key={unnorm_key} (requested: {dataset})")

    # Counters: (used = frames with non-empty GT label set AND L2 above threshold)
    n_used = 0
    n_skipped = 0
    n_skipped_l2 = 0  # explicitly skipped by L2 threshold (matches training's align_direction_l2_threshold)
    n_lp_correct = 0
    n_ap_correct = 0
    n_lp_ap_same = 0
    n_lp_ap_both_def = 0
    n11 = n10 = n01 = n00 = 0
    # For LP_gen breakdown (used on baseline as primary; on v7 as secondary)
    n_lp_gen_correct = 0
    n_lp_gen_ap_same = 0
    n_lp_gen_ap_both_def = 0
    lp_gen_words = Counter()

    it = make_rlds_iterator(dataset)
    for frame_idx in range(num_frames):
        try:
            sample = next(it)
        except StopIteration:
            print(f"Dataset exhausted at frame {frame_idx}")
            break

        # Chunk-averaged GT Δxyz (matches v7 alignment-training target:
        # `actions[:, :, :3].mean(dim=1)`). The L2 threshold is applied to this average
        # to match `align_direction_l2_threshold` from training.
        actions_arr = np.asarray(sample["actions"])
        gt = actions_arr[:, :3].mean(axis=0)  # (3,) chunk-averaged Δxyz
        gt_l2 = float(np.linalg.norm(gt))
        if gt_l2_threshold > 0.0 and gt_l2 < gt_l2_threshold:
            n_skipped_l2 += 1
            continue
        gt_set = gt_label_set_from_vec3(gt)
        if not gt_set:
            n_skipped += 1
            continue
        n_used += 1

        inputs = build_inputs(processor, sample["primary"], sample["wrist"], sample["lang"], num_images=2)
        proprio_vec = sample["proprio"][:8] if sample["proprio"] is not None else None

        action, pre_h = run_ap_with_hidden(
            vla, action_head, proprio_projector, inputs, proprio_vec, unnorm_key,
            capture_hidden=has_align,
        )
        # Chunk-averaged predicted Δxyz, mirroring how GT was averaged.
        ap_xyz = np.asarray(action)[:, :3].mean(axis=0)
        ap_lab = ap_top1_label_from_vec3(ap_xyz)
        ap_correct = ap_lab is not None and ap_lab in gt_set
        if ap_correct:
            n_ap_correct += 1

        # Primary LP: trained pathway for v7, generation for baseline
        lp_word_primary: Optional[str] = None
        if has_align and pre_h is not None and lm_head_weight is not None:
            lp_word_primary, _ = lp_trained_word(pre_h, align_dir_proj, lm_head_weight, dir_token_ids)
        # Always also compute LP_gen (secondary for v7, primary for baseline)
        lp_word_gen = ""
        if lm_head_weight is not None:
            try:
                lp_word_gen = lp_generate_word(vla, processor, sample["primary"], sample["wrist"], sample["lang"], dir_token_ids, lm_head_weight)
            except Exception as e:
                print(f"  frame {frame_idx}: LP_gen failed: {e}")
                lp_word_gen = ""
        # Track raw LP_gen distribution BEFORE remap (so we can see what the model actually outputs)
        lp_gen_words[lp_word_gen or "<EMPTY>"] += 1

        # Apply axis-swap remap if requested (for v7 checkpoints trained pre-2026-03-03 fix)
        if remap_axes:
            lp_word_primary = remap_axis_swap(lp_word_primary)
            lp_word_gen_normed = normalize_direction_label(lp_word_gen)
            lp_word_gen_for_scoring = remap_axis_swap(lp_word_gen_normed)
        else:
            lp_word_gen_for_scoring = normalize_direction_label(lp_word_gen)

        # Decide which is the primary LP for headline numbers:
        if has_align:
            lp_primary = lp_word_primary
        else:
            lp_primary = lp_word_gen_for_scoring

        lp_correct = lp_primary is not None and lp_primary in gt_set
        if lp_correct:
            n_lp_correct += 1

        # 2x2 cell
        if lp_correct and ap_correct:
            n11 += 1
        elif lp_correct and not ap_correct:
            n10 += 1
        elif (not lp_correct) and ap_correct:
            n01 += 1
        else:
            n00 += 1

        # LP↔AP same
        if lp_primary is not None and ap_lab is not None:
            n_lp_ap_both_def += 1
            if lp_primary == ap_lab:
                n_lp_ap_same += 1

        # LP_gen-side metrics (always compute for cross-comparison; uses remapped value if remap_axes)
        lp_gen_norm = lp_word_gen_for_scoring
        if lp_gen_norm is not None and lp_gen_norm in gt_set:
            n_lp_gen_correct += 1
        if lp_gen_norm is not None and ap_lab is not None:
            n_lp_gen_ap_both_def += 1
            if lp_gen_norm == ap_lab:
                n_lp_gen_ap_same += 1

        if (frame_idx + 1) % 100 == 0 or frame_idx < 5:
            elapsed = time.time() - t_start
            print(f"  frame {frame_idx+1}/{num_frames}: used={n_used} skipped={n_skipped} "
                  f"LP={lp_primary!r}({'Y' if lp_correct else 'N'}) AP={ap_lab!r}({'Y' if ap_correct else 'N'}) "
                  f"LP_gen={lp_word_gen!r} elapsed={elapsed:.1f}s")

    used = max(1, n_used)
    metrics = {
        "ckpt": ckpt,
        "dataset": dataset,
        "has_align_dir_proj": has_align,
        "primary_lp_pathway": "trained" if has_align else "generate",
        "remap_axes": remap_axes,
        "gt_l2_threshold": gt_l2_threshold,
        "num_skipped_L2": n_skipped_l2,
        "num_frames_seen": n_used + n_skipped,
        "num_used": n_used,
        "num_skipped_GT_eps": n_skipped,
        "LP_ACC_primary": n_lp_correct / used,
        "AP_ACC_top1": n_ap_correct / used,
        "LP_AP_SAME_RATE_OVER_USED_primary": n_lp_ap_same / used,
        "LP_AP_SAME_RATE_OVER_DEFINED_primary": (n_lp_ap_same / n_lp_ap_both_def) if n_lp_ap_both_def > 0 else 0.0,
        "N11_both_correct": n11,
        "N10_LP_correct_AP_wrong": n10,
        "N01_AP_correct_LP_wrong": n01,
        "N00_both_wrong": n00,
        # LP_gen secondary numbers (always reported for fair comparison)
        "LP_ACC_gen": n_lp_gen_correct / used,
        "LP_AP_SAME_RATE_OVER_USED_gen": n_lp_gen_ap_same / used,
        "LP_AP_SAME_RATE_OVER_DEFINED_gen": (n_lp_gen_ap_same / n_lp_gen_ap_both_def) if n_lp_gen_ap_both_def > 0 else 0.0,
        "lp_gen_word_dist": dict(lp_gen_words.most_common(10)),
        "elapsed_s": time.time() - t_start,
    }
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {out_path}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=["libero_spatial_no_noops", "libero_object_no_noops"])
    ap.add_argument("--num-frames", type=int, default=5)
    ap.add_argument("--phase", default="verify", choices=["verify", "full"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--remap-axes", action="store_true",
                    help="Apply X/Y axis-swap remap to LP outputs before scoring (use for v7 checkpoints "
                         "trained before commit 7f02c12 / 2026-03-03 fix)")
    ap.add_argument("--gt-l2-threshold", type=float, default=0.0,
                    help="Skip frames whose chunk-averaged GT Δxyz L2 norm is below this. "
                         "Use 0.15 to match v7 training's align_direction_l2_threshold.")
    args = ap.parse_args()
    if args.phase == "verify":
        run_verify(args.ckpt, args.dataset)
    else:
        run_full(args.ckpt, args.dataset, args.num_frames, args.out, remap_axes=args.remap_axes, gt_l2_threshold=args.gt_l2_threshold)


if __name__ == "__main__":
    main()
