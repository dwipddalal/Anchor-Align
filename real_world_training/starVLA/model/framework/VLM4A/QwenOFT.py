# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOFT
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOFTDefaultConfig:
    """QwenOFT framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenOFT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (MLP regression over action special tokens) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "MLP",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # Hidden dim for the action MLP (auto-set from VLM hidden_size at runtime)
            "action_hidden_dim": 2560,
            # How many future steps to predict
            "future_action_window_size": 8,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Action special token injected into the VLM sequence
      - MLP regression head over action token hidden states (L1 loss)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenOFTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align action_hidden_dim to VLM hidden_size at runtime
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_token = "🔍"  # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        # L1 loss
        self.l1_loss = nn.L1Loss()

        # ---- MSE anchoring (distillation from frozen pre-trained VLM) ----
        # Off by default; enabled by setting `framework.enable_mse_anchor: true`
        # in the YAML, with optional `framework.mse_loss_weight` (default 0.1),
        # `framework.mse_sigma` (default 1.0), `framework.mse_layers`
        # ('all' | 'last', default 'all').
        self.enable_mse_anchor = bool(self.config.framework.get("enable_mse_anchor", False))
        if self.enable_mse_anchor:
            self.mse_loss_weight = float(self.config.framework.get("mse_loss_weight", 0.1))
            self.mse_sigma = float(self.config.framework.get("mse_sigma", 1.0))
            self.mse_layers = str(self.config.framework.get("mse_layers", "all"))
            logger.info(
                f"[QwenOFT] MSE anchor ON: weight={self.mse_loss_weight}, "
                f"sigma={self.mse_sigma}, layers={self.mse_layers}. Building frozen teacher VLM..."
            )
            teacher = get_vlm_model(config=self.config)
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
            # CRITICAL: store inside a Python list (not as a direct attribute) to
            # bypass nn.Module's child registration. If we did
            # `self.teacher_qwen_vl = teacher`, its (frozen) params would appear
            # in self.parameters() and DeepSpeed ZeRO-2 would put them in the
            # "base" optimizer group — that group then becomes empty after
            # DeepSpeed filters out non-trainable params, breaking
            # _flatten_dense_tensors_aligned during ZeRO-2 init.
            # Lazily moved to the student's device on the first forward() call.
            self._teacher_holder = [teacher]
        else:
            self._teacher_holder = [None]

        # ---- Alignment v7: pre-action bottleneck direction prediction ----
        # Hidden state at position JUST BEFORE the first 🔍 action token has
        # attended to all vision + text via self-attention. Project via a learned
        # nn.Linear(D, D), multiply by lm_head.weight to get vocab logits, and
        # cross-entropy on direction-word target (forward/backward/left/right/up/down)
        # derived from the dominant axis sign of the XYZ action delta.
        self.enable_alignment_v7 = bool(self.config.framework.get("enable_alignment_v7", False))
        if self.enable_alignment_v7:
            self.align_weight = float(self.config.framework.get("align_weight", 0.02))
            self.align_direction_l2_threshold = float(self.config.framework.get("align_direction_l2_threshold", 0.15))
            hidden_dim = self.qwen_vl_interface.model.config.hidden_size
            self.align_dir_proj = nn.Linear(hidden_dim, hidden_dim)

            # Pre-tokenize 6 direction words to single Qwen tokens.
            direction_words = ["forward", "backward", "left", "right", "up", "down"]
            tokenizer = self.qwen_vl_interface.processor.tokenizer
            self.direction_token_ids_list = []
            for w in direction_words:
                # Leading-space convention common for BPE tokenizers
                ids = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
                if len(ids) != 1:
                    ids = tokenizer(w, add_special_tokens=False)["input_ids"]
                assert len(ids) == 1, f"Direction word '{w}' tokenizes to {ids}"
                self.direction_token_ids_list.append(ids[0])
            logger.info(
                f"[QwenOFT] Alignment v7 ON: weight={self.align_weight}, "
                f"direction_token_ids={dict(zip(direction_words, self.direction_token_ids_list))}"
            )
        else:
            self.align_dir_proj = None

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Optionally prepend discretised proprioceptive state tokens to each instruction (π₀.5 style).
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        if not self.enable_mse_anchor:
            # ===== Original anchor-disabled path =====
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # last_hidden_state: [B, seq_len, H]
                last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            # Step 4: Action Expert Forward and Loss
            with torch.autocast("cuda", dtype=torch.float32):
                # Extract action token embeddings as action prediction queries
                input_ids = qwen_inputs.get("input_ids", None)
                action_queries = self._gather_action_token_embeddings(
                    last_hidden, input_ids, action_token_id=self.action_token_id
                )  # [B, chunk_len, H]
                pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

                # Label alignment: take the last chunk_len segment
                actions = torch.tensor(
                    np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

                # Compute L1 loss
                action_loss = self.l1_loss(pred_actions, actions_target)

            return {"action_loss": action_loss}
        else:
            # ===== MSE anchor path: student forward + frozen teacher forward =====
            # Student forward (output all layer hidden states for MSE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]

            # Action prediction (same as the anchor-disabled path)
            with torch.autocast("cuda", dtype=torch.float32):
                input_ids = qwen_inputs.get("input_ids", None)
                action_queries = self._gather_action_token_embeddings(
                    last_hidden, input_ids, action_token_id=self.action_token_id
                )
                pred_actions = self.action_model.predict_action(action_queries)
                actions = torch.tensor(
                    np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
                )
                actions_target = actions[:, -self.action_horizon :, :]
                action_loss = self.l1_loss(pred_actions, actions_target)

            # Lazy-move teacher to the student's device on first forward call.
            # (Teacher is hidden from nn.Module children, so .to(device) on the
            # framework / accelerator.prepare() does NOT move it automatically.)
            teacher = self._teacher_holder[0]
            student_device = next(self.qwen_vl_interface.parameters()).device
            if next(teacher.parameters()).device != student_device:
                teacher = teacher.to(student_device)
                self._teacher_holder[0] = teacher

            # Teacher forward in no_grad — no activation storage, no backward.
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher_outputs = teacher(
                        **qwen_inputs,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )

            # Mask: True at positions to keep for MSE (everything except the action tokens).
            # input_ids shape: (B, L). Action token positions excluded so MSE only anchors
            # vision + language understanding, not the action-prediction queries.
            non_action_mask = (input_ids != self.action_token_id)  # (B, L) bool

            # Pick layers per mse_layers config.
            # qwenvl_outputs.hidden_states is a tuple of length (num_hidden_layers + 1):
            # index 0 = embeddings, indices 1..N = transformer layer outputs.
            if self.mse_layers == "all":
                student_layers = list(qwenvl_outputs.hidden_states[1:])
                teacher_layers = list(teacher_outputs.hidden_states[1:])
            else:  # "last"
                student_layers = [qwenvl_outputs.hidden_states[-1]]
                teacher_layers = [teacher_outputs.hidden_states[-1]]

            sigma = self.mse_sigma
            layer_mse_losses = []
            for student_h, teacher_h in zip(student_layers, teacher_layers):
                # student_h: (B, L, H). Mask via boolean indexing → (N_keep, H).
                student_vl = student_h[non_action_mask]
                teacher_vl = teacher_h[non_action_mask]
                # Preserve the original 0.5/sigma^2 scaling.
                layer_loss = (0.5 / (sigma ** 2)) * F.mse_loss(student_vl.float(), teacher_vl.float())
                layer_mse_losses.append(layer_loss)
            mse_loss = sum(layer_mse_losses) / len(layer_mse_losses)
            total_loss = action_loss + self.mse_loss_weight * mse_loss

            # ----- Alignment v7 (optional, additive) ---------------------------
            align_loss_raw = None
            if self.enable_alignment_v7:
                # Position of the LAST text token before the chunk of 🔍 placeholders.
                # input_ids shape (B, L); find first 🔍 position per sample, then -1.
                first_action_mask = (input_ids == self.action_token_id).int()
                first_action_pos = first_action_mask.argmax(dim=1)  # (B,)
                pre_action_pos = (first_action_pos - 1).clamp(min=0)  # (B,)
                # Gather pre-action hidden from the LAST transformer layer's output.
                B, L, H = last_hidden.shape
                idx = pre_action_pos.view(-1, 1, 1).expand(-1, 1, H)
                pre_action_hidden = last_hidden.gather(dim=1, index=idx).squeeze(1)  # (B, H)
                align_loss_raw = self._compute_alignment_loss_v7(pre_action_hidden, actions_target)
                total_loss = total_loss + self.align_weight * align_loss_raw

            out = {
                "action_loss": total_loss,
                "mse_loss_raw": mse_loss.detach(),
                "action_l1_loss_raw": action_loss.detach(),
            }
            if align_loss_raw is not None:
                out["align_loss_raw"] = align_loss_raw.detach()
            return out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Optionally prepend discretised proprioceptive state tokens to each instruction (π₀.5 style).
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> torch.Tensor:
        """
        Vectorized batch extraction of action token embeddings:
          - No per-sample for loop
          - Select the last chunk_len action placeholder tokens from each sample
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int or List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Support multiple ids (e.g., multiple variants)
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin requires PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.chunk_len}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        # Position indices
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1

        # Take the last chunk_len positions (higher indices = later in sequence)
        # Note: count sufficiency already verified, so -1 won't be incorrectly selected
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len] unsorted
        # Sort in temporal order
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

    # Discretised state → instruction prefix (π₀.5 style); shared with QwenPI_v3.
    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)

    def _compute_alignment_loss_v7(self, pre_action_hidden, actions_target):
        """Alignment v7: pre-action bottleneck direction prediction.

        Maps the hidden state at the LAST text position (right before the 🔍
        placeholders) to a direction word (forward/backward/left/right/up/down)
        via `align_dir_proj` + lm_head, with cross-entropy supervision.

        actions_target: (B, action_horizon, action_dim). Direction label is
        derived from the dominant axis (x/y/z) sign of the chunk-averaged XYZ
        delta. Samples with |xyz| < align_direction_l2_threshold are ignored.
        """
        B = pre_action_hidden.shape[0]
        device = pre_action_hidden.device

        # Chunk-averaged XYZ (first 3 dims of action: dx, dy, dz).
        xyz = actions_target[:, :, :3].float().mean(dim=1)  # (B, 3)
        xyz_l2 = xyz.norm(dim=1)
        abs_xyz = xyz.abs()
        dominant_axis = abs_xyz.argmax(dim=1)  # (B,) in {0,1,2}
        dominant_sign = torch.gather(xyz, 1, dominant_axis.unsqueeze(1)).squeeze(1)  # (B,)

        # LIBERO/robosuite convention: axis 0 -> forward(+)/backward(-),
        # axis 1 -> left(+)/right(-), axis 2 -> up(+)/down(-).
        # direction_token_ids_list order = [forward, backward, left, right, up, down]
        axis_to_pos = {0: 0, 1: 2, 2: 4}
        dir_labels = torch.full((B,), -100, dtype=torch.long, device=device)
        for i in range(B):
            if xyz_l2[i] < self.align_direction_l2_threshold:
                continue
            axis = int(dominant_axis[i].item())
            base_idx = axis_to_pos[axis]
            if dominant_sign[i] >= 0:
                dir_labels[i] = self.direction_token_ids_list[base_idx]      # forward/left/up
            else:
                dir_labels[i] = self.direction_token_ids_list[base_idx + 1]  # backward/right/down

        # Project pre-action hidden and read out via lm_head's weight to get
        # vocab logits. Use the LIVE lm_head (we full-finetune the VLM so it
        # is being trained, not frozen — same signal direction as v7).
        lm_head = self.qwen_vl_interface.model.lm_head
        projected = self.align_dir_proj(pre_action_hidden.to(self.align_dir_proj.weight.dtype))
        dir_logits = F.linear(projected, lm_head.weight, lm_head.bias)  # (B, vocab)
        return F.cross_entropy(dir_logits.float(), dir_labels, ignore_index=-100)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
