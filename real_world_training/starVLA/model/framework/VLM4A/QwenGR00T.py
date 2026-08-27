# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

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
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # ---- Alignment v7 (optional, opt-in via `framework.enable_alignment_v7`) ----
        # Predict a 6-class direction word from the LAST text-token hidden state of
        # the VLM, supervised by the dominant axis/sign of the chunk-averaged FK
        # Cartesian XYZ delta (provided per-sample by the dataset as `eef_xyz_delta`).
        # Mirrors `compute_alignment_loss_v7` from the prismatic VLA-Adapter trainer.
        #
        # Active iff `enable_alignment_v7=True` AND the batch carries `eef_xyz_delta`.
        # When inactive, behaviour is identical to the unpatched QwenGR00T.
        fw = self.config.framework
        self.enable_alignment_v7 = bool(fw.get("enable_alignment_v7", False))
        self.align_weight = float(fw.get("align_weight", 0.0))
        self.align_direction_l2_threshold = float(fw.get("align_direction_l2_threshold", 0.0))
        if self.enable_alignment_v7:
            H = int(self.qwen_vl_interface.model.config.hidden_size)
            self.align_dir_proj = nn.Linear(H, H, bias=True)
            tok = self.qwen_vl_interface.processor.tokenizer
            # First subword token of each direction word; used as the CE target id.
            direction_names = ["forward", "backward", "left", "right", "up", "down"]
            self.direction_token_ids = {}
            for d in direction_names:
                ids = tok(d, add_special_tokens=False)["input_ids"]
                if len(ids) == 0:
                    raise RuntimeError(f"alignment v7: tokenizer returned empty ids for '{d}'")
                self.direction_token_ids[d] = int(ids[0])
            logger.info(
                f"[align_v7] enabled  weight={self.align_weight}  "
                f"l2_thresh={self.align_direction_l2_threshold}  "
                f"direction_token_ids={self.direction_token_ids}"
            )

        # ---- MSE anchor (optional, opt-in via `framework.enable_mse_anchor`) ----
        # Frozen-teacher MSE: penalise drift of the student VLM's hidden states
        # away from a deep-copied snapshot of the same VLM taken at init time.
        # Mirrors the prismatic `vla-scripts/finetune.py` implementation
        # (per-layer 0.5/sigma**2 * MSE, averaged over `mse_layers`).
        #
        # Active iff `enable_mse_anchor=True` AND `mse_loss_weight > 0`.
        self.enable_mse_anchor = bool(fw.get("enable_mse_anchor", False))
        self.mse_loss_weight = float(fw.get("mse_loss_weight", 0.0))
        self.mse_sigma = float(fw.get("mse_sigma", 1.0))
        self.mse_layers = str(fw.get("mse_layers", "all"))
        self.mse_teacher_vl = None
        if self.enable_mse_anchor and self.mse_loss_weight > 0:
            import copy as _copy
            self.mse_teacher_vl = _copy.deepcopy(self.qwen_vl_interface)
            for _p in self.mse_teacher_vl.parameters():
                _p.requires_grad_(False)
            self.mse_teacher_vl.eval()
            # The teacher only contributes hidden states; drop its lm_head to save
            # ~150M params worth of GPU memory (matches the prismatic recipe).
            try:
                if hasattr(self.mse_teacher_vl, "model") and hasattr(self.mse_teacher_vl.model, "lm_head"):
                    self.mse_teacher_vl.model.lm_head = nn.Identity()
            except Exception:
                pass
            logger.info(
                f"[mse_anchor] enabled  weight={self.mse_loss_weight}  "
                f"sigma={self.mse_sigma}  layers={self.mse_layers}"
            )

    def _compute_alignment_v7_loss(
        self,
        last_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        eef_xyz_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Direction-word CE on the last-text-token hidden state.

        Args:
            last_hidden:     (B, L, H) VLM last-layer hidden states.
            attention_mask:  (B, L) or None — used to find the last non-pad position.
            eef_xyz_deltas:  (B, horizon, 3) precomputed FK Cartesian deltas (m).
        Returns:
            Scalar CE loss (un-weighted).  -100 mask is applied to samples whose
            chunk-averaged |xyz| < align_direction_l2_threshold so quasi-static
            samples don't contribute spurious direction supervision.
        """
        B = eef_xyz_deltas.shape[0]
        device = last_hidden.device
        # Chunk-averaged direction
        xyz = eef_xyz_deltas.float().mean(dim=1)               # (B, 3)
        xyz_l2 = xyz.norm(dim=1)                                # (B,)
        abs_xyz = xyz.abs()
        dominant_axis = abs_xyz.argmax(dim=1)                   # (B,)
        dominant_sign = torch.gather(xyz, 1, dominant_axis.unsqueeze(1)).squeeze(1)  # (B,)
        names = ["forward", "backward", "left", "right", "up", "down"]
        axis_to_pos = {0: 0, 1: 2, 2: 4}
        dir_labels = torch.full((B,), -100, dtype=torch.long, device=device)
        for i in range(B):
            if xyz_l2[i].item() < self.align_direction_l2_threshold:
                continue
            axis = int(dominant_axis[i].item())
            base_idx = axis_to_pos[axis]
            idx = base_idx if dominant_sign[i].item() >= 0 else (base_idx + 1)
            dir_labels[i] = self.direction_token_ids[names[idx]]
        # Last non-pad text-token hidden state per sample (the position that has
        # attended to ALL vision+text tokens — analogous to next-token prediction).
        if attention_mask is not None:
            last_pos = attention_mask.long().sum(dim=1) - 1     # (B,)
            last_pos = last_pos.clamp(min=0, max=last_hidden.shape[1] - 1)
            pre_action_hidden = last_hidden[torch.arange(B, device=device), last_pos]   # (B, H)
        else:
            pre_action_hidden = last_hidden[:, -1, :]            # (B, H)
        # Project, then predict through the (frozen-weights) lm_head.
        proj = self.align_dir_proj(pre_action_hidden.float())   # (B, H)
        lm_head = self.qwen_vl_interface.model.lm_head
        # Use weights without tracking grads through lm_head (matches the
        # prismatic `_frozen_lm_head_forward` semantics).
        w = lm_head.weight.detach().to(proj.dtype)
        b = lm_head.bias.detach().to(proj.dtype) if getattr(lm_head, "bias", None) is not None else None
        dir_logits = F.linear(proj, w, b)                       # (B, vocab_size)
        return F.cross_entropy(dir_logits, dir_labels, ignore_index=-100)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

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

        # MSE anchor: teacher VLM forward (frozen). Done OUTSIDE the action-model
        # autocast block so teacher activations are released before the action
        # forward, but the hidden states needed for MSE stay live until the MSE
        # loss is computed. `torch.no_grad` (not inference_mode) so the detached
        # teacher tensors interop cleanly with student tensors in MSE.
        teacher_hidden_states = None
        if self.enable_mse_anchor and self.mse_loss_weight > 0 and self.mse_teacher_vl is not None:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher_outputs = self.mse_teacher_vl(
                        **qwen_inputs,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
            if self.mse_layers == "all":
                teacher_hidden_states = [h.detach() for h in teacher_outputs.hidden_states[1:]]
            else:
                teacher_hidden_states = [teacher_outputs.hidden_states[-1].detach()]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )  # (B, chunk_len, action_dim)

        # ---- Alignment v7: direction-word CE on last text-token hidden state ----
        # Active iff (a) framework.enable_alignment_v7=True, (b) every sample in the
        # batch carries `eef_xyz_delta` (attached by the dataset when its `data_cfg`
        # has `eef_xyz_delta_cache`), and (c) align_weight > 0.  The raw CE is
        # returned alongside action_loss for trainer logging; the WEIGHTED term is
        # folded into action_loss (which the trainer treats as the total loss).
        output_dict = {"action_loss": action_loss}
        eef_present = self.enable_alignment_v7 and self.align_weight > 0.0 and all(
            isinstance(ex.get("eef_xyz_delta", None), np.ndarray) for ex in examples
        )
        if eef_present:
            eef_arr = np.stack([ex["eef_xyz_delta"] for ex in examples], axis=0).astype(np.float32)  # (B, H, 3)
            eef_t = torch.from_numpy(eef_arr).to(last_hidden.device)
            with torch.autocast("cuda", dtype=torch.float32):
                align_loss_raw = self._compute_alignment_v7_loss(
                    last_hidden=last_hidden,
                    attention_mask=qwen_inputs.get("attention_mask"),
                    eef_xyz_deltas=eef_t,
                )
            if torch.isfinite(align_loss_raw):
                output_dict["action_loss"] = action_loss + self.align_weight * align_loss_raw.to(action_loss.dtype)
                output_dict["align_loss_raw"] = align_loss_raw.detach()

        # ---- MSE anchor: per-layer (0.5/sigma**2) * MSE(student_hidden, teacher_hidden) ----
        # Folds `mse_loss_weight * mse_loss_raw` into action_loss; returns the unweighted
        # `mse_loss_raw` for trainer logging (loss/mse_raw, loss/mse_weighted in metrics).
        if teacher_hidden_states is not None:
            if self.mse_layers == "all":
                student_layer_hidden = list(qwenvl_outputs.hidden_states[1:])  # skip embedding layer
            else:
                student_layer_hidden = [qwenvl_outputs.hidden_states[-1]]
            assert len(student_layer_hidden) == len(teacher_hidden_states), (
                f"MSE layer count mismatch: student={len(student_layer_hidden)} teacher={len(teacher_hidden_states)}"
            )
            inv_two_sigma2 = 0.5 / (self.mse_sigma ** 2)
            with torch.autocast("cuda", dtype=torch.float32):
                layer_mse_losses = []
                for s, t in zip(student_layer_hidden, teacher_hidden_states):
                    layer_mse_losses.append(inv_two_sigma2 * F.mse_loss(s.float(), t.float()))
                mse_loss_raw = sum(layer_mse_losses) / len(layer_mse_losses)
            if torch.isfinite(mse_loss_raw):
                output_dict["action_loss"] = output_dict["action_loss"] + self.mse_loss_weight * mse_loss_raw.to(action_loss.dtype)
                output_dict["mse_loss_raw"] = mse_loss_raw.detach()

        return output_dict

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
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

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

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

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


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

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
