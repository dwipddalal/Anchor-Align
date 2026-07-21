"""
Test language generation ability of the finetuned VLA-Adapter model.

Loads the merged (LoRA baked-in) checkpoint and tests whether the inner
Qwen2.5-0.5B LLM can still generate coherent language output.

Class choices:
- AutoModelForVision2Seq: The checkpoint was saved as OpenVLAForActionPrediction
  (a subclass of PrismaticForConditionalGeneration). This is the outer VLA wrapper
  containing vision_backbone + projector + language_model. We must load it this way
  because the state_dict keys match this architecture.

- AutoTokenizer: Resolves to Qwen2Tokenizer (as specified in tokenizer_config.json).
  Includes the 256 extra action tokens added during VLA training.

- model.language_model: The inner Qwen2ForCausalLM. This is where the actual LM head
  lives (tied with embed_tokens via tie_word_embeddings=True). We call .generate()
  on this directly because the outer model's forward() requires pixel_values, labels,
  action masks, etc. which we don't have for pure text generation.
"""

import os
import sys
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer

# Register custom model classes so AutoModelForVision2Seq can find them
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints/libero-spatial")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Test prompts ──
PROMPTS = [
    # Simple completion
    "The capital of France is",
    # Instruction-style (Qwen2.5 chat format)
    "<|im_start|>user\nWhat is a robot arm used for?<|im_end|>\n<|im_start|>assistant\n",
    # Robotics-related
    "In robotic manipulation, grasping an object requires",
    # General knowledge
    "The three laws of robotics are",
]


def main():
    print("=" * 70)
    print("Language Generation Test — Finetuned VLA-Adapter (10K steps)")
    print("=" * 70)

    # ── Load tokenizer ──
    # AutoTokenizer resolves to Qwen2Tokenizer from tokenizer_config.json
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    print(f"    Tokenizer class: {type(tokenizer).__name__}")
    print(f"    Vocab size: {tokenizer.vocab_size}")
    print(f"    EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"    Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # ── Load full VLA model ──
    # Must use AutoModelForVision2Seq because config.json has
    # architectures: ["OpenVLAForActionPrediction"] and auto_map points here.
    # The state_dict has keys like vision_backbone.*, projector.*, language_model.*
    # so we need the full Prismatic wrapper to load correctly.
    print("\n[2] Loading model (OpenVLAForActionPrediction via registered AutoConfig)...")
    # Load config explicitly using our registered OpenVLAConfig class
    # (avoids HF trying to find configuration_prismatic.py in checkpoint dir)
    config = AutoConfig.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT_DIR,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE)
    model.eval()

    print(f"    Model class: {type(model).__name__}")
    print(f"    Device: {DEVICE}")

    # ── Verify LM head ──
    # Qwen2.5-0.5B uses tied weights: embed_tokens.weight == lm_head.weight
    # get_output_embeddings() delegates to language_model.get_output_embeddings()
    lm_head = model.get_output_embeddings()
    input_embeds = model.get_input_embeddings()
    print(f"\n[3] LM Head check:")
    print(f"    Output embeddings (lm_head): {type(lm_head).__name__}, shape={lm_head.weight.shape}")
    print(f"    Input embeddings: {type(input_embeds).__name__}, shape={input_embeds.weight.shape}")
    print(f"    Weights tied: {lm_head.weight.data_ptr() == input_embeds.weight.data_ptr()}")

    # ── Extract inner LLM for generation ──
    # The outer model's forward() requires pixel_values, labels, action masks.
    # For pure text generation, we use the inner Qwen2ForCausalLM directly,
    # which has the standard HF generate() method and the tied lm_head.
    llm = model.language_model
    print(f"\n[4] Inner LLM: {type(llm).__name__}")
    print(f"    Config: hidden_size={llm.config.hidden_size}, "
          f"num_layers={llm.config.num_hidden_layers}, "
          f"vocab_size={llm.config.vocab_size}")

    # ── Generate from prompts ──
    print("\n" + "=" * 70)
    print("GENERATION RESULTS")
    print("=" * 70)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1} ---")
        print(f"INPUT: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,       # Greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"OUTPUT ({len(generated_tokens)} tokens): {generated_text}")

    # ── Also test with base Qwen2.5-0.5B for comparison ──
    print("\n" + "=" * 70)
    print("COMPARISON: Base Qwen2.5-0.5B (not finetuned)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    base_model.eval()
    base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1} ---")
        print(f"INPUT: {prompt}")

        inputs = base_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=base_tokenizer.pad_token_id,
                eos_token_id=base_tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        generated_text = base_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"OUTPUT ({len(generated_tokens)} tokens): {generated_text}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
