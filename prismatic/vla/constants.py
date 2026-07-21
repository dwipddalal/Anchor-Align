"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Qwen2.5-0.5B token constants
IGNORE_INDEX = -100
# ACTION_TOKEN_BEGIN_IDX = 31743
ACTION_TOKEN_BEGIN_IDX  = 151386
STOP_INDEX = 2  # '</s>'
NUM_TOKENS = 64


# === Optional LLM-family override for ACTION_TOKEN_BEGIN_IDX ===
# When the new Llama-2-7B prism path is in use (--vlm_path prism-dinosiglip-224px+7b),
# the action tokens live at the *Llama* tail-of-vocab range [31744, 32000) instead of
# Qwen's [151387, 151643). The original Qwen-tuned constant above is preserved when
# the Llama path is not detected. Detection: scan sys.argv for the Llama-2-7B vlm id.
def _detect_llm_family() -> str:
    cmd_args = " ".join(sys.argv).lower()
    if "prism-dinosiglip-224px+7b" in cmd_args:
        return "LLAMA2"
    return "QWEN25"


_LLM_FAMILY = _detect_llm_family()
if _LLM_FAMILY == "LLAMA2":
    # Tokenizer vocab=32000, action tokens overwrite the last n_bins=256 of vocab.
    # action_token_begin_idx = 32000 - (256 + 1) = 31743 (matches the commented-out value).
    ACTION_TOKEN_BEGIN_IDX = 31743


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CALVIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

DEFAULT_TASK_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 8,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CLUTTER_TASK_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 64,
    "ACTION_DIM": 8,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

DATASET_CONSTANT_OVERRIDES = {
    "default_task": DEFAULT_TASK_CONSTANTS,
    "clutter_task": CLUTTER_TASK_CONSTANTS,
}


def _get_cli_arg_value(flag: str):
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1].lower()
        if arg.startswith(f"{flag}="):
            return arg.split("=", maxsplit=1)[1].lower()
    return None


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    dataset_name = _get_cli_arg_value("--dataset_name")
    if dataset_name in DATASET_CONSTANT_OVERRIDES:
        return dataset_name

    cmd_args = " ".join(sys.argv).lower()
    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "calvin" in cmd_args:
        return "CALVIN"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "CALVIN":
    constants = CALVIN_CONSTANTS
elif ROBOT_PLATFORM in DATASET_CONSTANT_OVERRIDES:
    constants = DATASET_CONSTANT_OVERRIDES[ROBOT_PLATFORM]

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
