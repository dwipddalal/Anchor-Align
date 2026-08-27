"""Real-world xArm7 (joint-position) — data config, embodiment tag, mixtures.

Reusable single config for all real-world xArm7 datasets with absolute joint
positions (7 joints + 1 gripper). Mirrors VLA-Adapter's `default_task` pattern:
one DataConfig, many datasets that share the same robot/format.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class XArm7JointPosDataConfig:
    """xArm7 with absolute joint positions (7 joints) + 1 gripper position."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_wrist"]
    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.joints", "action.gripper"]
    action_key_dims = {"action.joints": 7, "action.gripper": 1}
    state_key_dims = {"state.joints": 7, "state.gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))   # 8-step action chunk (matches VLA-Adapter)
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.joints": "min_max", "state.gripper": "min_max"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.joints": "min_max", "action.gripper": "min_max"},
            ),
        ])


class XArm7JointPosChunk64DataConfig(XArm7JointPosDataConfig):
    """xArm7 joint-pos with a 64-step action chunk (matches VLA-Adapter clutter)."""

    action_indices = list(range(64))


ROBOT_TYPE_CONFIG_MAP = {
    "xarm7_joint_pos": XArm7JointPosDataConfig(),
    "xarm7_joint_pos_chunk64": XArm7JointPosChunk64DataConfig(),
}


ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # auto-derived from classvar
}


DATASET_NAMED_MIXTURES = {
    "mug_xarm7": [
        ("mug_xarm7_lerobot", 1.0, "xarm7_joint_pos_chunk64"),
    ],
}
