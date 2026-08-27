# Source provenance

The `starVLA/` and `deployment/` directories are a source snapshot derived from
<https://github.com/starVLA/starVLA> at commit
`5ce9e2ed59dc94de926c335d36a3681c243a58f7`.

The snapshot includes the local changes used for the real-world xArm7 mug runs:

- LeRobot v3 real-world xArm7 registry and chunk-64 joint-position data path.
- Optional per-trajectory EEF Cartesian-delta cache injection.
- QwenGR00T frozen-teacher MSE anchoring and alignment-v7 losses.
- Multi-GPU trainer fixes, sub-loss logging, and complete checkpoint metadata.

The upstream project and this source snapshot are distributed under the MIT
license in `LICENSE`. Dataset recordings, pretrained models, generated EEF
caches, training logs, and checkpoints are not part of this repository.
