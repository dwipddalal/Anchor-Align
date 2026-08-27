#!/usr/bin/env python3
"""Build the alignment-v7 EEF-delta cache from a LeRobot v3 xArm7 dataset."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def state_fingerprint(states: np.ndarray) -> str:
    return hashlib.sha256(states[:10].astype(np.float32).tobytes()).hexdigest()[:16]


def build_fk(urdf_path: Path, eef_frame: str):
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError("Pinocchio is required; install it with `pip install pin`") from exc

    model = pin.buildModelFromUrdf(str(urdf_path))
    if not model.existFrame(eef_frame):
        available = [frame.name for frame in model.frames]
        raise ValueError(f"EEF frame {eef_frame!r} not found in URDF; available frames: {available}")
    frame_id = model.getFrameId(eef_frame)
    data = model.createData()
    if model.nq < 7:
        raise ValueError(f"URDF has only {model.nq} position variables; xArm7 requires at least 7")

    def fk(joints_radians: np.ndarray) -> np.ndarray:
        qpos = np.zeros(model.nq, dtype=np.float64)
        qpos[:7] = joints_radians[:7]
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        return np.asarray(data.oMf[frame_id].translation, dtype=np.float64).copy()

    print(f"FK model: nq={model.nq}, EEF frame={eef_frame} (id={frame_id})")
    return fk


def load_frames(dataset_dir: Path) -> pd.DataFrame:
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no trajectory parquet files under {dataset_dir / 'data'}")
    columns = ["episode_index", "frame_index", "observation.state"]
    frames = pd.concat([pd.read_parquet(path, columns=columns) for path in parquet_files], ignore_index=True)
    return frames.sort_values(["episode_index", "frame_index"], kind="stable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eef-frame", default="link_eef")
    parser.add_argument("--joint-unit", choices=("degrees", "radians"), default="degrees")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    frames = load_frames(dataset_dir)
    fk = build_fk(urdf_path, args.eef_frame)
    cache: dict[str, np.ndarray] = {}
    all_deltas: list[np.ndarray] = []

    grouped = frames.groupby("episode_index", sort=True)
    for episode_number, (episode_id, episode) in enumerate(grouped, start=1):
        states = np.stack([np.asarray(value, dtype=np.float32) for value in episode["observation.state"]])
        if states.shape[0] < 10 or states.shape[1] < 7:
            raise ValueError(f"episode {episode_id} has invalid state shape {states.shape}")
        joints = states[:, :7].astype(np.float64)
        if args.joint_unit == "degrees":
            joints = np.deg2rad(joints)

        positions = np.stack([fk(joint_row) for joint_row in joints])
        deltas = np.zeros((positions.shape[0], 3), dtype=np.float32)
        deltas[:-1] = np.diff(positions, axis=0).astype(np.float32)
        fingerprint = state_fingerprint(states)
        if fingerprint in cache:
            raise ValueError(f"duplicate state fingerprint {fingerprint} at episode {episode_id}")
        cache[fingerprint] = deltas
        all_deltas.append(deltas)
        if episode_number % 10 == 0:
            print(f"Processed {episode_number}/{grouped.ngroups} episodes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    combined = np.concatenate(all_deltas, axis=0)
    norms = np.linalg.norm(combined, axis=1)
    print(f"Saved {len(cache)} episodes and {len(combined)} frame deltas to {output_path}")
    print(
        "Delta L2 stats (m): "
        f"mean={norms.mean():.6f}, p50={np.percentile(norms, 50):.6f}, "
        f"p75={np.percentile(norms, 75):.6f}, max={norms.max():.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
