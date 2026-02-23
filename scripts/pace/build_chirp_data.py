# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0
"""Utility to convert recorded NPZ files into chirp_data.pt format."""

import argparse
from pathlib import Path

import numpy as np
import torch


def load_npz(path: Path, position_key: str, num_joints: int) -> torch.Tensor:
    data = np.load(path)
    if position_key not in data:
        raise KeyError(f"Missing key '{position_key}' in {path}")
    arr = data[position_key]
    if arr.shape[1] < num_joints:
        raise ValueError(
            f"{path} only has {arr.shape[1]} joints, expected at least {num_joints}"
        )
    return torch.from_numpy(arr[:, :num_joints]).float(), float(data["dt"][0])
    # return torch.from_numpy(arr[:, [0,1,2,3,4,9,10]]).float(), float(data["dt"][0])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert joint/motor NPZ logs into a chirp_data.pt file with "
            "time, dof_pos, and des_dof_pos tensors."
        )
    )
    parser.add_argument("--joint-npz", type=Path, required=True, help="NPZ with measured joint positions.")
    parser.add_argument("--motor-npz", type=Path, required=True, help="NPZ with commanded joint positions.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path for chirp_data.pt (directories are created)."
    )
    parser.add_argument("--num-joints", type=int, default=15, help="Number of joints to keep from the NPZ arrays.")
    parser.add_argument(
        "--current-key", type=str, default="current_positions", help="Key in joint NPZ for measured joint positions."
    )
    parser.add_argument(
        "--desired-key", type=str, default="desired_positions", help="Key in motor NPZ for commanded joint positions."
    )
    args = parser.parse_args()

    dof_pos, joint_dt = load_npz(args.joint_npz, args.current_key, args.num_joints)
    des_dof_pos, motor_dt = load_npz(args.motor_npz, args.desired_key, args.num_joints)

    if not np.isclose(joint_dt, motor_dt):
        raise ValueError(f"Joint dt ({joint_dt}) and motor dt ({motor_dt}) do not match.")
    dt = joint_dt

    num_steps = dof_pos.shape[0]
    time = torch.arange(num_steps, dtype=torch.float32) * dt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"time": time, "dof_pos": dof_pos, "des_dof_pos": des_dof_pos}, args.output)

    print(f"Saved chirp data with {num_steps} steps and {args.num_joints} joints to {args.output}")


if __name__ == "__main__":
    main()
