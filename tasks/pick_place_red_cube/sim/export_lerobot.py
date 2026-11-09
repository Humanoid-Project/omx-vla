import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SIM_GRIPPER_OPEN = 0.7


def to_robot_units(q, units, gripper_open_units):
    out = np.empty_like(q, dtype=np.float32)
    if units == "range":
        out[..., :5] = q[..., :5] * (100.0 / np.pi)
        out[..., 5] = gripper_open_units - (SIM_GRIPPER_OPEN - q[..., 5]) * (100.0 / (2 * np.pi))
    else:
        out[..., :5] = np.degrees(q[..., :5])
        out[..., 5] = np.clip(q[..., 5] / SIM_GRIPPER_OPEN, 0.0, 1.0) * 100.0
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert raw sim episodes from collect.py into a LeRobotDataset.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--repo_id", default="local/omx_pick_place_red_cube_sim")
    parser.add_argument("--root", default=None)
    parser.add_argument("--include_failed", action="store_true")
    parser.add_argument("--units", choices=["range", "degrees"], default="range")
    parser.add_argument("--gripper_open_units", type=float, default=60.3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw = Path(args.raw)
    info = json.loads((raw / "info.json").read_text())
    episodes = sorted(p for p in raw.glob("episode_*") if (p / "data.npz").exists())
    usable = [p for p in episodes if args.include_failed or bool(np.load(p / "data.npz")["success"])]
    if not usable:
        raise SystemExit(f"no usable episodes in {raw}")
    first = np.array(Image.open(next((usable[0] / "top").glob("*.jpg"))))
    height, width = first.shape[:2]

    names = [f"{m}.pos" for m in MOTOR_NAMES]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.top": {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]},
    }

    if args.root:
        root = Path(args.root)
    else:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        root = HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=info["fps"],
        features=features,
        root=root,
        robot_type="omx_follower",
        use_videos=True,
    )

    kept, skipped, frames_total = 0, 0, 0
    for ep_dir in episodes:
        data = np.load(ep_dir / "data.npz")
        if not bool(data["success"]) and not args.include_failed:
            skipped += 1
            continue
        state = to_robot_units(data["state"], args.units, args.gripper_open_units)
        action = to_robot_units(data["action"], args.units, args.gripper_open_units)
        for i in range(len(state)):
            dataset.add_frame({
                "observation.state": state[i],
                "action": action[i],
                "observation.images.top": np.array(Image.open(ep_dir / "top" / f"{i:06d}.jpg").convert("RGB")),
                "observation.images.wrist": np.array(Image.open(ep_dir / "wrist" / f"{i:06d}.jpg").convert("RGB")),
                "task": info["task"],
            })
        dataset.save_episode()
        kept += 1
        frames_total += len(state)
        print(f"{ep_dir.name}: {len(state)} frames ({len(state) / info['fps']:.1f} s)", flush=True)

    dataset.finalize()
    print(
        f"dataset {args.repo_id} at {root}: {kept} episodes, {frames_total} frames "
        f"({frames_total / info['fps'] / 60:.1f} min), skipped {skipped} failed",
        flush=True,
    )


if __name__ == "__main__":
    main()
