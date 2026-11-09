import argparse
import json
import math
from pathlib import Path

import numpy as np

ARM = ["joint1", "joint2", "joint3", "joint4", "joint5"]
TRANSIT = {"approach", "reapproach", "move", "home"}
CONTACT = {"descend", "grasp", "lift", "place", "release"}

parser = argparse.ArgumentParser(description="Summarize raw episodes written by collect.py.")
parser.add_argument("--raw", required=True)
args = parser.parse_args()

rows = [np.load(p, allow_pickle=False) for p in sorted(Path(args.raw).glob("episode_*/data.npz"))]
if not rows:
    raise SystemExit(f"no episodes in {args.raw}")

print(f"episodes {len(rows)}")
success = np.array([bool(r["success"]) for r in rows])
print(f"success {success.sum()}/{len(rows)} ({100 * success.mean():.1f} %)")
errors = [str(r["error"]) for r in rows if str(r["error"])]
print(f"ik errors {len(errors)}")
frames = np.array([len(r["state"]) for r in rows])
print(f"frames total {frames.sum()}  per episode min {frames.min()} mean {frames.mean():.0f} max {frames.max()}  ({frames.sum() / 30 / 60:.1f} min at 30 fps)")

knobs = [json.loads(str(r["knobs"])) for r in rows]
modes = {}
for k, ok in zip(knobs, success):
    m = k.get("start_mode", "?")
    modes.setdefault(m, []).append(ok)
for m, oks in sorted(modes.items()):
    print(f"mode {m}: {sum(oks)}/{len(oks)} success")
for name in ["approach_via", "move_via"]:
    print(f"{name} used in {100 * np.mean([name in k for k in knobs]):.0f} % of episodes")
for name in ["grasp_pitch_deg", "hover_pitch_deg", "grasp_z", "hover", "speed"]:
    v = np.array([k[name] for k in knobs if name in k])
    if len(v):
        print(f"{name}: min {v.min():.3f} mean {v.mean():.3f} max {v.max():.3f}")

grasp, gaps = [], {"transit": [], "contact": []}
for r in rows:
    phase = r["phase"].astype(str)
    if len(phase) == 0:
        continue
    hits = np.flatnonzero(phase == "grasp")
    if len(hits):
        grasp.append(r["state"][hits[0], :5])
    gap = np.abs(r["action"][:, :5] - r["state"][:, :5])
    gaps["transit"].append(gap[np.isin(phase, list(TRANSIT))])
    gaps["contact"].append(gap[np.isin(phase, list(CONTACT))])
if grasp:
    g = np.degrees(np.std(grasp, axis=0))
    print("joint std at grasp start (deg): " + "  ".join(f"{n} {v:.1f}" for n, v in zip(ARM, g)))
for name, chunks in gaps.items():
    a = np.concatenate(chunks)
    if len(a):
        print(f"mean |action - state| {name} (deg): " + "  ".join(f"{n} {math.degrees(v):.2f}" for n, v in zip(ARM, a.mean(0))))
print("gripper state min/max (rad): " + " ".join(f"{v:.3f}" for v in (min(r["state"][:, 5].min() for r in rows if len(r["state"])), max(r["state"][:, 5].max() for r in rows if len(r["state"])))))
