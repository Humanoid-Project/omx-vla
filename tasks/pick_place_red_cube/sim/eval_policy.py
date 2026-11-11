import argparse
import json
import math
import os
import sys
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a trained LeRobot policy (served by sim/omx/policy_server.py) in the red cube sim scene.")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--episode_s", type=float, default=25.0)
parser.add_argument("--success_hold_s", type=float, default=1.0)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--authkey", default="omx-vla")
parser.add_argument("--gripper_open_units", type=float, default=60.3)
parser.add_argument("--start_gripper_units", type=float, default=49.0)
parser.add_argument("--video_episodes", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from omx.robot_cfg import OMX_ARM_JOINT_NAMES, OMX_JOINT_NAMES

import pick_place_red_cube.sim  # noqa: F401
from pick_place_red_cube.sim import env_cfg as task_cfg
from pick_place_red_cube.sim.mdp import cube_in_square

TASK = "OMX-PickPlaceRedCube-v0"
FPS = task_cfg.FPS
SIM_GRIPPER_OPEN = 0.7
SETTLE_STEPS = 10
RAD_PER_UNIT = math.pi / 100.0
GRIP_RAD_PER_UNIT = 2 * math.pi / 100.0


def to_units(q):
    u = np.empty(6, dtype=np.float32)
    u[:5] = q[:5] / RAD_PER_UNIT
    u[5] = args.gripper_open_units - (SIM_GRIPPER_OPEN - q[5]) / GRIP_RAD_PER_UNIT
    return u


def to_rad(u):
    q = np.empty(6, dtype=np.float32)
    q[:5] = u[:5] * RAD_PER_UNIT
    q[5] = SIM_GRIPPER_OPEN - (args.gripper_open_units - u[5]) * GRIP_RAD_PER_UNIT
    return q


def abort_on_exception(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = abort_on_exception

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.seed = args.seed
cfg.terminations.time_out = None
cfg.terminations.success = None
env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
cube = env.scene["cube"]
joint_ids = [robot.joint_names.index(j) for j in OMX_JOINT_NAMES]
arm_ids = [robot.joint_names.index(j) for j in OMX_ARM_JOINT_NAMES]
home = np.array(task_cfg.HOME_JOINT_POS, dtype=np.float32)
half = (task_cfg.SQUARE_SIZE[0] / 2, task_cfg.SQUARE_SIZE[1] / 2)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
conn = Client(("127.0.0.1", args.port), authkey=args.authkey.encode())
writer = None
if args.video_episodes > 0:
    writer = cv2.VideoWriter(str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 520))


def in_square():
    return bool(cube_in_square(env, square_center=task_cfg.SQUARE_CENTER, square_half_size=half, min_gripper_opening=-1.0)[0])


server_info = None
results = []
for ep in range(args.episodes):
    torch.manual_seed(args.seed * 1_000_000 + ep)
    obs, _ = env.reset()
    start_g = to_rad(np.r_[np.zeros(5), args.start_gripper_units])[5]
    jp = robot.data.default_joint_pos.clone()
    jp[0, arm_ids] = torch.tensor(home, device=env.device, dtype=jp.dtype)
    jp[0, joint_ids[5]] = float(start_g)
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    hold = torch.tensor(np.r_[home, start_g], device=env.device, dtype=torch.float32).unsqueeze(0)
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(hold)
    cube_start = cube.data.root_pos_w[0, :3] - env.scene.env_origins[0]
    cube_start = cube_start.cpu().numpy().round(3).tolist()
    conn.send({"cmd": "reset"})
    server_info = conn.recv()

    t0 = time.time()
    steps = int(args.episode_s * FPS)
    inside_run, success, first_in, ms, max_grip_closed = 0, False, None, [], 99.0
    actions = []
    for i in range(steps):
        p = obs["policy"]
        top = p["top"][0].cpu().numpy()
        wrist = p["wrist"][0].cpu().numpy()
        state = to_units(robot.data.joint_pos[0, joint_ids].cpu().numpy())
        conn.send({
            "cmd": "act",
            "task": task_cfg.TASK_DESCRIPTION,
            "obs": {"observation.state": state, "observation.images.top": top, "observation.images.wrist": wrist},
        })
        reply = conn.recv()
        a_units = np.asarray(reply["action"], dtype=np.float32)
        ms.append(reply["ms"])
        actions.append(a_units)
        max_grip_closed = min(max_grip_closed, float(state[5]))
        obs, *_ = env.step(torch.tensor(to_rad(a_units), device=env.device).unsqueeze(0))
        if writer is not None and ep < args.video_episodes:
            frame = cv2.cvtColor(np.hstack([top, wrist]), cv2.COLOR_RGB2BGR)
            bar = np.full((40, 1280, 3), 30, dtype=np.uint8)
            text = f"ep {ep + 1}  t {i / FPS:5.1f}s  grip {state[5]:5.1f}  {'IN SQUARE' if inside_run else ''}"
            cv2.putText(bar, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(np.vstack([bar, frame]))
        if in_square():
            inside_run += 1
            first_in = first_in if first_in is not None else i / FPS
            if inside_run >= args.success_hold_s * FPS:
                success = True
                break
        else:
            inside_run = 0
    cube_end = (cube.data.root_pos_w[0, :3] - env.scene.env_origins[0]).cpu().numpy().round(3).tolist()
    r = dict(episode=ep, success=success, steps=i + 1, first_in_s=first_in, cube_start=cube_start, cube_end=cube_end,
             min_gripper_units=round(max_grip_closed, 1), infer_ms_median=round(float(np.median(ms)), 1),
             wall_s=round(time.time() - t0, 1))
    results.append(r)
    acts = np.array(actions)
    step = np.abs(np.diff(acts[:, :5], axis=0)).max(1) if len(acts) > 1 else np.zeros(1)
    r.update(arm_step_p99=round(float(np.percentile(step, 99)), 2), arm_step_max=round(float(step.max()), 2),
             arm_step_mean=round(float(step.mean()), 2))
    np.save(out / f"actions_{ep:03d}.npy", acts)
    print(json.dumps(r), flush=True)

conn.send({"cmd": "close"})
conn.recv()
if writer is not None:
    writer.release()
n_ok = sum(r["success"] for r in results)
(out / "results.json").write_text(json.dumps({"success": n_ok, "episodes": len(results), "server": server_info, "results": results}, indent=2))
print(f"SUMMARY success {n_ok}/{len(results)} out {out}", flush=True)
sys.stdout.flush()
os._exit(0)
