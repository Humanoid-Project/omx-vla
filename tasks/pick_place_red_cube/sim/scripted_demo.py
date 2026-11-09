import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted pick-and-place episodes with camera recording.")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--video", required=True)
parser.add_argument("--data_dir", default=None)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg
from omx.kinematics import inverse_top_down
from omx.robot_cfg import OMX_ARM_JOINT_NAMES, OMX_GRIPPER_CLOSED, OMX_GRIPPER_OPEN, OMX_HOME_JOINT_POS

import pick_place_red_cube.sim  # noqa: F401
from pick_place_red_cube.sim import env_cfg as task_cfg
from pick_place_red_cube.sim.mdp import cube_in_square

TASK = "OMX-PickPlaceRedCube-v0"
HOVER_Z = 0.09
GRASP_Z = 0.008
PLACE_Z = 0.014

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.seed = args.seed
cfg.terminations.time_out = None
cfg.terminations.success = None
env = gym.make(TASK, cfg=cfg).unwrapped
home = np.array([OMX_HOME_JOINT_POS[j] for j in OMX_ARM_JOINT_NAMES])

video_path = Path(args.video)
video_path.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), task_cfg.FPS, (1280, 520))


def wrap_quarter(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def arm_q(x, y, z, cube_yaw):
    q1, q2, q3, q4 = inverse_top_down(x, y, z)
    return np.array([q1, q2, q3, q4, wrap_quarter(q1 - cube_yaw)])


def render(obs, episode, phase, success):
    top = obs["policy"]["top"][0].cpu().numpy()
    wrist = obs["policy"]["wrist"][0].cpu().numpy()
    frame = cv2.cvtColor(np.hstack([top, wrist]), cv2.COLOR_RGB2BGR)
    bar = np.full((40, 1280, 3), 30, dtype=np.uint8)
    text = f"Episode {episode}/{args.episodes}  |  {phase}  |  task: \"{task_cfg.TASK_DESCRIPTION}\""
    cv2.putText(bar, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if success:
        cv2.putText(bar, "SUCCESS", (1170, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "top", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, "wrist", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    writer.write(np.vstack([bar, frame]))


results = []
for ep in range(1, args.episodes + 1):
    obs, _ = env.reset()
    cube = env.scene["cube"]
    origin = env.scene.env_origins[0]
    cube_xyz = (cube.data.root_pos_w[0] - origin).tolist()
    cube_yaw = euler_xyz_from_quat(cube.data.root_quat_w)[2][0].item()
    sq = task_cfg.SQUARE_CENTER
    plan = [
        ("approach", arm_q(cube_xyz[0], cube_xyz[1], HOVER_Z, cube_yaw), OMX_GRIPPER_OPEN, 1.5),
        ("descend", arm_q(cube_xyz[0], cube_xyz[1], GRASP_Z, cube_yaw), OMX_GRIPPER_OPEN, 1.0),
        ("grasp", arm_q(cube_xyz[0], cube_xyz[1], GRASP_Z, cube_yaw), OMX_GRIPPER_CLOSED, 0.8),
        ("lift", arm_q(cube_xyz[0], cube_xyz[1], HOVER_Z, cube_yaw), OMX_GRIPPER_CLOSED, 1.0),
        ("move", arm_q(sq[0], sq[1], HOVER_Z, cube_yaw), OMX_GRIPPER_CLOSED, 1.5),
        ("place", arm_q(sq[0], sq[1], PLACE_Z, cube_yaw), OMX_GRIPPER_CLOSED, 1.0),
        ("release", arm_q(sq[0], sq[1], PLACE_Z, cube_yaw), OMX_GRIPPER_OPEN, 0.6),
        ("retreat", arm_q(sq[0], sq[1], HOVER_Z, cube_yaw), OMX_GRIPPER_OPEN, 0.8),
        ("home", home, OMX_GRIPPER_OPEN, 1.5),
    ]
    q_prev, g_prev = home.copy(), OMX_GRIPPER_OPEN
    states, actions = [], []
    success = False
    for phase, q_goal, g_goal, dur in plan:
        n = max(1, int(dur * task_cfg.FPS))
        for i in range(n):
            s = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
            q = q_prev + s * (q_goal - q_prev)
            g = g_prev + s * (g_goal - g_prev)
            act = torch.tensor([list(q) + [g]], device=env.device, dtype=torch.float32)
            obs, *_ = env.step(act)
            states.append(obs["policy"]["joint_pos"][0].cpu().numpy())
            actions.append(act[0].cpu().numpy())
            success = bool(cube_in_square(env, square_center=sq, square_half_size=task_cfg.SQUARE_SIZE / 2)[0])
            render(obs, ep, phase, success)
        q_prev, g_prev = q_goal, g_goal
    for _ in range(int(0.5 * task_cfg.FPS)):
        render(obs, ep, "done", success)
    final_cube = (cube.data.root_pos_w[0] - origin).tolist()
    results.append(success)
    print(f"episode {ep}: cube start {[round(v, 3) for v in cube_xyz]} yaw {math.degrees(cube_yaw):.1f} deg -> end {[round(v, 3) for v in final_cube]} success {success} frames {len(states)}", flush=True)
    if args.data_dir:
        d = Path(args.data_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"episode_{ep:03d}.npz", state=np.array(states), action=np.array(actions), success=success)

writer.release()
print(f"SUMMARY success {sum(results)}/{len(results)} video {video_path}", flush=True)
sys.stdout.flush()
os._exit(0)
