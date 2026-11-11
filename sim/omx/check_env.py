import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render and sanity-check an OMX task environment.")
parser.add_argument("--task", default="OMX-PickPlaceRedCube-v0")
parser.add_argument("--task_module", default="pick_place_red_cube.sim")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import importlib

import cv2
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from omx.kinematics import inverse_top_down
from omx.robot_cfg import OMX_GRIPPER_CLOSED, OMX_GRIPPER_OPEN, OMX_HOME_JOINT_POS

importlib.import_module(args.task_module)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env_cfg.terminations.time_out = None
env_cfg.terminations.success = None
env = gym.make(args.task, cfg=env_cfg).unwrapped
obs, _ = env.reset()
robot = env.scene["robot"]
cube = env.scene["cube"]
origin = env.scene.env_origins[0]
print("joints:", robot.joint_names, flush=True)
print("bodies:", robot.body_names, flush=True)
ee_id = robot.body_names.index("end_effector_link")
f1, f2 = robot.body_names.index("link6"), robot.body_names.index("link7")
home = [OMX_HOME_JOINT_POS[j] for j in ["joint1", "joint2", "joint3", "joint4", "joint5"]]


def step(arm, grip, n):
    global obs
    act = torch.tensor([list(arm) + [grip]], device=env.device, dtype=torch.float32)
    for _ in range(n):
        obs, *_ = env.step(act)


def save(name):
    for cam in ("top", "wrist"):
        img = obs["policy"][cam][0].cpu().numpy()
        cv2.imwrite(str(out / f"{name}_{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


step(home, OMX_GRIPPER_OPEN, 60)
save("home")
print("cube start:", [round(v, 4) for v in (cube.data.root_pos_w[0] - origin).tolist()], flush=True)
for g in (OMX_GRIPPER_OPEN, OMX_GRIPPER_CLOSED):
    step(home, g, 45)
    d = (robot.data.body_pos_w[0, f1] - robot.data.body_pos_w[0, f2]).norm().item()
    print(f"gripper target {g}: joint {robot.data.joint_pos[0, -2:].tolist()} finger-link distance {d:.4f}", flush=True)
    save(f"gripper_{g}")

task_cfg = importlib.import_module(f"{args.task_module}.env_cfg")
task_mdp = importlib.import_module(f"{args.task_module}.mdp")
sq_x, sq_y = task_cfg.SQUARE_CENTER
sq_half = (task_cfg.SQUARE_SIZE[0] / 2, task_cfg.SQUARE_SIZE[1] / 2)

for target in [(0.20, 0.0, 0.08), (0.15, 0.10, 0.06), (0.24, -0.10, 0.05), (sq_x, sq_y, 0.05)]:
    q = inverse_top_down(*target)
    step(list(q) + [0.0], OMX_GRIPPER_OPEN, 60)
    ee = (robot.data.body_pos_w[0, ee_id] - origin).tolist()
    err = [round((ee[i] - target[i]) * 1000, 1) for i in range(3)]
    tracking = (robot.data.joint_pos[0, :5] - torch.tensor(list(q) + [0.0], device=env.device)).abs().max().item()
    print(f"IK target {target}: sim ee {[round(v, 4) for v in ee]} error mm {err} joint tracking err {tracking:.4f}", flush=True)
save("reach")

step(home, OMX_GRIPPER_OPEN, 60)
state = cube.data.default_root_state.clone()
state[:, :3] = torch.tensor([[sq_x, sq_y, 0.016]], device=env.device) + origin
cube.write_root_state_to_sim(state)
step(home, OMX_GRIPPER_OPEN, 30)
ok = task_mdp.cube_in_square(env, square_center=(sq_x, sq_y), square_half_size=sq_half)
print("success when cube placed in square:", bool(ok[0]), flush=True)
save("cube_in_square")

state[:, :3] = torch.tensor([[sq_x, sq_y + sq_half[1] + 0.05, 0.016]], device=env.device) + origin
cube.write_root_state_to_sim(state)
step(home, OMX_GRIPPER_OPEN, 30)
ok = task_mdp.cube_in_square(env, square_center=(sq_x, sq_y), square_half_size=sq_half)
print("success when cube outside:", bool(ok[0]), flush=True)
print("CHECK_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
