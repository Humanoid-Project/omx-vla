import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect randomized scripted pick-and-place episodes (raw frames + joint data).")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--recovery_prob", type=float, default=0.1)
parser.add_argument("--funnel_prob", type=float, default=0.25)
parser.add_argument("--wrist_modes", type=int, choices=[1, 3], default=1)
parser.add_argument("--no_randomize", action="store_true")
parser.add_argument("--preview_video", default=None)
parser.add_argument("--preview_episodes", type=int, default=5)
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
from omx.kinematics import JOINT1_XY, forward, inverse_top_down
from omx.robot_cfg import OMX_ARM_JOINT_NAMES, OMX_GRIPPER_CLOSED, OMX_GRIPPER_OPEN, OMX_JOINT_NAMES
from pxr import Gf, UsdGeom, UsdLux, UsdShade

import pick_place_red_cube.sim  # noqa: F401
from pick_place_red_cube.sim import env_cfg as task_cfg
from pick_place_red_cube.sim.mdp import cube_in_square

TASK = "OMX-PickPlaceRedCube-v0"
FPS = task_cfg.FPS
SETTLE_STEPS = 10
TAIL_S = 1.0
TRANSIT_PHASES = {"approach", "move", "home", "reapproach"}

rng = np.random.default_rng(args.seed)
randomize = not args.no_randomize

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.seed = args.seed
cfg.terminations.time_out = None
cfg.terminations.success = None
env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
cube = env.scene["cube"]
top_cam = env.scene["top"]
stage = env.sim.stage
joint_ids = [robot.joint_names.index(j) for j in OMX_JOINT_NAMES]
home = np.array(task_cfg.HOME_JOINT_POS)


def find_shader(root):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(root) and prim.IsA(UsdShade.Shader):
            return UsdShade.Shader(prim)
    raise RuntimeError(f"no shader under {root}")


SURFACES = {
    "table": (find_shader("/World/envs/env_0/Table"), task_cfg.WHITE),
    "front_wall": (find_shader("/World/envs/env_0/FrontWall"), task_cfg.YELLOW),
    "left_wall": (find_shader("/World/envs/env_0/LeftWall"), task_cfg.WHITE),
    "right_wall": (find_shader("/World/envs/env_0/RightWall"), task_cfg.WHITE),
    "cube": (find_shader("/World/envs/env_0/Cube"), task_cfg.RED),
    "square": (find_shader("/World/envs/env_0/TargetSquare"), task_cfg.BLACK),
}
DOME = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
SUN = UsdLux.DistantLight(stage.GetPrimAtPath("/World/CeilingLight"))


def xform_op(prim_path, op_type):
    for op in UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    raise RuntimeError(f"{prim_path} has no {op_type} op")


SUN_ORIENT = xform_op("/World/CeilingLight", UsdGeom.XformOp.TypeOrient)
SUN_ORIENT_TYPE = type(SUN_ORIENT.Get())
WRIST_TRANSLATE = xform_op("/World/envs/env_0/Robot/link5/WristCamera", UsdGeom.XformOp.TypeTranslate)
WRIST_ORIENT = xform_op("/World/envs/env_0/Robot/link5/WristCamera", UsdGeom.XformOp.TypeOrient)
WRIST_TRANSLATE_TYPE = type(WRIST_TRANSLATE.Get())
WRIST_ORIENT_TYPE = type(WRIST_ORIENT.Get())
WRIST_BASE_POS = Gf.Vec3d(WRIST_TRANSLATE.Get())
WRIST_BASE_ROT = Gf.Quatd(WRIST_ORIENT.Get())


def randomize_scene():
    params = {}
    for name, (shader, base) in SURFACES.items():
        color = np.array(base)
        if randomize:
            color = np.clip(color * rng.uniform(0.85, 1.12) * (1.0 + rng.normal(0.0, 0.04, 3)), 0.0, 1.0)
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color.tolist()))
        params[f"color_{name}"] = color.round(4).tolist()
        rough = u(0.4, 0.9)
        shader.GetInput("roughness").Set(float(rough))
        params[f"roughness_{name}"] = round(rough, 3)
    dome, sun = 800.0, 1500.0
    tint = np.ones(3)
    if randomize:
        dome *= rng.uniform(0.6, 1.4)
        sun *= rng.uniform(0.5, 1.5)
        tint = np.clip(1.0 + rng.normal(0.0, 0.05, 3), 0.85, 1.0)
    DOME.GetIntensityAttr().Set(dome)
    SUN.GetIntensityAttr().Set(sun)
    SUN.GetColorAttr().Set(Gf.Vec3f(*tint.tolist()))
    elev, azim = math.radians(u(15.0, 45.0)), math.radians(u(-60.0, 60.0))
    sun_q = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(azim)) * Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(elev))
    SUN_ORIENT.Set(SUN_ORIENT_TYPE(sun_q.GetQuat()))
    params.update(sun_tilt_deg=math.degrees(elev), sun_azimuth_deg=math.degrees(azim))
    w_pos = WRIST_BASE_POS + Gf.Vec3d(*(rng.uniform(-0.003, 0.003, 3) if randomize else np.zeros(3)))
    w_rot = Gf.Rotation(Gf.Vec3d(*rng.normal(0, 1, 3)) if randomize else Gf.Vec3d(1, 0, 0), u(-2.5, 2.5))
    WRIST_TRANSLATE.Set(WRIST_TRANSLATE_TYPE(w_pos))
    WRIST_ORIENT.Set(WRIST_ORIENT_TYPE((w_rot * Gf.Rotation(WRIST_BASE_ROT)).GetQuat()))
    params.update(wrist_cam_offset=list(w_pos - WRIST_BASE_POS))
    params.update(dome_intensity=dome, sun_intensity=sun, light_tint=tint.round(4).tolist())
    eye = np.array(task_cfg.TOP_CAMERA_EYE)
    target = np.array(task_cfg.TOP_CAMERA_TARGET)
    if randomize:
        eye = eye + rng.uniform(-0.02, 0.02, 3)
        target = target + rng.uniform(-0.015, 0.015, 3)
    origin = env.scene.env_origins[0]
    top_cam.set_world_poses_from_view(
        (origin + torch.tensor(eye, device=env.device, dtype=torch.float32)).unsqueeze(0),
        (origin + torch.tensor(target, device=env.device, dtype=torch.float32)).unsqueeze(0),
    )
    params.update(top_camera_eye=eye.round(4).tolist(), top_camera_target=target.round(4).tolist())
    return params


def wrap_quarter(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def cube_state():
    origin = env.scene.env_origins[0]
    xyz = (cube.data.root_pos_w[0] - origin).tolist()
    yaw = euler_xyz_from_quat(cube.data.root_quat_w)[2][0].item()
    return xyz, yaw


def arm_q(x, y, z, yaw, pitch=math.pi / 2, wrist=None):
    pitches = [pitch] if pitch >= math.pi / 2 else list(np.linspace(pitch, math.pi / 2, 4))
    pitches += list(np.arange(pitch - math.radians(2.5), math.radians(20.0) - 1e-9, -math.radians(2.5)))
    for p in pitches:
        zz = z
        while zz >= min(z, 0.05) - 1e-9:
            try:
                q1, q2, q3, q4 = inverse_top_down(x, y, zz, p)
                q5 = wrap_quarter(q1 - yaw) if wrist is None else wrist
                return np.array([q1, q2, q3, q4, q5])
            except ValueError:
                zz -= 0.005
    raise ValueError(f"target out of reach: {(x, y, z)}")


def reachable_z(x, y, z, pitch):
    while z > 0.0:
        try:
            inverse_top_down(x, y, z, pitch)
            return z
        except ValueError:
            z -= 0.002
    raise ValueError(f"target out of reach: {(x, y)} at pitch {math.degrees(pitch):.0f}")


def feasible_pitch(x, y, zs, pitch, lowest=math.radians(20.0)):
    p = pitch
    while p >= lowest - 1e-9:
        try:
            for z in zs:
                inverse_top_down(x, y, z, p)
            return p
        except ValueError:
            p -= math.radians(2.5)
    raise ValueError(f"target out of reach: {(x, y)} for z {zs}")


def u(lo, hi):
    return rng.uniform(lo, hi) if randomize else (lo + hi) / 2


def chance(p):
    return randomize and rng.random() < p


class Recorder:
    def __init__(self, ep_dir):
        self.ep_dir = ep_dir
        (ep_dir / "top").mkdir(parents=True)
        (ep_dir / "wrist").mkdir(parents=True)
        self.state, self.action, self.phase = [], [], []
        self.q_cmd = None
        self.g_cmd = None
        self.noise = np.zeros(5)
        self.obs = None
        self.preview = None
        self.recording = True
        self.profile = "cosine"

    def step(self, q, g, phase):
        if self.recording:
            self.save_frame(q, g, phase)
        executed = np.concatenate([q, [g]]).astype(np.float32)
        if randomize and phase in TRANSIT_PHASES:
            self.noise = 0.9 * self.noise + rng.normal(0.0, 0.006, 5)
        else:
            self.noise[:] = 0.0
        executed[:5] += self.noise
        self.obs, *_ = env.step(torch.tensor(executed, device=env.device).unsqueeze(0))
        self.q_cmd, self.g_cmd = q, g

    def save_frame(self, q, g, phase):
        policy = self.obs["policy"]
        i = len(self.state)
        top = policy["top"][0].cpu().numpy()
        wrist = policy["wrist"][0].cpu().numpy()
        cv2.imwrite(str(self.ep_dir / "top" / f"{i:06d}.jpg"), cv2.cvtColor(top, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(self.ep_dir / "wrist" / f"{i:06d}.jpg"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if self.preview is not None:
            self.preview(top, wrist, phase)
        self.state.append(robot.data.joint_pos[0, joint_ids].cpu().numpy().copy())
        self.action.append(np.concatenate([q, [g]]).astype(np.float32))
        self.phase.append(phase)

    def timing(self, duration):
        n = max(2, int(round(duration * FPS)))
        gamma = u(0.8, 1.25)
        for i in range(n):
            t = ((i + 1) / n) ** gamma
            if self.profile == "minjerk":
                yield t * t * t * (10 - 15 * t + 6 * t * t)
            else:
                yield 0.5 - 0.5 * math.cos(math.pi * t)

    def move(self, phase, q_goal, g_goal, duration, q_via=None):
        q0, g0 = self.q_cmd.copy(), self.g_cmd
        ctrl = None if q_via is None else 2.0 * q_via - 0.5 * (q0 + q_goal)
        for s in self.timing(duration):
            if ctrl is None:
                q = q0 + s * (q_goal - q0)
            else:
                q = (1 - s) ** 2 * q0 + 2 * s * (1 - s) * ctrl + s * s * q_goal
            self.step(q, g0 + s * (g_goal - g0), phase)

    def vertical(self, phase, x, y, z0, z1, pitch, wrist, g_goal, duration):
        z0, z1 = reachable_z(x, y, z0, pitch), reachable_z(x, y, z1, pitch)
        q_start = arm_q(x, y, z0, 0.0, pitch, wrist)
        if np.abs(q_start - self.q_cmd).max() > 0.02:
            self.move(phase, q_start, self.g_cmd, 0.3)
        g0 = self.g_cmd
        for s in self.timing(duration):
            self.step(arm_q(x, y, z0 + s * (z1 - z0), 0.0, pitch, wrist), g0 + s * (g_goal - g0), phase)

    def hold(self, phase, duration):
        for _ in range(max(1, int(round(duration * FPS)))):
            self.step(self.q_cmd, self.g_cmd, phase)


def grasp_wrist(x, y, yaw):
    q1 = math.atan2(y - JOINT1_XY[1], x - JOINT1_XY[0])
    base = wrap_quarter(q1 - yaw)
    if args.wrist_modes == 1:
        return float(base + math.radians(u(-5.0, 5.0)))
    options = [w for w in (base, base + math.pi / 2, base - math.pi / 2) if abs(w) <= 1.9]
    weights = np.array([0.6 if w == base else 0.2 for w in options])
    w = rng.choice(options, p=weights / weights.sum()) if randomize else base
    return float(w + math.radians(u(-5.0, 5.0)))


def run_episode(rec, speed, knobs):
    def d(base):
        return base * speed * u(0.8, 1.25)

    hover = u(0.07, 0.11)
    grasp_z = u(0.005, 0.009)
    grasp_pitch = math.radians(u(72.0, 90.0))
    hover_pitch = max(math.radians(60.0), grasp_pitch - math.radians(u(0.0, 15.0)))
    (cx0, cy0, _), _ = cube_state()
    grasp_pitch = feasible_pitch(cx0, cy0, (grasp_z, hover + 0.04), grasp_pitch)
    hover_pitch = min(hover_pitch, grasp_pitch)
    knobs.update(hover=hover, grasp_z=grasp_z, grasp_pitch_deg=math.degrees(grasp_pitch),
                 hover_pitch_deg=math.degrees(hover_pitch))

    mode = rng.choice(["normal", "recovery", "funnel"], p=[1 - args.recovery_prob - args.funnel_prob, args.recovery_prob, args.funnel_prob]) if randomize else "normal"
    funnel = mode == "funnel"
    knobs["start_mode"] = str(mode)
    rec.profile = "minjerk" if chance(0.5) else "cosine"
    knobs["profile"] = rec.profile
    if funnel:
        rec.recording = False
        (cx, cy, _), cyaw = cube_state()
        ang = rng.uniform(0, 2 * math.pi)
        off = rng.uniform(0.01, 0.025)
        w = grasp_wrist(cx, cy, cyaw) + u(-0.2, 0.2)
        rec.move("approach", arm_q(cx + off * math.cos(ang), cy + off * math.sin(ang), u(0.025, 0.055), cyaw, grasp_pitch, w),
                 OMX_GRIPPER_OPEN, d(1.8))
        rec.hold("approach", 0.2)
        rec.recording = True
    if mode == "recovery":
        rec.recording = False
        (cx, cy, _), cyaw = cube_state()
        ang = rng.uniform(0, 2 * math.pi)
        off = rng.uniform(0.012, 0.02)
        mx, my = cx + off * math.cos(ang), cy + off * math.sin(ang)
        w = grasp_wrist(mx, my, cyaw)
        rec.move("approach", arm_q(mx, my, hover, cyaw, hover_pitch, w), OMX_GRIPPER_OPEN, d(1.5))
        rec.move("descend", arm_q(mx, my, grasp_z, cyaw, grasp_pitch, w), OMX_GRIPPER_OPEN, d(1.0))
        rec.move("grasp", rec.q_cmd, OMX_GRIPPER_CLOSED, d(0.8))
        rec.move("lift", arm_q(mx, my, hover, cyaw, grasp_pitch, w), OMX_GRIPPER_CLOSED, d(0.8))
        rec.move("reopen", rec.q_cmd, OMX_GRIPPER_OPEN, d(0.5))
        rec.hold("reopen", u(0.2, 0.5))
        rec.recording = True
        recovered = True
    else:
        recovered = False
    rec.hold("start", u(0.0, 0.4))

    (cx, cy, _), cyaw = cube_state()
    gx, gy = cx + u(-0.003, 0.003), cy + u(-0.003, 0.003)
    w_pick = grasp_wrist(gx, gy, cyaw)
    knobs["wrist_pick"] = w_pick
    approach_via = None
    if not funnel and chance(0.5):
        sx0, sy0, _ = forward(rec.q_cmd)
        vx = (sx0 + gx) / 2 + u(-0.04, 0.04)
        vy = (sy0 + gy) / 2 + u(-0.04, 0.04)
        vz = u(0.08, 0.16)
        approach_via = arm_q(vx, vy, vz, 0.0, math.radians(u(60.0, 90.0)), (rec.q_cmd[4] + w_pick) / 2 + u(-0.2, 0.2))
        knobs["approach_via"] = [vx, vy, vz]
    if funnel:
        rec.move("approach", arm_q(gx, gy, u(0.02, 0.04), cyaw, grasp_pitch, w_pick), OMX_GRIPPER_OPEN, d(0.9))
    else:
        rec.move("reapproach" if recovered else "approach", arm_q(gx, gy, hover, cyaw, hover_pitch, w_pick), OMX_GRIPPER_OPEN,
                 d(2.2 if approach_via is not None else 1.5), q_via=approach_via)
    if funnel:
        rec.move("descend", arm_q(gx, gy, grasp_z, cyaw, grasp_pitch, w_pick), OMX_GRIPPER_OPEN, d(0.7))
    else:
        if hover_pitch != grasp_pitch:
            rec.move("descend", arm_q(gx, gy, hover, cyaw, grasp_pitch, w_pick), OMX_GRIPPER_OPEN, d(0.4))
        rec.vertical("descend", gx, gy, hover, grasp_z, grasp_pitch, w_pick, OMX_GRIPPER_OPEN, d(1.0))
    rec.move("grasp", rec.q_cmd, OMX_GRIPPER_CLOSED, d(0.8))
    rec.hold("grasp", u(0.0, 0.2))
    lift_z = hover + u(-0.01, 0.04)
    rec.vertical("lift", gx, gy, grasp_z, lift_z, grasp_pitch, w_pick, OMX_GRIPPER_CLOSED, d(1.0))

    sx = task_cfg.SQUARE_CENTER[0] + u(-0.012, 0.012)
    sy = task_cfg.SQUARE_CENTER[1] + u(-0.012, 0.012)
    w_place = w_pick + u(-0.35, 0.35)
    place_pitch = math.radians(u(72.0, 90.0))
    place_z = u(0.012, 0.018)
    place_pitch = feasible_pitch(sx, sy, (place_z, hover + 0.03), place_pitch)
    knobs.update(wrist_place=w_place, place_pitch_deg=math.degrees(place_pitch))
    move_via = None
    if chance(0.6):
        vx = (gx + sx) / 2 + u(-0.04, 0.04)
        vy = (gy + sy) / 2 + u(-0.04, 0.04)
        vz = max(hover, lift_z) + u(0.0, 0.07)
        move_via = arm_q(vx, vy, vz, 0.0, math.radians(u(60.0, 90.0)), (w_pick + w_place) / 2 + u(-0.3, 0.3))
        knobs["move_via"] = [vx, vy, vz]
    place_hover = hover + u(-0.01, 0.03)
    rec.move("move", arm_q(sx, sy, place_hover, 0.0, place_pitch, w_place), OMX_GRIPPER_CLOSED,
             d(2.0 if move_via is not None else 1.5), q_via=move_via)
    rec.vertical("place", sx, sy, place_hover, place_z, place_pitch, w_place, OMX_GRIPPER_CLOSED, d(1.0))
    rec.move("release", rec.q_cmd, OMX_GRIPPER_OPEN, d(0.6))
    rec.vertical("retreat", sx, sy, place_z, place_hover, place_pitch, w_place, OMX_GRIPPER_OPEN, d(0.8))
    end = home + rng.uniform(-0.05, 0.05, 5) * randomize
    rec.move("home", end, OMX_GRIPPER_OPEN, d(1.5))
    rec.move("close", rec.q_cmd, OMX_GRIPPER_CLOSED, d(0.5))
    rec.hold("done", TAIL_S)
    return recovered


preview_writer = None
if args.preview_video:
    Path(args.preview_video).parent.mkdir(parents=True, exist_ok=True)
    preview_writer = cv2.VideoWriter(args.preview_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 520))

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "info.json").write_text(json.dumps({
    "task": task_cfg.TASK_DESCRIPTION,
    "fps": FPS,
    "joint_names": OMX_JOINT_NAMES,
    "units": "rad; gripper_joint_1 0.7 open, 0 closed",
    "seed": args.seed,
    "randomize": randomize,
    "recovery_prob": args.recovery_prob,
}, indent=2))

def abort_on_exception(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = abort_on_exception

results = []
for ep in range(args.episodes):
    ep_dir = out / f"episode_{ep:06d}"
    if (ep_dir / "data.npz").exists():
        print(f"episode {ep}: exists, skipped", flush=True)
        continue
    rng = np.random.default_rng([args.seed, ep])
    torch.manual_seed(args.seed * 1_000_000 + ep)
    tmp_dir = out / f".tmp_episode_{ep:06d}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    t0 = time.time()

    obs, _ = env.reset()
    start_q = home + (rng.uniform(-0.05, 0.05, 5) if randomize else 0.0)
    jp = robot.data.default_joint_pos.clone()
    arm_ids = [robot.joint_names.index(j) for j in OMX_ARM_JOINT_NAMES]
    jp[0, arm_ids] = torch.tensor(start_q, device=env.device, dtype=jp.dtype)
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    scene_params = randomize_scene()
    settle = torch.tensor(np.concatenate([start_q, [OMX_GRIPPER_OPEN]]), device=env.device, dtype=torch.float32).unsqueeze(0)
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(settle)

    rec = Recorder(tmp_dir)
    rec.obs = obs
    rec.q_cmd, rec.g_cmd = start_q.copy(), OMX_GRIPPER_OPEN
    if preview_writer is not None and ep < args.preview_episodes:
        def preview(top, wrist, phase, ep=ep):
            frame = cv2.cvtColor(np.hstack([top, wrist]), cv2.COLOR_RGB2BGR)
            bar = np.full((40, 1280, 3), 30, dtype=np.uint8)
            text = f"Episode {ep + 1}  |  {phase}  |  task: \"{task_cfg.TASK_DESCRIPTION}\""
            cv2.putText(bar, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            preview_writer.write(np.vstack([bar, frame]))
        rec.preview = preview

    cube_start, cube_yaw = cube_state()
    speed = u(0.85, 1.2)
    knobs = {"start_q": np.round(start_q, 3).tolist(), "speed": speed}
    try:
        recovered = run_episode(rec, speed, knobs)
        error = None
    except ValueError as exc:
        recovered, error = False, str(exc)
    success = error is None and bool(
        cube_in_square(env, square_center=task_cfg.SQUARE_CENTER, square_half_size=(task_cfg.SQUARE_SIZE[0] / 2, task_cfg.SQUARE_SIZE[1] / 2), min_gripper_opening=-1.0)[0]
    )
    cube_end, _ = cube_state()
    np.savez(
        tmp_dir / "data.npz",
        state=np.array(rec.state, dtype=np.float32),
        action=np.array(rec.action, dtype=np.float32),
        phase=np.array(rec.phase),
        success=success,
        recovered=recovered,
        cube_start=np.array(cube_start),
        cube_yaw=cube_yaw,
        cube_end=np.array(cube_end),
        speed=speed,
        scene=json.dumps(scene_params),
        knobs=json.dumps(knobs, default=float),
        error=error or "",
    )
    tmp_dir.rename(ep_dir)
    results.append(success)
    print(
        f"episode {ep}: success {success} recovered {recovered} frames {len(rec.state)} "
        f"cube {np.round(cube_start, 3).tolist()} yaw {math.degrees(cube_yaw):.0f} -> {np.round(cube_end, 3).tolist()} "
        f"time {time.time() - t0:.1f}s{' error ' + error if error else ''}",
        flush=True,
    )

if preview_writer is not None:
    preview_writer.release()
print(f"SUMMARY success {sum(results)}/{len(results)} out {out}", flush=True)
sys.stdout.flush()
os._exit(0)
