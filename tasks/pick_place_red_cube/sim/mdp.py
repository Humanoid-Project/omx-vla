import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def cube_pos(env, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def cube_in_square(
    env,
    square_center: tuple[float, float],
    square_half_size: float | tuple[float, float],
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_name: str = "gripper_joint_1",
    max_height: float = 0.02,
    min_gripper_opening: float = 0.4,
) -> torch.Tensor:
    pos = cube_pos(env, cube_cfg)
    center = torch.tensor(square_center, device=pos.device)
    half = torch.as_tensor(square_half_size, device=pos.device, dtype=pos.dtype)
    inside = ((pos[:, :2] - center).abs() <= half).all(dim=1)
    resting = pos[:, 2] <= max_height
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_id = robot.find_joints(gripper_joint_name)[0][0]
    released = robot.data.joint_pos[:, gripper_id] >= min_gripper_opening
    return inside & resting & released


def reset_cube_around_square(
    env,
    env_ids: torch.Tensor,
    center: tuple[float, float],
    radius_range: tuple[float, float],
    n_directions: int,
    direction_jitter: float,
    min_x: float,
    max_reach: float,
    reach_origin: tuple[float, float],
    yaw_range: tuple[float, float],
    height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    cube: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)
    pos = torch.zeros(n, 3, device=env.device)
    for i in range(n):
        for _ in range(200):
            k = int(torch.randint(n_directions, (1,)))
            ang = 2 * math.pi * k / n_directions + float(torch.empty(1).uniform_(-direction_jitter, direction_jitter))
            r = float(torch.empty(1).uniform_(*radius_range))
            x, y = center[0] + r * math.cos(ang), center[1] + r * math.sin(ang)
            if x >= min_x and math.hypot(x - reach_origin[0], y - reach_origin[1]) <= max_reach:
                break
        pos[i, 0], pos[i, 1], pos[i, 2] = x, y, height
    yaw = torch.empty(n, device=env.device).uniform_(*yaw_range)
    quat = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)
    pose = torch.cat([pos + env.scene.env_origins[env_ids], quat], dim=-1)
    cube.write_root_pose_to_sim(pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
