import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def cube_pos(env, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def cube_in_square(
    env,
    square_center: tuple[float, float],
    square_half_size: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_name: str = "gripper_joint_1",
    max_height: float = 0.02,
    min_gripper_opening: float = 0.4,
) -> torch.Tensor:
    pos = cube_pos(env, cube_cfg)
    center = torch.tensor(square_center, device=pos.device)
    inside = (pos[:, :2] - center).abs().max(dim=1).values <= square_half_size
    resting = pos[:, 2] <= max_height
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_id = robot.find_joints(gripper_joint_name)[0][0]
    released = robot.data.joint_pos[:, gripper_id] >= min_gripper_opening
    return inside & resting & released
