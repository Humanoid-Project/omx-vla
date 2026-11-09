import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from omx.actions import MimicGripperActionCfg
from omx.camera_utils import focal_length_for_hfov, look_at_quat_world
from omx.robot_cfg import (
    OMX_ARM_JOINT_NAMES,
    OMX_F_CFG,
    OMX_GRIPPER_JOINT_NAME,
    OMX_GRIPPER_MIMIC_JOINT_NAME,
    OMX_JOINT_NAMES,
)

from . import mdp as task_mdp

TASK_DESCRIPTION = "Pick up the red cube and place it inside the black square."

FPS = 30
CUBE_SIZE = 0.03
SQUARE_SIZE = (0.081, 0.104)
SQUARE_CENTER = (0.21, -0.08)
CUBE_RING_RADIUS = (0.09, 0.15)
CUBE_RING_DIRECTIONS = 8
CUBE_RING_JITTER = math.radians(10.0)
CUBE_MIN_X = 0.11
CUBE_MAX_REACH = 0.35

TABLE_X = (-0.25, 0.65)
TABLE_Y = (-0.45, 0.45)
TABLE_THICKNESS = 0.04
WALL_HEIGHT = 0.6
WALL_THICKNESS = 0.02

CAMERA_HFOV_DEG = 51.0
CAMERA_APERTURE = 20.955
TOP_CAMERA_EYE = (0.57, 0.0, 0.40)
TOP_CAMERA_TARGET = (0.09, 0.02, 0.0)
WRIST_CAMERA_EYE = (0.05, 0.0, 0.072)
WRIST_CAMERA_TARGET = (0.13, 0.0, -0.03)
WRIST_CAMERA_HFOV_DEG = 80.0

WHITE = (0.9, 0.9, 0.9)
YELLOW = (0.95, 0.8, 0.1)
RED = (0.8, 0.05, 0.05)
BLACK = (0.02, 0.02, 0.02)
OMX_BLUE = (0.05, 0.2, 0.75)


def _box(size, color, collision=True):
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
    )


def _camera(prim_path, eye, target, hfov_deg=CAMERA_HFOV_DEG):
    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(pos=eye, rot=look_at_quat_world(eye, target), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_for_hfov(hfov_deg, CAMERA_APERTURE),
            horizontal_aperture=CAMERA_APERTURE,
            clipping_range=(0.01, 10.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1.0 / FPS,
    )


_table_center_x = sum(TABLE_X) / 2
_table_len_x = TABLE_X[1] - TABLE_X[0]
_table_len_y = TABLE_Y[1] - TABLE_Y[0]


@configclass
class PickPlaceRedCubeSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = OMX_F_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=_box((_table_len_x, _table_len_y, TABLE_THICKNESS), WHITE),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_table_center_x, 0.0, -TABLE_THICKNESS / 2)),
    )
    front_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FrontWall",
        spawn=_box((WALL_THICKNESS, _table_len_y, WALL_HEIGHT), YELLOW),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_X[0] - WALL_THICKNESS / 2, 0.0, WALL_HEIGHT / 2)),
    )
    left_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftWall",
        spawn=_box((_table_len_x, WALL_THICKNESS, WALL_HEIGHT), WHITE),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_table_center_x, TABLE_Y[1] + WALL_THICKNESS / 2, WALL_HEIGHT / 2)),
    )
    right_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RightWall",
        spawn=_box((_table_len_x, WALL_THICKNESS, WALL_HEIGHT), WHITE),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_table_center_x, TABLE_Y[0] - WALL_THICKNESS / 2, WALL_HEIGHT / 2)),
    )
    target_square = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetSquare",
        spawn=_box((SQUARE_SIZE[0], SQUARE_SIZE[1], 0.0005), BLACK, collision=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(SQUARE_CENTER[0], SQUARE_CENTER[1], 0.00025)),
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RED, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(SQUARE_CENTER[0], SQUARE_CENTER[1] + 0.13, CUBE_SIZE / 2)),
    )

    top = _camera("{ENV_REGEX_NS}/TopCamera", TOP_CAMERA_EYE, TOP_CAMERA_TARGET)
    wrist = _camera("{ENV_REGEX_NS}/Robot/link5/WristCamera", WRIST_CAMERA_EYE, WRIST_CAMERA_TARGET, WRIST_CAMERA_HFOV_DEG)

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=800.0),
    )
    ceiling_light = AssetBaseCfg(
        prim_path="/World/CeilingLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=1500.0, angle=20.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9659, 0.0, 0.2588, 0.0)),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=OMX_ARM_JOINT_NAMES, scale=1.0, use_default_offset=False, preserve_order=True
    )
    gripper_action = MimicGripperActionCfg(
        asset_name="robot", joint_name=OMX_GRIPPER_JOINT_NAME, mimic_joint_name=OMX_GRIPPER_MIMIC_JOINT_NAME
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=OMX_JOINT_NAMES, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)
        top = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False})
        wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )
        cube_pos = ObsTerm(func=task_mdp.cube_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=task_mdp.reset_cube_around_square,
        mode="reset",
        params={
            "center": SQUARE_CENTER,
            "radius_range": CUBE_RING_RADIUS,
            "n_directions": CUBE_RING_DIRECTIONS,
            "direction_jitter": CUBE_RING_JITTER,
            "min_x": CUBE_MIN_X,
            "max_reach": CUBE_MAX_REACH,
            "reach_origin": (-0.01125, 0.0),
            "yaw_range": (-math.pi / 4, math.pi / 4),
            "height": CUBE_SIZE / 2,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=task_mdp.cube_in_square,
        params={"square_center": SQUARE_CENTER, "square_half_size": (SQUARE_SIZE[0] / 2, SQUARE_SIZE[1] / 2)},
    )


@configclass
class RewardsCfg:
    pass


@configclass
class PickPlaceRedCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceRedCubeSceneCfg = PickPlaceRedCubeSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    task_description: str = TASK_DESCRIPTION

    def __post_init__(self):
        self.scene.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=OMX_BLUE, roughness=0.5)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (0.7, 0.5, 0.5)
        self.viewer.lookat = (0.15, 0.0, 0.05)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
