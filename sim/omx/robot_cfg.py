from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

OMX_DIR = Path(__file__).resolve().parent
OMX_F_USD_PATH = OMX_DIR / "usd" / "omx_f.usd"

OMX_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5"]
OMX_GRIPPER_JOINT_NAME = "gripper_joint_1"
OMX_GRIPPER_MIMIC_JOINT_NAME = "gripper_joint_2"
OMX_JOINT_NAMES = OMX_ARM_JOINT_NAMES + [OMX_GRIPPER_JOINT_NAME]

OMX_GRIPPER_OPEN = 0.7
OMX_GRIPPER_CLOSED = 0.0

OMX_HOME_JOINT_POS = {
    "joint1": 0.0,
    "joint2": -1.7,
    "joint3": 1.6,
    "joint4": 0.2,
    "joint5": 0.0,
    "gripper_joint_1": OMX_GRIPPER_OPEN,
    "gripper_joint_2": -OMX_GRIPPER_OPEN,
}

OMX_F_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(OMX_F_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=OMX_HOME_JOINT_POS,
    ),
    actuators={
        "xl430": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3"],
            effort_limit_sim=1.5,
            velocity_limit_sim=4.8,
            stiffness=50.0,
            damping=1.3,
        ),
        "xl330_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint4", "joint5"],
            effort_limit_sim=0.52,
            velocity_limit_sim=4.8,
            stiffness=50.0,
            damping=0.5,
        ),
        "xl330_gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_1", "gripper_joint_2"],
            effort_limit_sim=0.5,
            velocity_limit_sim=4.8,
            stiffness=20.0,
            damping=0.2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
