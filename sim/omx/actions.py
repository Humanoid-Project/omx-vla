from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class MimicGripperAction(ActionTerm):
    cfg: "MimicGripperActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "MimicGripperActionCfg", env):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints([cfg.joint_name, cfg.mimic_joint_name], preserve_order=True)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(self.cfg.lower, self.cfg.upper)

    def apply_actions(self):
        target = torch.cat([self._processed_actions, self.cfg.mimic_multiplier * self._processed_actions], dim=1)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


@configclass
class MimicGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = MimicGripperAction
    joint_name: str = MISSING
    mimic_joint_name: str = MISSING
    mimic_multiplier: float = -1.0
    lower: float = 0.0
    upper: float = 1.745
