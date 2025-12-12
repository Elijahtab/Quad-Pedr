# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm

from .rough_env_cfg import UnitreeGo2RoughEnvCfg, NavCommandsCfg
from . import custom_rewards


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 1
        self.rewards.base_height_err = RewTerm(
            func=custom_rewards.base_height_l2_flat,
            weight=-3.0,  # stronger anti-crouch
            params={
                "target_height": 0.38,               # same nominal COM height
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.track_cmd_height_flat = RewTerm(
            func=custom_rewards.track_commanded_height_flat,
            weight=1.0,   # tune: 0.5–2.0 is a reasonable range to try
            params={
                "command_name": "gait_params",
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_height": 0.38,
                "min_offset": 0.02,  # only care when >= 2 cm offset
                "std": 0.05,         # ~5 cm tolerance
            },
        )
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        #self.rewards.base_height_err = None
        self.rewards.track_foot_clearance = None
        #JANK ALERT: use this to change height for test
        #self.commands.gait_params.fixed_height = -0.18

class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        self.commands.gait_params.resampling_time_range = (1e9, 1e9)