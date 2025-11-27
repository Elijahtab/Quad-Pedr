# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from .rough_env_cfg import NavCommandsCfg
from .commands import GoalCommandCfg 

@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # -------- FIX: replace default navigation commands --------
        self.commands = NavCommandsCfg()
        self.commands.base_velocity = None                     # disable velocity-based nav
        self.commands.goal_pos = GoalCommandCfg()              # <-- YOUR CUSTOM COMMAND
        self.commands.goal_pos.debug_vis = True                # enable blue sphere

        # -------- rewards --------
        #self.rewards.flat_orientation_l2.weight = -2.5
        #self.rewards.feet_air_time.weight = 0.25

        # -------- flat terrain --------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # -------- disable height scan --------
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # -------- disable terrain curriculum --------
        self.curriculum.terrain_levels = None



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
