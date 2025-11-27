# go2_nav/rough_env_cfg.py

import torch
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg as BaseSceneCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg as BaseRewardsCfg,
    CommandsCfg as BaseCommandsCfg,
    TerminationsCfg as BaseTerminationsCfg,
)

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import custom_obs, custom_rewards
from .commands import GoalCommandCfg


# ======================================================================
# Scene
# ======================================================================
@configclass
class NavSceneCfg(BaseSceneCfg):
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
    )

    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.25, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-15, 15),
            horizontal_fov_range=(-180, 180),
            horizontal_res=1.0,
        ),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


# ======================================================================
# Observations
# ======================================================================
@configclass
class NavObservationsCfg(BaseObservationsCfg):

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        velocity_commands = None  # remove old command type

        goal_relative = ObsTerm(
            func=custom_obs.goal_relative_target,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        lidar = ObsTerm(
            func=custom_obs.placeholder_lidar,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )

        lookahead = ObsTerm(
            func=custom_obs.lookahead_hint
        )

        goal_dir_b = ObsTerm(
            func=custom_obs.goal_direction_body,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "goal_pos"
            }
        )

        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()


# ======================================================================
# Rewards
# ======================================================================
@configclass
class NavRewardsCfg(BaseRewardsCfg):

    # disable ALL inherited locomotion rewards
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    track_vel_norm = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    base_height_l2 = None
    undesired_contacts = None
    dof_torques_l2 = None
    flat_orientation_l2 = None
    feet_air_time = None

    # ----------------------------------------------------
    # Minimal navigation rewards
    # ----------------------------------------------------

    heading_align = RewTerm(
        func=custom_rewards.heading_alignment,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    )

    vel_to_goal = RewTerm(
        func=custom_rewards.vel_toward_goal,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    )

    progress_to_goal = RewTerm(
        func=custom_rewards.progress_to_goal,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    )




# ======================================================================
# Commands
# ======================================================================
@configclass
class NavCommandsCfg(BaseCommandsCfg):

    base_velocity = None  # disable old command

    goal_pos = GoalCommandCfg(
        resampling_time_range=(10.0, 15.0),
        radius_range=(1.0, 5.0),
        debug_vis=True,
    )


# ======================================================================
# Terminations
# ======================================================================
@configclass
class NavTerminationsCfg(BaseTerminationsCfg):
    base_contact = None

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 2.5},
    )


# ======================================================================
# Full Env Config
# ======================================================================
@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    scene: NavSceneCfg = NavSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: NavObservationsCfg = NavObservationsCfg()
    rewards: NavRewardsCfg = NavRewardsCfg()
    commands: NavCommandsCfg = NavCommandsCfg()
    terminations: NavTerminationsCfg = NavTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.max_init_terrain_level = 0

        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.robot.init_state.rot = (0, 0, 0, 1)

        self.actions.joint_pos.scale = 0.25

        # configure feet_air_time sensor
        if self.rewards.feet_air_time is not None:
            sensor_cfg = self.rewards.feet_air_time.params["sensor_cfg"]
            sensor_cfg.body_names = ".*_foot"      # regex for foot links

        self.events.push_robot = None
        self.curriculum.terrain_levels = None
