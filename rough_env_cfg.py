# go2_nav/rough_env_cfg.py

import torch
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg

from isaaclab.envs import ViewerCfg 

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
# from .commands import GoalCommandCfg


# ======================================================================
# Scene
# ======================================================================
@configclass
class NavSceneCfg(BaseSceneCfg):
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
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
        # velocity_commands = None  # remove old command type

        # goal_relative = ObsTerm(
        #     func=custom_obs.goal_relative_target,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )

        lidar = ObsTerm(
            func=custom_obs.placeholder_lidar,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )

        lookahead = ObsTerm(
            func=custom_obs.lookahead_hint
        )

        # goal_dir_b = ObsTerm(
        #     func=custom_obs.goal_direction_body,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "command_name": "goal_pos"
        #     }
        # )

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
    base_height_l2 = RewTerm(
        func=custom_rewards.base_height_l2,
        weight=-5.0,   # negative because it’s a penalty
        params={
            "target_height": 0.36,                 # tune this! ~standing base height in meters
            "asset_cfg": SceneEntityCfg("robot"),
            # For flat env we don’t need a sensor, so omit "sensor_cfg"
        },
    )
    # # disable ALL inherited locomotion rewards
    # track_lin_vel_xy_exp = None
    # track_ang_vel_z_exp = None
    # track_vel_norm = None
    # lin_vel_z_l2 = None
    # ang_vel_xy_l2 = None
    # base_height_l2 = None
    # undesired_contacts = None
    # dof_torques_l2 = None
    # flat_orientation_l2 = None
    # feet_air_time = None

    # # ----------------------------------------------------
    # # Minimal navigation rewards
    # # ----------------------------------------------------

    # heading_align = RewTerm(
    #     func=custom_rewards.heading_alignment,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    # )

    # vel_to_goal = RewTerm(
    #     func=custom_rewards.vel_toward_goal,
    #     weight=3.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    # )

    # progress_to_goal = RewTerm(
    #     func=custom_rewards.progress_to_goal,
    #     weight=4.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    # )
    pass




# ======================================================================
# Commands
# ======================================================================
@configclass
class NavCommandsCfg(BaseCommandsCfg):
    """Define our commands."""

    # 1. Velocity Command (Active for Milestones 1-3)
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # MILESTONE 1 SETTINGS: FROZEN (0.0)
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), # Forward/Back
            lin_vel_y=(-0.5, 0.5), # Strafe Left/Right
            ang_vel_z=(-1.0, 1.0), # Turn Left/Right
            heading=(-3.14, 3.14)  # Face any direction
        ),
    )

    # MILESTONE 2: (REPLACE WITH THIS) 
    # ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.0, 1.0), # CHANGED: Move forward up to 1 m/s
    #         lin_vel_y=(0.0, 0.0), 
    #         ang_vel_z=(0.0, 0.0), 
    #         heading=(0.0, 0.0)
    #     ),

    # MILESTONE 3: (REPLACE WITH THIS)
    # ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), # Forward/Back
    #         lin_vel_y=(-0.5, 0.5), # Strafe Left/Right
    #         ang_vel_z=(-1.0, 1.0), # Turn Left/Right
    #         heading=(-3.14, 3.14)  # Face any direction
    #     ),


    # MILESTONE 4: Goal Command
    # UNCOMMENT THIS BLOCK AFTER MILESTONE 3
    # COMMENT OUT THE PREVIOUS BLOCK (base_velocity = ...)
    # goal_pos = commands.GoalCommandCfg(
    #     resampling_time_range=(4.0, 8.0),
    #     debug_vis=True
    # )



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

    viewer: ViewerCfg = ViewerCfg(
        eye=(-4.0, 1.5, 2.0),      # behind & slightly to the left
        lookat=(0.0, 0.0, 0.7),    # look toward robot torso / stairs
        resolution=(1920, 1080),
        origin_type="env",
        env_index=0,
        cam_prim_path="/OmniverseKit_Persp",
    )

    scene: NavSceneCfg = NavSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: NavObservationsCfg = NavObservationsCfg()
    rewards: NavRewardsCfg = NavRewardsCfg()
    commands: NavCommandsCfg = NavCommandsCfg()
    terminations: NavTerminationsCfg = NavTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        
        if hasattr(self.rewards, "undesired_contacts"):
            sensor_cfg = self.rewards.undesired_contacts.params["sensor_cfg"]
            # Penalize contacts on torso + upper legs, not feet
            sensor_cfg.body_names = ["base", ".*_thigh", ".*_calf"]
            sensor_cfg.preserve_order = False
        
        # MILESTONE 2.5: ROUGH TERRAIN NAVIGATION
        # Only going forward
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-1.57, 1.57),
        )
        
        # MILESTONE 3: FULL ROUGH NAVIGATION
        # self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            # lin_vel_x=(-1.0, 1.0),
            # lin_vel_y=(-0.5, 0.5),
            # ang_vel_z=(-1.0, 1.0),
            # heading=(-3.14, 3.14)
        # )
        
        self.scene.terrain.max_init_terrain_level = 5

        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        
        self.scene.robot.visual_material = sim_utils.PreviewSurfaceCfg(
            func=sim_utils.spawn_preview_surface,
            diffuse_color=(0.1, 0.4, 0.9),  # bluish
            roughness=0.4,
            metallic=0.1,
        )
        
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.56)
        self.scene.robot.init_state.rot = (0, 0, 0, 1)

        self.actions.joint_pos.scale = 0.25
        
        # ---- Milestone 3 rough: encourage upright, 4-leg gait ----
        # ------- Reward shaping (only touch rewards that actually exist) -------
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -3.0   # strong penalty for tipping

        if hasattr(self.rewards, "undesired_contacts"):
            self.rewards.undesired_contacts.weight = -5.0
            
        if hasattr(self.rewards, "lin_vel_z_l2"):
            self.rewards.lin_vel_z_l2.weight = -4.0          # discourage vertical bouncing

        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2.weight = -0.2         # discourage roll/pitch motion

        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.weight = 0.25         # encourage stepping
            sensor_cfg = self.rewards.feet_air_time.params["sensor_cfg"]
            sensor_cfg.body_names = ".*_foot"

        self.events.push_robot = None
        self.curriculum.terrain_levels = None
