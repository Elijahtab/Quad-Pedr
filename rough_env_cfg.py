# go2_nav/rough_env_cfg.py

import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg

# Use the same mdp module as the base configs
import isaaclab.envs.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg as BaseSceneCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg as BaseRewardsCfg,
    CommandsCfg as BaseCommandsCfg,
    TerminationsCfg as BaseTerminationsCfg,
)

# Import the Robot Asset
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# IMPORT YOUR CUSTOM MODULES
from . import custom_obs, custom_rewards, commands

##
# Custom Configuration Classes
# We define these here to override the defaults
##

# ======================================================================
# Milestone flag (set this to 1, 2, 3, or 4)
# ======================================================================

TRAINING_MILESTONE = 3
# 1: stand & balance on flat, frozen commands
# 2: forward velocity on flat
# 3: full velocity command tracking (fwd/back/strafe/turn)
# 4: waypoint navigation using goal_pos command + nav rewards


# ======================================================================
# Scene
# ======================================================================
@configclass
class NavSceneCfg(BaseSceneCfg):
    """Scene config for Go2 nav / rough env.

    IMPORTANT: We **do not** override `contact_forces` here so that all
    base terms (base_contact, undesired_contacts, feet_air_time, etc.)
    see the full set of bodies (base, thighs, feet, ...), as expected
    by the upstream RoughEnv config.
    """

    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.25, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        mesh_prim_paths=["/World/ground"],   # FIXED
        debug_vis=False,
    )



# ======================================================================
# Observations
# ======================================================================
@configclass
class NavObservationsCfg(BaseObservationsCfg):
    """Observation config — extends the base with nav-specific terms."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):

        # ----------------------
        # NAV-SPECIFIC OBS TERMS
        # ---------------------
        if TRAINING_MILESTONE <= 3:
            goal_relative: ObsTerm = ObsTerm(
                func=custom_obs.goal_relative_placeholder,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
        else: 
            goal_relative: ObsTerm = ObsTerm(
                func=custom_obs.goal_relative_target,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )

        lidar: ObsTerm = ObsTerm(
            func=custom_obs.placeholder_lidar,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )

        lookahead: ObsTerm = ObsTerm(
            func=custom_obs.lookahead_hint,
        )

        def __post_init__(self):
            # Run base init FIRST (this creates inherited terms,
            # including velocity_commands which uses generated_commands("base_velocity"))
            super().__post_init__()

            # Standard Isaac Lab pattern
            self.concatenate_terms = True
            self.enable_corruption = True

            # IMPORTANT: we DO NOT delete velocity_commands.
            # It will query command_name="base_velocity", which we define
            # (and keep registered) in NavCommandsCfg.

    policy: PolicyCfg = PolicyCfg()


@configclass
class NavRewardsCfg(BaseRewardsCfg):
    """Reward config with milestone-gated nav + locomotion terms."""

    # ---------------------------------------------------------
    # NEW locomotion rewards
    # ---------------------------------------------------------
    lin_vel_track_exp: RewTerm = RewTerm(
        func=custom_rewards.lin_vel_tracking_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )

    ang_vel_track_exp: RewTerm = RewTerm(
        func=custom_rewards.ang_vel_tracking_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )

    base_height_err: RewTerm = RewTerm(
        func=custom_rewards.base_height_error,
        # penalty → negative weight
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # target CoM height in meters for Go2 standing
            "target_height": 0.38,
        },
    )

    pose_similarity: RewTerm = RewTerm(
        func=custom_rewards.pose_similarity,
        weight=-0.1,  # mild penalty; prevents crazy joint configs
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    vertical_vel_pen: RewTerm = RewTerm(
        func=custom_rewards.vertical_vel_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    roll_pitch_pen: RewTerm = RewTerm(
        func=custom_rewards.roll_pitch_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    forward_vel_bonus: RewTerm = RewTerm(
        func=custom_rewards.forward_velocity_bonus,
        weight=2.0,   # tune this (1.0–5.0 is typical)
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


    # ---------------------------------------------------------
    # Existing NAV-specific rewards (for milestone 4)
    # ---------------------------------------------------------
    heading_align: RewTerm = RewTerm(
        func=custom_rewards.heading_alignment,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    )

    vel_to_goal: RewTerm = RewTerm(
        func=custom_rewards.vel_toward_goal,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    )

    progress_to_goal: RewTerm = RewTerm(
        func=custom_rewards.progress_to_goal,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_pos"},
    )
        
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------
        # Disable default velocity-tracking rewards
        # ------------------------------------------
        for name in [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "track_lin_vel_xy",
            "track_ang_vel_z",
        ]:
            if hasattr(self, name):
                setattr(self, name, None)

        # ------------------------------------------
        # Disable nav rewards until milestone 4
        # ------------------------------------------
        if TRAINING_MILESTONE <= 3:
            self.heading_align = None
            self.vel_to_goal = None
            self.progress_to_goal = None

        # ------------------------------------------
        # Milestone 1 (stand)
        # ------------------------------------------
        if TRAINING_MILESTONE == 1:
            self.lin_vel_track_exp.weight = 1.0
            self.ang_vel_track_exp.weight = 0.3
            self.base_height_err.weight = -15.0
            self.roll_pitch_pen.weight = -3.0
            # keep pose_similarity at mild level

        # ------------------------------------------
        # Milestone 2 (forward walking only)
        # ------------------------------------------
        elif TRAINING_MILESTONE == 2:
            self.lin_vel_track_exp.weight = 3.0
            self.ang_vel_track_exp.weight = 0.3
            self.base_height_err.weight = -6.0
            self.roll_pitch_pen.weight = -0.7
            self.pose_similarity.weight = -0.15     
            self.feet_air_time.weight = 0.25
            self.forward_vel_bonus.weight = 1.5

        elif TRAINING_MILESTONE == 0:
            # Stronger forward velocity tracking, moderate turning, mild strafing
            self.lin_vel_track_exp.weight = 3.5
            self.ang_vel_track_exp.weight = 0.4
            self.base_height_err.weight = -8.0
            self.roll_pitch_pen.weight = -1.0   # slightly stronger than M3
            self.pose_similarity = None
            self.feet_air_time.weight = 0.25
            self.backward_drift_pen = RewTerm(
                func=custom_rewards.negative_forward_velocity_penalty,
                weight = -1.0,
                params={"asset_cfg": SceneEntityCfg("robot")}
            )

            self.forward_vel_bonus.weight = 1.2

        # ------------------------------------------
        # Milestone 3 (full locomotion)
        # ------------------------------------------
        elif TRAINING_MILESTONE == 3:
            self.lin_vel_track_exp.weight = 9.0
            self.ang_vel_track_exp.weight = 1.75
            
            self.base_height_err.weight = -10.0  # MUCH stronger anti-collapse

            self.roll_pitch_pen.weight = -0.4    # help counter lean-back collapse

            self.pose_similarity.weight = -0.01  # re-enable small symmetry prior
            
            if hasattr(self, "feet_air_time") and self.feet_air_time:
                self.feet_air_time.weight = 0.2  # reduce exaggerated stepping

            # additional backward reward
            if hasattr(self, "backward_vel_bonus"):
                self.backward_vel_bonus.weight = 2.0

        # ------------------------------------------
        # Milestone 4 (navigation)
        # ------------------------------------------
        elif TRAINING_MILESTONE >= 4:
            # same locomotion tuning as milestone 3
            self.lin_vel_track_exp.weight = 4.0
            self.ang_vel_track_exp.weight = 0.75
            self.base_height_err.weight = -4.0
            self.roll_pitch_pen.weight = -0.5
            self.pose_similarity = None

            if hasattr(self, "feet_air_time") and self.feet_air_time:
                self.feet_air_time.weight = 0.2
            

@configclass
class NavCommandsCfg(BaseCommandsCfg):
    """Command config — milestone-gated velocity + goal_pos."""

    # Crucial: type annotation so configclass registers this field.
    base_velocity: mdp.UniformVelocityCommandCfg = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0, 3.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # Default (milestone 1) = frozen commands (stand & balance)
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

    goal_pos: GoalCommandCfg = GoalCommandCfg(
        resampling_time_range=(100.0, 100.0),
        radius_range=(1.0, 5.0),
        debug_vis=True,
        # MILESTONE 1 SETTINGS: FROZEN (0.0)
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0)
        ),
    )

    def __post_init__(self):
        super().__post_init__()

        # Milestone tuning for base_velocity ranges
        if TRAINING_MILESTONE == 1:
            self.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            self.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.base_velocity.ranges.heading = (0.0, 0.0)

        elif TRAINING_MILESTONE == 2:
            self.base_velocity.heading_command = False 
            self.base_velocity.ranges.heading = (0.0, 0.0)
            self.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
            self.base_velocity.resampling_time_range = (0.3, 0.3)
        
        elif TRAINING_MILESTONE == 25:
            # allow gentle strafe + small backward commands
            self.base_velocity.ranges.lin_vel_x = (-0.1, 0.8)
            self.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
            self.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
            self.base_velocity.resampling_time_range = (0.5, 0.8)


        elif TRAINING_MILESTONE == 3:
            self.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
            self.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
            self.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
            self.base_velocity.ranges.heading = (-3.1415, 3.1415)

        elif TRAINING_MILESTONE >= 4:
            # For waypoint nav, conceptually switch to goal_pos as the
            # main high-level command, but keep base_velocity defined
            # and frozen so velocity_commands obs still has something
            # to read.
            self.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            self.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.base_velocity.ranges.heading = (0.0, 0.0)


# Early termination config (if the bot flips over)
@configclass
class NavTerminationsCfg(BaseTerminationsCfg):
    """Termination config — keep timeout and add orientation constraint."""

    bad_orientation: DoneTerm = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 2.0},
    )

    def __post_init__(self):
        super().__post_init__()
        # We leave base_contact and others intact.
        # They now see the same contact_forces sensor as upstream
        # (which we did NOT override in NavSceneCfg).


##
# The Main Environment Class
##

@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Main env config for Go2 rough / nav with milestone gating."""

    scene: NavSceneCfg = NavSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: NavObservationsCfg = NavObservationsCfg()
    rewards: NavRewardsCfg = NavRewardsCfg()
    commands: NavCommandsCfg = NavCommandsCfg()
    terminations: NavTerminationsCfg = NavTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Fix undesired_contacts to match Go2 naming
        if hasattr(self.rewards, "undesired_contacts") and self.rewards.undesired_contacts:
            term = self.rewards.undesired_contacts
            term.params["sensor_cfg"].body_names = [
                "FL_thigh",
                "FR_thigh",
                "RL_thigh",
                "RR_thigh",
            ]

        # -----------------------------------------------------------------
        # Scene / terrain setup per milestone
        # -----------------------------------------------------------------
        if TRAINING_MILESTONE <= 3:
            # Force flat terrain for early milestones
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.prim_path = "/World/ground"
            self.scene.terrain.collision_group = -1
            self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            )

            # Remove mesh terrain if inherited
            mpaths = getattr(self.scene.terrain, "mesh_prim_paths", None)
            if isinstance(mpaths, list):
                mpaths.clear()  # safest: no mesh prims at all for plane terrain

            self.curriculum.terrain_levels = None
            if hasattr(self.curriculum, "terms"):
                for k in list(self.curriculum.terms.keys()):
                    if "terrain" in k:
                        print(f"[DEBUG] Removing curriculum term: {k}")
                        del self.curriculum.terms[k]
  
        # -----------------------------------------------------------------
        # Initial robot pose
        # -----------------------------------------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        # Action scaling
        self.actions.joint_pos.scale = 0.25  # standard Go2 value

        # -----------------------------------------------------------------
        # Reward tweaks shared across milestones
        # -----------------------------------------------------------------
        if hasattr(self.rewards, "feet_air_time") and self.rewards.feet_air_time:
            sensor_cfg: SceneEntityCfg = self.rewards.feet_air_time.params["sensor_cfg"]
            sensor_cfg.body_names = ".*_foot"
            self.rewards.feet_air_time.weight = 0.5

        if hasattr(self.rewards, "dof_torques_l2") and self.rewards.dof_torques_l2:
            self.rewards.dof_torques_l2.weight = -0.0002

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
