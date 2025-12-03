# go2_nav/rough_env_cfg.py

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg

from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

# Use the same mdp module as the base configs
import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards as vel_mdp

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

# IMPORT CUSTOM MODULES
from . import custom_obs, custom_rewards, commands
import math

# How “busy” the env can be.
MAX_OBSTACLES_PER_ENV = 3        # bump this later to 5, 8, ...
OBSTACLE_SIZE = (0.4, 0.4, 0.4)  # (x, y, z) in meters
OBSTACLE_HEIGHT = OBSTACLE_SIZE[2]
NUM_OBSTACLES = 4  # <-- bump this up later when your nav policy gets better






@configclass
class NavCommandsCfg(BaseCommandsCfg):
    """Command config — milestone-gated velocity + goal_pos."""

    # Crucial: type annotation so configclass registers this field.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), 
            lin_vel_y=(-1.0, 1.0), 
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi)
        ),
    )
    
    # NEW GAIT COMMANDS (Size: 3) [Height, Freq, Clearance]
    gait_params: commands.GaitParamCommandCfg = commands.GaitParamCommandCfg()


@configclass
class NavRewardsCfg(BaseRewardsCfg):
    """Reward config with milestone-gated nav + locomotion terms."""
    #base_height_err = None
    base_height_err = RewTerm(
        func=custom_rewards.base_height_l2_safe,
        weight=-1.5,
        params={
            "target_height": 0.38,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    track_foot_clearance = RewTerm(
        func=custom_rewards.track_feet_clearance_exp,  # use the ray version
        weight=0.04,  # low-ish so it doesn't dominate
        params={
            "command_name": "gait_params",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "sigma": 0.01,
            "swing_vel_thresh": 0.1,
        },
    )


    # Clearance: "Lift feet when I say lift"
    # track_foot_clearance = RewTerm(
    #     func=custom_rewards.track_feet_clearance_exp,
    #     weight=0.4,
    #     params={
    #         "command_name": "gait_params",
    #         "asset_cfg":  SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "sigma": 0.01,
    #         "swing_vel_thresh": 0.1,
    #     },
    # )

    # CAN WE DO THIS?
    # TRACK HEIGHT (Dynamic)
    # track_cmd_height = RewTerm(
    #     func=custom_rewards.track_commanded_height_exp,
    #     weight=1.0,  # or whatever
    #     params={
    #         "command_name": "gait_params",
    #         "asset_cfg":  SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     },
    # )
    # roll_pitch_pen: RewTerm = RewTerm(
    #     func=custom_rewards.roll_pitch_penalty,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


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
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # Nice visual floor
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg(
    #         size=(50.0, 50.0),       # make it big enough for all envs
    #         color=(0.25, 0.25, 0.25),# mid-grey instead of black
    #         visible=True,
    #     ),
    # )

    obstacles: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"obstacle_{i}": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/NavObstacle_" + str(i),

                spawn=sim_utils.CuboidCfg(
                    size=(0.4, 0.4, 0.4),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.1, 0.1),
                        metallic=0.0,
                        roughness=0.9,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,    # <- static rigid object
                        disable_gravity=True,      # doesn’t sag or fall
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                        contact_offset=0.02,
                        rest_offset=0.0,
                    ),
                ),

                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.3),
                    # rot=(0.0, 0.0, 0.0, 1.0),
                ),

                # this doesn't exist on RigidObjectCfg
                # activate_contact_sensors=True,
            )
            for i in range(4)
        }
    )


# ======================================================================
# Observations
# ======================================================================
@configclass
class NavObservationsCfg(BaseObservationsCfg):
    """Observation config — extends the base with nav-specific terms."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):

        # lidar: ObsTerm = ObsTerm(
        #     func=custom_obs.placeholder_lidar,
        #     params={"sensor_cfg": SceneEntityCfg("lidar")},
        # )

        # 1. Velocity Inputs (Size 3)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )

        # gait commands (dim 3: height, freq, clearance)
        gait_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "gait_params"},
        )
        def __post_init__(self):
            # Run base init FIRST (this creates inherited terms,
            # including velocity_commands which uses generated_commands("base_velocity"))
            super().__post_init__()

            # Standard Isaac Lab pattern
            self.concatenate_terms = True
            self.enable_corruption = True
        

    policy: PolicyCfg = PolicyCfg()


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    scene: NavSceneCfg = NavSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: NavObservationsCfg = NavObservationsCfg()
    rewards: NavRewardsCfg = NavRewardsCfg()
    commands: NavCommandsCfg = NavCommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25 # standard Go2 value

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 1
        #self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 3
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.flat_orientation_l2.weight = -1.75
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.track_foot_clearance = None
        self.rewards.feet_slide = RewTerm(
            func=vel_mdp.feet_slide,
            weight=-0.325,  # good starting point; can go to -0.25 later
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        
        if self.rewards.undesired_contacts is not None:
            # Point it at the contact sensor
            self.rewards.undesired_contacts.params["sensor_cfg"].name = "contact_forces"

            # Hard-code the bodies we *don't* want touching the ground.
            # Feet are allowed, everything else is undesired.
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
                "base",
                "Head_upper",
                "Head_lower",
                "FL_hip", "FL_thigh", "FL_calf",
                "FR_hip", "FR_thigh", "FR_calf",
                "RL_hip", "RL_thigh", "RL_calf",
                "RR_hip", "RR_thigh", "RR_calf",
            ]
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        