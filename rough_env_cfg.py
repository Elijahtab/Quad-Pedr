# go2_nav/rough_env_cfg.py

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils

# Import Parent Configs
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Import the Robot Asset
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# IMPORT YOUR CUSTOM MODULES
from . import custom_obs, custom_rewards, commands

##
# Custom Configuration Classes
# We define these here to override the defaults
##

@configclass
class NavObservationsCfg(LocomotionVelocityRoughEnvCfg.ObservationsCfg):
    """We extend the parent observations to add our new sensors."""
    
    @configclass
    class PolicyCfg(LocomotionVelocityRoughEnvCfg.ObservationsCfg.PolicyCfg):
        # 1. Add Contact Forces (Feeling the ground)
        contact_forces = mdp.ObservationTermCfg(
            func=mdp.contact_sensor_data,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"), 
                "data_type": "force"
            },
            clip=(-100.0, 100.0),
            scale=0.01,
        )

        # 2. Add Relative Goal (Placeholder for now)
        goal_relative = mdp.ObservationTermCfg(
            func=custom_obs.goal_relative_placeholder,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # MILESTONE 4: (SWAP ABOVE WITH THIS ONE)
        # goal_relative = mdp.ObservationTermCfg(
        #     func=custom_obs.goal_relative_target, # <--- USE REAL TARGET FUNC
        #     params={"asset_cfg": SceneEntityCfg("robot")}
        # )

        # 3. Add Lidar (Placeholder for now)
        lidar = mdp.ObservationTermCfg(
            func=custom_obs.placeholder_lidar
        )

        # 4. Add Lookahead (Placeholder for now)
        lookahead = mdp.ObservationTermCfg(
            func=custom_obs.lookahead_hint
        )

    # Assign the updated policy group
    policy: PolicyCfg = PolicyCfg()


@configclass
class NavRewardsCfg(LocomotionVelocityRoughEnvCfg.RewardsCfg):
    """We add our custom rewards here."""
    
    # Add Anti-Drift (Penalize sideways movement)
    lin_vel_y_penalty = mdp.RewardTermCfg(
        func=custom_rewards.lin_vel_y_l2, 
        weight=-2.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Add Flat Orientation Reward
    flat_orientation_l2 = mdp.RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # MILESTONE 4: Add Progress to Goal Rewards
    # progress_to_goal = mdp.RewardTermCfg(
    #     func=custom_rewards.progress_to_goal,
    #     weight=1.0,
    #     params={
    #         "command_name": "goal_pos", # MUST MATCH COMMAND NAME
    #         "asset_cfg": SceneEntityCfg("robot")
    #     }
    # )
    # arrival_bonus = mdp.RewardTermCfg(
    #     func=custom_rewards.arrival_reward,
    #     weight=100.0,
    #     params={
    #         "command_name": "goal_pos",
    #         "threshold": 0.5,
    #         "asset_cfg": SceneEntityCfg("robot")
    #     }
    # )


@configclass
class NavCommandsCfg(LocomotionVelocityRoughEnvCfg.CommandsCfg):
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
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0)
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


# Early termination config (if the bot flips over)
@configclass
class NavTerminationsCfg(LocomotionVelocityRoughEnvCfg.TerminationsCfg):
    """We extend terminations to add the flip-over check."""
    
    # 1. Keep Time Out (Inherited, but good to be explicit)
    time_out = mdp.TerminationTermCfg(func=mdp.time_out, time_out=True)
    
    # 2. Base Contact (Fall detection)
    base_contact = mdp.TerminationTermCfg(
        func=mdp.illegal_contact,
        # NOTE: Go2 body name is usually "base"
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0}
    )

    bad_orientation = mdp.TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.5}
    )


##
# The Main Environment Class
##

@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    # Override with our custom configs
    observations: NavObservationsCfg = NavObservationsCfg()
    rewards: NavRewardsCfg = NavRewardsCfg()
    commands: NavCommandsCfg = NavCommandsCfg()
    terminations: NavTerminationsCfg = NavTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 1. ROBOT SETUP
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.25 # Go2 Standard

        # 2. ADD SENSORS TO SCENE
        # self.scene.height_scanner = None # RE-ENABLED for Milestone 2.5 (Rough Terrain)
        # Lidar (RayCaster)
        # Target: 360 degrees, 64 rays -> Res = 5.625 deg
        self.scene.lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.25, 0.0, 0.1)), # Chin mount
            attach_yaw_only=True,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(-180.0, 180.0), # 360 deg
                horizontal_res=1.0,                   # 1 deg resolution -> 360 rays
            ),
            debug_vis=False, # Toggle this to True to show visualization
        )
        # Ensure contact forces are tracking air time
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot", 
            history_length=3, 
            track_air_time=True
        )

        # 3. TERRAIN SETUP (MILESTONE 1: FLAT)
        # COMMENT THIS OUT AFTER MILESTONE 2
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane", # FORCE FLAT GROUND
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        # COMMENT ABOVE OUT AFTER MILESTONE 2

        # 4. REWARD TUNING (Go2 Specifics)
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.5 # Increased for stability
        # self.rewards.feet_air_time.params["command_name"] = "goal_pos" # MILESTONE 4: UNCOMMENT THIS LINE
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        
        # Remove rewards that might cause issues early on
        self.rewards.undesired_contacts = None
        self.events.push_robot = None # Disable pushing for Milestone 1

        # 5. DOMAIN RANDOMIZATION (if we want it)
        # We will need this for Sim-to-Real
        # self.events.randomize_mass = mdp.EventTermCfg(
        #     func=mdp.randomize_rigid_body_mass,
        #     mode="startup",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        #         "mass_distribution_params": (-1.0, 1.0), # +/- 1kg
        #         "operation": "add",
        #     },
        # )
