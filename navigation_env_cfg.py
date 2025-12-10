import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.utils.stage import get_current_stage


# Low-level locomotion env for Go2 (velocity / rough terrain)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_nav.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
    NavSceneCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_nav.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_nav import commands, custom_events

# Your nav-specific scene + sensor config (LiDAR, etc.)
from . import custom_obs
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_nav import commands, custom_events, custom_rewards, custom_terminations

# If you keep custom Obs / commands in a nav mdp package, import here.
# For stage 1 we only need standard mdp.* terms, so no custom imports are required.
# import isaaclab_tasks.manager_based.navigation.mdp.custom_obs as custom_obs

# Pre-trained-policy action lives under navigation.mdp in Isaac Lab 2.x
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

# Low-level config instance (ONLY used as a template; we never step it directly)
LOW_LEVEL_ENV_CFG = UnitreeGo2RoughEnvCfg()
# LOW_LEVEL_ENV_CFG = UnitreeGo2FlatEnvCfg()
MODEL_DIR = "/home/elijah/IsaacLab/logs/rsl_rl/unitree_go2_rough/2025-12-01_12-59-50/exported/"
# MODEL_DIR = "/home/elijah/IsaacLab/logs/rsl_rl/unitree_go2_flat/2025-12-01_03-24-08/exported/"
MODEL_NAME = "policy.pt"

# How “busy” the env can be.
GOAL_RADIUS = 5.0
OBSTACLE_RADIUS = 6.0
RESAMPLE_TIME = 15.0


@configclass
class EventCfg:
    """Configuration for events (resets, randomizations)."""

    # Simple flat, no-obstacle reset for stage 1.
    # Robot spawns near origin with almost zero velocity.
    reset_base = EventTerm(
        func=nav_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # One-time static obstacle placement per env at startup
    randomize_obstacles_static = EventTerm(
        func=custom_events.randomize_obstacles_static_startup,
        mode="reset",
        params={
            "inner_radius": 1.0,   # larger than robot spawn radius (0.5)
            "outer_radius": OBSTACLE_RADIUS,
        },
    )

    # BAD BC ITS NOT STATIC: randomize obstacle positions on reset.
    # randomize_obstacles = EventTerm(
    #     func=custom_events.randomize_obstacles,
    #     mode="reset",
    #     params={
    #         # you can tweak these to control sparsity / difficulty:
    #         "spawn_radius": 3.0,
    #         "min_gap_from_robot": 0.7,
    #         "max_active_obstacles": 3,
    #         "obstacle_density": 0.25,
    #     },
    # )


@configclass
class ActionsCfg:
    """High-level action terms (nav policy -> low-level Go2 policy)."""

    # This is the *hierarchical* part:
    # - The RL nav policy outputs "commands" that are fed into a *pre-trained*
    #   low-level Go2 locomotion policy.
    # - That low-level policy outputs joint position targets (same as in the
    #   UnitreeGo2RoughEnvCfg).
    #
    # The raw actions from the NAV policy correspond to the *command inputs*
    # expected by the pre-trained Go2 policy (e.g., [v_x, v_y, ω_z, height, freq, clearance]).
    #
    # low_level_actions / low_level_observations must match what you used when
    # training the low-level Go2 velocity policy.

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path= MODEL_DIR + MODEL_NAME,

        # How often the low-level controller runs relative to physics.
        low_level_decimation=4,
        # These two come *directly* from the low-level velocity env cfg:
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the *navigation* policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the nav policy."""

        # --- Low-dimensional robot state (proprioception) ---

        # Base linear velocity in body frame (3D)
        base_lin_vel = ObsTerm(func=nav_mdp.base_lin_vel)

        # Gravity vector projected into base frame (orientation)
        projected_gravity = ObsTerm(func=nav_mdp.projected_gravity)

        # Optional: base yaw rate if you want it explicitly.
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        # --- Navigation goal ---

        # 2D pose command (x, y, heading) sampled by CommandsCfg.pose_command.
        # The nav policy sees the *desired* goal, and the rewards (below)
        # encourage the robot to minimize position + heading error.
        pose_command = ObsTerm(
            func=nav_mdp.generated_commands,
            params={"command_name": "pose_command"},
        )
        
        goal_relative = ObsTerm(
            func=custom_obs.goal_relative_target,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # LiDAR scan (flattened distances)
        lidar_scan = ObsTerm(
            func=custom_obs.lidar_scan,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "max_distance": 10.0,
            },
            clip=(0.0, 1.0),
        )

        # # Lookahead (placeholder for now)
        lookahead_hint = ObsTerm(
            func=custom_obs.lookahead_placeholder
        )

        # # REAL Lookahead hint: 3 floats (dx, dy, dist_norm), 1 binary (active flag)
        # lookahead_hint = ObsTerm(
        #     func=custom_obs.lookahead_hint,
        #     params={
        #         "map_half_extent": max(GOAL_RADIUS, OBSTACLE_RADIUS) + 1.0,
        #         "grid_resolution": 0.25,
        #         "min_lookahead_distance": 1.0,  # lower bound
        #         "max_lookahead_distance": 4.0,  # upper bound
        #         "obstacle_inflation": 0.35,
        #         "max_astar_steps": 8192,
        #     },
        # )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the navigation policy."""

    # Big negative when the episode terminates (falls, time-out, etc.)
    termination_penalty = RewTerm(func=nav_mdp.is_terminated, weight=-400.0)

    # Coarse position tracking: encourages moving toward the goal in XY.
    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=0.75,
        params={
            "std": 2.0,                # wide basin
            "command_name": "pose_command",
        },
    )

    # Fine position tracking: tightens around the goal as you get closer.
    position_tracking_fine_grained = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=5,
        params={
            "std": 0.2,                # narrow basin
            "command_name": "pose_command",
        },
    )

    # # Penalize heading error so the robot turns to face the goal heading.
    # orientation_tracking = RewTerm(
    #     func=nav_mdp.heading_command_error_abs,
    #     weight=-0.1,
    #     params={"command_name": "pose_command"},
    # )

    # Optional: small alive bonus to encourage longer survival
    # alive_bonus = RewTerm(func=mdp.is_alive, weight=1.0)
    # alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.1)

    # goal_proximity = RewTerm(
    #     func=custom_rewards.goal_proximity_exp,
    #     weight=1.0,  # tune up/down if needed
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "alpha": 1.0,
    #     },
    # )

    # lateral walking penalty (penalizes crab-walk)
    # lateral_penalty = RewTerm(
    #     func=custom_rewards.motion_shape_penalty,
    #     weight=-0.05,  # start here, tune
    #     params={
    #         "speed_deadzone": 0.05,
    #     },
    # )

    # LiDAR-based obstacle clearance reward
    # obstacle_clearance = RewTerm(
    #     func=custom_rewards.obstacle_clearance_reward,
    #     weight=0.05,  # start small; tune later
    #     params={
    #         "sensor_cfg": SceneEntityCfg("lidar"),
    #         "min_clearance": 0.25,
    #         "max_distance": 1.5,
    #     },
    # )



@configclass
class CommandsCfg:
    """Command terms for the MDP (goal specification)."""

    # 2D target pose in world frame: the "goal" that the robot should reach.
    pose_command = nav_mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(RESAMPLE_TIME, RESAMPLE_TIME),  # goal fixed for each episode here
        debug_vis=True,
        # Stage 1: flat world with no obstacles, moderate workspace around origin.
        ranges=nav_mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-GOAL_RADIUS, GOAL_RADIUS),
            pos_y=(-GOAL_RADIUS, GOAL_RADIUS),
            heading=(-math.pi, math.pi),
        ),
    )

    gait_params: commands.GaitParamCommandCfg = commands.GaitParamCommandCfg() 


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode ends after fixed horizon (handled via env.episode_length_s).
    time_out = DoneTerm(func=nav_mdp.time_out, time_out=True)

    # End episode if base collides with ground (fall / crash).
    base_contact = DoneTerm(
        func=nav_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "base",
                "Head_upper",
                "Head_lower",
                "FL_hip", "FL_thigh", "FL_calf",
                "FR_hip", "FR_thigh", "FR_calf",
                "RL_hip", "RL_thigh", "RL_calf",
                "RR_hip", "RR_thigh", "RR_calf",
            ]),
            "threshold": 1.0,
        },
    )

    # # Lidar-based Obstacle collision
    # # (not needed anymore, but we can always bring it back if ghosting starts happening, to be safe)
    # obstacle_collision = DoneTerm(
    #     func=custom_terminations.obstacle_collision_from_lidar,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("lidar"),
    #         "min_distance": 0.2,
    #         "max_distance": 10.0,
    #     },
    # )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 navigation environment (stage 1)."""

    # Scene: reuse your nav-specific scene (Go2 + terrain + LiDAR)
    scene: NavSceneCfg = NavSceneCfg(
        num_envs=LOW_LEVEL_ENV_CFG.scene.num_envs,
        env_spacing=LOW_LEVEL_ENV_CFG.scene.env_spacing,
        robot=LOW_LEVEL_ENV_CFG.scene.robot,
    )

    # Managers
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization to keep nav env consistent with low-level env."""
        # Flatten the terrain taken from NavSceneCfg (which came from rough_env_cfg)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Use the same physics dt as the low-level env.
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt

        # Render every low-level control step (nice for debugging).
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation

        # One nav step = N low-level steps. Here we keep the same pattern as
        # the ANYmal navigation env: nav policy updates 10x slower.
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10

        # Fix patch buffer overflow
        # Default is 5 * 2**15, but we need at least 262144
        # 10 * 2**15 = 327680 (> 262144)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16

        # Episode length is tied to the goal resampling window.
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        # Keep sensor update periods consistent with the low-level controller.
        if getattr(self.scene, "lidar", None) is not None:
            self.scene.lidar.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # print("\n\n----------------")
        # if getattr(self.scene, "obstacles", None) is not None:
        #     print("amount of obstacles:", len(self.scene.obstacles.rigid_objects.keys()))
        # else:
        #     print("NO OBSTACLES TYPE IN THE SCENE")
        # print("----------------\n\n")


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    """Smaller debug / play variant."""

    def __post_init__(self) -> None:
        # Call parent first
        super().__post_init__()

        # Fewer envs for interactive runs.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable observation corruption / noise if your low-level policy
        # used noise; you can also disable at low-level if needed.
        if hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = False