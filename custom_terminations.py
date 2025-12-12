# custom_terminations.py
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils
import isaaclab.envs.mdp as mdp
from . import custom_obs
import math

def obstacle_collision_from_lidar(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,   # not actually used, for API symmetry
    min_distance: float = 0.15,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """
    Returns a bool tensor [num_envs] that is True if any ray is closer
    than `min_distance` to an obstacle, using analytic LiDAR.
    """
    d_norm = custom_obs.lidar_scan(env, sensor_cfg, max_distance=max_distance)
    d = d_norm * max_distance             # meters
    min_d, _ = d.min(dim=1)               # [num_envs]
    return min_d < min_distance


# def reached_goal(
#     env: ManagerBasedRLEnv,
#     command_name: str = "pose_command",
#     pos_tol: float = 0.2,
#     yaw_tol: float = 0.3,
# ) -> torch.Tensor:
#     """
#     Returns a bool tensor [num_envs] that is True if the robot is
#     'close enough' to its goal position AND heading.

#     - pos_tol: position tolerance (meters)
#     - yaw_tol: heading tolerance (radians)
#     """
#     # Reuse nav_mdp utilities instead of redoing math if you like,
#     # but simplest is to pull from the same data they use:

#     scene = env.scene
#     robot = scene["robot"]

#     # current base pose
#     base_pos = robot.data.root_pos_w[:, :2]  # [N, 2]
#     base_yaw = nav_mdp.base_yaw(env)         # [N], helper that extracts yaw

#     # desired command (x, y, heading) in world frame
#     cmd = nav_mdp.generated_commands(env, command_name=command_name)  # [N, 3]
#     goal_xy = cmd[:, :2]
#     goal_yaw = cmd[:, 2]

#     pos_err = torch.linalg.norm(base_pos - goal_xy, dim=1)
#     yaw_err = torch.abs(nav_mdp.wrap_to_pi(base_yaw - goal_yaw))

#     return (pos_err < pos_tol) & (yaw_err < yaw_tol)
