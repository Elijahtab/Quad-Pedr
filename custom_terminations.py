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
