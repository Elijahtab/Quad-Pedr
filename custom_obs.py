import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

# Correct math imports from isaaclab.utils.math
from isaaclab.utils.math import (
    wrap_to_pi,
    euler_xyz_from_quat,
    matrix_from_quat,
)

# ======================================================================
# Placeholder goal observation
# ======================================================================
def goal_relative_placeholder(env, asset_cfg):
    """Return zeros so the obs space shape is stable."""
    return torch.zeros((env.num_envs, 3), device=env.device)


# ======================================================================
# Real goal-relative observation
# ======================================================================
def goal_relative_target(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Computes the goal position relative to the robot in the robot's body frame.
    Returns: [distance, sin(relative_heading), cos(relative_heading)]
    """
    robot = env.scene[asset_cfg.name]

    # Goals: [num_envs, 2]
    goals = env.command_manager.get_command("goal_pos")[:, :2]

    # Robot XY position
    pos = robot.data.root_pos_w[:, :2]

    # Robot quaternion
    quat = robot.data.root_quat_w

    # World-frame vector to goal
    vec_w = goals - pos
    target_dist = torch.norm(vec_w, dim=1)

    # Goal yaw direction in world frame
    target_yaw_world = torch.atan2(vec_w[:, 1], vec_w[:, 0])

    # Robot orientation → yaw
    roll, pitch, yaw = euler_xyz_from_quat(quat)

    # Heading error
    rel_heading = wrap_to_pi(target_yaw_world - yaw)

    return torch.stack([
        target_dist,
        torch.sin(rel_heading),
        torch.cos(rel_heading)
    ], dim=-1)


# ======================================================================
# Lidar placeholders
# ======================================================================
def placeholder_lidar(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    num_envs = env.num_envs
    device = env.device
    num_measurements = 360
    return torch.zeros((num_envs, num_measurements), device=device)


def lidar_scan_normalized(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]

    ranges = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    dist = torch.norm(ranges, dim=-1)

    max_range = 5.0
    dist = torch.clamp(dist, 0.0, max_range)
    return 1.0 - (dist / max_range)


# ======================================================================
# Lookahead experiments
# ======================================================================
def lookahead_hint(env: ManagerBasedRLEnv) -> torch.Tensor:
    noise_vec = torch.randn((env.num_envs, 3), device=env.device)
    flag = torch.bernoulli(torch.full((env.num_envs, 1), 0.1, device=env.device))
    return torch.cat([noise_vec, flag], dim=-1)


def lookahead_hint2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    goals = env.command_manager.get_command("goal_pos")[:, :2]
    robot = env.scene[asset_cfg.name]

    pos = robot.data.root_pos_w[:, :2]
    quat = robot.data.root_quat_w

    target_vec_w = goals - pos

    roll, pitch, yaw = euler_xyz_from_quat(quat)

    cos_y = torch.cos(-yaw)
    sin_y = torch.sin(-yaw)

    # Rotate into body frame (2D)
    target_vec_b_x = target_vec_w[:, 0] * cos_y - target_vec_w[:, 1] * sin_y
    target_vec_b_y = target_vec_w[:, 0] * sin_y + target_vec_w[:, 1] * cos_y

    target_dist = torch.norm(
        torch.stack([target_vec_b_x, target_vec_b_y], dim=-1),
        dim=-1,
        keepdim=True
    )

    scale = torch.clamp(target_dist, max=1.0) / (target_dist + 1e-5)

    real_vec = torch.stack([target_vec_b_x, target_vec_b_y], dim=-1) * scale
    real_vec_3d = torch.cat([real_vec, torch.zeros_like(target_dist)], dim=-1)

    noise_vec = torch.randn((env.num_envs, 3), device=env.device)
    mask = torch.bernoulli(torch.full((env.num_envs, 1), 0.1, device=env.device))

    final_vec = mask * real_vec_3d + (1 - mask) * noise_vec
    final_flag = mask

    return torch.cat([final_vec, final_flag], dim=-1)


# ======================================================================
# Goal direction in body frame
# ======================================================================
def goal_direction_body(env, asset_cfg, command_name):
    robot = env.scene[asset_cfg.name]

    pos_w = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]

    vec_w = goals - pos_w   # [N,2]

    # 3×3 rotation matrices
    rot_mats = matrix_from_quat(robot.data.root_quat_w)

    # Extract upper 2×2
    rot_2x2 = rot_mats[:, :2, :2]

    # World → Body transform (R^T)
    vec_b = torch.einsum("bij,bj->bi", rot_2x2.transpose(1, 2), vec_w)

    return vec_b
