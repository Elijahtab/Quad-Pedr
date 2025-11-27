import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
import isaaclab_tasks.manager_based.locomotion.velocity.config.go2_nav.custom_obs as custom_obs


# ---------------------------------------------------------
# 1. Heading alignment reward
# ---------------------------------------------------------
def heading_alignment(env, asset_cfg: SceneEntityCfg, command_name):
    robot = env.scene[asset_cfg.name]

    # robot yaw
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)

    # direction to goal (world frame)
    pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]
    vec = goals - pos
    goal_yaw = torch.atan2(vec[:, 1], vec[:, 0])

    # relative angle (wrap)
    rel = wrap_to_pi(goal_yaw - yaw)

    # reward large when facing goal (cosine smooth)
    return torch.cos(rel)
    

# ---------------------------------------------------------
# 2. Velocity toward goal (body frame)
# ---------------------------------------------------------
def vel_toward_goal(env, asset_cfg: SceneEntityCfg, command_name):
    robot = env.scene[asset_cfg.name]
    vel_b = robot.data.root_lin_vel_b[:, :2]   # vx, vy in body frame

    # direction to goal in body frame
    goal_b = custom_obs.goal_direction_body(env, asset_cfg, command_name)

    goal_b = goal_b / (goal_b.norm(dim=1, keepdim=True) + 1e-6)

    # dot product = projected velocity toward goal
    reward = (vel_b * goal_b).sum(dim=1)

    # only positive forward
    return torch.clamp(reward, min=0.0)


# ---------------------------------------------------------
# 3. Progress toward goal (world frame)
# ---------------------------------------------------------
def progress_to_goal(env, asset_cfg: SceneEntityCfg, command_name):
    robot = env.scene[asset_cfg.name]

    pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]
    dist_now = torch.norm(goals - pos, dim=-1)

    dist_prev = env.extras.get("pg_dist_prev", None)
    if dist_prev is None:
        env.extras["pg_dist_prev"] = dist_now.clone()
        return torch.zeros_like(dist_now)

    reward = dist_prev - dist_now
    env.extras["pg_dist_prev"] = dist_now.clone()

    return reward

