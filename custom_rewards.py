import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize Y-axis (sideways) velocity."""
    asset = env.scene[asset_cfg.name]
    # asset.data.root_lin_vel_b is [num_envs, 3] (x, y, z)
    # We want index 1 (y)
    return torch.square(asset.data.root_lin_vel_b[:, 1])

def progress_to_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward for moving closer to the goal.
    Formula: (Old_Distance - New_Distance) + (Velocity * Goal_Direction)
    We typically use a "Potential-Based" reward: current_dist - prev_dist.
    """
    # 1. Get the robot and goals
    robot = env.scene[asset_cfg.name]
    
    # Get only the (X, Y) position [num_envs, 2]
    current_pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]
    
    # 2. Calculate current distance to goal
    target_vec = goals - current_pos
    current_dist = torch.norm(target_vec, dim=1)
    
    # 3. Calculate "Previous" distance
    # We approximate previous pos using velocity: prev_pos = curr_pos - (vel * dt)
    dt = env.step_dt
    # root_lin_vel_w is [num_envs, 3], we need [:, :2]
    current_vel = robot.data.root_lin_vel_w[:, :2]
    prev_pos = current_pos - (current_vel * dt)
    
    prev_target_vec = goals - prev_pos
    prev_dist = torch.norm(prev_target_vec, dim=1)
    
    # 4. Reward = Improvement in distance
    # If we got closer, prev_dist > current_dist, so result is positive.
    reward = prev_dist - current_dist
    
    return reward

def arrival_reward(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Sparse reward: +1.0 if within [threshold] meters of the goal.
    """
    robot = env.scene[asset_cfg.name]
    
    current_pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]
    
    distance = torch.norm(goals - current_pos, dim=1)
    
    # Return 1.0 where distance < threshold, else 0.0
    # converting bool to float gives us 1.0 or 0.0
    return (distance < threshold).float()

