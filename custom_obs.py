import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils  # Contains the crucial conversion tools

# Placeholder function (fills with 0's)
def goal_relative_placeholder(env, asset_cfg):
    """
    Phase 1-3: Returns zeros [num_envs, 3] (Distance, Sin, Cos)
    We reserve this seat for later!
    """
    return torch.zeros((env.num_envs, 3), device=env.device)

# Actual function once we have our goals
def goal_relative_target(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Computes the goal position relative to the robot in the robot's body frame.
    Returns: [Distance, Sin(Relative_Heading), Cos(Relative_Heading)]
    """
    # 1. Get the Robot and Goal Data
    robot = env.scene[asset_cfg.name]
    
    # goals shape: [num_envs, 2] (X, Y)
    goals = env.command_manager.get_command("goal_pos")[:, :2]
    
    # robot pos shape: [num_envs, 3] (X, Y, Z) - we only need X,Y
    pos = robot.data.root_pos_w[:, :2]
    
    # robot orientation (Quaternion): [num_envs, 4] (w, x, y, z)
    quat = robot.data.root_quat_w

    # 2. Calculate Vector to Goal (World Frame)
    target_vec = goals - pos
    target_dist = torch.norm(target_vec, dim=1)
    
    # 3. Calculate Target Angle in World Frame (atan2)
    # atan2(y, x) gives the absolute angle of the goal vector
    target_yaw_w = torch.atan2(target_vec[:, 1], target_vec[:, 0])

    # 4. Get Robot Yaw (Rotation around Z) from Quaternion
    # We use the Isaac Lab utility to convert Quat -> Euler (XYZ)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)

    # 5. Calculate Relative Heading (Delta Yaw)
    # "How much do I need to turn to face the goal?"
    relative_heading = target_yaw_w - yaw
    
    # Wrap angle to [-Pi, Pi] so 350 degrees becomes -10 degrees
    relative_heading = math_utils.wrap_to_pi(relative_heading)

    # 6. Build Output Tensor
    # We return Sin/Cos because they are continuous and easier for NN to learn than raw radians
    return torch.stack([
        target_dist, 
        torch.sin(relative_heading), 
        torch.cos(relative_heading)
    ], dim=-1)

def placeholder_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns random noise to occupy the 64 inputs reserved for Lidar.
    Range: [0.0, 1.0] to mimic normalized distance.
    """
    return torch.rand((env.num_envs, 64), device=env.device)

def lidar_scan_normalized(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reads the RayCaster, clips distance, and normalizes 0-1."""
    # 1. Get the sensor object
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. Get raw data (Shape: [num_envs, num_rays])
    # .data.ray_hits_w is the hit position, we want the distance
    # Isaac Lab RayCaster usually provides a 'data.ray_hits_w' 
    # We might need to compute distance manually or use a helper if available.
    # NOTE: Isaac Lab RayCaster has a helper 'data.ray_hits_w'. 
    # Standard implementation often just uses the raw buffer.
    
    # Let's assume we calculate distance from ray_hits_w to sensor_pos_w
    # (Simplified for readability)
    ranges = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    dist = torch.norm(ranges, dim=-1)

    # 3. Normalize (Invert so 1.0 = Close Wall, 0.0 = Far/Safe)
    max_range = 5.0
    dist = torch.clamp(dist, min=0.0, max=max_range)
    normalized_scan = 1.0 - (dist / max_range)
    
    return normalized_scan

def lookahead_hint(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Phase 1: Returns random noise (to keep neurons alive).
    Phase 2: Returns vector to SLAM point.
    """
    # Generate random noise [num_envs, 3]
    # We use env.device to make sure it's on GPU
    noise = torch.rand((env.num_envs, 3), device=env.device) 
    
    # Set the "Availability Flag" (3rd element) to 0 for now
    noise[:, 2] = 0.0 
    
    return noise

