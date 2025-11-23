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
    return torch.rand((env.num_envs, 360), device=env.device)

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
    Phase 1: Returns random noise with a RARE availability flag.
    Strategy: 90% Off / 10% On.
    Why: Prevents the network from learning to "hate" this input.
    """
    # 1. Random Vector (The Garbage)
    noise_vec = torch.randn((env.num_envs, 3), device=env.device)
    
    # 2. Bernoulli Mask (The Coin Flip)
    # 0.1 means 10% chance of being 1.0, 90% chance of being 0.0
    flag = torch.bernoulli(torch.full((env.num_envs, 1), 0.1, device=env.device))
    
    # 3. Combine
    return torch.cat([noise_vec, flag], dim=-1)


def lookahead_hint2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    End of Phase 1 (maybe) (Smart Placeholder): 
    Returns a 'Synthetic Hint' to pre-train the switching logic.
    - 10% of the time: Returns the Real Goal Vector + Flag 1.0 (Teaches trust)
    - 90% of the time: Returns Random Noise + Flag 0.0 (Teaches ignore)
    """
    # 1. Create the container
    obs = torch.zeros((env.num_envs, 4), device=env.device) # [x, y, z, flag] (Wait, your lookahead is 3 dims right? Dist, Sin, Cos + Flag?)
    
    # Let's assume your lookahead obs is size 4: [Dist, Sin, Cos, Flag]
    # If your defined size is 3, modify accordingly. 
    # Based on our previous talk, let's assume standard vector [x, y, z] + flag? 
    # Or [Dist, Angle, Flag]? Let's stick to the vector logic you likely used.
    
    # 2. Generate the "Noise" case (Standard)
    noise_vec = torch.randn((env.num_envs, 3), device=env.device) # Random vector
    noise_flag = torch.zeros((env.num_envs, 1), device=env.device) # Flag 0
    
    # 3. Generate the "Real Goal" case (The Synthetic Signal)
    # We can reuse the logic from `goal_relative_target` here!
    # Ideally, refactor that logic into a helper function to avoid code duplication.
    # For now, let's just grab the goal command directly:
    goals = env.command_manager.get_command("goal_pos")[:, :2]
    robot = env.scene[asset_cfg.name]
    pos = robot.data.root_pos_w[:, :2]
    quat = robot.data.root_quat_w
    
    # Calculate Goal Vector (Body Frame) - simplified for brevity
    target_vec_w = goals - pos
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    # Rotate vector by -yaw to get body frame
    # (Using simple 2D rotation for brevity, real implementation should use standard math utils)
    cos_n = torch.cos(-yaw)
    sin_n = torch.sin(-yaw)
    target_vec_b_x = target_vec_w[:, 0] * cos_n - target_vec_w[:, 1] * sin_n
    target_vec_b_y = target_vec_w[:, 0] * sin_n + target_vec_w[:, 1] * cos_n
    
    # Normalize or cap the distance to look like a "Lookahead" (usually 1-2m away)
    target_dist = torch.norm(torch.stack([target_vec_b_x, target_vec_b_y], dim=-1), dim=-1, keepdim=True)
    scale = torch.clamp(target_dist, max=1.0) / (target_dist + 1e-5) # Scale to max 1m length
    
    real_vec = torch.stack([target_vec_b_x, target_vec_b_y], dim=-1) * scale
    # We need 3 dims for vector? Or 2? Let's assume 3 dims [x, y, z] for consistency with noise
    real_vec_3d = torch.cat([real_vec, torch.zeros_like(target_dist)], dim=-1) 
    
    # 4. The Mask (Bernoulli Coin Flip)
    # 10% chance of being "Real" (1.0), 90% "Noise" (0.0)
    mask = torch.bernoulli(torch.full((env.num_envs, 1), 0.1, device=env.device))
    
    # 5. Combine
    # If mask is 1, use Real Vector. If 0, use Noise.
    final_vec = mask * real_vec_3d + (1 - mask) * noise_vec
    final_flag = mask # 1.0 or 0.0
    
    # Return combined [Vector, Flag]
    # Check your `NavObservationsCfg` to see if you defined Lookahead as 3 items or 4.
    # If it is [Dist, Sin, Cos] + Flag, adapt the math above to return that format.
    
    # Assuming just returning the tensor:
    return torch.cat([final_vec, final_flag], dim=-1)

