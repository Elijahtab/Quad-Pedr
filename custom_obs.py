import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils  # Contains the crucial conversion tools
import isaaclab.envs.mdp as mdp
import math

# Placeholder function (fills with 0's)
def goal_relative_placeholder(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Phase 1-3: Returns zeros [num_envs, 3] (Distance, Sin, Cos)"""
    return torch.rand((env.num_envs, 3), device=env.device)


# Actual function once we have our goals
def goal_relative_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Computes the goal position relative to the robot in the robot's body frame.
    Returns: [Distance, Sin(Relative_Heading), Cos(Relative_Heading)]
    """
    # 1. Get the Robot and Goal Data
    robot = env.scene[asset_cfg.name]

    # goals shape: [num_envs, 2] (X, Y)
    goals = env.command_manager.get_command("pose_command")[:, :2]

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
    relative_heading = math_utils.wrap_to_pi(target_yaw_w - yaw)

    # 6. Build Output Tensor
    # We return Sin/Cos because they are continuous and easier for NN to learn than raw radians
    return torch.stack([
        target_dist,
        torch.sin(relative_heading),
        torch.cos(relative_heading)
    ], dim=-1)


def placeholder_lidar(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns random noise to occupy the 64 inputs reserved for Lidar.
    Range: [0.0, 1.0] to mimic normalized distance.
    """
    return torch.rand((env.num_envs, 360), device=env.device)


def lidar_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,      # kept for API compatibility; we won't use it
    max_distance: float = 10.0,
) -> torch.Tensor:
    ...
    scene = env.scene
    robot = scene["robot"]

    base_xy = robot.data.root_pos_w[:, :2]
    quat = robot.data.root_quat_w
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)

    obstacles = scene["obstacles"]  # RigidObjectCollection
    num_objs = obstacles.num_objects
    num_rays = 360

    if num_objs == 0:
        return torch.ones((env.num_envs, num_rays), device=env.device, dtype=torch.float32)

    # Use collection data API (Isaac Lab 2.3+)
    # Shape: [num_envs, num_objs, 7] (pos xyz, quat wxyz)
    obj_pose = obstacles.data.object_link_pose_w
    obs_xy_world = obj_pose[..., :2]  # [N, M, 2]

    # ... everything else stays the same ...
    rel_world = obs_xy_world - base_xy.unsqueeze(1)   # [N, M, 2]

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx = rel_world[..., 0]
    dy = rel_world[..., 1]

    x_body =  cos_yaw.unsqueeze(-1) * dx + sin_yaw.unsqueeze(-1) * dy
    y_body = -sin_yaw.unsqueeze(-1) * dx + cos_yaw.unsqueeze(-1) * dy
    rel_body = torch.stack((x_body, y_body), dim=-1)   # [N, M, 2]

    angles = torch.linspace(-math.pi, math.pi, steps=num_rays + 1, device=env.device)[:-1]
    dir_x = torch.cos(angles)
    dir_y = torch.sin(angles)
    dirs = torch.stack((dir_x, dir_y), dim=-1)         # [R, 2]

    rel_exp = rel_body[:, None, :, :]   # [N, R=1, M, 2]
    dirs_exp = dirs[None, :, None, :]   # [1, R, 1, 2]

    radius = 0.3
    dot_dc = (dirs_exp * rel_exp).sum(dim=-1)         # [N, R, M]
    c2 = (rel_exp ** 2).sum(dim=-1)                   # [N, R, M]
    disc = dot_dc ** 2 - (c2 - radius ** 2)
    disc_clamped = torch.clamp(disc, min=0.0)
    sqrt_disc = torch.sqrt(disc_clamped)

    t1 = dot_dc - sqrt_disc
    t2 = dot_dc + sqrt_disc

    t_candidates = torch.where(t1 > 0.0, t1, t2)
    valid = (disc >= 0.0) & (t_candidates > 0.0)

    inf = torch.full_like(t_candidates, float("inf"))
    t_candidates = torch.where(valid, t_candidates, inf)

    dist, _ = t_candidates.min(dim=-1)  # [N, R]
    dist = torch.where(torch.isinf(dist),
                       torch.full_like(dist, max_distance),
                       dist)

    dist = torch.clamp(dist, 0.0, max_distance)
    dist_norm = dist / max_distance
    return dist_norm





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


def safe_height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Wrapper around mdp.height_scan that:
      - masks NaN/Inf ray hits from the RayCaster
      - clamps ray z-values into a sane range (e.g. [-2, 2])
    before computing the height scan vector.
    """
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w              # (num_envs, num_rays, 3)

    # 1) Mask non-finite hits
    finite = torch.isfinite(hits).all(dim=-1)  # (num_envs, num_rays)

    if (~finite).any():
        bad_envs = torch.nonzero((~finite).any(dim=1), as_tuple=False).squeeze(-1)
        print("safe_height_scan: masking non-finite ray hits in envs:", bad_envs)

        fixed_hits = hits.clone()
        fixed_hits[~finite] = 0.0             # zero-out any bad rays entirely
    else:
        fixed_hits = hits

    # 2) Clamp z-component into [-2, 2] (or whatever range you want)
    fixed_hits[..., 2] = torch.clamp(fixed_hits[..., 2], -2.0, 2.0)

    # 3) Write back to sensor data so mdp.height_scan sees the cleaned values
    sensor.data.ray_hits_w = fixed_hits

    # 4) Delegate to normal height_scan on cleaned + clamped data
    out = mdp.height_scan(env, sensor_cfg=sensor_cfg)

    # 5) Belt-and-suspenders: ensure obs is finite
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("safe_height_scan: produced non-finite output, this should not happen")
        raise RuntimeError("safe_height_scan output non-finite")

    return out


# def lookahead_hint2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """
#     End of Phase 1 (maybe) (Smart Placeholder):
#     Returns a 'Synthetic Hint' to pre-train the switching logic.
#     - 10% of the time: Returns the Real Goal Vector + Flag 1.0 (Teaches trust)
#     - 90% of the time: Returns Random Noise + Flag 0.0 (Teaches ignore)
#     """
#     # 1. Create the container
#     obs = torch.zeros((env.num_envs, 4),
#                       device=env.device)  # [x, y, z, flag] (Wait, your lookahead is 3 dims right? Dist, Sin, Cos + Flag?)
#
#     # Let's assume your lookahead obs is size 4: [Dist, Sin, Cos, Flag]
#     # If your defined size is 3, modify accordingly.
#     # Based on our previous talk, let's assume standard vector [x, y, z] + flag?
#     # Or [Dist, Angle, Flag]? Let's stick to the vector logic you likely used.
#
#     # 2. Generate the "Noise" case (Standard)
#     noise_vec = torch.randn((env.num_envs, 3), device=env.device)  # Random vector
#     noise_flag = torch.zeros((env.num_envs, 1), device=env.device)  # Flag 0
#
#     # 3. Generate the "Real Goal" case (The Synthetic Signal)
#     # We can reuse the logic from `goal_relative_target` here!
#     # Ideally, refactor that logic into a helper function to avoid code duplication.
#     # For now, let's just grab the goal command directly:
#     goals = env.command_manager.get_command("goal_pos")[:, :2]
#     robot = env.scene[asset_cfg.name]
#     pos = robot.data.root_pos_w[:, :2]
#     quat = robot.data.root_quat_w
#
#     # Calculate Goal Vector (Body Frame) - simplified for brevity
#     target_vec_w = goals - pos
#     roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
#     # Rotate vector by -yaw to get body frame
#     # (Using simple 2D rotation for brevity, real implementation should use standard math utils)
#     cos_n = torch.cos(-yaw)
#     sin_n = torch.sin(-yaw)
#     target_vec_b_x = target_vec_w[:, 0] * cos_n - target_vec_w[:, 1] * sin_n
#     target_vec_b_y = target_vec_w[:, 0] * sin_n + target_vec_w[:, 1] * cos_n
#
#     # Normalize or cap the distance to look like a "Lookahead" (usually 1-2m away)
#     target_dist = torch.norm(torch.stack([target_vec_b_x, target_vec_b_y], dim=-1), dim=-1, keepdim=True)
#     scale = torch.clamp(target_dist, max=1.0) / (target_dist + 1e-5)  # Scale to max 1m length
#
#     real_vec = torch.stack([target_vec_b_x, target_vec_b_y], dim=-1) * scale
#     # We need 3 dims for vector? Or 2? Let's assume 3 dims [x, y, z] for consistency with noise
#     real_vec_3d = torch.cat([real_vec, torch.zeros_like(target_dist)], dim=-1)
#
#     # 4. The Mask (Bernoulli Coin Flip)
#     # 10% chance of being "Real" (1.0), 90% "Noise" (0.0)
#     mask = torch.bernoulli(torch.full((env.num_envs, 1), 0.1, device=env.device))
#
#     # 5. Combine
#     # If mask is 1, use Real Vector. If 0, use Noise.
#     final_vec = mask * real_vec_3d + (1 - mask) * noise_vec
#     final_flag = mask  # 1.0 or 0.0
#
#     # Return combined [Vector, Flag]
#     # Check your `NavObservationsCfg` to see if you defined Lookahead as 3 items or 4.
#     # If it is [Dist, Sin, Cos] + Flag, adapt the math above to return that format.
#
#     # Assuming just returning the tensor:
#     return torch.cat([final_vec, final_flag], dim=-1)
