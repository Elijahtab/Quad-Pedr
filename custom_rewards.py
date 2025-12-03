import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_apply_inverse
import isaaclab_tasks.manager_based.locomotion.velocity.config.go2_nav.custom_obs as custom_obs
import math
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils
import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

def motion_shape_penalty(
    env: ManagerBasedRLEnv,
    speed_deadzone: float = 0.05,
    lateral_weight: float = 1.0,
    backward_weight: float = 1.0,
) -> torch.Tensor:
    """
    Pure penalty shaping for 'bad' motion patterns:

    - 0 when nearly still.
    - 0 when moving purely forward in body frame.
    - < 0 when moving sideways and/or backwards.

    Returns: [num_envs] tensor of rewards in [- (lateral_weight+backward_weight), 0].
    """
    base_lin_vel = mdp.base_lin_vel(env)   # [N, 3], body frame
    vx = base_lin_vel[:, 0]                # forward/back
    vy = base_lin_vel[:, 1]                # lateral

    # Total horizontal speed
    speed = torch.sqrt(vx * vx + vy * vy)  # [N]
    eps = 1e-6
    speed_safe = speed + eps

    # Fraction of speed that is lateral: |vy| / speed
    frac_lateral = torch.abs(vy) / speed_safe               # [N] in [0, 1]

    # Fraction that points backwards: max( -vx / speed, 0 )
    backward_frac = torch.clamp(-vx / speed_safe, min=0.0)  # [N] in [0, 1]

    # Only penalize if actually moving
    moving_mask = (speed > speed_deadzone).float()          # [N]

    penalty = moving_mask * (
        lateral_weight * frac_lateral +
        backward_weight * backward_frac
    )  # [N], in [0, lateral_weight + backward_weight]

    return penalty


def track_commanded_height_flat(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    nominal_height: float = 0.38,
    min_offset: float = 0.02,
    std: float = 0.05,
):
    """
    FLAT ENV VERSION: reward matching commanded base height using ONLY base_z.

    - Commands come from "gait_params":
        gait_params[:, 0] = height offset (m) relative to nominal_height.
    - Terrain is a plane at z=0, so base_z == height above ground.
    - We only give a reward when |height_offset| >= min_offset
      (i.e., "this timestep is actually a height task").

    Returns: [num_envs] reward in [0, 1].
    """
    robot = env.scene[asset_cfg.name]

    # --- 1. Commanded height offset from gait_params ---
    gait_cmd = env.command_manager.get_command(command_name)    # (num_envs, 3)
    height_offset = gait_cmd[:, 0]                              # (num_envs,)

    # When offset is tiny, treat it as "no height task"
    active = (torch.abs(height_offset) >= min_offset).float()   # (num_envs,)

    # --- 2. Target height above flat ground ---
    target_height = nominal_height + height_offset              # (num_envs,)

    # On flat plane, ground_z ~ 0, so base_z is the height above ground.
    base_z = robot.data.root_pos_w[:, 2]                        # (num_envs,)

    # --- 3. Error + exponential kernel ---
    error = base_z - target_height                              # (num_envs,)
    err2 = error * error

    # exp(-err^2 / std^2)  → 1 when on target, decays as we move away
    result = torch.exp(-err2 / (std ** 2))                      # (num_envs,)

    # Only count reward when a meaningful height command is active
    reward = result * active

    # Safety checks
    if torch.isnan(reward).any() or torch.isinf(reward).any():
        print("NaN/Inf in track_commanded_height_flat")
        raise RuntimeError("NaN/Inf in track_commanded_height_flat")

    return reward

def base_height_l2_flat(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg,
):
    """
    Flat-ground version:
    Penalizes being BELOW target_height using world-frame base.z.
    No sensor / RayCaster needed.
    """
    robot = env.scene[asset_cfg.name]

    base_z = robot.data.root_pos_w[:, 2]    # [num_envs]
    # positive diff means we're too low
    diff = target_height - base_z
    below = torch.clamp(diff, min=0.0)
    rew = below * below                      # >= 0

    if torch.isnan(rew).any() or torch.isinf(rew).any():
        print("NaN/Inf in base_height_l2_flat")
        raise RuntimeError("NaN/Inf in base_height_l2_flat")

    return rew


def ground_height_from_scanner_safe(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns per-env ground height from a RayCaster, with:
    - NaN/Inf masked
    - z clamped to a sane band
    Guaranteed finite output of shape [num_envs].
    """
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w  # (num_envs, num_rays, 3)

    # Mask non-finite
    finite = torch.isfinite(hits)
    if (~finite).any():
        bad_envs = torch.nonzero((~finite).any(dim=1), as_tuple=False).squeeze(-1)
        print("ground_height_from_scanner_safe: masking non-finite ray hits in envs:", bad_envs)

        fixed_hits = hits.clone()
        fixed_hits[~finite] = 0.0
        hits = fixed_hits

    # Clamp Z to a plausible band to avoid crazy spikes
    hits_z = hits[..., 2].clamp(min=-2.0, max=2.0)

    # If some rays are zeroed out, average still stays finite
    ground_z = hits_z.mean(dim=1)  # (num_envs,)

    if torch.isnan(ground_z).any() or torch.isinf(ground_z).any():
        print("ground_height_from_scanner_safe: still non-finite ground_z")
        raise RuntimeError("ground_height_from_scanner_safe produced non-finite ground_z")

    # Optionally write back sanitized hits if you want everyone else to see them:
    sensor.data.ray_hits_w[..., 2] = hits_z

    return ground_z

def base_height_l2_safe(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
):
    """
    Same idea as mdp.base_height_l2, but using ground_height_from_scanner_safe.
    Penalizes (base_height - target_height)^2.
    """
    robot = env.scene[asset_cfg.name]

    ground_z = ground_height_from_scanner_safe(env, sensor_cfg)     # [num_envs]
    current_height = robot.data.root_pos_w[:, 2] - ground_z         # [num_envs]

    error = current_height - target_height
    rew = error * error

    if torch.isnan(rew).any() or torch.isinf(rew).any():
        print("NaN/Inf in base_height_l2_safe")
        raise RuntimeError("NaN/Inf in base_height_l2_safe")

    return rew

def goal_proximity_exp(
    env,
    asset_cfg: SceneEntityCfg,
    alpha: float = 1.0,
):
    """
    Reward for being close to the goal based on goal_relative_target.

    goal_relative_target returns [distance, sin(d_heading), cos(d_heading)].

    We use an exponential: reward = exp(-alpha * distance)
    so:
        - distance = 0   → reward = 1
        - distance ~ 3m  → reward ~ exp(-3) ≈ 0.05 (if alpha=1)
    """
    # [num_envs, 3] = [dist, sin(d_heading), cos(d_heading)]
    rel = custom_obs.goal_relative_target(env, asset_cfg)
    dist = rel[:, 0]  # (num_envs,)

    # Stable exponential reward in [0, 1]
    dist_clamped = torch.clamp(dist, min=0.0)
    reward = torch.exp(-alpha * dist_clamped)

    # Safety
    if torch.isnan(reward).any() or torch.isinf(reward).any():
        raise RuntimeError("goal_proximity_exp produced non-finite values")

    return reward

def obstacle_clearance_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    min_clearance: float = 0.5,
    max_distance: float = 10.0,
):
    """
    Reward staying away from obstacles based on analytic LiDAR.

    - Uses custom_obs.lidar_scan: normalized distances in [0, 1]
      where 1.0 = no hit within max_distance, 0.0 = on top of obstacle.

    We compute, per env:
        closeness = max over rays of clamp((min_clearance - d) / min_clearance, 0, 1)
    and then:
        reward = 1 - closeness

    So:
        - All obstacles farther than min_clearance → reward = 1
        - Something right up against the robot     → reward ~ 0
    """
    # [num_envs, num_rays] in [0,1]
    d_norm = custom_obs.lidar_scan(env, sensor_cfg, max_distance=max_distance)

    # Convert back to meters
    d = d_norm * max_distance  # (num_envs, num_rays)

    # 1 when at 0m, 0 when >= min_clearance
    closeness = torch.clamp((min_clearance - d) / min_clearance, min=0.0, max=1.0)

    # Worst-case closeness per env
    k = 10
    topk_vals, _ = torch.topk(closeness, k=k, dim=1)
    worst = topk_vals.mean(dim=1)
    
    # worst = closeness.max(dim=1).values  # (num_envs,)

    # Reward is high when worst closeness is low (i.e., far from obstacles)
    reward = 1.0 - worst

    # Safety check
    if torch.isnan(reward).any() or torch.isinf(reward).any():
        raise RuntimeError("obstacle_clearance_reward produced non-finite values")

    return reward


def track_feet_clearance_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sigma: float = 0.01,
    swing_vel_thresh: float = 0.1,
    imbalance_coef: float = 0.8,
):
    """
    Real foot-clearance reward using RayCaster terrain height.

    - Command: gait_params[:, 2]  = target clearance (m)
    - Clearance: foot_z - ground_z_from_scanner
    - Penalize only when below target, only on swinging feet
    - Aggregation: reward = mean(rew_per_foot) - imbalance_coef * std(rew_per_foot)
      (clamped at 0), so one overachieving leg can't hide three lazy ones.
    """
    # --- 1. Commanded clearance ---
    gait_cmd = env.command_manager.get_command(command_name)     # (num_envs, 3)
    target_clearance = gait_cmd[:, 2]                            # (num_envs,)

    # --- 2. Base + foot positions ---
    robot = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :] # (num_envs, num_feet, 3)
    foot_z = foot_pos_w[..., 2]                                  # (num_envs, num_feet)

    # --- 3. Terrain height from RayCaster (per env) ---
    ground_z = ground_height_from_scanner_safe(env, sensor_cfg)  # (num_envs,)
    clearance = foot_z - ground_z.unsqueeze(1)                   # (num_envs, num_feet)

    # --- 4. Error relative to commanded clearance (only when too LOW) ---
    target_expand = target_clearance.unsqueeze(1)                # (num_envs, 1)
    below_target = torch.clamp(target_expand - clearance, min=0.0)
    error = below_target**2                                      # (num_envs, num_feet)

    # --- 5. Swing mask (only moving feet matter) ---
    foot_vel = torch.norm(
        robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :],
        dim=-1,
    )                                                             # (num_envs, num_feet)
    is_swinging = (foot_vel > swing_vel_thresh).float()

    # --- 6. Exponential per-foot reward, stabilized ---
    exp_arg = (-error / sigma).clamp(min=-20.0, max=0.0)
    rew_per_foot = torch.exp(exp_arg) * is_swinging              # (num_envs, num_feet)

    # If no feet are swinging in an env, rew_per_foot will be all zeros there,
    # which is fine: mean=0, std=0 → reward=0.

    # --- 7. Aggregate across feet with imbalance penalty ---
    mean = rew_per_foot.mean(dim=1)                              # (num_envs,)
    std = rew_per_foot.std(dim=1, unbiased=False)                # (num_envs,)

    reward = mean - imbalance_coef * std
    reward = torch.clamp(reward, min=0.0)                        # (num_envs,)

    if torch.isnan(reward).any() or torch.isinf(reward).any():
        print("NaN/Inf in track_feet_clearance_exp reward")
        raise RuntimeError("NaN/Inf in track_feet_clearance_exp")

    return reward


# ---------------------------------------------------------
# Existing NAV rewards (keep)
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


# =========================================================
# NEW BASE LOCOMOTION REWARDS
# =========================================================

# 1. Linear velocity tracking (exp(-||v_ref - v||^2))
def lin_vel_tracking_exp(env, asset_cfg: SceneEntityCfg, command_name: str):
    """Exponential tracking of commanded (vx, vy) in BODY frame."""
    robot = env.scene[asset_cfg.name]

    # commanded [vx_ref, vy_ref, wz_ref, heading] from command manager
    cmd = env.command_manager.get_command(command_name)
    v_ref = cmd[:, :2]  # (vx_ref, vy_ref)

    # actual base linear velocity in body frame: (vx, vy)
    v = robot.data.root_lin_vel_b[:, :2]

    diff = v_ref - v
    sq = torch.sum(diff * diff, dim=1)
    
    return torch.exp(-sq)


# 2. Angular velocity tracking (exp(-(wz_ref - wz)^2))
def ang_vel_tracking_exp(env, asset_cfg: SceneEntityCfg, command_name: str):
    """Exponential tracking of commanded yaw rate (wz)."""
    robot = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command(command_name)
    wz_ref = cmd[:, 2]  # assuming [vx, vy, wz, heading]

    wz = robot.data.root_ang_vel_b[:, 2]  # yaw rate in body frame

    diff = wz_ref - wz
    return torch.exp(-(diff * diff))


def track_commanded_height_exp(env, command_name: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    """
    Reward for matching commanded base height (gait_params[:, 0]) relative to terrain.
    Returns exp(-error^2 / std^2).
    """
    # 1. Commanded offset
    gait_cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)
    target_height_offset = gait_cmd[:, 0]                     # (num_envs,)

    nominal_height = 0.38
    target_height = nominal_height + target_height_offset     # (num_envs,)

    # 2. Current height relative to terrain via RayCaster
    robot = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)  # (num_envs,)
    current_height = robot.data.root_pos_w[:, 2] - ground_z       # (num_envs,)

    # 3. Error + exp kernel
    error = torch.square(current_height - target_height)
    std = 0.05
    result = torch.exp(-error / (std**2))

    if torch.isnan(result).any() or torch.isinf(result).any():
        print("NaN/Inf in track_commanded_height_exp")
        raise RuntimeError("NaN/Inf in track_commanded_height_exp")

    return result

def track_feet_clearance_body_l2(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    nominal_height: float = 0.38,
    swing_vel_thresh: float = 0.1,
):
    """
    Safer clearance reward that does NOT touch RayCaster or terrain.
    Approximate ground under the base as:
        z_ground ≈ base_z - nominal_height

    Then:
        clearance = foot_z - z_ground

    We penalize (clearance - target_clearance)^2 on SWINGING feet only.
    """
    robot = env.scene[asset_cfg.name]

    # --- 1. Commanded clearance (index 2: [height, freq, clearance]) ---
    gait_cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)
    target_clearance = gait_cmd[:, 2]                         # (num_envs,)

    # --- 2. Base + feet z positions ---
    base_z = robot.data.root_pos_w[:, 2]                      # (num_envs,)
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    foot_z = foot_pos_w[..., 2]                                   # (num_envs, num_feet)

    # approximate ground plane from base height:
    z_ground_approx = base_z - nominal_height                  # (num_envs,)
    clearance = foot_z - z_ground_approx.unsqueeze(1)          # (num_envs, num_feet)

    # --- 3. Error vs target, clamp only when *below* target ---
    target_expand = target_clearance.unsqueeze(1)              # (num_envs, 1)
    below_target = torch.clamp(target_expand - clearance, min=0.0)
    error = below_target**2                                    # (num_envs, num_feet)

    # --- 4. Apply only to swinging feet ---
    foot_vel = torch.norm(
        robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :],
        dim=-1,
    )                                                          # (num_envs, num_feet)
    is_swinging = (foot_vel > swing_vel_thresh).float()

    # L2 penalty on error for swinging feet
    per_foot_penalty = error * is_swinging                     # (num_envs, num_feet)
    penalty = per_foot_penalty.mean(dim=1)                     # (num_envs,)

    # Optional sanity check
    if torch.isnan(penalty).any() or torch.isinf(penalty).any():
        print("NaN/Inf in track_feet_clearance_body_l2")
        raise RuntimeError("NaN in track_feet_clearance_body_l2")

    # NOTE: This returns a *penalty*; use a negative weight or negate here.
    return penalty







# 4. Pose similarity ||q - q_default||^2
def pose_similarity(env, asset_cfg: SceneEntityCfg):
    """Squared distance to the robot's default joint configuration."""
    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos

    # Many Isaac/IsaacLab articulations expose default joint pose; if not,
    # fall back to zero as a safe no-op.
    if hasattr(robot.data, "default_joint_pos") and robot.data.default_joint_pos is not None:
        q_def = robot.data.default_joint_pos
    else:
        q_def = torch.zeros_like(q)

    diff = q - q_def
    return torch.sum(diff * diff, dim=1)


# 6. Vertical velocity penalty v_z^2
def vertical_vel_penalty(env, asset_cfg: SceneEntityCfg):
    """Penalty on vertical (z-axis) linear velocity of the base."""
    robot = env.scene[asset_cfg.name]
    vz = robot.data.root_lin_vel_b[:, 2]
    return vz * vz  # use negative weight


# 7. Roll/pitch stabilization penalty (phi^2 + theta^2)
def roll_pitch_penalty(env, asset_cfg: SceneEntityCfg):
    """
    Penalty for non-flat base orientation.
    Optimization: Uses projected gravity instead of Euler angles (faster/safer).
    """
    robot = env.scene[asset_cfg.name]

    #ORIGINAL:
    # roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    # return roll * roll + pitch * pitch  # use negative weight


    #NEW:
    # The gravity vector in World frame is [0, 0, -1]
    # We rotate it into the Robot's Body frame
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    projected_gravity_b = quat_apply_inverse(robot.data.root_quat_w, gravity_vec_w)
    
    # If perfectly flat, projected_gravity_b should be [0, 0, -1]
    # The x and y components represent roll/pitch tilt
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

def forward_velocity_bonus(env, asset_cfg):
    robot = env.scene[asset_cfg.name]          # FIX: IsaacLab uses dict-style access
    base_lin_vel_b = robot.data.root_lin_vel_b # (num_envs, 3)

    forward_vel = base_lin_vel_b[:, 0]         # body-frame X velocity
    return forward_vel

def negative_forward_velocity_penalty(env, env_ids=None, asset_cfg=None):
    """
    Penalize backward motion (negative x-velocity).
    """
    robot = env.scene[asset_cfg.name]

    # Always fix env_ids to slice(None) when empty
    if env_ids is None:
        env_ids = slice(None)

    # Extract forward velocity — world frame X
    base_lin_vel = robot.data.root_vel_w[env_ids, 0]   # shape [num_envs]

    # Penalize backward motion only
    penalty = torch.clamp(-base_lin_vel, min=0.0)
    return penalty
