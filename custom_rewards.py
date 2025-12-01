import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_apply
from . import custom_obs


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


# ---------------------------------------------------------
# 4. Base height penalty (L2)
# ---------------------------------------------------------
def base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize deviation of robot base height from a target height (L2).

    For flat terrain, we just compare to a fixed target height.
    For rough terrain, an optional sensor can be used to shift the target.
    """
    asset = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        adjusted_target_height = target_height + sensor.data.pos_w[:, 2]
    else:
        adjusted_target_height = target_height

    z = asset.data.root_link_pos_w[:, 2]
    return (z - adjusted_target_height) ** 2


def foot_forward_separation(
    env,
    asset_cfg: SceneEntityCfg,
    min_separation: float = 0.10,   # meters, tune this
) -> torch.Tensor:
    """
    Penalize gaits where hind feet come too far forward (overlap with front feet)
    along the robot's forward (x) axis in the *base frame*.

    Returns: (num_envs,) tensor – one penalty scalar per environment.
    """
    robot = env.scene[asset_cfg.name]
    data = robot.data

    # ---- Base pose ----
    root_pos = data.root_link_pos_w        # (N, 3)  (usually)
    root_quat = data.root_link_quat_w      # (N, 4)
    num_envs = root_pos.shape[0]

    # ---- Body positions in world ----
    # IsaacLab might store this as either:
    #  - (N, num_bodies, 3), or
    #  - (N * num_bodies, 3)
    body_pos_raw = getattr(data, "body_pos_w", None)
    if body_pos_raw is None:
        # fallback if the name differs in your build
        body_pos_raw = data.link_pos_w

    if body_pos_raw.ndim == 3:
        # already (N, num_bodies, 3)
        body_pos_w = body_pos_raw
    elif body_pos_raw.ndim == 2:
        # flattened (N * num_bodies, 3)
        total = body_pos_raw.shape[0]
        num_bodies = total // num_envs
        body_pos_w = body_pos_raw.view(num_envs, num_bodies, 3)
    else:
        # unexpected shape: just return zero penalty to be safe
        return torch.zeros(num_envs, device=root_pos.device, dtype=root_pos.dtype)

    # ---- Foot indices ----
    # If your API uses find_links instead of find_bodies, just swap that call.
    idx_fr = robot.find_bodies("FR_foot")[0]
    idx_fl = robot.find_bodies("FL_foot")[0]
    idx_rr = robot.find_bodies("RR_foot")[0]
    idx_rl = robot.find_bodies("RL_foot")[0]

    p_fr = body_pos_w[:, idx_fr, :]    # (N, 3)
    p_fl = body_pos_w[:, idx_fl, :]
    p_rr = body_pos_w[:, idx_rr, :]
    p_rl = body_pos_w[:, idx_rl, :]

    # ---- Base forward axis in world frame ----
    fwd_local = torch.tensor(
        [1.0, 0.0, 0.0],
        device=root_pos.device,
        dtype=root_pos.dtype,
    ).unsqueeze(0).repeat(num_envs, 1)   # (N, 3)

    fwd_world = quat_apply(root_quat, fwd_local)   # (N, 3)

    # ---- Relative positions wrt base ----
    rel_fr = p_fr - root_pos    # (N, 3)
    rel_fl = p_fl - root_pos
    rel_rr = p_rr - root_pos
    rel_rl = p_rl - root_pos

    # ---- Project onto forward axis ----
    proj_fr = (rel_fr * fwd_world).sum(dim=-1)   # (N,)
    proj_fl = (rel_fl * fwd_world).sum(dim=-1)
    proj_rr = (rel_rr * fwd_world).sum(dim=-1)
    proj_rl = (rel_rl * fwd_world).sum(dim=-1)

    # Front – hind distances along forward axis on each side
    sep_right = proj_fr - proj_rr   # (N,)
    sep_left  = proj_fl - proj_rl   # (N,)

    # Smaller separation is the “dangerous” one
    sep = torch.minimum(sep_right, sep_left)   # (N,)

    # Quadratic penalty – grows fast when they overlap
    diff = torch.clamp(min_separation - sep, min=0.0)
    penalty = diff * diff
    
    penalty = penalty.view(num_envs, -1).mean(dim=1)

    return penalty