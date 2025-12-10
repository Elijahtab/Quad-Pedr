import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils  # Contains the crucial conversion tools
import isaaclab.envs.mdp as mdp
import math

#New imports for A*:
import heapq
from typing import List, Optional, Tuple


# Optional: debug drawing of global A* path + lookahead
try:
    from isaacsim.util.debug_draw import _debug_draw
except Exception:
    # In pure headless / IsaacLab-python builds this may not exist
    _debug_draw = None

_DEBUG_DRAW = None

# Toggle this to enable/disable visualization globally
# ENABLE_LOOKAHEAD_DEBUG_DRAW: bool = False
# DEBUG_LOOKAHEAD_ENV_ID: int = 0  # which env index to visualize (usually 0)
# For toggling this on:
ENABLE_LOOKAHEAD_DEBUG_DRAW = True
DEBUG_LOOKAHEAD_ENV_ID = 0   # whichever env index you care about



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






#### SLAM/PATH PLANNING CHANGES START #####



def _get_debug_draw():
    """Lazily acquire the global debug-draw interface from Isaac Sim."""
    global _DEBUG_DRAW
    if _debug_draw is None:
        return None
    if _DEBUG_DRAW is None:
        try:
            _DEBUG_DRAW = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            _DEBUG_DRAW = None
    return _DEBUG_DRAW


def debug_draw_path_and_lookahead(
    path_world_xy: torch.Tensor,
    base_pos_world: torch.Tensor,
    lookahead_world_xy: torch.Tensor,
    path_color=(0.0, 1.0, 0.0, 1.0),
    lookahead_color=(1.0, 0.3, 0.0, 1.0),
    link_color=(1.0, 1.0, 0.0, 1.0),
    path_width: float = 2.5,
    link_width: float = 2.0,
):
    """
    Visualize the global A* path and lookahead using isaacsim.util.debug_draw.

    Args:
        path_world_xy:      (P, 2) tensor of [x, y] path points in world frame.
        base_pos_world:     (3,)   tensor of base [x, y, z] in world frame.
        lookahead_world_xy: (2,)   tensor of [x, y] for lookahead in world frame.
    """
    draw = _get_debug_draw()
    if draw is None:
        return

    # If no path, just clear any existing drawings
    if path_world_xy.numel() == 0:
        try:
            draw.clear_lines()
            draw.clear_points()
        except Exception:
            pass
        return

    # Use base z for all points so the path lies in a horizontal plane at robot height
    z = float(base_pos_world[2].item())

    # Path points to 3D list
    pts_np = path_world_xy.detach().cpu().numpy()
    path_pts_3d = [(float(p[0]), float(p[1]), z) for p in pts_np]

    # Lookahead point (3D)
    lx = float(lookahead_world_xy[0].item())
    ly = float(lookahead_world_xy[1].item())
    lookahead_pt_3d = [(lx, ly, z)]

    # Robot base (3D)
    bx = float(base_pos_world[0].item())
    by = float(base_pos_world[1].item())
    base_pt_3d = [(bx, by, z)]

    # Clear previous drawings so we only see the current step
    try:
        draw.clear_lines()
        draw.clear_points()
    except Exception:
        # If something goes wrong, don't crash the sim
        pass

    # Draw the path as a spline (API from Isaac Sim Debug Drawing extension docs)
    draw.draw_lines_spline(path_pts_3d, path_color, float(path_width), False)

    # Draw lookahead as a point
    draw.draw_points(lookahead_pt_3d, [lookahead_color], [10.0])

    # Draw a line from robot base to lookahead
    draw.draw_lines(base_pt_3d, lookahead_pt_3d, [link_color], [float(link_width)])


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion (w, x, y, z). Returns scalar tensor."""
    w, x, y, z = q.unbind(-1)
    # standard yaw-from-quat formula (Z-up, yaw around Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _world_to_grid(
    x: float,
    y: float,
    map_half_extent: float,
    grid_resolution: float,
    grid_size: int,
) -> Optional[Tuple[int, int]]:
    """Convert world (x, y) to integer grid indices (ix, iy). Return None if out of bounds."""
    ix = int((x + map_half_extent) / grid_resolution)
    iy = int((y + map_half_extent) / grid_resolution)
    if ix < 0 or iy < 0 or ix >= grid_size or iy >= grid_size:
        return None
    return ix, iy


def _grid_to_world(
    ix: int,
    iy: int,
    map_half_extent: float,
    grid_resolution: float,
) -> Tuple[float, float]:
    """Convert grid indices (ix, iy) to world (x, y) at cell center."""
    x = (ix + 0.5) * grid_resolution - map_half_extent
    y = (iy + 0.5) * grid_resolution - map_half_extent
    return x, y

def _build_occupancy_grid(
    obstacles_xy: List[Tuple[float, float]],
    map_half_extent: float,
    grid_resolution: float,
    obstacle_inflation: float,
    grid_size: int,
) -> List[List[bool]]:
    """Build a binary occupancy grid (True = occupied) with inflated obstacles."""
    occ = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    infl_radius_sq = obstacle_inflation * obstacle_inflation

    for ix in range(grid_size):
        for iy in range(grid_size):
            cx, cy = _grid_to_world(ix, iy, map_half_extent, grid_resolution)
            for ox, oy in obstacles_xy:
                dx = ox - cx
                dy = oy - cy
                if dx * dx + dy * dy <= infl_radius_sq:
                    occ[iy][ix] = True
                    break
    return occ


def _plan_astar_single_env(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    occ: List[List[bool]],
    map_half_extent: float,
    grid_resolution: float,
    max_astar_steps: int,
) -> Optional[List[Tuple[float, float]]]:
    """Compute an 8-connected A* path in 2D grid from start to goal on a given occupancy grid.

    Returns:
        List of (x, y) world points along the path, or None if no path found / invalid.
    """
    if occ is None:
        return None

    grid_size = len(occ)
    if grid_size <= 1:
        return None

    start_idx = _world_to_grid(*start_xy, map_half_extent, grid_resolution, grid_size)
    goal_idx = _world_to_grid(*goal_xy, map_half_extent, grid_resolution, grid_size)
    if start_idx is None or goal_idx is None:
        return None

    sx, sy = start_idx
    gx, gy = goal_idx

    # If start or goal in obstacle, bail
    if occ[sy][sx] or occ[gy][gx]:
        return None

    # A* search
    def heuristic(ix: int, iy: int) -> float:
        dx = ix - gx
        dy = iy - gy
        return math.sqrt(dx * dx + dy * dy)

    neighbors = [  # 8-connected
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    open_heap: List[Tuple[float, int, int]] = []
    heapq.heappush(open_heap, (0.0, sx, sy))

    came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sx, sy): None}
    g_cost: dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    steps = 0
    while open_heap and steps < max_astar_steps:
        f, x, y = heapq.heappop(open_heap)
        steps += 1

        if (x, y) == (gx, gy):
            break

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= grid_size or ny >= grid_size:
                continue
            if occ[ny][nx]:
                continue

            step_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2.0)
            new_g = g_cost[(x, y)] + step_cost
            nkey = (nx, ny)

            if nkey not in g_cost or new_g < g_cost[nkey]:
                g_cost[nkey] = new_g
                heapq.heappush(open_heap, (new_g + heuristic(nx, ny), nx, ny))
                came_from[nkey] = (x, y)

    if (gx, gy) not in came_from:
        return None

    # Reconstruct path (grid -> world)
    path_grid: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = (gx, gy)
    while cur is not None:
        path_grid.append(cur)
        cur = came_from[cur]
    path_grid.reverse()

    path_world: List[Tuple[float, float]] = [
        _grid_to_world(ix, iy, map_half_extent, grid_resolution) for (ix, iy) in path_grid
    ]
    return path_world


def _has_line_of_sight(
    occ: List[List[bool]],
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    map_half_extent: float,
    grid_resolution: float,
) -> bool:
    """Check if the straight segment start_xy→end_xy is free of occupied cells."""
    if occ is None:
        return False

    grid_size = len(occ)
    sx_sy = _world_to_grid(
        float(start_xy[0]),
        float(start_xy[1]),
        map_half_extent,
        grid_resolution,
        grid_size,
    )
    ex_ey = _world_to_grid(
        float(end_xy[0]),
        float(end_xy[1]),
        map_half_extent,
        grid_resolution,
        grid_size,
    )
    if sx_sy is None or ex_ey is None:
        return False

    x0, y0 = sx_sy
    x1, y1 = ex_ey

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        # Out of bounds or occupied → no LOS
        if y < 0 or y >= grid_size or x < 0 or x >= grid_size:
            return False
        if occ[y][x]:
            return False

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return True


def _get_pose_command_goal_xy_world(env: ManagerBasedRLEnv, robot) -> torch.Tensor:
    """Best-effort retrieval of a stable world-frame goal for planning.

    Preference order:
    1) Read a world-goal tensor directly from the pose command term (if exposed).
    2) Reconstruct a world goal from the body-frame tracking command as:
         goal_w = base_pos_w + R(yaw) * cmd_b_xy
    """
    # 1) Try to access the underlying command term buffer (names may differ across Isaac Lab versions)
    term_dict = getattr(env.command_manager, "command_terms", {})
    term = term_dict.get("pose_command", None)

    if term is not None:
        # These attribute names are guesses; keep the list broad but safe.
        for attr in ("goal_pos_w", "target_pos_w", "command_w", "command_pos_w", "pose_w", "pos_w"):
            if hasattr(term, attr):
                val = getattr(term, attr)
                if isinstance(val, torch.Tensor) and val.ndim == 2 and val.shape[1] >= 2:
                    return val[:, :2]

    # 2) Fallback: assume get_command returns a body-frame XY tracking vector
    cmd_b = env.command_manager.get_command("pose_command")
    base_pos_w = robot.data.root_pos_w[:, :2]
    yaw = _yaw_from_quat(robot.data.root_quat_w)

    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    dx_b = cmd_b[:, 0]
    dy_b = cmd_b[:, 1]

    dx_w = cos_y * dx_b - sin_y * dy_b
    dy_w = sin_y * dx_b + cos_y * dy_b

    return base_pos_w + torch.stack([dx_w, dy_w], dim=1)



def lookahead_hint(
    env: ManagerBasedRLEnv,
    map_half_extent: float = 7.0,
    grid_resolution: float = 0.25,
    min_lookahead_distance: float = 1.0,
    max_lookahead_distance: float = 4.0,
    obstacle_inflation: float = 0.35,
    max_astar_steps: int = 8192,
) -> torch.Tensor:
    """Return [dx, dy, path_progress, hint_active] for each env.

    Heavy work on CPU, output on GPU.
    """
    # ---- output stays on GPU ----
    device_out = env.device
    num_envs = env.num_envs
    out = torch.zeros((num_envs, 4), device=cpu, dtype=torch.float32)

    #TODO: WHEN UPGRADING TO 4D LOOKAHEAD (plus hint):
    # 5 dims: dx, dy, path_progress, detour_norm, hint_active
    # out = torch.zeros((num_envs, 5), device=device, dtype=torch.float32)

    # ---- heavy work device ----
    cpu = torch.device("cpu")

    scene = env.scene
    if "robot" not in scene.keys() or "obstacles" not in scene.keys():
        return out

    robot = scene["robot"]
    obstacles = scene["obstacles"]

    # Keep full base position for debug draw (can remain on GPU; debug fn .cpu()s anyway)
    base_pos_w_full = robot.data.root_pos_w[:, :3]  # (N, 3)

    # Move planning inputs to CPU
    base_pos_w = robot.data.root_pos_w[:, :2].to(cpu)     # (N, 2)
    base_quat_w = robot.data.root_quat_w.to(cpu)          # (N, 4)

    goal_xy_world = _get_pose_command_goal_xy_world(env, robot).to(cpu)  # (N, 2)
    obs_xy_all_world = obstacles.data.object_link_pose_w[:, :, :2].to(cpu)  # (N, num_obs, 2)

    # Origins (only used for debug in your current code)
    env_origins_xy = env.scene.env_origins[:, :2].to(cpu)  # noqa: F841

    # If you later want true env-frame math, subtract origins here.
    base_pos_env = base_pos_w - env_origins_xy
    goal_xy_env = goal_xy_world - env_origins_xy
    obs_xy_all_env = obs_xy_all_world - env_origins_xy.unsqueeze(1)

    # ---- cache on CPU ----
    if not hasattr(env, "_lookahead_cache"):
        env._lookahead_cache = {
            "last_goal_xy": goal_xy_env.clone(),
            "paths": [None] * num_envs,
            "path_total_len": torch.zeros(num_envs, device=cpu),
            "occ_grids": [None] * num_envs,
            "path_tensors": [None] * num_envs,
            "seg_lens": [None] * num_envs,
            "cum_lens": [None] * num_envs,
            "progress_idx": torch.zeros(num_envs, dtype=torch.long, device=cpu),

        }

    cache = env._lookahead_cache
    occ_grids: List[Optional[List[List[bool]]]] = cache["occ_grids"]
    last_goal_xy: torch.Tensor = cache["last_goal_xy"]
    paths: List[Optional[List[Tuple[float, float]]]] = cache["paths"]
    path_total_len: torch.Tensor = cache["path_total_len"]

    # Replan only when the goal moves significantly, or when we don't have a path yet
    goal_delta = torch.linalg.norm(goal_xy_env - last_goal_xy, dim=1)
    need_replan_goal = goal_delta > grid_resolution * 0.5

    replan_ids = set(torch.nonzero(need_replan_goal, as_tuple=False).squeeze(-1).tolist())
    for env_idx in range(num_envs):
        if paths[env_idx] is None:
            replan_ids.add(env_idx)

    for env_idx in replan_ids:
        start = tuple(float(v) for v in base_pos_env[env_idx].tolist())
        goal = tuple(float(v) for v in goal_xy_env[env_idx].tolist())
        obs_list = [tuple(float(x) for x in xy) for xy in obs_xy_all_env[env_idx].tolist()]

        grid_size = int((2.0 * map_half_extent) / grid_resolution)
        if grid_size > 1:
            occ_grid = _build_occupancy_grid(
                obstacles_xy=obs_list,
                map_half_extent=map_half_extent,
                grid_resolution=grid_resolution,
                obstacle_inflation=obstacle_inflation,
                grid_size=grid_size,
            )
        else:
            occ_grid = None

        occ_grids[env_idx] = occ_grid

        path = _plan_astar_single_env(
            start_xy=start,
            goal_xy=goal,
            occ=occ_grid,
            map_half_extent=map_half_extent,
            grid_resolution=grid_resolution,
            max_astar_steps=max_astar_steps,
        )

        paths[env_idx] = path
        last_goal_xy[env_idx] = goal_xy_env[env_idx]

        if path is not None and len(path) >= 2:
            pt = torch.tensor(path, dtype=torch.float32, device=cpu)
            seg_vecs = pt[1:] - pt[:-1]
            seg_lens = torch.linalg.norm(seg_vecs, dim=1)
            total_len = float(seg_lens.sum().item())
        else:
            total_len = 0.0

        pt = torch.tensor(path, dtype=torch.float32, device=cpu)
        seg = torch.linalg.norm(pt[1:] - pt[:-1], dim=1)
        cum = torch.cat([torch.zeros(1), torch.cumsum(seg, dim=0)])
        cache["path_tensors"][env_idx] = pt
        cache["seg_lens"][env_idx] = seg
        cache["cum_lens"][env_idx] = cum
        cache["progress_idx"][env_idx] = 0

        path_total_len[env_idx] = max(total_len, 1e-6)

    cache["last_goal_xy"] = last_goal_xy
    cache["paths"] = paths
    cache["path_total_len"] = path_total_len
    cache["occ_grids"] = occ_grids

    # ---- compute hints from cached paths (CPU) ----
    for env_idx in range(num_envs):
        path = paths[env_idx]
        if not path or len(path) < 2:
            continue

        total_len = float(path_total_len[env_idx].item())
        if total_len <= 1e-6:
            continue

        path_tensor = cache["path_tensors"][env_idx]  # (P, 2)
        robot_xy = base_pos_env[env_idx]                                   # (2,)

        seg_lens = cache["seg_lens"][env_idx]
        _ = torch.clamp(seg_lens.sum(), min=1e-6)  # curr_total_len (unused beyond debug)

        diff = path_tensor - robot_xy.unsqueeze(0)
        dists_sq = torch.sum(diff * diff, dim=1)
        closest_idx = int(torch.argmin(dists_sq).item())

        if closest_idx == 0:
            s_travelled = 0.0
        else:
            s_travelled = float(seg_lens[:closest_idx].sum().item())

        path_progress = max(0.0, min(1.0, s_travelled / total_len))

        #TODO: WHEN UPGRADING TO 4D LOOKAHEAD (plus hint):
        # # --- UPGRADE FOR 4D: compute detour amount = (remaining path length - straight-line distance) ---
        # # Remaining path length from robot to goal along the A* path
        # # TODO: After choosing final path progress method, ensure this matches
        # remaining_len = max(float(total_len - s_travelled.item()), 0.0)

        # # Straight-line distance from robot to goal
        # goal_xy_env = goal_xy[env_idx]  # (2,)
        # straight_dist = torch.linalg.norm(goal_xy_env - robot_xy)

        # # Difference (how much longer the optimal path is than a straight line)
        # detour = remaining_len - float(straight_dist.item())

        # # Normalize by map_half_extent and clamp to [0, 1]
        # detour_norm = detour / map_half_extent
        # detour_norm = max(0.0, min(1.0, detour_norm))
        # detour_norm = torch.tensor(detour_norm, device=device, dtype=torch.float32)

        occ_grid = occ_grids[env_idx]

        best_idx: Optional[int] = None
        best_eucl: float = 0.0

        if occ_grid is not None:
            for j in range(closest_idx + 1, path_tensor.shape[0]):
                cand_xy = path_tensor[j]
                eucl = float(torch.linalg.norm(cand_xy - robot_xy).item())

                if eucl < min_lookahead_distance or eucl > max_lookahead_distance:
                    continue

                if _has_line_of_sight(occ_grid, robot_xy, cand_xy, map_half_extent, grid_resolution):
                    if eucl > best_eucl:
                        best_eucl = eucl
                        best_idx = j

        if best_idx is not None:
            look_idx = best_idx
        else:
            look_idx = closest_idx
            travelled = 0.0
            for j in range(closest_idx, path_tensor.shape[0] - 1):
                seg_len = float(torch.linalg.norm(path_tensor[j + 1] - path_tensor[j]).item())
                travelled += seg_len
                look_idx = j + 1
                if travelled >= min_lookahead_distance:
                    break

        look_xy = path_tensor[look_idx]

        # Debug draw (unchanged semantics)
        if ENABLE_LOOKAHEAD_DEBUG_DRAW and int(env_idx) == int(DEBUG_LOOKAHEAD_ENV_ID):
            # try:
            origin_xy = env_origins_xy[env_idx]  # (2,)

            # Convert ENV coords back to WORLD for visualization
            path_world_xy = path_tensor + origin_xy            # (P, 2)
            lookahead_world_xy = look_xy + origin_xy           # (2,)

            debug_draw_path_and_lookahead(
                path_world_xy=path_world_xy,
                base_pos_world=base_pos_w_full[env_idx],       # (3,) already world
                lookahead_world_xy=lookahead_world_xy,
            )
            # except Exception:
            #     pass

        delta_world = look_xy - robot_xy

        yaw = _yaw_from_quat(base_quat_w[env_idx])  # CPU scalar tensor
        cos_y = torch.cos(-yaw)
        sin_y = torch.sin(-yaw)

        dx_b = (cos_y * delta_world[0] - sin_y * delta_world[1]).item()
        dy_b = (sin_y * delta_world[0] + cos_y * delta_world[1]).item()

        # Write small scalars into GPU output
        out[env_idx, 0] = float(dx_b)
        out[env_idx, 1] = float(dy_b)
        out[env_idx, 2] = float(path_progress)
        out[env_idx, 3] = 1.0
        #TODO: WHEN UPGRADING TO 4D LOOKAHEAD (plus hint):
        # out[env_idx, 3] = detour_norm        # normalized detour amount
        # out[env_idx, 4] = 1.0                # hint active

    # ---- optional flicker mask (MUST be on GPU) ----
    out_gpu = out.to(device_out)

    # keep your mask on GPU
    hint_prob = 0.8
    mask = torch.bernoulli(torch.full((num_envs, 1), hint_prob, device=device_out))
    out_gpu = out_gpu * mask

    return out_gpu



def lookahead_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Phase 1: Returns random noise with a RARE availability flag.
    Strategy: 90% Off / 10% On.
    Why: Prevents the network from learning to "hate" this input.
    """
    # 1. 3 Random Vectors (The Garbage)
    noise_vec = torch.randn((env.num_envs, 3), device=env.device)
    #TODO: WHEN UPGRADING TO 4D LOOKAHEAD (plus hint):
    # 1. 4 Random Vectors (The Garbage)
    # noise_vec = torch.randn((env.num_envs, 4), device=env.device)

    # 2. Bernoulli Mask (The Coin Flip)
    # 0.1 means 10% chance of being 1.0, 90% chance of being 0.0
    flag = torch.bernoulli(torch.full((env.num_envs, 1), 0.1, device=env.device))

    # 3. Combine
    return torch.cat([noise_vec, flag], dim=-1)



#### SLAM/PATH PLANNING CHANGES END #####



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
