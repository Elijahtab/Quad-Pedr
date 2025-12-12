import torch
import math
from isaaclab.envs import ManagerBasedRLEnv

# You can reuse your existing constants, or hardcode.
# If you want to reuse OBSTACLE_HEIGHT, you can either:
# - import it from rough_env_cfg, OR
# - just rely on the default z already stored in object_link_pose_w.
# I'll do the second (simpler) option.

# How “busy” the env can be.
MAX_OBSTACLES_PER_ENV = 3        # bump this later to 5, 8, ...
OBSTACLE_SIZE = (0.4, 0.4, 0.4)  # (x, y, z) in meters
OBSTACLE_HEIGHT = OBSTACLE_SIZE[2]

def randomize_obstacles_static_startup(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    inner_radius: float = 1.0,   # no cubes inside this radius
    outer_radius: float = 3.0,   # max radius where cubes can appear
) -> None:
    """
    Spawn obstacles in a donut AROUND EACH ROBOT (per env), not at world origin.

    - inner_radius: keep a small "safe bubble" around the base
    - outer_radius: how far out obstacles can appear (in meters, in XY plane)
    """

    device = env.device

    # Which envs are we randomizing?
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    if env_ids.numel() == 0:
        return

    # Grab robot and obstacle collections
    scene = env.scene
    robot = scene["robot"]
    obstacles = scene["obstacles"]  # RigidObjectCollection

    # Current obstacle poses: [num_envs, num_objs, 7] (x, y, z, qw, qx, qy, qz)
    obj_pose = obstacles.data.object_link_pose_w.clone()
    num_envs, num_objs, _ = obj_pose.shape

    # Robot base positions in world: [num_envs, 3]
    base_pos_w = robot.data.root_pos_w  # (x, y, z)
    base_xy = base_pos_w[:, :2]         # [num_envs, 2]

    # Only operate on the selected env IDs
    E = env_ids.shape[0]

    # ------------------------------------------------------------------
    # SAMPLE DONUT POSITIONS PER-ENV, PER-OBJECT
    # ------------------------------------------------------------------
    # Angles: [E, num_objs] in [0, 2π)
    angles = 2.0 * math.pi * torch.rand(E, num_objs, device=device)
    # Radii: uniform in [inner_radius, outer_radius]
    radii = torch.empty(E, num_objs, device=device).uniform_(inner_radius, outer_radius)

    offset_x = radii * torch.cos(angles)  # [E, num_objs]
    offset_y = radii * torch.sin(angles)  # [E, num_objs]

    # Base positions for these envs: [E, 2]
    base_xy_sel = base_xy[env_ids]        # [E, 2]

    # Broadcast and set world positions for obstacles
    # x
    obj_pose[env_ids, :, 0] = base_xy_sel[:, 0:1] + offset_x
    # y
    obj_pose[env_ids, :, 1] = base_xy_sel[:, 1:1+1] + offset_y

    # z: keep whatever the spawner set (usually top of terrain)
    # If you want to enforce a fixed height, uncomment something like:
    # obj_pose[env_ids, :, 2] = some_height

    # ------------------------------------------------------------------
    # WRITE BACK TO SIM
    # ------------------------------------------------------------------
    # obj_pose: [num_envs, num_objs, 7]
    # env_ids: [E]
    obj_pose_subset = obj_pose[env_ids]  # [E, num_objs, 7]

    obstacles.write_object_pose_to_sim(
        object_pose=obj_pose_subset,
        env_ids=env_ids,
        object_ids=None,  # all objects in those envs
    )


def randomize_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    spawn_radius: float = 3.0,
    min_gap_from_robot: float = 0.7,
    max_active_obstacles: int = 4,
    obstacle_density: float = 0.3,
) -> None:
    """Randomize obstacle positions around the robot on reset.

    env_ids: the env indices this event is being applied to (EventManager passes this in).
    Other args come from EventTerm.params.
    """
    scene = env.scene
    robot = scene["robot"]
    obstacles = scene["obstacles"]  # RigidObjectCollection

    # All envs if none given
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Get robot base positions [num_envs, 3]
    base_pos_w = robot.data.root_pos_w[env_ids]  # (N, 3)

    num_envs = env_ids.shape[0]
    num_objs = obstacles.num_objects

    # Decide how many obstacles per env (sparse by default)
    # around ~obstacle_density * num_objs, clipped by max_active_obstacles
    max_per_env = min(max_active_obstacles, num_objs)
    # sample number of active obstacles (0..max_per_env)
    # simple: Bernoulli per object, then clamp
    bern = torch.rand((num_envs, num_objs), device=env.device)
    active_mask = (bern < obstacle_density)
    # force at most max_per_env per env
    counts = active_mask.sum(dim=1, keepdim=True)
    too_many = counts > max_per_env
    # if too many, randomly trim
    if too_many.any():
        # generate random scores and keep the smallest `max_per_env`
        scores = torch.rand_like(bern)
        scores[~active_mask] = 1e9
        kth = torch.topk(scores, k=max_per_env, dim=1, largest=False).values[:, -1:]
        # new mask: only those with score <= kth and originally active
        active_mask = active_mask & (scores <= kth)

    # Sample random polar offsets in XY for each env/obstacle
    # shape: (num_envs, num_objs)
    rand_r = spawn_radius * torch.sqrt(torch.rand((num_envs, num_objs), device=env.device))
    rand_theta = 2.0 * math.pi * torch.rand((num_envs, num_objs), device=env.device)

    offset_x = rand_r * torch.cos(rand_theta)
    offset_y = rand_r * torch.sin(rand_theta)

    # Enforce min distance from robot in XY
    too_close = (rand_r < min_gap_from_robot)
    # if too close, push them out to min_gap_from_robot
    rand_r = torch.where(too_close, torch.full_like(rand_r, min_gap_from_robot), rand_r)
    offset_x = rand_r * torch.cos(rand_theta)
    offset_y = rand_r * torch.sin(rand_theta)

    # Build object poses: [N, num_objs, 7] (pos xyz, quat wxyz)
    # Start from default poses, just tweak XY
    # base position per env, expanded to match objects
    base_xy = base_pos_w[:, :2].unsqueeze(1)  # (N, 1, 2) -> (N, num_objs, 2) via broadcast

    # Build object poses: [N, num_objs, 7] (pos xyz, quat wxyz)
    obj_pose = obstacles.data.object_link_pose_w.clone()[env_ids]  # (N, num_objs, 7)

    # (x, y) for all obstacles
    obj_pose[..., 0] = base_xy[..., 0] + offset_x
    obj_pose[..., 1] = base_xy[..., 1] + offset_y

    # Now set z based on active / inactive
    z_active   = torch.full_like(obj_pose[..., 2], OBSTACLE_HEIGHT * 0.5)
    z_inactive = torch.full_like(obj_pose[..., 2], -10.0)

    obj_pose[..., 2] = torch.where(active_mask, z_active, z_inactive)

    # Write poses back to sim
    obstacles.write_object_pose_to_sim(
        object_pose=obj_pose,
        env_ids=env_ids,
        object_ids=None,  # all objects
    )
