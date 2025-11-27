from __future__ import annotations
import math
from dataclasses import dataclass, field
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils


class BaseVelocityCommand(CommandTerm):
    """Standard velocity command: (vx, vy, yaw_rate). Generated from goal direction."""

    @property
    def command(self):
        return self._command

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._num_envs = env.num_envs
        self._device = env.device
        self._command = torch.zeros(self._num_envs, 3, device=self._device)

    def _update_command(self):
        return self._command

    def _update_metrics(self):
        return {}

    def reset(self, env_ids):
        return {}

    def set_velocities(self, vx, vy, yaw_rate):
        """External interface: used by GoalCommand to set desired velocities."""
        self._command[:, 0] = vx
        self._command[:, 1] = vy
        self._command[:, 2] = yaw_rate


class GoalCommand(CommandTerm):

    @property
    def command(self):
        # This is what the rest of the system reads
        return self._command_tensor

    # ------------------------------------------------------------------
    # REQUIRED OVERRIDES
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: torch.Tensor):
        """Called by CommandManager on env reset. We *only* want to
        reset the timer here, not change the goal."""
        if env_ids.numel() == 0:
            return {}
        n = env_ids.numel()
        self._time_left[env_ids] = torch.empty(n, device=self._device).uniform_(
            *self.cfg.resampling_time_range
        )
        # keep command positions as-is
        return {}

    def _update_command(self):
        # IsaacLab doesn’t pass dt, so use env.step_dt
        dt = self._env.step_dt
        return self._compute(dt)

    def _update_metrics(self):
        # not using custom metrics
        return {}

    # ------------------------------------------------------------------
    # INTERNAL IMPLEMENTATION
    # ------------------------------------------------------------------

    def __init__(self, cfg: "GoalCommandCfg", env):
        super().__init__(cfg, env)

        self._device = self._env.device
        self._num_envs = self._env.num_envs

        # (x, y, heading) in *local* env frame
        self._command_tensor = torch.zeros(self._num_envs, 3, device=self._device)
        self._time_left = torch.zeros(self._num_envs, device=self._device)

        # for “was I externally overwritten?” debugging
        self._prev_command = self._command_tensor.clone()

        # Optional visual marker
        self._marker = VisualizationMarkers(cfg.visualizer_cfg) if cfg.debug_vis else None

        print("=== USING CUSTOM GOAL COMMAND ===")

        # Initial sample for all envs
        env_ids = torch.arange(self._num_envs, device=self._device)
        self._initial_resample(env_ids)

    # ------------------------------------------------------------------

    def _initial_resample(self, env_ids: torch.Tensor):
        """Used only at construction time."""
        n = env_ids.numel()
        if n == 0:
            return

        r = torch.empty(n, device=self._device).uniform_(*self.cfg.radius_range)
        theta = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)
        heading = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)

        self._command_tensor[env_ids, 0] = r * torch.cos(theta)
        self._command_tensor[env_ids, 1] = r * torch.sin(theta)
        self._command_tensor[env_ids, 2] = heading

        self._time_left[env_ids] = torch.empty(n, device=self._device).uniform_(
            *self.cfg.resampling_time_range
        )

        self._visualize()

    def _timer_resample(self, env_ids: torch.Tensor):
        """Timer-based resampling. Only called when time_left <= 0."""
        n = env_ids.numel()
        if n == 0:
            return

        r = torch.empty(n, device=self._device).uniform_(*self.cfg.radius_range)
        theta = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)
        heading = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)

        self._command_tensor[env_ids, 0] = r * torch.cos(theta)
        self._command_tensor[env_ids, 1] = r * torch.sin(theta)
        self._command_tensor[env_ids, 2] = heading

        self._time_left[env_ids] = torch.empty(n, device=self._device).uniform_(
            *self.cfg.resampling_time_range
        )

    # ------------------------------------------------------------------

    def _visualize(self):
        if not self._marker:
            return

        # Local command (N, 3)
        local = self._command_tensor

        # Env origins from IsaacLab (N, 3)
        env_origins = self._env.scene.env_origins.to(self._device)

        # Convert to world frame
        world = torch.zeros(self._num_envs, 3, device=self._device)
        world[:, :2] = local[:, :2] + env_origins[:, :2]
        world[:, 2] = 0.3   # fixed Z height

        self._marker.visualize(world)

    # ------------------------------------------------------------------

    def _compute(self, dt: float):
        # Debug: timer should start big and count down slowly
        # print("Before countdown, time_left[:10]:", self._time_left[:10])

        # countdown
        self._time_left -= dt

        # resample only when timer hits zero
        env_ids = torch.nonzero(self._time_left <= 0.0, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            # print("Resampling for envs:", env_ids)
            self._timer_resample(env_ids)

        # debug: detect external overwrites (shouldn’t happen)
        if torch.any(torch.ne(self._prev_command, self._command_tensor)):
            # comment this out once stable if it’s spammy
            print("Command changed (by timer or something else). Command[0]:", self._command_tensor[0])
        self._prev_command = self._command_tensor.clone()

        # update marker positions
        self._visualize()

        # === Convert goal direction → velocity command ===

        goal_xy = self._command_tensor[:, :2]   # local frame
        goal_dist = torch.norm(goal_xy, dim=1) + 1e-6
        goal_dir = goal_xy / goal_dist.unsqueeze(1)

        desired_vx = goal_dir[:, 0] * 0.6        # 0.6 m/s forward
        desired_vy = goal_dir[:, 1] * 0.2        # small sideways allowed
        desired_yaw = self._command_tensor[:, 2] # heading target from goal

        # write into base_velocity command (through command manager)
        #self._env.command_manager.commands["base_velocity"].set_velocities(
        #    desired_vx, desired_vy, desired_yaw
        #)


        return self._command_tensor


# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------

@dataclass
class GoalCommandCfg(CommandTermCfg):
    class_type: type = GoalCommand

    # Make these generous so you’re *sure* it’s not resampling constantly
    resampling_time_range: tuple[float, float] = (100.0, 100.0)
    radius_range: tuple[float, float] = (1.0, 5.0)
    debug_vis: bool = True

    visualizer_cfg: VisualizationMarkersCfg = field(
        default_factory=lambda: VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarker",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.15,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.2, 1.0)
                    ),
                ),
            },
        )
    )


