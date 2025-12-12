from dataclasses import dataclass
import torch
from collections.abc import Sequence
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
# from isaaclab.markers.config import BLUE_SPHERE_MARKER_CFG  # or your custom marker
from isaaclab.utils import configclass

# Use the same mdp module as the base configs
import isaaclab.envs.mdp as mdp


class GaitParamCommand(CommandTerm):
    """
    Generates randomized gait parameters:
    [0] Body Height offset (m)
    [1] Step Frequency (Hz)
    [2] Foot Clearance (m)
    """

    def __init__(self, cfg: "GaitParamCommandCfg", env):
        super().__init__(cfg, env)

        # Pull ranges from cfg so they can be overridden by Hydra/YAML
        r = cfg.ranges
        self.height_range = r.height
        self.freq_range = r.freq
        self.clearance_range = r.clearance

        # Internal command buffer (num_envs, 3)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

        self.fixed_height = cfg.fixed_height

    # ---- required abstract property ---------------------------------
    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape: (num_envs, 3)."""
        return self._command

    # ---- abstract methods required by CommandTerm -------------------
    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        cmd = self._command

        # Convert env_ids to a tensor of indices on the right device
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = env_ids_t.shape[0]

        # -----------------------
        # 0: Height offset (with gating)
        # -----------------------
        p_height = 0.3  # 30% of envs get a non-zero height command

        # which envs in this batch will get a non-zero height?
        height_mask = (torch.rand(n, device=self.device) < p_height).float()

        # sample candidate heights
        height_samples = torch.empty(n, device=self.device).uniform_(*self.height_range)

        # write: height = sample * mask  (others → 0)
        cmd[env_ids_t, 0] = height_samples * height_mask

        # -----------------------
        # 1: Step frequency (always sample)
        # -----------------------
        freq_samples = torch.empty(n, device=self.device).uniform_(*self.freq_range)
        cmd[env_ids_t, 1] = freq_samples

        # -----------------------
        # 2: Foot clearance (always sample)
        # -----------------------
        clear_samples = torch.empty(n, device=self.device).uniform_(*self.clearance_range)
        cmd[env_ids_t, 2] = clear_samples

        # -----------------------
        # Debug print
        # -----------------------
        # with torch.no_grad():
        #     sample_envs = env_ids_t[:5]
        #     print(
        #         "[GaitParamCommand] resample: envs",
        #         sample_envs.tolist(),
        #         "height offsets",
        #         cmd[sample_envs, 0].tolist(),
        #         "freq",
        #         cmd[sample_envs, 1].tolist(),
        #         "clearance",
        #         cmd[sample_envs, 2].tolist(),
        #     )

    def _update_command(self):
        """
        Update the command every step.

        For piecewise-constant commands (only change on resample), this can be a no-op.
        If later you want to e.g. low-pass filter, you’d do it here.
        """
        if self.fixed_height is not None:
            # Hardcode crouch height for all envs
            self._command[:, 0] = self.fixed_height
            # Optionally lock freq/clearance as well:
            # self._command[:, 1] = 3.0    # mid-range frequency
            # self._command[:, 2] = 0.15   # mid-range clearance

    def _update_metrics(self):
        """Optional: log metrics if you care. For now, nothing."""
        # Example if you want later:
        # self.metrics.setdefault("height_mean", torch.zeros(self.num_envs, device=self.device))
        # self.metrics["height_mean"] += self._command[:, 0]
        pass


@configclass
class GaitParamCommandCfg(CommandTermCfg):
    """Configuration for the gait parameter command."""
    class_type: type = GaitParamCommand

    # how often to resample
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    debug_vis: bool = False

    fixed_height: float | None = None

    @configclass
    class Ranges:
        """Uniform ranges for gait parameters."""
        height: tuple[float, float] = (0, 0)   # m offset from nominal - Normally (-0.1, 0.1)
        freq: tuple[float, float] = (2.0, 4.0)      # Hz
        clearance: tuple[float, float] = (0.1, 0.20)  # m

    ranges: Ranges = Ranges()


# # 1. The Implementation Class (Logic)
# class GoalCommand(CommandTerm):
#     """
#     Generates a random (X,Y) goal within a radius.
#     """
#     cfg: "GoalCommandCfg"

#     def __init__(self, cfg: CommandTermCfg, env):
#         super().__init__(cfg, env)
#         # Create a visualization marker
#         self.marker = VisualizationMarkers(cfg.visualizer_cfg)

#     def _resample_command(self, env_ids):
#         """
#         Called when the timer runs out or robot reaches goal.
#         """
#         # Sample Random Radius (1m to 5m)
#         r = torch.empty(len(env_ids), device=self.device).uniform_(1.0, 5.0)
#         theta = torch.empty(len(env_ids), device=self.device).uniform_(0, 2 * 3.14159)

#         self.command[env_ids, 0] = r * torch.cos(theta)  # Goal X
#         self.command[env_ids, 1] = r * torch.sin(theta)  # Goal Y
#         self.command[env_ids, 2] = 0.0  # Heading (optional)

#     def _update_command(self, dt: float):
#         """
#         Called every step to update visuals.
#         """
#         if self.cfg.debug_vis:
#             self.marker.visualize(self.command[:, :3])

#     def _update_metrics(self):
#         # not using custom metrics
#         return {}

#     # ------------------------------------------------------------------
#     # INTERNAL IMPLEMENTATION
#     # ------------------------------------------------------------------

#     def __init__(self, cfg: "GoalCommandCfg", env):
#         super().__init__(cfg, env)

#         self._device = self._env.device
#         self._num_envs = self._env.num_envs

#         # (x, y, heading) in *local* env frame
#         self._command_tensor = torch.zeros(self._num_envs, 3, device=self._device)
#         self._time_left = torch.zeros(self._num_envs, device=self._device)

#         # for “was I externally overwritten?” debugging
#         self._prev_command = self._command_tensor.clone()

#         # Optional visual marker
#         self._marker = VisualizationMarkers(cfg.visualizer_cfg) if cfg.debug_vis else None

#         print("=== USING CUSTOM GOAL COMMAND ===")

#         # Initial sample for all envs
#         env_ids = torch.arange(self._num_envs, device=self._device)
#         self._initial_resample(env_ids)

#     # ------------------------------------------------------------------

#     def _initial_resample(self, env_ids: torch.Tensor):
#         """Used only at construction time."""
#         n = env_ids.numel()
#         if n == 0:
#             return

#         r = torch.empty(n, device=self._device).uniform_(*self.cfg.radius_range)
#         theta = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)
#         heading = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)

#         self._command_tensor[env_ids, 0] = r * torch.cos(theta)
#         self._command_tensor[env_ids, 1] = r * torch.sin(theta)
#         self._command_tensor[env_ids, 2] = heading

#         self._time_left[env_ids] = torch.empty(n, device=self._device).uniform_(
#             *self.cfg.resampling_time_range
#         )

#         self._visualize()

#     def _timer_resample(self, env_ids: torch.Tensor):
#         """Timer-based resampling. Only called when time_left <= 0."""
#         n = env_ids.numel()
#         if n == 0:
#             return

#         r = torch.empty(n, device=self._device).uniform_(*self.cfg.radius_range)
#         theta = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)
#         heading = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)

#         self._command_tensor[env_ids, 0] = r * torch.cos(theta)
#         self._command_tensor[env_ids, 1] = r * torch.sin(theta)
#         self._command_tensor[env_ids, 2] = heading

#         self._time_left[env_ids] = torch.empty(n, device=self._device).uniform_(
#             *self.cfg.resampling_time_range
#         )

#     # ------------------------------------------------------------------

#     def _visualize(self):
#         if not self._marker:
#             return

#         # Local command (N, 3)
#         local = self._command_tensor

#         # Env origins from IsaacLab (N, 3)
#         env_origins = self._env.scene.env_origins.to(self._device)

#         # Convert to world frame
#         world = torch.zeros(self._num_envs, 3, device=self._device)
#         world[:, :2] = local[:, :2] + env_origins[:, :2]
#         world[:, 2] = 0.3   # fixed Z height

#         self._marker.visualize(world)

#     # ------------------------------------------------------------------

#     def _compute(self, dt: float):
#         # Debug: timer should start big and count down slowly
#         # print("Before countdown, time_left[:10]:", self._time_left[:10])

#         # countdown
#         self._time_left -= dt

#         # resample only when timer hits zero
#         env_ids = torch.nonzero(self._time_left <= 0.0, as_tuple=False).squeeze(-1)
#         if env_ids.numel() > 0:
#             # print("Resampling for envs:", env_ids)
#             self._timer_resample(env_ids)

#         # debug: detect external overwrites (shouldn’t happen)
#         #if torch.any(torch.ne(self._prev_command, self._command_tensor)):
#         #    comment this out once stable if it’s spammy
#         #    print("Command changed (by timer or something else). Command[0]:", self._command_tensor[0])
#         self._prev_command = self._command_tensor.clone()

#         # update marker positions
#         self._visualize()

#         # === Convert goal direction → velocity command ===

#         goal_xy = self._command_tensor[:, :2]   # local frame
#         goal_dist = torch.norm(goal_xy, dim=1) + 1e-6
#         goal_dir = goal_xy / goal_dist.unsqueeze(1)

#         desired_vx = goal_dir[:, 0] * 0.6        # 0.6 m/s forward
#         desired_vy = goal_dir[:, 1] * 0.2        # small sideways allowed
#         desired_yaw = self._command_tensor[:, 2] # heading target from goal

#         # write into base_velocity command (through command manager)
#         #self._env.command_manager.commands["base_velocity"].set_velocities(
#         #    desired_vx, desired_vy, desired_yaw
#         #)


# # 2. The Configuration Class (Settings)
# @dataclass
# class GoalCommandCfg(CommandTermCfg):
#     # [CRITICAL] This links the Config to the Logic class above!
#     class_type: type = GoalCommand

#     # Default settings
#     resampling_time_range: tuple[float, float] = (5.0, 10.0)
#     visualizer_cfg: object = BLUE_SPHERE_MARKER_CFG
#     debug_vis: bool = True
#     num_commands: int = 3
