from dataclasses import dataclass
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_SPHERE_MARKER_CFG # or your custom marker

# 1. The Implementation Class (Logic)
class GoalCommand(CommandTerm):
    """
    Generates a random (X,Y) goal within a radius.
    """
    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        # Create a visualization marker
        self.marker = VisualizationMarkers(cfg.visualizer_cfg)

    def _resample_command(self, env_ids):
        """
        Called when the timer runs out or robot reaches goal.
        """
        # Sample Random Radius (1m to 5m)
        r = torch.empty(len(env_ids), device=self.device).uniform_(1.0, 5.0)
        theta = torch.empty(len(env_ids), device=self.device).uniform_(0, 2*3.14159)
        
        self.command[env_ids, 0] = r * torch.cos(theta) # Goal X
        self.command[env_ids, 1] = r * torch.sin(theta) # Goal Y
        self.command[env_ids, 2] = 0.0 # Heading (optional)

    def _update_command(self):
        """
        Called every step to update visuals.
        """
        self.marker.visualize(self.command[:, :3])

# 2. The Configuration Class (Settings)
@dataclass
class GoalCommandCfg(CommandTermCfg):
    # [CRITICAL] This links the Config to the Logic class above!
    class_type: type = GoalCommand 
    
    # Default settings
    resampling_time_range: tuple[float, float] = (5.0, 10.0)
    visualizer_cfg: object = BLUE_SPHERE_MARKER_CFG
    debug_vis: bool = True
