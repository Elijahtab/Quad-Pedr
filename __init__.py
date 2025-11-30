# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from . import agents
from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from .flat_env_cfg import UnitreeGo2FlatEnvCfg   
from .agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
    UnitreeGo2FlatPPORunnerCfg,                    
)


def make_go2_nav_env(**kwargs):
    """
    Gym entry point for Isaac-Nav-Go2-*-v0.

    Hydra passes env_cfg via env_cfg_entry_point; we just forward cfg to ManagerBasedRLEnv.
    """
    incoming_cfg = kwargs.pop("cfg", None)
    if incoming_cfg is not None:
        cfg = incoming_cfg
    else:
        # Fallback: rough if nothing is passed (doesn’t matter much once Hydra is set up)
        cfg = UnitreeGo2RoughEnvCfg()

    num_envs = kwargs.pop("num_envs", None)
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    kwargs.pop("env_cfg_entry_point", None)
    kwargs.pop("rsl_rl_cfg_entry_point", None)

    env = ManagerBasedRLEnv(cfg=cfg, **kwargs)
    return env


# Flat task – Milestone 1 & 2
gym.register(
    id="Isaac-Nav-Go2-Flat-v0",
    entry_point="isaaclab_tasks.go2_navigation:make_go2_nav_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.go2_navigation.flat_env_cfg:UnitreeGo2FlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.go2_navigation.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg"
        ),
    },
)

# Rough task – Milestone 2.5 & 3
gym.register(
    id="Isaac-Nav-Go2-Rough-v0",
    entry_point="isaaclab_tasks.go2_navigation:make_go2_nav_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.go2_navigation.rough_env_cfg:UnitreeGo2RoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.go2_navigation.agents.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
        ),
    },
)

