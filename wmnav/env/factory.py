"""Register the navigation task with aerial_gym and build environments.

    from wmnav.env.factory import make_env
    env = make_env(mode="world_model", level="L3", num_panels=5, num_envs=64, seed=5)

Entry points build it from their ``env`` config group with ``make_env_from_cfg``.
"""

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.robots.base_multirotor import BaseMultirotor

from wmnav.env.configs.env import build_env_cfg
from wmnav.env.configs.robot import LMF2Cfg
from wmnav.env.configs.sim import SimCfg
from wmnav.env.configs.task import make_task_cfg
from wmnav.env.levels import TRAINING_PANELS
from wmnav.env.task import NavigationTask
from wmnav.env import patches

# the sample-factory experiment configs of the released PPO runs name this task
TASK_NAME = "static_env"


def register(mode: str, level: str, num_panels: int = TRAINING_PANELS, num_envs: int = 64, seed: int = 5,
             device: str = "cuda:0", headless: bool = True, episode_len_steps: int = 300):
    """Register sim, scene, robot and task; returns the task config class."""
    patches.apply()
    task_cfg = make_task_cfg(mode, num_envs, seed, device, headless, episode_len_steps)
    sim_config_registry.register(task_cfg.sim_name, SimCfg)
    env_config_registry.register(task_cfg.env_name, build_env_cfg(level, num_panels))
    robot_registry.register(task_cfg.robot_name, BaseMultirotor, LMF2Cfg)
    task_registry.register_task(TASK_NAME, NavigationTask, task_cfg)
    return task_cfg


def make_env(mode: str, level: str, num_panels: int = TRAINING_PANELS, num_envs: int = 64, seed: int = 5,
             device: str = "cuda:0", headless: bool = True, episode_len_steps: int = 300) -> NavigationTask:
    """Register the task for this level / panel count and create it."""
    register(mode, level, num_panels, num_envs, seed, device, headless, episode_len_steps)
    return task_registry.make_task(TASK_NAME)


def env_kwargs(env_cfg, seed: int, mode: str = "world_model") -> dict:
    """``register`` / ``make_env`` arguments from an ``env`` config group (plain types,
    so they can travel as JSON to the processes sample-factory spawns)."""
    return {"mode": mode, "level": env_cfg.level, "num_panels": int(env_cfg.num_panels),
            "num_envs": int(env_cfg.num_envs), "seed": int(seed), "device": env_cfg.device,
            "headless": bool(env_cfg.headless), "episode_len_steps": int(env_cfg.episode_len_steps)}


def make_env_from_cfg(env_cfg, seed: int, mode: str = "world_model") -> NavigationTask:
    """Build and reset the task described by an ``env`` config group."""
    env = make_env(**env_kwargs(env_cfg, seed, mode))
    env.reset()
    return env
