"""sample-factory glue: env registration, the vectorised-env adapter, config."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import torch
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env

from wmnav.env.factory import TASK_NAME, register
from wmnav.ppo.encoder import INPUTS, make_encoder

# the task instance sample-factory created in this process, for reading episode stats
CREATED: List = []


class VecEnv(gym.Env):
    """The navigation task as one multi-agent sample-factory env (one agent per sim env).

    With ``stats_file`` set, every ``stats_every`` env steps it appends the episodes
    finished since the last line and their win / crash / timeout rates (the
    training curve; ``frames`` counts from the start of this process).
    """

    def __init__(self, task, stats_file: Optional[str] = None, stats_every: int = 200):
        self.env = task
        self.num_agents = task.num_envs
        self.action_space = convert_space(task.action_space)
        self.keys = [key for _, key in INPUTS]
        self.observation_space = gym.spaces.Dict({k: convert_space(task.observation_space[k]) for k in self.keys})
        self.stats_file, self.stats_every = stats_file, stats_every
        self._steps = 0

    def _obs(self, obs) -> Dict[str, torch.Tensor]:
        return {k: obs[k] for k in self.keys}

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        return self._obs(obs), info

    def step(self, action):
        obs, rew, terminated, truncated, infos = self.env.step(action)
        self._steps += 1
        if self.stats_file and self._steps % self.stats_every == 0:
            stats = self.env.episode_stats(reset=True)
            stats.update(frames=self._steps * self.num_agents, time=time.time())
            with open(self.stats_file, "a") as f:
                f.write(json.dumps(stats) + "\n")
        return self._obs(obs), rew, terminated, truncated, infos

    def render(self):
        pass


def _make(full_env_name, cfg, env_config=None, render_mode: Optional[str] = None):
    """Env factory. sample-factory also calls it in a spawned process (to read the
    observation/action spaces), where nothing has been registered with aerial_gym
    yet, so the task is registered here from parameters carried in the config."""
    from aerial_gym.registry.task_registry import task_registry
    register(**json.loads(cfg.wmnav_env))
    stats_file = None
    if not getattr(cfg, "no_render", None) and getattr(cfg, "train_dir", None) and getattr(cfg, "experiment", None):
        exp_dir = Path(cfg.train_dir) / cfg.experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = str(exp_dir / "episode_stats.jsonl")
    env = VecEnv(task_registry.make_task(TASK_NAME), stats_file=stats_file)
    CREATED.append(env.env)
    return env


def build_cfg(sf_defaults: dict, argv: List[str], env_kwargs: dict, evaluation: bool = False):
    """sample-factory config from our defaults plus explicit arguments.

    ``env_kwargs`` (the arguments of ``wmnav.env.factory.register``) travel inside
    the config as ``--wmnav_env``, for the processes sample-factory spawns.
    """
    register_env(TASK_NAME, _make)
    global_model_factory().register_encoder_factory(make_encoder)
    argv = list(argv) + [f"--wmnav_env={json.dumps(env_kwargs)}"]
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    parser.add_argument("--wmnav_env", type=str, default=None, help="wmnav environment parameters (JSON)")
    parser.set_defaults(**{k: v for k, v in sf_defaults.items() if not isinstance(v, dict)})
    return parse_full_cfg(parser, argv=argv)
