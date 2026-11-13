"""Run a fixed action sequence in the legacy or the new environment and save a
per-step record; used by tests/test_env_rollout.py to compare the two.

    python -m tests.rollout_probe {legacy|new} OUT.pt [--mode world_model|ppo] [--level L3]
"""

import isaacgym  # noqa: F401

import argparse
import os

import torch

N_ENVS, N_STEPS, SEED = 16, 400, 5


def legacy_env(mode, level):
    from tests import _legacy as L
    L.require_legacy_wms()
    os.chdir(L.LEGACY_WMS)  # legacy configs are read relative to the repo root
    import wandb
    wandb.init(mode="disabled")
    from omegaconf import OmegaConf
    from aerial_gym.robots.base_multirotor import BaseMultirotor
    from aerial_gym.registry.env_registry import env_config_registry
    from aerial_gym.registry.robot_registry import robot_registry
    from aerial_gym.registry.sim_registry import sim_config_registry
    from aerial_gym.registry.task_registry import task_registry
    from src.configs.assets.dataset_dynamic import pillar_params
    from src.configs.envs.levels import apply_level
    from src.configs.envs.static_env import StaticEnvCfg
    from src.configs.robots.lmf2 import LMF2Cfg
    from src.configs.sims.procedural_world import BaseSimConfig
    from src.configs.tasks.static_env import task_config
    from src.tasks.static_env_task import StaticEnv
    from src.utils.merge_settings import merge_settings, yaml_file

    apply_level(level, StaticEnvCfg, pillar_params, n_pillars=5)
    task_config.num_envs = N_ENVS
    task_config.seed = SEED
    task_config.reset_in_step = mode == "ppo"
    task_config.trjs_cfg = yaml_file("./src/configs/cfgs/", filename="trajectories.yaml")
    cfgs = {"seed": SEED}
    cfg_env = {"task": {"name": "static_env", "class": StaticEnv, "config": task_config},
               "env": {"name": "static_env", "config": StaticEnvCfg},
               "sim": {"name": "base_sim", "config": BaseSimConfig},
               "robot": {"name": "lmf2", "class": BaseMultirotor, "config": LMF2Cfg}}
    merge_settings(cfg_env, cfgs)
    task_config.cfgs = cfgs
    task_config.cfgs_custom_algo = OmegaConf.create({"average_to_save": 64, "print": 0, "debug": False})
    task_registry.register_task("static_env", StaticEnv, task_config)
    env_config_registry.register("static_env", StaticEnvCfg)
    sim_config_registry.register("base_sim", BaseSimConfig)
    robot_registry.register("lmf2", BaseMultirotor, LMF2Cfg)
    return task_registry.make_task("static_env")


def new_env(mode, level):
    from wmnav.env.factory import make_env
    return make_env(mode=mode, level=level, num_panels=5, num_envs=N_ENVS, seed=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("impl", choices=["legacy", "new"])
    ap.add_argument("out")
    ap.add_argument("--mode", default="world_model")
    ap.add_argument("--level", default="L3")
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    env = (legacy_env if args.impl == "legacy" else new_env)(args.mode, args.level)
    env.reset()
    g = torch.Generator(device="cuda:0").manual_seed(0)
    done = torch.ones(N_ENVS, dtype=torch.bool, device="cuda:0")
    record = {k: [] for k in ("reward", "done", "terminated", "truncated", "robot_state", "depth_sum",
                              "is_first", "is_terminal", "action_agent")}
    for _ in range(N_STEPS):
        if args.mode == "world_model" and done.any():
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            env.reset_idx(idx)
        # mostly forward, some climbing and turning: reaches targets and walls
        a = torch.rand(N_ENVS, 4, device="cuda:0", generator=g) * 2 - 1
        a[:, 0] = a[:, 0] * 0.5 + 0.5
        out_t = env.step(a)
        if args.mode == "world_model":
            obs, rew, done, info = out_t
            term, trunc = obs["is_terminal"], torch.zeros_like(done)
        else:
            obs, rew, term, trunc, info = out_t
            done = torch.logical_or(term.bool(), trunc.bool())
        record["reward"].append(rew.cpu())
        record["done"].append(done.cpu())
        record["terminated"].append(term.cpu())
        record["truncated"].append(trunc.cpu())
        record["robot_state"].append(obs["robot_state"].cpu())
        record["depth_sum"].append(obs["depth_image"].double().sum(dim=(1, 2, 3)).cpu())
        record["is_first"].append(obs["is_first"].cpu())
        record["is_terminal"].append(obs["is_terminal"].cpu())
        record["action_agent"].append(obs["action_agent"].cpu())
    torch.save({k: torch.stack(v) for k, v in record.items()}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
