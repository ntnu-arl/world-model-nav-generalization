"""Collect a small dataset with the legacy pipeline (legacy env, MultiStrategyActor,
tools.simulate, ReplayBuffer with min_length) and save it, for comparison with
``wmnav.collect`` (tests/test_collect.py).

    python -m tests.collect_probe OUT.pt --policy safe_explore|uniform_biased \
        --level L2 --envs 16 --episodes 30 --seed 5
"""

import isaacgym  # noqa: F401

import argparse
import os
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--policy", default="safe_explore")
    ap.add_argument("--level", default="L2")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    out = Path(os.path.abspath(args.out))

    from tests import _legacy as L
    from tests.rollout_probe import legacy_env
    from tests import rollout_probe

    L.require_legacy_wms()
    import world_models.tools as tools
    # as main_collect_dataset.py: seed everything (deterministic_run True) before the env
    seed = tools.set_seed(args.seed, True)
    rollout_probe.N_ENVS, rollout_probe.SEED = args.envs, seed
    env = legacy_env("world_model", args.level)
    env.reset()

    from world_models.agents import MultiStrategyActor, create_multi_strategy_agent
    from world_models.memory.memory import ReplayBuffer
    from world_models.utils import LoggerDreamer

    buf = ReplayBuffer(64, 64, 2 ** 40, min_length=50, seed=seed)
    logger = LoggerDreamer(net_dir=tempfile.mkdtemp(), window_size=64)
    actor = MultiStrategyActor(num_envs=args.envs, device="cuda:0", dt=0.1)
    if args.policy == "uniform_biased":
        # the as-run high_freq variant (reconstructed in tests/test_exploration.py)
        from tests.test_exploration import legacy_high_freq_get_action
        actor.get_action = lambda depth: legacy_high_freq_get_action(actor, depth)
    agent = create_multi_strategy_agent(actor)
    tools.simulate(agent, env, buf, logger=logger, episodes=args.episodes, is_eval=False)
    buf.save(out.parent, name=out.name)


if __name__ == "__main__":
    main()
