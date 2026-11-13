"""Collect exploration episodes for SSL pretraining (``configs/collect.yaml``)."""

import isaacgym  # noqa: F401  (before torch)

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from wmnav.data.exploration import ExplorationPolicy
from wmnav.data.replay_buffer import ReplayBuffer
from wmnav.data.rollout import OutcomeCounter, simulate
from wmnav.env.factory import make_env_from_cfg
from wmnav.utils.setup import start_run, write_json

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="collect")
def main(cfg: DictConfig) -> None:
    """Collect exploration episodes into a replay-buffer file (``configs/collect.yaml``)."""
    run_dir, seed = start_run(cfg)
    e = cfg.env
    env = make_env_from_cfg(e, seed)

    buffer = ReplayBuffer(1, 1, cfg.buffer_bytes, min_length=cfg.min_length, seed=seed)
    policy = ExplorationPolicy(env.num_envs, e.device, cfg.policy)
    counter = OutcomeCounter()
    with torch.no_grad():
        result = simulate(policy, env, buffer, episodes=cfg.episodes, on_episodes=counter.add)

    output = Path(cfg.output)
    out = buffer.save(output.parent, output.name)
    summary = {"output": str(out), "episodes_kept": result["episodes"], "frames": buffer.num_steps,
               "episodes_stored": len(buffer), **{f"collected_{k}": v for k, v in counter.summary().items()}}
    write_json(run_dir / "summary.json", summary)
    log.info("saved %s: %d episodes (incl. unfinished), %d frames", out, len(buffer), buffer.num_steps)


if __name__ == "__main__":
    main()
