"""Deterministic evaluation of a fine-tuned agent (``configs/evaluate.yaml``).

Writes ``summary.json`` (games, wins, crashes, timeouts, rates, mean episode
length/return) and ``episodes.csv`` (one row per finished episode). With
``record_trajectories`` it also saves every step together with the obstacle
layout, for ``wmnav.analysis.clearance``.
"""

import isaacgym  # noqa: F401  (before torch)

import csv
import functools
import logging

import hydra
import torch
from omegaconf import DictConfig

from wmnav.data.replay_buffer import ReplayBuffer
from wmnav.data.rollout import OutcomeCounter, layout_specs, simulate
from wmnav.dreamer.agent import DreamerAgent
from wmnav.env.factory import make_env_from_cfg
from wmnav.utils.setup import model_cfg_for, observation_shapes, start_run, write_json

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Evaluate a Dreamer checkpoint (``configs/evaluate.yaml``)."""
    run_dir, seed = start_run(cfg)
    e = cfg.env
    env = make_env_from_cfg(e, seed)

    model_cfg = model_cfg_for(cfg.model, cfg.checkpoint)
    agent = DreamerAgent(observation_shapes(env), env.action_space.shape[0], model_cfg).to(e.device)
    meta = agent.load(cfg.checkpoint)["meta"]

    buffer = None
    if cfg.record_trajectories:
        buffer = ReplayBuffer(1, 1, 2 ** 62, extra_specs=layout_specs(e.num_panels))

    counter = OutcomeCounter()
    rows = []

    def on_episodes(outcomes):
        """Count the finished episodes and keep one row per episode."""
        counter.add(outcomes)
        rows.extend({k: float(v[i]) for k, v in outcomes.items()} for i in range(len(outcomes["wins"])))

    with torch.no_grad():
        simulate(functools.partial(agent, training=False), env, buffer, episodes=cfg.episodes,
                 on_episodes=on_episodes, record_layout=bool(cfg.record_trajectories))

    summary = {"name": cfg.name, "level": e.level, "num_panels": e.num_panels, "checkpoint": str(cfg.checkpoint),
               "checkpoint_meta": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in meta.items()},
               **counter.summary()}
    write_json(run_dir / "summary.json", summary)
    with open(run_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rewards", "wins", "terminations", "truncations", "dist", "length"])
        w.writeheader()
        w.writerows(rows)
    if buffer is not None:
        buffer.save(run_dir, "trajectories.pt")
    log.info("%s on %s with %d panels: %d games, win %.3f, crash %.3f, timeout %.3f", cfg.name, e.level,
             e.num_panels, counter.games, summary["win_rate"], summary["crash_rate"], summary["timeout_rate"])


if __name__ == "__main__":
    main()
