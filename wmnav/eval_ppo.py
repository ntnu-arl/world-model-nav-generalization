"""Deterministic evaluation of a PPO experiment (``configs/eval_ppo.yaml``).

Episodes are counted by the task itself (every episode that ends in any env),
so ``summary.json`` has the same games / wins / crashes / timeouts as the world
model evaluation.
"""

import isaacgym  # noqa: F401  (before torch)

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from wmnav.env.factory import env_kwargs
from wmnav.ppo import sample_factory as sf
from wmnav.utils.setup import start_run, write_json

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_ppo")
def main(cfg: DictConfig) -> None:
    """Evaluate a trained PPO experiment deterministically (``configs/eval_ppo.yaml``)."""
    from sample_factory.enjoy import enjoy

    run_dir, seed = start_run(cfg)
    e = cfg.env
    argv = ["--env=static_env", f"--experiment={cfg.experiment}", f"--train_dir={cfg.train_dir}",
            f"--load_checkpoint_kind={cfg.checkpoint}", "--eval_deterministic=True",
            f"--max_num_episodes={int(cfg.episodes)}", "--no_render"]
    sf_cfg = sf.build_cfg(OmegaConf.to_container(cfg.sample_factory, resolve=True), argv,
                          env_kwargs(cfg.env, seed, mode="ppo"), evaluation=True)
    status, avg_reward = enjoy(sf_cfg)
    task = sf.CREATED[-1]
    stats = task.episode_stats()
    summary = {"experiment": cfg.experiment, "level": e.level, "num_panels": e.num_panels,
               "checkpoint": cfg.checkpoint, "status": int(status), "sf_avg_reward": float(avg_reward),
               "games": int(stats["episodes"]), **{k: v for k, v in stats.items() if k != "episodes"}}
    write_json(run_dir / "summary.json", summary)
    log.info("%s on %s with %d panels: %d games, win %.3f, crash %.3f, timeout %.3f", cfg.experiment, e.level,
             e.num_panels, summary["games"], stats["win_rate"], stats["crash_rate"], stats["timeout_rate"])


if __name__ == "__main__":
    main()
