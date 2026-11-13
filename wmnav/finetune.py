"""Fine-tune a pretrained world model with the actor-critic (``configs/finetune.yaml``).

Order of the IROS26 runs: prefill the replay buffer with random actions; build the
agent and load the SSL world model (the actor-critic starts from scratch); one
deterministic evaluation of ``eval_episodes`` episodes; then ``episodes`` training
episodes, gradient steps interleaved with acting. ``best_train_win_rate.pt`` -
the checkpoint the paper evaluates - is written whenever the mean win rate of the
last ``best_window`` training episodes reaches a new maximum.
"""

import isaacgym  # noqa: F401  (before torch)

import functools
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from wmnav.data.exploration import UniformRandomPolicy
from wmnav.data.replay_buffer import ReplayBuffer
from wmnav.data.rollout import BestTracker, OutcomeCounter, simulate
from wmnav.dreamer.agent import DreamerAgent
from wmnav.env.factory import make_env_from_cfg
from wmnav.utils.run import RunLogger
from wmnav.utils.setup import model_cfg_for, observation_shapes, start_run, write_json

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    """Fine-tune a pretrained world model with an actor-critic (``configs/finetune.yaml``)."""
    run_dir, seed = start_run(cfg)
    with RunLogger(run_dir, cfg.logging.project, f"finetune-{cfg.run_name}",
                   OmegaConf.to_container(cfg, resolve=True), cfg.logging.wandb_mode) as logger:
        summary = finetune(cfg, run_dir, seed, logger)
        logger.summary(summary)
        write_json(run_dir / "summary.json", summary)


def finetune(cfg: DictConfig, run_dir: Path, seed: int, logger: RunLogger) -> dict:
    """Run the fine-tuning loop (prefill, world-model pretraining steps, training with
    periodic evaluation and checkpoints) and return the run summary.
    """
    ckpt_dir = run_dir / "checkpoints"
    e, t = cfg.env, cfg.train
    env = make_env_from_cfg(e, seed)

    pin = bool(cfg.perf.pin_memory)
    train_buffer = ReplayBuffer(t.batch_sz, t.batch_length, t.buffer_bytes, seed=seed, pin_memory=pin)
    train_best = BestTracker(t.best_window)
    eval_best = BestTracker(t.best_window)

    # 1. random-action prefill (counts towards the training moving averages, as before)
    simulate(UniformRandomPolicy(env.num_envs), env, train_buffer, steps=t.prefill, on_episodes=train_best.add)

    # 2. agent: SSL world model + fresh actor-critic
    model_cfg = model_cfg_for(cfg.model, cfg.init_from)
    agent = DreamerAgent(observation_shapes(env), env.action_space.shape[0], model_cfg, t,
                         replay=train_buffer.on_device(e.device))
    agent.to(e.device)
    agent.configure_optimizers(t.model_opt)
    agent.load(cfg.init_from, world_model_only=True)

    def saver(tracker, phase):
        """Callback that feeds finished episodes to ``tracker`` and, once its window is
        full, logs the ``phase`` means and saves a checkpoint on a new best."""
        def on_episodes(outcomes):
            res = tracker.add(outcomes)
            if not res["full"]:
                return
            logger.log({f"{phase}/avg_{k}": v for k, v in res["means"].items()}, step=agent.step)
            meta = {"phase": phase, "score_r": tracker.best_reward, "score_w": tracker.best_win_rate}
            if res["new_best_reward"]:
                agent.save(ckpt_dir / f"best_{phase}_reward.pt", meta)
            if res["new_best_win_rate"]:
                agent.save(ckpt_dir / f"best_{phase}_win_rate.pt", meta)
        return on_episodes

    # 3. deterministic evaluation before training
    eval_counter = OutcomeCounter()
    on_eval = saver(eval_best, "eval")

    def on_eval_episodes(outcomes):
        """Best-checkpoint tracking plus the counts of the initial evaluation."""
        on_eval(outcomes)
        eval_counter.add(outcomes)

    with torch.no_grad():
        simulate(functools.partial(agent, training=False), env, None, episodes=t.eval_episodes,
                 on_episodes=on_eval_episodes)
    logger.log({f"initial_eval/{k}": v for k, v in eval_counter.summary().items()}, step=agent.step)

    # 4. training
    last_logged = 0

    def on_step(frames):
        """Log the training metrics every ``logging.log_every`` env steps."""
        nonlocal last_logged
        env_steps = frames // env.num_envs
        if env_steps - last_logged >= cfg.logging.log_every:
            last_logged = env_steps
            logger.log({**agent.metrics.flush(), **{f"env/{k}": v for k, v in env.episode_stats(reset=True).items()},
                        "updates": agent.update_count}, step=agent.step)

    result = simulate(agent, env, train_buffer, episodes=t.episodes, on_episodes=saver(train_best, "train"),
                      on_step=on_step)
    agent.save(ckpt_dir / "last.pt", {"phase": "train"})
    return {"env_frames": agent.step, "updates": agent.update_count, "train_episodes": result["episodes"],
            "best_train_win_rate": train_best.best_win_rate, "best_train_reward": train_best.best_reward}

if __name__ == "__main__":
    main()
