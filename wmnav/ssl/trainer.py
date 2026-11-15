"""Self-supervised world-model pretraining on recorded episodes.

The loop of the paper's models (``TrainerSSL.wtrain`` in ssl_wms): ``epochs`` x
``iterations`` gradient steps, each on a fresh (batch_sz, batch_length) batch
sampled from the training episodes; after every epoch, ``eval_iterations``
validation batches. The objective is the full world-model loss (image, robot
state, reward and continue heads + KL).

Checkpoints: ``last.pt`` after every epoch, ``best_eval.pt`` / ``best_train.pt``
when the epoch's mean model loss improves. In ssl_wms the "best" bookkeeping
never updated its reference value, so ``agent_eval_best.pth`` and
``agent_train_best.pth`` were overwritten every epoch - the released SSL
checkpoints are the last-epoch weights. This trainer tracks the best properly;
``last.pt`` is the equivalent of the released files.
"""

import logging
import math
from pathlib import Path

import torch

from wmnav.data.replay_buffer import SSL_KEYS, ReplayBuffer
from wmnav.dreamer.checkpoints import save_checkpoint
from wmnav.dreamer.world_model import WorldModel
from wmnav.utils.metrics import MetricsAccumulator

log = logging.getLogger(__name__)


def load_episodes(files, batch_sz, batch_length, seed, pin_memory=False) -> ReplayBuffer:
    """A replay buffer with the SSL observation keys, loaded from ``files``."""
    buf = ReplayBuffer(batch_sz, batch_length, capacity_bytes=2 ** 62, seed=seed, keys=SSL_KEYS,
                       pin_memory=pin_memory)
    buf.load([str(f) for f in files])
    log.info("loaded %d episodes / %d frames from %d file(s)", len(buf), buf.num_steps, len(files))
    return buf


class SSLTrainer:
    """Trains a world model alone on recorded episodes: fixed numbers of updates per
    epoch, a validation pass after each, ``last`` / ``best_train`` / ``best_eval``
    checkpoints.
    """

    def __init__(self, cfg, run_dir: Path, logger=None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.logger = logger
        pin = bool(cfg.get("perf", {}).get("pin_memory", False))
        train = load_episodes(cfg.data.train, cfg.batch_sz, cfg.batch_length, cfg.data.seed, pin)
        valid = load_episodes(cfg.data.valid, cfg.batch_sz, cfg.batch_length, cfg.data.seed, pin)
        self.train, self.valid = train.on_device(cfg.device), valid.on_device(cfg.device)
        ep = next(iter(valid.data.values())).tensors
        shapes = {k: tuple(v.shape[1:]) for k, v in ep.items()}
        num_actions = shapes["action"][0]
        self.wm = WorldModel(shapes, num_actions, cfg.model).to(cfg.device)
        self.wm.requires_grad_(False)
        self.wm.configure_optimizer(cfg.optimizer)
        self.best = {"train": math.inf, "eval": math.inf}
        self.step = 0

    def evaluate(self) -> dict:
        """Mean world-model metrics over ``eval_iterations`` validation batches."""
        acc = MetricsAccumulator()
        with torch.no_grad():
            for _ in range(self.cfg.eval_iterations):
                acc.add(self.wm.eval_step(self.valid.sample()))
        return acc.flush()

    def _checkpoint(self, name, epoch, metrics):
        meta = {"epoch": epoch, "gradient_steps": self.step,
                **{k: v for k, v in metrics.items() if k == "model_loss"}}
        save_checkpoint(self.run_dir / "checkpoints" / f"{name}.pt", self.wm, model_config=self.cfg.model, meta=meta)

    def _maybe_best(self, phase, loss, epoch, metrics):
        if loss < self.best[phase]:
            self.best[phase] = loss
            self._checkpoint(f"best_{phase}", epoch, metrics)

    def fit(self) -> dict:
        """Train for ``epochs`` epochs; returns the final summary."""
        cfg = self.cfg
        ev = self.evaluate()
        self._log({f"eval/{k}": v for k, v in ev.items()}, epoch=0)
        self._maybe_best("eval", ev["model_loss"], 0, ev)

        for epoch in range(1, cfg.epochs + 1):
            acc = MetricsAccumulator()
            for _ in range(cfg.iterations):
                _, metrics = self.wm.train_step(self.train.sample())
                acc.add(metrics)
                self.step += 1
            tr = acc.flush()
            ev = self.evaluate()
            self._log({**{f"train/{k}": v for k, v in tr.items()}, **{f"eval/{k}": v for k, v in ev.items()}}, epoch)
            self._maybe_best("train", tr["model_loss"], epoch, tr)
            self._maybe_best("eval", ev["model_loss"], epoch, ev)
            self._checkpoint("last", epoch, ev)
            log.info("epoch %d/%d  train %.3f  eval %.3f", epoch, cfg.epochs, tr["model_loss"], ev["model_loss"])
        return {"best_train_loss": self.best["train"], "best_eval_loss": self.best["eval"], "gradient_steps": self.step}

    def _log(self, data, epoch):
        if self.logger is not None:
            data = dict(data, epoch=epoch)
            self.logger.log(data, step=self.step)
