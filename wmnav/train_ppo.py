"""Train the PPO baseline with sample-factory (``configs/train_ppo.yaml``).

With ``snapshot_at`` set, training first runs to that many frames, copies the
checkpoint to ``<experiment>-snapshot`` (a standalone experiment that
``wmnav.eval_ppo`` can evaluate), then continues to ``env_steps``. The second
phase runs in a fresh process (the command re-executes itself): Isaac Gym cannot
build a second simulator in a process that already built one.
"""

import isaacgym  # noqa: F401  (before torch)

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from wmnav.env.factory import env_kwargs
from wmnav.ppo.sample_factory import build_cfg
from wmnav.utils.setup import start_run

log = logging.getLogger(__name__)


def _train(cfg, seed, steps):
    from sample_factory.train import run_rl

    argv = ["--env=static_env", f"--experiment={cfg.experiment}", f"--train_dir={cfg.train_dir}",
            f"--train_for_env_steps={int(steps)}", f"--seed={seed}", f"--restart_behavior={cfg.restart_behavior}"]
    sf_cfg = build_cfg(OmegaConf.to_container(cfg.sample_factory, resolve=True), argv,
                       env_kwargs(cfg.env, seed, mode="ppo"))
    status = run_rl(sf_cfg)
    if status != 0:
        raise SystemExit(f"sample-factory exited with status {status}")


def _snapshot(cfg):
    src = Path(cfg.train_dir) / cfg.experiment
    dst = Path(cfg.train_dir) / f"{cfg.experiment}-snapshot"
    if dst.exists():
        log.info("snapshot %s exists, kept", dst)
        return
    dst.mkdir(parents=True)
    shutil.copytree(src / "checkpoint_p0", dst / "checkpoint_p0")
    sf_cfg = json.loads((src / "config.json").read_text())
    sf_cfg["experiment"] = dst.name  # the evaluator resolves paths from it
    (dst / "config.json").write_text(json.dumps(sf_cfg, indent=2))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_ppo")
def main(cfg: DictConfig) -> None:
    """Train the PPO baseline (``configs/train_ppo.yaml``)."""
    _, seed = start_run(cfg)
    snap = cfg.get("snapshot_at")
    if snap and not (Path(cfg.train_dir) / f"{cfg.experiment}-snapshot").exists():
        _train(cfg, seed, snap)
        _snapshot(cfg)
        # continue in a fresh process, with the seed this run resolved
        keep = [a for a in sys.argv[1:] if a.split("=", 1)[0].lstrip("+~") not in ("snapshot_at", "seed")]
        args = [sys.executable, "-m", "wmnav.train_ppo", *keep, "snapshot_at=null", f"seed={seed}"]
        log.info("phase 2 in a new process: %s", " ".join(args))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, args)
    _train(cfg, seed, cfg.env_steps)


if __name__ == "__main__":
    main()
