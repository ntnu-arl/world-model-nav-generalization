"""SSL world-model pretraining (``configs/pretrain.yaml``). Needs no simulator."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from wmnav.ssl.trainer import SSLTrainer
from wmnav.utils.run import RunLogger
from wmnav.utils.setup import start_run, write_json

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    """Self-supervised world-model pretraining on recorded episodes (``configs/pretrain.yaml``)."""
    run_dir, _ = start_run(cfg)
    with RunLogger(run_dir, cfg.logging.project, f"pretrain-{cfg.run_name}",
                   OmegaConf.to_container(cfg, resolve=True), cfg.logging.wandb_mode) as logger:
        summary = SSLTrainer(cfg, run_dir, logger).fit()
        logger.summary(summary)
        write_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
