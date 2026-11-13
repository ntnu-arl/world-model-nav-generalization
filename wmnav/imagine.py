"""Open-loop prediction quality of a world model on recorded episodes
(``configs/imagine.yaml``). Needs no simulator.

Writes ``summary.json`` (MSE/SSIM over total / context / imagination) and
``episodes.csv``; with ``save_predictions`` also one ``.npy`` per episode with the
true and predicted images, robot state, reward and continue flag.
"""

import csv
import logging

import hydra
import torch
from omegaconf import DictConfig

from wmnav.analysis.imagination import ImaginationScores
from wmnav.data.replay_buffer import SSL_KEYS, ReplayBuffer
from wmnav.dreamer.checkpoints import load_checkpoint
from wmnav.dreamer.world_model import WorldModel
from wmnav.utils.setup import model_cfg_for, start_run, write_json

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="imagine")
def main(cfg: DictConfig) -> None:
    """Score open-loop predictions of a world model (``configs/imagine.yaml``)."""
    run_dir, _ = start_run(cfg)
    data = ReplayBuffer(1, 1, 2 ** 62, keys=SSL_KEYS)
    data.load([str(p) for p in cfg.data])
    trajectories, ids = data.get_full_trajectories(min_length=cfg.context + cfg.min_after_context)
    if cfg.max_episodes:
        trajectories, ids = trajectories[: cfg.max_episodes], ids[: cfg.max_episodes]
    log.info("%d eligible episodes", len(trajectories))

    model_cfg = model_cfg_for(cfg.model, cfg.checkpoint)
    shapes = {k: tuple(v.shape[2:]) for k, v in trajectories[0].items()}
    wm = WorldModel(shapes, shapes["action"][0], model_cfg).to(cfg.device)
    wm.load_state_dict(load_checkpoint(cfg.checkpoint)["world_model"])
    wm.eval()

    scores = ImaginationScores(cfg.context, cfg.depth_space)
    save_dir = run_dir / "predictions" if cfg.save_predictions else None
    if save_dir:
        save_dir.mkdir()
    rows = []
    with torch.no_grad():
        for traj, episode_id in zip(trajectories, ids):
            pred = wm.video_pred(traj, context=cfg.context, save_dir=save_dir, name=f"{episode_id}.npy")
            rows.append({"episode_id": episode_id, "length": traj["reward"].shape[1], **scores.add(pred)})
    summary = {"checkpoint": str(cfg.checkpoint), "data": [str(p) for p in cfg.data], "episodes": len(rows),
               "context": cfg.context, "depth_space": cfg.depth_space, **scores.summary()}
    write_json(run_dir / "summary.json", summary)
    with open(run_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("%s", {k: round(v, 4) for k, v in scores.summary().items()})


if __name__ == "__main__":
    main()
