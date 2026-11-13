"""Boilerplate shared by the entry points."""

import json
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from wmnav.utils.run import save_git_state
from wmnav.utils.seed import set_seed


def run_dir_from_hydra() -> Path:
    """Output directory of the current Hydra run."""
    from hydra.core.hydra_config import HydraConfig
    return Path(HydraConfig.get().runtime.output_dir)


def start_run(cfg: DictConfig) -> Tuple[Path, int]:
    """Run directory (Hydra's), git provenance, resolved config, seed, GPU flags."""
    run_dir = run_dir_from_hydra()
    run_dir.mkdir(parents=True, exist_ok=True)
    save_git_state(run_dir)
    seed = set_seed(cfg.get("seed"), bool(cfg.get("deterministic", False)))
    with open_dict(cfg):
        cfg.seed = seed
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    apply_perf(cfg.get("perf"))
    return run_dir, seed


def apply_perf(perf) -> None:
    """Apply the ``perf`` config group (TF32, cuDNN benchmark) to torch."""
    if not perf:
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(perf.get("tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(perf.get("tf32", False)) or torch.backends.cudnn.allow_tf32
    if perf.get("cudnn_benchmark", False):
        cudnn = torch.backends.cudnn
        cudnn.benchmark, cudnn.deterministic = True, False


def observation_shapes(env) -> dict:
    """``{key: shape}`` of an env's observation space."""
    return {k: tuple(space.shape) for k, space in env.observation_space.spaces.items()}


def model_cfg_for(model_cfg: DictConfig, checkpoint) -> DictConfig:
    """The model config, with latent sizes taken from ``checkpoint`` (they are not
    stored in legacy checkpoints and loading with the wrong ones fails)."""
    from wmnav.dreamer.checkpoints import infer_latent_dims, load_checkpoint

    ckpt = load_checkpoint(checkpoint)
    cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
    if ckpt.get("model_config"):
        stored = OmegaConf.create(ckpt["model_config"])
        cfg = OmegaConf.merge(cfg, stored)
    with open_dict(cfg):
        cfg.latent.update(infer_latent_dims(ckpt["world_model"], groups=cfg.latent.groups))
    return cfg


def write_json(path: Path, data: dict) -> None:
    """Write ``data`` as indented, key-sorted JSON."""
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))
