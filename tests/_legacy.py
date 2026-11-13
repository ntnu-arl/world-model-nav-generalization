"""Helpers to import the legacy repositories for equivalence tests.

Set WMNAV_LEGACY_WMS (default /home/luca/work/wms_navigation) and
WMNAV_LEGACY_SSL (default /home/luca/work/ssl_wms); tests that need them are
skipped when they are not present.
"""

import os
import sys
import unittest
from pathlib import Path

LEGACY_WMS = Path(os.environ.get("WMNAV_LEGACY_WMS", "/home/luca/work/wms_navigation"))
LEGACY_SSL = Path(os.environ.get("WMNAV_LEGACY_SSL", "/home/luca/work/ssl_wms"))
REPO = Path(__file__).resolve().parents[1]


def require_legacy_wms():
    if not (LEGACY_WMS / "world_models" / "models.py").exists():
        raise unittest.SkipTest(f"legacy wms_navigation not found at {LEGACY_WMS}")
    if str(LEGACY_WMS) not in sys.path:
        sys.path.insert(0, str(LEGACY_WMS))


class Dummy:
    """Stands in for the legacy loggers."""

    def log(self, *a, **k):
        pass

    def add(self, *a, **k):
        pass


def model_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load(REPO / "configs" / "model" / "dreamer.yaml")


def legacy_cfg(device="cpu", lr=1e-4, dyn_discrete=32):
    """The legacy config_wms.yaml with the as-run architecture."""
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.load(LEGACY_WMS / "src" / "configs" / "cfgs" / "config_wms.yaml")
    with open_dict(cfg):
        cfg.device = device
        cfg.num_actions = 4
        cfg.compile = False
        cfg.model_lr = lr
        cfg.dyn_discrete = dyn_discrete
    return cfg


SHAPES = {
    "action_agent": (4,),
    "depth_image": (64, 96, 1),
    "is_first": (1,),
    "is_terminal": (1,),
    "robot_state": (22,),
}


def obs_space():
    import gym
    import numpy as np
    return gym.spaces.Dict({k: gym.spaces.Box(-1.0, 10.0, shape=v, dtype=np.float32) for k, v in SHAPES.items()})


def make_batch(B=3, T=12, seed=0, device="cpu"):
    """A batch shaped like ReplayBuffer.sample(), with episode starts inside it."""
    import torch
    g = torch.Generator().manual_seed(seed)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    is_first[1, 5] = 1.0
    is_terminal = torch.zeros(B, T, dtype=torch.bool)
    is_terminal[1, 4] = True
    batch = {
        "depth_image": torch.rand(B, T, 64, 96, 1, generator=g),
        "robot_state": torch.randn(B, T, 22, generator=g),
        "action": torch.rand(B, T, 4, generator=g) * 2 - 1,
        "action_agent": torch.rand(B, T, 4, generator=g) * 2 - 1,
        "reward": torch.rand(B, T, generator=g) * 4,
        "is_first": is_first,
        "is_terminal": is_terminal,
        "done": is_terminal.long(),
        "discount": (~is_terminal).float(),
        "logprob": torch.zeros(B, T),
    }
    return {k: v.to(device) for k, v in batch.items()}


def clone(batch):
    return {k: v.clone() for k, v in batch.items()}
