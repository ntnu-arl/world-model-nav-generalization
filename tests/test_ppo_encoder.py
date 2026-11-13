"""PPO encoder against the legacy CustomEncoder: same initialisation (parameters in
the same order, legacy names translated), and the same outputs with the released
weights loaded under their legacy names."""

import isaacgym  # noqa: F401  (sample-factory/aerial_gym import chain)

import glob
import types
import unittest

import gymnasium as gym
import numpy as np
import torch

from tests import _legacy as L


def obs_space():
    return gym.spaces.Dict({
        "depth_image": gym.spaces.Box(-1.0, 10.0, (64, 96, 1), np.float32),
        "robot_state": gym.spaces.Box(-1.0, 1.0, (22,), np.float32),
        "action_agent": gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
    })


def legacy_encoder():
    L.require_legacy_wms()
    from omegaconf import OmegaConf
    from src.configs.tasks.static_env import task_config
    from src.utils.merge_settings import yaml_file
    cfgs_dir = str(L.LEGACY_WMS / "src/configs/cfgs")
    task_config.cfgs_custom_algo = OmegaConf.create(yaml_file(cfgs_dir, "config_ppo.yaml"))
    from ppo.networks.network import CustomEncoder
    return CustomEncoder


class TestPPOEncoder(unittest.TestCase):
    def test_fresh_init_matches(self):
        Old = legacy_encoder()
        from wmnav.ppo.encoder import PPOEncoder
        cfg = types.SimpleNamespace()
        torch.manual_seed(4)
        old = Old(cfg, obs_space())
        after_old = torch.rand(3)
        torch.manual_seed(4)
        new = PPOEncoder(cfg, obs_space())
        after_new = torch.rand(3)
        from wmnav.dreamer.compat import ppo_encoder_rules, rename
        so, sn = rename(old.state_dict(), ppo_encoder_rules("")), new.state_dict()
        self.assertEqual(list(so), list(sn))
        for k in so:
            self.assertTrue(torch.equal(so[k], sn[k]), k)
        self.assertTrue(torch.equal(after_old, after_new))  # same random stream afterwards
        self.assertEqual(old.get_out_size(), new.get_out_size())

    def test_released_weights_same_output(self):
        ckpt_dir = L.LEGACY_WMS / "outputs/iros26/ppo_baseline/train_dir/ppo-L1/checkpoint_p0"
        files = sorted(glob.glob(str(ckpt_dir / "checkpoint_*.pth")))
        if not files:
            self.skipTest("released PPO checkpoints not available")
        Old = legacy_encoder()
        from wmnav.ppo.encoder import PPOEncoder
        sd = torch.load(files[-1], map_location="cpu", weights_only=False)["model"]
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        old, new = Old(None, obs_space()), PPOEncoder(None, obs_space())
        old.load_state_dict(enc)
        new.load_state_dict(enc)
        g = torch.Generator().manual_seed(0)
        obs = {"depth_image": torch.rand(7, 64, 96, 1, generator=g), "robot_state": torch.randn(7, 22, generator=g),
               "action_agent": torch.rand(7, 4, generator=g) * 2 - 1}
        with torch.no_grad():
            self.assertTrue(torch.equal(old({k: v.clone() for k, v in obs.items()}), new(obs)))


if __name__ == "__main__":
    unittest.main()
