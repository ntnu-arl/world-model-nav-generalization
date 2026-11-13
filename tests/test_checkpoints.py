"""Every released checkpoint must load into wmnav with strict key matching.

Uses the run directories on the external drive; skipped when they are absent.
"""

import unittest
from pathlib import Path

import torch

from tests import _legacy as L

SSL = {
    "WM1": L.LEGACY_SSL / "outputs/iros26/env2/2026-03-12/03-45-45/net/agent_eval_best.pth",
    "WM2": L.LEGACY_SSL / "outputs/iros26/env3/2026-04-07/14-33-56/net/agent_eval_best.pth",
    "WM3": L.LEGACY_SSL / "outputs/iros26/env4/2026-04-12/06-43-25/net/agent_eval_best.pth",
    "WM4": L.LEGACY_SSL / "outputs/iros26/env6/2026-03-21/05-08-41/net/agent_eval_best.pth",
}
FT = L.LEGACY_WMS / "outputs/iros26/new_dataset"
FINETUNED = {
    "WM1": FT / "fine_tuned_env2/2026-03-25/19-58-15/data/2026-03-25-19-58-15/net/best_train_win_rate.pt",
    "WM2": FT / "fine_tuned_env3/2026-04-10/22-14-58/data/2026-04-10-22-14-58/net/best_train_win_rate.pt",
    "WM3": FT / "fine_tuned_env4/2026-04-15/11-19-29/data/2026-04-15-11-19-30/net/best_train_win_rate.pt",
    "WM4": FT / "fine_tuned_env6/2026-04-17/10-39-54/data/2026-04-17-10-39-55/net/best_train_win_rate.pt",
}
EXPECTED_DISCRETE = {"WM1": 32, "WM2": 64, "WM3": 32, "WM4": 32}


def agent_for(sd):
    from omegaconf import open_dict
    from wmnav.dreamer.agent import DreamerAgent
    from wmnav.dreamer.checkpoints import infer_latent_dims

    cfg = L.model_cfg()
    with open_dict(cfg):
        cfg.latent.update(infer_latent_dims(sd))
    return DreamerAgent(L.SHAPES, 4, cfg)


class TestReleasedCheckpoints(unittest.TestCase):
    def test_ssl_checkpoints_load_strict(self):
        from wmnav.dreamer.checkpoints import infer_latent_dims, load_checkpoint
        found = 0
        for name, path in SSL.items():
            if not path.exists():
                continue
            found += 1
            ck = load_checkpoint(path)
            self.assertEqual(ck["source"], "legacy-ssl")
            self.assertEqual(infer_latent_dims(ck["world_model"])["classes"], EXPECTED_DISCRETE[name])
            agent = agent_for(ck["world_model"])
            agent.load(path, world_model_only=True, strict=True)
        if not found:
            self.skipTest("SSL checkpoints not available")

    def test_finetuned_checkpoints_load_strict_and_roundtrip(self):
        import tempfile
        from wmnav.dreamer.checkpoints import infer_latent_dims, load_checkpoint
        found = 0
        for name, path in FINETUNED.items():
            if not path.exists():
                continue
            found += 1
            ck = load_checkpoint(path)
            self.assertEqual(ck["source"], "legacy-agent")
            self.assertEqual(infer_latent_dims(ck["world_model"])["classes"], EXPECTED_DISCRETE[name])
            agent = agent_for(ck["world_model"])
            agent.load(path, strict=True)

            # the new format holds the same weights under one key each (legacy files
            # repeat the keys; torch.save shares the storage, so the size is similar)
            with tempfile.TemporaryDirectory() as tmp:
                out = agent.save(Path(tmp) / "agent.pt", meta=ck["meta"])
                again = load_checkpoint(out)
                for k, v in ck["world_model"].items():
                    self.assertTrue(torch.equal(v, again["world_model"][k]), k)
                for k, v in ck["actor_critic"].items():
                    self.assertTrue(torch.equal(v, again["actor_critic"][k]), k)
        if not found:
            self.skipTest("fine-tuned checkpoints not available")


if __name__ == "__main__":
    unittest.main()
