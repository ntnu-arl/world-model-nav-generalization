"""Gradient checkpointing of the imagination recomputes the same forward pass
(same random samples), so the actor-critic update is unchanged."""

import copy
import unittest

import torch
from omegaconf import OmegaConf

from tests import _legacy as L
from wmnav.dreamer.agent import DreamerAgent


class TestImaginationCheckpointing(unittest.TestCase):
    def test_same_update(self):
        torch.manual_seed(0)
        agent = DreamerAgent(L.SHAPES, 4, L.model_cfg())
        agent.configure_optimizers(OmegaConf.create({"opt": "adam", "lr": 1e-4, "eps": 1e-8, "clip": 1000.0}))
        batch = L.make_batch(B=2, T=8, seed=1)
        with torch.no_grad():
            wm = agent.world_model
            data = wm.preprocess(L.clone(batch))
            torch.manual_seed(2)
            post, _ = wm.rssm.observe(wm.encoder(data), data["action"], data["is_first"])
        post = {k: v.detach() for k, v in post.items()}
        ac = agent.actor_critic
        start = copy.deepcopy(ac.state_dict())

        def rew(f, s, a):
            return wm.reward(wm.rssm.features(s)).mode()


        results = []
        for ck in (False, True):
            ac.load_state_dict(start)
            ac.configure_optimizers()
            ac.updates = 0
            ac.checkpoint_imagination = ck
            torch.manual_seed(3)
            metrics, (feat, _, action, _) = ac.train_step({k: v.clone() for k, v in post.items()}, rew)
            results.append((feat.detach(), action.detach(), copy.deepcopy(ac.state_dict()), metrics))

        (f0, a0, s0, m0), (f1, a1, s1, m1) = results
        self.assertTrue(torch.equal(f0, f1))
        self.assertTrue(torch.equal(a0, a1))
        self.assertTrue(torch.equal(m0["actor_loss"], m1["actor_loss"]))
        for k in s0:
            self.assertTrue(torch.allclose(s0[k].float(), s1[k].float(), rtol=1e-5, atol=1e-7), k)


if __name__ == "__main__":
    unittest.main()
