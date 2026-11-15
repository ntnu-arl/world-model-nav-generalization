"""Exploration policies against the legacy MultiStrategyActor (same seed, same
inputs -> same actions and the same random stream)."""

import unittest

import torch

from tests import _legacy as L
from wmnav.data.exploration import ExplorationPolicy


def legacy_high_freq_get_action(actor, depth):
    """MultiStrategyActor.get_action as run for the *_high_freq datasets
    (reconstructed from the git diff stored with the paper's runs)."""
    high_freq_idx = actor.group_size
    actions = torch.zeros((actor.num_envs, 4), device=actor.device)
    idx_uni = actor.indices[:high_freq_idx]
    num_uni = idx_uni.shape[0]
    split1, split2 = num_uni // 3, 2 * num_uni // 3
    raw_rand = torch.rand((num_uni, 4), device=actor.device)
    raw_rand[split1:split2, 1] = raw_rand[split1:split2, 1] ** (1 / 8.)
    raw_rand[split2:, 1] = raw_rand[split2:, 1] ** (1 / 16.)
    actions[idx_uni] = raw_rand * 2 - 1
    idx_ou = actor.indices[high_freq_idx:2 * high_freq_idx]
    noise = torch.randn_like(actor.ou_state[idx_ou])
    dx = (-actor.theta * actor.ou_state[idx_ou] * actor.dt
          + actor.sigma * torch.sqrt(torch.tensor(actor.dt, device=actor.device)) * noise)
    actor.ou_state[idx_ou] += dx
    actions[idx_ou] = actor.ou_state[idx_ou]
    idx_liss = actor.indices[2 * high_freq_idx:3 * high_freq_idx]
    t = (actor.steps[idx_liss] * actor.dt).unsqueeze(-1)
    actions[idx_liss] = torch.sin(actor.liss_freq[idx_liss] * t + actor.liss_phase[idx_liss])
    idx_spline = actor.indices[3 * high_freq_idx:4 * high_freq_idx]
    target = torch.rand((idx_spline.shape[0], 4), device=actor.device) * 2 - 1
    actor.spline_state[idx_spline] = 0.95 * actor.spline_state[idx_spline] + 0.05 * target
    actions[idx_spline] = actor.spline_state[idx_spline]
    actor.steps += 1
    return torch.clamp(actions, -1, 1)


class TestExploration(unittest.TestCase):
    def run_both(self, mode, device):
        L.require_legacy_wms()
        from world_models.agents import MultiStrategyActor

        n = 24
        g = torch.Generator(device=device).manual_seed(0)
        torch.manual_seed(3)
        old = MultiStrategyActor(n, device, dt=0.1)
        torch.manual_seed(3)
        new = ExplorationPolicy(n, device, mode)
        for t in range(40):
            depth = torch.rand(n, 64, 96, 1, device=device, generator=g)
            if t:
                done = torch.rand(n, device=device, generator=g) < 0.1
            else:
                done = torch.ones(n, dtype=torch.bool, device=device)
            torch.manual_seed(100 + t)
            old.reset(done)
            a_old = old.get_action(depth) if mode == "safe_explore" else legacy_high_freq_get_action(old, depth)
            torch.manual_seed(100 + t)
            a_new, _ = new({"depth_image": depth}, done)
            self.assertTrue(torch.equal(a_old, a_new["action"]), (mode, t))
            self.assertTrue(torch.equal(old.vx_coeff, new.vx_coeff))

    def test_safe_explore(self):
        self.run_both("safe_explore", "cpu")

    def test_uniform_biased(self):
        self.run_both("uniform_biased", "cpu")

    def test_random_stream_preserved(self):
        L.require_legacy_wms()
        from world_models.agents import MultiStrategyActor
        torch.manual_seed(9)
        MultiStrategyActor(16, "cpu", dt=0.1)
        after_old = torch.rand(5)
        torch.manual_seed(9)
        ExplorationPolicy(16, "cpu", "safe_explore")
        after_new = torch.rand(5)
        self.assertTrue(torch.equal(after_old, after_new))


if __name__ == "__main__":
    unittest.main()
