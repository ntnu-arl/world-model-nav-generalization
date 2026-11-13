"""The merged replay buffer against both legacy buffers: same contents, same byte
accounting and eviction, same short-episode filter, bit-identical batches."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from tests import _legacy as L
from wmnav.data.replay_buffer import SSL_KEYS, ReplayBuffer


def legacy_rl_buffer_cls():
    L.require_legacy_wms()
    from world_models.memory.memory import ReplayBuffer as Legacy
    return Legacy


def legacy_ssl_buffer_cls():
    path = L.LEGACY_SSL / "src" / "datasets" / "replay_buffer.py"
    if not path.exists():
        raise unittest.SkipTest("legacy ssl_wms not found")
    spec = importlib.util.spec_from_file_location("legacy_ssl_replay_buffer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ReplayBuffer


def episode_stream(n_envs=6, n_steps=90, seed=0):
    """Vectorised-env style stream: per step, one dict per env; episodes end at
    random times and the env then starts a new episode id."""
    g = torch.Generator().manual_seed(seed)
    ids = [f"ep{e}-0" for e in range(n_envs)]
    counters = [0] * n_envs
    for t in range(n_steps):
        step = {}
        for e in range(n_envs):
            done = bool(torch.rand(1, generator=g).item() < 0.06)
            step[ids[e]] = {
                "depth_image": torch.rand(64, 96, 1, generator=g),
                "robot_state": torch.randn(22, generator=g),
                "action": torch.rand(4, generator=g) * 2 - 1,
                "action_agent": torch.rand(4, generator=g) * 2 - 1,
                "reward": torch.rand((), generator=g),
                "is_first": torch.tensor(1.0 if counters[e] == 0 else 0.0),
                "is_terminal": torch.tensor(done and bool(torch.rand(1, generator=g).item() < 0.5)),
                "done": torch.tensor(1.0 if done else 0.0),
                "discount": torch.tensor(1.0),
                "logprob": torch.randn((), generator=g),
            }
            counters[e] += 1
            if done:
                counters[e] = 0
                ids[e] = f"ep{e}-{t + 1}"
        yield step, list(ids)


def legacy_state(buf):
    return {i: dict(tr) for i, tr in buf.data.items()}


class TestAgainstLegacyRLBuffer(unittest.TestCase):
    def fill(self, legacy, new, **kw):
        for step, active in episode_stream(**kw):
            legacy.add(step)
            new.add(step)
            removed_old = legacy.cleanup_short_ended_trajectories(active)
            removed_new = new.drop_short_finished(active)
            self.assertEqual(removed_old, removed_new)

    def assert_same_contents(self, legacy, new):
        self.assertEqual(list(legacy.data), list(new.data))
        self.assertEqual(legacy.current_memory_usage, new.current_memory_usage)
        for i, tr in legacy.data.items():
            view = new.episode(i)
            for k, v in tr.items():
                self.assertTrue(torch.equal(v, view[k]), (i, k))

    def test_contents_sampling_and_save(self):
        Legacy = legacy_rl_buffer_cls()
        legacy = Legacy(8, 16, 10 ** 12, seed=3)
        new = ReplayBuffer(8, 16, 10 ** 12, seed=3)
        self.fill(legacy, new)
        self.assert_same_contents(legacy, new)

        gen = legacy()
        for _ in range(4):
            b_old, b_new = next(gen), new.sample()
            self.assertEqual(list(b_old), list(b_new))
            for k in b_old:
                self.assertTrue(torch.equal(b_old[k], b_new[k]), k)

        with tempfile.TemporaryDirectory() as tmp:
            legacy.save(tmp, "old.pt")
            new.save(tmp, "new.pt")
            old = torch.load(Path(tmp) / "old.pt")
            again = torch.load(Path(tmp) / "new.pt")
            self.assertEqual(list(old), list(again))
            for i in old:
                for k in old[i]:
                    self.assertTrue(torch.equal(old[i][k], again[i][k]))

    def test_short_episode_filter(self):
        Legacy = legacy_rl_buffer_cls()
        legacy = Legacy(4, 8, 10 ** 12, min_length=5, seed=1)
        new = ReplayBuffer(4, 8, 10 ** 12, min_length=5, seed=1)
        self.fill(legacy, new, seed=4)
        self.assert_same_contents(legacy, new)

    def test_capacity_eviction(self):
        Legacy = legacy_rl_buffer_cls()
        cap = 120 * 25_000  # ~120 steps
        legacy = Legacy(4, 8, cap, seed=2)
        new = ReplayBuffer(4, 8, cap, seed=2)
        self.fill(legacy, new, seed=5)
        self.assert_same_contents(legacy, new)

    def test_full_trajectories(self):
        Legacy = legacy_rl_buffer_cls()
        legacy, new = Legacy(4, 8, 10 ** 12), ReplayBuffer(4, 8, 10 ** 12)
        self.fill(legacy, new, seed=6)
        t_old, ids_old = legacy.get_full_trajectories(min_length=10)
        t_new, ids_new = new.get_full_trajectories(min_length=10)
        self.assertEqual(ids_old, ids_new)
        for a, b in zip(t_old, t_new):
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), k)


class TestAgainstLegacySSLBuffer(unittest.TestCase):
    def test_load_and_sample_like_ssl(self):
        Legacy = legacy_ssl_buffer_cls()
        src = ReplayBuffer(4, 8, 10 ** 12)
        for step, _ in episode_stream(seed=7):
            src.add(step)
        with tempfile.TemporaryDirectory() as tmp:
            f = src.save(tmp, "data.pt")
            legacy = Legacy(batch_sz=16, batch_length=32, capacity_bytes=10 ** 12, seed=0,
                            device="cpu", expected_keys_set=list(SSL_KEYS))
            legacy.load([f])
            new = ReplayBuffer(16, 32, 10 ** 12, seed=0, keys=SSL_KEYS)
            new.load([f])
        gen = legacy()
        for _ in range(3):
            b_old, b_new = next(gen), new.sample()
            self.assertEqual(list(b_old), list(b_new))
            for k in b_old:
                self.assertTrue(torch.equal(b_old[k], b_new[k]), k)


if __name__ == "__main__":
    unittest.main()
