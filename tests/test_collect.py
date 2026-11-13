"""wmnav.collect against the legacy collection pipeline: same seed and settings
give the same dataset, episode by episode (ids differ: they embed time and a UUID).

Needs the simulator, ~1-2 min per policy: enable with WMNAV_SIM_TESTS=1.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tests import _legacy as L

REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("WMNAV_SIM_TESTS") == "1", "set WMNAV_SIM_TESTS=1 to run simulator tests")
class TestCollectMatchesLegacy(unittest.TestCase):
    def collect_both(self, policy, level="L2", envs=16, episodes=30, seed=5):
        L.require_legacy_wms()
        env = dict(os.environ, PYTHONPATH=str(REPO), WANDB_MODE="disabled")
        with tempfile.TemporaryDirectory() as tmp:
            old_f, new_f = Path(tmp) / "old.pt", Path(tmp) / "new.pt"
            subprocess.run([sys.executable, "-m", "tests.collect_probe", str(old_f), "--policy", policy,
                            "--level", level, "--envs", str(envs), "--episodes", str(episodes), "--seed", str(seed)],
                           cwd=REPO, env=env, check=True, capture_output=True)
            subprocess.run([sys.executable, "-m", "wmnav.collect", f"env.level={level}", f"env.num_envs={envs}",
                            f"episodes={episodes}", f"policy={policy}", f"seed={seed}", f"output={new_f}",
                            "logging.wandb_mode=disabled", f"hydra.run.dir={tmp}/run"],
                           cwd=REPO, env=env, check=True, capture_output=True)
            return torch.load(old_f, weights_only=False), torch.load(new_f, weights_only=False)

    def assertSameDataset(self, old, new):
        self.assertEqual(len(old), len(new))
        for i, (a, b) in enumerate(zip(old.values(), new.values())):
            self.assertEqual(set(a), set(b))
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), f"episode {i}, key {k}")

    def test_safe_explore(self):
        old, new = self.collect_both("safe_explore")
        self.assertGreater(len(old), 30)
        self.assertSameDataset(old, new)

    def test_uniform_biased(self):
        old, new = self.collect_both("uniform_biased")
        self.assertSameDataset(old, new)


if __name__ == "__main__":
    unittest.main()
