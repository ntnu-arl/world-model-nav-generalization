"""Step-by-step comparison of the legacy and the new environment under the same
seed and action sequence (separate processes: one Isaac Gym sim per process).

Needs the simulator and ~1 min per mode: enable with WMNAV_SIM_TESTS=1.
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
class TestRolloutMatchesLegacy(unittest.TestCase):
    def run_probe(self, impl, out, mode, level):
        env = dict(os.environ, PYTHONPATH=str(REPO))
        subprocess.run([sys.executable, "-m", "tests.rollout_probe", impl, str(out), "--mode", mode, "--level", level],
                       cwd=REPO, env=env, check=True, capture_output=True)
        return torch.load(out)

    def compare(self, mode, level):
        L.require_legacy_wms()
        with tempfile.TemporaryDirectory() as tmp:
            old = self.run_probe("legacy", Path(tmp) / "old.pt", mode, level)
            new = self.run_probe("new", Path(tmp) / "new.pt", mode, level)
        self.assertGreater(int(old["done"].sum()), 50)  # many resets exercised
        for k in old:
            self.assertTrue(torch.equal(old[k].float(), new[k].float()), f"{mode}/{level}: {k} differs")

    def test_world_model_mode_L3(self):
        self.compare("world_model", "L3")

    def test_ppo_mode_L1(self):
        self.compare("ppo", "L1")


if __name__ == "__main__":
    unittest.main()
