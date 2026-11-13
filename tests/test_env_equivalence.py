"""Environment configuration and task maths against the legacy code.

Imports Isaac Gym (through aerial_gym), so it runs in the simulator environment.
"""

import isaacgym  # noqa: F401  (before torch)

import inspect
import math
import unittest

import torch

from tests import _legacy as L


def attrs(obj, depth=0):
    """Public attributes of a config class, recursing into nested classes."""
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        v = getattr(obj, k)
        nested = depth < 4 and not k.endswith("_config") or k == "motor_model_config"
        if inspect.isclass(v) and nested:
            out[k] = attrs(v, depth + 1)
        elif inspect.isclass(v):
            ours = v.__module__.startswith(("src.", "wmnav."))
            out[k] = attrs(v, depth + 1) if ours else f"<class {v.__module__}.{v.__name__}>"
        elif callable(v):
            continue
        else:
            out[k] = v
    return out


def approx_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(approx_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(approx_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=0, abs_tol=0)
    return a == b


class TestConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L.require_legacy_wms()

    def assertSameAttrs(self, old, new, skip=()):
        a, b = attrs(old), attrs(new)
        for k in skip:
            a.pop(k, None), b.pop(k, None)
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(approx_equal(a[k], b[k]), f"{old.__name__}.{k}: {a[k]!r} != {b[k]!r}")

    def test_robot(self):
        from src.configs.robots.lmf2 import LMF2Cfg as Old
        from wmnav.env.configs.robot import LMF2Cfg as New
        self.assertSameAttrs(Old, New)

    def test_sim(self):
        from src.configs.sims.procedural_world import BaseSimConfig as Old
        from wmnav.env.configs.sim import SimCfg as New
        self.assertSameAttrs(Old, New)

    def test_walls_and_panels(self):
        import src.configs.assets.dataset_dynamic as old
        import wmnav.env.configs.assets as new
        for name in ("left_wall", "right_wall", "back_wall", "front_wall", "bottom_wall", "top_wall"):
            self.assertSameAttrs(getattr(old, name), getattr(new, name))
        # panel count is per level now; everything else must match
        self.assertSameAttrs(old.pillar_params, new.pillar_params, skip=("num_assets",))

    def test_levels_match_the_as_run_scene(self):
        """Each level's scene equals the legacy StaticEnvCfg as patched by legacy levels.py."""
        import importlib
        import src.configs.envs.static_env as old_env_mod
        from src.configs.assets.dataset_dynamic import pillar_params as old_pillars
        from wmnav.env.configs.env import build_env_cfg
        for level in ("L1", "L2", "L3", "L4"):
            old_env = importlib.reload(old_env_mod).StaticEnvCfg
            old_levels = importlib.import_module("src.configs.envs.levels")
            old_levels.apply_level(level, old_env, old_pillars, n_pillars=5)
            new_env = build_env_cfg(level, 5)
            for k in ("lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
                      "distribution", "num_physics_steps_per_env_step_mean", "num_physics_steps_per_env_step_std",
                      "collision_force_threshold", "reset_on_collision"):
                self.assertEqual(getattr(old_env.env, k), getattr(new_env.env, k), (level, k))
            new_pillars = new_env.env_config.asset_type_to_dict_map["pillars"]
            self.assertEqual(new_pillars.num_assets, 5)
            used_old = [k for k, v in old_env.env_config.include_asset_type.items() if v]
            self.assertEqual(sorted(used_old), sorted(new_env.env_config.include_asset_type))
            order_old = [k for k in old_env.env_config.asset_type_to_dict_map if k in used_old]
            self.assertEqual(order_old, list(new_env.env_config.asset_type_to_dict_map))


class TestTaskMaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L.require_legacy_wms()
        import src.tasks.static_env_task as old
        cls.old = old

    def test_reward(self):
        from aerial_gym.utils.math import quat_to_rotation_matrix
        from wmnav.env.configs.task import make_task_cfg
        from wmnav.env.task import compute_reward

        g = torch.Generator().manual_seed(0)
        n = 4096
        pos = torch.rand(n, 3, generator=g) * torch.tensor([15.0, 6.0, 7.0]) - torch.tensor([0.0, 3.0, 0.0])
        target = pos + torch.randn(n, 3, generator=g) * torch.rand(n, 1, generator=g) * 3
        quat = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
        crash = torch.rand(n, generator=g) < 0.1
        left = torch.randint(0, 300, (n,), generator=g)
        params = {k: torch.tensor(v) for k, v in make_task_cfg("world_model", 1, 0).reward_parameters.items()}
        r_old, w_old, _ = self.old.compute_reward_r(pos, target, quat_to_rotation_matrix(quat), crash, left, params)
        r_new, w_new = compute_reward(pos, target, quat_to_rotation_matrix(quat), crash, left, params)
        self.assertTrue(torch.equal(r_old, r_new))
        self.assertTrue(torch.equal(w_old, w_new))
        self.assertGreater(int(w_new.sum()), 0)

    def test_action_transform(self):
        from wmnav.env.task import transform_action

        class Holder:
            device = "cpu"

        g = torch.Generator().manual_seed(1)
        a = torch.rand(1000, 4, generator=g) * 3 - 1.5
        # the legacy transform is a closure inside StaticEnv.step: re-create it from its source
        src = inspect.getsource(self.old.StaticEnv.step)
        start = src.index("        def action_transformation_function")
        end = src.index("        if getattr(self.task_config, \"preprocess_actions\"")
        body = "\n".join(line[8:] for line in src[start:end].splitlines())
        scope = {"torch": torch, "self": Holder()}
        exec(body, scope)
        old = scope["action_transformation_function"](torch.clamp(a.clone(), -1, 1))
        new = transform_action(torch.clamp(a.clone(), -1, 1))
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(bool((new[:, 0] >= 0).all()) and bool((new[:, 1] == 0).all()))


if __name__ == "__main__":
    unittest.main()
