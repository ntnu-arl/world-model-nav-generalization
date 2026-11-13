"""The ported Dreamer must compute exactly what the legacy code computed.

Runs on CPU, where kernels are deterministic, so the comparisons are exact
(``torch.equal``), not approximate: same initialisation, same losses, same
weights after a world-model step and after an actor-critic step, same actions.
"""

import unittest

import torch

from tests import _legacy as L
from wmnav.dreamer.compat import upgrade_actor_critic, upgrade_world_model


class TestScanAndReturns(unittest.TestCase):
    def setUp(self):
        L.require_legacy_wms()

    def test_unroll_matches_legacy_scan(self):
        import world_models.tools as legacy_tools
        from wmnav.dreamer.blocks import unroll

        torch.manual_seed(0)
        xs = torch.randn(7, 3, 5)

        def fn_dict(prev, x):
            return {"a": prev["a"] * 0.5 + x, "b": (prev["a"] + x).sum(-1)}

        def fn_tuple(prev, x):
            d, _, _ = prev
            nd = {"s": d["s"] + x}
            return nd, x * 2, x.sum(-1)

        start = {"a": torch.zeros(3, 5)}
        new = unroll(fn_dict, start, [xs])
        old = legacy_tools.static_scan(fn_dict, [xs], start)
        for k in new:
            self.assertTrue(torch.equal(new[k], old[0][k]), k)

        start = ({"s": torch.zeros(3, 5)}, None, None)
        new = unroll(fn_tuple, start, [xs])
        old = legacy_tools.static_scan(fn_tuple, [xs], start)
        self.assertTrue(torch.equal(new[0]["s"], old[0]["s"]))
        self.assertTrue(torch.equal(new[1], old[1]))
        self.assertTrue(torch.equal(new[2], old[2]))

    def test_disc_dist_matches_legacy(self):
        import world_models.distributions as legacy_d
        from wmnav.dreamer.outputs import TwoHot

        torch.manual_seed(1)
        logits = torch.randn(4, 6, 255)
        x = torch.randn(4, 6, 1) * 30
        new, old = TwoHot(logits), legacy_d.DiscDist(logits, device="cpu")
        self.assertTrue(torch.equal(new.mode(), old.mode()))
        self.assertTrue(torch.equal(new.log_prob(x), old.log_prob(x)))


class TestWorldModelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L.require_legacy_wms()
        import world_models.models as legacy_models
        from omegaconf import OmegaConf
        from wmnav.dreamer.world_model import WorldModel

        cls.lr = 1e-4  # large enough for a step to move every weight visibly
        cls.legacy_cfg = L.legacy_cfg(lr=cls.lr)
        cls.cfg = L.model_cfg()

        torch.manual_seed(123)
        cls.legacy = legacy_models.WorldModel(L.obs_space(), 0, cls.legacy_cfg, L.Dummy())
        torch.manual_seed(123)
        cls.new = WorldModel(L.SHAPES, 4, cls.cfg)
        cls.new.configure_optimizer(OmegaConf.create({"opt": "adam", "lr": cls.lr, "eps": 1e-8, "clip": 1000.0}))

    def test_fresh_initialisation_is_identical(self):
        sd_old, sd_new = upgrade_world_model(self.legacy.state_dict()), self.new.state_dict()
        self.assertEqual(set(sd_old), set(sd_new))
        for k in sd_old:
            self.assertTrue(torch.equal(sd_old[k], sd_new[k]), k)

    def _step_both(self, batch):
        """One legacy and one new world-model step from the same weights; returns the
        posteriors and the gradients each optimizer saw before clipping."""
        grads = {}
        clip = torch.nn.utils.clip_grad_norm_

        def spy(tag):
            def f(params, max_norm, *a, **k):
                params = list(params)
                grads[tag] = [p.grad.detach().clone() for p in params]
                return clip(params, max_norm, *a, **k)
            return f

        self.new.load_state_dict(upgrade_world_model(self.legacy.state_dict()))
        try:
            torch.nn.utils.clip_grad_norm_ = spy("legacy")
            torch.manual_seed(7)
            post_old, _ = self.legacy._train(L.clone(batch))
            torch.nn.utils.clip_grad_norm_ = spy("new")
            torch.manual_seed(7)
            post_new, _ = self.new.train_step(L.clone(batch))
        finally:
            torch.nn.utils.clip_grad_norm_ = clip
        return post_old, post_new, grads

    def test_train_step_is_bit_identical_with_the_legacy_scan(self):
        """Everything but the sequence scan is exactly the legacy computation."""
        import world_models.tools as legacy_tools
        import wmnav.dreamer.latent as latent

        def legacy_unroll(step, carry, sequences, length=None):
            out = legacy_tools.static_scan(step, list(sequences), carry)
            return tuple(out) if len(out) > 1 else out[0]

        fast = latent.unroll
        latent.unroll = legacy_unroll
        try:
            post_old, post_new, grads = self._step_both(L.make_batch())
        finally:
            latent.unroll = fast
        for k in post_old:
            self.assertTrue(torch.equal(post_old[k], post_new[k]), k)
        for g_old, g_new in zip(grads["legacy"], grads["new"]):
            self.assertTrue(torch.equal(g_old, g_new))
        sd_old, sd_new = upgrade_world_model(self.legacy.state_dict()), self.new.state_dict()
        for k in sd_old:
            self.assertTrue(torch.equal(sd_old[k], sd_new[k]), f"weights differ after step: {k}")

    def test_train_step_matches_with_the_fast_scan(self):
        """The stacked scan changes only the order autograd sums the recurrent
        gradient contributions: same forward, gradients equal to float rounding."""
        post_old, post_new, grads = self._step_both(L.make_batch())
        for k in post_old:
            self.assertTrue(torch.equal(post_old[k], post_new[k]), k)
        for g_old, g_new in zip(grads["legacy"], grads["new"]):
            scale = g_old.abs().max().clamp(min=1e-12)
            self.assertLess(((g_old - g_new).abs().max() / scale).item(), 1e-4)

    def test_eval_losses_match_legacy_target_convention(self):
        """The decoder target is depth - 0.5, as produced by the legacy in-place shift."""
        self.new.load_state_dict(upgrade_world_model(self.legacy.state_dict()))
        batch = L.make_batch(seed=3)
        logs = {}

        class Grab:
            def log(self, d):
                logs.update(d)

        self.legacy.logger = Grab()
        torch.manual_seed(11)
        self.legacy._eval(L.clone(batch), Grab())
        torch.manual_seed(11)
        metrics = self.new.eval_step(L.clone(batch))
        self.legacy.logger = L.Dummy()
        for key in ("depth_image_not_scaled", "robot_state_not_scaled", "reward_not_scaled", "cont_not_scaled"):
            self.assertTrue(torch.equal(torch.as_tensor(logs[key]), metrics[key].cpu()), key)


class TestBehaviorAndAgentEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L.require_legacy_wms()
        import world_models.models as legacy_models
        from omegaconf import OmegaConf
        from wmnav.dreamer.agent import DreamerAgent

        cls.legacy_cfg = L.legacy_cfg(lr=1e-4)
        torch.manual_seed(5)
        cls.legacy_wm = legacy_models.WorldModel(L.obs_space(), 0, cls.legacy_cfg, L.Dummy())
        cls.legacy_beh = legacy_models.ImagBehavior(cls.legacy_cfg, cls.legacy_wm, L.Dummy())

        cls.agent = DreamerAgent(L.SHAPES, 4, L.model_cfg())
        cls.agent.configure_optimizers(OmegaConf.create({"opt": "adam", "lr": 1e-4, "eps": 1e-8, "clip": 1000.0}))
        cls.agent.world_model.load_state_dict(upgrade_world_model(cls.legacy_wm.state_dict()))
        legacy_beh_sd = {k: v for k, v in cls.legacy_beh.state_dict().items() if not k.startswith("world_model.")}
        cls.agent.actor_critic.load_state_dict(upgrade_actor_critic(legacy_beh_sd))

    def test_behavior_step_is_identical(self):
        # start states from the (identical) world models
        batch = L.make_batch(B=2, T=8, seed=9)
        with torch.no_grad():
            wm = self.agent.world_model
            data = wm.preprocess(L.clone(batch))
            embed = wm.encoder(data)
            torch.manual_seed(21)
            post, _ = wm.rssm.observe(embed, data["action"], data["is_first"])
        post = {k: v.detach() for k, v in post.items()}

        def reward_new(f, s, a):
            return wm.reward(wm.rssm.features(s)).mode()

        def reward_old(f, s, a):
            return self.legacy_wm.heads["reward"](self.legacy_wm.dynamics.get_feat(s)).mode()

        torch.manual_seed(33)
        feat_old, _, act_old, w_old = self.legacy_beh._train({k: v.clone() for k, v in post.items()}, reward_old)
        torch.manual_seed(33)
        _, (feat_new, _, act_new, w_new) = self.agent.actor_critic.train_step(
            {k: v.clone() for k, v in post.items()}, reward_new)

        self.assertTrue(torch.equal(feat_old, feat_new))
        self.assertTrue(torch.equal(act_old, act_new))
        self.assertTrue(torch.equal(w_old, w_new))
        old_sd = upgrade_actor_critic({k: v for k, v in self.legacy_beh.state_dict().items()
                                       if not k.startswith("world_model.")})
        new_sd = self.agent.actor_critic.state_dict()
        self.assertEqual(set(old_sd), set(new_sd))
        for k in old_sd:
            self.assertTrue(torch.equal(old_sd[k], new_sd[k]), f"behaviour differs after step: {k}")

    def test_policy_actions_are_identical(self):
        import world_models.agents as legacy_agents

        class Holder:  # minimal stand-in for the legacy Dreamer._policy's self
            pass

        h = Holder()
        h.wm, h.task_behavior, h.expl_behavior = self.legacy_wm, self.legacy_beh, self.legacy_beh
        h.config = self.legacy_cfg
        h.step = 0
        h.should_expl = lambda step: True

        g = torch.Generator().manual_seed(4)
        obs = {
            "depth_image": torch.rand(5, 64, 96, 1, generator=g),
            "robot_state": torch.randn(5, 22, generator=g),
            "action_agent": torch.zeros(5, 4),
            "is_first": torch.ones(5),
            "is_terminal": torch.zeros(5),
        }
        for training in (False, True):
            torch.manual_seed(8)
            out_old, _ = legacy_agents.Dreamer._policy(h, {k: v.clone() for k, v in obs.items()}, None, training)
            torch.manual_seed(8)
            out_new, _ = self.agent.policy({k: v.clone() for k, v in obs.items()}, None, training)
            self.assertTrue(torch.equal(out_old["action"], out_new["action"]), f"training={training}")



class TestSyncFreeSampling(unittest.TestCase):
    def test_one_hot_sampling_matches_torch(self):
        from wmnav.dreamer.outputs import MixedOneHot
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for dev in devices:
            for shape in ((64, 32, 32), (8, 32, 64)):
                logits = torch.randn(shape, device=dev)
                d = MixedOneHot(logits, 0.01)
                torch.manual_seed(1)
                ref = torch.distributions.OneHotCategorical.sample(d)
                after_ref = torch.rand(2, device=dev)
                torch.manual_seed(1)
                new = d._draw()
                after_new = torch.rand(2, device=dev)
                self.assertTrue(torch.equal(ref, new), (dev, shape))
                self.assertTrue(torch.equal(after_ref, after_new), (dev, shape))


if __name__ == "__main__":
    unittest.main()
