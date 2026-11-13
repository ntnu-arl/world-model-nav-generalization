"""Actor-critic trained on trajectories imagined by the world model (DreamerV3).

From each posterior state of a training batch the actor is rolled out for
``horizon`` steps in the latent dynamics. Rewards and continuation probabilities
come from the world-model heads; TD(lambda) returns bootstrap from the critic.
The actor maximises the returns - normalised by the spread between their 5th and
95th percentiles (an EMA), backpropagated through the dynamics - plus an entropy
bonus. The critic (a two-hot regression on the returns) is also regressed
towards a slowly updated copy of itself.
"""

import copy
from typing import Callable, Dict, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from wmnav.dreamer.blocks import ClippedOptimizer, DenseStack, trainable, unroll
from wmnav.dreamer.outputs import NUM_BINS, TwoHot, gaussian_policy

# share of the new return percentiles in their running estimate, per update
PERCENTILE_RATE = 1e-2
PERCENTILES = (0.05, 0.95)


def td_lambda(reward, value, continuation, bootstrap, lam):
    """TD(lambda) returns, computed backwards along the leading (time) axis.

    ``R_t = r_t + c_t ((1 - lam) v_{t+1} + lam R_{t+1})`` with ``R_T = bootstrap``.
    Returns (time, batch, 1).
    """
    next_value = torch.cat((value[1:], bootstrap.unsqueeze(0)))
    one_step = reward + continuation * next_value * (1 - lam)
    ret = bootstrap
    backwards = []
    for t in range(len(one_step) - 1, -1, -1):
        ret = one_step[t] + continuation[t] * lam * ret
        backwards.append(ret)
    # (batch, time) layout built in reverse order, then put back in time order
    out = torch.cat(backwards, dim=-1)
    out = torch.flip(out.reshape(out.shape[0], out.shape[1], 1), [1])
    return torch.stack(torch.unbind(out, 0), 1)


class ActorCritic(nn.Module):
    """Actor, critic and the critic's slow copy; the world model is held by
    reference (it is owned, and saved, by the agent)."""

    def __init__(self, cfg, world_model):
        super().__init__()
        self.cfg = cfg
        object.__setattr__(self, "world_model", world_model)
        feat = world_model.rssm.feat_dim
        act, norm, width = cfg["activation"], cfg["layer_norm"], cfg["mlp_width"]
        a, c = cfg["actor"], cfg["critic"]
        self.actor = DenseStack(feat, width, a["layers"], act, norm, out_dim=world_model.num_actions,
                                out_scale=a["out_scale"], tanh_out=True, learned_std=True)
        self.critic = DenseStack(feat, width, c["layers"], act, norm, out_dim=NUM_BINS, out_scale=c["out_scale"])
        self.critic_ema = copy.deepcopy(self.critic)
        self.register_buffer("return_range", torch.zeros(2))  # EMA of the 5th / 95th return percentiles
        self.updates = 0
        self.checkpoint_imagination = bool(cfg.get("checkpoint_imagination", False))
        self.actor_optimizer = self.critic_optimizer = None
        self._percentiles = {}

    def policy(self, feat: torch.Tensor):
        """Action distribution for model features."""
        a = self.cfg["actor"]
        mean, raw_std = self.actor(feat)
        return gaussian_policy(mean, raw_std, a["min_std"], a["max_std"])

    def value(self, feat: torch.Tensor, ema: bool = False) -> TwoHot:
        """Critic (or its slow copy) on model features."""
        return TwoHot((self.critic_ema if ema else self.critic)(feat))

    def configure_optimizers(self, opt: str = "adam", amp: bool = False) -> None:
        """Separate optimizers for the actor and the critic."""
        a, c = self.cfg["actor"], self.cfg["critic"]
        self.actor_optimizer = ClippedOptimizer(self.actor.parameters(), a["lr"], a["eps"], a["grad_clip"],
                                                kind=opt, amp=amp)
        self.critic_optimizer = ClippedOptimizer(self.critic.parameters(), c["lr"], c["eps"], c["grad_clip"],
                                                 kind=opt, amp=amp)

    # ------------------------------------------------------------------ update

    def train_step(self, start: Dict[str, torch.Tensor], reward_fn: Callable) -> Tuple[dict, tuple]:
        """One actor and one critic update from imagined rollouts of ``start``.

        ``reward_fn(feat, state, action)`` gives the reward of each imagined state.
        Returns ``(metrics, (feat, state, action, weight))``: detached metrics and
        the imagined trajectory.
        """
        self._follow_critic()
        metrics = {}
        with trainable(self.actor):
            feat, state, action = self._imagine(start)
            reward = reward_fn(feat, state, action)
            actor_entropy = self.policy(feat).entropy()
            state_entropy = self.world_model.rssm.distribution(state["logit"]).entropy()
            metrics.update(actor_ent=actor_entropy.detach(), state_ent=state_entropy.detach(),
                           reward=reward.detach())
            returns, weight, baseline = self._returns(feat, state, reward)
            actor_loss = self._actor_loss(returns, weight, baseline)
            actor_loss = actor_loss - self.cfg["actor"]["entropy_weight"] * actor_entropy[:-1, ..., None]
            metrics["actor_loss"] = actor_loss.detach()
            actor_loss = actor_loss.mean()

        with trainable(self.critic):
            inputs = feat[:-1].detach()
            value = self.value(inputs)
            critic_loss = -value.log_prob(returns.detach())
            critic_loss = critic_loss - value.log_prob(self.value(inputs, ema=True).mode().detach())
            critic_loss = torch.mean(weight[:-1] * critic_loss[:, :, None])
            metrics["value_loss"] = critic_loss.detach()

        with trainable(self):
            metrics["actor_norm"] = self.actor_optimizer(actor_loss, self.actor.parameters()).detach()
            metrics["value_norm"] = self.critic_optimizer(critic_loss, self.critic.parameters()).detach()
        return metrics, (feat, state, action, weight)

    def _follow_critic(self) -> None:
        """Move the slow critic towards the critic every ``ema_every`` updates."""
        c = self.cfg["critic"]
        if self.updates % c["ema_every"] == 0:
            rate = c["ema_rate"]
            for fast, slow in zip(self.critic.parameters(), self.critic_ema.parameters()):
                slow.data = rate * fast.data + (1 - rate) * slow.data
        self.updates += 1

    def _imagine(self, start):
        """Roll the actor out in the latent dynamics from every start state;
        returns time-major features, states (each step's input state) and actions."""
        rssm = self.world_model.rssm
        flat = {}
        for key, tensor in start.items():  # (batch, time, ...) -> (batch * time, ...)
            flat[key] = tensor.reshape([-1] + list(tensor.shape[2:]))

        def one_step(state):
            feat = rssm.features(state)
            action = self.policy(feat.detach()).sample()
            return rssm.prior_step(state, action), feat, action

        def step(carry):
            if self.checkpoint_imagination and torch.is_grad_enabled():
                # store only each step's inputs and outputs and recompute the rest in
                # the backward pass, with the same random state (same samples)
                return checkpoint(one_step, carry[0], use_reentrant=False, preserve_rng_state=True)
            return one_step(carry[0])

        succ, feat, action = unroll(step, (flat, None, None), (), length=self.cfg["imagination_horizon"])
        # the state each action was taken in: the start, then all but the last successor
        state = {key: torch.cat((flat[key].unsqueeze(0), nxt[:-1])) for key, nxt in succ.items()}
        return feat, state, action

    def _returns(self, feat, state, reward):
        """TD(lambda) returns, the weight of each step (the probability that the
        imagined episode is still running) and the critic's values as baseline."""
        discount = self.cfg["discount"] * self.world_model.continuation(self.world_model.rssm.features(state)).mean
        value = self.value(feat).mode()
        returns = td_lambda(reward=reward[1:], value=value[:-1], continuation=discount[1:],
                            lam=self.cfg["return_lambda"], bootstrap=value[-1])
        first = torch.ones_like(discount[:1])  # the start state is always alive
        alive = torch.cat((first, discount[:-1]))
        weight = alive.cumprod(0).detach()
        return returns, weight, value[:-1]

    def _actor_loss(self, returns, weight, baseline):
        """Minus the weighted advantage of the normalised returns over the baseline."""
        low, high = self._update_return_range(returns)
        scale = torch.clip(high - low, min=1.0).detach()
        low = low.detach()
        advantage = (returns - low) / scale - (baseline - low) / scale
        return -weight[:-1] * advantage

    def _update_return_range(self, returns):
        flat = torch.flatten(returns.detach())
        if flat.device not in self._percentiles:
            self._percentiles[flat.device] = torch.tensor(PERCENTILES, device=flat.device)
        current = torch.quantile(input=flat, q=self._percentiles[flat.device])
        self.return_range[:] = PERCENTILE_RATE * current + (1 - PERCENTILE_RATE) * self.return_range
        return self.return_range[0], self.return_range[1]
