"""Latent dynamics: a recurrent state-space model with categorical latents.

The model state at time t is ``{"deter": h_t, "stoch": z_t, "logit": ...}``: a
GRU state h_t and ``groups`` one-hot categorical variables z_t of ``classes``
classes each. Per step,

    h_t = GRU(h_{t-1}, MLP([z_{t-1}, a_{t-1}]))
    prior      p(z_t | h_t)         - used to imagine without observations
    posterior  q(z_t | h_t, e_t)    - uses the observation embedding e_t

and model features are ``[flatten(z_t), h_t]``. An episode starts from a learned
h_0 (``tanh`` of a parameter) and the prior's most likely z_0.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import distributions as D
from torch import nn

from wmnav.dreamer.blocks import NormGRUCell, dense_block, init_hidden, init_output, unroll
from wmnav.dreamer.outputs import MixedOneHot

State = Dict[str, torch.Tensor]


def _time_major(x):
    return x.transpose(0, 1)


def _batch_major(states: State) -> State:
    return {name: seq.transpose(0, 1) for name, seq in states.items()}


class LatentDynamics(nn.Module):
    """Prior / posterior steps, sequence observation and open-loop imagination."""

    def __init__(self, groups: int, classes: int, recurrent: int, hidden: int, num_actions: int,
                 embed_dim: int, activation: str = "SiLU", layer_norm: bool = True, uniform_mix: float = 0.01):
        super().__init__()
        self.groups, self.classes, self.recurrent = groups, classes, recurrent
        self.num_actions = num_actions
        self.uniform_mix = uniform_mix
        latent = groups * classes

        def hidden_layer(in_dim):
            return init_hidden(nn.Sequential(*dense_block(in_dim, hidden, activation, layer_norm,
                                                          bias=False, eps=1e-3)))

        self.prior_in = hidden_layer(latent + num_actions)
        self.gru = init_hidden(NormGRUCell(hidden, recurrent))
        self.prior_hidden = hidden_layer(recurrent)
        self.post_hidden = hidden_layer(recurrent + embed_dim)
        self.prior_logits = init_output(nn.Linear(hidden, latent), 1.0)
        self.post_logits = init_output(nn.Linear(hidden, latent), 1.0)
        self.h0 = nn.Parameter(torch.zeros(1, recurrent))

    @property
    def feat_dim(self) -> int:
        """Size of :meth:`features`."""
        return self.groups * self.classes + self.recurrent

    # ------------------------------------------------------------------ state

    def distribution(self, logits: torch.Tensor) -> D.Independent:
        """The latent distribution with these logits (``groups`` independent one-hots)."""
        return D.Independent(MixedOneHot(logits, self.uniform_mix), 1, validate_args=False)

    def features(self, state: State) -> torch.Tensor:
        """Model features ``[flatten(stoch), deter]``."""
        stoch = state["stoch"]
        flat = stoch.reshape(stoch.shape[:-2] + (self.groups * self.classes,))
        return torch.cat([flat, state["deter"]], -1)

    def _logits(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        out = layer(x)
        return out.reshape(out.shape[:-1] + (self.groups, self.classes))

    def initial(self, batch: int) -> State:
        """Start-of-episode state for ``batch`` sequences."""
        device = self.h0.device
        deter = torch.tanh(self.h0).repeat(batch, 1)
        stoch = self.distribution(self._logits(self.prior_logits, self.prior_hidden(deter))).mode()
        return {
            "logit": torch.zeros(batch, self.groups, self.classes, device=device),
            "stoch": stoch,
            "deter": deter,
        }

    def _restart(self, state: State, action: torch.Tensor, is_first: torch.Tensor) -> Tuple[State, torch.Tensor]:
        """Replace the state and action of sequences that start an episode."""
        mask = is_first[:, None]
        action = action * (1.0 - mask)
        fresh = self.initial(len(is_first))
        blended = {}
        for key, value in state.items():
            m = mask.reshape(mask.shape + (1,) * (value.dim() - mask.dim()))
            blended[key] = value * (1.0 - m) + fresh[key] * m
        return blended, action

    # ------------------------------------------------------------------ steps

    def prior_step(self, state: State, action: torch.Tensor, sample: bool = True) -> State:
        """Advance one step without an observation."""
        flat = state["stoch"].reshape(state["stoch"].shape[:-2] + (self.groups * self.classes,))
        deter = self.gru(self.prior_in(torch.cat([flat, action], -1)), state["deter"])
        logits = self._logits(self.prior_logits, self.prior_hidden(deter))
        dist = self.distribution(logits)
        return {"stoch": dist.sample() if sample else dist.mode(), "deter": deter, "logit": logits}

    def posterior_step(self, state: Optional[State], action: Optional[torch.Tensor], embed: torch.Tensor,
                       is_first: torch.Tensor, sample: bool = True,
                       starts: Optional[Tuple[bool, bool]] = None) -> Tuple[State, State]:
        """Advance one step with the observation embedding; returns (posterior, prior).

        ``state=None`` starts every sequence. ``starts`` is an optional precomputed
        ``(every sequence starts, some sequence starts)`` for ``is_first``; without
        it the flags are read from the device (one synchronisation).
        """
        if state is None:
            every, some = True, True
        elif starts is not None:
            every, some = starts
        else:
            n = float(is_first.sum())
            every, some = n == len(is_first), n > 0
        if state is None or every:
            state = self.initial(len(is_first))
            action = torch.zeros(len(is_first), self.num_actions, device=embed.device)
        elif some:
            state, action = self._restart(state, action, is_first)

        prior = self.prior_step(state, action)
        joint = torch.cat((prior["deter"], embed), -1)
        logits = self._logits(self.post_logits, self.post_hidden(joint))
        dist = self.distribution(logits)
        post = {"stoch": dist.sample() if sample else dist.mode(), "deter": prior["deter"], "logit": logits}
        return post, prior

    # -------------------------------------------------------------- sequences

    def observe(self, embed: torch.Tensor, action: torch.Tensor, is_first: torch.Tensor,
                state: Optional[State] = None) -> Tuple[State, State]:
        """Posteriors and priors along (batch, time) sequences of embeddings, the
        previous actions and episode-start flags."""
        action, embed, is_first = _time_major(action), _time_major(embed), _time_major(is_first)
        # the per-step start flags, read to the host once for the whole sequence
        width = is_first.shape[1]
        starts = [(n == width, n > 0) for n in is_first.sum(-1).tolist()]

        def step(carry, act, emb, first, flags):
            return self.posterior_step(carry[0], act, emb, first, starts=flags)

        post, prior = unroll(step, (state, state), (action, embed, is_first, starts))
        return _batch_major(post), _batch_major(prior)

    def imagine(self, actions: torch.Tensor, state: State) -> State:
        """Open-loop prior rollout from ``state`` under (batch, time) actions."""
        return _batch_major(unroll(self.prior_step, state, (_time_major(actions),)))

    # ------------------------------------------------------------------- loss

    def kl_terms(self, post: State, prior: State, free_nats: float, dynamics_weight: float,
                 representation_weight: float):
        """KL balancing with free nats.

        The dynamics term trains the prior towards the (fixed) posterior, the
        representation term the posterior towards the (fixed) prior; each is
        clipped below at ``free_nats``. Returns ``(loss, unclipped representation
        KL, metrics)``; metrics are detached tensors.
        """
        kl = D.kl.kl_divergence
        representation = kl(self.distribution(post["logit"]), self.distribution(prior["logit"].detach()))
        dynamics = kl(self.distribution(post["logit"].detach()), self.distribution(prior["logit"]))
        raw = representation
        metrics = {"rep_loss": representation.detach(), "dyn_loss": dynamics.detach()}
        representation = torch.clip(representation, min=free_nats)
        dynamics = torch.clip(dynamics, min=free_nats)
        loss = dynamics_weight * dynamics + representation_weight * representation
        metrics.update({
            "rep_loss_clipped": representation.detach(),
            "dyn_loss_clipped": dynamics.detach(),
            "dyn_scaled": dynamics.detach() * dynamics_weight,
            "rep_scaled": representation.detach() * representation_weight,
            "kl_loss": loss.detach(),
        })
        return loss, raw, metrics
