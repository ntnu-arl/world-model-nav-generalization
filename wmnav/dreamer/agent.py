"""Dreamer agent: acts in a vectorised env and trains the world model and the
actor-critic from a replay buffer at a fixed ratio of updates to frames."""

from typing import Dict, Optional

import torch
from torch import nn

from wmnav.dreamer.actor_critic import ActorCritic
from wmnav.dreamer.blocks import TrainSchedule
from wmnav.dreamer.checkpoints import load_checkpoint, save_checkpoint
from wmnav.dreamer.world_model import WorldModel
from wmnav.utils.metrics import MetricsAccumulator


class DreamerAgent(nn.Module):
    """``agent(obs, done, state, training)`` returns ``(policy_output, state)``.

    In training mode a call first runs the updates that are due - ``pretrain`` on
    the first call, then one per ``batch_sz * batch_length / train_ratio``
    environment frames - and then acts with a *sampled* action. In evaluation mode
    it acts with the actor's mean and does not train.
    """

    def __init__(self, obs_shapes: Dict[str, tuple], num_actions: int, model_cfg, train_cfg=None, replay=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.world_model = WorldModel(obs_shapes, num_actions, model_cfg)
        self.actor_critic = ActorCritic(model_cfg, self.world_model)
        self.replay = replay
        self.step = 0           # environment frames acted on in training mode
        self.update_count = 0   # gradient steps
        self.metrics = MetricsAccumulator()
        if train_cfg is not None:
            frames = train_cfg["batch_sz"] * train_cfg["batch_length"] / train_cfg["train_ratio"]
            self.schedule = TrainSchedule(frames, int(train_cfg["pretrain"]))
        self.requires_grad_(False)

    def configure_optimizers(self, world_model_opt, actor_critic_opt: str = "adam") -> None:
        """Optimizers of the world model (config ``world_model_opt``) and of the actor
        and critic (torch optimizer name ``actor_critic_opt``)."""
        self.world_model.configure_optimizer(world_model_opt)
        self.actor_critic.configure_optimizers(actor_critic_opt)

    # ------------------------------------------------------------------ acting

    def forward(self, obs, done, state=None, training: bool = True):
        if not training:
            return self.policy(obs, state, training=False)
        for _ in range(self.schedule(self.step)):
            self.train_step(self.replay.sample())
        result = self.policy(obs, state, training=True)
        self.step += len(done)
        return result

    @torch.no_grad()
    def policy(self, obs, state=None, training: bool = False):
        """Fold ``obs`` into the latent state and choose an action - sampled in
        training, the mean otherwise. ``state`` is the ``(latent, action)`` returned
        by the previous call, or None at the start."""
        latent, action = state if state is not None else (None, None)
        wm = self.world_model
        obs = wm.preprocess(obs)
        latent, _ = wm.rssm.posterior_step(latent, action, wm.encoder(obs), obs["is_first"])
        dist = self.actor_critic.policy(wm.rssm.features(latent))
        action = dist.sample() if training else dist.mode()
        logprob = dist.log_prob(action)
        latent = {name: t.detach() for name, t in latent.items()}
        out = {"action": action.detach(), "logprob": logprob}
        return out, (latent, out["action"])

    # ---------------------------------------------------------------- training

    def train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        """One world-model update on ``batch``, then one actor-critic update in
        imagination from its posterior states."""
        wm = self.world_model
        post, wm_metrics = wm.train_step(batch)

        def reward(feat, state, action):
            return wm.reward(wm.rssm.features(state)).mode()

        ac_metrics, _ = self.actor_critic.train_step(post, reward)
        self.metrics.add({f"wm/{k}": v for k, v in wm_metrics.items()})
        self.metrics.add({f"behavior/{k}": v for k, v in ac_metrics.items()})
        self.update_count += 1

    # ------------------------------------------------------------- checkpoints

    def save(self, path, meta: Optional[dict] = None):
        """Save world model, actor-critic, model config and training counters."""
        meta = {"env_frames": self.step, "updates": self.update_count, **(meta or {})}
        return save_checkpoint(path, self.world_model, self.actor_critic, self.model_cfg, meta)

    def load(self, path, world_model_only: bool = False, strict: bool = True) -> dict:
        """Load any supported checkpoint (``wmnav.dreamer.checkpoints``).

        ``world_model_only=True`` is the fine-tuning start: the world model comes
        from an SSL checkpoint and the actor-critic keeps its fresh initialisation.
        """
        ckpt = load_checkpoint(path, map_location=self.world_model.device)
        self.world_model.load_state_dict(ckpt["world_model"], strict=strict)
        if not world_model_only:
            if ckpt["actor_critic"] is None:
                raise ValueError(f"{path} holds no actor-critic ({ckpt['source']}); "
                                 "use world_model_only=True to start fine-tuning from it")
            self.actor_critic.load_state_dict(ckpt["actor_critic"], strict=strict)
        self.requires_grad_(False)
        return ckpt
