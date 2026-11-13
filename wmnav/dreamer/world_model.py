"""The world model: observation encoder, latent dynamics, and the decoder, reward
and continuation heads, with their losses and optimizer.

One class serves both stages - SSL pretraining (``wmnav.ssl``) and fine-tuning
with the actor-critic (``wmnav.dreamer.agent``).

Depth-target convention
-----------------------
The legacy encoder subtracted the 0.5 depth offset from its input *in place*, so
the batch's ``depth_image`` - next used as the decoder's reconstruction target -
was shifted as well. Every released decoder was therefore trained to predict
``depth - 0.5``. ``legacy_depth_target: true`` (the default) uses that target
explicitly, which keeps the losses of the released checkpoints identical to the
ones they were trained with; ``false`` trains against the true depth (only for
models trained from scratch). ``video_pred`` returns images in the target space.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from wmnav.dreamer.blocks import ClippedOptimizer, DenseStack, trainable
from wmnav.dreamer.latent import LatentDynamics, State
from wmnav.dreamer.normalizer import RunningNormalizer
from wmnav.dreamer.outputs import NUM_BINS, ContinueFlag, TwoHot
from wmnav.dreamer.vision import DEPTH_OFFSET, ObservationDecoder, ObservationEncoder

Batch = Dict[str, torch.Tensor]


class WorldModel(nn.Module):
    """Learns the latent dynamics of the task from (batch, time) sequences."""

    def __init__(self, obs_shapes: Dict[str, Tuple[int, ...]], num_actions: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.legacy_depth_target = bool(cfg.get("legacy_depth_target", True))
        self.saved_videos = 0  # numbers video_pred files saved without a name
        shapes = {name: tuple(shape) for name, shape in obs_shapes.items()}
        act, norm, width = cfg["activation"], cfg["layer_norm"], cfg["mlp_width"]
        lat = cfg["latent"]

        self.encoder = ObservationEncoder(shapes, cfg["encoder"])
        self.rssm = LatentDynamics(lat["groups"], lat["classes"], lat["recurrent"], lat["hidden"], num_actions,
                                   self.encoder.out_dim, act, norm, lat["uniform_mix"])
        feat = self.rssm.feat_dim
        self.decoder = ObservationDecoder(feat, shapes, cfg["decoder"], self.encoder.image.out_hw)
        rew, cont = cfg["reward"], cfg["continuation"]
        self.reward_head = DenseStack(feat, width, rew["layers"], act, norm, out_dim=NUM_BINS,
                                      out_scale=rew["out_scale"])
        self.continue_head = DenseStack(feat, width, cont["layers"], act, norm, out_dim=1,
                                        out_scale=cont["out_scale"])
        self.loss_weights = {"reward": rew["loss_weight"], "cont": cont["loss_weight"]}
        self.normalizer = RunningNormalizer(shapes["robot_state"])
        self.optimizer = None

    @property
    def device(self) -> torch.device:
        """Device of the model's parameters."""
        return self.rssm.h0.device

    def configure_optimizer(self, opt_cfg) -> ClippedOptimizer:
        """One optimizer over every world-model parameter."""
        self.optimizer = ClippedOptimizer(
            self.parameters(), lr=opt_cfg["lr"], eps=opt_cfg.get("eps", 1e-8), clip=opt_cfg.get("clip", 1000.0),
            kind=opt_cfg.get("opt", "adam"), amp=bool(opt_cfg.get("amp", False)), extra=opt_cfg.get("kwargs"),
        )
        return self.optimizer

    # ------------------------------------------------------------------ heads

    def reward(self, feat: torch.Tensor) -> TwoHot:
        """Predicted reward distribution."""
        return TwoHot(self.reward_head(feat))

    def continuation(self, feat: torch.Tensor) -> ContinueFlag:
        """Predicted probability that the episode continues."""
        return ContinueFlag(self.continue_head(feat))

    # ------------------------------------------------------------------ losses

    def preprocess(self, obs: Batch, train: bool = False) -> Batch:
        """Move to the model device, standardise the robot state and add ``cont``
        (1 - is_terminal). ``train=True`` also updates the robot-state statistics."""
        device = self.device
        out = {}
        for name, value in obs.items():
            out[name] = (value.to(device) if torch.is_tensor(value)
                         else torch.tensor(value, dtype=torch.float32, device=device))
        if "robot_state" in out:
            out["robot_state"] = self.normalizer(out["robot_state"], update=train)
        alive = 1.0 - out["is_terminal"].float()
        out["cont"] = alive.unsqueeze(-1)
        return out

    def _target(self, data: Batch, key: str) -> torch.Tensor:
        if self.legacy_depth_target and key == self.encoder.image_key:
            return data[key] - DEPTH_OFFSET
        return data[key]

    def _losses(self, data: Batch):
        """Total loss per (batch, time) element, posterior, unclipped KL and metrics."""
        post, prior = self.rssm.observe(self.encoder(data), action=data["action"], is_first=data["is_first"])
        kl, kl_raw, metrics = self.rssm.kl_terms(post, prior, **self.cfg["kl"])

        # Every head gets its own feature tensor: sharing one would change the order
        # in which autograd sums the heads' gradients (last-bit differences).
        outputs = dict(self.decoder(self.rssm.features(post)))
        outputs["reward"] = self.reward(self.rssm.features(post))
        outputs["cont"] = self.continuation(self.rssm.features(post))
        nll = {key: -dist.log_prob(self._target(data, key)) for key, dist in outputs.items()}
        weighted = {key: value * self.loss_weights.get(key, 1.0) for key, value in nll.items()}
        total = sum(weighted.values()) + kl
        for key in nll:
            metrics[f"{key}_not_scaled"] = nll[key].detach()
            metrics[f"{key}_scaled"] = weighted[key].detach()
        metrics["model_loss"] = total.detach()
        return total, post, kl_raw, metrics

    def train_step(self, data: Batch) -> Tuple[State, Batch]:
        """One gradient step on a (batch, time) batch; returns the detached posterior
        (start states for imagination) and detached metrics."""
        data = self.preprocess(data, train=True)
        with trainable(self):
            total, post, _, metrics = self._losses(data)
            metrics["norm"] = self.optimizer(total.mean(), self.parameters()).detach()
        return {name: t.detach() for name, t in post.items()}, metrics

    @torch.no_grad()
    def eval_step(self, data: Batch) -> Batch:
        """Loss terms on a batch without updating anything (normalizer included)."""
        return self._losses(self.preprocess(data, train=False))[3]

    # -------------------------------------------------------------- prediction

    @torch.no_grad()
    def video_pred(self, episode: Batch, context: Optional[int] = None,
                   save_dir: Optional[Path] = None, name: str = "") -> Dict[str, np.ndarray]:
        """Reconstruct the first ``context`` steps of a (1, T, ...) episode, then
        imagine the rest from its actions.

        ``context=None`` reconstructs every step. The arrays put ground truth
        (``*_true``) next to the model output (``*_model``) for the image (``i_``),
        reward (``r_``), continuation (``c_``) and robot state (``rs_``).
        """
        x = self.preprocess(episode, train=False)
        image_key, vector_key = self.encoder.image_key, self.encoder.vector_key
        steps = x["is_terminal"].shape[-1]
        ctx = steps if context is None else int(context)
        states, _ = self.rssm.observe(self.encoder(x), x["action"][:, :ctx], x["is_first"][:, :ctx])

        def decode(s):
            feat = self.rssm.features(s)
            obs = self.decoder(feat)
            return [obs[image_key].mode(), obs[vector_key].mode(),
                    self.continuation(feat).mode(), self.reward(feat).mode()]

        parts = decode(states)
        if ctx < steps:
            last = {name: seq[:, -1] for name, seq in states.items()}
            imagined = decode(self.rssm.imagine(x["action"][:, ctx:steps], last))
            parts = [torch.cat(pair, 1) for pair in zip(parts, imagined)]
        def host(t):
            return t.detach().to("cpu").numpy()

        image, robot_state, cont, reward = map(host, parts)

        out = {
            "i_true": host(self._target(x, image_key)), "i_model": image,
            "r_true": host(x["reward"]), "r_model": reward,
            "c_true": host(x["cont"]), "c_model": cont,
            "rs_true": host(x[vector_key]), "rs_model": robot_state,
            "context": ctx,
        }
        self.saved_videos += 1
        if save_dir is not None:
            np.save(Path(save_dir) / (name or f"video_{self.saved_videos}.npy"), out)
        return out
