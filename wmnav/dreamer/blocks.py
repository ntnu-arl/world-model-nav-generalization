"""Building blocks of the Dreamer networks.

Initialisation follows DreamerV3: weights drawn with a variance set by the average
of fan-in and fan-out - a truncated normal for hidden layers, a uniform
distribution (optionally scaled down, even to zero) for output layers - and zero
biases. Layers are created and initialised in a fixed order, because every
creation and every initialisation draws from torch's random stream: the order is
what makes a seeded model reproducible.
"""

import contextlib
import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# standard deviation of a unit normal truncated to [-2, 2]; dividing by it gives the
# truncated draw the intended standard deviation
_TRUNCATED_STD = 0.87962566103423978


def _fan_average(layer: nn.Module) -> float:
    if isinstance(layer, nn.Linear):
        return (layer.in_features + layer.out_features) / 2.0
    taps = layer.kernel_size[0] * layer.kernel_size[1]
    return (taps * layer.in_channels + taps * layer.out_channels) / 2.0


def _weighted_layers(module: nn.Module):
    return [m for m in module.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))]


def init_hidden(module: nn.Module) -> nn.Module:
    """Truncated-normal init (std = sqrt(1 / fan_avg)) of every linear / conv layer
    inside ``module``, in registration order; biases set to zero."""
    for layer in _weighted_layers(module):
        std = math.sqrt(1.0 / _fan_average(layer)) / _TRUNCATED_STD
        nn.init.trunc_normal_(layer.weight, 0.0, std, -2.0 * std, 2.0 * std)  # cut at +-2 std
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return module


def init_output(module: nn.Module, scale: float) -> nn.Module:
    """Uniform init with variance ``scale / fan_avg`` (``scale=0`` gives zero weights)."""
    for layer in _weighted_layers(module):
        bound = math.sqrt(3 * (scale / _fan_average(layer)))
        nn.init.uniform_(layer.weight, -bound, bound)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return module


class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW feature map."""

    def __init__(self, channels: int):
        super().__init__(channels, eps=1e-3)

    def forward(self, x):
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)


class SameConv2d(nn.Conv2d):
    """Conv2d padded so that the output size is ceil(input / stride), the padding
    split as evenly as possible with the extra pixel at the end."""

    def forward(self, x):
        pads = []
        for size, k, s, d in zip(reversed(x.shape[-2:]), reversed(self.kernel_size),
                                 reversed(self.stride), reversed(self.dilation)):
            out = -(-size // s)              # ceil(size / s)
            span = d * (k - 1) + 1           # receptive field of one output
            total = max(s * (out - 1) + span - size, 0)
            pads += [total // 2, total - total // 2]
        if any(pads):
            x = F.pad(x, pads)
        return super().forward(x)


def dense_block(in_dim: int, out_dim: int, activation: str, layer_norm: bool = True,
                bias: bool = True, eps: float = 1e-5) -> list:
    """Linear -> (LayerNorm) -> activation, as a list of modules."""
    mods = [nn.Linear(in_dim, out_dim, bias=bias)]
    if layer_norm:
        mods.append(nn.LayerNorm(out_dim, eps=eps))
    mods.append(getattr(nn, activation)())
    return mods


class NormGRUCell(nn.Module):
    """GRU cell with one fused, layer-normalised projection and an update gate biased
    towards keeping the state (DreamerV3)."""

    def __init__(self, in_dim: int, size: int):
        super().__init__()
        self.size = size
        self.proj = nn.Linear(in_dim + size, 3 * size, bias=False)
        self.norm = nn.LayerNorm(3 * size, eps=1e-3)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = self.norm(self.proj(torch.cat([x, h], -1)))
        r, c, u = torch.split(gates, [self.size] * 3, -1)
        c = torch.tanh(torch.sigmoid(r) * c)
        u = torch.sigmoid(u - 1.0)
        return u * c + (1 - u) * h


class DenseStack(nn.Module):
    """A trunk of ``layers`` x (Linear, LayerNorm, activation) and an optional linear
    output layer (``out``) - with a second one (``out_std``) for a learned standard
    deviation.

    ``forward`` returns the trunk features, or ``(mean, std)`` / ``mean`` when the
    stack has an output layer; ``tanh_out`` squashes that mean.
    """

    def __init__(self, in_dim: int, width: int, layers: int, activation: str = "SiLU",
                 layer_norm: bool = True, out_dim: Optional[int] = None, out_scale: float = 1.0,
                 tanh_out: bool = False, learned_std: bool = False):
        super().__init__()
        mods = []
        for i in range(layers):
            mods += dense_block(in_dim if i == 0 else width, width, activation, layer_norm)
        self.trunk = init_hidden(nn.Sequential(*mods))
        self.tanh_out = tanh_out
        if out_dim is not None:
            self.out = init_output(nn.Linear(width, out_dim), out_scale)
            if learned_std:
                self.out_std = init_output(nn.Linear(width, out_dim), out_scale)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        if not hasattr(self, "out"):
            return h
        mean = self.out(h)
        if self.tanh_out:
            mean = torch.tanh(mean)
        if hasattr(self, "out_std"):
            return mean, self.out_std(h)
        return mean


def unroll(step: Callable, carry, sequences: Sequence, length: Optional[int] = None):
    """Run ``carry = step(carry, *(seq[t] for seq in sequences))`` for t = 0..T-1 and
    stack every step's result along a new leading axis.

    A result may be a tensor, a dict of tensors, or a tuple of those; the stacked
    output has the same structure. ``length`` is needed only without sequences.
    """
    steps = length if length is not None else len(sequences[0])
    results = []
    for t in range(steps):
        carry = step(carry, *(seq[t] for seq in sequences))
        results.append(carry)
    return _stack(results)


def _stack(items: list):
    first = items[0]
    if isinstance(first, dict):
        return {k: torch.stack([it[k] for it in items], 0) for k in first}
    if isinstance(first, tuple):
        return tuple(_stack(list(part)) for part in zip(*items))
    return torch.stack(items, 0)


class ClippedOptimizer:
    """A torch optimizer with gradient-norm clipping, driven by ``opt(loss, params)``.

    ``kind`` is a ``torch.optim`` class name, case-insensitive (``adam``, ``adamw``,
    ...); ``extra`` goes to its constructor. ``amp=True`` scales the loss for
    float16 autocast.
    """

    def __init__(self, params, lr: float, eps: float = 1e-4, clip: Optional[float] = None,
                 kind: str = "adam", amp: bool = False, extra: Optional[dict] = None):
        classes = {name.lower(): cls for name, cls in vars(torch.optim).items()
                   if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer)}
        kwargs = dict(extra or {}, lr=lr)
        if kind.lower() != "sgd":
            kwargs["eps"] = eps
        self.torch_optimizer = classes[kind.lower()](params, **kwargs)
        self.clip = clip
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def __call__(self, loss: torch.Tensor, params) -> torch.Tensor:
        """Back-propagate ``loss``, clip the gradients of ``params``, step; returns the
        gradient norm before clipping."""
        if loss.dim():
            raise ValueError(f"expected a scalar loss, got shape {tuple(loss.shape)}")
        params = list(params)
        opt = self.torch_optimizer
        opt.zero_grad()
        scaled = self.scaler.scale(loss)
        scaled.backward()
        self.scaler.unscale_(opt)
        norm = nn.utils.clip_grad_norm_(params, self.clip)
        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad()
        return norm

    def state_dict(self) -> dict:
        """Optimizer and loss-scaler state."""
        return {"optimizer": self.torch_optimizer.state_dict(), "scaler": self.scaler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore :meth:`state_dict`."""
        self.torch_optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])


@contextlib.contextmanager
def trainable(module: nn.Module):
    """Gradients on for ``module``'s parameters inside the block, off afterwards."""
    module.requires_grad_(True)
    try:
        yield module
    finally:
        module.requires_grad_(False)


class TrainSchedule:
    """How many gradient steps are due at each call, given the environment frames.

    The first call returns ``pretrain`` (the burst of updates before acting); after
    that, one update is due every ``frames_per_update`` frames, counted from the
    second call. ``frames_per_update`` of 0 disables training.
    """

    def __init__(self, frames_per_update: float, pretrain: int):
        self.period = frames_per_update
        self.pretrain = pretrain
        self.anchor = None
        self.started = False

    def __call__(self, frames: int) -> int:
        if not self.started:
            self.started = True
            return self.pretrain
        if not self.period:
            return 0
        if self.anchor is None:
            self.anchor = frames
            return 1
        due = int((frames - self.anchor) / self.period)
        self.anchor += self.period * due
        return due
