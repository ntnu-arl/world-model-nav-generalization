"""Observation encoder and decoder: a strided CNN for the depth image and an MLP for
the robot-state vector, and their mirror images for reconstruction."""

import math
from typing import Dict, Tuple

import torch
from torch import nn

from wmnav.dreamer.blocks import ChannelLayerNorm, DenseStack, SameConv2d, init_hidden, init_output
from wmnav.dreamer.outputs import SquaredError, SymlogSquaredError

# Depth images are in [0, 1] (-1 where nothing is in range). The encoder centres
# them by this offset; the decoder adds it back.
DEPTH_OFFSET = 0.5


def _halvings(height: int, min_resolution: int) -> int:
    return int(math.log2(height) - math.log2(min_resolution))


class ImageEncoder(nn.Sequential):
    """Stride-2 convolutions, doubling the channels, until the height reaches
    ``min_resolution``; input (..., H, W, C), output (..., out_dim)."""

    def __init__(self, image_shape: Tuple[int, int, int], channels: int, kernel: int, min_resolution: int,
                 activation: str, layer_norm: bool):
        h, w, c_in = image_shape
        mods = []
        for i in range(_halvings(h, min_resolution)):
            c_out = channels * 2 ** i
            mods.append(SameConv2d(c_in, c_out, kernel, stride=2, bias=False))
            if layer_norm:
                mods.append(ChannelLayerNorm(c_out))
            mods.append(getattr(nn, activation)())
            c_in, h, w = c_out, h // 2, w // 2
        super().__init__(*mods)
        init_hidden(self)
        self.out_hw = (h, w)
        self.out_dim = c_in * h * w

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        centred = image - DEPTH_OFFSET
        lead = centred.shape[:-3]
        x = centred.reshape((-1,) + tuple(centred.shape[-3:])).movedim(-1, 1)
        return super().forward(x).flatten(1).reshape(*lead, -1)


class ImageDecoder(nn.Module):
    """Linear projection to a (start_hw, widest channels) map, then stride-2
    transposed convolutions back to the image; output (..., H, W, C)."""

    def __init__(self, feat_dim: int, image_shape: Tuple[int, int, int], channels: int, kernel: int,
                 min_resolution: int, start_hw: Tuple[int, int], activation: str, layer_norm: bool,
                 out_scale: float):
        super().__init__()
        h, w, c = image_shape
        stages = _halvings(h, min_resolution)
        self.image_shape = (c, h, w)
        self.start_hw = tuple(start_hw)
        widest = channels * 2 ** (stages - 1)
        self.project = init_output(nn.Linear(feat_dim, self.start_hw[0] * self.start_hw[1] * widest), out_scale)

        # padding that exactly doubles the size at stride 2
        excess = (kernel - 1) - 2 + 1
        pad = -(-excess // 2)
        out_pad = 2 * pad - excess
        mods = []
        for i in range(stages):
            last = i == stages - 1
            c_in = widest // 2 ** i
            c_out = c if last else c_in // 2
            mods.append(nn.ConvTranspose2d(c_in, c_out, kernel, 2, padding=(pad, pad),
                                           output_padding=(out_pad, out_pad), bias=last))
            if not last:
                if layer_norm:
                    mods.append(ChannelLayerNorm(c_out))
                mods.append(getattr(nn, activation)())
        for m in mods[:-1]:
            init_hidden(m)
        init_output(mods[-1], out_scale)
        self.blocks = nn.Sequential(*mods)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h0, w0 = self.start_hw
        x = self.project(feat)
        grid = x.reshape(-1, h0, w0, x.shape[-1] // (h0 * w0)).movedim(-1, 1)
        image = self.blocks(grid).reshape(feat.shape[:-1] + self.image_shape).movedim(2, -1)
        return image + DEPTH_OFFSET


class ObservationEncoder(nn.Module):
    """Concatenated embeddings of the image (CNN) and of the state vector (MLP)."""

    def __init__(self, obs_shapes: Dict[str, tuple], cfg):
        super().__init__()
        self.image_key, self.vector_key = cfg["image_key"], cfg["vector_key"]
        act, norm = cfg["activation"], cfg["layer_norm"]
        shapes = obs_shapes
        self.image = ImageEncoder(shapes[self.image_key], cfg["channels"], cfg["kernel"], cfg["min_resolution"],
                                  act, norm)
        self.vector = DenseStack(shapes[self.vector_key][0], cfg["mlp_width"], cfg["mlp_layers"], act, norm)
        self.out_dim = self.image.out_dim + cfg["mlp_width"]

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.image(obs[self.image_key]), self.vector(obs[self.vector_key])], -1)


class ObservationDecoder(nn.Module):
    """Reconstructs the image and the state vector from model features; returns
    ``{key: output distribution}``."""

    def __init__(self, feat_dim: int, shapes: Dict[str, tuple], cfg, start_hw: Tuple[int, int]):
        super().__init__()
        self.image_key, self.vector_key = cfg["image_key"], cfg["vector_key"]
        act, norm = cfg["activation"], cfg["layer_norm"]
        self.image = ImageDecoder(feat_dim, shapes[self.image_key], cfg["channels"], cfg["kernel"],
                                  cfg["min_resolution"], start_hw, act, norm, cfg["out_scale"])
        self.vector = DenseStack(feat_dim, cfg["mlp_width"], cfg["mlp_layers"], act, norm,
                                 out_dim=shapes[self.vector_key][0], out_scale=cfg["out_scale"])

    def forward(self, feat: torch.Tensor) -> dict:
        return {self.image_key: SquaredError(self.image(feat)),
                self.vector_key: SymlogSquaredError(self.vector(feat))}
