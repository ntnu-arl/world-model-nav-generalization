"""PPO encoder sized like the world model's: a 16-32-64-128 CNN on the depth image
(3072 features), 2x512 MLPs on the robot state and on the last action, and a
2x512 MLP over their concatenation.

Built from the world model's own image encoder and MLP. The released PPO
checkpoints use the legacy names inside those modules; a load hook renames them,
so sample-factory loads them unchanged (``tests/test_ppo_encoder.py``).
"""

import torch
from torch import nn
from sample_factory.model.encoder import Encoder

from wmnav.dreamer.blocks import DenseStack
from wmnav.dreamer.compat import ppo_encoder_rules, rename
from wmnav.dreamer.vision import ImageEncoder

INPUTS = (("cnn_encoder", "depth_image"), ("mlp_state", "robot_state"), ("mlp_action", "action_agent"))
DEFAULT_SPEC = {"cnn_depth": 16, "units": 512, "layers": 2, "act": "SiLU", "norm": True, "kernel_size": 4, "minres": 4}


class PPOEncoder(Encoder):
    """sample-factory encoder with the world model's encoder sizes (module docstring)."""

    spec = dict(DEFAULT_SPEC)

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        s = self.spec
        heads = {}
        latent = 0
        for head, key in INPUTS:
            shape = tuple(obs_space[key].shape)
            if head == "cnn_encoder":
                heads[head] = ImageEncoder(shape, s["cnn_depth"], s["kernel_size"], s["minres"], s["act"], s["norm"])
                # the legacy encoder sized each head with a dummy forward; the draws are
                # kept so the random stream (and so a fresh initialisation) is unchanged
                torch.rand((1,) + shape[::-1])
                latent += heads[head].out_dim
            else:
                heads[head] = DenseStack(shape[0], s["units"], s["layers"], s["act"], s["norm"])
                torch.rand((1, shape[0]))
                latent += s["units"]
        self.heads = nn.ModuleDict(heads)
        self.net_concat = DenseStack(latent, s["units"], s["layers"], s["act"], s["norm"])
        torch.rand((1, latent))
        self.out_size = s["units"]
        self._register_load_state_dict_pre_hook(self._rename_legacy_keys)

    @staticmethod
    def _rename_legacy_keys(state_dict, prefix, *_):
        renamed = rename({k: state_dict.pop(k) for k in [k for k in state_dict if k.startswith(prefix)]},
                         ppo_encoder_rules(prefix))
        state_dict.update(renamed)

    def forward(self, obs_dict):
        out = []
        for head, key in INPUTS:
            out.append(self.heads[head](obs_dict[key]))
        return self.net_concat(torch.cat(out, -1))

    def get_out_size(self) -> int:
        """Size of the encoder output (sample-factory API)."""
        return self.out_size


def make_encoder(cfg, obs_space) -> Encoder:
    """Encoder factory registered with sample-factory."""
    return PPOEncoder(cfg, obs_space)
