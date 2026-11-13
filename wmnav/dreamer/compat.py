"""Reading checkpoints written before the current parameter names.

Two older layouts exist: the legacy repositories' (wms_navigation / ssl_wms, i.e.
every released model) and wmnav's own first format (``wmnav/v1``). Both use the
same tensors in the same order as the current modules, under other names;
``upgrade_*`` renames them, ``upgrade_model_config`` translates a stored v1
model config to the current schema. ``tests/test_checkpoints.py`` loads every
released checkpoint through this module with strict key matching.
"""

import re
from typing import Dict, List, Tuple

import torch

StateDict = Dict[str, torch.Tensor]
Rules = List[Tuple[str, str]]


def _dense(old: str, new: str) -> Rules:
    """Rules for an MLP (trunk of Linear/LayerNorm/activation + output layers)."""
    return [
        (rf"^{old}net\.[A-Za-z]*_layer_(\d+)\.", lambda m, p=new: f"{p}trunk.{3 * int(m.group(1))}."),
        (rf"^{old}net\.[A-Za-z]*_norm_(\d+)\.", lambda m, p=new: f"{p}trunk.{3 * int(m.group(1)) + 1}."),
        (rf"^{old}net_mean\.[A-Za-z]*_mean\.", f"{new}out."),
        (rf"^{old}net_std\.", f"{new}out_std."),
    ]


def _conv_stack(old: str, new: str) -> Rules:
    """Rules for a Sequential of convolutions with channel LayerNorms."""
    return [
        (rf"^{old}(\d+)\.norm\.", lambda m, p=new: f"{p}{m.group(1)}."),
        (rf"^{old}(\d+)\.", lambda m, p=new: f"{p}{m.group(1)}."),
    ]


WORLD_MODEL_RULES: Rules = [
    *_conv_stack(r"encoder\.net_cnn\.net\.", "encoder.image."),
    *_dense(r"encoder\.net_mlp\.", "encoder.vector."),
    (r"^dynamics\.img_in_layer\.", "rssm.prior_in."),
    (r"^dynamics\.net_cell\.layers\.GRU_linear\.", "rssm.gru.proj."),
    (r"^dynamics\.net_cell\.layers\.GRU_norm\.", "rssm.gru.norm."),
    (r"^dynamics\.img_out_layer\.", "rssm.prior_hidden."),
    (r"^dynamics\.obs_out_layer\.", "rssm.post_hidden."),
    (r"^dynamics\.imgs_stat_layer\.", "rssm.prior_logits."),
    (r"^dynamics\.obs_stat_layer\.", "rssm.post_logits."),
    (r"^dynamics\.w$", "rssm.h0"),
    (r"^heads\.decoder\.net\.linear_layer\.", "decoder.image.project."),
    *_conv_stack(r"heads\.decoder\.net\.net\.", "decoder.image.blocks."),
    *_dense(r"heads\.decoder\.mlp\.", "decoder.vector."),
    *_dense(r"heads\.reward\.", "reward_head."),
    *_dense(r"heads\.cont\.", "continue_head."),
    (r"^state_preproces\.running_mean$", "normalizer.mean"),
    (r"^state_preproces\.running_variance$", "normalizer.var"),
    (r"^state_preproces\.current_count$", "normalizer.count"),
]

ACTOR_CRITIC_RULES: Rules = [
    *_dense(r"actor\.", "actor."),
    *_dense(r"value\.", "critic."),
    *_dense(r"slow_value\.", "critic_ema."),
    (r"^ema_vals$", "return_range"),
]


def ppo_encoder_rules(prefix: str) -> Rules:
    """Rules for the PPO encoder's keys under ``prefix`` (sample-factory checkpoints)."""
    p = re.escape(prefix)
    rules = _conv_stack(rf"{p}heads\.cnn_encoder\.net\.", f"{prefix}heads.cnn_encoder.")
    for head in ("heads.mlp_state.", "heads.mlp_action.", "net_concat."):
        rules += _dense(p + re.escape(head), prefix + head)
    return rules


def rename(sd: StateDict, rules: Rules) -> StateDict:
    """Apply the first matching rule to every key; keys no rule matches are kept."""
    compiled = [(re.compile(pattern), repl) for pattern, repl in rules]
    out = {}
    for key, value in sd.items():
        for pattern, repl in compiled:
            new, hits = pattern.subn(repl, key, count=1)
            if hits:
                key = new
                break
        out[key] = value
    return out


def upgrade_world_model(sd: StateDict) -> StateDict:
    """World-model state dict with current names (unchanged if already current)."""
    return rename(sd, WORLD_MODEL_RULES)


def upgrade_actor_critic(sd: StateDict) -> StateDict:
    """Actor-critic state dict with current names (unchanged if already current)."""
    return rename(sd, ACTOR_CRITIC_RULES)


# --------------------------------------------------------------- model config

_UNSUPPORTED = {  # v1 options the current code no longer implements, and the value it assumes
    "dyn_rec_depth": 1, "initial": "learned", "imag_gradient": "dynamics", "eval_state_mean": False,
    "grad_heads": None, "dyn_mean_act": None, "dyn_std_act": None, "dyn_min_std": None,
}


def upgrade_model_config(old: dict) -> dict:
    """Translate a model config stored by ``wmnav/v1`` to the current schema."""
    for key, assumed in _UNSUPPORTED.items():
        if assumed is not None and key in old and old[key] != assumed:
            raise ValueError(f"checkpoint uses {key}={old[key]!r}; only {assumed!r} is supported")
    if old.get("dyn_discrete", 32) == 0:
        raise ValueError("checkpoint uses Gaussian latents (dyn_discrete=0), which are not supported")

    def obs(part):
        return {"image_key": part["cnn_keys"][0], "vector_key": part["mlp_keys"][0],
                "activation": part["act"], "layer_norm": part["norm"], "channels": part["cnn_depth"],
                "kernel": part["kernel_size"], "min_resolution": part["minres"],
                "mlp_layers": part["mlp_layers"], "mlp_width": part["mlp_units"]}

    actor, critic = old["actor"], old["critic"]
    return {
        "latent": {"groups": old["dyn_stoch"], "classes": old["dyn_discrete"], "recurrent": old["dyn_deter"],
                   "hidden": old["dyn_hidden"], "uniform_mix": old["unimix_ratio"]},
        "mlp_width": old["units"], "activation": old["act"], "layer_norm": old["norm"],
        "encoder": obs(old["encoder"]),
        "decoder": {**obs(old["decoder"]), "out_scale": old["decoder"]["outscale"]},
        "reward": {"layers": old["reward_head"]["layers"], "out_scale": old["reward_head"]["outscale"],
                   "loss_weight": old["reward_head"]["loss_scale"]},
        "continuation": {"layers": old["cont_head"]["layers"], "out_scale": old["cont_head"]["outscale"],
                         "loss_weight": old["cont_head"]["loss_scale"]},
        "kl": {"free_nats": old["kl_free"], "dynamics_weight": old["dyn_scale"],
               "representation_weight": old["rep_scale"]},
        "legacy_depth_target": old.get("legacy_depth_target_shift", True),
        "discount": old["discount"], "return_lambda": old["discount_lambda"],
        "imagination_horizon": old["imag_horizon"],
        "checkpoint_imagination": old.get("checkpoint_imagination", False),
        "actor": {"layers": actor["layers"], "min_std": actor["min_std"], "max_std": actor["max_std"],
                  "out_scale": actor["outscale"], "lr": actor["lr"], "eps": actor["eps"],
                  "grad_clip": actor["grad_clip"], "entropy_weight": actor["entropy"]},
        "critic": {"layers": critic["layers"], "out_scale": critic["outscale"], "lr": critic["lr"],
                   "eps": critic["eps"], "grad_clip": critic["grad_clip"],
                   "ema_every": critic["slow_target_update"], "ema_rate": critic["slow_target_fraction"]},
    }
