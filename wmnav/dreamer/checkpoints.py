"""Checkpoint I/O for world models and Dreamer agents.

Written format (``wmnav/v2``)::

    {"format": "wmnav/v2", "kind": "agent" | "world_model",
     "world_model": state dict, "actor_critic": state dict or None,
     "model_config": dict, "meta": dict}

``load_checkpoint`` also reads, and converts to current parameter names
(``wmnav.dreamer.compat``):

* ``wmnav/v1`` - this package's first format (``behavior`` instead of
  ``actor_critic``, older names and model-config schema);
* legacy SSL (ssl_wms ``agent_{eval,train}_best.pth``): a flat state dict whose
  world model sits under ``wm._orig_mod.`` (``torch.compile``'s wrapper, which
  never changed the computation) or ``wm.``;
* legacy agent (wms_navigation ``best_{train,eval}_{win_rate,reward}.pt``):
  ``{"agent": state dict, "score_r", "score_w", "step"}``, the world model under
  ``wm.`` and the actor-critic under ``task_behavior.`` (other prefixes in these
  files repeat the same tensors).
"""

from pathlib import Path
from typing import Dict, Optional

import torch

from wmnav.dreamer.compat import upgrade_actor_critic, upgrade_model_config, upgrade_world_model

FORMAT = "wmnav/v2"


def _under(sd: Dict[str, torch.Tensor], *prefixes: str) -> Dict[str, torch.Tensor]:
    """The entries under the first prefix that occurs, with the prefix removed."""
    for prefix in prefixes:
        found = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if found:
            return found
    return {}


def _normalised(source, world_model, actor_critic, model_config, meta) -> dict:
    return {
        "source": source,
        "world_model": upgrade_world_model(world_model),
        "actor_critic": upgrade_actor_critic(actor_critic) if actor_critic else None,
        "model_config": model_config,
        "meta": meta,
    }


def load_checkpoint(path, map_location="cpu") -> dict:
    """Any supported checkpoint as ``{"source", "world_model", "actor_critic",
    "model_config", "meta"}`` with current parameter names."""
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: unrecognised checkpoint format")
    fmt = blob.get("format")
    if fmt == FORMAT:
        return _normalised(FORMAT, blob["world_model"], blob.get("actor_critic"), blob.get("model_config"),
                           blob.get("meta", {}))
    if fmt == "wmnav/v1":
        cfg = blob.get("model_config")
        return _normalised(fmt, blob["world_model"], blob.get("behavior"),
                           upgrade_model_config(cfg) if cfg else None, blob.get("meta", {}))
    if isinstance(blob.get("agent"), dict):
        sd = blob["agent"]
        actor_critic = {k: v for k, v in _under(sd, "task_behavior._orig_mod.", "task_behavior.").items()
                        if not k.startswith("world_model.")}
        meta = {k: blob[k] for k in ("score_r", "score_w", "step") if k in blob}
        return _normalised("legacy-agent", _under(sd, "wm._orig_mod.", "wm."), actor_critic, None, meta)
    if any(k.startswith("wm.") for k in blob):
        return _normalised("legacy-ssl", _under(blob, "wm._orig_mod.", "wm."), None, None, {})
    raise ValueError(f"{path}: unrecognised checkpoint format")


def infer_latent_dims(world_model_sd: Dict[str, torch.Tensor], groups: int = 32) -> Dict[str, int]:
    """Latent sizes implied by a world-model state dict (current names).

    The number of classes per latent variable is not stored in legacy checkpoints
    (WM2 of the paper uses 64, the others 32). The logit layers have
    ``groups * classes`` outputs, so it follows from ``groups``.
    """
    outputs, hidden = world_model_sd["rssm.prior_logits.weight"].shape
    recurrent = world_model_sd["rssm.prior_hidden.0.weight"].shape[1]
    if outputs % groups:
        raise ValueError(f"logit layer has {outputs} outputs, not a multiple of groups={groups}")
    return {"groups": groups, "classes": outputs // groups, "recurrent": recurrent, "hidden": hidden}


def _plain(cfg):
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return cfg
    return OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg


def save_checkpoint(path, world_model, actor_critic=None, model_config=None, meta: Optional[dict] = None) -> Path:
    """Save a world model (and optionally the actor-critic) with the model config
    and ``meta`` counters; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT,
        "kind": "agent" if actor_critic is not None else "world_model",
        "world_model": world_model.state_dict(),
        "actor_critic": actor_critic.state_dict() if actor_critic is not None else None,
        "model_config": _plain(model_config),
        "meta": dict(meta or {}),
    }, path)
    return path
