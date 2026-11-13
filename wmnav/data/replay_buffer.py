"""Episode replay buffer: storage, sequence sampling, and the ``.pt`` dataset format.

Merges the two legacy buffers (``world_models/memory/memory.py`` used for collection
and fine-tuning, ``ssl_wms/src/datasets/replay_buffer.py`` used for SSL), which
sampled identically and differed only in loading and in the short-episode filter.

On-disk format (unchanged, so every existing dataset loads)::

    OrderedDict{ episode_uuid: { key: tensor(T, *shape) } }

Sampling (unchanged - batches are bit-identical to the legacy code for the same
seed and contents): each of the ``batch_sz`` sequences starts in an episode drawn
with probability proportional to its length, at a uniform offset in
``[0, T-2]``; if the episode ends first, the sequence continues at the start of
another length-weighted episode, until it holds ``batch_length`` steps. The first
step of every segment gets ``is_first = 1``. Episodes still being collected are
sampled too.

What changed is the cost: an episode is no longer re-concatenated on every
appended step (quadratic in its length), and a batch is written straight into a
preallocated tensor instead of being concatenated and then stacked.
"""

import collections
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

# Every key the navigation task produces, with its per-step shape and dtype.
DEFAULT_SPECS = {
    "depth_image": ((64, 96, 1), torch.float32),
    "robot_state": ((22,), torch.float32),
    "action": ((4,), torch.float32),
    "action_agent": ((4,), torch.float32),  # raw agent action before clamping; not trained on
    "reward": ((), torch.float32),
    "is_first": ((), torch.float32),
    "is_terminal": ((), torch.bool),
    "done": ((), torch.int64),
    "discount": ((), torch.float32),
    "logprob": ((), torch.float32),          # not trained on
}

# Keys the SSL stage samples (ssl_wms config x_labels).
SSL_KEYS = ("depth_image", "robot_state", "action", "reward", "done", "is_terminal", "is_first")


def _nbytes(shape, dtype) -> int:
    return math.prod(shape) * torch.tensor([], dtype=dtype).element_size()


class _Episode:
    """Per-key tensors with spare capacity, so appending a step is O(1)."""

    __slots__ = ("tensors", "length")

    def __init__(self, tensors: Dict[str, torch.Tensor], length: int):
        self.tensors = tensors
        self.length = length

    @classmethod
    def empty(cls, specs, device, capacity):
        """An episode with preallocated, zeroed storage for ``capacity`` steps."""
        return cls({k: torch.zeros((capacity,) + tuple(shape), dtype=dtype, device=device)
                    for k, (shape, dtype) in specs.items()}, 0)

    def append(self, step: Dict[str, torch.Tensor]):
        """Add one step, doubling the storage when it is full."""
        cap = next(iter(self.tensors.values())).shape[0]
        if self.length == cap:
            for k, t in self.tensors.items():
                grown = torch.zeros((2 * cap,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
                grown[:cap] = t
                self.tensors[k] = grown
        for k, t in self.tensors.items():
            v = step.get(k)
            if v is not None:
                t[self.length] = v
            # missing keys stay zero (the legacy buffer zero-padded them)
        self.length += 1

    def shrink(self):
        """Drop spare capacity once the episode is complete."""
        for k, t in self.tensors.items():
            if t.shape[0] != self.length:
                self.tensors[k] = t[: self.length].clone()

    def view(self) -> Dict[str, torch.Tensor]:
        """The recorded steps (views, not copies) as ``{key: (length, ...)}``."""
        return {k: t[: self.length] for k, t in self.tensors.items()}


class ReplayBuffer:
    """
    Args:
        batch_sz: sequences per sampled batch.
        batch_length: steps per sequence.
        capacity_bytes: soft cap on stored data; whole episodes are evicted
            oldest-first once it is exceeded.
        min_length: if set, ``drop_short_finished`` removes finished episodes of at
            most this many steps (data collection used 50).
        seed: seed of the sampling RNG (numpy RandomState, as in the legacy code).
        device: where episodes live (``cpu``).
        keys: keys to store and sample; defaults to all of ``DEFAULT_SPECS``.
        extra_specs: ``{key: (shape, dtype)}`` of additional keys to store, e.g.
            obstacle positions for the clearance analysis.
        episode_capacity: steps preallocated per new episode (episode length + 1
            covers the navigation task without regrowth).
        pin_memory: return sampled batches in pinned memory, so the copy to the GPU
            can be asynchronous.
    """

    def __init__(self, batch_sz: int, batch_length: int, capacity_bytes: int,
                 min_length: Optional[int] = None, seed: int = 0, device: str = "cpu",
                 keys: Optional[Sequence[str]] = None, extra_specs: Optional[dict] = None,
                 episode_capacity: int = 301, pin_memory: bool = False):
        self.batch_size = batch_sz
        self.sequence_length = batch_length
        self.capacity_bytes = capacity_bytes
        self.min_length = min_length
        self.device = device
        self.episode_capacity = episode_capacity
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.np_random = np.random.RandomState(seed)

        specs = dict(DEFAULT_SPECS)
        for k, (shape, dtype) in (extra_specs or {}).items():
            specs[k] = (tuple(shape), dtype)
        if keys is not None:
            specs = {k: specs[k] for k in keys}
        self.specs = specs
        self._step_bytes = sum(_nbytes(s, d) for s, d in specs.values())

        self.data: collections.OrderedDict[str, _Episode] = collections.OrderedDict()
        self.current_memory_usage = 0

    # ------------------------------------------------------------------ storage

    def __len__(self):
        return len(self.data)

    @property
    def num_steps(self) -> int:
        """Total number of steps stored across all episodes."""
        return sum(ep.length for ep in self.data.values())

    def episode(self, episode_id) -> Dict[str, torch.Tensor]:
        """The stored steps of one episode as ``{key: (length, ...)}`` views."""
        return self.data[episode_id].view()

    def add(self, steps: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Append one step to each listed episode: ``{episode_id: {key: value}}``.

        Values are converted to the key's dtype and copied to the buffer device;
        keys the buffer does not store are ignored, missing ones are zero-filled.
        An appended step with ``done != 0`` completes its episode.
        """
        for episode_id, step in steps.items():
            ep = self.data.get(episode_id)
            if ep is None:
                ep = _Episode.empty(self.specs, self.device, self.episode_capacity)
                self.data[episode_id] = ep
            converted = {}
            for k, v in step.items():
                if k not in self.specs:
                    continue
                shape, dtype = self.specs[k]
                v = torch.as_tensor(v)
                if v.shape != torch.Size(shape) and ep.length > 0:
                    raise ValueError(f"shape mismatch for '{k}' in {episode_id}: "
                                     f"expected {shape}, got {tuple(v.shape)}")
                converted[k] = v
            ep.append(converted)
            self.current_memory_usage += self._step_bytes
            done = step.get("done")
            if done is not None and bool(torch.as_tensor(done).item()):
                ep.shrink()
        self._enforce_capacity()

    def add_batch(self, episode_ids: Sequence[str], batch: Dict[str, torch.Tensor],
                  rows: Optional[Iterable[int]] = None) -> None:
        """Append row ``i`` of every tensor in ``batch`` to episode ``episode_ids[i]``.

        The vectorised-env form of ``add``: each key is moved to the buffer device
        once for all rows (one transfer per key instead of one per env and key).
        """
        rows = range(len(episode_ids)) if rows is None else rows
        host = {}
        for k, v in batch.items():
            if k not in self.specs:
                continue
            v = torch.as_tensor(v)
            host[k] = v.detach().to(self.device, self.specs[k][1])
        self.add({episode_ids[i]: {k: v[i] for k, v in host.items()} for i in rows})

    def drop_short_finished(self, active_ids: Iterable[str]) -> int:
        """Remove episodes that are no longer active and have <= ``min_length`` steps.

        Returns the number of episodes removed (the legacy
        ``cleanup_short_ended_trajectories``).
        """
        if self.min_length is None:
            return 0
        active = set(active_ids)
        drop = [i for i, ep in self.data.items() if i not in active and ep.length <= self.min_length]
        for i in drop:
            ep = self.data.pop(i)
            self.current_memory_usage -= ep.length * self._step_bytes
        return len(drop)

    def _enforce_capacity(self):
        while self.current_memory_usage > self.capacity_bytes and self.data:
            _, ep = self.data.popitem(last=False)
            self.current_memory_usage -= ep.length * self._step_bytes

    def reset(self):
        """Drop every stored episode."""
        self.data = collections.OrderedDict()
        self.current_memory_usage = 0

    # ----------------------------------------------------------------- sampling

    def sample(self, keys: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        """One (batch_sz, batch_length, ...) batch; ``{}`` if nothing can be sampled."""
        keys = list(self.specs) if keys is None else list(keys)
        ids = [i for i, ep in self.data.items() if ep.length >= 2]
        if not ids:
            return {}
        lengths = np.array([self.data[i].length for i in ids])
        probabilities = np.array(lengths / lengths.sum())

        B, T = self.batch_size, self.sequence_length
        out = {}
        for k in keys:
            shape, dtype = self.specs[k]
            out[k] = torch.empty((B, T) + tuple(shape), dtype=dtype, pin_memory=self.pin_memory)

        for b in range(B):
            filled = 0
            while filled < T:
                ep = self.data[ids[self.np_random.choice(len(ids), p=probabilities)]]
                start = self.np_random.randint(0, ep.length - 1) if filled == 0 else 0
                end = min(start + (T - filled), ep.length)
                n = end - start
                for k in keys:
                    if k == "is_first":
                        seg = out[k][b, filled:filled + n]
                        seg.zero_()
                        seg[0] = 1.0
                    else:
                        out[k][b, filled:filled + n] = ep.tensors[k][start:end]
                filled += n
        return out

    def on_device(self, device) -> "DeviceSampler":
        """A view whose ``sample()`` returns batches already moved to ``device``."""
        return DeviceSampler(self, device)

    def __call__(self):
        """Legacy interface: an endless generator of batches."""
        while True:
            yield self.sample()

    # --------------------------------------------------------------- whole runs

    def get_full_trajectories(self, min_length: int = 1, remove_data: bool = False, all_trj: bool = False):
        """Finished episodes (last ``done`` set) longer than ``min_length``, each as a
        dict of (1, T, ...) tensors; with ``all_trj`` the length is not checked."""
        trajectories, ids = [], []
        for i in list(self.data):
            ep = self.data[i]
            view = ep.view()
            if "done" in view and ep.length and bool(view["done"][-1].item()) and (ep.length > min_length or all_trj):
                trajectories.append({k: v.clone().unsqueeze(0) for k, v in view.items()})
                ids.append(i)
        if remove_data:
            for i in ids:
                ep = self.data.pop(i)
                self.current_memory_usage -= ep.length * self._step_bytes
        return trajectories, ids

    # --------------------------------------------------------------------- files

    def state(self) -> "collections.OrderedDict[str, Dict[str, torch.Tensor]]":
        """A copy of all episodes in the on-disk format (``{episode id: {key: tensor}}``)."""
        return collections.OrderedDict((i, {k: v.clone() for k, v in ep.view().items()}) for i, ep in self.data.items())

    def save(self, path, name: Optional[str] = None) -> Path:
        """Write the buffer to ``path / name`` (default ``replay_buffer_<n>.pt``) and
        return the file path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        out = path / (name or f"replay_buffer_{len(self):05}.pt")
        torch.save(self.state(), out)
        return out

    def load(self, paths: Union[str, Path, List[Union[str, Path]]], mmap: bool = False) -> None:
        """Add the episodes of one or more ``.pt`` files (keys not stored are dropped)."""
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        for p in paths:
            blob = torch.load(p, map_location="cpu", mmap=mmap, weights_only=False)
            for i, episode in blob.items():
                tensors = {}
                length = None
                for k in self.specs:
                    if k in episode:
                        tensors[k] = episode[k].to(self.device, self.specs[k][1])
                        length = tensors[k].shape[0]
                if length is None:
                    continue
                for k, (shape, dtype) in self.specs.items():  # absent keys: zeros, as on add
                    if k not in tensors:
                        tensors[k] = torch.zeros((length,) + tuple(shape), dtype=dtype, device=self.device)
                self.data[i] = _Episode(tensors, length)
                self.current_memory_usage += length * self._step_bytes


class DeviceSampler:
    """``sample()`` from a buffer, moved to ``device`` (asynchronously from pinned memory)."""

    def __init__(self, buffer: ReplayBuffer, device):
        self.buffer, self.device = buffer, device

    def sample(self, keys: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        """One training batch, moved to the device without blocking the host."""
        return {k: v.to(self.device, non_blocking=True) for k, v in self.buffer.sample(keys).items()}
