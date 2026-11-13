"""Metric accumulation without a GPU sync per training step.

The legacy loggers converted every metric to numpy (``.cpu()``) and called
``wandb.log`` at every gradient step - several device syncs and a W&B call per
update. ``MetricsAccumulator`` keeps running sums *on the device* and reduces them
once per flush, so logging costs one sync per ``log_every`` updates and never
changes what is computed, only how often it is reported.
"""

from typing import Dict, Union

import torch

Number = Union[int, float]


class MetricsAccumulator:
    """Running sums of metrics kept on their device until :meth:`flush`."""

    def __init__(self):
        self._acc = {}  # key -> [stats tensor (sum, sumsq, max, min), count]

    def add(self, metrics: Dict[str, Union[torch.Tensor, Number]]) -> None:
        """Add one update's metrics (tensors or numbers); no host synchronisation."""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                v = value.detach().float()
                count = v.numel()
                if count == 0:
                    continue
                stats = torch.stack([v.sum(), (v * v).sum(), v.max(), v.min()])
            else:
                v = float(value)
                count = 1
                stats = torch.tensor([v, v * v, v, v])

            if key not in self._acc:
                self._acc[key] = [stats, count]
                continue
            prev, prev_count = self._acc[key]
            if prev.device != stats.device:
                stats = stats.to(prev.device)
            merged = torch.stack([
                prev[0] + stats[0],
                prev[1] + stats[1],
                torch.maximum(prev[2], stats[2]),
                torch.minimum(prev[3], stats[3]),
            ])
            self._acc[key] = [merged, prev_count + count]

    def __len__(self):
        return len(self._acc)

    def flush(self) -> Dict[str, float]:
        """Reduce everything added since the last flush to python floats.

        A key that only ever held scalars is reported as its mean; a key that held
        arrays also gets ``/std``, ``/max`` and ``/min``.
        """
        if not self._acc:
            return {}
        keys = list(self._acc)
        stats = torch.stack([self._acc[k][0].cpu() for k in keys]).double()  # one sync per device
        out = {}
        for i, key in enumerate(keys):
            count = self._acc[key][1]
            mean = stats[i, 0].item() / count
            out[key] = mean
            if count > 1:
                var = max(stats[i, 1].item() / count - mean * mean, 0.0)
                out[f"{key}/std"] = var ** 0.5
                out[f"{key}/max"] = stats[i, 2].item()
                out[f"{key}/min"] = stats[i, 3].item()
        self._acc = {}
        return out
