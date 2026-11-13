"""Running standardisation of the robot-state vector."""

import torch
from torch import nn


class RunningNormalizer(nn.Module):
    """Standardises inputs with running estimates of their mean and variance and
    clips the result to [-clip, clip].

    The estimates are float64 buffers, merged with each new batch by the pairwise
    update of Chan, Golub & LeVeque (1979). Being buffers, they are saved with the
    model: fine-tuned models inherit them from SSL pretraining and keep updating
    them.
    """

    def __init__(self, dim, eps: float = 1e-8, clip: float = 5.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        f64 = {"dtype": torch.float64}
        self.register_buffer("mean", torch.zeros(dim, **f64))
        self.register_buffer("var", torch.ones(dim, **f64))
        self.register_buffer("count", torch.tensor(1.0, **f64))  # starts at 1, like a prior sample

    def _merge(self, batch_mean, batch_var, n) -> None:
        delta = batch_mean - self.mean
        merged = self.count + n
        sum_sq = self.var * self.count + batch_var * n + delta ** 2 * self.count * n / merged
        self.mean = self.mean + delta * n / merged
        self.var = sum_sq / merged
        self.count = merged

    @torch.no_grad()
    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        """Standardised ``x`` (..., size); ``update=True`` first folds the batch
        (all leading axes) into the statistics."""
        if update:
            axes = tuple(range(x.dim() - 1))
            n = 1
            for a in axes:
                n *= x.shape[a]
            self._merge(x.mean(dim=axes), x.var(dim=axes), n)
        std = torch.sqrt(self.var.float()) + self.eps
        return torch.clamp((x - self.mean.float()) / std, min=-self.clip, max=self.clip)
