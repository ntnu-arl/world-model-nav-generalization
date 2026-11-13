"""Output distributions of the world-model and actor-critic heads.

Following DreamerV3 (Hafner et al., 2023):

* rewards and values are a categorical over 255 bins spaced uniformly in symlog
  space, trained against a "two-hot" target that splits each scalar between its
  two neighbouring bins (``TwoHot``);
* the depth image is regressed with a summed squared error (``SquaredError``),
  the robot state with a squared error in symlog space (``SymlogSquaredError``);
* the continuation flag is a Bernoulli with a straight-through mode
  (``ContinueFlag``), the discrete latents a one-hot categorical mixed with 1 %
  uniform probability (``MixedOneHot``), the actor a Gaussian.

No torch distribution here validates its arguments: validation reads tensors back
to the host, which on a GPU stalls the update for every head call, and it never
changes a value.
"""

import contextlib
import functools

import torch
from torch import distributions as D
from torch.nn.functional import one_hot, softplus

NUM_BINS = 255
BIN_LIMIT = 20.0  # bins span [-20, 20] in symlog space


def symlog(x):
    """sign(x) * log(1 + |x|): logarithmic for large magnitudes, linear near zero."""
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x):
    """Inverse of :func:`symlog`."""
    return x.sign() * (x.abs().exp() - 1.0)


@functools.lru_cache(maxsize=None)
def _bin_centres(device: torch.device) -> torch.Tensor:
    return torch.linspace(-BIN_LIMIT, BIN_LIMIT, NUM_BINS, device=device)


def _sum_event(x: torch.Tensor) -> torch.Tensor:
    """Sum over every axis after (batch, time) in one reduction."""
    return x.sum(list(range(2, x.dim())))


@contextlib.contextmanager
def _unvalidated():
    """OneHotCategorical builds an inner Categorical without passing ``validate_args``
    on, so the class default is switched off while it is constructed."""
    saved = D.Distribution._validate_args
    D.Distribution._validate_args = False
    try:
        yield
    finally:
        D.Distribution._validate_args = saved


class TwoHot:
    """Categorical over symlog-spaced bins, read out as its expected value."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.bins = _bin_centres(logits.device)

    def mode(self) -> torch.Tensor:
        """Expected value (in the original space), with a trailing singleton axis."""
        expected = (torch.softmax(self.logits, -1) * self.bins).sum(-1, keepdim=True)
        return symexp(expected)

    mean = mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Cross-entropy-style score of ``value`` against its two-hot encoding.

        Values outside the bin range are assigned entirely to the outermost bin.
        """
        y = symlog(value)
        last = NUM_BINS - 1
        lo = ((self.bins <= y[..., None]).sum(-1) - 1).clamp(0, last)
        hi = (NUM_BINS - (self.bins > y[..., None]).sum(-1)).clamp(0, last)
        same = lo == hi
        lo_gap = torch.where(same, 1.0, (self.bins[lo] - y).abs())
        hi_gap = torch.where(same, 1.0, (self.bins[hi] - y).abs())
        total = lo_gap + hi_gap
        target = (one_hot(lo, NUM_BINS) * (hi_gap / total)[..., None]
                  + one_hot(hi, NUM_BINS) * (lo_gap / total)[..., None])
        logits = self.logits
        normalised = logits - logits.logsumexp(-1, keepdim=True)
        return (target.squeeze(-2) * normalised).sum(-1)


class SquaredError:
    """Deterministic prediction scored by minus the summed squared error."""

    def __init__(self, prediction: torch.Tensor):
        self.prediction = prediction

    def mode(self) -> torch.Tensor:
        """The prediction."""
        return self.prediction

    mean = mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Minus the squared error, summed over every axis after (batch, time)."""
        if self.prediction.shape != value.shape:
            raise ValueError(f"prediction {tuple(self.prediction.shape)} vs target {tuple(value.shape)}")
        return -_sum_event((self.prediction - value) ** 2)


class SymlogSquaredError:
    """Prediction made in symlog space, scored by the squared error there."""

    TOLERANCE = 1e-8  # squared errors below this count as zero

    def __init__(self, prediction: torch.Tensor):
        self.prediction = prediction

    def mode(self) -> torch.Tensor:
        """The prediction mapped back to the original space."""
        return symexp(self.prediction)

    mean = mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Minus the squared symlog error, summed over every axis after (batch, time)."""
        if self.prediction.shape != value.shape:
            raise ValueError(f"prediction {tuple(self.prediction.shape)} vs target {tuple(value.shape)}")
        err = (self.prediction - symlog(value)) ** 2.0
        return -_sum_event(torch.where(err < self.TOLERANCE, 0, err))


class ContinueFlag:
    """Bernoulli over "the episode continues", with a straight-through mode."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    @functools.cached_property
    def mean(self) -> torch.Tensor:
        """Probability of continuing."""
        return torch.sigmoid(self.logits)

    def mode(self) -> torch.Tensor:
        """The rounded probability, with the probability's gradient."""
        p = self.mean
        return p.round().detach() + p - p.detach()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of the 0/1 targets ``value``, summed over the last axis."""
        log_stop = -softplus(self.logits)
        log_continue = -softplus(-self.logits)
        return (log_stop * (1 - value) + log_continue * value).sum(-1)


class GaussianPolicy:
    """Diagonal Gaussian over the action vector (reparameterised samples)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.dist = D.Independent(D.Normal(mean, std, validate_args=False), 1, validate_args=False)

    def mode(self) -> torch.Tensor:
        """The mean action."""
        return self.dist.mean

    def sample(self) -> torch.Tensor:
        """A reparameterised sample (gradients flow into mean and std)."""
        return self.dist.rsample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Log-density of ``action``, summed over the action dimensions."""
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Entropy, summed over the action dimensions."""
        return self.dist.entropy()


def gaussian_policy(mean: torch.Tensor, raw_std: torch.Tensor, min_std: float, max_std: float) -> GaussianPolicy:
    """Actor distribution: the mean is squashed by tanh, the standard deviation is
    mapped into [min_std, max_std] by a shifted sigmoid."""
    std = (max_std - min_std) * torch.sigmoid(raw_std + 2.0) + min_std
    return GaussianPolicy(torch.tanh(mean), std)


class MixedOneHot(D.OneHotCategorical):
    """One-hot categorical whose probabilities are mixed with a uniform distribution.

    The mix keeps every class reachable, so the KL terms between latent
    distributions stay bounded. Samples and the mode carry straight-through
    gradients to the probabilities / logits.
    """

    def __init__(self, logits: torch.Tensor, uniform: float):
        if uniform > 0.0:
            p = torch.softmax(logits, -1)
            logits = torch.log(p * (1.0 - uniform) + uniform / p.shape[-1])
        with _unvalidated():
            super().__init__(logits=logits, validate_args=False)

    def mode(self) -> torch.Tensor:
        """The most likely class, one-hot, with the logits' gradient."""
        logits = self.logits
        hard = one_hot(logits.argmax(-1), logits.shape[-1])
        return hard.detach() + logits - logits.detach()

    def sample(self, shape=()) -> torch.Tensor:
        """A one-hot sample with the probabilities' gradient."""
        if len(shape):
            raise ValueError("only single draws are supported")
        probs = self.probs
        return self._draw() + (probs - probs.detach())

    def _draw(self) -> torch.Tensor:
        """One draw without host synchronisation.

        ``torch.multinomial`` (behind ``OneHotCategorical.sample``) picks
        ``argmax(p / e)`` with ``e ~ Exp(1)`` per class, after two checks on ``p``
        that each read a value back to the host. This does the same draw from the
        same random stream, without the checks (``tests/test_dreamer_equivalence.py``
        compares it with torch's sampler).
        """
        probs = self._categorical.probs
        classes = probs.shape[-1]
        flat = probs.reshape(-1, classes)
        noise = torch.empty_like(flat).exponential_(1)
        index = (flat / noise).argmax(-1).reshape(probs.shape[:-1])
        return one_hot(index, classes).to(probs)
