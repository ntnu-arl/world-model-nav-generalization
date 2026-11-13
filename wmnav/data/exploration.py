"""Exploration policies used to collect the SSL datasets.

Two policies produced the IROS26 data (the dataset file names call them
``low_freq`` and ``high_freq``; both are recorded at 10 Hz - the names refer to
how fast the *actions* change, not to the sampling rate):

* ``safe_explore`` ("low_freq"): a depth-based corridor follower. Constant
  forward command drawn per episode in [1.0, 1.2] (clipped to 1, i.e. full
  speed), yaw rate steering towards the more open side of the image, a hard turn
  when the centre third is closer than 0.3 (normalised depth).
* ``uniform_biased`` ("high_freq"): i.i.d. uniform actions in [-1, 1] every step;
  the climb command of the second and last thirds of the envs is raised to the
  power 1/8 and 1/16 so that some trajectories hold or gain altitude instead of
  sinking.

The legacy actor also contained OU, Lissajous and smoothed-random strategies, but
with ``group_size = num_envs`` their env groups were empty; they were never used
and are not ported.
"""

import torch


class ExplorationPolicy:
    """Batched scripted exploration over ``num_envs`` environments (``safe_explore`` or
    ``uniform_biased``, see the module docstring).
    """

    def __init__(self, num_envs: int, device, mode: str = "safe_explore"):
        assert mode in ("safe_explore", "uniform_biased"), mode
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        self.steps = torch.zeros(num_envs, device=device)
        self._draw_unused(num_envs)
        self.vx_coeff = torch.rand((num_envs,), device=device) * 0.2 + 1.0

    def reset(self, done_mask):
        """Restart the per-episode state (step counter, forward command) of the envs
        flagged in ``done_mask``.
        """
        if done_mask is None or not done_mask.any():
            return
        idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
        self.steps[idx] = 0.0
        self._draw_unused(idx.shape[0])
        self.vx_coeff[idx] = torch.rand((idx.shape[0],), device=self.device) * 0.2 + 1.0

    def _draw_unused(self, n):
        # The legacy actor drew Lissajous frequency and phase here (never used). The
        # draws are kept so that the random stream - and with it, since the simulator
        # is deterministic, the collected data - is the same for the same seed.
        torch.rand((n, 4), device=self.device)
        torch.rand((n, 4), device=self.device)

    def _safe_explore(self, depth):
        if depth.ndim == 4:
            depth = depth[..., 0]
        w = depth.shape[-1]
        left = depth[:, :, : w // 3].mean(dim=(1, 2))
        mid = depth[:, :, w // 3: 2 * w // 3].mean(dim=(1, 2))
        right = depth[:, :, 2 * w // 3:].mean(dim=(1, 2))
        yaw = 3.0 * (left - right)
        yaw = yaw + (mid < 0.3).float() * 10.0 * torch.sign(left - right)
        action = torch.zeros((depth.shape[0], 4), device=self.device)
        action[:, 0] = self.vx_coeff
        action[:, 2] = yaw
        return action

    def _uniform_biased(self):
        n = self.num_envs
        raw = torch.rand((n, 4), device=self.device)
        s1, s2 = n // 3, 2 * n // 3
        raw[s1:s2, 1] = raw[s1:s2, 1] ** (1 / 8.0)
        raw[s2:, 1] = raw[s2:, 1] ** (1 / 16.0)
        return raw * 2 - 1

    def get_action(self, depth):
        """Actions in [-1, 1] for every env, from the current depth images."""
        action = self._safe_explore(depth) if self.mode == "safe_explore" else self._uniform_biased()
        self.steps += 1
        return torch.clamp(action, -1, 1)

    def __call__(self, obs, done, state=None):
        """Agent interface of the rollout loop."""
        self.reset(done)
        action = self.get_action(obs["depth_image"])
        return {"action": action, "logprob": torch.zeros(self.num_envs, device=self.device)}, None


class UniformRandomPolicy:
    """Uniform actions in [-1, 1]; prefills the replay buffer before fine-tuning."""

    def __init__(self, num_envs: int, num_actions: int = 4):
        low = -torch.ones(num_envs, num_actions)
        self.dist = torch.distributions.Independent(torch.distributions.Uniform(low, -low), 1)

    def __call__(self, obs, done, state=None):
        action = self.dist.sample()
        return {"action": action, "logprob": self.dist.log_prob(action)}, None
