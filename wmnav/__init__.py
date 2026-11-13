"""wmnav - world-model based drone navigation in Aerial Gym.

One package for the whole pipeline: collecting exploration data, self-supervised
world-model pretraining, fine-tuning the world model with an actor-critic in
imagination (DreamerV3), the PPO baseline, and evaluation.

Simulator-dependent modules (``wmnav.env`` and everything that steps an
environment) import Isaac Gym, which must be imported before torch; the entry
points take care of that. ``wmnav.dreamer`` and ``wmnav.data`` are simulator-free.
"""

__version__ = "1.0.0"
