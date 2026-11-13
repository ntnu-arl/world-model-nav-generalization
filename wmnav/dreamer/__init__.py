"""DreamerV3 for depth-based drone navigation.

An implementation of the method of Hafner et al., "Mastering Diverse Domains
through World Models" (2023, arXiv:2301.04104), configured as in the IROS26
experiments: a world model with categorical latents (``world_model``,
``latent``, ``vision``), an actor-critic trained in imagination
(``actor_critic``), the agent that ties them to the environment (``agent``), and
checkpoint I/O including the legacy formats (``checkpoints``, ``compat``).
"""
