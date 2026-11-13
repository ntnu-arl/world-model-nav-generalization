"""Seeding of every random generator a run uses."""

import os
import random
import secrets
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """Seed python's ``random``, numpy and torch (CPU and every GPU); returns the seed.

    ``seed=None`` picks a fresh one below 2**31. ``deterministic=True`` also
    selects deterministic cuDNN / cuBLAS kernels. Isaac Gym's GPU physics stays
    nondeterministic regardless, so simulator runs with the same seed can diverge.
    """
    value = secrets.randbelow(2 ** 31) if seed is None else int(seed)
    for seed_fn in (torch.manual_seed, np.random.seed, random.seed):  # torch: CPU and every GPU
        seed_fn(value)
    if deterministic:
        cudnn = torch.backends.cudnn
        cudnn.deterministic, cudnn.benchmark = True, False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS
    return value
