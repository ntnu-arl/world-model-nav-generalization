"""Obstacle-placement levels L1-L4 of the IROS26 experiments.

The levels differ in how the panels are *placed*, not in how many there are:

=====  ==========================  =============  ============  ===================================
level  placement                   base pairs     bounds        what the sampler actually produces
=====  ==========================  =============  ============  ===================================
L1     Halton, no random shift     (2,3)          fixed         ONE layout, identical in every env
                                                                and every episode
L2     Halton, no random shift     5 pairs        fixed         5 layouts, one drawn per env reset
L3     scrambled Sobol             (ignored)      fixed         a fresh layout per reset call,
                                                                shared by all envs reset in it
L4     uniform                     (ignored)      +-0.5 m       independent per env
=====  ==========================  =============  ============  ===================================

The last column follows from aerial_gym's ``quasi_random_sampling`` with
``temperature: 0`` (no shift): Halton points depend only on the bases, and Sobol
ignores ``base_list`` and draws one (num_points, 2) sequence per call.

Reconstructed from the git diffs stored in each fine-tuning run's W&B config;
WM<i> was fine-tuned on L<i> with 5 panels.
"""

BASE_LIST_SINGLE = [[2, 3]]
BASE_LIST_FIVE = [[2, 3], [2, 5], [3, 5], [2, 7], [3, 7]]

BOUNDS_FIXED = {
    "lower_bound_min": [0.0, -3.0, 0.0],
    "lower_bound_max": [0.0, -3.0, 0.0],
    "upper_bound_min": [15.0, 3.0, 7.0],
    "upper_bound_max": [15.0, 3.0, 7.0],
}
BOUNDS_SAMPLED = {
    "lower_bound_min": [-0.5, -3.5, 0.0],
    "lower_bound_max": [0.5, -2.5, 0.0],
    "upper_bound_min": [14.5, 2.5, 7.0],
    "upper_bound_max": [15.5, 3.5, 7.0],
}


def _quasi_random(base_list, dist):
    return {"distribution": "quasi_random", "ndim": 2, "temperature": 0.0,
            "base_list": base_list, "prime_list": [], "dist": dist}


LEVELS = {
    "L1": {"distribution": _quasi_random(BASE_LIST_SINGLE, "halton"), "bounds": BOUNDS_FIXED},
    "L2": {"distribution": _quasi_random(BASE_LIST_FIVE, "halton"), "bounds": BOUNDS_FIXED},
    "L3": {"distribution": _quasi_random(BASE_LIST_FIVE, "sobol"), "bounds": BOUNDS_FIXED},
    "L4": {"distribution": {"distribution": "uniform", "ndim": 2, "temperature": 0.0,
                            "base_list": BASE_LIST_FIVE, "prime_list": [], "dist": "sobol"},
           "bounds": BOUNDS_SAMPLED},
}

TRAINING_PANELS = 5  # pillar count all four world models were trained with

# Paper names of the released models, and the level each was fine-tuned on.
WORLD_MODEL_LEVEL = {"WM1": "L1", "WM2": "L2", "WM3": "L3", "WM4": "L4"}


def level_spec(level: str) -> dict:
    """A copy of the panel distribution and corridor bounds of ``level`` (L1-L4)."""
    if level not in LEVELS:
        raise KeyError(f"unknown level {level!r}; expected one of {sorted(LEVELS)}")
    spec = LEVELS[level]
    return {"distribution": dict(spec["distribution"]), "bounds": {k: list(v) for k, v in spec["bounds"].items()}}
