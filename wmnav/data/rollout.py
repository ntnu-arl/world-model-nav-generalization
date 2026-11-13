"""The rollout loop shared by data collection, fine-tuning and evaluation.

Same order of operations as the legacy ``tools.simulate``:

1. reset the envs that finished at the previous step and record their first
   observation (``is_first = 1``, reward 0) in the buffer;
2. ``agent(obs, done, state)`` - in training mode the agent first runs the
   gradient steps that are due;
3. step the env and record the transition (observation, action, log-prob,
   reward, done);
4. report the episodes that just ended.

It stops when ``steps`` frames (num_envs per env step) or ``episodes`` finished
episodes are reached. Finished episodes are counted in batches, so a run asked
for 200 episodes can finish 201.
"""

import collections
from typing import Callable, Dict, Optional

import numpy as np
import torch


def simulate(agent, env, buffer=None, steps: int = 0, episodes: int = 0,
             on_episodes: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None,
             on_step: Optional[Callable[[int], None]] = None, record_layout: bool = False) -> Dict[str, int]:
    """Run ``agent`` in ``env`` until ``steps`` frames or ``episodes`` episodes.

    ``buffer``: a ReplayBuffer receiving every step (None to record nothing).
    ``on_episodes(outcomes)``: called with the per-episode results of the envs that
    just finished (``env.episode_outcomes``), before they are reset.
    ``on_step(frames)``: called after every env step with the frames so far.
    ``record_layout``: also record obstacle positions, bounds and target per step
    (the buffer needs the matching extra keys; see ``LAYOUT_SPECS``).
    """
    n = env.num_envs
    frame, episode = 0, 0
    done = torch.ones(n, dtype=torch.bool, device=env.device)
    finished = torch.arange(n, device=env.device)  # envs to reset: all of them at the start
    state = None
    obs = None
    while (steps and frame < steps) or (episodes and episode < episodes):
        if len(finished):
            obs, info = env.reset_idx(finished)
            if buffer is not None:
                data = {**obs, **info}
                if record_layout:
                    data.update(env.layout())
                buffer.add_batch(env.env_ids, data, rows=finished.tolist())
                episode -= buffer.drop_short_finished(env.env_ids)

        output, state = agent(obs, done, state)
        obs, reward, done, info = env.step(output["action"])
        obs = {**obs, **info}
        if buffer is not None:
            data = {**obs, **output, "reward": reward, "done": done}
            if record_layout:
                data.update(env.layout())
            buffer.add_batch(env.env_ids, data)

        frame += n
        if on_step is not None:
            on_step(frame)
        # the one host synchronisation of the loop: which envs just finished
        finished = done.nonzero(as_tuple=False).squeeze(-1)
        if len(finished):
            episode += len(finished)
            if on_episodes is not None:
                on_episodes(env.episode_outcomes(finished))
    return {"frames": frame, "episodes": episode}




def layout_specs(num_panels: int) -> dict:
    """Replay-buffer extra keys for ``record_layout`` (6 walls + the panels)."""
    return {
        "obstacle_position": ((6 + num_panels, 3), torch.float32),
        "target_position": ((3,), torch.float32),
        "env_bounds_min": ((3,), torch.float32),
        "env_bounds_max": ((3,), torch.float32),
    }


class BestTracker:
    """Moving averages over the last ``window`` episodes, and when they improve.

    Same rule as the legacy LoggerDreamer, which selected the released
    ``best_train_win_rate`` checkpoints: nothing is reported until the window is
    full; afterwards a new best mean reward or mean win rate is flagged.
    """

    KEYS = ("rewards", "wins", "terminations", "truncations", "dist")

    def __init__(self, window: int = 64):
        self.window = window
        self.history = {k: collections.deque(maxlen=window) for k in self.KEYS}
        self.best_reward = -float("inf")
        self.best_win_rate = -float("inf")

    def add(self, outcomes: Dict[str, torch.Tensor]) -> Dict[str, object]:
        """Add finished episodes; once the window is full, report the moving averages and
        whether the mean reward or mean win rate reached a new best.
        """
        for k in self.KEYS:
            self.history[k].extend(outcomes[k].float().tolist())
        if len(self.history["rewards"]) < self.window:
            return {"full": False, "new_best_reward": False, "new_best_win_rate": False}
        means = {k: float(np.mean(v)) for k, v in self.history.items()}
        out = {"full": True, "means": means, "new_best_reward": False, "new_best_win_rate": False}
        if means["rewards"] > self.best_reward:
            self.best_reward = means["rewards"]
            out["new_best_reward"] = True
        if means["wins"] > self.best_win_rate:
            self.best_win_rate = means["wins"]
            out["new_best_win_rate"] = True
        return out


class OutcomeCounter:
    """Win / crash / timeout counts of every episode reported to it."""

    def __init__(self):
        self.games = self.wins = self.crashes = self.timeouts = 0
        self.lengths, self.returns = [], []

    def add(self, outcomes: Dict[str, torch.Tensor]) -> None:
        """Count the outcomes, lengths and returns of finished episodes."""
        self.games += len(outcomes["wins"])
        self.wins += int(outcomes["wins"].sum())
        self.crashes += int(outcomes["terminations"].sum())
        self.timeouts += int(outcomes["truncations"].sum())
        self.lengths += outcomes["length"].tolist()
        self.returns += outcomes["rewards"].tolist()

    def summary(self) -> Dict[str, float]:
        """Totals and rates over every episode counted so far."""
        g = max(self.games, 1)
        return {
            "games": self.games, "wins": self.wins, "crashes": self.crashes, "timeouts": self.timeouts,
            "win_rate": self.wins / g, "crash_rate": self.crashes / g, "timeout_rate": self.timeouts / g,
            "mean_length": float(np.mean(self.lengths)) if self.lengths else float("nan"),
            "mean_return": float(np.mean(self.returns)) if self.returns else float("nan"),
        }
