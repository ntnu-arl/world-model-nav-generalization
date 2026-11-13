"""Navigation task: fly through a corridor of panels to a target, from depth images.

Episode (300 steps of 0.1 s = 30 s): the drone spawns at rest in the first 10-20 %
of the corridor; the target is drawn in x in [13, 14] m, y in [-2, 2] m, z in
[1, 5] m. The episode ends on a crash (contact force > 0.05 N), on a win
(distance to target < 0.5 m) or at the time limit.

Action (4 values in [-1, 1], only the first three used): a[0] -> forward speed in
[0, 2] m/s, a[1] -> climb angle in [-45, 45] deg (the speed is split into vx/vz),
a[2] -> yaw rate in [-60, 60] deg/s; vy = 0. Sent to lmf2_velocity_control.

Observation: depth image (64, 96, 1) in [0, 1] (-1 = too near), robot state (22):
unit vector to the target in the vehicle frame (3), distance (1), rotation matrix
(9), world position (3), body linear velocity (3), body angular velocity (3);
plus the last agent action, is_first, is_terminal.

Reward: 2 exp(-d^2) + 2 exp(-0.008 d^2) + left_steps * win (a3 = a4 = 0).

Two reset modes, matching how each learner was trained (both are reproduced
because trained checkpoints depend on them):

* ``mode="world_model"``: ``step`` never resets; it returns
  ``(obs, reward, done, info)`` and the rollout loop resets finished envs with
  ``reset_idx``, which marks ``is_first``, clears ``is_terminal`` and zeroes the
  stored last action. A win sets ``done`` but not ``is_terminal``, so the world
  model keeps bootstrapping past it.
* ``mode="ppo"``: ``step`` resets finished envs itself (sample-factory API) and
  returns ``(obs, reward, terminated, truncated, info)`` with ``terminated`` =
  crash and ``truncated`` = timeout OR win, so ``value_bootstrap`` treats a win
  like the world model does. In this mode the legacy reset did not zero the
  stored last action or set ``is_first`` (that code sat under a debug flag); the
  PPO policies were trained that way and it is kept.
"""

import datetime
import uuid
from typing import Dict

import numpy as np
import torch
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.task.base_task import BaseTask
from aerial_gym.utils.math import quat_rotate_inverse, quat_to_rotation_matrix
from gym.spaces import Box, Dict as DictSpace

OBS_KEYS = ("robot_state", "depth_image", "is_terminal", "is_first", "action_agent")


def _episode_id():
    return f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex}"


def _uniform(low, high):
    def sample(tensor):
        return torch.rand_like(tensor) * (high - low) + low
    return sample


@torch.jit.script
def exp_func(x, a, b):
    # type: (Tensor, float, float) -> Tensor
    """a * exp(-b x^2), the reward's distance kernel."""
    return a * torch.exp(-b * x * x)


def compute_reward(position_w, target_w, rotation_wv, crash, left_steps, p: Dict[str, torch.Tensor]):
    """Reward and win flag; unchanged from the legacy ``compute_reward_r``.

    r = a1 exp(-b1 d^2) + a2 exp(-b2 d^2) + a3 crash + a4 win + a5 win * left_steps,
    with d the distance to the target (computed in the vehicle frame).
    """
    target_v = torch.matmul(rotation_wv.transpose(1, 2), (target_w - position_w).unsqueeze(-1)).squeeze(-1)
    distance = torch.linalg.norm(target_v, dim=1)
    wins = distance < p["dist_thr"]
    reward = (exp_func(distance, p["a1"], p["b1"]) + exp_func(distance, p["a2"], p["b2"])
              + p["a3"] * crash + p["a4"] * wins + p["a5"] * wins * left_steps)
    return reward, wins


def transform_action(action: torch.Tensor) -> torch.Tensor:
    """Agent action in [-1, 1]^4 -> velocity command [vx, vy, vz, yaw rate]."""
    clamped = torch.clamp(action, -1.0, 1.0)
    max_speed = 2.0                      # [m/s]
    max_yawrate = torch.pi / 3           # [rad/s]
    max_inclination_angle = torch.pi / 4 # [rad]
    clamped[:, 0] += 1.0
    out = torch.zeros((clamped.shape[0], 4), device=action.device)
    out[:, 0] = clamped[:, 0] * torch.cos(max_inclination_angle * clamped[:, 1]) * max_speed / 2.0
    out[:, 1] = 0
    out[:, 2] = clamped[:, 0] * torch.sin(max_inclination_angle * clamped[:, 1]) * max_speed / 2.0
    out[:, 3] = clamped[:, 2] * max_yawrate
    return out


class NavigationTask(BaseTask):
    """The navigation task on top of aerial_gym (see the module docstring for the
    episode, observation, action, reward and the two reset modes).
    """

    def __init__(self, task_config, seed=None, num_envs=None, headless=None, use_warp=None):
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)  # seeds numpy and torch with task_config.seed

        cfg = self.task_config
        self.mode = cfg.mode
        assert self.mode in ("world_model", "ppo"), self.mode
        self.device = cfg.device
        self.reward_parameters = {k: torch.tensor(v, device=self.device) for k, v in cfg.reward_parameters.items()}

        self.sim_env = SimBuilder().build_env(
            sim_name=cfg.sim_name, env_name=cfg.env_name, robot_name=cfg.robot_name,
            controller_name=cfg.controller_name, args=cfg.args, device=self.device,
            num_envs=cfg.num_envs, use_warp=cfg.use_warp, headless=cfg.headless,
        )
        self.num_envs = self.sim_env.num_envs
        n, a_dim = self.num_envs, cfg.action_space_dim

        self.obs_dict = self.sim_env.get_obs()  # live tensors, updated in place by the sim
        self.terminations = self.obs_dict["crashes"]          # alias: crash in the last env step
        self.truncations = torch.zeros_like(self.terminations)
        self.wins = torch.zeros_like(self.terminations)
        self.dones = torch.zeros_like(self.terminations)
        self.rewards = torch.zeros(n, device=self.device)
        self.discounts = torch.ones(n, device=self.device)
        self.cumulative_rewards = torch.zeros(n, device=self.device)
        self.target_position = torch.zeros_like(self.obs_dict["robot_position"])
        self.env_dist = torch.zeros(n, device=self.device)
        self.step_env = torch.zeros(n, device=self.device)
        self.action_agent = torch.zeros((n, a_dim), device=self.device)
        self.actions = torch.zeros((n, a_dim), device=self.device)
        self.env_to_reset = None

        self.x_target_fn = _uniform(*cfg.target_x)
        self.y_target_fn = _uniform(*cfg.target_y)
        self.z_target_fn = _uniform(*cfg.target_z)

        cam = cfg.camera_shape  # (height, width, 1)
        self.depth_images = self.obs_dict["depth_range_pixels"].view(n, *cam)
        self.action_space = Box(low=-1.0, high=1.0, shape=(a_dim,), dtype=np.float32)
        self.observation_space = DictSpace({
            "robot_state": Box(low=-1.0, high=1.0, shape=(cfg.observation_space_dim,), dtype=np.float32),
            "depth_image": Box(low=-1.0, high=10.0, shape=tuple(cam), dtype=np.float32),
            "action_agent": Box(low=-1.0, high=1.0, shape=(a_dim,), dtype=np.float32),
            "is_first": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "is_terminal": Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        self.task_obs = {
            "robot_state": torch.zeros((n, cfg.observation_space_dim), device=self.device),
            "depth_image": self.depth_images,
            "is_terminal": self.terminations,
            "is_first": torch.zeros(n, device=self.device),
            "action_agent": self.action_agent,
        }
        self.env_ids = [_episode_id() for _ in range(n)]

        # obstacles are static: zero velocity commands, passed as the legacy task did
        n_assets = self.sim_env.IGE_env.num_assets_per_env - 1
        self.asset_twist = torch.zeros((n, n_assets, 6), device=self.device)

        # episode counters, kept on the device (read with episode_stats())
        self._stats = torch.zeros(4, device=self.device)  # episodes, wins, crashes, timeouts

    # ----------------------------------------------------------------- basics

    def __len__(self):
        return self.num_envs

    def close(self):
        self.sim_env.delete_env()

    def render(self, mode="human"):
        self.sim_env.render(render_components="sensors")

    def randomize_target(self, env_ids):
        """Draw new target positions for ``env_ids``."""
        self.target_position[env_ids, 0] = self.x_target_fn(self.target_position[env_ids, 0])
        self.target_position[env_ids, 1] = self.y_target_fn(self.target_position[env_ids, 1])
        self.target_position[env_ids, 2] = self.z_target_fn(self.target_position[env_ids, 2])

    # ----------------------------------------------------------------- resets

    def reset(self):
        """Reset every env; returns (observations, {})."""
        all_envs = torch.arange(self.num_envs)
        self.env_to_reset = all_envs
        # two full resets, as the legacy task did (kept: it changes which layouts
        # and targets the first episodes get)
        self.reset_idx(all_envs)
        self.reset_idx(all_envs)
        self._process_obs()
        return self.task_obs, {}

    def reset_idx(self, env_ids):
        """Reset the envs in ``env_ids`` the way the current ``mode`` requires."""
        if self.mode == "world_model":
            return self._reset_idx_world_model(env_ids)
        return self._reset_idx_ppo(env_ids)

    def _reset_common(self, env_ids):
        self.sim_env.reset_idx(env_ids)
        self.randomize_target(env_ids)

    def _reset_idx_world_model(self, env_ids):
        if len(env_ids) == 0:
            return None
        self._reset_common(env_ids)
        self.step_env[env_ids] = 0
        self.task_obs["is_first"][env_ids] = 1.0
        self.terminations[env_ids] = False       # the first observation is never terminal
        self.action_agent[env_ids] = 0.0
        self.dones[env_ids] = False
        self.wins[env_ids] = False
        self.truncations[env_ids] = False
        self.discounts[env_ids] = 1.0
        self.rewards[env_ids] = 0.0
        self.render()
        self.sim_env.IGE_env.post_physics_step()
        self.cumulative_rewards[env_ids] = 0.0
        for i in env_ids.tolist():  # one transfer, not one per env
            self.env_ids[i] = _episode_id()
        info = {"reward": self.rewards.clone(), "discount": self.discounts.clone()}
        obs, _, _, _ = self._return_tuple()
        return obs, info

    def _reset_idx_ppo(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_common(env_ids)
        self.render()
        self.sim_env.IGE_env.post_physics_step()
        self.cumulative_rewards[env_ids] = 0.0
        return

    # ------------------------------------------------------------------- step

    def step(self, actions):
        """Apply ``actions`` (in [-1, 1]) for one env step and return the step tuple of
        the current mode.
        """
        self.dones[:] = False
        self.task_obs["is_first"][:] = 0
        self.action_agent[:] = actions.clone()
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)
        if self.task_config.preprocess_actions:
            self.actions[:] = transform_action(self.actions)

        self.sim_env.step(actions=self.actions, env_actions=self.asset_twist)
        self.rewards[:], self.wins[:] = compute_reward(
            self.obs_dict["robot_position"], self.target_position,
            quat_to_rotation_matrix(self.obs_dict["robot_orientation"]), self.obs_dict["crashes"],
            self.task_config.episode_len_steps - self.sim_env.sim_steps, self.reward_parameters,
        )
        self.cumulative_rewards += self.rewards
        self.truncations[:] = torch.where(self.sim_env.sim_steps >= self.task_config.episode_len_steps, 1, 0)
        self.render()

        self.env_to_reset = (
            self.sim_env.collision_tensor * int(self.sim_env.cfg.env.reset_on_collision)
            + self.truncations + self.wins
        ).nonzero(as_tuple=False).squeeze(-1)
        self.discounts[:] = torch.where(self.terminations > 0.0, 0.0, 1.0)  # 0 only on a crash
        self.dones[self.env_to_reset] = True
        self.step_env += 1
        self.env_dist[:] = torch.linalg.norm(self.target_position - self.obs_dict["robot_position"], dim=-1)
        self._count_episodes(self.env_to_reset)

        if self.mode == "ppo":
            self.reset_idx(self.env_to_reset)
        return self._return_tuple()

    # ------------------------------------------------------------ observations

    def _process_obs(self):
        to_target_w = self.target_position - self.obs_dict["robot_position"]
        vec = quat_rotate_inverse(self.obs_dict["robot_vehicle_orientation"], to_target_w)
        dist = torch.norm(vec, dim=-1)
        rs = self.task_obs["robot_state"]
        rs[:, 0:3] = vec / dist.unsqueeze(1)
        rs[:, 3] = dist
        rs[:, 4:13] = quat_to_rotation_matrix(self.obs_dict["robot_orientation"]).reshape(-1, 9)
        rs[:, 13:16] = self.obs_dict["robot_position"]
        rs[:, 16:19] = self.obs_dict["robot_body_linvel"]
        rs[:, 19:22] = self.obs_dict["robot_body_angvel"]

    def _return_tuple(self):
        self._process_obs()
        obs = {k: v.detach().clone() for k, v in self.task_obs.items()}
        if self.mode == "ppo":
            truncated = torch.logical_or(self.truncations.bool(), self.wins.bool())
            return (obs, self.rewards.detach().clone(), self.terminations.detach().clone(),
                    truncated.detach().clone(), {"discount": self.discounts.detach().clone()})
        return (obs, self.rewards.detach().clone(), self.dones.detach().clone(),
                {"discount": self.discounts.detach().clone()})

    # ---------------------------------------------------------------- metrics

    def _count_episodes(self, env_ids):
        if len(env_ids):
            self._stats += torch.stack([
                torch.tensor(float(len(env_ids)), device=self.device),
                self.wins[env_ids].float().sum(),
                self.terminations[env_ids].float().sum(),
                self.truncations[env_ids].float().sum(),
            ])

    def episode_stats(self, reset: bool = False) -> Dict[str, float]:
        """Episodes finished since the last reset of the counters, and their outcome
        rates (a win and a crash in the same step count as both)."""
        episodes, wins, crashes, timeouts = self._stats.tolist()
        if reset:
            self._stats.zero_()

        def rate(x):
            return x / episodes if episodes else float("nan")

        return {"episodes": episodes, "win_rate": rate(wins), "crash_rate": rate(crashes),
                "timeout_rate": rate(timeouts)}

    def episode_outcomes(self, env_ids) -> Dict[str, torch.Tensor]:
        """Per-episode results for envs that just finished (read before resetting them)."""
        return {
            "rewards": self.cumulative_rewards[env_ids],
            "wins": self.wins[env_ids],
            "terminations": self.terminations[env_ids],
            "truncations": self.truncations[env_ids],
            "dist": self.env_dist[env_ids],
            "length": self.step_env[env_ids],
        }

    def layout(self) -> Dict[str, torch.Tensor]:
        """Obstacle positions, bounds and target - for trajectory logging."""
        return {
            "obstacle_position": self.obs_dict["obstacle_position"],
            "target_position": self.target_position,
            "env_bounds_min": self.obs_dict["env_bounds_min"],
            "env_bounds_max": self.obs_dict["env_bounds_max"],
        }
