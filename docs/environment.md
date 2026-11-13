# Environment

`wmnav/env/task.py` (`NavigationTask`), configured by `wmnav/env/configs/` and built
by `wmnav.env.factory.make_env(mode, level, num_panels, num_envs, seed)` (entry points use
`make_env_from_cfg(cfg.env, seed)`, which also resets it).

## Scene

A corridor bounded by six walls, 15 m long, 6 m wide, 7 m high (L1-L3; L4 jitters
each bound by up to +-0.5 m per episode). Obstacles are `num_panels` axis-aligned
boxes of 0.1 x 1.2 x 50 m (`big_panel.urdf`), taller than the corridor, placed in x
within 20-80 % of its length and anywhere in y. The drone (LMF2 quadrotor) spawns
at rest in the first 10-20 % of the corridor, anywhere in y and z (10-90 %); the
target is drawn in x in [13, 14] m, y in [-2, 2] m, z in [1, 5] m.

## Time

One environment step = 10 physics steps of 0.01 s = **0.1 s (10 Hz)**. Episodes
last at most **300 steps (30 s)**.

## Observations

| key | shape | content |
|---|---|---|
| `depth_image` | 64 x 96 x 1 | forward depth camera, 87 deg horizontal FOV, range 0.2-10 m normalised to [0, 1], -1 for too-near pixels |
| `robot_state` | 22 | unit vector to the target in the vehicle frame (3), distance to the target (1), rotation matrix (9), world position (3), body linear velocity (3), body angular velocity (3) |
| `action_agent` | 4 | the last action the agent sent |
| `is_first` | 1 | first observation of an episode |
| `is_terminal` | 1 | the last step ended in a crash |

## Actions

Four values in [-1, 1]; the first three are used. `a0` -> forward speed in
[0, 2] m/s, `a1` -> climb angle in [-45, 45] deg (the speed is split into vx and
vz), `a2` -> yaw rate in [-60, 60] deg/s; vy = 0. The result goes to aerial_gym's
`lmf2_velocity_control`, whose gains are re-sampled every episode within the
ranges of `aerial_gym/config/controller_config/lmf2_controller_config.py`; the
motor thrust and spin-up constants are randomised by the motor model
(`wmnav/env/configs/robot.py`).

## Episode end and reward

* **crash**: contact force above 0.05 N during the step;
* **win**: distance to the target below 0.5 m;
* **timeout**: 300 steps.

Reward per step: `2 exp(-d^2) + 2 exp(-0.008 d^2) + win * steps_left`, with `d` the
distance to the target. See [known issues](known_issues.md) on why the win must not
be treated as terminal.

## Modes

The world model and PPO were trained with different reset protocols, and trained
models depend on them, so the task implements both:

| | `mode="world_model"` | `mode="ppo"` |
|---|---|---|
| who resets | the rollout loop calls `reset_idx` on finished envs | `step` resets finished envs itself |
| `step` returns | `(obs, reward, done, info)` | `(obs, reward, terminated, truncated, info)` |
| win | `done = 1`, `is_terminal = 0` (continue predicted 1) | `truncated = 1` (bootstrapped) |
| reset clears `is_first` / last action | yes | no (as trained) |

## Levels

How the panels are placed (`wmnav/env/levels.py`); WM*i* was fine-tuned on L*i*.

| level | placement | bounds | layouts actually produced |
|---|---|---|---|
| L1 | Halton, base pair (2,3) | fixed | one, identical everywhere |
| L2 | Halton, 5 base pairs | fixed | 5 |
| L3 | scrambled Sobol | fixed | one per reset call, shared by the envs reset in it |
| L4 | uniform | +-0.5 m | independent per env |

The panel count is separate (`env.num_panels`, 5 during training); the paper's
out-of-distribution condition is L4 with 10 panels.
