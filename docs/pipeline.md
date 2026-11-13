# Pipeline

```
collect ──► pretrain ──► finetune ──► evaluate · imagine · analysis
train_ppo ─────────────────────────► eval_ppo
```

Each stage is `python -m wmnav.<stage> [overrides]` with `configs/<stage>.yaml`.
Run directories: `$WMNAV_OUTPUTS/<stage>/<run_name>/<timestamp>/` with
`config.yaml` (resolved), `git_commit.txt`, `git.diff`, `git_untracked.txt`,
`metrics.jsonl`, `summary.json`, and stage outputs.

## 1. collect - exploration data

`wmnav.collect env.level=L3 policy=safe_explore episodes=10000`

Runs an exploration policy (`wmnav/data/exploration.py`) in 64 envs and saves the
episodes as one `.pt` file.

* `safe_explore` (the paper's `*_low_freq` files): full forward speed, yaw towards
  the more open side of the depth image, hard turn near obstacles.
* `uniform_biased` (`*_high_freq`): uniform random actions every step, with the
  climb command biased upwards for two thirds of the envs.

Finished episodes of at most `min_length` = 50 steps are dropped and do not count
towards `episodes`; episodes still running at the end are kept unfinished.

**Dataset format** (unchanged from the legacy repos; every existing file loads):
`OrderedDict{episode_id: {key: tensor(T, ...)}}` with keys `depth_image`
(T, 64, 96, 1), `robot_state` (T, 22), `action` (T, 4), `action_agent`, `reward`,
`is_first`, `is_terminal`, `done`, `discount`, `logprob`. Step 0 of an episode is
the reset observation (reward 0, action 0).

## 2. pretrain - SSL world model

`wmnav.pretrain data.train=[a.pt,b.pt] data.valid=[c.pt]`

Trains the world model alone on sequences sampled from the recorded episodes
(`wmnav/ssl/trainer.py`): 200 epochs x 200 steps of 128 x 64 sequences, validation
after every epoch. Writes `checkpoints/last.pt`, `best_train.pt`, `best_eval.pt`.
No simulator.

## 3. finetune - world model + actor-critic

`wmnav.finetune env.level=L3 init_from=<SSL checkpoint>`

1. 2,500 frames of uniform random actions fill the replay buffer;
2. the agent is built, its world model loaded from `init_from` (the actor-critic
   starts from scratch);
3. 128 deterministic evaluation episodes (writes `best_eval_*` checkpoints);
4. 10,000 training episodes: the agent acts with sampled actions and runs a
   gradient step every `batch_sz * batch_length / train_ratio` frames.

`checkpoints/best_train_win_rate.pt` is written whenever the mean win rate of the
last 64 training episodes reaches a new maximum - the checkpoint the paper
evaluates. Also `best_train_reward.pt`, `best_eval_*.pt`, `last.pt`.

## 4. evaluate - deterministic episodes

`wmnav.evaluate checkpoint=<agent> env.level=L2 env.num_panels=7 episodes=1000`

Writes `summary.json` (games, wins, crashes, timeouts, rates, mean length and
return) and `episodes.csv`. `record_trajectories=true` also saves every step with
the obstacle layout for the clearance analysis. See [evaluation](evaluation.md).

## 5. imagine - open-loop prediction

`wmnav.imagine checkpoint=<world model> data=[eval_env_3_low_freq.pt]`

Reconstructs the first `context` steps of each recorded episode and imagines the
rest from its actions; MSE and SSIM over context and imagination.

## PPO baseline

`wmnav.train_ppo env.level=L3` trains sample-factory PPO with 512 envs for 15M
frames, keeping a standalone snapshot at 1M frames (`ppo-L3-snapshot`; the
command then continues in a fresh process with the same seed), and logs
win/crash/timeout rates to `<train_dir>/<experiment>/episode_stats.jsonl`
(`frames` there count from the start of each process).
`wmnav.eval_ppo experiment=ppo-L3 env.level=L3` evaluates deterministically with
the same `summary.json` fields as the world-model evaluation.
