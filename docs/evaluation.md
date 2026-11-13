# Evaluation

## World models

```bash
python -m wmnav.evaluate name=WM2 checkpoint=<best_train_win_rate.pt> \
    env.level=L4 env.num_panels=7 env.num_envs=16 episodes=1000
```

Deterministic actor (the RSSM latent is still sampled), `env.num_envs` parallel
envs, until `episodes` episodes have finished (they finish in batches, so the count
can overshoot by a few). `summary.json`: `games`, `wins`, `crashes`, `timeouts`,
`win_rate`, `crash_rate`, `timeout_rate`, `mean_length`, `mean_return`, and the
checkpoint's metadata. A win and a crash in the same step count as both.

The paper's protocol - 16 envs, seed 5, non-deterministic cuDNN - is the default
of `configs/evaluate.yaml`. With it the evaluator reproduces the legacy evaluator's
counts exactly (WM3 on L3, 5 panels, 200 episodes: 187 wins, 13
crashes). Statistical precision: the standard error of a win rate p over n games is
sqrt(p(1-p)/n) - about 2.5 points at n = 200 and 1 point at n = 1000; prefer 1000.

**Cross-evaluation** (every model on every level): `scripts/reproduce/cross_eval.sh`
with `PANELS=5|7|10`, then

```bash
python -m wmnav.analysis.tables $WMNAV_OUTPUTS/evaluate --out cross_eval
```
which writes the per-run table and model x level matrices of win, crash and
timeout rate (with row/column means) for each panel count.

## PPO

```bash
python -m wmnav.eval_ppo experiment=ppo-L2 env.level=L2 episodes=1000
```
Use `checkpoint=latest` (default). sample-factory's `best` is selected by mean
episode reward, which in this task favours hovering next to the target over
reaching it (on L2 it picked a 5.6M-frame policy with 40 % wins over the final
one with 99 %).

## Open-loop prediction (world model quality)

```bash
python -m wmnav.imagine checkpoint=<world model or agent> \
    data=[/media/luca/hd5t/datasets/iros26/datasets_new/eval_env_3/eval_env_3_low_freq.pt] \
    context=10 min_after_context=25
```
MSE and SSIM of the predicted depth over all steps, the 10 context steps (1 s) and
the imagined steps; per-episode values in `episodes.csv`. `depth_space=legacy`
(default) scores the images as the paper did; `raw` scores true depth (matters for
SSIM only).

## Clearance

```bash
python -m wmnav.evaluate ... record_trajectories=true
python -m wmnav.analysis.clearance --buffers WM3_L3=<run>/trajectories.pt --csv clearance.csv
```
Per episode, the minimum distance to the nearest panel or wall along the flown
path; distribution summary per run.
