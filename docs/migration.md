# Migration from wms_navigation / ssl_wms

The legacy repositories are unchanged and still work; wmnav replaces them. Their
data and checkpoints load as they are.

## Where things went

| legacy | wmnav |
|---|---|
| `wms_navigation/main_collect_dataset.py` (output path edited per run) | `python -m wmnav.collect env.level=.. policy=..` |
| `ssl_wms/main.py` + `src/trainers/ssl.py` (`wtrain`) | `python -m wmnav.pretrain` + `wmnav/ssl/trainer.py` |
| `ssl_wms/src/configs/sweep_cfg_*.yaml` (W&B sweeps) | `scripts/reproduce/ssl_sweep.sh` (Hydra multirun) |
| `ssl_wms/cross_validation.py`, notebooks | `python -m wmnav.imagine` + `wmnav/analysis/imagination.py` |
| `wms_navigation/main_train_wms.py` | `python -m wmnav.finetune` |
| `wms_navigation/main_eval_wms.py` (+level, +num_pillars) | `python -m wmnav.evaluate` |
| `wms_navigation/scripts/cross_eval_wms.sh` | `scripts/reproduce/cross_eval.sh` |
| `wms_navigation/main_train_ppo.py`, `main_eval_ppo.py`, `scripts/*ppo*` | `python -m wmnav.train_ppo` / `wmnav.eval_ppo`, `scripts/reproduce/ppo_levels.sh` |
| `wms_navigation/main_eval_imagination_rl_wms.py` (broken) | `python -m wmnav.imagine` |
| `src/analysis/ppo_win_rate.py` (parsed printed blocks) | `<train_dir>/<exp>/episode_stats.jsonl` and `summary.json` |
| `src/analysis/clearance.py` | `wmnav/analysis/clearance.py` |
| `world_models/` and `ssl_wms/src/world_models/` (two copies) | `wmnav/dreamer/` |
| `world_models/memory/memory.py`, `ssl_wms/src/datasets/replay_buffer.py` | `wmnav/data/replay_buffer.py` |
| `world_models/tools.py: simulate` | `wmnav/data/rollout.py: simulate` |
| `world_models/agents.py: MultiStrategyActor` | `wmnav/data/exploration.py` |
| `src/tasks/static_env_task.py: StaticEnv` | `wmnav/env/task.py: NavigationTask` |
| `src/configs/{robots,sims,envs,assets,tasks}` | `wmnav/env/configs/` |
| `src/configs/envs/levels.py` | `wmnav/env/levels.py` |
| `src/wrappers/{sf_wrapper,aerial_gym_vec_env}.py`, `ppo/networks/` | `wmnav/ppo/` |
| `config_wms.yaml` / `config_ssl_wms.yaml` model sections | `configs/model/dreamer.yaml` |
| `static_env_ppo.yaml`, `config_ppo.yaml` | `configs/ppo/sample_factory.yaml`, `wmnav/ppo/encoder.py` |

## Deliberate changes

* **SSL checkpoints**: `last.pt`, `best_train.pt`, `best_eval.pt` with the best
  actually tracked (legacy "best" files were the last epoch).
* **Checkpoint format** `wmnav/v2` (each network once, model config stored); legacy
  and `wmnav/v1` files still load and are renamed on the fly
  ([checkpoints](checkpoints.md)).
* **Outputs**: `summary.json` / `episodes.csv` / `metrics.jsonl` per run instead of
  `wtt.csv`, printed blocks and W&B-only logs; git commit + diff + untracked files
  saved with every run; W&B defaults to offline.
* **Levels and obstacle counts are configuration** (`env.level`, `env.num_panels`)
  instead of edits to config files; no shared config class is mutated.
* **Evaluation does not run video prediction** between episodes (legacy
  `video_pred_log`, which also consumed random numbers and so changed outcomes);
  use `wmnav.imagine`. Trajectory recording for the clearance analysis is
  `record_trajectories=true` and no longer exits the process after N steps.
* **Logging cadence**: training metrics are aggregated over `logging.log_every`
  env steps (mean/std/min/max) instead of being sent at every update.
* **Hydra** 1.3 semantics: the working directory is not changed; run directories
  are `$WMNAV_OUTPUTS/<stage>/<run_name>/<timestamp>`.

* **The Dreamer implementation was rewritten** (2026-11-13) with its own structure
  and names - see below. It computes exactly what the port did (same initialisation
  from the same seed, same updates, same actions; [testing](testing.md)).

Everything else - environment, reward, observation, model, losses, schedules,
sampling, checkpoint selection - computes what the legacy code computed
([testing](testing.md)).

## Dreamer rewrite: names

### Model config (`configs/model/dreamer.yaml`)

| legacy / wmnav v1 | now |
|---|---|
| `dyn_stoch`, `dyn_discrete`, `dyn_deter`, `dyn_hidden`, `unimix_ratio` | `latent.groups`, `latent.classes`, `latent.recurrent`, `latent.hidden`, `latent.uniform_mix` |
| `units`, `act`, `norm` | `mlp_width`, `activation`, `layer_norm` |
| `encoder` / `decoder`: `cnn_keys`, `mlp_keys`, `cnn_depth`, `kernel_size`, `minres`, `mlp_units`, `outscale` | `image_key`, `vector_key`, `channels`, `kernel`, `min_resolution`, `mlp_width`, `out_scale` |
| `reward_head`, `cont_head` (`outscale`, `loss_scale`) | `reward`, `continuation` (`out_scale`, `loss_weight`) |
| `kl_free`, `dyn_scale`, `rep_scale` | `kl.free_nats`, `kl.dynamics_weight`, `kl.representation_weight` |
| `legacy_depth_target_shift` | `legacy_depth_target` |
| `imag_horizon`, `discount_lambda` | `imagination_horizon`, `return_lambda` |
| `actor.outscale`, `actor.entropy` | `actor.out_scale`, `actor.entropy_weight` |
| `critic.outscale`, `critic.slow_target_update`, `critic.slow_target_fraction` | `critic.out_scale`, `critic.ema_every`, `critic.ema_rate` |

### Parameters (state-dict keys)

| legacy / wmnav v1 | now |
|---|---|
| `encoder.net_cnn.net.*`, `encoder.net_mlp.*` | `encoder.image.*`, `encoder.vector.*` |
| `dynamics.img_in_layer`, `net_cell`, `img_out_layer`, `obs_out_layer` | `rssm.prior_in`, `rssm.gru`, `rssm.prior_hidden`, `rssm.post_hidden` |
| `dynamics.imgs_stat_layer`, `obs_stat_layer`, `w` | `rssm.prior_logits`, `rssm.post_logits`, `rssm.h0` |
| `heads.decoder.net.*`, `heads.decoder.mlp.*` | `decoder.image.*`, `decoder.vector.*` |
| `heads.reward.*`, `heads.cont.*` | `reward_head.*`, `continue_head.*` |
| `state_preproces.{running_mean, running_variance, current_count}` | `normalizer.{mean, var, count}` |
| MLP `net.<Name>_layer_i`, `net.<Name>_norm_i`, `net_mean`, `net_std` | `trunk.<3i>`, `trunk.<3i+1>`, `out`, `out_std` |
| behaviour `actor`, `value`, `slow_value`, `ema_vals` | `actor`, `critic`, `critic_ema`, `return_range` |

Checkpoints in the old names are renamed when loaded (`wmnav/dreamer/compat.py`);
released PPO checkpoints are renamed by a load hook of the PPO encoder.

### Options removed with the rewrite

No IROS26 model used them: Gaussian latents (`dyn_discrete: 0`), `dyn_rec_depth`
> 1, a zero initial state (`initial: zeros`), REINFORCE actor gradients
(`imag_gradient`), acting from the latent mean (`eval_state_mean`), heads without
gradients (`grad_heads`), symlog encoder inputs, and the alternative output
distributions. A v1 checkpoint that used a non-default value of one of these is
refused when loaded rather than silently changed.

## Dropped

Code that no IROS26 experiment used: dynamic obstacles and trajectory generators,
the other observation layouts (`unit vector`, `L1 distance`, `pos unit vector`),
the other tasks (`position_setpoint`, `procedural_world`, the old `static_env`
variants), panels/trees/objects/thin assets, the VAE, world-model-in-the-loop PPO
(`wms_in_the_loop`, `wms_ssl`, `ppo_wm`), PCGrad, the cartpole script,
`distributions/`, `sf_wrapper_old*`, `memory_old.py`, the old loggers, the
Minecraft config, `envs.py`, and the non-`wtrain` SSL loop (`WMDataset`). They
remain in the legacy repositories.
