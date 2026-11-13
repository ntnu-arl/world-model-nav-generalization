# Configuration

Hydra 1.3 (`version_base="1.3"`: the working directory is not changed). Each entry
point has `configs/<entry>.yaml`, composed from groups:

| group | file | contents |
|---|---|---|
| `model` | `configs/model/dreamer.yaml` | world model + actor-critic architecture and losses |
| `env` | `configs/env/default.yaml` | `level`, `num_panels`, `num_envs`, `episode_len_steps`, `headless`, `device` |
| `paths` | `configs/paths/default.yaml` | `output_root` ($WMNAV_OUTPUTS), `data_root` ($WMNAV_DATA) |
| `logging` | `configs/logging/default.yaml` | `wandb_mode`, `project`, `log_every` |
| `perf` | `configs/perf/default.yaml` | GPU settings (TF32, cuDNN autotuning, pinned memory) |
| `sample_factory` | `configs/ppo/sample_factory.yaml` | PPO hyperparameters |

Override anything on the command line: `python -m wmnav.finetune env.level=L2
train.batch_sz=32 model.imagination_horizon=10 logging.wandb_mode=disabled`. The resolved
configuration is saved as `config.yaml` in the run directory, and
`python -m wmnav.<entry> --cfg job` prints it without running.

## Settings that matter

| setting | default | meaning |
|---|---|---|
| `env.level` | L1 | obstacle placement, [environment](environment.md) |
| `env.num_panels` | 5 | obstacle count (training value) |
| `env.num_envs` | 64 (512 for PPO) | parallel simulated drones |
| `seed`, `deterministic` | 5, true | seeds python/numpy/torch and the simulator; `deterministic` selects deterministic cuDNN kernels (GPU training is still not bit-reproducible) |
| `train.batch_sz`, `train.batch_length` | 64, 64 | sequences per batch, steps per sequence (6.4 s) |
| `train.train_ratio` | 512 | replayed frames per collected frame; sets the update frequency |
| `train.model_opt.lr` | 1e-7 | world-model learning rate during fine-tuning |
| `train.best_window` | 64 | moving-average window of the best-checkpoint rule |
| `model.legacy_depth_target` | true | reconstruct `depth - 0.5` like the released decoders, [known issues](known_issues.md) |
| `model.latent.classes` | 32 | classes per latent variable; read from the checkpoint when one is loaded |
| `perf.tf32` | false | faster, changes numerics, [performance](performance.md) |

## Paper settings that differ per model

| | WM1 | WM2 | WM3 | WM4 |
|---|---|---|---|---|
| level | L1 | L2 | L3 | L4 |
| `model.latent.classes` | 32 | 64 | 32 | 32 |
| `train.batch_sz` | 128 | 64 | 64 | 64 |
| SSL seed | 60 | 20 | 20 | 20 |

`scripts/reproduce/finetune_all.sh` applies these.
