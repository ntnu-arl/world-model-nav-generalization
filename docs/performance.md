# Performance

Measured on the laptop used for the paper (RTX 3500 Ada Laptop, 12 GB; Python 3.8, torch 2.4.1).

## Where the time goes

Fine-tuning is dominated by gradient steps. One step trains the world model on a
`batch_sz x 64` batch and the actor-critic on 15-step imaginations from every state
of that batch:

| batch | world-model step | full update (WM + actor-critic) | updates per frame | training per collected frame |
|---|---|---|---|---|
| 8 x 64 | - | 287 ms | 1 | 287 ms |
| 16 x 64 | - | 429 ms | 1/2 | 214 ms |
| 32 x 64 | - | 723 ms | 1/4 | 181 ms |
| 64 x 64 | 600 ms | (does not fit next to the simulator) | 1/8 | - |

At small batches the cost is per-kernel overhead of the sequential RSSM scan (64
steps forward and backward), not arithmetic; the paper's 64 x 64 batches use the GPU
better. A million frames - the paper's fine-tuning budget - is ~30-40 h of training
on the workstation the paper used. Environment stepping and replay are small in
comparison, which is why their speed-ups below barely move fine-tuning wall-clock;
they matter for collection, evaluation and SSL data loading.

## Optimisations made (numerically identical)

| change | before | after |
|---|---|---|
| replay buffer: append a step for 64 envs | 17.3 ms | 3.7 ms (O(1) appends; one host transfer per key) |
| replay buffer: sample a 64 x 64 batch | 45 ms | 16 ms (written into a preallocated batch) |
| sample + copy to GPU | 48 ms | 25 ms (pinned memory, asynchronous copy) |
| world-model step, peak memory at 64 x 64 | 7.27 GiB | 6.14 GiB (no `retain_graph`, linear scans) |
| world-model step, peak memory at 96 x 64 | 10.42 GiB | 9.16 GiB |
| world-model step time at 64 x 64 | 612 ms | 591 ms (stacked scan) |
| metric logging | several GPU->host syncs + a W&B call per update | device-side accumulation, one sync per `log_every` |
| GPU->host syncs per training update (16 x 64) | 900 | 2: distributions built without argument validation, `is_first` read once per sequence, one-hot sampling without multinomial's `.item()` checks; update 424 -> 394 ms (16 x 64), 710 -> 679 ms (32 x 64) |
| environment step, 16 / 64 / 512 envs | 56 / 105 / 67 ms | 33 / 37 / 51 ms: aerial_gym formatted CUDA tensors into debug messages that were then discarded (~1,700-6,000 syncs per step); `wmnav/env/patches.py` replaces the two methods with copies without that logging |
| environment step with resets, 16 envs | 34.0 ms, 115 syncs | 32.2 ms, 96 syncs: the same kind of debug f-strings on every reset (robot initial state, env bounds, reset ids) patched out too; the rollout loop reads the finished envs once per step |
| evaluation, WM3 / L3, 200 episodes, 16 envs (end to end, incl. start-up) | 78 s (legacy evaluator) | 45 s, same 187 wins / 13 crashes |

Each is covered by an equivalence test against the legacy code
([testing](testing.md)); the stacked scan changes only the order in which autograd
adds gradient contributions (float rounding). The environment patch is checked by
comparing rollouts and collected datasets against the unpatched legacy code
(bit-identical).

Where it shows: collection and evaluation are environment-bound and get the
environment speed-up. PPO training (512 envs) is bound by its learner and stays at
~3,300 frames/s. Fine-tuning is bound by its gradient steps (below), so it gains
only the few percent of the update.

## GPU settings (change numerics; off by default)

`configs/perf/default.yaml`:

| setting | effect measured | change |
|---|---|---|
| `perf.tf32=true` | full update 424 -> 330 ms at 16 x 64 (-22 %), 710 -> 520 ms at 32 x 64 (-27 %); world-model step alone at 64 x 64: -7 % | TF32 for matrix multiplies (10-bit mantissa inputs). Convolutions already use TF32 by PyTorch default, as they did in the paper runs |
| `perf.cudnn_benchmark=true` | no measurable gain here | non-deterministic algorithm choice |
| `deterministic=false` | 612 -> 599 ms | non-deterministic cuDNN kernels (the paper's fine-tuning used `true`) |

Leave them off to reproduce the paper's numerics; turn on `tf32` for new training.

## Memory

| component | peak GPU memory |
|---|---|
| Isaac Gym + depth renderer, 16-64 envs | ~5 GB (outside PyTorch) |
| world-model step, batch 8 / 16 / 32 / 64 x 64 | 0.9 / 1.9 / 3.4 / 6.1 GiB |
| actor-critic step, batch 8 / 16 / 32 x 64 | 1.8 / 3.3 / 6.2 GiB (roughly doubling with the batch) |

So on a 12 GB GPU fine-tuning fits up to `train.batch_sz=16` (the paper used 64 and
128), or 32 with `model.checkpoint_imagination=true`, and SSL at the paper's 128 x 64 needs a larger GPU (it OOMs on 12 GB with the
legacy code as well). Lowering `batch_sz` keeps the replay ratio (`train_ratio`) and
therefore increases the number of updates.

At 16 x 64 an update is CPU-bound: ~36,000 kernel launches, 440 ms of CPU time
against 379 ms of GPU work; FP32 matrix multiplies are the largest GPU cost (35 %).

## Not done (candidates)

Tried and measured, no gain (so not kept):

* Assembling a sampled batch with one `torch.cat` per key instead of one copy per
  segment: unchanged (~20 ms at 64 x 64). Sampling is bound by memory bandwidth
  (a 64 x 64 batch is ~100 MB of depth images), not by Python overhead.
* Sampling the next batch in a background thread while the GPU trains: unchanged
  for SSL training and evaluation steps at 16 and 64 x 64. The GPU work is queued
  asynchronously, so the CPU already samples while the GPU computes.
* The remaining ~95 syncs per environment step are inside aerial_gym's controller
  and pytorch3d's `matrix_to_quaternion` (boolean indexing). Removing them means
  rewriting simulator maths, which risks changing the physics, so they are left.
* The update itself: ~13,800 kernel launches with CPU and GPU time roughly equal,
  so removing launch overhead without changing numerics can gain at most a few
  percent.


* `torch.compile` of the RSSM step, actor and posterior layer: measured only 5-6 %
  faster (427 -> 403 ms at 16 x 64, 722 -> 685 ms at 32 x 64) for an 18 s compile and
  changed numerics, so not adopted. (The legacy `compile: true` never compiled
  anything, [known issues](known_issues.md).)
* CUDA graphs for the update, to remove the launch overhead that dominates small
  batches.
* bf16 autocast for the world model.
* Gradient checkpointing of the imagination is implemented
  (`model.checkpoint_imagination=true`, off by default): actor-critic peak 6.22 ->
  3.76 GiB at 32 x 64 and 7.17 GiB at 64 x 64 (out of memory without it), at +27 %
  actor-critic time. 64 x 64 plus the ~5 GB simulator still exceeds 12 GB.
