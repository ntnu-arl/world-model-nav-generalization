# Model

DreamerV3 (Hafner et al., 2023, arXiv:2301.04104), implemented in `wmnav/dreamer/`
and configured by `configs/model/dreamer.yaml`. The same world model class is
pretrained self-supervised and then fine-tuned with the actor-critic.

| module | contents |
|---|---|
| `world_model.py` | `WorldModel`: encoder, latent dynamics, heads, losses, optimizer, `video_pred` |
| `latent.py` | `LatentDynamics`: prior / posterior steps, `observe`, `imagine`, KL terms |
| `vision.py` | image encoder / decoder and the observation encoder / decoder |
| `actor_critic.py` | `ActorCritic`, TD(lambda) returns, return normalisation |
| `outputs.py` | output distributions (two-hot, squared errors, continuation flag, Gaussian policy, mixed one-hot) |
| `blocks.py` | initialisation, layers, GRU cell, `DenseStack` (MLP), the time loop, optimizer wrapper, training schedule |
| `normalizer.py` | running standardisation of the robot state |
| `agent.py` | `DreamerAgent`: acting, training schedule, save / load |
| `checkpoints.py`, `compat.py` | checkpoint I/O; renaming of legacy / v1 checkpoints |

## World model (`world_model.py`)

* **Encoder**: depth image -> 4 stride-2 "same" convolutions (16-32-64-128 channels,
  kernel 4, channel LayerNorm, SiLU) -> 3072 features; robot state (standardised by
  running statistics) -> 2 x 512 MLP. Concatenated: 3584.
* **Latent dynamics** (`latent.py`): GRU with 1024 units (LayerNorm, update bias -1); stochastic
  state of 32 categoricals with 32 classes (64 for WM2), 1 % uniform mix; learned
  initial state. Features = flattened stochastic state + GRU state (2048, or 3072
  for WM2).
* **Heads**: image decoder (transposed convolutions back to 64 x 96), robot-state
  decoder (2 x 512 MLP), reward (2 x 512 MLP, two-hot over 255 symlog buckets in
  [-20, 20]), continue (2 x 512 MLP, Bernoulli).

Loss per step: image summed squared error (target `depth - 0.5`, see
[known issues](known_issues.md)) + robot-state summed squared error in symlog space
+ reward two-hot cross-entropy + continue log-loss + KL balancing with free bits 1
(0.5 dynamics / 0.1 representation). `continue = 1 - is_terminal`, so a win does
not end the predicted episode.

## Actor-critic (`actor_critic.py`)

Imagination of 15 steps (1.5 s) from every posterior state of a training batch.
Actor: 2 x 512 MLP, Gaussian with learned std in [0.1, 1], entropy bonus 1e-3,
trained through the dynamics. Critic: 2 x 512 MLP, two-hot symlog, slow target (EMA
0.02). Discount 0.997, lambda 0.95, returns normalised by the 5-95 % percentile
range (EMA). Adam, lr 3e-5 for both.

## Training schedules

| stage | batch | steps | optimiser |
|---|---|---|---|
| SSL (`wmnav.pretrain`) | 128 x 64 | 200 epochs x 200 = 40,000 | AdamW, lr 1e-4, wd 0.01, clip 1000 |
| fine-tuning (`wmnav.finetune`) | 64 x 64 (WM1: 128 x 64) | one per `batch_sz * batch_length / train_ratio` frames (8 at 64 x 64, 16 at 128 x 64) | world model Adam lr 1e-7; actor/critic as above |

Fine-tuning keeps training the world model (at a very low learning rate) together
with the actor-critic, which starts from scratch. The robot-state statistics come
from the SSL checkpoint and keep updating.

## Agent (`agent.py`)

`DreamerAgent(obs, done, state, training)`: in training mode it first runs the due
gradient steps (100 extra on the first call), then acts with a sampled action; in
evaluation mode it acts with the actor's mode. The recurrent state is reset per env
on `is_first`.
