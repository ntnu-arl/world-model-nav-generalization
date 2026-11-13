# Known issues

Two lists: behaviours **kept** because the released models were trained with them
(changing them would change what a loaded model sees or computes), and **bugs found**
in the legacy code while porting it, with what wmnav does about each.

## Kept for compatibility

### The decoder reconstructs `depth - 0.5`
The legacy `ConvEncoder.forward` did `obs -= 0.5` *in place*. The tensor it modified
was the batch's `depth_image`, which the loss then used as the reconstruction
target, so every released decoder was trained to output `depth - 0.5` (its raw
output, before the `+0.5` the decoder adds, learned `depth - 1`). The encoder input
centring was as intended.

wmnav: the encoder is out of place and the target shift is explicit,
`model.legacy_depth_target: true` (default). With it, losses of loaded models
are identical to the legacy ones (`tests/test_dreamer_equivalence.py`). Set `false`
only for models trained from scratch. `video_pred` returns images in the target
space, as before.

Consequence for reported numbers: image MSE is unaffected by the shift, **SSIM is
not** (its luminance term is not shift-invariant), so the paper's SSIM values are
SSIM of `depth - 0.5` images. `wmnav.imagine depth_space=raw` scores true depth.

### Continue head initialised with the reward head's output scale
The legacy world model built the continue head with `reward_head.outscale` (0.0)
although its config said `cont_head.outscale: 1.0`. wmnav reads `continuation.out_scale`
and the default config now says 0.0, which is what was used.

### The actor mean goes through tanh twice
The legacy `MLP` added a `Tanh` to the mean layer of a head named `"Actor"`, and the
`normal` distribution applied `tanh` to the mean again. The trained actors depend on
it, so wmnav keeps it: the actor's `DenseStack(tanh_out=True)` squashes the mean and
`outputs.gaussian_policy` squashes it again.

### PPO resets did not clear the last action
In PPO mode the code that resets per-episode observation fields (`is_first`, the
stored last action, ...) sat under a debug flag, so the first observation of a new
episode carried the previous episode's final action in `action_agent`, which the
PPO encoder consumes. The released PPO policies were trained that way; wmnav's
`mode="ppo"` keeps it (`wmnav/env/task.py`). World-model mode resets them, as before.

### Obstacle layouts are much less diverse than the level names suggest
With `temperature: 0` (no random shift) aerial_gym's quasi-random sampler produces:

| level | layouts |
|---|---|
| L1 | **one** layout, identical in every env and every episode |
| L2 | 5 layouts (one per Halton base pair), drawn per env reset |
| L3 | one scrambled-Sobol layout per reset call, shared by all envs reset in it (`base_list` is ignored) |
| L4 | independent uniform layouts per env |

So WM1's L1 results measure flying one memorised layout. The levels are defined in
`wmnav/env/levels.py` exactly as run; change `temperature` only for new experiments.

### Reaching the target is not what the reward pays for
`r = 2 exp(-d^2) + 2 exp(-0.008 d^2) + win * steps_left`. Hovering just outside the
0.5 m win radius earns up to 3.55 per step for the rest of the episode, while a win
earns `steps_left` once and ends it. Both learners reach the target only because a
win is not treated as terminal: the world model keeps `cont = 1` on a win, and PPO
gets the win as a *truncation* so `value_bootstrap` adds `gamma * V(s_win)`. Kept
identical on both sides so the comparison is fair, not fixed.

### `reset()` resets every env twice
The legacy task did; it changes which layouts and targets the first episodes get,
so it is kept for bit-identical rollouts.

### Evaluation samples the latent state
The policy uses the actor's mode, but the RSSM posterior is sampled
(`obs_step(..., sample=True)`), so evaluation outcomes depend on the seed, not only
on the weights.

## Bugs found in the legacy code

| issue | effect | wmnav |
|---|---|---|
| ssl_wms never updated its best-loss reference (`previous_m_eval/train`) | `agent_eval_best.pth` / `agent_train_best.pth` were overwritten every epoch: the released SSL checkpoints are the **last epoch**, and the cross-validation picked runs by min eval loss but loaded their last-epoch weights | best tracked properly; `last.pt` is the equivalent of the released files |
| aerial_gym builds `logger.debug(f"...{tensor}...")` messages in the per-physics-step path and on every reset | the f-string copies CUDA tensors to the host even with debug logging off: thousands of GPU stalls per env step | `wmnav/env/patches.py` (aerial_gym sources untouched); env step 1.3-2.8x faster |
| `torch.compile(model)` with the model driven through methods other than `forward` | compiled nothing; only added the `_orig_mod.` prefix to checkpoint keys | not used; loader strips the prefix |
| `ImagBehavior` registered the world model as a submodule; `expl_behavior` aliased `task_behavior` | checkpoints repeat the world model's keys three times and the actor-critic's twice (same storage, so no size cost) | each network once; loader takes one copy |
| `Optimizer` back-propagated with `retain_graph=True` | ~1.1 GiB more GPU memory at batch 64x64 | released after each step |
| `Optimizer` passed a parameter *generator* to clipping, then reused it for weight decay | manual weight decay could never apply (it was 0 in every config) | list of parameters |
| quadratic `torch.cat` growth in the sequence scan, the lambda-return scan and the replay buffer | slower and more memory than needed | linear; see performance.md |
| `config_wms.yaml: size: [96, 54]` | never read; the camera is 64x96 | removed |
| `static_env.py` comment: episode time = `episode_len_steps * sim.dt` | would be 3 s; an env step is 10 physics steps, so episodes are 30 s | documented correctly |
| `main_eval_imagination_rl_wms.py` | broken (invalid logger argument, 18-d state layout, hard-coded 2025 dataset) | replaced by `wmnav.imagine` |
| `deterministic_run: True` in the fine-tuning runs | did not make training reproducible: twin runs with seed 5 diverged (env3, env4, env6) - the simulator is deterministic, training on the GPU is not | documented |
| levels were changed by editing config files between runs | the only record of what a run used was the git diff stored in W&B | levels are a config value; each run stores its git state locally |
| SSL runs recorded their git state only in W&B, and the W&B files were lost; the SSL training sets were deleted | the exact SSL code state and data size cannot be recovered | every run saves `git_commit.txt`, `git.diff`, `git_untracked.txt` |

PPO pipeline bugs fixed in `wms_navigation` before the merge (wins not reported as
episode ends, stale encoder input sizes, a normaliser built empty, double depth
centring, best-checkpoint selection by reward) are fixed in wmnav as well; see
`docs/paper_params.md` and the `wms_navigation` history.

## Data notes

* Collection drops finished episodes of at most 50 steps (5 s) and does not count
  them; the eval sets on disk have exactly 3000 finished episodes each, plus the
  episodes still running when collection stopped (the "length 2" ones).
* SSL sampling used RNG seed 0 in every run; the run seed (20/40/60) only changed
  the model initialisation.
* SSL read `*.pt` from `train_env_N/`, so it probably mixed safe-explore and
  uniform-random episodes; the folders are gone, so this cannot be checked.
