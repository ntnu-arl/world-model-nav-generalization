# Checkpoints

## Format

`wmnav/dreamer/checkpoints.py`. wmnav writes

```python
{"format": "wmnav/v2", "kind": "agent" | "world_model",
 "world_model": state_dict, "actor_critic": state_dict | None,
 "model_config": {...},        # the full model config, latent sizes included
 "meta": {...}}                # e.g. env_frames, updates, score_w, epoch
```

`load_checkpoint(path)` also reads wmnav's first format (`wmnav/v1`, older
parameter names and config schema) and the two legacy formats, and returns every
one with the current parameter names (`wmnav/dreamer/compat.py` holds the
renaming rules):

* **legacy SSL** (`ssl_wms`, `agent_{eval,train}_best.pth`): a flat state dict with
  keys `wm._orig_mod.*`;
* **legacy agent** (`wms_navigation`, `best_{train,eval}_{win_rate,reward}.pt`):
  `{"agent": state_dict, "score_r", "score_w", "step"}`, world model under
  `wm._orig_mod.*`, actor-critic under `task_behavior._orig_mod.*` (the repeated
  copies under `task_behavior.world_model.*` and `expl_behavior.*` are ignored).

Latent sizes are not stored in legacy files; `infer_latent_dims` reads them from the
tensor shapes (`latent.classes` = 64 for WM2, 32 for the others), so loading never
needs a manual override.

```python
from wmnav.dreamer.agent import DreamerAgent
from wmnav.utils.setup import model_cfg_for
cfg = model_cfg_for(OmegaConf.load("configs/model/dreamer.yaml"), path)
agent = DreamerAgent(shapes, 4, cfg).to("cuda:0")
agent.load(path)                         # agent checkpoint
agent.load(ssl_path, world_model_only=True)   # SSL world model, fresh actor-critic
```

## Released checkpoints

On the external drive (paths through the legacy repos' `outputs` symlinks):

| model | SSL (pretraining) | fine-tuned (paper) |
|---|---|---|
| WM1 / L1 | `ssl_wms/outputs/iros26/env2/2026-03-12/03-45-45/net/agent_eval_best.pth` | `wms_navigation/outputs/iros26/new_dataset/fine_tuned_env2/2026-03-25/19-58-15/data/2026-03-25-19-58-15/net/best_train_win_rate.pt` |
| WM2 / L2 | `.../env3/2026-04-07/14-33-56/net/agent_eval_best.pth` | `.../fine_tuned_env3/2026-04-10/22-14-58/data/2026-04-10-22-14-58/net/best_train_win_rate.pt` |
| WM3 / L3 | `.../env4/2026-04-12/06-43-25/net/agent_eval_best.pth` | `.../fine_tuned_env4/2026-04-15/11-19-29/data/2026-04-15-11-19-30/net/best_train_win_rate.pt` |
| WM4 / L4 | `.../env6/2026-03-21/05-08-41/net/agent_eval_best.pth` | `.../fine_tuned_env6/2026-04-17/10-39-54/data/2026-04-17-10-39-55/net/best_train_win_rate.pt` |

The SSL files are last-epoch weights despite their names ([known issues](known_issues.md)).
Each fine-tuning run has a twin in the sibling timestamp directory: same seed and
settings, different weights (except WM1's twin, which is bit-identical). The twins
were never evaluated in the L1-L4 setup; `wmnav.evaluate` on them would give a
second run per model without retraining. PPO: `wms_navigation/outputs/iros26/ppo_baseline/train_dir/ppo-L{1..4}` (15M) and `ppo-L{1..4}-wm_budget` (1M);
sample-factory checkpoints load with `wmnav.eval_ppo train_dir=... experiment=...`.

All eight Dreamer checkpoints load with strict key matching (`tests/test_checkpoints.py`).
