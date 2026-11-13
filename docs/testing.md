# Testing

```bash
python -m unittest discover -s tests -t . -v                  # all non-simulator tests, ~1 min
WMNAV_SIM_TESTS=1 python -m unittest tests.test_env_rollout tests.test_collect   # simulator comparisons, ~5 min
```

Tests that compare with the legacy code import it from `WMNAV_LEGACY_WMS` and
`WMNAV_LEGACY_SSL` (default: `wms_navigation` and `ssl_wms` next to this
repository) and are skipped when those are missing; checkpoint tests are skipped
when the released checkpoints are not available. pytest is not needed
(stdlib `unittest`), but works if installed.

| file | what it establishes |
|---|---|
| `test_dreamer_equivalence.py` | same initialisation as the legacy world model; identical posteriors, losses (including the `depth - 0.5` target) and actor-critic update; world-model update bit-identical with the legacy scan and equal to float rounding with the stacked one; identical actions in both modes. CPU, exact comparisons. |
| `test_checkpoints.py` | all 4 SSL and 4 fine-tuned released checkpoints load strictly; latent sizes are inferred correctly (WM2: 64 classes); the new format round-trips the same weights |
| `test_replay_buffer.py` | contents, byte accounting, eviction, short-episode filter, saved files and sampled batches bit-identical to both legacy buffers |
| `test_env_equivalence.py` | robot, simulator, wall and panel configs attribute-identical; each level's scene equal to the legacy one; reward and action mapping bit-identical |
| `test_env_rollout.py` | 400 steps x 16 envs under the same seed and actions: observations, rewards and flags bit-identical in world-model mode (L3) and PPO mode (L1) |
| `test_checkpointing.py` | gradient checkpointing of the imagination gives the same imagined trajectories, losses and update |
| `test_dreamer_equivalence.py` (`TestSyncFreeSampling`) | the sync-free one-hot sampler draws the same samples and leaves the RNG in the same state as torch's, on CPU and GPU |
| `test_collect.py` | `wmnav.collect` against the legacy collection pipeline (legacy env, explorer, `tools.simulate`, buffer): same seed gives the same dataset, episode by episode, for both exploration policies (simulator, `WMNAV_SIM_TESTS=1`) |
| `test_exploration.py` | both exploration policies give the legacy actor's actions and leave the random stream where it left it |
| `test_ppo_encoder.py` | PPO encoder: same initialisation and parameter names; identical outputs with the released 15M weights |

End to end, `wmnav.evaluate` on the released WM3 / L3 (16 envs, seed 5, 200
episodes) gives the legacy evaluator's exact counts (187 wins, 13 crashes, 0
timeouts).
