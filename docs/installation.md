# Installation

Isaac Gym preview 4 supports Python 3.8 only, so wmnav targets 3.8. On the IROS26
machine everything is already installed in the conda env `rlwms`
(`/home/luca/miniconda3/envs/rlwms`); wmnav runs from the repository root without
being installed.

## Dependencies

| package | version used | notes |
|---|---|---|
| Python | 3.8.20 | |
| torch | 2.4.1 (CUDA) | |
| Isaac Gym | preview 4 | from NVIDIA; `pip install -e isaacgym/python` |
| aerial_gym | `luca_dev` @ `dcd757dd` | editable install of `aerial_gym_dev`; the robot, controller and camera configs come from it |
| sample-factory | 2.x | PPO baseline only |
| hydra-core / omegaconf | 1.3.2 / 2.3.0 | |
| gym / gymnasium | 0.23.1 / 0.29.1 | |
| numpy | 1.23.0 | |
| wandb | 0.12.21 | optional (`logging.wandb_mode=disabled`) |
| scikit-image, scipy, pandas | | analysis |

`pip install -r requirements.txt` installs the pinned PyPI dependencies (after Isaac Gym
and aerial_gym); `pip install -e .` installs wmnav itself (optional, extras `ppo`,
`analysis`, `all`).

## Running

```bash
conda activate rlwms            # or put $CONDA_PREFIX/bin first on PATH
cd /home/luca/work/wmnav
python -m wmnav.evaluate ...
```

* Isaac Gym must be imported before torch; every simulator entry point does it on
  its first line. Import `wmnav.env` (not `wmnav.dreamer`) first in your own scripts.
* Isaac Gym compiles `gymtorch` on first import and needs `ninja` on `PATH`; scripts
  launched from another shell should put the env's `bin/` first on `PATH`
  (`scripts/reproduce/_env.sh` does).
* **One simulator per process.** Isaac Gym cannot build a second sim in a process
  that already built one (it hangs). Script several runs as separate commands;
  `wmnav.train_ppo` re-executes itself for its second phase for this reason, and
  sample-factory reads env spaces from a temporary env in a spawned process.
* W&B: the stored key of the paper runs is expired. Runs default to
  `logging.wandb_mode=offline`; sync later with `wandb sync <run_dir>/wandb/offline-run-*`
  after `wandb login --relogin`. Every run also writes `metrics.jsonl`.

## Data and outputs

Runs are written to `$WMNAV_OUTPUTS` (default `./outputs`, git-ignored) and datasets
to `$WMNAV_DATA` (default `/media/luca/hd5t/datasets/iros26`). On the IROS26 machine
both should point to the external drive `/media/luca/hd5t`, which also holds the
released checkpoints and datasets of the legacy repos (see
[checkpoints](checkpoints.md)).

## GPU memory

Isaac Gym with 16-64 envs and the depth renderer takes ~5 GB. Fine-tuning at the
paper's batch size (64 x 64) needs ~12 GB more, SSL at 128 x 64 does not fit in
12 GB either: the paper runs used a larger GPU. On a 12 GB card lower `batch_sz`
(see [performance](performance.md)).

## Tests

```bash
python -m unittest discover -s tests -t . -v                 # ~1 min
WMNAV_SIM_TESTS=1 python -m unittest tests.test_env_rollout  # simulator, ~2 min
```
Equivalence tests import the legacy repos (`WMNAV_LEGACY_WMS`, `WMNAV_LEGACY_SSL`)
and are skipped if they are absent; see [testing](testing.md).
