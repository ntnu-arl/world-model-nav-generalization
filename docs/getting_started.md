# Getting started: install and use, step by step

This guide goes from an empty machine to evaluating a released model, training the
full pipeline and the PPO baseline. Every command is run from the repository root.
The versions are those of the IROS26 runs; the reference doc for each topic is linked
at the end of each step.

**You need:** Linux (x86-64), an NVIDIA GPU with a recent driver, conda, git.
GPU memory: ~6 GB to evaluate, ~24 GB to fine-tune or pretrain at the paper's batch
sizes (smaller GPUs: lower the batch size, see [performance](performance.md)).

---

## Part 1 - Installation

### Step 1. Create the Python environment

Isaac Gym preview 4 runs on Python 3.8 only.

```bash
conda create -n wmnav python=3.8 -y
conda activate wmnav
```

### Step 2. Install PyTorch (CUDA build)

Install it first: pytorch3d (step 4) is built against it.

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.4.1+cu121 True
```

### Step 3. Install Isaac Gym preview 4

Isaac Gym is not on PyPI. Download *Isaac Gym - Preview 4* from NVIDIA
(<https://developer.nvidia.com/isaac-gym>, free account), then:

```bash
tar -xf IsaacGym_Preview_4_Package.tar.gz -C ~/tools      # any location
pip install -e ~/tools/isaacgym/python
pip install ninja==1.11.1.3                               # Isaac Gym compiles gymtorch with it
python -c "import isaacgym; import torch; print('ok')"    # the first import compiles gymtorch (~1 min)
```

If this fails with `libpython3.8.so.1.0: cannot open shared object file`, run
`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` (add it to your shell
profile or to the env's activation script).

### Step 4. Install the libraries aerial_gym needs from source

aerial_gym imports pytorch3d (controllers) and urdfpy (robot assets). The paper
environment has pytorch3d 0.7.8 and urdfpy from its GitHub repository.

```bash
# pytorch3d 0.7.8, built against the torch above (needs the CUDA toolkit, nvcc; ~10-20 min)
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

# urdfpy from source
git clone https://github.com/mmatl/urdfpy.git ~/src/urdfpy
pip install -e ~/src/urdfpy
```

### Step 5. Install aerial_gym (the version of the paper)

wmnav uses the `luca_dev` branch of `ntnu-arl/aerial_gym_dev` at commit `dcd757dd`
(the robot, controller and camera configs come from it). You need access to that
repository.

```bash
git clone git@github.com:ntnu-arl/aerial_gym_dev.git ~/src/aerial_gym_dev
cd ~/src/aerial_gym_dev && git checkout dcd757dd && cd -
pip install -e ~/src/aerial_gym_dev          # also installs warp-lang 0.10.1, trimesh, rl-games
```

### Step 6. Install wmnav

```bash
git clone <wmnav repository URL> wmnav && cd wmnav
pip install -r requirements.txt              # pinned dependencies, PPO and analysis included
pip install -e .                             # optional: makes `wmnav` importable from anywhere
```

`requirements.txt` installs the exact versions of the paper runs. To install only what
you need instead: `pip install -e .` (core), `".[ppo]"`, `".[analysis]"`, `".[all]"`.

### Step 7. Check the installation

```bash
python -m unittest discover -s tests -t .                  # ~1 min, no simulator; ends with "OK"
WMNAV_SIM_TESTS=1 python -m unittest tests.test_env_rollout  # starts the simulator, ~2 min
```

Tests that compare against the legacy repositories or need the released checkpoints
are reported as *skipped* when those are not on the machine; that is expected.
Always pass `-t .` (it makes Isaac Gym load before torch). See [testing](testing.md).

### Step 8. Choose where runs and datasets go

```bash
export WMNAV_OUTPUTS=/path/to/outputs     # run directories (default: ./outputs)
export WMNAV_DATA=/path/to/datasets       # collected datasets
```

Put both lines in your shell profile. Every run writes to
`$WMNAV_OUTPUTS/<stage>/<run name>/<date_time>/`.

---

## Part 2 - Using wmnav

All entry points are run as `python -m wmnav.<stage> key=value ...`. The defaults
are in `configs/<stage>.yaml`; any key can be overridden on the command line (nested
keys with dots: `env.level=L2`). Show the full resolved config without running:
`python -m wmnav.evaluate checkpoint=x --cfg job`. Settings: [configuration](configuration.md).

Every run directory contains `config.yaml` (the resolved config), `git_commit.txt`,
`git.diff`, `metrics.jsonl`, `summary.json` and the stage's outputs.

`env.level` picks one of the four obstacle-placement levels `L1`-`L4` (from one fixed
layout in L1 to independent random layouts per env in L4, see `wmnav/env/levels.py`);
`env.num_panels` is the number of obstacles (every model was trained with 5).

### Step 9. Evaluate a released model

```bash
python -m wmnav.evaluate name=WM3 env.level=L3 env.num_panels=5 episodes=1000 \
    checkpoint=/path/to/best_train_win_rate.pt
```

The result is in `$WMNAV_OUTPUTS/evaluate/WM3_L3_p5/<date_time>/summary.json`
(`win_rate`, `crash_rate`, `timeout_rate`, mean length and return) with one row per
episode in `episodes.csv`. The released checkpoints and where they are:
[checkpoints](checkpoints.md). Any checkpoint can be evaluated on any level and panel
count - that is the cross-evaluation of the paper. More: [evaluation](evaluation.md).

### Step 10. Train your own world model: the full pipeline

The pipeline has four stages; each one's output is the next one's input.

```
collect ──► pretrain ──► finetune ──► evaluate
```

**10a. Collect exploration data** (simulator, ~64 envs in parallel).

```bash
python -m wmnav.collect env.level=L3 policy=safe_explore episodes=10000          # training set
python -m wmnav.collect env.level=L3 policy=safe_explore episodes=1000 seed=6    # validation set
```

Each run saves one `.pt` file under `$WMNAV_DATA/collected/` and prints its path (also
in the run's `summary.json`). Use a different `seed` for the validation set so it holds
different episodes. `policy=uniform_biased` gives the second kind of exploration data
(random actions); you can train on both. See [pipeline](pipeline.md#1-collect---exploration-data).

**10b. Pretrain the world model** (self-supervised, no simulator).

```bash
python -m wmnav.pretrain \
    data.train=[$WMNAV_DATA/collected/L3_safe_explore_A.pt] \
    data.valid=[$WMNAV_DATA/collected/L3_safe_explore_B.pt]
```

Several files: `data.train=[a.pt,b.pt]`. Writes
`$WMNAV_OUTPUTS/pretrain/ssl/<date_time>/checkpoints/{last,best_train,best_eval}.pt`.
`last.pt` corresponds to the released SSL files (they hold the last epoch);
`best_eval.pt` is the epoch with the lowest validation loss.

**10c. Fine-tune with reinforcement learning** (simulator; ~30-40 h, ~24 GB GPU).

```bash
python -m wmnav.finetune env.level=L3 run_name=my_L3 \
    init_from=$WMNAV_OUTPUTS/pretrain/ssl/<date_time>/checkpoints/last.pt
```

The world model is loaded from `init_from`, the actor-critic starts from scratch.
On a smaller GPU add `train.batch_sz=32`. The checkpoint to use afterwards is
`$WMNAV_OUTPUTS/finetune/my_L3/<date_time>/checkpoints/best_train_win_rate.pt`.
`init_from` also accepts the released SSL checkpoints, so you can skip 10a-10b.

**10d. Evaluate it** - as in step 9, with `checkpoint=` the file from 10c.

### Step 11. Analyse world models and trajectories

```bash
# open-loop prediction: reconstruct 1 s of context, imagine the rest; MSE and SSIM
python -m wmnav.imagine checkpoint=<world model or agent .pt> data=[<dataset .pt>]

# distance to the obstacles along the flown paths
python -m wmnav.evaluate checkpoint=<agent .pt> env.level=L3 record_trajectories=true
python -m wmnav.analysis.clearance --buffers WM3_L3=<evaluate run dir>/trajectories.pt --csv clearance.csv

# tables (per run, and model x level matrices) from any set of evaluation runs
python -m wmnav.analysis.tables $WMNAV_OUTPUTS/evaluate --out results
```

### Step 12. Train and evaluate the PPO baseline

```bash
python -m wmnav.train_ppo env.level=L3          # 15M frames, 512 envs, ~85 min; snapshot at 1M frames
python -m wmnav.eval_ppo experiment=ppo-L3 env.level=L3           # the 15M policy
python -m wmnav.eval_ppo experiment=ppo-L3-snapshot env.level=L3  # the 1M snapshot
```

The evaluation writes the same `summary.json` fields as step 9; tabulate the PPO runs
with `python -m wmnav.analysis.tables $WMNAV_OUTPUTS/eval_ppo --by experiment --out ppo_results`.

### Step 13. Reproduce the paper's experiments

`scripts/reproduce/` holds the paper's sweeps as shell scripts:

| script | what it runs |
|---|---|
| `collect_datasets.sh [L1 ...]` | exploration data for each level and both policies |
| `ssl_sweep.sh` | the SSL architecture grid (`TRAIN=... VALID=...`) |
| `finetune_all.sh [WM1 ...]` | fine-tune WM1-WM4 from the released SSL checkpoints |
| `cross_eval.sh [WM1 ...]` | every model on every level (`PANELS=5\|7\|10`) |
| `ppo_levels.sh [L1 ...]` | PPO on each level, then both evaluations |

They read `scripts/reproduce/_env.sh`, which points to the conda env and the
released checkpoints of the original machine. On another machine set
`CONDA_ENV_DIR`, `LEGACY_FT` and `LEGACY_SSL` (or edit that file) first:

```bash
CONDA_ENV_DIR=$CONDA_PREFIX LEGACY_FT=/data/wms_navigation/outputs/iros26/new_dataset \
    PANELS=7 ./scripts/reproduce/cross_eval.sh WM3
```

Every number of the paper's setup and where it comes from: [paper parameters](paper_params.md).

---

## Logging

Runs log to Weights & Biases in offline mode by default, and always to
`metrics.jsonl` in the run directory. Upload later with
`wandb sync <run dir>/wandb/offline-run-*` (after `wandb login`), log live with
`logging.wandb_mode=online`, or turn it off with `logging.wandb_mode=disabled`.

## Troubleshooting

| symptom | fix |
|---|---|
| `ImportError: PyTorch was imported before isaacgym` | import `wmnav.env` / `isaacgym` before `torch` in your own scripts; run the tests with `-t .` |
| `libpython3.8.so.1.0: cannot open shared object file` | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` |
| gymtorch fails to build / `ninja` not found | `pip install ninja`; when launching from another shell put the env's `bin/` first on `PATH` |
| a script hangs when it creates a second environment | Isaac Gym allows one simulator per process: run each stage as its own command |
| CUDA out of memory in `finetune` / `pretrain` | lower the batch size: `train.batch_sz=32` / `batch_sz=64` ([performance](performance.md)) |
| results differ slightly between identical runs | Isaac Gym's GPU physics is nondeterministic; see [known issues](known_issues.md) |
