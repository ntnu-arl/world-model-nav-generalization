# world-model-nav-generalization
This repository contains the source code for the paper: [_**Generalization of World Models under Environmental Variability for Vision-based Quadrotor Navigation**_](https://arxiv.org/abs/2606.05015).

A video demonstrating the work is available [here](https://youtu.be/830cnuQ555o?si=jKiP89B6B7eEsgpN).

[<img width="1280" height="720" alt="image_with_play_button" src="https://github.com/user-attachments/assets/af96bbdd-6547-4599-bea3-ce0fe00c9128" />](https://youtu.be/830cnuQ555o?si=jKiP89B6B7eEsgpN)

World-model based drone navigation in [Aerial Gym](https://github.com/ntnu-arl/aerial_gym_simulator):
the code behind the IROS26 experiments, merged from `wms_navigation` (data collection,
RL fine-tuning, PPO baseline, evaluation) and `ssl_wms` (self-supervised world-model
pretraining) into one package, with one implementation of every component.

A quadrotor with a forward depth camera flies down a walled corridor (15 x 6 x 7 m)
past randomly placed panels to a target. A DreamerV3 world model is pretrained on
exploration data, then fine-tuned together with an actor-critic that learns in
imagination; a PPO agent with the same encoder size and reward is the model-free
baseline.

```
 collect ──► pretrain (SSL) ──► finetune (Dreamer) ──► evaluate / imagine / analysis
 exploration   world model        world model +          win/crash/timeout, cross-eval,
 episodes      on recorded data   actor-critic in env    open-loop prediction, clearance

 train_ppo ───────────────────────────────────────────► eval_ppo          (baseline)
```

## Installation

wmnav needs Python 3.8 (the only version Isaac Gym preview 4 supports) and an NVIDIA GPU.

```bash
conda create -n wmnav python=3.8 && conda activate wmnav

# 1. not on PyPI - install from source
pip install -e /path/to/isaacgym/python          # Isaac Gym preview 4 (NVIDIA)
pip install -e /path/to/aerial_gym_dev           # ntnu-arl/aerial_gym_dev, branch luca_dev @ dcd757dd

# 2. everything else, pinned to the versions of the paper runs
pip install -r requirements.txt
pip install -e .                                 # optional: makes `wmnav` importable from anywhere
```

`requirements.txt` includes the PPO baseline (sample-factory) and the analysis tools;
with `pip install -e .` they are extras: `pip install -e ".[ppo,analysis]"` or `".[all]"`.
A full step-by-step guide (Isaac Gym, aerial_gym and its source dependencies, then
every stage of the pipeline): [docs/getting_started.md](docs/getting_started.md).
Details, GPU memory and common pitfalls: [docs/installation.md](docs/installation.md).

Check the install with the test suite (about a minute, no simulator needed):

```bash
python -m unittest discover -s tests -t .   # -t . : tests/__init__.py imports Isaac Gym before torch
```

## Quick start

```bash
export WMNAV_OUTPUTS=/path/to/outputs   # where runs go (default ./outputs)

# evaluate a released model: WM3 on level L3 with 7 panels, 1000 episodes
python -m wmnav.evaluate name=WM3 env.level=L3 env.num_panels=7 \
    checkpoint=/path/to/fine_tuned_env4/.../best_train_win_rate.pt

# the full pipeline on one level
python -m wmnav.collect   env.level=L3 policy=safe_explore episodes=10000
python -m wmnav.pretrain  data.train=[/path/L3_train.pt] data.valid=[/path/L3_valid.pt]
python -m wmnav.finetune  env.level=L3 init_from=$WMNAV_OUTPUTS/pretrain/ssl/<run>/checkpoints/last.pt
python -m wmnav.evaluate  env.level=L3 checkpoint=$WMNAV_OUTPUTS/finetune/L3/<run>/checkpoints/best_train_win_rate.pt

# PPO baseline (1M snapshot + 15M) and its evaluation
python -m wmnav.train_ppo env.level=L3
python -m wmnav.eval_ppo  experiment=ppo-L3 env.level=L3

# tables from any set of evaluation runs
python -m wmnav.analysis.tables $WMNAV_OUTPUTS/evaluate --out results
```

Every entry point is a Hydra app: `configs/<entry>.yaml`, overridable on the command
line; each run directory gets the resolved config, the git commit, `git diff` and
untracked files, `metrics.jsonl` and `summary.json`. Paper experiments are scripted
in `scripts/reproduce/`.

## Repository layout

| path | contents |
|---|---|
| `wmnav/dreamer/` | world model (encoder, RSSM, heads), actor-critic, agent, checkpoint I/O |
| `wmnav/data/` | replay buffer / dataset format, exploration policies, rollout loop |
| `wmnav/env/` | navigation task, robot / scene / simulator configs, levels L1-L4, factory |
| `wmnav/ssl/` | SSL pretraining loop |
| `wmnav/ppo/` | PPO encoder and sample-factory glue |
| `wmnav/analysis/` | result tables, open-loop prediction metrics, clearance |
| `wmnav/{collect,pretrain,finetune,evaluate,imagine,train_ppo,eval_ppo}.py` | entry points |
| `configs/` | Hydra configs (`model/`, `env/`, `ppo/`, `perf/`, `paths/`, `logging/`) |
| `scripts/reproduce/` | the paper's experiment sweeps |
| `tests/` | unit tests and equivalence tests against the legacy code |
| `docs/` | documentation (below) |

## Documentation

| doc | |
|---|---|
| [getting started](docs/getting_started.md) | **step-by-step: install, check, evaluate, train, reproduce** |
| [installation](docs/installation.md) | environment, dependencies, external drive |
| [pipeline](docs/pipeline.md) | each stage: what it does, inputs, outputs, file formats |
| [environment](docs/environment.md) | task, observations, actions, reward, levels L1-L4 |
| [model](docs/model.md) | world model, actor-critic, losses, training schedule |
| [configuration](docs/configuration.md) | config groups and the settings that matter |
| [checkpoints](docs/checkpoints.md) | formats, loading the released models |
| [evaluation](docs/evaluation.md) | evaluation, cross-evaluation, PPO, tables, metrics |
| [performance](docs/performance.md) | measured speed-ups, memory, GPU settings |
| [testing](docs/testing.md) | test suite and what the equivalence tests prove |
| [migration](docs/migration.md) | mapping from `wms_navigation` / `ssl_wms`, what changed |
| [known issues](docs/known_issues.md) | behaviours kept for reproducibility, and bugs found |
| [paper parameters](docs/paper_params.md) | every number of the paper's setup, with its source |

## Relationship to the paper runs

Default configurations are the as-run settings of the IROS26 experiments, and the
code computes what the legacy code computed: the test suite checks the port against
the legacy repositories (identical initialisation, losses, updates, replay batches,
environment rollouts) and every released checkpoint loads strictly. Evaluating the
released WM3 on L3 with the paper's evaluation settings gives the same 187/200 wins
as the legacy evaluator. Deviations that are deliberate are listed in
[migration](docs/migration.md); quirks that were kept because trained models depend
on them are in [known issues](docs/known_issues.md).

## Method and third-party code

The world model and actor-critic follow DreamerV3:

> Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap.
> *Mastering Diverse Domains through World Models.* arXiv:2301.04104, 2023.

`wmnav/dreamer/` is an implementation of that method written for this project.
The simulator configs in `wmnav/env/configs/` and the methods in
`wmnav/env/patches.py` are derived from
[Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) (BSD 3-Clause);
its license is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citing

If you use or reference this work in your research, please cite the following paper:

L. Zanatta, G. Malczyk and K. Alexis, "Generalization of World Models under Environmental Variability for Vision-based Quadrotor Navigation"

```bibtex
@article{zanatta2026generalization,
  title={Generalization of World Models under Environmental Variability for Vision-based Quadrotor Navigation},
  author={Zanatta, Luca and Malczyk, Grzegorz and Alexis, Kostas},
  journal={arXiv preprint arXiv:2606.05015},
  year={2026}
}
```

## Contact
For inquiries, feel free to reach out to the authors:
- **Luca Zanatta**

  [Email](mailto:luca.zanatta@ntnu.no) | [GitHub](https://github.com/LucaZanatta) | [LinkedIn](https://www.linkedin.com/in/luca-zanatta-54317a200/)
  
- **Grzegorz Malczyk**
  
  [Email](mailto:grzegorz.malczyk@ntnu.no) | [GitHub](https://github.com/grzemal) | [LinkedIn](https://www.linkedin.com/in/grzegorz-malczyk/) | [X (formerly Twitter)](https://twitter.com/grzemalige)

- **Kostas Alexis**

  [Email](mailto:konstantinos.alexis@ntnu.no) | [GitHub](https://github.com/kostas-alexis) | [LinkedIn](https://www.linkedin.com/in/kostas-alexis-67713918/) | [X (formerly Twitter)](https://twitter.com/arlteam)

This research was conducted at the [Autonomous Robots Lab](https://www.autonomousrobotslab.com/), [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no). 

For more information, visit our website.

## Acknowledgements
This material was supported by the Horizon Europe Grant Agreement No. 101120732 and the Research Council of Norway under Award NO-338694.
