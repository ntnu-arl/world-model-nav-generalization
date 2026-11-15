# Sourced by the reproduction scripts. Pins the interpreter of the Isaac Gym conda
# env and puts its bin/ first on PATH (Isaac Gym JIT-builds gymtorch with ninja).
# CONDA_ENV_DIR defaults to the active conda env.
CONDA_ENV_DIR=${CONDA_ENV_DIR:-${CONDA_PREFIX:?activate the wmnav conda env or set CONDA_ENV_DIR}}
PYTHON=${PYTHON:-$CONDA_ENV_DIR/bin/python}
export PATH="$CONDA_ENV_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

# The released checkpoints, by paper name. LEGACY_FT / LEGACY_SSL default to
# the legacy repos' output folders checked out next to this repository.
LEGACY_FT=${LEGACY_FT:-$REPO/../wms_navigation/outputs/iros26/new_dataset}
declare -A FINETUNED=(
  [WM1]=$LEGACY_FT/fine_tuned_env2/2026-03-25/19-58-15/data/2026-03-25-19-58-15/net/best_train_win_rate.pt
  [WM2]=$LEGACY_FT/fine_tuned_env3/2026-04-10/22-14-58/data/2026-04-10-22-14-58/net/best_train_win_rate.pt
  [WM3]=$LEGACY_FT/fine_tuned_env4/2026-04-15/11-19-29/data/2026-04-15-11-19-30/net/best_train_win_rate.pt
  [WM4]=$LEGACY_FT/fine_tuned_env6/2026-04-17/10-39-54/data/2026-04-17-10-39-55/net/best_train_win_rate.pt
)
LEGACY_SSL=${LEGACY_SSL:-$REPO/../ssl_wms/outputs/iros26}
declare -A PRETRAINED=(
  [WM1]=$LEGACY_SSL/env2/2026-03-12/03-45-45/net/agent_eval_best.pth
  [WM2]=$LEGACY_SSL/env3/2026-04-07/14-33-56/net/agent_eval_best.pth
  [WM3]=$LEGACY_SSL/env4/2026-04-12/06-43-25/net/agent_eval_best.pth
  [WM4]=$LEGACY_SSL/env6/2026-03-21/05-08-41/net/agent_eval_best.pth
)
declare -A LEVEL_OF=( [WM1]=L1 [WM2]=L2 [WM3]=L3 [WM4]=L4 )
