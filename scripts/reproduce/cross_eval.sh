#!/usr/bin/env bash
# Every world model on every level: the 4x4 matrix of the paper.
#
#   PANELS=5 EPISODES=1000 ./scripts/reproduce/cross_eval.sh
#   PANELS=7 ./scripts/reproduce/cross_eval.sh WM3          # one model
#
# Results: $WMNAV_OUTPUTS/evaluate/<WM>_<level>_p<panels>/<time>/summary.json;
# tables with:  python -m wmnav.analysis.tables $WMNAV_OUTPUTS/evaluate --out cross_eval
set -euo pipefail
source "$(dirname "$0")/_env.sh"
PANELS=${PANELS:-5}
EPISODES=${EPISODES:-1000}
NUM_ENVS=${NUM_ENVS:-16}
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(WM1 WM2 WM3 WM4)

for wm in "${MODELS[@]}"; do
  for level in L1 L2 L3 L4; do
    echo "=== $(date -Is) $wm on $level, $PANELS panels ==="
    "$PYTHON" -m wmnav.evaluate name="$wm" checkpoint="${FINETUNED[$wm]}" \
      env.level="$level" env.num_panels="$PANELS" env.num_envs="$NUM_ENVS" episodes="$EPISODES" \
      logging.wandb_mode=disabled
  done
done
