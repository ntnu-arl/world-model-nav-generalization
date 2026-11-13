#!/usr/bin/env bash
# PPO baseline on each level: snapshot at 1M frames (the WMs' budget), train to 15M,
# then evaluate both deterministically. ~85 min of training per level on the paper
# setup (512 envs).
#
#   ./scripts/reproduce/ppo_levels.sh            # L1-L4
#   SEED=7 ./scripts/reproduce/ppo_levels.sh L2  # an independent seed
set -euo pipefail
source "$(dirname "$0")/_env.sh"
LEVELS=("$@"); [ ${#LEVELS[@]} -eq 0 ] && LEVELS=(L1 L2 L3 L4)
SEED=${SEED:-}
EPISODES=${EPISODES:-1000}
for level in "${LEVELS[@]}"; do
  exp="ppo-$level${SEED:+-s$SEED}"
  echo "=== $(date -Is) PPO $exp ==="
  "$PYTHON" -m wmnav.train_ppo env.level="$level" experiment="$exp" ${SEED:+seed=$SEED}
  for e in "$exp-snapshot" "$exp"; do
    "$PYTHON" -m wmnav.eval_ppo experiment="$e" env.level="$level" episodes="$EPISODES"
  done
done
