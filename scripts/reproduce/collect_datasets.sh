#!/usr/bin/env bash
# Exploration datasets for SSL pretraining, for each level and both policies
# (the paper's *_low_freq / *_high_freq files). The original training sets are
# lost; this regenerates data of the same kind, not the same files.
#
#   EPISODES=10000 ./scripts/reproduce/collect_datasets.sh L3
set -euo pipefail
source "$(dirname "$0")/_env.sh"
EPISODES=${EPISODES:-10000}
LEVELS=("$@"); [ ${#LEVELS[@]} -eq 0 ] && LEVELS=(L1 L2 L3 L4)
for level in "${LEVELS[@]}"; do
  for policy in safe_explore uniform_biased; do
    echo "=== $(date -Is) collecting $level / $policy ==="
    "$PYTHON" -m wmnav.collect env.level="$level" policy="$policy" episodes="$EPISODES"
  done
done
