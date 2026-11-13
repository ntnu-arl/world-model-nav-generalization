#!/usr/bin/env bash
# Fine-tune WM1-WM4 from their SSL checkpoints, each on its own level, with the
# settings of the paper runs (WM1 used batch 128, the others 64). Needs a GPU
# with ~24 GB at batch 64 (see docs/performance.md); ~30-40 h per model.
#
#   ./scripts/reproduce/finetune_all.sh            # all four
#   ./scripts/reproduce/finetune_all.sh WM3
set -euo pipefail
source "$(dirname "$0")/_env.sh"
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(WM1 WM2 WM3 WM4)
for wm in "${MODELS[@]}"; do
  batch=64; [ "$wm" = WM1 ] && batch=128
  echo "=== $(date -Is) fine-tuning $wm on ${LEVEL_OF[$wm]} (batch $batch) ==="
  "$PYTHON" -m wmnav.finetune run_name="$wm" env.level="${LEVEL_OF[$wm]}" \
    init_from="${PRETRAINED[$wm]}" train.batch_sz="$batch"
done
