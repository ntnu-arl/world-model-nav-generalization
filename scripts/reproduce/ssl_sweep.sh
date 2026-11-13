#!/usr/bin/env bash
# The SSL architecture grid of the paper (ssl_wms sweep_cfg_*.yaml) as a Hydra
# multirun: sequence length x GRU size x GRU input size x categorical classes, on
# one level's data. The paper then compared the runs by validation loss and
# re-ran the chosen configuration with seeds 20, 40, 60.
#
#   TRAIN=/path/train_L3.pt VALID=/path/eval_L3.pt ./scripts/reproduce/ssl_sweep.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
: "${TRAIN:?set TRAIN to the training .pt file(s), comma-separated}"
: "${VALID:?set VALID to the validation .pt file(s), comma-separated}"
"$PYTHON" -m wmnav.pretrain -m "data.train=[$TRAIN]" "data.valid=[$VALID]" \
  batch_length=16,32,64 model.latent.recurrent=256,512,1024 model.latent.hidden=256,512,1024 \
  model.latent.classes=${DISCRETE:-32} seed=${SEED:-20} \
  hydra.sweep.dir='${paths.output_root}/pretrain/sweep/${now:%Y-%m-%d_%H-%M-%S}'
