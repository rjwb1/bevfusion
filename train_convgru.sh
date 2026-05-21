#!/usr/bin/env bash
# Train the temporal ConvGRU fuser, initialised from the trained ConvFuser
# multi-decoder model (runs/det-seg-multidecoder/latest.pth).
#
# Step 1 builds a ConvGRU init checkpoint by remapping the ConvFuser fusion
# weights onto ConvGRUFuser.input_proj (GRU gates start random).
# Step 2 launches clip-based training (each batch = one temporal clip).
set -euo pipefail
cd "$(dirname "$0")"

SRC=runs/det-seg-multidecoder/latest.pth
INIT=runs/det-seg-multidecoder/convgru_init.pth
CONFIG=configs/nuscenes/det+seg-multi/resnet50-convgru-temporal.yaml
RUN_DIR=runs/convgru-temporal

if [[ ! -f "$INIT" ]]; then
  echo "==> Building ConvGRU init checkpoint from $SRC"
  python tools/convert_convfuser_to_convgru.py "$SRC" "$INIT"
fi

echo "==> Starting ConvGRU temporal training"
bash scripts/run_local.sh train "$CONFIG" \
  --run-dir "$RUN_DIR" \
  load_from "$INIT" \
  fp16.loss_scale=dynamic \
  checkpoint_config.by_epoch=False \
  checkpoint_config.interval=2000 \
  checkpoint_config.max_keep_ckpts=3
