#!/usr/bin/env bash
# Visualize the ag20m model: per-camera boxes, LiDAR BEV, and BEV map seg.
#
# Usage:
#   bash viz_ag_model.sh                       # GT on val split  -> viz/ag20m
#   bash viz_ag_model.sh <checkpoint>          # predictions on val split
#   bash viz_ag_model.sh <checkpoint> [args]   # extra args forwarded to visualize.py
#
# Examples:
#   bash viz_ag_model.sh                                  # sanity-check GT geometry
#   bash viz_ag_model.sh runs/ag20m/latest.pth           # model predictions
#   bash viz_ag_model.sh runs/ag20m/latest.pth --split train --bbox-score 0.2
#
# Output lands under viz/ag20m/{camera-*,lidar,map}/<timestamp>-<token>.png

set -euo pipefail

CFG=configs/nuscenes/det+seg-multi/resnet50-convfuser-ag20m.yaml
OUT_DIR=viz/ag20m

if [[ $# -ge 1 && -f "$1" ]]; then
  CKPT="$1"; shift
  bash scripts/run_local.sh viz "$CFG" \
    --mode pred --checkpoint "$CKPT" --out-dir "$OUT_DIR" "$@"
else
  bash scripts/run_local.sh viz "$CFG" \
    --mode gt --out-dir "$OUT_DIR" "$@"
fi
