bash scripts/run_local.sh train \
  configs/nuscenes/det+seg-multi/resnet50-convfuser-ag20m-sdg.yaml \
  --run-dir runs/ag20msdg \
  load_from runs/ag20m/latest.pth