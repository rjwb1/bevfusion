bash scripts/run_local.sh train \
  configs/nuscenes/det+seg-multi/resnet50-convfuser-ag20m.yaml \
  --run-dir runs/ag20m \
  resume_from runs/ag20m/latest.pth