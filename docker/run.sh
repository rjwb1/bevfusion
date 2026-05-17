#!/bin/bash
docker run -it \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace \
  -v "$(cd "$(dirname "$0")/.." && pwd)":/workspace\
  -v /home/rob/nuscenes:/workspace/data/nuscenes \
  --shm-size 16g \
  -w /workspace \
  bevfusion:torch2 \
  bash -c "FORCE_CUDA=1 python setup.py build_ext --inplace && torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox"

#torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox