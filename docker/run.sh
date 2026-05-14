#!/bin/bash
docker run -it \
  --gpus all \
  -v "$(cd "$(dirname "$0")/.." && pwd)":/workspace\
  -v /home/rob/nuscenes:/workspace/data/nuscenes \
  --shm-size 16g \
  -w /workspace \
  bevfusion \
  bash -c "FORCE_CUDA=1 python setup.py develop && bash"
