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
  bevfusion \
  bash -c "FORCE_CUDA=1 python setup.py build_ext --inplace && bash"
