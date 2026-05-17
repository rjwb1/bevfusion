# PyTorch 2 Upgrade

This document describes the PyTorch 2 compatibility changes made to this repo,
how to actually run the upgrade, and the things that have **not** been
verified end-to-end.

## What was changed

### C++ / CUDA extensions
The deprecated `THC/THC.h` header and `THCState *state` were removed from
every custom extension. `tensor.data<T>()` was replaced with
`tensor.data_ptr<T>()` where it still appeared.

Files touched:

- `mmdet3d/ops/ball_query/src/ball_query.cpp`
- `mmdet3d/ops/knn/src/knn.cpp`
- `mmdet3d/ops/interpolate/src/interpolate.cpp`
- `mmdet3d/ops/group_points/src/group_points.cpp`
- `mmdet3d/ops/gather_points/src/gather_points.cpp`
- `mmdet3d/ops/furthest_point_sample/src/furthest_point_sample.cpp`
- `mmdet3d/ops/spconv/include/torch_utils.h` (`tensor.type().scalarType()` → `tensor.scalar_type()`)

### Python source
`torch.jit._unwrap_optional`, a private API that no longer exists in
PyTorch 2, was replaced with plain `assert ... is not None` guards in
`mmdet3d/models/utils/transformer.py`.

### Build system
`setup.py`:
- Added `sm_89` (Ada / RTX 40xx) and `sm_90` (Hopper / H100) gencodes.
- Honors `TORCH_CUDA_ARCH_LIST` if set so users can compile for one arch
  during local iteration.
- Bumped the spconv extension from `-std=c++14` to `-std=c++17`
  (PyTorch 2 requires C++17).
- Updated Python version classifiers to 3.8 / 3.9 / 3.10 and set
  `python_requires=">=3.8"`.

### Environment
- `docker/Dockerfile.torch2`: new image based on CUDA 11.8, Python 3.10,
  PyTorch 2.1.2, mmcv-full 1.7.2, mmdet 2.28.2. The original
  `docker/Dockerfile` is untouched as a fallback.
- `requirements.txt`: rewritten to install the PyTorch 2 stack.

## How to actually build and test

The local virtualenv at `.venv` is still on PyTorch 1.13 with the
extensions compiled for Python 3.8. Do **not** try to reuse the existing
`build/` directory — it contains stale artifacts.

```bash
# 1. Build the new image
docker build -f docker/Dockerfile.torch2 -t bevfusion:torch2 .

# 2. Inside the container
cd /workspace            # or wherever you mount the repo
rm -rf build/ *.egg-info  # wipe stale build outputs
pip install -e .          # rebuild all CUDA extensions against torch 2.1

# 3. Smoke test
python -c "import torch; print(torch.__version__); import mmdet3d.ops"

# 4. Tiny training sanity check
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det+seg/resnet50-convfuser.yaml \
    --run-dir runs/torch2-smoke
```

## What is NOT verified

I made the source-level changes needed to compile and import the
extensions against PyTorch 2.1, but I did **not**:

- Actually compile the extensions on PyTorch 2 (the local venv is on
  1.13 and rebuilding there would break the current debugging setup).
- Run a full training epoch end-to-end on PyTorch 2.
- Confirm that the original published checkpoints still load cleanly.
- Confirm numerical parity (loss curves, mAP) against the PyTorch 1.10
  baseline.

Plan to budget time for the items below — none of them is unusual for a
major-version upgrade, but they will surface bugs that this static
pass cannot catch.

## Known risk areas to watch

1. **mmcv 1.4 → 1.7**: `Fp16OptimizerHook` and runner classes still
   exist but some hook signatures changed. If the runner errors on
   startup, check `mmdet3d/apis/train.py` against mmcv 1.7's
   `OptimizerHook.__init__`.
2. **mmdet 2.20 → 2.28**: `BBOX_ASSIGNERS` registry path is unchanged
   but `multi_apply` and a few sampler internals were refactored. The
   `HungarianAssigner3D` in this repo overrides the default so it
   should be unaffected.
3. **spconv**: this repo vendors its own copy of the old `spconv` v1
   code (not the pip-installable `spconv 2.x`). The CUDA kernels there
   are pre-PyTorch-1.5 era. They are the most likely thing to fail to
   build on a modern CUDA toolkit. If `sparse_conv_ext` fails to
   compile, the alternatives are (a) patch the kernels, or (b) migrate
   to `spconv-cu118==2.3.6` which is a non-trivial code change.
4. **`torch.cuda.amp.custom_fwd` / `custom_bwd`** in
   `mmdet3d/ops/spconv/functional.py` are deprecated in 2.x but still
   present in 2.1. Will need to switch to `torch.amp.custom_fwd(device_type="cuda", ...)`
   when upgrading past 2.4.
5. **CUDA 11.8 vs 12.x**: the new Dockerfile uses CUDA 11.8 because it
   has the broadest PyTorch wheel coverage. If you need CUDA 12, switch
   the base image and the wheel index URL to `cu121` and bump
   `torch==2.2.x`.

## If you need to roll back

The original `docker/Dockerfile` (PyTorch 1.10) is still in place, and
the C++ edits are backwards-compatible — modern PyTorch removed
`THCState` and `.data<>` but never relied on them being present, so
files that no longer include `THC/THC.h` still compile on 1.10+.
