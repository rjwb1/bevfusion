#!/usr/bin/env bash
# Bring the existing .venv up to PyTorch 2.1 + the rest of the stack we
# proved works inside docker/Dockerfile.torch2. Run from the repo root:
#     bash scripts/setup_venv.sh
#
# Assumes:
#   - .venv/ already exists at the repo root (Python 3.10)
#   - System CUDA toolkit (nvcc) is available — used for building the C++
#     extensions. We use PyTorch's cu118 wheels for runtime; mismatched
#     minor CUDA versions between nvcc and PyTorch are fine.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ ! -x .venv/bin/python ]]; then
    echo "Expected .venv/ at $REPO. Create it first: python3.10 -m venv .venv"
    exit 1
fi

source .venv/bin/activate
echo "Using python: $(which python) ($(python --version))"

# 1. NumPy must be < 2 BEFORE torch, or pip will pull NumPy 2.x and
#    every C extension compiled against NumPy 1.x will abort at import.
pip install --upgrade pip 'setuptools<70.0.0' wheel
pip install 'numpy<2'

# 2. PyTorch 2.1 + cu121. The cu118 wheels won't compile extensions
#    against this machine's nvcc 12.8 — PyTorch enforces a CUDA
#    major-version match. cu121 still triggers a minor-mismatch warning
#    against 12.8 but builds fine. If a torch is already installed at the
#    same version string but with cu118, pip would skip the reinstall; we
#    detect that and force the swap.
CURRENT_CUDA=$(python -c "import torch, sys; sys.stdout.write(torch.version.cuda or '')" 2>/dev/null || true)
if [[ "$CURRENT_CUDA" != 12.* ]]; then
    pip install --force-reinstall --no-deps \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121
fi

# 3. mmcv 1.7.x + mmdet 2.28.x. The mmcv wheel must match torch's CUDA
#    runtime, so we use the cu121 index here too. Force-reinstall in case
#    an older cu118 build is cached at the same version.
pip install --force-reinstall --no-deps mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet==2.28.2

# 4. Apply the mmcv runtime patches the docker image needed. The wheel
#    pip serves may already have them (different build than openmmlab.com's
#    wheel index); in that case we just confirm the file is already safe
#    and move on instead of failing.
python - <<'PY'
import re, pathlib
import mmcv.parallel.distributed as ddp_mod
import mmcv.parallel._functions as fn_mod

# Patch 1: drop the _use_replicated_tensor_module ternary (gone in torch 2.x)
p = pathlib.Path(ddp_mod.__file__)
src = p.read_text()
if "_use_replicated_tensor_module" not in src:
    print(f"OK {p}: already safe (no _use_replicated_tensor_module reference)")
else:
    pattern = re.compile(
        r"self\._replicated_tensor_module"
        r"[\s\\]+if[\s\\]+self\._use_replicated_tensor_module"
        r"[\s\\]+else[\s\\]+self\.module",
    )
    new, n = pattern.subn("self.module", src)
    if n == 0:
        new, n = re.subn(r"self\._use_replicated_tensor_module", "False", src)
    if n == 0:
        raise SystemExit(f"distributed.py still references the missing attr but no known pattern matched: {p}")
    p.write_text(new)
    print(f"Patched {p} ({n})")

# Patch 2: wrap int device IDs in torch.device for _get_stream
p = pathlib.Path(fn_mod.__file__)
src = p.read_text()
if "torch.device('cuda', device)" in src or 'torch.device("cuda", device)' in src:
    print(f"OK {p}: already wraps int device ids")
else:
    new, n = re.subn(
        r"streams\s*=\s*\[_get_stream\(device\)\s+for\s+device\s+in\s+target_gpus\]",
        "streams = [_get_stream(torch.device('cuda', device) "
        "if isinstance(device, int) else device) for device in target_gpus]",
        src,
    )
    if n == 0:
        raise SystemExit(f"_functions.py needs patching but no known pattern matched: {p}")
    p.write_text(new)
    print(f"Patched {p} ({n})")
PY

# 5. Remaining runtime deps. shapely<2 because nuscenes-devkit iterates
#    over MultiLineString directly (shapely 1.x behaviour).
pip install \
    Pillow==10.2.0 \
    tqdm \
    torchpack \
    nuscenes-devkit \
    'shapely<2' \
    numba==0.58.1 \
    'yapf<0.40.0' \
    scipy \
    Cython

# 6. Build the custom CUDA extensions in-place. RTX 3090 is sm_86 so we
#    limit the arch list to avoid wasting time on other gencodes.
#    --no-build-isolation: setup.py imports torch, so the build process must
#    use our venv (which has torch) instead of pip's clean isolated env.
rm -rf build/ mmdet3d.egg-info
TORCH_CUDA_ARCH_LIST="8.6" FORCE_CUDA=1 \
    pip install -e . --no-build-isolation

# 7. Sanity check.
python - <<'PY'
import torch, mmcv, mmdet, mmdet3d.ops
print("torch:",   torch.__version__, "CUDA:", torch.version.cuda, "cuda_ok:", torch.cuda.is_available())
print("mmcv:",    mmcv.__version__)
print("mmdet:",   mmdet.__version__)
print("mmdet3d.ops: imported OK")
PY

echo
echo "venv is ready. To run inference on 1 GPU without MPI:"
echo "  source .venv/bin/activate"
echo "  bash scripts/run_local.sh test  configs/.../convfuser.yaml pretrained/bevfusion-det.pth"
echo "  bash scripts/run_local.sh train configs/.../convfuser.yaml --run-dir runs/local"
