#!/usr/bin/env bash
# Run BEVFusion entry points on a single local GPU without going through
# torchpack/mpirun. Sets the env vars torchpack.distributed.init() reads,
# so dist.init() works with world_size=1 over a no-op TCP rendezvous.
#
# Usage:
#   bash scripts/run_local.sh test  <config> <checkpoint> [extra args...]
#   bash scripts/run_local.sh train <config> [extra args...]
#   bash scripts/run_local.sh viz   <config> [extra args...]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# Pick a free local port for the (single-rank) rendezvous.
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

export MASTER_HOST="127.0.0.1:${PORT}"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_SIZE=1
export OMPI_COMM_WORLD_LOCAL_RANK=0

# RTX 3090 quirks: disable libfabric paths that torch's DataLoader fork
# breaks, and keep CUDA allocator from fragmenting too aggressively under
# spconv's many small allocations.
export RDMAV_FORK_SAFE=1
export FI_PROVIDER="^efa,verbs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="$1"; shift
case "$CMD" in
    test)
        CFG="$1"; CKPT="$2"; shift 2
        python tools/test.py "$CFG" "$CKPT" --eval bbox "$@"
        ;;
    train)
        CFG="$1"; shift
        python tools/train.py "$CFG" "$@"
        ;;
    viz|visualize)
        CFG="$1"; shift
        python tools/visualize.py "$CFG" "$@"
        ;;
    *)
        echo "Usage: $0 {test|train|viz} <config> [args...]"
        exit 1
        ;;
esac
