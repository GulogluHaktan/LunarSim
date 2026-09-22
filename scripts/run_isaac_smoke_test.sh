#!/usr/bin/env bash
# Runs scripts/isaac_smoke_test.py (or any script passed as $1) inside a real
# Isaac Sim container, against the lunar-rocket-isaaclab:6.0.1 image already
# built on this machine (falls back to the upstream
# nvcr.io/nvidia/isaac-sim:6.0.1 image if that one isn't present). No Isaac
# Lab checkout is needed for USD-authoring/physics smoke tests -- plain Isaac
# Sim's python.sh is enough. --network=host --ipc=host + the ulimits below
# are required (verified empirically on this machine) for the CUDA/render
# pipeline to come up cleanly in the container.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${1:-scripts/isaac_smoke_test.py}"
shift || true

IMAGE="${LUNARSIM_ISAAC_IMAGE:-lunarsim-isaacsim:6.0.1}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[LunarSim] image '$IMAGE' not found -- run ./scripts/install_isaac_docker.sh first." >&2
  echo "[LunarSim] falling back to lunar-rocket-isaaclab:6.0.1 / nvcr.io/nvidia/isaac-sim:6.0.1 if present..." >&2
  if docker image inspect lunar-rocket-isaaclab:6.0.1 >/dev/null 2>&1; then
    IMAGE="lunar-rocket-isaaclab:6.0.1"
  else
    IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1"
  fi
fi

CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}
mkdir -p "$PROJECT_ROOT/out"

echo "[LunarSim] using image: $IMAGE"

docker run --rm \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONUNBUFFERED=1 \
  -v "$PROJECT_ROOT":/workspace/LunarSim \
  -v "$CACHE_ROOT/cache/ov":/root/.cache/ov \
  -v "$CACHE_ROOT/cache/pip":/root/.cache/pip \
  -v "$CACHE_ROOT/cache/glcache":/root/.cache/nvidia \
  -v "$CACHE_ROOT/cache/kit":/root/.cache/kit \
  -v "$CACHE_ROOT/data":/root/.local/share/ov \
  -v "$CACHE_ROOT/documents":/root/Documents \
  --entrypoint /isaac-sim/python.sh \
  "$IMAGE" \
  "/workspace/LunarSim/$SCRIPT" \
  --headless \
  --lunarsim-root /workspace/LunarSim \
  "$@"
