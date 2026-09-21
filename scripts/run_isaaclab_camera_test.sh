#!/usr/bin/env bash
# Runs scripts/isaaclab_test_camera.py against the lunarsim-isaaclab:6.0.1
# image (built by ./scripts/install_isaac_docker.sh). Needs the Isaac Lab
# image specifically -- bare isaacsim.core.api.World (what
# run_isaac_smoke_test.sh's other scripts use) never advances the render
# product's frame counter for camera capture in this headless setup; Isaac
# Lab's own SimulationContext + isaaclab.sensors.camera.Camera does (see
# lunarsim/adapters/isaac/README.md's "RESOLVED" section for why).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}
mkdir -p "$PROJECT_ROOT/out"
chmod -R a+rwX "$PROJECT_ROOT/out" 2>/dev/null || true

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[LunarSim] image '$IMAGE' not found -- run ./scripts/install_isaac_docker.sh first" \
       "(or set LUNARSIM_SKIP_ISAACLAB=0 if you skipped it before)." >&2
  exit 1
fi

docker run --rm \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "$PROJECT_ROOT":/workspace/LunarSim \
  -v "$CACHE_ROOT/cache/ov":/root/.cache/ov \
  -v "$CACHE_ROOT/cache/pip":/root/.cache/pip \
  -v "$CACHE_ROOT/cache/glcache":/root/.cache/nvidia \
  -v "$CACHE_ROOT/cache/kit":/root/.cache/kit \
  -v "$CACHE_ROOT/data":/root/.local/share/ov \
  -v "$CACHE_ROOT/documents":/root/Documents \
  --entrypoint /workspace/IsaacLab/_isaac_sim/python.sh \
  -w /workspace/IsaacLab \
  "$IMAGE" \
  /workspace/LunarSim/scripts/isaaclab_test_camera.py \
  --headless --enable_cameras \
  --lunarsim-root /workspace/LunarSim \
  --out /workspace/LunarSim/out/isaaclab_camera_validation.json \
  "$@"
