#!/usr/bin/env bash
# Opens an interactive Isaac Sim window (X11) with the real lunarsim scene
# loaded (real terrain, regolith material + micro-bump texture, fixed sun,
# no ambient, path tracing by default) so you can fly/orbit around it
# yourself and judge realism live, instead of watching a pre-baked video.
#
# Needs a real display on THIS machine (not a pure SSH-without-X session --
# use scripts/run_isaaclab_orbit_demo.sh for a headless video instead).
#
# Usage: ./scripts/run_isaaclab_live_view.sh [--no-path-tracing] [--spp 256] [--mesh-lod 2] [--size-m 150]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[LunarSim] image '$IMAGE' not found -- run ./scripts/install_isaac_docker.sh first." >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[LunarSim] No \$DISPLAY set -- this needs a real display on this machine" \
       "(not a headless/SSH-without-X session). Use run_isaaclab_orbit_demo.sh instead." >&2
  exit 1
fi

xhost +local:docker >/dev/null 2>&1 || echo "[LunarSim] warning: 'xhost +local:docker' failed -- the container may not be able to open the display."

docker run --rm \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
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
  /workspace/LunarSim/scripts/isaaclab_live_view.py \
  --lunarsim-root /workspace/LunarSim \
  "$@"
