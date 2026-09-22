#!/usr/bin/env bash
# Runs the interactive lunarsim scene fully headless and streams it out over
# WebRTC instead of opening a local X11 window -- see isaaclab_live_view.py's
# module docstring for why (a real, unresolved GLX/window-surface hang was
# found when forwarding X11 into the container on this project's dev
# machine: ~4% GPU util, no window ever appeared, real PhysX/render work
# clearly wasn't happening). This needs no $DISPLAY at all and works over a
# pure SSH session.
#
# To view it once this prints "scene ready":
#   1. Preferred: download the "Isaac Sim WebRTC Streaming Client" from
#      NVIDIA (search "Isaac Sim WebRTC Streaming Client download" on
#      NVIDIA's developer site / Isaac Sim release page), point it at
#      this machine's IP (or 127.0.0.1 if running the client on the same
#      machine), default signaling port 8211.
#   2. Or try a browser at http://<this-machine-ip>:8211/streaming/webrtc-demo/
#      (the exact path can vary by Isaac Sim version -- if that 404s, check
#      the container's startup log for the actual URL it prints, or fall
#      back to the streaming client app above).
#
# Usage: ./scripts/run_isaaclab_livestream_view.sh [--mesh-lod 2] [--size-m 150] [--center-detail-radius-m 40]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[LunarSim] image '$IMAGE' not found -- run ./scripts/install_isaac_docker.sh first." >&2
  exit 1
fi

STALE=$(pgrep -f isaac_sim || true)
if [[ -n "$STALE" ]]; then
  echo "[LunarSim] killing stale isaac_sim process(es): $STALE"
  pkill -9 -f isaac_sim || true
  sleep 1
fi
STALE_CTR=$(docker ps -q --filter "ancestor=$IMAGE")
if [[ -n "$STALE_CTR" ]]; then
  echo "[LunarSim] removing stale container(s) from image $IMAGE: $STALE_CTR"
  docker rm -f $STALE_CTR || true
fi
rm -f /dev/shm/carb-* /dev/shm/omni-* 2>/dev/null || true
find "$CACHE_ROOT" -iname "*hub-root*" -delete 2>/dev/null || true

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "[LunarSim] starting WebRTC livestream -- once you see 'scene ready', connect to:"
echo "[LunarSim]   this machine's IP: ${HOST_IP:-<check with 'hostname -I'>}, port 8211"

docker run --rm \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
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
  --livestream 2 \
  "$@"
