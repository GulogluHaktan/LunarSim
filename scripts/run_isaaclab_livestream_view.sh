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
#   Download the "Isaac Sim WebRTC Streaming Client" from NVIDIA (search
#   "Isaac Sim WebRTC Streaming Client download" on NVIDIA's developer
#   site / Isaac Sim release page) and point it at this machine's IP,
#   signaling port 49100 (TCP; media itself flows over UDP 47998) -- NOT
#   port 8211, which was wrong (leftover from an older Isaac Sim version's
#   docs). Confirmed by reading the actual extension config on this image:
#   omni.kit.livestream.app's extension.toml sets signalPort=49100,
#   streamPort=47998, and `ss -tlnp` on the host showed 49100 genuinely
#   LISTENING while the app was running -- there is no browser/HTTP demo
#   page served by this extension, only the raw signaling socket the
#   streaming client app speaks to.
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

# REAL BUG FIXED: `hostname -I` is a GNU/inetutils extension -- on a
# machine whose `hostname` doesn't support it (confirmed: "invalid option
# -- 'I'", exit 64), `set -e` killed this WHOLE script silently with zero
# output, before any of the log lines below ever printed. `ip addr` is
# the portable way to get this; `|| true` guarantees it can never abort
# the script even if IP detection itself fails for some other reason.
HOST_IP=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1) || true
echo "[LunarSim] starting WebRTC livestream -- once you see 'scene ready', connect the"
echo "[LunarSim] Isaac Sim WebRTC Streaming Client to: ${HOST_IP:-<check with 'ip addr'>}, signaling port 49100"

docker run --rm \
  --gpus all \
  --network=host \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTHONUNBUFFERED=1 \
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
