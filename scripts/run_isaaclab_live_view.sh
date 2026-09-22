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

echo "[LunarSim] DISPLAY=$DISPLAY (session type: ${XDG_SESSION_TYPE:-unknown})"
if ! xhost +local:docker; then
  echo "[LunarSim] 'xhost +local:docker' failed -- the container will likely not be able to" \
       "open the display. Install 'xhost' (x11-xserver-utils / xorg-xhost package) and retry." >&2
fi

# A stray isaac_sim process/container from a previous run that never
# actually opened a window (killed with Ctrl-C, terminal closed, etc.)
# can sit holding the GPU/display, so a new launch looks "stuck" at high
# GPU util with no window ever appearing -- clear those first.
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

# The repeated "OmniHub ... Hub failed to launch: child exited with exit
# status: 1 without writing file ..." retries seen on this machine point
# at a stale/corrupt Hub state in the PERSISTENT mounted cache (it's the
# only thing carried over between runs) -- clear it so a bad leftover
# lock/config can't be the reason startup later stalls silently.
find "$CACHE_ROOT" -iname "*hub-root*" -delete 2>/dev/null || true

echo "[LunarSim] If this hangs again with no more log output after" \
     "'AppLauncher initialization complete', open a SECOND terminal and run:" \
     "  pgrep -af 'kit|isaac-sim' | head -20" \
     "then 'sudo strace -p <pid> -f -tt 2>&1 | tail -50' on whichever PID is the" \
     "actual simulation process (not this docker/python.sh wrapper) -- that will" \
     "show the exact syscall it's blocked on, which we need to diagnose this" \
     "further. (--pid=host below makes the real process visible from the host.)"

# XAUTHORITY forwarding is more reliable than xhost alone on some setups
# (notably GNOME/Wayland via XWayland) -- mount the real cookie file too.
XAUTH_FILE="${XAUTHORITY:-$HOME/.Xauthority}"
XAUTH_MOUNT=()
if [[ -f "$XAUTH_FILE" ]]; then
  XAUTH_MOUNT=(-e XAUTHORITY=/root/.Xauthority -v "$XAUTH_FILE":/root/.Xauthority:ro)
else
  echo "[LunarSim] warning: no Xauthority file found at $XAUTH_FILE -- if the window still" \
       "doesn't appear, run 'xauth list' on the host to check it exists." >&2
fi

docker run --rm \
  --gpus all \
  --network=host \
  --pid=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONUNBUFFERED=1 \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e __NV_PRIME_RENDER_OFFLOAD=1 \
  "${XAUTH_MOUNT[@]}" \
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
