#!/usr/bin/env bash
# Runs scripts/isaaclab_static_telemetry_capture.py in the Isaac Sim docker
# image: real production environment, real physics, vehicle held static (no
# RL/controller), two camera feeds + LiDAR + IMU telemetry logged -- then
# stitches both frame sequences into mp4s with ffmpeg (host-side).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
CAPTURE_DIR="$PROJECT_ROOT/out/static_capture"
NAV_MP4="$PROJECT_ROOT/out/static_capture_nav.mp4"
CHASE_MP4="$PROJECT_ROOT/out/static_capture_chase.mp4"

mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}
mkdir -p "$CAPTURE_DIR"

# REAL BUG FOUND: the container runs as root by default, so every file it
# writes under out/ is root-owned -- a host-side `chmod`/`rm` on those files
# is a no-op (permission denied to chmod/rm a file you don't own, silently
# swallowed by `|| true`), which left root-owned frames behind and made the
# NEXT run's plain `rm -rf` below fail outright (set -e killed the script
# before docker ever ran). Fix ownership from INSIDE a throwaway root
# container instead, both before (defensive, in case a prior run left
# root-owned files) and after the real capture run.
fix_ownership() {
  docker run --rm -v "$PROJECT_ROOT/out":/out alpine chmod -R a+rwX /out 2>/dev/null || true
}
fix_ownership
rm -rf "$CAPTURE_DIR"/chase_frames "$CAPTURE_DIR"/nav_frames "$CAPTURE_DIR"/lidar_scans

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[LunarSim] image '$IMAGE' not found -- run ./scripts/install_isaac_docker.sh first." >&2
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
  /workspace/LunarSim/scripts/isaaclab_static_telemetry_capture.py \
  --headless --enable_cameras \
  --lunarsim-root /workspace/LunarSim \
  --out-dir /workspace/LunarSim/out/static_capture \
  "$@"

fix_ownership

n_nav=$(find "$CAPTURE_DIR/nav_frames" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l)
n_chase=$(find "$CAPTURE_DIR/chase_frames" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l)
if [[ "$n_nav" -eq 0 || "$n_chase" -eq 0 ]]; then
  echo "[LunarSim] no frames were rendered -- refusing to encode empty video(s)." >&2
  exit 1
fi

echo "[LunarSim] encoding $n_nav nav frames -> $NAV_MP4"
ffmpeg -y -framerate 30 -i "$CAPTURE_DIR/nav_frames/frame_%04d.png" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p "$NAV_MP4"

echo "[LunarSim] encoding $n_chase chase frames -> $CHASE_MP4"
ffmpeg -y -framerate 30 -i "$CAPTURE_DIR/chase_frames/frame_%04d.png" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p "$CHASE_MP4"

echo "[LunarSim] telemetry.csv + lidar_scans/ under $CAPTURE_DIR"
echo "[LunarSim] wrote $NAV_MP4 and $CHASE_MP4"
