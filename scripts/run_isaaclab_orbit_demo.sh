#!/usr/bin/env bash
# Renders an orbit-camera flythrough of a real lunarsim tile via
# scripts/isaaclab_orbit_demo.py, then stitches the frames into an mp4 with
# ffmpeg (host-side). Needs the lunarsim-isaaclab:6.0.1 image.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
FRAMES_DIR="$PROJECT_ROOT/out/orbit_frames"
OUT_MP4="$PROJECT_ROOT/out/orbit_demo.mp4"

mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}
mkdir -p "$FRAMES_DIR"
rm -f "$FRAMES_DIR"/frame_*.png
chmod -R a+rwX "$PROJECT_ROOT/out" 2>/dev/null || true

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
  /workspace/LunarSim/scripts/isaaclab_orbit_demo.py \
  --headless --enable_cameras \
  --lunarsim-root /workspace/LunarSim \
  --out-dir /workspace/LunarSim/out/orbit_frames \
  "$@"

chmod -R a+rwX "$PROJECT_ROOT/out" 2>/dev/null || true

n_frames=$(find "$FRAMES_DIR" -maxdepth 1 -name 'frame_*.png' | wc -l)
if [[ "$n_frames" -eq 0 ]]; then
  echo "[LunarSim] no frames were rendered -- refusing to encode an empty video." >&2
  exit 1
fi

echo "[LunarSim] encoding $n_frames frames -> $OUT_MP4"
ffmpeg -y -framerate 24 -i "$FRAMES_DIR/frame_%04d.png" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p "$OUT_MP4"
echo "[LunarSim] wrote $OUT_MP4"
