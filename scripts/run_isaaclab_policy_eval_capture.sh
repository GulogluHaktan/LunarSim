#!/usr/bin/env bash
# Runs scripts/isaaclab_policy_eval_capture.py in the Isaac Sim docker image:
# loads a trained SAC checkpoint and actually flies it through a real
# landing attempt in the real production environment, recording camera +
# lidar + telemetry -- then stitches frames into mp4s and builds the 4-panel
# dashboard + standalone lidar video via scripts/compose_capture_video.py.
#
# Usage: ./scripts/run_isaaclab_policy_eval_capture.sh <checkpoint.zip> [out-subdir] [extra python args...]
#   checkpoint.zip: path relative to the project root (e.g.
#     out/sac_training_run/sac_lunar_lander_isaac_hover_only.zip)
#   out-subdir: where under out/eval_snapshots/ this run's files go
#     (default: the checkpoint's own basename + a timestamp)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"

CHECKPOINT="${1:?usage: $0 <checkpoint.zip> [out-subdir] [extra args...]}"
shift
CKPT_BASENAME="$(basename "$CHECKPOINT" .zip)"
OUT_SUBDIR="${1:-${CKPT_BASENAME}_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then shift; fi

CAPTURE_DIR="$PROJECT_ROOT/out/eval_snapshots/$OUT_SUBDIR"
mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}
mkdir -p "$CAPTURE_DIR"

fix_ownership() {
  docker run --rm -v "$PROJECT_ROOT/out":/out alpine chmod -R a+rwX /out 2>/dev/null || true
}
fix_ownership

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[LunarSim] image '$IMAGE' not found -- run ./scripts/install_isaac_docker.sh first." >&2
  exit 1
fi

# REAL GAP FOUND: the docker image's Isaac Sim python has no
# stable_baselines3 (needed to load the .zip checkpoint) -- same gap the
# LOCAL isaac-env venv had. Installed with --user into a volume that
# PERSISTS across `docker run --rm` invocations (unlike the container's own
# site-packages), so this only actually downloads/installs once, not on
# every eval run.
mkdir -p "$CACHE_ROOT/pylibs"
if ! docker run --rm -v "$CACHE_ROOT/pylibs":/root/.local --entrypoint /workspace/IsaacLab/_isaac_sim/python.sh "$IMAGE" \
     -c "import stable_baselines3" >/dev/null 2>&1; then
  echo "[LunarSim] installing stable-baselines3 into the persistent pylibs volume (one-time)..."
  docker run --rm \
    -v "$CACHE_ROOT/pylibs":/root/.local \
    -v "$CACHE_ROOT/cache/pip":/root/.cache/pip \
    --entrypoint /workspace/IsaacLab/_isaac_sim/python.sh \
    "$IMAGE" \
    -m pip install --user --no-warn-script-location stable-baselines3
fi

echo "[LunarSim] evaluating $CHECKPOINT -> $CAPTURE_DIR"
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
  -v "$CACHE_ROOT/pylibs":/root/.local \
  --entrypoint /workspace/IsaacLab/_isaac_sim/python.sh \
  -w /workspace/IsaacLab \
  "$IMAGE" \
  /workspace/LunarSim/scripts/isaaclab_policy_eval_capture.py \
  --headless --enable_cameras \
  --lunarsim-root /workspace/LunarSim \
  --checkpoint "/workspace/LunarSim/$CHECKPOINT" \
  --out-dir "/workspace/LunarSim/out/eval_snapshots/$OUT_SUBDIR" \
  "$@"

fix_ownership

n_chase=$(find "$CAPTURE_DIR/chase_frames" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l)
if [[ "$n_chase" -eq 0 ]]; then
  echo "[LunarSim] no frames were rendered -- skipping video encode." >&2
  exit 1
fi

echo "[LunarSim] encoding raw camera videos..."
ffmpeg -y -framerate 30 -i "$CAPTURE_DIR/chase_frames/frame_%04d.png" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "$CAPTURE_DIR/chase.mp4"
ffmpeg -y -framerate 30 -i "$CAPTURE_DIR/nav_frames/frame_%04d.png" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "$CAPTURE_DIR/nav.mp4"

echo "[LunarSim] building 4-panel dashboard + standalone lidar video..."
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/compose_capture_video.py" \
  --capture-dir "$CAPTURE_DIR" \
  --out "$CAPTURE_DIR/dashboard.mp4" \
  --lidar-out "$CAPTURE_DIR/lidar.mp4"

echo "[LunarSim] done -> $CAPTURE_DIR"
