#!/usr/bin/env bash
# Runs scripts/train_sac_isaac.py inside the Isaac Sim docker image, the
# same image/cache/volume setup run_isaaclab_policy_eval_capture.sh uses --
# for when the local bare isaac-env venv training path isn't working.
#
# Usage: ./scripts/run_train_sac_isaac_docker.sh [extra train_sac_isaac.py args...]
#   e.g. ./scripts/run_train_sac_isaac_docker.sh --steps-per-stage 50000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"

mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents,pylibs}
mkdir -p "$PROJECT_ROOT/out"

fix_ownership() {
  docker run --rm -v "$PROJECT_ROOT/out":/out alpine chmod -R a+rwX /out 2>/dev/null || true
}
fix_ownership

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[LunarSim] image '$IMAGE' not found -- run ./scripts/install_isaac_docker.sh first." >&2
  exit 1
fi

# same stable_baselines3 gap as the eval-capture docker wrapper -- installed
# into a persistent volume so it's a one-time cost, not per-run.
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

echo "[LunarSim] training (docker) -> out/sac_training_run/ args: $*"
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
  /workspace/LunarSim/scripts/train_sac_isaac.py \
  --headless \
  --lunarsim-root /workspace/LunarSim \
  --out-dir /workspace/LunarSim/out \
  "$@"

fix_ownership

echo "[LunarSim] done"
