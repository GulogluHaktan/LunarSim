#!/usr/bin/env bash
# Runs scripts/isaaclab_multi_env_speed_test.py against the lunarsim-isaaclab:6.0.1
# image -- once per env count (a fresh process each time; tearing down and
# rebuilding an InteractiveScene mid-process was unreliable) -- and reports
# real PhysX step throughput at each. Uses Isaac Lab's InteractiveScene
# env-cloning with ONE shared real terrain (TerrainImporterCfg terrain_type="usd"),
# not a unique mesh per env (see the script's docstring for why).
#
# Usage: ./scripts/run_isaaclab_speed_test.sh [env_count ...]
#   defaults to 16 64 256
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

ENV_COUNTS=("$@")
if [[ ${#ENV_COUNTS[@]} -eq 0 ]]; then
  ENV_COUNTS=(16 64 256)
fi

for n in "${ENV_COUNTS[@]}"; do
  echo "[LunarSim] running num_envs=$n ..."
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
    /workspace/LunarSim/scripts/isaaclab_multi_env_speed_test.py \
    --headless \
    --lunarsim-root /workspace/LunarSim \
    --out /workspace/LunarSim/out/isaaclab_multi_env_speed.jsonl \
    --num-envs "$n"
  chmod -R a+rwX "$PROJECT_ROOT/out" 2>/dev/null || true
done

echo ""
echo "=== results ==="
cat "$PROJECT_ROOT/out/isaaclab_multi_env_speed.jsonl"
