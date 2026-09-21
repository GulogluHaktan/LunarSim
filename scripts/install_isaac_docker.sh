#!/usr/bin/env bash
# One-shot installer: builds LunarSim's own Isaac Sim Docker image from
# scratch (base nvcr.io/nvidia/isaac-sim:6.0.1 + rasterio/skyfield/pyyaml),
# and checks every prerequisite along the way. Safe to re-run (each step
# checks whether it's already done).
#
# Usage:
#   ./scripts/install_isaac_docker.sh
#   LUNARSIM_ISAAC_IMAGE=my-tag:1 ./scripts/install_isaac_docker.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUNARSIM_ISAAC_IMAGE:-lunarsim-isaacsim:6.0.1}"
BASE_IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"

log() { echo "[LunarSim] $*"; }
die() { echo "[LunarSim] ERROR: $*" >&2; exit 1; }

log "1/5 checking docker..."
command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker first: https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 || die "docker daemon not reachable (permission? try 'sudo usermod -aG docker \$USER' then re-login, or run with sudo)."

log "2/5 checking NVIDIA driver + GPU..."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- install the NVIDIA proprietary driver first."
nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed to run -- driver installed but not working (reboot? check 'nvidia-smi' output directly)."
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
log "    GPU: $GPU_NAME (${GPU_VRAM_MB} MB VRAM)"

log "3/5 checking NVIDIA Container Toolkit (GPU access inside Docker)..."
# A small, already-common CUDA base image -- avoids the big Isaac Sim image's
# EULA-gate (it exits non-zero without ACCEPT_EULA=Y, which would look like a
# toolkit failure here) and keeps this check fast.
TOOLKIT_TEST_IMAGE="nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04"
docker image inspect "$TOOLKIT_TEST_IMAGE" >/dev/null 2>&1 || docker pull "$TOOLKIT_TEST_IMAGE" >/dev/null
if ! docker run --rm --gpus all "$TOOLKIT_TEST_IMAGE" nvidia-smi >/dev/null 2>&1; then
  log "    GPU not visible in a container yet -- attempting to install the NVIDIA Container Toolkit."
  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "    nvidia-ctk already present; configuring the docker runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  else
    die "NVIDIA Container Toolkit not installed and this script won't install it unattended. \
Follow https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html, \
then re-run this script. (On Arch/CachyOS: 'sudo pacman -S nvidia-container-toolkit', \
then 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'.)"
  fi
  docker run --rm --gpus all "$TOOLKIT_TEST_IMAGE" nvidia-smi >/dev/null 2>&1 || die "GPU still not visible in a container after configuring the toolkit -- check 'nvidia-ctk runtime configure' output and docker daemon logs."
fi
log "    GPU visible inside containers: OK"

log "4/5 pulling base image ($BASE_IMAGE) if needed (this is ~10 GB, first pull can take a while)..."
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  docker pull "$BASE_IMAGE"
fi

log "5/5 building $IMAGE from docker/Dockerfile.isaacsim..."
docker build -f "$PROJECT_ROOT/docker/Dockerfile.isaacsim" -t "$IMAGE" "$PROJECT_ROOT/docker"

mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents}

log "Done. Image: $IMAGE"
log "Try it: LUNARSIM_ISAAC_IMAGE=$IMAGE ./scripts/run_isaac_smoke_test.sh scripts/isaac_smoke_test.py"

if [[ "${LUNARSIM_SKIP_ISAACLAB:-0}" != "1" ]]; then
  ISAACLAB_IMAGE="${LUNARSIM_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
  log "optional: building $ISAACLAB_IMAGE (adds a real Isaac Lab checkout, needed for" \
      "isaaclab.sim.SimulationContext-backed camera rendering -- this pulls torch+cu128" \
      "and can take 10-20+ minutes on the first build). Skip with LUNARSIM_SKIP_ISAACLAB=1."
  docker build -f "$PROJECT_ROOT/docker/Dockerfile.isaaclab" -t "$ISAACLAB_IMAGE" "$PROJECT_ROOT/docker"
  log "Done. Isaac Lab image: $ISAACLAB_IMAGE"
fi
