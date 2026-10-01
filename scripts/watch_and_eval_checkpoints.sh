#!/usr/bin/env bash
# Watches out/sac_training_run/ for new/updated SAC checkpoints while training
# runs, and for each one automatically runs the policy-controlled eval capture
# (scripts/run_isaaclab_policy_eval_capture.sh): real policy flight, camera +
# lidar + telemetry, 4-panel dashboard video + standalone lidar video.
#
# Meant to run alongside the training process (background), so that by the
# time training finishes there is already a reviewable video/telemetry/log
# snapshot for every completed curriculum stage under out/eval_snapshots/.
#
# Usage: ./scripts/watch_and_eval_checkpoints.sh [checkpoints-dir] [poll-seconds]
#   checkpoints-dir: default out/sac_training_run
#   poll-seconds:    default 60
#
# Stop it with: kill $(cat out/eval_snapshots/.watcher.pid)
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CKPT_DIR="${1:-out/sac_training_run}"
POLL_S="${2:-60}"
SNAP_ROOT="out/eval_snapshots"
STATE_FILE="$SNAP_ROOT/.watcher_seen.txt"
LOG_FILE="$SNAP_ROOT/watcher.log"

mkdir -p "$SNAP_ROOT"
touch "$STATE_FILE"
echo $$ > "$SNAP_ROOT/.watcher.pid"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "watcher started (pid=$$), watching '$CKPT_DIR' every ${POLL_S}s"

# REAL BUG FOUND (after 3 full training runs' worth of eval results all
# looking equally bad on vxy, no matter what reward/threshold change was
# tried): this watcher called run_isaaclab_policy_eval_capture.sh with NO
# spawn/tile args, so EVERY checkpoint -- regardless of which curriculum
# stage actually produced it -- was evaluated against the eval script's own
# generic defaults (0-5 m/s random spawn speed, 15m random xy offset, 150m
# tile). `hover_only`/`hover_only_easy` were trained with EXACTLY ZERO
# spawn horizontal speed and ZERO xy offset (dead-center hover only) -- so
# every "hover_only" eval was silently testing the policy against an
# out-of-distribution scenario it was never trained to handle, which is
# fully consistent with the touchdown horizontal velocity never trending
# down across any of the three attempts, no matter what was changed on the
# training side. This table mirrors STAGES in scripts/train_sac_isaac.py --
# keep the two in sync if that file's spawn parameters ever change.
_stage_args() {
  # REAL BUG FOUND: checkpoint files are saved as
  # sac_lunar_lander_isaac_<stage>.zip, so `basename ... .zip` (stage_name)
  # is "sac_lunar_lander_isaac_orbit_descent", not "orbit_descent" -- the
  # case patterns below never matched anything and silently fell through to
  # the empty-args default every single time, for every stage, since this
  # function was first added. Matched on a glob suffix instead.
  case "$1" in
    *hover_only_easy|*hover_only)
      echo "--spawn-altitude-min-m 20 --spawn-altitude-max-m 20 --spawn-xy-radius-m 0 \
            --spawn-speed-min-m-s 0 --spawn-speed-max-m-s 0 --tile-size-m 40 --terrain-roughness-scale 0.0" ;;
    *final_approach)
      echo "--spawn-altitude-min-m 35 --spawn-altitude-max-m 35 --spawn-xy-radius-m 15 \
            --spawn-speed-min-m-s 0 --spawn-speed-max-m-s 3 --tile-size-m 60 --terrain-roughness-scale 1.0" ;;
    *orbit_descent)
      echo "--spawn-altitude-min-m 200 --spawn-altitude-max-m 200 --spawn-xy-radius-m 80 \
            --spawn-speed-min-m-s 10 --spawn-speed-max-m-s 30 --tile-size-m 600 --terrain-roughness-scale 1.0 \
            --max-episode-s 60" ;;
    *)
      echo "" ;;
  esac
}

while true; do
  if [[ -d "$CKPT_DIR" ]]; then
    for ckpt in "$CKPT_DIR"/*.zip; do
      [[ -e "$ckpt" ]] || continue
      ckpt_abs="$(readlink -f "$ckpt")"
      mtime="$(stat -c %Y "$ckpt_abs" 2>/dev/null || echo 0)"
      key="${ckpt_abs}@${mtime}"
      if grep -qxF "$key" "$STATE_FILE" 2>/dev/null; then
        continue
      fi

      ckpt_rel="${ckpt_abs#$PROJECT_ROOT/}"
      stage_name="$(basename "$ckpt" .zip)"
      out_subdir="${stage_name}_$(date +%Y%m%d_%H%M%S)"
      stage_args="$(_stage_args "$stage_name")"

      log "new/updated checkpoint: $ckpt_rel -> capturing into $SNAP_ROOT/$out_subdir (stage args: $stage_args)"
      if ./scripts/run_isaaclab_policy_eval_capture.sh "$ckpt_rel" "$out_subdir" $stage_args \
           >> "$SNAP_ROOT/${out_subdir}.capture.log" 2>&1; then
        log "capture OK: $SNAP_ROOT/$out_subdir"
      else
        log "capture FAILED for $ckpt_rel (see $SNAP_ROOT/${out_subdir}.capture.log)"
      fi

      echo "$key" >> "$STATE_FILE"
    done
  fi
  sleep "$POLL_S"
done
