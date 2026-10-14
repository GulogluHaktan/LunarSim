#!/usr/bin/env bash
# Second half of the deliverables chain, RL video only (no controller video):
# wait for the paired validation already running, record the validation wave,
# render the RL policy in Isaac Sim, add the telemetry dashboard, build the
# report, back everything up outside the repo.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/haktan/isaac-env/bin/python
CK=out/gg_isaac_orbit_v4_seed21_repro_best_frozen.zip
VAL=out/gg_isaac_orbit_v4_seed21_repro_validation.json
REC=out/recordings
VID=out/videos
BACKUP=$HOME/lunarsim_artifacts
RW=(--reward-weight reward_scale=0 --reward-weight touchdown_k=0
    --reward-weight touchdown_penalty_cap=60 --reward-weight landed_step_k=100)

echo "[chain] waiting for the paired validation to finish"
while pgrep -f "validate_guided_gain_isaac.py $CK" >/dev/null; do sleep 20; done
test -s "$VAL"
grep -E "^(controller \(|guided|difference|  two-sided|  policy landed|  controller landed)" "${VAL%.json}.log" || true
mkdir -p "$BACKUP" && cp "$CK" "$VAL" "${VAL%.json}.log" "$BACKUP/"

echo "[chain] recording wave 90000"
$PY scripts/record_guided_gain_isaac.py "$CK" --wave-seed 90000 --check-json "$VAL" \
  --out-dir "$REC" "${RW[@]}" 2>&1 | grep -E "^\[(controller|policy|discordant|check)\]"
cp "$REC"/wave90000_*.npz "$BACKUP/"

ENV=$($PY - <<'EOF'
import json, numpy as np
C = np.load("out/recordings/wave90000_controller.npz"); P = np.load("out/recordings/wave90000_policy.npz")
best, score = None, -1.0
for i in range(int(C["n_envs"])):
    oc = json.loads(str(C[f"e{i}_outcome"])); op = json.loads(str(P[f"e{i}_outcome"]))
    if op["landed"] and not oc["landed"]:
        vc = np.hypot(oc["impact"].get("vx", 0), oc["impact"].get("vy", 0))
        vp = np.hypot(op["impact"].get("vx", 0), op["impact"].get("vy", 0))
        if vc - vp > score:
            best, score = i, vc - vp
print(best if best is not None else 0)
EOF
)
echo "[chain] rendering RL policy, env $ENV, in Isaac Sim"
mkdir -p "$VID"
$PY scripts/isaaclab_replay_recording.py --recording "$REC/wave90000_policy.npz" --env "$ENV" \
  --label "RL politika (PPO)" --accent "#3987e5" --out "$VID/policy_env$ENV.mp4" 2>&1 \
  | grep -E "^\[replay\] (env|wrote)"
test -s "$VID/policy_env$ENV.mp4"

echo "[chain] adding dashboard"
$PY scripts/compose_rl_dashboard_video.py --recording "$REC/wave90000_policy.npz" --env "$ENV" \
  --frames "$VID/policy_env${ENV}_frames" --out "$VID/rl_landing_dashboard.mp4" \
  --still "$VID/rl_landing_dashboard_final.png"
cp "$VID/rl_landing_dashboard.mp4" "$VID/rl_landing_dashboard_final.png" "$BACKUP/"

echo "[chain] building report"
$PY report/build_report.py
cp report/lunarsim_rl_rapor.pdf "$BACKUP/"
echo "[chain] DONE -> $VID/rl_landing_dashboard.mp4 + report/lunarsim_rl_rapor.pdf (backup: $BACKUP)"
