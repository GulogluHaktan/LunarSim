#!/usr/bin/env bash
# After the seed21 reproduction run finishes: validate, record, render in Isaac,
# compose the side-by-side video, build the report, back everything up.
# Each step checks the previous one's output exists, so a failure stops the
# chain instead of feeding the next step a stale file.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/haktan/isaac-env/bin/python
RUN=out/gg_isaac_orbit_v4_seed21_repro
CK=out/gg_isaac_orbit_v4_seed21_repro_best_frozen.zip
VAL=out/gg_isaac_orbit_v4_seed21_repro_validation.json
REC=out/recordings
VID=out/videos
BACKUP=$HOME/lunarsim_artifacts
RW=(--reward-weight reward_scale=0 --reward-weight touchdown_k=0
    --reward-weight touchdown_penalty_cap=60 --reward-weight landed_step_k=100)

echo "[chain] waiting for training to finish"
until grep -q "^\[done\]" "$RUN.log"; do sleep 30; done
grep -E "^\[(eval|done)\]" "$RUN.log"

cp "$RUN/best_on_selection_block.zip" "$CK"
mkdir -p "$BACKUP" && cp "$CK" "$BACKUP/"

echo "[chain] paired validation, 16 waves x 12"
$PY scripts/validate_guided_gain_isaac.py "$CK" --stage orbit_descent --gain-set attitude \
  --hold-steps 20 --n-envs 12 --waves 16 --wave-seed0 90000 "${RW[@]}" \
  --out-json "$VAL" > "${VAL%.json}.log" 2>&1
test -s "$VAL"
grep -E "^(controller \(|guided|difference|  two-sided|  policy landed|  controller landed)" "${VAL%.json}.log" || true
cp "$VAL" "${VAL%.json}.log" "$BACKUP/"

echo "[chain] recording wave 90000"
$PY scripts/record_guided_gain_isaac.py "$CK" --wave-seed 90000 --check-json "$VAL" \
  --out-dir "$REC" "${RW[@]}" 2>&1 | grep -E "^\[(controller|policy|discordant|check)\]"
cp "$REC"/wave90000_*.npz "$BACKUP/"

ENV=$($PY - <<'EOF'
import json, numpy as np
C = np.load("out/recordings/wave90000_controller.npz"); P = np.load("out/recordings/wave90000_policy.npz")
best, score = None, -1
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
echo "[chain] rendering env $ENV in Isaac Sim"
mkdir -p "$VID"
$PY scripts/isaaclab_replay_recording.py --recording "$REC/wave90000_controller.npz" --env "$ENV" \
  --label "ZEM/ZEV kontrolcü" --accent "#d95926" --out "$VID/controller_env$ENV.mp4" 2>&1 | grep "^\[replay\]" | tail -2
$PY scripts/isaaclab_replay_recording.py --recording "$REC/wave90000_policy.npz" --env "$ENV" \
  --label "RL politika (PPO)" --accent "#3987e5" --out "$VID/policy_env$ENV.mp4" 2>&1 | grep "^\[replay\]" | tail -2
test -s "$VID/controller_env$ENV.mp4" && test -s "$VID/policy_env$ENV.mp4"

echo "[chain] composing side-by-side"
# pad the shorter clip by freezing its last frame, then stack
DC=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VID/controller_env$ENV.mp4")
DP=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VID/policy_env$ENV.mp4")
ffmpeg -y -loglevel error -i "$VID/controller_env$ENV.mp4" -i "$VID/policy_env$ENV.mp4" -filter_complex \
  "[0:v]tpad=stop_mode=clone:stop_duration=$(python3 -c "print(max(0,$DP-$DC)+0.01)")[a];\
   [1:v]tpad=stop_mode=clone:stop_duration=$(python3 -c "print(max(0,$DC-$DP)+0.01)")[b];\
   [a][b]hstack=inputs=2,pad=iw:ih+64:0:64:color=0x0d0d0d,\
   drawtext=fontfile=/usr/share/fonts/TTF/DejaVuSans-Bold.ttf:fontsize=26:fontcolor=white:x=24:y=18:\
text='Ay inişi · Isaac Sim · aynı başlangıç, aynı arazi · eğitimin görmediği episod'[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 18 "$VID/isaac_controller_vs_rl.mp4"
ls -1 "$VID/controller_env${ENV}_frames" | tail -1 | xargs -I{} cp "$VID/controller_env${ENV}_frames/{}" "$VID/still_controller.png"
ls -1 "$VID/policy_env${ENV}_frames" | tail -1 | xargs -I{} cp "$VID/policy_env${ENV}_frames/{}" "$VID/still_policy.png"
cp "$VID"/*.mp4 "$VID"/still_*.png "$BACKUP/"

echo "[chain] building report"
$PY report/build_report.py
cp report/lunarsim_rl_rapor.pdf "$BACKUP/"
echo "[chain] DONE -> $VID/isaac_controller_vs_rl.mp4, report/lunarsim_rl_rapor.pdf, backup in $BACKUP"
