#!/usr/bin/env bash
# Final deliverable: validated RL flight, Isaac Sim cameras, computed LiDAR map
# (legacy 4-panel dashboard, unchanged composer) + RL telemetry column, report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/haktan/isaac-env/bin/python
CAP=out/videos/rl_capture_env10
REC=out/recordings/wave90000_policy.npz
OUTV=out/videos/rl_landing_lidar_dashboard.mp4
BACKUP=$HOME/lunarsim_artifacts

echo "[final] waiting for camera render"
while pgrep -f "isaaclab_replay_capture.py --recording $REC" >/dev/null; do sleep 15; done
test -s "$CAP/frames_index.csv"
echo "[final] computed LiDAR scans"
$PY scripts/synth_lidar_scans.py --recording "$REC" --env 10 --capture-dir "$CAP" | tail -1
echo "[final] legacy 4-panel dashboard (compose_capture_video.py, unchanged)"
$PY scripts/compose_capture_video.py --capture-dir "$CAP" --out "$CAP/dashboard.mp4" \
  --map-out "$CAP/map.mp4" --fps 30 --scan-format world --pose-lag-s 0 2>&1 | tail -3
test -s "$CAP/dashboard.mp4"
echo "[final] adding RL telemetry column"
$PY scripts/compose_rl_dashboard_video.py --recording "$REC" --env 10 --frames "$CAP/dashboard_frames" \
  --pattern 'frame_{:04d}.png' --index-csv "$CAP/frames_index.csv" --out "$CAP/_with_rl.mp4" \
  --still "$CAP/_with_rl_final.png"
echo "[final] labelling the LiDAR as computed"
ffmpeg -y -loglevel error -i "$CAP/_with_rl.mp4" -vf "pad=iw:ih+44:0:0:color=0x0d0d0d,\
drawtext=fontfile=/usr/share/fonts/TTF/DejaVuSans.ttf:fontsize=19:fontcolor=0xc3c2b7:x=18:y=h-34:\
text='Doğrulanmış RL uçuşu (Isaac Sim, eğitimin görmediği episod). LiDAR haritası HESAPLANMIŞ\: OS2 deseni + gerçek montaj, kayıtlı araziye ışın izleme — RTX sensör çıktısı değil.'" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUTV"
cp "$CAP/_with_rl_final.png" out/videos/rl_landing_lidar_dashboard_final.png
echo "[final] report"
$PY report/build_report.py
mkdir -p "$BACKUP"; cp "$OUTV" out/videos/rl_landing_lidar_dashboard_final.png "$CAP/map.mp4" report/lunarsim_rl_rapor.pdf "$BACKUP/"
echo "[final] DONE -> $OUTV"
