#!/usr/bin/env bash
# Re-encode a capture's videos for the demo site under docs/ and refresh the
# stills that go with them.
#
# The site shows the COMBINED 4-panel dashboard rather than the individual
# chase/nav/map feeds, so only two videos per capture are needed: the
# dashboard, and (for the landing capture) the standalone LiDAR mapping
# video that the mapping section is built around.
#
# Usage:
#   scripts/refresh_demo_site.sh landing  out/eval_snapshots/hero_fixed
#   scripts/refresh_demo_site.sh freefall out/static_capture
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROLE="${1:?usage: refresh_demo_site.sh <landing|freefall> <capture-dir>}"
D="${2:?usage: refresh_demo_site.sh <landing|freefall> <capture-dir>}"
[ -d "$D" ] || { echo "no such capture dir: $D" >&2; exit 1; }

web() {  # src dst [crf]
  [ -f "$1" ] || { echo "missing: $1" >&2; exit 1; }
  ffmpeg -y -loglevel error -i "$1" -c:v libx264 -crf "${3:-25}" -preset slow \
         -pix_fmt yuv420p -movflags +faststart -an "$2"
  echo "  $(du -h "$2" | cut -f1)  $2"
}

# the capture scripts write dashboard.mp4/map.mp4 inside the capture dir;
# compose_capture_video.py's own defaults put them alongside it instead.
pick() {
  for c in "$D/$1.mp4" "${D%/}_$1.mp4"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo "missing $1 video for $D" >&2; exit 1
}

web "$(pick dashboard)" "docs/assets/video/${ROLE}_dashboard.mp4"
if [ "$ROLE" = "landing" ]; then
  web "$(pick map)" "docs/assets/video/landing_map.mp4"
fi

# Posters by image contrast, not by a fixed timestamp: a nav/dashboard frame
# sampled at an arbitrary time can be flat grey or (as happened before) solid
# black, which makes a useless poster.
TARGETS="${ROLE}_dashboard"
[ "$ROLE" = "landing" ] && TARGETS="$TARGETS landing_map"
.venv/bin/python - $TARGETS <<'PY'
import subprocess, sys
import numpy as np
from PIL import Image

for name in sys.argv[1:]:
    src = f"docs/assets/video/{name}.mp4"
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
        capture_output=True, text=True, check=True).stdout.strip())
    best = (0.0, -1.0)
    for frac in np.linspace(0.15, 0.9, 10):
        t = dur * frac
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src,
                        "-frames:v", "1", "/tmp/_poster.png"], check=True)
        a = np.asarray(Image.open("/tmp/_poster.png").convert("L"), dtype=float)
        if a.std() > best[1]:
            best = (t, float(a.std()))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{best[0]:.2f}", "-i", src,
                    "-frames:v", "1", "-q:v", "4", f"docs/assets/img/{name}_poster.jpg"], check=True)
    print(f"  poster {name}: t={best[0]:.1f}s contrast={best[1]:.1f}")
PY

for cand in "${D%/}_map.png" "$D/map.png"; do
  [ -f "$cand" ] || continue
  out="docs/assets/img/${ROLE}_map_final.jpg"
  [ "$ROLE" = "landing" ] && out="docs/assets/img/landing_map_final.jpg"
  ffmpeg -y -loglevel error -i "$cand" -vf scale=1600:-1 -q:v 4 "$out"
  echo "  $(du -h "$out" | cut -f1)  $out"
  break
done

echo "docs/ is now $(du -sh docs | cut -f1)"
echo "remember to refresh the hero numbers in docs/index.html from $D/eval_summary.txt"
