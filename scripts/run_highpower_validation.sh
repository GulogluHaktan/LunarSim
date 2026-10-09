#!/usr/bin/env bash
# THE PROTOCOL, declared before the numbers exist.
#
# Every comparison in this project has been a 192-episode pooled matrix, and the power
# analysis says that resolves about 14 points with a +/-10 point interval. Every RL result
# so far is inside that, so none of them can be acted on -- and picking the best checkpoint
# from a run makes it worse, because two of four runs peaked mid-training and then declined
# (v77 at 310k, v87 at 750k), so the peak of a declining run is a selection artifact.
#
# Two separate fixes, and they have to be declared in advance or they are just two more
# ways to flatter a result:
#
#   SELECTION. Pick the checkpoint on seed block 7000 ONLY. Never on the pooled number,
#   never on the training-time log. The chosen checkpoint's score then comes from blocks it
#   had no part in choosing.
#
#   POWER. Score it on K FRESH seed blocks, never on 7000. At 96 episodes a block the 95%
#   interval on a difference is +/-14.1 points with one block, +/-10.0 with two, +/-7.1 with
#   four, +/-5.0 with eight. The honest estimates on the table right now are +2.1 (v86) and
#   +5.2 (v87) against the clone, so FOUR blocks is the minimum that could resolve the
#   larger of them and eight is what it takes to be sure.
#
# Applied to v87 after the fact this protocol reads +5.2 points, which is why it is being
# written down now rather than after v88 -- a protocol chosen once the numbers are in is not
# a protocol.
#
# Usage: run_highpower_validation.sh CLONE.zip CANDIDATE.zip [N_BLOCKS]
set -u
cd /home/haktan/Projects/LunarSim
PY=/home/haktan/isaac-env/bin/python
CLONE="${1:?clone checkpoint}"
CAND="${2:?candidate checkpoint, selected on block 7000 only}"
K="${3:-4}"
# Blocks disjoint from 7000 (selection) and 41000 (already used in every matrix).
BLOCKS=(90000 120000 150000 180000 210000 240000 270000 300000)

echo "high-power validation: $K fresh blocks x 96 episodes = $((K*96)) per arm"
echo "  clone     : $CLONE"
echo "  candidate : $CAND   (must have been selected on block 7000 alone)"
echo
for ck in "$CLONE" "$CAND"; do
  tot=0; n=0
  for i in $(seq 0 $((K-1))); do
    s0=${BLOCKS[$i]}
    line=$(timeout 2400 $PY scripts/diag_stage_landing_rate.py --checkpoint "$ck" \
      --stage orbit_descent --episodes 96 --seed0 "$s0" --terrain-seed 7 2>/dev/null \
      | grep -oE "[0-9]+/96 landed_safely" | head -1)
    got=${line%%/*}
    printf '  %-34s seed0=%-7s %s\n' "$(basename "$ck" .zip)" "$s0" "${line:-NO OUTPUT}"
    [ -n "${got:-}" ] && { tot=$((tot+got)); n=$((n+96)); }
  done
  printf '  %-34s TOTAL %d/%d\n\n' "$(basename "$ck" .zip)" "$tot" "$n"
  echo "$tot $n" >> /tmp/hp_$$.txt
done
$PY - <<'PYEOF' /tmp/hp_$$.txt
import math, sys
rows = [tuple(map(int, l.split())) for l in open(sys.argv[1]) if l.strip()]
if len(rows) != 2:
    raise SystemExit("need both arms")
(a, n), (b, m) = rows
p1, p2 = a/n, b/m
pool = (a+b)/(n+m)
se = math.sqrt(pool*(1-pool)*(1/n+1/m))
z = (p2-p1)/se
p = math.erfc(abs(z)/math.sqrt(2))
print(f"  clone {p1:.1%}   candidate {p2:.1%}   diff {(p2-p1)*100:+.1f} pts")
print(f"  z={z:+.2f}  p={p:.3f}  95% CI [{(p2-p1-1.96*se)*100:+.1f}, "
      f"{(p2-p1+1.96*se)*100:+.1f}] pts")
print("  SIGNIFICANT" if p < 0.05 else "  not resolved at this budget")
PYEOF
rm -f /tmp/hp_$$.txt
