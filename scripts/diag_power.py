"""How big an improvement can a given evaluation budget actually detect?

Written after reporting "statistically indistinguishable" a dozen times without checking
what the measurement could have distinguished. At 192 episodes per arm -- the matrix used
for every comparison in this project -- the smallest effect detectable at p<0.05 with 80%
power is about 15 points. So:

    improvement   episodes/arm   96-blocks   time/arm
       2.0 pts        9,759         102       ~25 h
       3.0 pts        4,330          46       ~11 h
       5.0 pts        1,552          17       ~4 h
       7.5 pts          685           8        ~2 h
      10.0 pts          382           4        ~1 h
      15.0 pts          166           2       ~30 min

v86's observed +1.6 points would need 15,258 episodes per arm. It was never resolvable at
192, so "indistinguishable from zero" was true and incomplete: the honest statement is
that the matrix rules out improvements larger than about 10 points and says nothing below
that. Every RL result here is consistent with anywhere from -10 to +10 points.

The practical consequence is to stop chasing increments the budget cannot see and aim at a
large effect. The project's own target helps: 50% to 75% is +25 points, which 166 episodes
per arm resolves comfortably. What is unmeasurable is the +2 to +5 point steps on the way,
not the destination.

Run:
    .venv/bin/python scripts/diag_power.py --baseline 0.526
    .venv/bin/python scripts/diag_power.py --baseline 0.526 --n 192   # what 192 can see
"""
from __future__ import annotations

import argparse
import math

p = argparse.ArgumentParser()
p.add_argument("--baseline", type=float, default=0.526, help="the arm being improved on")
p.add_argument("--n", type=int, default=None,
               help="if given, report the smallest effect this many episodes/arm resolves")
p.add_argument("--alpha", type=float, default=0.05)
p.add_argument("--power", type=float, default=0.80)
p.add_argument("--episodes-per-block", type=int, default=96)
p.add_argument("--minutes-per-block", type=int, default=15)
args = p.parse_args()

ZA = {0.05: 1.959964, 0.10: 1.644854, 0.01: 2.575829}[round(args.alpha, 2)]
ZB = {0.80: 0.841621, 0.90: 1.281552}[round(args.power, 2)]


def n_per_arm(p1, p2):
    pbar = (p1 + p2) / 2
    return (ZA * math.sqrt(2 * pbar * (1 - pbar))
            + ZB * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2


b = args.baseline
print(f"baseline {b:.1%}, alpha {args.alpha}, power {args.power:.0%}\n")
if args.n:
    lo, hi = 0.0005, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if n_per_arm(b, b + mid) > args.n:
            lo = mid
        else:
            hi = mid
    print(f"  {args.n} episodes/arm resolves effects of about "
          f"{hi * 100:.1f} points and larger")
    se = math.sqrt(2 * 0.25 / args.n)
    print(f"  and its 95% interval on a difference is about "
          f"+/-{1.96 * se * 100:.1f} points")
else:
    print(f"  {'improvement':>12} {'episodes/arm':>14} {'blocks':>8} {'time/arm':>10}")
    for d in (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.25):
        n = n_per_arm(b, b + d)
        blocks = math.ceil(n / args.episodes_per_block)
        mins = blocks * args.minutes_per_block
        t = f"{mins} min" if mins < 120 else f"~{mins / 60:.0f} h"
        print(f"  {d * 100:>11.1f}p {n:>14,.0f} {blocks:>8,} {t:>10}")
