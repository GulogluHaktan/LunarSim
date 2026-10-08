"""Where do the controller's successes and failures separate, and does the reward's
velocity field ask for the successful profile or the average one?

A reward fitted to demonstrations encodes whatever the demonstrator did, FAILURES
INCLUDED, and a policy that maximises it perfectly then inherits the demonstrator's
failure rate. That is a reward-design ceiling no amount of critic work can lift, so it is
worth checking before blaming the critic -- and it checked out: the field asks for less
speed than even the successful episodes fly, at every altitude.

MEASURED on 480 orbit_descent demo episodes (52% landed), lateral speed in m/s:

      band      field target   LANDED   FAILED    diff
      0-1 m         0.30        0.54     1.68    +1.14
      2-5 m         0.51        0.56     1.20    +0.64
     10-20 m        1.79        2.01     2.74    +0.73
     20-50 m        3.75        4.32     6.08    +1.76
    50-100 m        7.27        9.07    12.99    +3.92
   100-200 m       13.29       15.94    21.80    +5.86

Three things come out of it.

The field is TIGHTER than the controller everywhere, including tighter than its successful
episodes, and at touchdown it asks for 0.30 m/s against a 1.2 m/s limit. So the reward
already points at better-than-controller behaviour; a policy that tracked it would beat
the teacher. The ceiling is not in the reward.

The failures are separated from the successes already at the RELEASE state, by 5.86 m/s of
lateral speed. orbit_descent releases between 10 and 30 m/s, so the outcome is strongly
determined by how hard the draw was, not only by control quality during descent.

And the vertical channel is the opposite story: landed and failed episodes fly the same vz
within 0.02 m/s below 2 m, and both sit AT the field's -0.80 target, against a -1.0 limit.
There is 0.2 m/s of margin there, which is why v_z still binds on 28% of touchdowns while
v_xy binds on 62%.

AND WHICH CRITERION ACTUALLY BINDS. Separating failures from successes over 32 clone
touchdowns (17 exceeded a limit, 15 were inside every one), by how far each criterion is
over its limit rather than by the saturating margins:

    on the FAILURES          on the SUCCESSES (tightest criterion)
      v_xy   16  (94%)         v_z    9  (60%)
      tilt    1   (6%)         v_xy   4  (27%)
      v_z     0                tilt   2  (13%)
      leg_diff 0
    v_xy overshoot on the failures: median 1.39x, max 3.90x

So the whole gap between 51% and 100% is ONE scalar: lateral speed at touchdown, arriving
at about 1.67 m/s against a 1.2 m/s limit. Nothing fails on leg_diff, which rules out
terrain slope -- the one thing in the state that the observation genuinely does not carry
-- as the ceiling. The failures are control error, not missing information.

It does NOT follow that the lateral penalty should simply be weighted up. The two channels
are physically coupled: killing lateral speed needs tilt, tilt costs vertical thrust, and
vertical speed is already the tightest criterion on 60% of the SUCCESSES, sitting on the
field's -0.80 target against a -1.0 limit. Charging lateral error harder would buy v_xy
margin with v_z margin that is not there to spend. The equal weighting in
`_velocity_field_penalty`'s Euclidean error may be a reasonable point on that trade-off,
and a change to it has to be measured on BOTH criteria, not on the one that currently
fails.

Run:
    .venv/bin/python scripts/diag_profile_landed_vs_failed.py
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--demos", default="out/zemzev_orbit_demos_s*.npz",
               help="glob; all matching files are pooled")
args = p.parse_args()

from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402
from lunarsim.rl.reward import RewardWeights, target_velocity  # noqa: E402

w = RewardWeights()
files = sorted(glob.glob(args.demos))
if not files:
    raise SystemExit(f"no demo files matched {args.demos!r}")

O, LAND = [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    ends, landed = d["episode_ends"], d["landed"].astype(bool)
    starts = np.concatenate([[0], ends[:-1]])
    for s, e, ok in zip(starts, ends, landed):
        O.append(obs[s:e])
        LAND.append(np.full(e - s, ok))
obs = np.concatenate(O)
land = np.concatenate(LAND)
alt = obs[:, 2] * float(OBS_SCALE[2])
lat = np.hypot(obs[:, 3] * OBS_SCALE[3], obs[:, 4] * OBS_SCALE[4])
vz = obs[:, 5] * float(OBS_SCALE[5])

print(f"{len(obs)} transitions, {len(O)} episodes from {len(files)} file(s), "
      f"{np.mean([l[0] for l in LAND]):.0%} landed\n")
BANDS = ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200))
for name, series, slot in (("LATERAL speed", lat, 0), ("VERTICAL speed", vz, 2)):
    print(f"{name} by altitude (m/s), against the reward's velocity field")
    print(f"  {'band':>12} {'target':>8} {'LANDED':>9} {'FAILED':>9} {'diff':>8}"
          f" {'n_land':>8} {'n_fail':>8}")
    for lo, hi in BANDS:
        m = (alt >= lo) & (alt < hi)
        ml, mf = m & land, m & ~land
        if ml.sum() < 30 or mf.sum() < 30:
            continue
        tgt = target_velocity(w, (lo + hi) / 2)[slot]
        a, b = series[ml].mean(), series[mf].mean()
        print(f"  {f'{lo}-{hi} m':>12} {tgt:>8.2f} {a:>9.2f} {b:>9.2f} {b - a:>+8.2f}"
              f" {int(ml.sum()):>8} {int(mf.sum()):>8}")
    print()
