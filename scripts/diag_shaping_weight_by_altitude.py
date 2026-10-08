"""Where does the shaping reward actually LIVE, and is that where the task is decided?

A step-weighted dense reward pays wherever the steps are and wherever the error magnitudes
are, and on a 200 m descent both favour the cruise. Measured on 480 orbit_descent demo
episodes, per-step reward with the terminal removed:

    band        share of steps   share of |shaping|
    0-2 m            6.8%              0.8%
    2-5 m            7.9%              0.8%
    5-20 m          12.0%              4.5%
    20-50 m         13.4%             11.5%
    50-100 m        16.7%             22.8%
    100-200 m       43.2%             59.6%

    the decisive 0-5 m band:  14.8% of steps,  1.5% of the shaping reward
    the 50-200 m cruise:      59.9% of steps, 82.5% of the shaping reward

So the cruise is over-weighted 55:1 against the decisive band relative to its share of
the reward, while 94% of the failures are in the last five metres -- lateral speed at
touchdown, median overshoot 1.39x against a 1.2 m/s limit.

The cause is in the shape of the field rather than in a weight: `_velocity_field_penalty`
charges the ABSOLUTE velocity error, and the field's own targets are 17 m/s at 200 m and
0.30 m/s at touchdown, so the error available to be charged scales with altitude. The
reward has the priority backwards: the landing criteria are absolute, so a 0.3 m/s error
one metre up matters MORE than a 3 m/s error at a hundred.

THIS IS WHAT v86 MEASURED FROM THE OTHER SIDE. PPO at gamma 0.999 improved the proxy
return with a 95% interval that excludes zero (+0.027 to +2.408 per 1M steps) and left
the landing rate unresolved (+0.92 pts/1M, CI -3.90 to +5.74). Since the landing rate did
not move, the terminal contribution did not move, so the whole proxy gain came from
shaping -- i.e. from the cruise, which is 82.5% of it. The agent did exactly what it was
paid to do.

Run:
    .venv/bin/python scripts/diag_shaping_weight_by_altitude.py
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--demos", default="out/zemzev_orbit_demos_s*.npz")
args = p.parse_args()

from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402
from lunarsim.rl.reward import RewardWeights, target_velocity  # noqa: E402

ALT = float(OBS_SCALE[2])
files = sorted(glob.glob(args.demos))
if not files:
    raise SystemExit(f"no demos matched {args.demos!r}")

alts, shap = [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    r = d["rewards"].astype(np.float64)
    ends = d["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    for s, e in zip(starts, ends):
        seg = r[s:e].copy()
        seg[-1] = 0.0            # the terminal lives entirely in the last step
        alts.append(obs[s:e, 2] * ALT)
        shap.append(seg)
alt = np.concatenate(alts)
sh = np.concatenate(shap)
tot = float(np.abs(sh).sum())

print(f"{len(alt)} transitions from {len(files)} demo file(s); terminal removed\n")
print(f"  {'band':>12} {'steps':>9} {'share of steps':>15} {'share of |shaping|':>19} "
      f"{'field target':>13}")
w = RewardWeights()
for lo, hi in ((0, 2), (2, 5), (5, 20), (20, 50), (50, 100), (100, 201)):
    m = (alt >= lo) & (alt < hi)
    if m.sum() == 0:
        continue
    tgt = target_velocity(w, (lo + hi) / 2)[0]
    print(f"  {f'{lo}-{hi} m':>12} {int(m.sum()):>9} {m.mean():>14.1%} "
          f"{np.abs(sh[m]).sum() / tot:>18.1%} {tgt:>12.2f} m/s")
dec, cru = alt < 5, alt >= 50
print(f"\n  decisive 0-5 m : {dec.mean():>5.1%} of steps, "
      f"{np.abs(sh[dec]).sum() / tot:>5.1%} of the shaping reward")
print(f"  cruise 50-200 m: {cru.mean():>5.1%} of steps, "
      f"{np.abs(sh[cru]).sum() / tot:>5.1%} of the shaping reward")
ratio = (np.abs(sh[cru]).sum() / max(np.abs(sh[dec]).sum(), 1e-9))
print(f"\n  the cruise carries {ratio:.0f}x the decisive band's shaping reward, on "
      f"{cru.mean() / dec.mean():.1f}x the steps")
print(f"  and 94% of the failures are in the decisive band (v_xy at touchdown).")
