"""Does the velocity field's schedule match the trajectories that actually land?

It did not. The shipped constants are |vz| = 0.547*alt^0.431, and a power-law fit to the
current 480-episode orbit_descent demo set gives 1.262*alt^0.324 on the landed episodes --
roughly 2.3x the coefficient. Per band, against what landed episodes really fly:

    band        schedule   landed actual   error
    5-20 m        1.62         3.66        +2.03
    20-50 m       2.53         6.44        +3.90
    50-100 m      3.52         8.45        +4.93

So through the whole middle of the descent the schedule is about half the real descent
rate, and the symmetric field penalty charges every trajectory -- including the ones that
land -- for descending at the rate that works.

That explains a dead zone found separately. Resolved by altitude, the per-step shaping
reward's ability to predict the eventual outcome is bimodal: AUC 0.87 at 0-2 m, 0.67-0.69
at 50-200 m, and 0.46-0.48 at 5-50 m, i.e. BELOW chance in the middle. In that band the
schedule error (2-4 m/s) swamps the landed-versus-failed difference (0.2-1.8 m/s), so the
penalty is mostly a constant offset carrying no information about the outcome.

Refitting the schedule on the LANDED episodes fixes two of the three weak bands and costs
nothing anywhere:

    band        current AUC   refitted AUC
    0-2 m          0.872         0.909
    2-5 m          0.516         0.746
    5-20 m         0.457         0.489
    20-50 m        0.475         0.510
    50-100 m       0.665         0.662
    100-200 m      0.688         0.682

A CAUTION THAT MATTERS MORE THAN THE FIX. Changing the schedule changes
`target_velocity`, which is what observation slots 16-19 are built from -- and shifting
those under a trained policy is exactly what took this project's orbit clone from 22% to
2% on the same checkpoint and seed block. A schedule refit therefore requires repairing
the demos' guidance slots (scripts/repair_demo_guidance_slots.py) and refitting the clone
before any run, or the warm start is being fed an input whose meaning moved.

Run:
    .venv/bin/python scripts/diag_field_schedule_fit.py
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--demos", default="out/zemzev_orbit_demos_s[0-9]*000.npz")
p.add_argument("--landed-only", action="store_true", default=True)
args = p.parse_args()

from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402
from lunarsim.rl.reward import RewardWeights  # noqa: E402

w = RewardWeights()
ALT = float(OBS_SCALE[2])
files = [f for f in sorted(glob.glob(args.demos)) if "_vfnorm" not in f]
if not files:
    raise SystemExit(f"no demos matched {args.demos!r}")

A, VZ, LAT, OK = [], [], [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    o = d["obs"].astype(np.float32)
    ends, lg = d["episode_ends"], d["landed"].astype(bool)
    starts = np.concatenate([[0], ends[:-1]])
    for s, e, ok in zip(starts, ends, lg):
        A.append(o[s:e, 2] * ALT)
        VZ.append(-o[s:e, 5] * OBS_SCALE[5])
        LAT.append(np.hypot(o[s:e, 3] * OBS_SCALE[3], o[s:e, 4] * OBS_SCALE[4]))
        OK.append(np.full(e - s, bool(ok)))
alt, vz, lat, ok = (np.concatenate(x) for x in (A, VZ, LAT, OK))
mask = ok if args.landed_only else np.ones_like(ok)


def fit(v):
    k = mask & (alt > 0.5) & (v > 0.05)
    c, pp = np.polyfit(np.log(alt[k]), np.log(v[k]), 1)[::-1]
    return float(np.exp(c)), float(pp)


cz, pz = fit(vz)
cl, pl = fit(lat)
print(f"{len(files)} file(s), {int(mask.sum())} transitions "
      f"({'landed only' if args.landed_only else 'all'})\n")
print(f"  shipped : |vz| = {w.vfield_vz_c:.3f}*alt^{w.vfield_vz_p:.3f}   "
      f"lat = {w.vfield_vxy_c:.3f}*alt^{w.vfield_vxy_p:.3f}")
print(f"  refitted: |vz| = {cz:.3f}*alt^{pz:.3f}   lat = {cl:.3f}*alt^{pl:.3f}\n")


def auc(cz_, pz_, cl_, pl_, label):
    tz = np.maximum(cz_ * np.maximum(alt, 1e-6) ** pz_, w.vfield_vz_floor_m_s)
    tl = np.maximum(cl_ * np.maximum(alt, 1e-6) ** pl_, w.vfield_vxy_floor_m_s)
    err = np.sqrt((lat - tl) ** 2 + (vz - tz) ** 2)
    print(f"  {label}")
    print(f"  {'band':>12} {'landed err':>11} {'failed err':>11} {'AUC':>7}")
    for lo, hi in ((0, 2), (2, 5), (5, 20), (20, 50), (50, 100), (100, 201)):
        m = (alt >= lo) & (alt < hi)
        x, y = err[m & ok], err[m & ~ok]
        if len(x) < 50 or len(y) < 50:
            continue
        s = np.sort(y)
        a_ = 1.0 - float(np.searchsorted(s, x, side="left").mean() / len(s))
        print(f"  {f'{lo}-{hi} m':>12} {x.mean():>11.3f} {y.mean():>11.3f} {a_:>7.3f}")


auc(w.vfield_vz_c, w.vfield_vz_p, w.vfield_vxy_c, w.vfield_vxy_p, "shipped schedule:")
print()
auc(cz, pz, cl, pl, "refitted on landed episodes:")
print("\n  AUC = P(a landed step's field error < a failed step's). 0.5 carries no "
      "information.")
print("  Changing these constants moves observation slots 16-19: repair the demos and "
      "refit\n  the clone before any run, or the warm start is fed an input whose "
      "meaning moved.")
