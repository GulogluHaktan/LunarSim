"""Does the reward know which trajectory landed? Scanned against a shaping parameter.

Written because a reward change shipped on the wrong criterion. The shaping reward was
rebalanced across altitude on the measurement that the decisive 0-5 m band carried 1.5%
of it against 14.8% of the steps and 94% of the failures -- and the exponent was chosen
to give that band its step share. That criterion was assumed, not measured.

This measures a different and more direct one: the value difference at the RELEASE state
between demo episodes that landed and ones that crashed, i.e. how much the reward knows,
in hindsight, about which trajectory was good. Scanned over the normalisation exponent:

    p        c        shaping separation    relative
    0.000   1.0000         +1.084            1.00x
    0.250   1.6662         +0.916            0.84x
    0.375   2.1087         +0.826            0.76x
    0.500   2.6242         +0.734            0.68x
    0.750   3.8171         +0.545            0.50x
    1.000   5.0031         +0.370            0.34x

Monotone, and p = 0 -- the original absolute-error form -- is the maximum. The terminal's
contribution is p-independent at +2.073, so the normalisation trades discriminative power
for band balance, and it is the discriminative power that the policy needs.

WHY: the land-versus-crash signal lives high up. Failing episodes are already +5.86 m/s
faster laterally at 100-200 m, which is the CAUSE; by the decisive band the difference
(0.54 against 1.68 m/s) is the CONSEQUENCE. Down-weighting the cruise removes the part of
the reward that can see the failure coming.

The two criteria genuinely conflict, and NEITHER settles it on its own: separation is what
the reward knows in hindsight, while policy improvement needs the GRADIENT -- whether
raising the reward from where the policy currently is moves toward landing. A reward can
rank finished trajectories well and still have a useless gradient, or the reverse. Only a
run adjudicates, which is why v87 was left running at p = 0.75 with the expectation,
written down first, that it comes out equal or worse.

Run:
    .venv/bin/python scripts/diag_reward_separation.py
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--demos", default="out/zemzev_orbit_demos_s[0-9]*000.npz")
p.add_argument("--gamma-shaping", type=float, default=0.99)
p.add_argument("--gamma-terminal", type=float, default=0.999)
p.add_argument("--scan", type=float, nargs="+",
               default=[0.0, 0.25, 0.375, 0.5, 0.75, 1.0])
args = p.parse_args()

from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402
from lunarsim.rl.reward import RewardWeights, target_velocity  # noqa: E402

w0 = RewardWeights()
ALT = float(OBS_SCALE[2])
files = [f for f in sorted(glob.glob(args.demos)) if "_vfnorm" not in f]
if not files:
    raise SystemExit(f"no demos matched {args.demos!r}")

eps = []
for f in files:
    d = np.load(f, allow_pickle=True)
    o = d["obs"].astype(np.float32)
    r = d["rewards"].astype(np.float64)
    ends, lg = d["episode_ends"], d["landed"].astype(bool)
    starts = np.concatenate([[0], ends[:-1]])
    for s, e, ok in zip(starts, ends, lg):
        seg = o[s:e]
        a = seg[:, 2] * ALT
        lat = np.hypot(seg[:, 3] * OBS_SCALE[3], seg[:, 4] * OBS_SCALE[4])
        vz = seg[:, 5] * OBS_SCALE[5]
        tl = np.array([target_velocity(w0, float(x))[0] for x in a])
        tz = np.array([target_velocity(w0, float(x))[2] for x in a])
        err = np.sqrt((lat - tl) ** 2 + (vz - tz) ** 2)
        tgt = np.maximum(tl, w0.vfield_vxy_floor_m_s)
        sh = r[s:e].copy()
        term = float(sh[-1])
        sh[-1] = 0.0
        eps.append((bool(ok), a, err, tgt, sh, term, len(sh)))

SC = w0.vfield_k * w0.reward_scale * w0.reward_total_scale
ALLERR = np.concatenate([e[2] for e in eps])
ALLTGT = np.concatenate([e[3] for e in eps])

print(f"{len(eps)} episodes from {len(files)} file(s), "
      f"{sum(e[0] for e in eps)} landed\n")
print(f"  {'p':>7} {'c':>9} {'shaping sep':>13} {'relative':>10}")
base = None
for pw in args.scan:
    c = 1.0 if pw == 0 else float(ALLERR.mean() / (ALLERR / ALLTGT ** pw).mean())
    got = {True: [], False: []}
    for ok, a, err, tgt, sh, term, n in eps:
        new_f = err if pw == 0 else c * err / tgt ** pw
        d_field = -SC * (new_f - err)
        t = np.arange(n)
        got[ok].append(float((args.gamma_shaping ** t * (sh + d_field)).sum()))
    s = float(np.mean(got[True]) - np.mean(got[False]))
    if base is None:
        base = s
    print(f"  {pw:>7.3f} {c:>9.4f} {s:>+13.3f} {s / base:>9.2f}x")

tl_ = np.mean([args.gamma_terminal ** (e[6] - 1) * e[5] for e in eps if e[0]])
tf_ = np.mean([args.gamma_terminal ** (e[6] - 1) * e[5] for e in eps if not e[0]])
print(f"\n  terminal separation, independent of p: {tl_ - tf_:+.3f}")
print("\n  Separation is what the reward knows in hindsight. Policy improvement needs the")
print("  GRADIENT, which this does not measure -- so a run, not this table, decides.")
