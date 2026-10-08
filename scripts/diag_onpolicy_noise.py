"""Is the on-policy (realised-return) estimator quieter than the learned critic here?

Pairs with scripts/diag_action_signal_to_noise.py, which found that a learned Q must
resolve an action difference 2559x smaller than its own residual std. That number
condemns Q-based ranking but says nothing about the alternative, so this measures the
alternative's noise on the same data.

The quantity: group demo transitions by near-identical observations taken FROM DIFFERENT
EPISODES, and look at how much the realised return-to-go varies inside a group. That
spread is the irreducible sampling noise of an on-policy advantage at a state, the way the
critic's residual std is the noise of the Q estimator -- with one decisive difference. The
critic's error is a function-approximation bias: SYSTEMATIC, and it does not average out
over gradient steps. The realised return's spread is sampling variance: ZERO-MEAN, and it
does.

MEASURED on 480 orbit_descent demo episodes (neighbour distance 0.216 z-units):

    calibrated critic residual std   1.884    systematic
    realised-return spread           0.266    zero-mean
    one-step action signal           0.000736

    samples needed to resolve the action difference by averaging
      over 1 step            signal 0.0007   ->  130,295
      over 100 steps         signal 0.0736   ->       13
      over a whole episode   signal 0.5255   ->        0

So the on-policy estimator is 7x quieter AND unbiased, and over the horizon a policy
change actually persists the sample requirement is trivially met -- a 400k-step run
collects 400,000 transitions. That is the quantitative case for PPO here, and it was
written before the run rather than after it.

Two honest caveats. The 0.266 spread comes from ONE policy's demonstrations (the
controller's), so it understates the return variance of a stochastic policy still
exploring. And the "13 samples" figure assumes the action difference persists over the
horizon, which is true of a parameter change -- what a policy gradient actually makes --
but not of a single sampled action.

Run:
    .venv/bin/python scripts/diag_onpolicy_noise.py
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--demos", default="out/zemzev_orbit_demos_s*.npz")
p.add_argument("--gamma-shaping", type=float, default=0.99)
p.add_argument("--gamma-terminal", type=float, default=0.999)
p.add_argument("--pool", type=int, default=40000, help="transitions searched")
p.add_argument("--queries", type=int, default=1500)
p.add_argument("--k", type=int, default=8, help="neighbours per query")
p.add_argument("--critic-noise", type=float, default=1.884,
               help="for the comparison line; from diag_action_signal_to_noise.py")
p.add_argument("--action-signal", type=float, default=0.000736)
args = p.parse_args()

files = sorted(glob.glob(args.demos))
if not files:
    raise SystemExit(f"no demos matched {args.demos!r}")

O, V, EP = [], [], []
n_ep = 0
for f in files:
    d = np.load(f, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    r = d["rewards"].astype(np.float64)
    ends = d["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    for s, e in zip(starts, ends):
        seg = r[s:e]
        n = len(seg)
        shap = seg.copy()
        shap[-1] = 0.0
        vs = np.zeros(n)
        acc = 0.0
        for i in range(n - 1, -1, -1):
            acc = shap[i] + args.gamma_shaping * acc
            vs[i] = acc
        vt = (args.gamma_terminal ** np.arange(n - 1, -1, -1)) * seg[-1]
        O.append(obs[s:e])
        V.append(vs + vt)
        EP.append(np.full(n, n_ep))
        n_ep += 1
obs, V, ep = np.concatenate(O), np.concatenate(V), np.concatenate(EP)

rng = np.random.default_rng(0)
pool = rng.choice(len(obs), min(args.pool, len(obs)), replace=False)
P, VP, EPP = obs[pool], V[pool], ep[pool]
spreads, dists = [], []
for i in rng.choice(len(pool), min(args.queries, len(pool)), replace=False):
    dd = np.linalg.norm(P - P[i], axis=1)
    dd[EPP == EPP[i]] = np.inf          # neighbours from OTHER episodes only
    idx = np.argpartition(dd, args.k)[:args.k]
    if not np.isfinite(dd[idx]).all():
        continue
    spreads.append(VP[idx].std())
    dists.append(dd[idx].mean())
spreads = np.asarray(spreads)
sp = float(np.median(spreads))

print(f"[onpolicy-noise] {len(obs)} transitions, {n_ep} episodes, {len(files)} file(s)")
print(f"  neighbour observation distance (z-units): median {np.median(dists):.3f}\n")
print(f"  realised return-to-go spread among near-identical states from other episodes")
print(f"    median {sp:.3f}   mean {spreads.mean():.3f}   "
      f"p90 {np.percentile(spreads, 90):.3f}\n")
print(f"  calibrated critic residual std   {args.critic_noise:.3f}   "
      f"SYSTEMATIC - does not average out")
print(f"  realised-return spread           {sp:.3f}   ZERO-MEAN - does")
print(f"  one-step action signal           {args.action_signal:.6f}\n")
print(f"  samples needed to resolve the action difference by averaging:")
for n, lab in ((1, "1 step"), (100, "100 steps"), (714, "a whole episode")):
    sig = args.action_signal * n
    print(f"    over {lab:<16} signal {sig:.4f}  ->  {(sp / sig) ** 2:>12,.0f}")
