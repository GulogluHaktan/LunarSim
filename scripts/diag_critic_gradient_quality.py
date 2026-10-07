"""Does the critic's ACTION gradient point toward the expert's action?

The distinction this measures is the one v65 forced into the open. Its Q0-vs-MC gap is
+0.02 to +0.96 -- the critic predicts its own policy's discounted return almost exactly,
after dual-gamma, CQL and the Cal-QL clamp took the gap down from +20..+51. And the policy
still dips from 75% to 4.5% in the window after the actor is released.

A calibrated VALUE is not a correct GRADIENT. Q can predict the return of the action the
policy actually takes while having the wrong derivative with respect to that action, and
the actor only ever uses the derivative: its loss is `ent_coef*log_prob - Q(s, pi(s))`, so
it moves in the direction of the steepest increase of Q, not toward any value Q got right.

So: at states drawn from the demonstrator's own trajectories, compute

    g      = grad_a Q(s, a) at a = pi(s)        the direction the actor is pushed
    d      = a_expert - pi(s)                   the direction of the 91% policy
    cos    = <g, d> / (|g| |d|)

A positive cosine means the critic is pushing the actor toward the behaviour that lands.
A negative one means the actor is being pushed away from it by a critic whose values are
nonetheless right -- which no amount of further calibration would fix, and which would make
the whole conservatism line of attack beside the point.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch as th
from stable_baselines3 import SAC

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--demos", default="out/zemzev_ramp35_demos_v20.npz")
p.add_argument("--samples", type=int, default=4000)
p.add_argument("--landed-only", action="store_true", default=True,
               help="only states from successful demo episodes define 'toward the expert'")
args = p.parse_args()

d = np.load(args.demos, allow_pickle=True)
obs, act = d["obs"].astype(np.float32), d["actions"].astype(np.float32)
if args.landed_only and "landed" in d:
    ends = d["episode_ends"]; landed = d["landed"].astype(bool)
    starts = np.concatenate([[0], ends[:-1]])
    keep = np.zeros(len(obs), bool)
    for s, e, lg in zip(starts, ends, landed):
        if lg:
            keep[s:e] = True
    obs, act = obs[keep], act[keep]

rng = np.random.default_rng(0)
idx = rng.choice(len(obs), size=min(args.samples, len(obs)), replace=False)
obs, act = obs[idx], act[idx]

m = SAC.load(args.checkpoint, device="cpu")
o = th.as_tensor(obs)
a_exp = th.as_tensor(act)

with th.no_grad():
    a_pi, _ = m.actor.action_log_prob(o)
a = a_pi.clone().requires_grad_(True)
qs = m.critic(o, a)
# the actor maximises the MIN over the ensemble, or the SUM of the two stream minima
n = len(qs)
if getattr(m, "dual_gamma", None):
    half = n // 2
    q = (th.cat(qs[:half], dim=1).min(dim=1).values
         + th.cat(qs[half:], dim=1).min(dim=1).values)
    which = f"dual-gamma: min(first {half}) + min(last {n-half})"
else:
    q = th.cat(qs, dim=1).min(dim=1).values
    which = f"single: min over {n}"
q.sum().backward()
g = a.grad.detach()

dvec = a_exp - a_pi
cos = (g * dvec).sum(1) / (g.norm(dim=1) * dvec.norm(dim=1) + 1e-12)
cos = cos.numpy()

print(f"checkpoint: {args.checkpoint}")
print(f"objective the actor ascends -- {which}")
print(f"states: {len(obs)} from {'landed' if args.landed_only else 'all'} demo episodes")
print()
print(f"  cos(grad_a Q, a_expert - pi(s))")
print(f"    mean   {cos.mean():+.4f}")
print(f"    median {np.median(cos):+.4f}")
print(f"    p10    {np.percentile(cos, 10):+.4f}   p90 {np.percentile(cos, 90):+.4f}")
print(f"    fraction POSITIVE (pushed toward the expert): {float((cos > 0).mean()):.1%}")
print()
print(f"  |pi(s) - a_expert| mean {dvec.norm(dim=1).mean():.4f}  "
      f"(action box is [-1,1]^{a_exp.shape[1]})")
print(f"  |grad_a Q| mean {g.norm(dim=1).mean():.4f}")
print()
if cos.mean() > 0.05:
    print("  => the critic pushes the actor TOWARD the expert. The gradient is not the problem.")
elif cos.mean() < -0.05:
    print("  => the critic pushes the actor AWAY from the expert while its VALUES are")
    print("     calibrated. Further calibration cannot fix this; the objective is wrong.")
else:
    print("  => the gradient is essentially ORTHOGONAL to the expert direction: the critic")
    print("     carries no information about which way the 91% behaviour lies.")
