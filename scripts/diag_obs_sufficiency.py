"""Is the controller's action a function of the observation?

(The observation is whatever width the demo file carries -- 16 before the
2026-10-07 widening, 20 after it; this script reads `obs.shape[1]` and never
assumes.)

This bounds everything else. The controller lands 92% on ramp_35m; a behaviour
clone of it lands 50.5%; and no RL configuration tried so far exceeds the clone.
If the controller's action is NOT determined by what the policy can see, then no
memoryless policy can reproduce it, the 42-point imitation gap is a ceiling rather
than a bug, and every actor/critic fix is secondary.

Method. For each demo transition, find its nearest neighbours in observation space
among transitions from OTHER episodes, then compare the actions taken. If the
observation determines the action, near-identical observations must carry
near-identical actions.

The "other episodes" restriction is the whole point. Consecutive samples within one
trajectory are trivially similar in both observation and action, so including them
measures trajectory smoothness and would report sufficiency no matter what. This is
the same confound as evaluating a policy on the terrain it trained on.

Reported: the conditional action spread at matched observations against the
marginal spread. A ratio near 0 means the observation determines the action; a ratio
near 1 means knowing the observation tells you almost nothing about the action, and
the demonstrator is using information the policy does not receive.
"""
from __future__ import annotations

import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--demos", default="out/zemzev_ramp35_demos.npz")
p.add_argument("--k", type=int, default=5, help="neighbours per query")
p.add_argument("--queries", type=int, default=4000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

d = np.load(args.demos, allow_pickle=True)
obs = d["obs"].astype(np.float64)
act = d["actions"].astype(np.float64)
ends = d["episode_ends"]
starts = np.concatenate([[0], ends[:-1]])
ep_id = np.zeros(len(obs), dtype=np.int64)
for i, (s, e) in enumerate(zip(starts, ends)):
    ep_id[s:e] = i

print(f"{args.demos}: {len(obs)} transitions, {len(ends)} episodes, "
      f"obs dim {obs.shape[1]}, act dim {act.shape[1]}")

# z-score so no single observation dimension dominates the distance
mu, sd = obs.mean(0), obs.std(0)
sd[sd < 1e-9] = 1.0
z = (obs - mu) / sd
dead = int((d["obs"].std(0) < 1e-9).sum())
if dead:
    print(f"  note: {dead} observation dimensions are constant across the data")

rng = np.random.default_rng(args.seed)
qi = rng.choice(len(obs), size=min(args.queries, len(obs)), replace=False)

marg = act.std(0)
print(f"\nmarginal action std per dim: {np.round(marg, 4)}  (mean {marg.mean():.4f})")

obs_d, act_d = [], []
for i in qi:
    # mask out the query's OWN episode entirely
    same = ep_id == ep_id[i]
    dist = np.sqrt(((z - z[i]) ** 2).sum(1))
    dist[same] = np.inf
    nn = np.argpartition(dist, args.k)[: args.k]
    obs_d.append(dist[nn])
    act_d.append(np.abs(act[nn] - act[i]))

obs_d = np.concatenate(obs_d)
act_d = np.concatenate(act_d)          # (queries*k, act_dim)

# |a_i - a_j| for two independent draws of the same conditional has
# E|a_i - a_j| = 2*sigma/sqrt(pi) for a Gaussian, so sigma_cond ~ mean/1.128
cond = act_d.mean(0) / 1.128
print(f"\nnearest-neighbour observation distance (z-units): "
      f"median {np.median(obs_d):.4f}  p10 {np.percentile(obs_d, 10):.4f}  "
      f"p90 {np.percentile(obs_d, 90):.4f}")
print(f"conditional action std at matched obs: {np.round(cond, 4)}")
print(f"ratio conditional/marginal per dim:    {np.round(cond / marg, 3)}")
print(f"\n  MEAN RATIO = {float((cond / marg).mean()):.3f}")
print("  ~0.0 -> the observation determines the action (BC can in principle match)")
print("  ~1.0 -> the observation says almost nothing (hard ceiling for any")
print("          memoryless policy, and the imitation gap is not a bug)")

# A tighter read: restrict to the CLOSEST matches, where any residual
# disagreement cannot be blamed on the neighbour being far away.
tight = obs_d < np.percentile(obs_d, 10)
if tight.sum() > 50:
    c2 = act_d[tight].mean(0) / 1.128
    print(f"\nrestricted to the closest 10% of matches (n={int(tight.sum())}):")
    print(f"  conditional action std: {np.round(c2, 4)}")
    print(f"  ratio to marginal:      {np.round(c2 / marg, 3)}   "
          f"MEAN {float((c2 / marg).mean()):.3f}")
