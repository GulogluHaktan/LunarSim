"""Can the critic RANK actions the policy would actually take?

This is a different and weaker question than the one the action gradient answers, and it
is the only one a sampling-based improvement operator needs. Best-of-N -- sample N actions
from the policy, keep the one the critic scores highest -- never leaves the policy's own
support, so it cannot drift out of distribution, and it does not use grad_a Q at all. It
needs only that Q order in-distribution actions better than chance.

Measured against the expert's action as the reference: for each state, sample N actions
from the policy and compare
    |a_Qbest - a_expert|     the action best-of-N would take
    |a_random - a_expert|    the action the policy takes on its own (mean over samples)
    |a_oracle - a_expert|    the closest of the N, i.e. what a perfect ranker could reach
If the Q-selected action is no closer than a random sample, the critic cannot rank, and
best-of-N would add nothing. If it sits between random and oracle, it can.

The expert is a reference, not the target: the aim is to EXCEED the teacher, and a critic
that ranks toward the teacher is being asked for the one thing we can check cheaply. A
critic that ranked away from the teacher but toward higher true value would show up here
as no better than random, so this test can only under-credit, never over-credit.

Run:
    .venv/bin/python scripts/diag_best_of_n.py \
        --checkpoint out/sac_v78_altreplay/snapshots/orbit_descent_410000_steps.zip
"""
from __future__ import annotations

import argparse

import numpy as np
import torch as th

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", action="append", required=True)
p.add_argument("--demos", default="out/zemzev_orbit_demos_s20000.npz")
p.add_argument("--samples", type=int, default=2000, help="states")
p.add_argument("--n", type=int, default=16, help="actions sampled per state")
args = p.parse_args()

from lunarsim.rl.eval_loader import load_with_critic_for_diag  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402

d = np.load(args.demos, allow_pickle=True)
obs, act = d["obs"].astype(np.float32), d["actions"].astype(np.float32)
ends, landed = d["episode_ends"], d["landed"].astype(bool)
starts = np.concatenate([[0], ends[:-1]])
keep = np.zeros(len(obs), bool)
for s, e, ok in zip(starts, ends, landed):
    if ok:
        keep[s:e] = True
obs, act = obs[keep], act[keep]
rng = np.random.default_rng(0)
idx = rng.choice(len(obs), min(args.samples, len(obs)), replace=False)
o = th.as_tensor(obs[idx])
a_exp = th.as_tensor(act[idx])
alt = obs[idx][:, 2] * float(OBS_SCALE[2])
N = args.n

print(f"best-of-{N} on {len(idx)} held-out expert states\n")
print(f"  {'checkpoint':<30} {'random':>8} {'Q-best':>8} {'oracle':>8} "
      f"{'Q gain':>8} {'of oracle':>10}")
for ck in args.checkpoint:
    m, dual = load_with_critic_for_diag(ck, "cpu")
    with th.no_grad():
        # N stochastic samples per state, from the policy's own distribution
        o_rep = o.repeat_interleave(N, dim=0)
        a_s, _ = m.policy.actor.action_log_prob(o_rep)
        qs = m.critic(o_rep, a_s)
        if dual:
            h = len(qs) // 2
            q = (th.min(th.cat(qs[:h], dim=1), dim=1).values
                 + th.min(th.cat(qs[h:], dim=1), dim=1).values)
        else:
            q = th.min(th.cat(qs, dim=1), dim=1).values
        q = q.reshape(len(idx), N)
        dist = th.norm(a_s - a_exp.repeat_interleave(N, dim=0), dim=1).reshape(len(idx), N)
        pick = q.argmax(dim=1)
        d_q = dist.gather(1, pick.unsqueeze(1)).squeeze(1)
        d_rand = dist.mean(dim=1)
        d_oracle = dist.min(dim=1).values
    gain = float((d_rand - d_q).mean())
    head = float((d_rand - d_oracle).mean())
    name = ck.replace("out/", "").replace("/snapshots/orbit_descent", "@").replace(
        "_steps.zip", "")
    print(f"  {name[-30:]:<30} {float(d_rand.mean()):>8.4f} {float(d_q.mean()):>8.4f} "
          f"{float(d_oracle.mean()):>8.4f} {gain:>+8.4f} "
          f"{(gain / head if head > 1e-9 else 0):>9.1%}")
    lo = alt < 2
    print(f"  {'':<30} 0-2 m: random {float(d_rand[lo].mean()):.4f} -> Q-best "
          f"{float(d_q[lo].mean()):.4f} (oracle {float(d_oracle[lo].mean()):.4f})")
print("\n  'of oracle' is the fraction of the achievable improvement the critic's ranking")
print("  actually captures. 0% means it ranks no better than chance.")
