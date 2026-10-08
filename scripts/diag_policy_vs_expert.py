"""How far is a policy from the teacher on HELD-OUT expert states, by altitude band?

The number that matters is the held-out one, and this project measured the training-set
one for a long time. The orbit clone's action error reads 0.1009 on the states it was
fitted on and 0.4757 on expert states from seed blocks it never saw -- a factor of 4.7.
The first number says the clone imitates well; the second says it does not, and the second
is the one that predicts the landing rate.

MEASURED, |pi_det(s) - a_expert| in L2 over a [-1,1]^4 action box, on 6000 held-out
expert states from landed episodes:

    checkpoint                 overall    0-2 m    80+ m
    bc_orbit_fixed  (48 landed eps)  0.4757   0.6305   0.3026   -> lands 19.8%
    v77@310k  (RL from it)           0.4737   0.6405   0.3019   -> lands 27.1%
    bc_orbit_big   (255 landed eps)  0.0590   0.0654   0.0449   -> lands 51.0%

Two things come out of that table. RL moved the policy by 0.002 in action space and gained
7.3 points of landing rate, which is only possible because competence here sits on a ~96%
saturated policy -- a tiny action change is a large behavioural one. And the clone's
8x-closer imitation is what took it from 19.8% to 51.0%, so the imitation gap was a
coverage problem, as the learning curve said.

It also bounds what is left: the observation's own floor, from the conditional action
spread among the closest 10% of observation matches, is about 0.047 in L2. The new clone
is at 0.059, within ~25% of a conservatively estimated floor -- so the large imitation win
is already banked and further demonstrations buy much less than the first 480 did. The
remaining distance to the target has to come from RL exceeding the teacher.

Run:
    .venv/bin/python scripts/diag_policy_vs_expert.py \
        --checkpoint out/bc_orbit_big.zip --checkpoint out/sac_v78_altreplay/... \
        --demos out/zemzev_orbit_demos_s20000.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch as th

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", action="append", required=True)
p.add_argument("--demos", action="append", default=None)
p.add_argument("--samples", type=int, default=6000)
p.add_argument("--vs", action="append", default=None,
               help="also report the distance BETWEEN policies: each --checkpoint "
                    "against this one. Distance to the expert cannot answer 'how far did "
                    "RL move from its warm start' -- two policies can sit equally far "
                    "from the expert in opposite directions, and reading it that way led "
                    "to the claim that v77 moved 0.002 when it had moved 0.078.")
args = p.parse_args()

from lunarsim.rl.eval_loader import load_with_critic_for_diag  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402

demos = args.demos or ["out/zemzev_orbit_demos_s20000.npz"]
ALT = float(OBS_SCALE[2])

O, A = [], []
for f in demos:
    d = np.load(f, allow_pickle=True)
    # Demos store ALREADY-NORMALISED observations; normalising again would feed the actor
    # states 200x too small (it invalidated three measurements before this was noticed).
    obs, act = d["obs"].astype(np.float32), d["actions"].astype(np.float32)
    ends, landed = d["episode_ends"], d["landed"].astype(bool)
    starts = np.concatenate([[0], ends[:-1]])
    for s, e, ok in zip(starts, ends, landed):
        if ok:
            O.append(obs[s:e])
            A.append(act[s:e])
obs, act = np.concatenate(O), np.concatenate(A)
rng = np.random.default_rng(0)
idx = rng.choice(len(obs), min(args.samples, len(obs)), replace=False)
o = th.as_tensor(obs[idx])
a_exp = th.as_tensor(act[idx])
alt = obs[idx][:, 2] * ALT

print(f"|pi_det(s) - a_expert| on {len(idx)} expert states from {len(demos)} demo file(s)")
print(f"(action box is [-1,1]^4; the observation's own floor is ~0.047)\n")
bands = ((0, 2), (2, 5), (5, 20), (20, 80), (80, 1e9))
hdr = "".join(f"{(f'{lo}-{hi:g}m' if hi < 1e9 else f'{lo}+m'):>9}" for lo, hi in bands)
print(f"  {'checkpoint':<34}{'overall':>9}{hdr}")
acts = {}
for ck in args.checkpoint:
    m, _ = load_with_critic_for_diag(ck, "cpu")
    with th.no_grad():
        a = m.policy.actor(o, deterministic=True)
    dist = th.norm(a - a_exp, dim=1).numpy()
    row = "".join(f"{dist[(alt >= lo) & (alt < hi)].mean():>9.4f}"
                  if ((alt >= lo) & (alt < hi)).sum() >= 20 else f"{'-':>9}"
                  for lo, hi in bands)
    name = ck.replace("out/", "").replace("/snapshots/orbit_descent", "@")
    print(f"  {name[-34:]:<34}{dist.mean():>9.4f}{row}")
    acts[ck] = a

if args.vs:
    print(f"\n  distance BETWEEN policies -- how far a run moved from its warm start")
    print(f"  {'pair':<34}{'overall':>9}{hdr}")
    for ref in args.vs:
        m, _ = load_with_critic_for_diag(ref, "cpu")
        with th.no_grad():
            a_ref = m.policy.actor(o, deterministic=True)
        for ck, a in acts.items():
            if ck == ref:
                continue
            dist = th.norm(a - a_ref, dim=1).numpy()
            row = "".join(f"{dist[(alt >= lo) & (alt < hi)].mean():>9.4f}"
                          if ((alt >= lo) & (alt < hi)).sum() >= 20 else f"{'-':>9}"
                          for lo, hi in bands)
            short = lambda q: q.replace("out/", "").replace(
                "/snapshots/orbit_descent", "@").replace("_steps.zip", "")[-15:]
            print(f"  {short(ref) + ' -> ' + short(ck):<34}{dist.mean():>9.4f}{row}")
