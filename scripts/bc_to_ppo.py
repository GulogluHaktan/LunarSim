"""Fit a PPO policy to the controller's demonstrations, so PPO can start competent.

WHY THIS EXISTS. Three things were measured on orbit_descent, and this is the only
combination that addresses all three on the stage the goal is defined on:

  1. An untrained policy cannot sample the terminal reward there at all. Thrust-to-weight at
     the centre of the action box is 1.0159, so an untrained actor's mean action IS a hover:
     0/20 touchdowns, median minimum altitude 200.0 m over sixty seconds. So PPO from scratch
     on this stage is futile, and that was measured before a run was wasted on it a second
     time. A behaviour-cloned start removes it -- the policy already descends.
  2. The critic's action GRADIENT is uninformative where it matters: 40.8% of states point
     toward the expert against 47.6% for an untrained critic, and in the decisive 0-2 m band
     it is 54.3% while being fifteen times stronger than at altitude. Every SAC-side
     mechanism that trusts grad_a Q failed on it (--bc-anchor, --cql-alpha, --td3bc-alpha),
     and so did the one that trusts its ranking (--pex-temperature: 22% -> 1%).
     PPO never differentiates its value with respect to an action.
  3. Q overestimated its own policy by +40 for an entire run. That comes from bootstrapping
     Q(s', a') onto the policy's own, possibly out-of-distribution action. PPO's V(s) has no
     action argument, so the mechanism is structurally absent.

WHAT IS DIFFERENT FROM THE SAC CLONE. SB3's SAC actor emits an unbounded Gaussian mean and
the environment sees tanh(u), so that clone is fitted to `atanh(action)`. PPO's Box policy
uses a DiagGaussianDistribution with NO squashing -- actions are clipped to the box instead --
so the target here is the raw recorded action. Fitting atanh into it would be wrong by
construction, which is the sort of silent mismatch this project has already been bitten by.

The VALUE head is deliberately left untrained: it has no supervision here (the demos carry
rewards but not on-policy returns for THIS policy), and PPO fits it on-policy from observed
returns, which is exactly the property that makes PPO attractive.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch as th
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from lunarsim.rl.curriculum import STAGES_BY_NAME  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--demos", required=True)
p.add_argument("--stage", required=True)
p.add_argument("--out", required=True)
p.add_argument("--net-arch", type=int, nargs="+", default=[512, 512])
p.add_argument("--epochs", type=int, default=1200)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--learning-rate", type=float, default=1e-3)
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--device", default="cpu")
p.add_argument("--include-failures", action="store_true")
args = p.parse_args()

if args.stage not in STAGES_BY_NAME:
    raise SystemExit(f"no stage named {args.stage!r}")

d = np.load(args.demos, allow_pickle=True)
obs, act = d["obs"].astype(np.float32), d["actions"].astype(np.float32)
if obs.shape[1] != len(OBS_SCALE):
    raise SystemExit(
        f"{args.demos} has {obs.shape[1]}-wide observations but the envs produce "
        f"{len(OBS_SCALE)}. Re-record the demos; a clone fitted to a stale layout trains and "
        f"saves without complaint and is wrong at every step.")
if not args.include_failures and "landed" in d:
    ends, landed = d["episode_ends"], d["landed"].astype(bool)
    starts = np.concatenate([[0], ends[:-1]])
    keep = np.zeros(len(obs), bool)
    for s, e, lg in zip(starts, ends, landed):
        if lg:
            keep[s:e] = True
    obs, act = obs[keep], act[keep]
    print(f"[bc-ppo] {args.demos}: {len(obs)} transitions from "
          f"{int(landed.sum())}/{len(landed)} episodes (landings only)", flush=True)

rng = np.random.default_rng(0)
perm = rng.permutation(len(obs))
nva = max(1, int(len(obs) * args.val_frac))
va, tr = perm[:nva], perm[nva:]
print(f"[bc-ppo] train {len(tr)}, val {len(va)}", flush=True)


class _Spaces(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (obs.shape[1],), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (act.shape[1],), np.float32)

    def reset(self, **kw):
        return np.zeros(obs.shape[1], np.float32), {}

    def step(self, a):
        return np.zeros(obs.shape[1], np.float32), 0.0, False, False, {}


th.manual_seed(0)
model = PPO("MlpPolicy", _Spaces(), device=args.device, verbose=0, seed=0,
            n_steps=64, batch_size=32,
            policy_kwargs={"net_arch": list(args.net_arch)})
pol = model.policy
print(f"[bc-ppo] net_arch={list(args.net_arch)}  dist={type(pol.action_dist).__name__}",
      flush=True)

xt = th.as_tensor(obs[tr], device=args.device)
xv = th.as_tensor(obs[va], device=args.device)
# RAW action, not atanh: PPO's Box policy does not squash.
tt = th.as_tensor(act[tr], device=args.device)
tv = th.as_tensor(act[va], device=args.device)

params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
opt = th.optim.Adam(params, lr=args.learning_rate)
loss_fn = nn.MSELoss()


def mean_action(x):
    return pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(x)))


best, best_state = float("inf"), None
for ep in range(args.epochs):
    idx = th.randperm(len(xt), device=args.device)
    for i in range(0, len(xt), args.batch_size):
        b = idx[i:i + args.batch_size]
        opt.zero_grad()
        loss_fn(mean_action(xt[b]), tt[b]).backward()
        opt.step()
    if ep % max(1, args.epochs // 12) == 0 or ep == args.epochs - 1:
        with th.no_grad():
            ltr = loss_fn(mean_action(xt), tt).item()
            lva = loss_fn(mean_action(xv), tv).item()
            mae = (mean_action(xv) - tv).abs().mean().item()
        print(f"[bc-ppo] epoch {ep:5d}  train {ltr:.5f}  val {lva:.5f}  mean|da| {mae:.4f}",
              flush=True)
        if lva < best:
            best = lva
            best_state = {k: v.detach().clone() for k, v in pol.state_dict().items()}

if best_state is not None:
    pol.load_state_dict(best_state)
    print(f"[bc-ppo] restored best val {best:.5f}", flush=True)

# log_std from the fit residual, so exploration starts at the scale of what the fit cannot
# explain rather than at SB3's default of 0 (std 1.0, which on this action box is enormous).
with th.no_grad():
    resid = (mean_action(xv) - tv).std(dim=0).clamp_min(1e-3)
    pol.log_std.data = resid.log()
print(f"[bc-ppo] log_std set from residual: std={np.round(resid.cpu().numpy(), 4)} "
      f"(SB3 default would be 1.0)", flush=True)

model.save(args.out)
print(f"[bc-ppo] saved PPO checkpoint -> {args.out}", flush=True)
