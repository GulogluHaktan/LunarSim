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
p.add_argument("--dagger-iters", type=int, default=0,
               help="DAgger iterations. The PPO clone needs them for a MEASURED reason: fitted "
                    "on the demos alone it reached val MSE 0.00279 and a mean action error of "
                    "0.026 -- a good fit by every loss the regression can see -- and landed "
                    "3%% against the SAC clone's 22%%. Its causal gain d(throttle)/d(vz) is the "
                    "right sign and 4-10x too weak: -0.115 against -0.520 at 0-2 m, -0.015 "
                    "against -0.152 at 15-40 m. MSE cannot see a flattened gain when the "
                    "expert's data is collinear, which this project already learned on the SAC "
                    "side; DAgger breaks the collinearity by labelling states the CLONE visits.")
p.add_argument("--dagger-episodes", type=int, default=24)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--gain-check", action="store_true", default=True,
               help="reject a clone whose d(throttle)/d(vz) is positive in any altitude band. "
                    "Positive is POSITIVE FEEDBACK: the vehicle accelerates away from the "
                    "ground instead of arresting.")
p.add_argument("--no-gain-check", dest="gain_check", action="store_false")
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


def _fit(xt_, tt_, xv_, tv_, epochs, tag=""):
    global best, best_state
    best, best_state = float("inf"), None
    for e in range(epochs):
        idx = th.randperm(len(xt_), device=args.device)
        for i in range(0, len(xt_), args.batch_size):
            b = idx[i:i + args.batch_size]
            opt.zero_grad()
            loss_fn(mean_action(xt_[b]), tt_[b]).backward()
            opt.step()
        if e % max(1, epochs // 6) == 0 or e == epochs - 1:
            with th.no_grad():
                ltr = loss_fn(mean_action(xt_), tt_).item()
                lva = loss_fn(mean_action(xv_), tv_).item()
                mae = (mean_action(xv_) - tv_).abs().mean().item()
            print(f"[bc-ppo]{tag} epoch {e:5d}  train {ltr:.5f}  val {lva:.5f}  "
                  f"mean|da| {mae:.4f}", flush=True)
            if lva < best:
                best, best_state = lva, {k: v.detach().clone()
                                         for k, v in pol.state_dict().items()}
    if best_state is not None:
        pol.load_state_dict(best_state)


if args.dagger_iters > 0:
    # Same structure as train_bc_from_demos.py's DAgger, on the ANALYTIC env so it needs no
    # GPU, with two changes for PPO: the clone's action is the RAW distribution mean clipped
    # to the box (PPO does not squash), and the targets are raw recorded actions rather than
    # atanh. Getting either wrong would train a policy whose actions mean something different
    # from what the env receives.
    from lunarsim.control.zemzev_controller import ZemZevController
    from lunarsim.core.terrain.generate import generate_tile
    from lunarsim.rl import AnalyticLanderEnv
    from lunarsim.rl.curriculum import terrain_config

    stage = STAGES_BY_NAME[args.stage]
    sp = stage.params
    max_steps = int(sp.max_episode_s / sp.dt_s) + 5
    agg_o, agg_a = [obs], [act]

    for it in range(1, args.dagger_iters + 1):
        new_o, new_a = [], []
        landed = timeouts = 0
        for ep in range(args.dagger_episodes):
            seed = 90000 + it * 1000 + ep
            env = AnalyticLanderEnv(
                tile=generate_tile(terrain_config(stage, args.seed, 80)),
                params=sp, seed=seed)
            o, _ = env.reset(seed=seed)
            ctrl = ZemZevController(sp)
            info = {}
            for _ in range(max_steps):
                # label the state the CLONE reached with what the TEACHER would do there
                new_o.append(np.asarray(o, dtype=np.float32))
                new_a.append(np.asarray(ctrl.act(env), dtype=np.float32))
                with th.no_grad():
                    a = mean_action(th.as_tensor(o, dtype=th.float32,
                                                 device=args.device).unsqueeze(0))[0]
                o, _, term, trunc, info = env.step(
                    np.clip(a.cpu().numpy(), -1.0, 1.0))
                if term or trunc:
                    break
            landed += bool(info.get("landed_safely"))
            timeouts += bool(trunc and not term)
        agg_o.append(np.asarray(new_o, dtype=np.float32))
        agg_a.append(np.clip(np.asarray(new_a, dtype=np.float32), -1.0, 1.0))
        print(f"[dagger {it}] rollout: {landed}/{args.dagger_episodes} landed, "
              f"{timeouts} timeout, +{len(new_o)} relabelled states", flush=True)

        all_o = np.concatenate(agg_o)
        all_a = np.concatenate(agg_a)
        pm = np.random.default_rng(args.seed + it).permutation(len(all_o))
        all_o, all_a = all_o[pm], all_a[pm]
        nv = max(1, int(len(all_o) * args.val_frac))
        _fit(th.as_tensor(all_o[nv:], device=args.device),
             th.as_tensor(all_a[nv:], device=args.device),
             th.as_tensor(all_o[:nv], device=args.device),
             th.as_tensor(all_a[:nv], device=args.device),
             args.epochs, tag=f"[dagger {it}]")

if args.gain_check:
    # The acceptance test the loss cannot replace. A clone at val MSE 0.00279 with a mean
    # action error of 0.026 landed 3%, because its gain was flat -- so MSE is not evidence.
    from lunarsim.rl.obs_norm import OBS_SCALE as _OS
    alt_m = obs[:, 2] * _OS[2]
    bands = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 1e9)]
    rng2 = np.random.default_rng(0)
    bad = []
    print("[gain] d(throttle)/d(vz), by altitude band:", flush=True)
    for lo, hi in bands:
        m = (alt_m >= lo) & (alt_m < hi)
        if m.sum() < 100:
            continue
        idx = rng2.choice(np.where(m)[0], size=min(800, int(m.sum())), replace=False)
        h = 0.5 / _OS[5]
        op, om = obs[idx].copy(), obs[idx].copy()
        op[:, 5] += h
        om[:, 5] -= h
        with th.no_grad():
            ap = mean_action(th.as_tensor(op, device=args.device))[:, 0]
            am = mean_action(th.as_tensor(om, device=args.device))[:, 0]
        g = float(((ap - am) / (2 * 0.5)).mean().item())
        flag = "  <-- POSITIVE FEEDBACK" if g > 0 else ""
        print(f"[gain] {lo:7.0f}-{hi if hi < 1e8 else 999:<7.0f} m  n={int(m.sum()):6d}  "
              f"d/dvz={g:+.3f}{flag}", flush=True)
        if g > 0:
            bad.append((lo, hi, g))
    if bad:
        raise SystemExit(
            f"[gain] REJECTED: positive d(throttle)/d(vz) in {len(bad)} band(s): "
            + ", ".join(f"{lo:.0f}-{hi:.0f} m: {g:+.3f}" for lo, hi, g in bad)
            + ". The policy will accelerate away from the ground instead of arresting. "
              "Action MSE cannot see this -- run with --dagger-iters 3.")
    print("[gain] OK: vertical feedback is negative in every band", flush=True)

# log_std from the fit residual, so exploration starts at the scale of what the fit cannot
# explain rather than at SB3's default of 0 (std 1.0, which on this action box is enormous).
with th.no_grad():
    resid = (mean_action(xv) - tv).std(dim=0).clamp_min(1e-3)
    pol.log_std.data = resid.log()
print(f"[bc-ppo] log_std set from residual: std={np.round(resid.cpu().numpy(), 4)} "
      f"(SB3 default would be 1.0)", flush=True)

model.save(args.out)
print(f"[bc-ppo] saved PPO checkpoint -> {args.out}", flush=True)
