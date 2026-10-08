"""Refit ONLY the value head of an existing PPO clone, at a different gamma.

The policy half of a BC-to-PPO clone takes about 40 minutes with DAgger; the value head
takes about one. When the discount changes, only the value head's target changes -- so
redoing the policy fit to change gamma wastes the expensive half and, worse, produces a
DIFFERENT policy, which would confound the comparison between two gamma settings.

Why gamma changed. With a single gamma = 0.99 the terminal reward's weight in the return
is 0.0054 at a 200 m release and 0.786 at touchdown, so PPO's advantage sees almost none
of it where it matters -- and the lateral speed that causes 94% of the failures is already
+5.86 m/s worse in the failing episodes at 100-200 m. Measured on 480 demo episodes, the
terminal's land-versus-crash difference at the release state against the return's own
noise at a state:

    gamma    return std   noise at a state   terminal signal   signal/noise
    0.990       2.27          0.272               0.004            0.01
    0.995       3.79          0.419               0.129            0.31
    0.999       7.63          0.693               2.260            3.26

The noise grows 2.5x and the signal 565x. And the divergence that stopped SAC from using
gamma 0.998 does not apply here: PPO's value is a regression onto the realised return, not
a bootstrap fixed point, so a per-step error is not amplified by 1/(1-gamma).

Run:
    /home/haktan/isaac-env/bin/python scripts/refit_ppo_value_head.py \
        --in out/bc_ppo_orbit_v2.zip --out out/bc_ppo_orbit_g999.zip --gamma 0.999 \
        --demos out/zemzev_orbit_demos_s20000.npz [--demos ...]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch as th
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
p.add_argument("--gamma", type=float, required=True)
p.add_argument("--demos", action="append", required=True)
p.add_argument("--epochs", type=int, default=600)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--learning-rate", type=float, default=1e-3)
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--device", default="cpu")
args = p.parse_args()

from stable_baselines3 import PPO  # noqa: E402

model = PPO.load(args.inp, device=args.device)
pol = model.policy

O, T = [], []
for f in args.demos:
    d = np.load(f, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    r = d["rewards"].astype(np.float64)
    ends = d["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    for s, e in zip(starts, ends):
        seg = r[s:e]
        g = np.zeros(len(seg))
        acc = 0.0
        for i in range(len(seg) - 1, -1, -1):
            acc = seg[i] + args.gamma * acc
            g[i] = acc
        O.append(obs[s:e])
        T.append(g)
X = np.concatenate(O).astype(np.float32)
Y = np.concatenate(T).astype(np.float32).reshape(-1, 1)
perm = np.random.default_rng(0).permutation(len(X))
X, Y = X[perm], Y[perm]
nv = max(1, int(len(X) * args.val_frac))
xt, yt = th.as_tensor(X[nv:]), th.as_tensor(Y[nv:])
xv, yv = th.as_tensor(X[:nv]), th.as_tensor(Y[:nv])
print(f"[refit] {len(xt)} train / {len(xv)} val transitions, gamma={args.gamma}, "
      f"target mean {float(yt.mean()):+.3f} std {float(yt.std()):.3f}", flush=True)


def value(x):
    return pol.value_net(pol.mlp_extractor.forward_critic(pol.extract_features(x)))


# The policy half is NOT touched, so the two gamma settings differ in exactly one thing.
before = {k: v.clone() for k, v in pol.state_dict().items()
          if k.startswith("action_net") or k.startswith("mlp_extractor.policy_net")}
opt = th.optim.Adam(list(pol.mlp_extractor.value_net.parameters())
                    + list(pol.value_net.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()
for ep in range(args.epochs):
    idx = th.randperm(len(xt))
    for i in range(0, len(xt), args.batch_size):
        b = idx[i:i + args.batch_size]
        opt.zero_grad()
        loss_fn(value(xt[b]), yt[b]).backward()
        opt.step()
    if ep % 100 == 0 or ep == args.epochs - 1:
        with th.no_grad():
            print(f"[refit] epoch {ep:>5}  train {float(loss_fn(value(xt), yt)):.5f}  "
                  f"val {float(loss_fn(value(xv), yv)):.5f}", flush=True)
with th.no_grad():
    r2 = 1.0 - float(((value(xv) - yv) ** 2).mean() / yv.var())
print(f"[refit] value head R^2 on held-out states: {r2:+.3f}")
for k, v in before.items():
    if not th.allclose(pol.state_dict()[k], v):
        raise SystemExit(f"the POLICY changed at {k}; only the value head may move, or "
                         f"the two gamma settings differ in more than gamma")
print("[refit] policy parameters verified unchanged")
model.save(args.out)
print(f"[refit] saved -> {args.out}")
