"""Behaviour cloning: fit the SAC actor to the ZemZev controller's actions.

Why this exists. The lunar-landing guidance literature mostly does NOT solve
this problem with RL -- it trains networks SUPERVISED on optimal or
near-optimal trajectories (Pontryagin-derived datasets in Wang/Chen/Li 2024,
Guidance & Control Networks in Origer & Izzo 2024, optimality-informed networks
in Wang 2026). That branch has no training-stability problem at all, which is
the problem that has cost this project every run so far: a policy measured at
22.9% went to 0/24 after 600k more SAC steps, invariant to gamma, learning
rate, entropy coefficient and replay ratio (see handover.md).

We have a controller that lands 22/24 on orbit_descent. Cloning it is the
field's main road, and it produces a working artefact to build on rather than
asking SAC to rediscover the same solution.

The output is a REAL SAC checkpoint (`.zip`), not a bare torch file, so
everything downstream -- diag_stage_landing_rate.py, --warm-start, the snapshot
machinery -- works on it unchanged. Only the actor is fitted; the critic stays
at init, which is correct for a starting point that will either be evaluated
directly or RL-finetuned (the finetune then learns a critic for a policy that
already lands, instead of for one that does not).

Run:
    .venv/bin/python scripts/train_bc_from_demos.py \
        --demos out/zemzev_orbit_descent_demos_v3.npz \
        --stage orbit_descent --epochs 200 \
        --out out/bc_orbit_descent.zip
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--demos", type=str, required=True, action="append",
                    help="a demo .npz from collect_zemzev_demos.py, repeatable")
parser.add_argument("--stage", type=str, required=True,
                    help="stage the demos were flown on; sets the action/obs spaces")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--val-frac", type=float, default=0.1)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def main():
    import gymnasium as gym
    from stable_baselines3 import SAC

    from lunarsim.rl.curriculum import STAGES_BY_NAME

    if args.stage not in STAGES_BY_NAME:
        raise SystemExit(f"no stage named {args.stage!r}")

    obs_l, act_l = [], []
    for path in args.demos:
        d = np.load(path)
        obs_l.append(np.asarray(d["obs"], dtype=np.float32))
        act_l.append(np.asarray(d["actions"], dtype=np.float32))
        print(f"[bc] {path}: {len(d['obs'])} transitions, "
              f"{len(d['episode_ends'])} episodes")
    obs = np.concatenate(obs_l)
    act = np.concatenate(act_l)

    # The demo actions were recorded in the env's action space, so they are
    # already in [-1, 1]; clipping guards a boundary value that would make
    # atanh infinite below.
    act = np.clip(act, -1.0 + 1e-6, 1.0 - 1e-6)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(obs))
    obs, act = obs[perm], act[perm]
    n_val = max(1, int(len(obs) * args.val_frac))
    obs_tr, act_tr = obs[n_val:], act[n_val:]
    obs_va, act_va = obs[:n_val], act[:n_val]
    print(f"[bc] train {len(obs_tr)}, val {len(obs_va)}")

    # A throwaway env only to give SAC the right spaces; SAC.load/predict and
    # every downstream script rebuild the env themselves.
    class _Spaces(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (obs.shape[1],), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (act.shape[1],), np.float32)

        def reset(self, **kw):
            return np.zeros(obs.shape[1], np.float32), {}

        def step(self, a):
            return np.zeros(obs.shape[1], np.float32), 0.0, False, False, {}

    torch.manual_seed(args.seed)
    model = SAC("MlpPolicy", _Spaces(), device=args.device, verbose=0,
                seed=args.seed)
    actor = model.policy.actor

    # Fit the PRE-TANH mean, not the squashed action. SAC's actor emits an
    # unbounded Gaussian mean `u` and the environment sees `tanh(u)`, so the
    # supervised target is `atanh(a)`. Regressing on the squashed output
    # instead would put the loss through a saturating nonlinearity and give
    # almost no gradient exactly where the controller commands extremes --
    # which, per handover.md, is most of the time (95% of actions sit at the
    # bounds).
    tgt_tr = torch.atanh(torch.as_tensor(act_tr, device=args.device))
    tgt_va = torch.atanh(torch.as_tensor(act_va, device=args.device))
    x_tr = torch.as_tensor(obs_tr, device=args.device)
    x_va = torch.as_tensor(obs_va, device=args.device)

    # atanh of a near-bound action is large; clamp so a handful of saturated
    # samples cannot dominate the regression.
    tgt_tr = tgt_tr.clamp(-5.0, 5.0)
    tgt_va = tgt_va.clamp(-5.0, 5.0)

    params = list(actor.parameters())
    opt = torch.optim.Adam(params, lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    def forward_mean(x):
        feats = actor.extract_features(x, actor.features_extractor)
        latent = actor.latent_pi(feats)
        return actor.mu(latent)

    best_val, best_state = float("inf"), None
    n = len(x_tr)
    for epoch in range(args.epochs):
        actor.train()
        idx = torch.randperm(n, device=args.device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            b = idx[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(forward_mean(x_tr[b]), tgt_tr[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        actor.eval()
        with torch.no_grad():
            val = float(loss_fn(forward_mean(x_va), tgt_va))
            # the number that actually matters: error in the ACTION the env
            # sees, which is what the landing criteria respond to
            act_err = float((torch.tanh(forward_mean(x_va))
                             - torch.tanh(tgt_va)).abs().mean())
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"[bc] epoch {epoch:4d}  train {tot / n:.5f}  val {val:.5f}  "
                  f"mean |da| {act_err:.4f}")

    if best_state is not None:
        actor.load_state_dict(best_state)
        print(f"[bc] restored best val {best_val:.5f}")

    # make the saved policy deterministic-ish: a cloned policy has no reason
    # to carry the wide exploration std SAC initialises, and a large pre-tanh
    # std would saturate tanh and throw away the fit (handover.md measured
    # std 4.5-5.8 producing 95% saturated actions).
    with torch.no_grad():
        ls = getattr(actor, "log_std", None)
        if isinstance(ls, nn.Linear):
            # SB3 uses a state-dependent std head here, not a bare parameter:
            # zero the weight so the std stops depending on the observation and
            # set the bias so it is small and constant.
            ls.weight.zero_()
            ls.bias.fill_(-3.0)
        elif ls is not None:
            ls.fill_(-3.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    model.save(args.out)
    print(f"[bc] saved SAC checkpoint -> {args.out}")
    print("[bc] evaluate it with:")
    print(f"    scripts/diag_stage_landing_rate.py --checkpoint {args.out} "
          f"--stage {args.stage} --episodes 96 --seed0 41000")


if __name__ == "__main__":
    main()
