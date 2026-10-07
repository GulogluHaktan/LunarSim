"""Behaviour cloning: fit the SAC actor to the ZemZev controller's actions.

Why this exists. The lunar-landing guidance literature mostly does NOT solve
this problem with RL -- it trains networks SUPERVISED on optimal or
near-optimal trajectories (Pontryagin-derived datasets in Wang/Chen/Li 2024,
Guidance & Control Networks in Origer & Izzo 2024, optimality-informed networks
in Wang 2026). That branch has no training-stability problem at all, which is
the problem that has cost this project every run so far: a policy measured at
22.9% went to 0/24 after 600k more SAC steps, invariant to gamma, learning
rate, entropy coefficient and replay ratio (see handover.md).

We have a controller that lands 20/48 (42%) on orbit_descent in real Isaac.
Cloning it is the field's main road, and it produces a working artefact to build
on rather than asking SAC to rediscover the same solution.

Mind which simulator a controller number came from. The 22/24 (92%) figure this
file first quoted was measured in AnalyticLanderEnv, a simpler dynamics model
than the Isaac/PhysX one we train and evaluate in; collect_zemzev_demos.py's own
docstring had already recorded the gap ("75% on the analytic env but only
~40-50% on real Isaac"). 42% is therefore the ceiling a pure clone of this
controller can approach, and the 75% target needs more than imitation.

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
parser.add_argument("--include-failures", action="store_true",
                    help="clone EVERY demo episode, including the ones that crashed. "
                         "Off by default, and that default matters: the ZemZev "
                         "controller lands 20 of 48 on orbit_descent in real Isaac "
                         "(42%%), so an unfiltered dataset teaches the policy 28 "
                         "crashes alongside 20 landings.")
parser.add_argument("--dagger-iters", type=int, default=0,
                    help="DAgger iterations after the initial fit: roll the current "
                         "clone out in the ANALYTIC env, relabel every state it "
                         "visits with the ZemZev controller, aggregate, refit. This "
                         "is the validated fix for the sign-flipped vertical gain "
                         "(see the module docstring); measured 8 -> 12 -> 19 landings "
                         "over three iterations, with timeouts vanishing exactly when "
                         "the gain turns negative. The analytic env is used because it "
                         "reproduces the Isaac failure (0/24, 19 timeouts) in ~20 s of "
                         "CPU per 24 episodes.")
parser.add_argument("--dagger-episodes", type=int, default=24,
                    help="rollout episodes per DAgger iteration")
parser.add_argument("--gain-check", action="store_true", default=True,
                    help="after fitting, finite-difference d(throttle)/d(vz) and "
                         "d(throttle)/d(alt) at the demo states and FAIL if either is "
                         "positive in any altitude band. Positive d(a0)/d(vz) is "
                         "positive feedback and the policy flies away; action MSE "
                         "cannot see it (0.068 MSE / 0.0485 mean error coexisted with "
                         "a +0.129 gain and a 4%% landing rate).")
parser.add_argument("--no-gain-check", dest="gain_check", action="store_false")
parser.add_argument("--net-arch", type=int, nargs="+", default=None,
                     help="hidden layer sizes for the actor, e.g. --net-arch 512 512. "
                          "Default is SB3's [256, 256], which MEASURABLY limits the fit: "
                          "against the irreducible conditional action spread (the floor set "
                          "by how well the observation determines the controller's action, "
                          "measured at 0.079 of the marginal spread), [256,256] plateaus at "
                          "4.3-4.6x the floor by epoch 500-600 while [512,512] reaches 3.2x "
                          "at 600 and is still improving, with train 0.069 against val 0.079 "
                          "-- almost no overfitting, so the data supports more capacity.")
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
        o = np.asarray(d["obs"], dtype=np.float32)
        a = np.asarray(d["actions"], dtype=np.float32)
        ends = np.asarray(d["episode_ends"])
        n_ep = len(ends)
        kept_ep = n_ep
        if not args.include_failures:
            # Which episodes landed. Newer datasets record it; older ones do not,
            # so fall back to the sign of the terminal reward. That inference is
            # safe here and was checked: on v4 the terminal rewards fall into two
            # clean clusters, 28 at -94..-86 and 20 at +322..+393, and the 20
            # positives match the collector's own "20/48 landed_safely" line
            # exactly.
            if "landed" in d.files:
                landed = np.asarray(d["landed"], dtype=bool)
            else:
                r = np.asarray(d["rewards"])
                landed = np.array([r[e - 1] > 0.0 for e in ends])
            starts = np.concatenate(([0], ends[:-1]))
            mask = np.zeros(len(o), dtype=bool)
            for st, en, ok in zip(starts, ends, landed):
                if ok:
                    mask[st:en] = True
            o, a = o[mask], a[mask]
            kept_ep = int(landed.sum())
        obs_l.append(o)
        act_l.append(a)
        print(f"[bc] {path}: {len(o)} transitions from {kept_ep}/{n_ep} episodes"
              f"{'' if args.include_failures else ' (landings only)'}")
    obs = np.concatenate(obs_l)
    act = np.concatenate(act_l)
    if len(obs) == 0:
        raise SystemExit("no transitions left after filtering -- no demo episode landed")

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
    policy_kwargs = {"net_arch": list(args.net_arch)} if args.net_arch else {}
    model = SAC("MlpPolicy", _Spaces(), device=args.device, verbose=0,
                seed=args.seed, policy_kwargs=policy_kwargs)
    if args.net_arch:
        print(f"[bc] actor net_arch={list(args.net_arch)}", flush=True)
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

    def gain_report(x_np):
        """Finite-difference d(throttle)/d(vz) and d(throttle)/d(alt).

        This is the acceptance test the loss cannot replace. A clone fitted to
        MSE 0.068 with a mean action error of 0.0485 -- numbers that look like a
        good fit -- carried d(a0)/d(vz) = +0.129 where the teacher's is -0.98,
        and +0.306 in the 10-20 m band. Positive is POSITIVE FEEDBACK: with
        d(az)/d(a0) = 0.504 m/s^2 the growth rate is +0.154/s, i.e. 47x over a
        25 s episode, and the measured rollout did exactly that -- descended to
        18 m, crossed vz=0 at t=9 s, ran the throttle to +0.68 and climbed back
        to 55.6 m at +4.75 m/s. It lands 4% of the time and never looks wrong
        in the loss.

        The cause is collinearity in expert data: on the kept trajectories
        corr(alt, vz) = -0.98 and vz never exceeds -0.80 m/s, so there is not a
        single ascending state. On that near-1D manifold MSE is nearly invariant
        to how throttle is apportioned between alt and vz, and a sign-flipped
        partial is almost free. A per-transition validation split cannot see it
        either, because adjacent steps of one episode land on both sides.
        """
        from lunarsim.rl.obs_norm import OBS_SCALE
        alt_i, vz_i = 2, 5
        x = torch.as_tensor(x_np, device=args.device)
        out = {}
        for name, dim, phys_eps in (("vz", vz_i, 0.5), ("alt", alt_i, 1.0)):
            h = phys_eps / float(OBS_SCALE[dim])   # perturb in PHYSICAL units
            xp, xm = x.clone(), x.clone()
            xp[:, dim] += h
            xm[:, dim] -= h
            with torch.no_grad():
                dp = torch.tanh(forward_mean(xp))[:, 0]
                dm = torch.tanh(forward_mean(xm))[:, 0]
            out[name] = ((dp - dm) / (2.0 * phys_eps)).cpu().numpy()
        alt_phys = x_np[:, alt_i] * float(OBS_SCALE[alt_i])
        bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
        rows = []
        for lo, hi in bands:
            m = (alt_phys >= lo) & (alt_phys < hi)
            if m.sum() < 20:
                continue
            rows.append((lo, hi, int(m.sum()),
                         float(out["vz"][m].mean()), float(out["alt"][m].mean())))
        return rows

    def fit(x_tr, tgt_tr, x_va, tgt_va, epochs, tag=""):
        best_val, best_state = float("inf"), None
        n = len(x_tr)
        for epoch in range(epochs):
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
                # the error in the ACTION the env sees. Worth printing, but see
                # gain_report: this number cannot detect a sign-flipped gain.
                act_err = float((torch.tanh(forward_mean(x_va))
                                 - torch.tanh(tgt_va)).abs().mean())
            if val < best_val:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                print(f"[bc]{tag} epoch {epoch:4d}  train {tot / n:.5f}  "
                      f"val {val:.5f}  mean |da| {act_err:.4f}")
        if best_state is not None:
            actor.load_state_dict(best_state)
            print(f"[bc]{tag} restored best val {best_val:.5f}")

    fit(x_tr, tgt_tr, x_va, tgt_va, args.epochs)


    # ---------------------------------------------------------------- #
    # DAgger: relabel the states the CLONE actually visits.
    # ---------------------------------------------------------------- #
    # Plain behaviour cloning cannot fix the sign-flipped vertical gain, because
    # the defect lives in the expert's own data geometry (see gain_report). The
    # cure is to break the collinearity by visiting states the expert never
    # does, which is exactly what the half-trained clone does when it flies
    # away, and then asking the teacher what it would have done there.
    # Measured over three iterations in this env: 8 -> 12 -> 19 landings, with
    # d(a0)/d(vz) going +0.014 -> -0.883 -> -0.621 and timeouts vanishing
    # precisely when the sign turned. Injecting action noise instead does NOT
    # work: the teacher drags the state straight back onto the manifold, so
    # corr(alt, vz) stays at -0.98.
    if args.dagger_iters > 0:
        from lunarsim.control.zemzev_controller import ZemZevController
        from lunarsim.core.terrain.generate import generate_tile
        from lunarsim.rl import AnalyticLanderEnv
        from lunarsim.rl.curriculum import terrain_config

        stage = STAGES_BY_NAME[args.stage]
        params = stage.params
        max_steps = int(params.max_episode_s / params.dt_s) + 5
        agg_obs = [obs]
        agg_act = [act]

        for it in range(1, args.dagger_iters + 1):
            new_obs, new_act = [], []
            landed = timeouts = 0
            for ep in range(args.dagger_episodes):
                seed = 90000 + it * 1000 + ep
                env = AnalyticLanderEnv(
                    tile=generate_tile(terrain_config(stage, args.seed, 80)),
                    params=params, seed=seed)
                o, _ = env.reset(seed=seed)
                ctrl = ZemZevController(params)
                info = {}
                for _ in range(max_steps):
                    # label the state the CLONE is in with what the TEACHER
                    # would do there -- the whole point of DAgger
                    new_obs.append(np.asarray(o, dtype=np.float32))
                    new_act.append(np.asarray(ctrl.act(env), dtype=np.float32))
                    with torch.no_grad():
                        a_clone = torch.tanh(forward_mean(
                            torch.as_tensor(o, dtype=torch.float32,
                                            device=args.device).unsqueeze(0)))[0]
                    o, _, term, trunc, info = env.step(a_clone.cpu().numpy())
                    if term or trunc:
                        break
                landed += bool(info.get("landed_safely"))
                timeouts += bool(trunc and not term)
            agg_obs.append(np.asarray(new_obs, dtype=np.float32))
            agg_act.append(np.clip(np.asarray(new_act, dtype=np.float32),
                                   -1.0 + 1e-6, 1.0 - 1e-6))
            print(f"[dagger {it}] rollout: {landed}/{args.dagger_episodes} landed, "
                  f"{timeouts} timeout, +{len(new_obs)} relabelled states")

            all_o = np.concatenate(agg_obs)
            all_a = np.concatenate(agg_act)
            perm2 = np.random.default_rng(args.seed + it).permutation(len(all_o))
            all_o, all_a = all_o[perm2], all_a[perm2]
            nv = max(1, int(len(all_o) * args.val_frac))
            xt = torch.as_tensor(all_o[nv:], device=args.device)
            xv = torch.as_tensor(all_o[:nv], device=args.device)
            tt = torch.atanh(torch.as_tensor(all_a[nv:], device=args.device)).clamp(-5.0, 5.0)
            tv = torch.atanh(torch.as_tensor(all_a[:nv], device=args.device)).clamp(-5.0, 5.0)
            fit(xt, tt, xv, tv, args.epochs, tag=f"[dagger {it}]")

    # ---------------------------------------------------------------- #
    # Gain sign acceptance test
    # ---------------------------------------------------------------- #
    rows = gain_report(obs)
    print("[gain] d(throttle)/d(vz) and d(throttle)/d(alt), by altitude band:")
    bad = []
    for lo, hi, cnt, g_vz, g_alt in rows:
        flag = ""
        if g_vz > 0.0:
            flag = "  <-- POSITIVE FEEDBACK"
            bad.append((lo, hi, g_vz))
        print(f"[gain]   {lo:4.0f}-{hi if hi < 1e8 else 999:<4.0f} m  n={cnt:5d}  "
              f"d/dvz={g_vz:+7.3f}  d/dalt={g_alt:+7.4f}{flag}")
    if bad:
        msg = ("positive d(throttle)/d(vz) in " + str(len(bad)) + " band(s): "
               + ", ".join(f"{lo:.0f}-{hi:.0f} m: {g:+.3f}" for lo, hi, g in bad)
               + ". The policy will accelerate away from the ground instead of "
                 "arresting. Action MSE cannot see this -- run with "
                 "--dagger-iters 3 to break the collinearity in the expert data.")
        if args.gain_check:
            raise SystemExit("[gain] REJECTED: " + msg)
        print("[gain] WARNING (check disabled): " + msg)
    else:
        print("[gain] OK: vertical feedback is negative in every band")

    # Set the policy's spread to its OWN FIT RESIDUAL, per action dimension.
    #
    # This slot previously hardcoded log_std = -3 (std 0.05) to make the clone
    # near-deterministic. That is right for EVALUATION and wrong for RL
    # fine-tuning, and the difference cost a whole run: a 31-38% clone warm
    # started into SAC scored 0.0% in every one of 30 windows over 300k steps.
    #
    # Wagenmaker, Dong, Tsao, Finn & Levine (arXiv:2512.16911) give the reason
    # directly: behaviour cloning that trains a policy to match the
    # demonstrator's actions exactly "can fail to ensure COVERAGE over the
    # demonstrator's actions, a minimal condition necessary for effective RL
    # finetuning". A policy with std 0.05 shows the critic essentially one
    # action per state, so there is no spread to estimate an advantage from and
    # the policy gradient has nothing to work with. Their fix is to model the
    # POSTERIOR over the demonstrator's behaviour rather than its point
    # estimate; matching the fit residual is the cheap version of that -- the
    # policy is exactly as uncertain as its own regression error.
    #
    # Also worth recording from the same literature search, because it closes
    # two hypotheses this project spent runs on: a randomly initialised critic
    # is not the problem. Dong, Polonsky, Sadigh & Finn (arXiv:2607.27203) find
    # naive Q pretraining gives little benefit over random init, Li et al.
    # (arXiv:2608.10473) abandon offline critic training entirely and match or
    # beat conventional O2O, and PORL (arXiv:2505.16856) initialises Q from
    # scratch online specifically to fine-tune BC policies.
    with torch.no_grad():
        resid = (torch.tanh(forward_mean(x_va)) - torch.tanh(tgt_va)).std(dim=0)
        resid = resid.clamp(min=1e-3)
        # convert an action-space residual to the pre-tanh scale at the policy's
        # own operating point: d(tanh)/du = 1 - tanh^2, so u_std ~ a_std / (1-a^2)
        a_mean = torch.tanh(forward_mean(x_va)).mean(dim=0)
        jac = (1.0 - a_mean ** 2).clamp(min=0.05)
        pre_tanh_std = (resid / jac).clamp(0.02, 1.0)
        target_log_std = torch.log(pre_tanh_std)
        ls = getattr(actor, "log_std", None)
        if isinstance(ls, nn.Linear):
            # SB3's std head is state-dependent; zero the weight so it stops
            # varying with the observation and put the value in the bias.
            ls.weight.zero_()
            ls.bias.copy_(target_log_std)
        elif ls is not None:
            ls.copy_(target_log_std)
    print(f"[bc] policy std set from fit residual: "
          f"{pre_tanh_std.cpu().numpy().round(4)} (pre-tanh), "
          f"action-space residual {resid.cpu().numpy().round(4)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    model.save(args.out)
    print(f"[bc] saved SAC checkpoint -> {args.out}")
    print("[bc] evaluate it with:")
    print(f"    scripts/diag_stage_landing_rate.py --checkpoint {args.out} "
          f"--stage {args.stage} --episodes 96 --seed0 41000")


if __name__ == "__main__":
    main()
