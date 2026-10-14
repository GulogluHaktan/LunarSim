"""Train PPO to modulate the ZEM/ZEV controller's gains on the analytic env.

This is the fast CPU path for `lunarsim/rl/guided_gain_env.py`. Read that
module's docstring first -- it carries the measurements and the citations that
chose this action space.

THE HYPERPARAMETERS HERE ARE DERIVED, NOT COPIED. Three of them are set
against specific defects measured in this project's previous PPO runs, and the
numbers are worth stating because each one was individually sufficient to
explain a null result.

1. `--gae-lambda 0.98`, not 0.95/0.99.
   GAE weights the reward at step t+l by (gamma*lambda)^l, so the advantage --
   the only thing the POLICY gradient sees -- has an effective horizon of
   1/(1-gamma*lambda). The previous runs set gamma=0.999 explicitly so the
   landing bonus would be visible at release (it keeps 0.999^714 = 49% of it),
   and then ran lambda=0.99, which gives a 91-step advantage horizon and keeps
   (0.999*0.99)^714 = 0.037% of that bonus. At the shipped default lambda=0.95
   it is 6e-15%. So gamma was chosen for the terminal and lambda threw it away;
   the policy was climbing the dense shaping alone, whose decisive 0-5 m band
   holds 2.0% of the reward mass.
   Here an episode is ~12 decisions on land_handoff and ~37 on orbit_descent,
   so at gamma=0.99, lambda=0.98 the advantage horizon is 1/(1-0.9702) = 33.6
   decisions and the terminal keeps (0.9702)^12 = 70% of its weight. That is
   the same quantity that used to be 0.00037.

2. `--log-std-init -1.0` (sigma = 0.368 on a [-1, 1] box), set explicitly.
   The previous runs inherited log_std from the behaviour clone's fit residual:
   sigma = [0.0221, 0.0493, 0.0490, 0.0037], i.e. 1.1-2.5% of the half-range
   and 0.19% in yaw. Measured across v86's whole run, mean sigma went
   0.0310608 -> 0.0310872 in 1.5M steps: the entropy bonus's gradient on
   log_std is a constant -ent_coef per dimension, which at 0.003 and lr=1e-5
   is orders of magnitude under the policy-gradient term, so it cannot widen
   anything. Wagenmaker et al. (arXiv:2512.16911) prove standard BC can fail
   the action-COVERAGE condition that is minimally necessary for RL
   fine-tuning; 1.2% of the box is not coverage. Here there is no clone to
   inherit from, so the scale is a declared choice.

3. Decisions, not control steps. `--hold-steps 20` = 1.0 s.
   Tallec, Blier & Ollivier (ICML 2019) Thm 2 makes the action-ranking signal
   O(dt), and their Thm 4 shows per-step i.i.d. exploration noise provably
   collapses to a deterministic policy as dt -> 0. Measured here: the attitude
   channel needs 6.3 s at full torque to reach 10 deg of tilt, against a 0.05 s
   control step -- a 125:1 ratio -- so one lateral braking manoeuvre is 250-380
   consecutive commands. The gains are held for a whole second while the
   controller's inner loop keeps running at 20 Hz, which is the
   zero-order-hold form Furfaro et al. (arXiv:2003.02182) used on this same
   guidance law and the action persistence Metelli et al. (ICML 2020)
   formalise.

WHY A FRESH POLICY AND NOT A WARM START. SB3 orthogonally initialises
`action_net` with gain 0.01, so a fresh policy's mean action is ~0 -- and in
this env a zero action reproduces the controller EXACTLY (verified: identical
return and outcome on 24 orbit_descent seeds). The jump-start property that
Jiang et al. (arXiv:2407.15083) needed an annealed guide horizon to get, and
that JSRL (arXiv:2204.02372) bounds the benefit of, is here by construction.
There is nothing to clone and no coverage to lose.

STAGE CHOICE. `land_handoff` is the default because it is where the headroom
is and where the failure lives: the controller lands 6.2% of 48 analytic
episodes (12.0% on Isaac), 12 decisions per episode, and 94 points of room.
It is also the stage whose five identical 10/96 readings across 1M steps the
handover calls "the strongest single constraint" -- so it is the stage where a
real gain is hardest to explain away. `orbit_descent` has the goal stated on
it but the analytic backend gives the controller 89.6% there, so improvement
is not measurable on this backend; measure that one on Isaac.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS, GuidedGainEnv  # noqa: E402
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides  # noqa: E402


def _rfn(overrides):
    """A reward_fn for `--reward-weight` overrides, or None for the shipped
    reward. Built from a TUPLE of strings so it can be passed explicitly into
    SubprocVecEnv workers: on Python 3.14 the default start method is
    forkserver, which does not inherit globals set in the parent after import,
    so a module-level override would silently fall back to the defaults in
    every training worker while the evaluator used the override -- the exact
    train/eval reward skew this project has paid for twice."""
    return make_apollo_reward_fn(weights_from_overrides(list(overrides))) if overrides else None


def make_env(stage, hold_steps: int, residual_scale: float, seed: int,
             terrain_seed: int | None, gain_spec, reward_overrides=()):
    def _init():
        if terrain_seed is None:
            # Fresh terrain every episode. This is what stops the policy
            # fitting one map's craters; `make_tile_fn` is the curriculum's own
            # helper so the tile geometry matches what the trainer uses.
            return GuidedGainEnv(tile_fn=make_tile_fn(stage), params=stage.params,
                                 reward_fn=_rfn(reward_overrides),
                                 hold_steps=hold_steps, gain_spec=gain_spec,
                                 residual_scale=residual_scale, seed=seed)
        from lunarsim.rl.curriculum import terrain_config
        return GuidedGainEnv(tile=generate_tile(terrain_config(stage, terrain_seed)),
                             params=stage.params, hold_steps=hold_steps,
                             reward_fn=_rfn(reward_overrides),
                             gain_spec=gain_spec, residual_scale=residual_scale,
                             seed=seed)
    return _init


def evaluate(model, stage, hold_steps: int, residual_scale: float,
             episodes: int, seed0: int, terrain_seed: int, gain_spec,
             deterministic: bool = True, reward_overrides=()) -> dict:
    """Deterministic rollouts on a FIXED seed block, the way
    `diag_stage_landing_rate.py` scores -- same terrain seed, consecutive
    episode seeds -- so a number here is comparable to one from there.
    """
    from lunarsim.rl.curriculum import terrain_config
    landed = touched = 0
    rets = []
    for i in range(episodes):
        seed = seed0 + i
        env = GuidedGainEnv(tile=generate_tile(terrain_config(stage, terrain_seed)),
                            params=stage.params, hold_steps=hold_steps,
                            reward_fn=_rfn(reward_overrides),
                            gain_spec=gain_spec, residual_scale=residual_scale)
        obs, _ = env.reset(seed=seed)
        total = 0.0
        info: dict = {}
        while True:
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(act)
            total += float(rew)
            if term or trunc:
                break
        landed += bool(info.get("landed_safely", False))
        touched += bool(info.get("landing_margins"))
        rets.append(total)
    return {"landed": landed, "episodes": episodes, "touched": touched,
            "mean_return": float(np.mean(rets))}


def baseline(stage, hold_steps: int, residual_scale: float, episodes: int,
             seed0: int, terrain_seed: int, gain_spec, reward_overrides=()) -> dict:
    """The zero-action arm: the controller itself, measured through the SAME
    env and the same seeds. This is the number any RL result has to beat, and
    measuring it here rather than quoting it removes the possibility that the
    two arms differ by anything but the policy.
    """
    class _Zero:
        def __init__(self, n):
            self.n = n

        def predict(self, obs, deterministic=True):
            return np.zeros(self.n, dtype=np.float32), None

    n = len(gain_spec) + (4 if residual_scale > 0.0 else 0)
    return evaluate(_Zero(n), stage, hold_steps, residual_scale, episodes,
                    seed0, terrain_seed, gain_spec, reward_overrides=reward_overrides)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="land_handoff")
    ap.add_argument("--steps", type=int, default=200_000,
                    help="PPO timesteps, counted in DECISIONS (each is "
                         "--hold-steps control steps)")
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--hold-steps", type=int, default=20)
    ap.add_argument("--reward-weight", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="override a RewardWeights field; reaches the training "
                         "workers, the evaluator and the baseline alike")
    ap.add_argument("--gain-set", default="lateral", choices=sorted(GAIN_SETS),
                    help="'lateral' is the orbit_descent failure mode; "
                         "'attitude' adds the inner-loop gains, which is what "
                         "land_handoff's tilted, already-rotating entry needs")
    ap.add_argument("--residual-scale", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.98)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--ent-coef", type=float, default=0.002)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-terrain-seed", type=int, default=None,
                    help="fix the training terrain; default is fresh terrain "
                         "every episode")
    ap.add_argument("--eval-episodes", type=int, default=48)
    ap.add_argument("--eval-seed0", type=int, default=7000)
    ap.add_argument("--eval-terrain-seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--out", default="out/guided_gain")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    stage = STAGES_BY_NAME[args.stage]
    gain_spec = GAIN_SETS[args.gain_set]
    os.makedirs(args.out, exist_ok=True)

    print(f"[stage] {args.stage}  hold={args.hold_steps} steps "
          f"({args.hold_steps * stage.params.dt_s:.2f} s/decision)  "
          f"budget {stage.params.max_episode_s:.0f} s = "
          f"~{int(stage.params.max_episode_s / stage.params.dt_s / args.hold_steps)} "
          f"decisions/episode")
    print(f"[gains] {args.gain_set}: {[k.name for k in gain_spec]}")
    if args.residual_scale > 0:
        print(f"[resid] +4 residual dims at scale {args.residual_scale}")

    # The advantage horizon, printed because it is the quantity that silently
    # invalidated the previous runs. 1/(1-gamma*lambda) in DECISIONS.
    gl = args.gamma * args.gae_lambda
    n_dec = max(1, int(stage.params.max_episode_s / stage.params.dt_s / args.hold_steps))
    print(f"[horizon] gamma={args.gamma} lambda={args.gae_lambda} -> advantage "
          f"horizon {1.0 / (1.0 - gl):.1f} decisions; terminal reward keeps "
          f"{gl ** n_dec:.3f} of its weight at a {n_dec}-decision release "
          f"(previous runs: 0.00037)")

    print(f"\n[baseline] zero action = the controller, {args.eval_episodes} "
          f"episodes, seed block {args.eval_seed0}, terrain {args.eval_terrain_seed}",
          flush=True)
    rw = tuple(args.reward_weight or ())
    if rw:
        print(f"[reward] overrides {list(rw)} -> training workers, evaluator and baseline",
              flush=True)
    base = baseline(stage, args.hold_steps, args.residual_scale,
                    args.eval_episodes, args.eval_seed0, args.eval_terrain_seed,
                    gain_spec, reward_overrides=rw)
    print(f"[baseline] {base['landed']}/{base['episodes']} landed "
          f"({base['landed'] / base['episodes']:.1%}), "
          f"{base['touched']} touched down, mean return {base['mean_return']:.2f}",
          flush=True)

    venv = SubprocVecEnv([
        make_env(stage, args.hold_steps, args.residual_scale, args.seed + i,
                 args.train_terrain_seed, gain_spec, rw)
        for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        policy_kwargs={"net_arch": list(args.net_arch),
                       "log_std_init": args.log_std_init},
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # Assert on the OBJECT, not on the argument -- the project's own rule, and
    # the reason nine mechanism flags were silently dropped here before.
    got = float(model.policy.log_std.detach().cpu().numpy().mean())
    assert abs(got - args.log_std_init) < 1e-6, \
        f"log_std_init did not reach the policy: asked {args.log_std_init}, got {got}"
    assert abs(model.gae_lambda - args.gae_lambda) < 1e-12, "gae_lambda did not reach PPO"
    assert abs(model.ent_coef - args.ent_coef) < 1e-12, "ent_coef did not reach PPO"
    print(f"[ppo] log_std={np.round(np.exp(model.policy.log_std.detach().cpu().numpy()), 4)} "
          f"(sigma on a [-1,1] box)  rollout={args.n_steps * args.n_envs} decisions/update",
          flush=True)

    best = -1
    rows = []
    done = 0
    while done < args.steps:
        chunk = min(args.eval_every, args.steps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    progress_bar=False)
        done += chunk
        ev = evaluate(model, stage, args.hold_steps, args.residual_scale,
                      args.eval_episodes, args.eval_seed0, args.eval_terrain_seed,
                      gain_spec, reward_overrides=rw)
        sigma = np.exp(model.policy.log_std.detach().cpu().numpy())
        rate = ev["landed"] / ev["episodes"]
        print(f"[eval] {done:>8} dec  {ev['landed']:>3}/{ev['episodes']} "
              f"({rate:>6.1%})  base {base['landed']}/{base['episodes']} "
              f"({base['landed'] / base['episodes']:.1%})  ret {ev['mean_return']:>8.2f}  "
              f"sigma {np.round(sigma, 3)}", flush=True)
        rows.append((done, ev["landed"], ev["mean_return"], float(sigma.mean())))
        model.save(os.path.join(args.out, f"snap_{done}"))
        if ev["landed"] > best:
            best = ev["landed"]
            model.save(os.path.join(args.out, "best_on_selection_block"))

    print(f"\n[done] best on the SELECTION block: {best}/{args.eval_episodes} "
          f"({best / args.eval_episodes:.1%}) vs baseline "
          f"{base['landed']}/{base['episodes']} "
          f"({base['landed'] / base['episodes']:.1%})")
    print("[done] that number chose the checkpoint, so it cannot also score it "
          "-- score `best_on_selection_block` on fresh blocks "
          "(scripts/run_highpower_validation.sh's protocol) before believing it.")
    with open(os.path.join(args.out, "curve.csv"), "w") as fh:
        fh.write("decisions,landed,mean_return,mean_sigma\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
