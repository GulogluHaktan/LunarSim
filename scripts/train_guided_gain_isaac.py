"""Train PPO to modulate the ZEM/ZEV controller's gains on the REAL Isaac env.

The Isaac counterpart of `scripts/train_guided_gain.py`. Read
`lunarsim/rl/guided_gain_env.py` for why the action space is the controller's
gains, and `lunarsim/rl/guided_gain_vec_env.py` for the three things the
vectorized Isaac port has to get right (belly altitude, pad-relative
coordinates, mid-hold termination).

WHY THIS RUNS ON ISAAC AND NOT ON THE FAST BACKEND. The two backends disagree
about the task by more than they agree, so a result on one is not a result on
the other:

    stage            controller on analytic     controller on Isaac
    orbit_descent          89.6%                  ~50% (handover.md)
    land_handoff            6.2%                  the figure in handover.md
                                                  came from a harness with a
                                                  3.52 m altitude bug

The goal is stated on `orbit_descent` -- 50% -> 75% -- and only Isaac has that
headroom. The analytic backend's job here was to establish that the mechanism
learns at all, and it did: on `land_handoff` it took 6.2% -> 14.6% against a
measured state-independent gain-space ceiling of 20.8%
(`scripts/diag_gain_space_ceiling.py`).

WHY NOT land_handoff ON ISAAC EITHER. That same ceiling measurement found NO
gain vector lands on 79.2% of `land_handoff`'s episodes. Its release draw is
8-25 m with vz in [-5, -1] and up to 21.1 deg of tilt, while the RCS needs
~8.7 s to slew to the 20 deg final-approach cap -- and an entry at 8 m and
-5 m/s has 1.6 s of flight. Most of that stage is attitude-infeasible for this
vehicle, so the "88 points of headroom" it appears to offer is about 15.

MEASUREMENT PROTOCOL, declared before the numbers exist, following
`scripts/run_highpower_validation.sh`: the checkpoint is selected on seed block
7000 ALONE and scored afterwards on fresh blocks. `--eval-seed0` is therefore
the selection block and nothing else; the number this script prints cannot also
be the number that is believed. The zero-action baseline is measured through
this same env on the same seeds rather than quoted, because two figures in
handover.md turned out to be harness artifacts rather than physics.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--stage", type=str, default="orbit_descent")
parser.add_argument("--gain-set", type=str, default="lateral")
parser.add_argument("--hold-steps", type=int, default=20)
parser.add_argument("--residual-scale", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=120_000,
                    help="PPO timesteps in DECISIONS (each is --hold-steps "
                         "control steps)")
parser.add_argument("--n-envs", type=int, default=12)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.98)
parser.add_argument("--log-std-init", type=float, default=-1.0)
parser.add_argument("--ent-coef", type=float, default=0.002)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--n-steps", type=int, default=32)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--n-epochs", type=int, default=10)
parser.add_argument("--target-kl", type=float, default=0.05)
parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80)
parser.add_argument("--max-rocks-per-env", type=int, default=0)
parser.add_argument("--eval-episodes", type=int, default=48)
parser.add_argument("--eval-seed0", type=int, default=7000,
                    help="the SELECTION block; never report a score from it")
parser.add_argument("--eval-every", type=int, default=15_000)
parser.add_argument("--out", type=str, default="out/gg_isaac")
parser.add_argument("--torch-device", type=str, default="cuda")
parser.add_argument("--reward-weight", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="override a RewardWeights field for the whole run")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})
sys.path.insert(0, args.lunarsim_root)

from stable_baselines3 import PPO  # noqa: E402

from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS  # noqa: E402
from lunarsim.rl.guided_gain_vec_env import GuidedGainVecEnv  # noqa: E402
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides  # noqa: E402


def evaluate(gvenv, model, episodes: int, seed0: int, max_decisions: int) -> dict:
    """Deterministic rollouts, scored one episode at a time per env slot.

    Episode seeds are consumed in order across the env slots, so the SET of
    episodes is the same block `diag_stage_landing_rate.py` would score even
    though they are flown in parallel. `model=None` is the zero-action arm,
    i.e. the controller itself through this same env -- the baseline has to come
    from the same harness as the candidate or the comparison measures the
    harness.
    """
    n = gvenv.num_envs
    n_act = int(gvenv.action_space.shape[0])
    landed = 0
    done_count = 0
    rets: list[float] = []

    # `IsaacLanderVecEnv` draws its own spawn from its internal rng, so an
    # explicit per-episode seed is not available; what IS controlled is the
    # terrain seed and the episode count, which is what the block means here.
    obs = gvenv.reset()
    live = np.zeros(n, dtype=np.float64)
    while done_count < episodes:
        if model is None:
            act = np.zeros((n, n_act), dtype=np.float32)
        else:
            act, _ = model.predict(obs, deterministic=True)
        obs, rew, dones, infos = gvenv.step(act)
        live += rew
        for i in range(n):
            if not dones[i]:
                continue
            if done_count < episodes:
                landed += bool(infos[i].get("landed_safely", False))
                rets.append(float(live[i]))
                done_count += 1
            live[i] = 0.0
    return {"landed": landed, "episodes": done_count,
            "mean_return": float(np.mean(rets)) if rets else float("nan")}


def main() -> None:
    stage = STAGES_BY_NAME[args.stage]
    gain_spec = GAIN_SETS[args.gain_set]
    os.makedirs(args.out, exist_ok=True)

    dt = stage.params.dt_s
    n_dec = max(1, int(stage.params.max_episode_s / dt / args.hold_steps))
    gl = args.gamma * args.gae_lambda
    print(f"[stage] {args.stage} on ISAAC  hold={args.hold_steps} "
          f"({args.hold_steps * dt:.2f} s/decision)  "
          f"~{n_dec} decisions/episode", flush=True)
    print(f"[gains] {args.gain_set}: {[k.name for k in gain_spec]}", flush=True)
    print(f"[horizon] gamma={args.gamma} lambda={args.gae_lambda} -> advantage "
          f"horizon {1.0 / (1.0 - gl):.1f} decisions; terminal keeps "
          f"{gl ** n_dec:.3f} of its weight (previous runs: 0.00037)", flush=True)

    # Env spacing is sized off the LARGEST stage's tile, as the existing
    # trainer does, so a tile cannot overlap a neighbouring env slot.
    max_tile = max(s.tile_size_m for s in STAGES)
    venv = IsaacLanderVecEnv(
        num_envs=args.n_envs,
        tile_fn=make_tile_fn(stage, args.terrain_grid_n),
        params=stage.params,
        env_spacing_m=max_tile * 1.5,
        seed=args.terrain_seed,
        max_rocks_per_env=args.max_rocks_per_env,
        reward_fn=make_apollo_reward_fn(weights_from_overrides(args.reward_weight)),
    )
    gvenv = GuidedGainVecEnv(venv, gain_spec=gain_spec,
                             hold_steps=args.hold_steps,
                             residual_scale=args.residual_scale)

    print(f"\n[baseline] zero action = the controller, through THIS env, "
          f"{args.eval_episodes} episodes", flush=True)
    base = evaluate(gvenv, None, args.eval_episodes, args.eval_seed0, n_dec + 2)
    base_rate = base["landed"] / max(base["episodes"], 1)
    print(f"[baseline] {base['landed']}/{base['episodes']} landed "
          f"({base_rate:.1%}), mean return {base['mean_return']:.2f}", flush=True)

    model = PPO(
        "MlpPolicy", gvenv,
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
        device=args.torch_device,
    )
    # Assert on the OBJECT. Nine mechanism flags were silently dropped in this
    # project by code that printed the argument it had just passed.
    got = float(model.policy.log_std.detach().cpu().numpy().mean())
    assert abs(got - args.log_std_init) < 1e-6, \
        f"log_std_init did not reach the policy: asked {args.log_std_init}, got {got}"
    assert abs(model.gae_lambda - args.gae_lambda) < 1e-12, "gae_lambda did not reach PPO"
    assert abs(model.ent_coef - args.ent_coef) < 1e-12, "ent_coef did not reach PPO"
    print(f"[ppo] sigma={np.round(np.exp(model.policy.log_std.detach().cpu().numpy()), 4)}"
          f"  rollout={args.n_steps * args.n_envs} decisions/update", flush=True)

    best = -1
    rows = []
    done = 0
    while done < args.steps:
        chunk = min(args.eval_every, args.steps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
        ev = evaluate(gvenv, model, args.eval_episodes, args.eval_seed0, n_dec + 2)
        sigma = np.exp(model.policy.log_std.detach().cpu().numpy())
        rate = ev["landed"] / max(ev["episodes"], 1)
        print(f"[eval] {done:>8} dec  {ev['landed']:>3}/{ev['episodes']} "
              f"({rate:>6.1%})  base {base['landed']}/{base['episodes']} "
              f"({base_rate:.1%})  ret {ev['mean_return']:>8.2f}  "
              f"sigma~{sigma.mean():.3f}", flush=True)
        rows.append((done, ev["landed"], ev["episodes"], ev["mean_return"],
                     float(sigma.mean())))
        model.save(os.path.join(args.out, f"snap_{done}"))
        if ev["landed"] > best:
            best = ev["landed"]
            model.save(os.path.join(args.out, "best_on_selection_block"))

    print(f"\n[done] best on the SELECTION block: {best}/{args.eval_episodes} "
          f"vs controller {base['landed']}/{base['episodes']} ({base_rate:.1%})")
    print("[done] the selection block chose the checkpoint, so it cannot score "
          "it. Run scripts/run_highpower_validation.sh's fresh blocks next.")
    with open(os.path.join(args.out, "curve.csv"), "w") as fh:
        fh.write("decisions,landed,episodes,mean_return,mean_sigma\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


main()
simulation_app.close()
