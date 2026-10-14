"""Does the policy's exploration produce any BEHAVIOURAL difference at all?

This is the measurement that decides between two explanations of why every RL
run on this project moved in action space and changed nothing:

  (A) handover.md's reading: the ESTIMATOR cannot rank actions -- a learned
      Q(s,a) sits under its own residual noise, so the gradient is noise.
  (B) the alternative: the POLICY never explored. The perturbations PPO
      actually samples are so small, so temporally uncorrelated, and applied
      at so fine a control step that they produce no measurable change in the
      outcome -- so there was no signal to estimate, noisy estimator or not.

(A) and (B) make opposite predictions about a quantity neither
`diag_action_signal_to_noise.py` nor `diag_onpolicy_noise.py` measures: the
spread of REALISED episode returns and outcomes induced by the exploration
noise itself. Under (A) that spread is large (the outcome responds to actions;
only the critic cannot see it). Under (B) it is ~0 for the noise PPO uses, and
grows when the noise is made coherent -- at the SAME magnitude.

So this script holds the policy fixed and varies only the EXPLORATION:

    white   sigma * N(0,1), independent every step  -- what PPO does now
    repeat  the same white draw held for k steps    -- action repeat
    ou      Ornstein-Uhlenbeck, correlation time tau -- coloured noise

and reports, per scheme: landing rate, mean return, and -- the column that
matters -- the WITHIN-EPISODE-SEED std of the return.

That last quantity has to be measured the right way or it says nothing. Across
a seed set, returns differ for two reasons: the terrain/spawn draw changed, and
the noise draw changed. Only the second is exploration. A policy gradient
cannot use the first: it is the same for every scheme and it is not something
the policy chose. So this script runs each episode seed `--draws` times with
DIFFERENT noise and pools the std computed WITHIN each seed. Terrain is held
fixed inside that std, so what is left is the perturbation's own effect -- the
numerator of the policy gradient's signal-to-noise ratio. A scheme whose
within-seed std is ~0 explored nothing, and no estimator could have recovered
a gradient from it.

Scale note: the clone's own log_std is [0.0221, 0.0493, 0.0490, 0.0037] on a
[-1, 1] box, and v86's is identical to four decimals after 1.5M steps at
ent_coef=0.003 and lr=1e-5. So `--sigma-scale 1` is not a hypothetical, it is
what 1.5M steps of PPO actually sampled from.

Backend: the CPU `AnalyticLanderEnv`, deliberately. The question is about the
VEHICLE's response to a perturbation, which this env models with the same
`apollo_lm` constants Isaac uses, and it lets the sweep run in minutes on one
core instead of hours of Isaac. Confirm the winning scheme on Isaac before
spending a training run on it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402


def noise_sequence(scheme: str, n: int, sigma: np.ndarray, k: int, tau: float,
                   rng: np.random.Generator) -> np.ndarray:
    """An (n, 4) perturbation sequence whose PER-STEP marginal std is `sigma`
    for every scheme, so the schemes differ only in their time correlation and
    never in magnitude. That is the whole point of the comparison: if a
    coherent sequence changes the outcome and an independent one of the same
    size does not, the missing ingredient is correlation, not scale.
    """
    dim = sigma.shape[0]
    if scheme == "none":
        return np.zeros((n, dim))
    if scheme == "white":
        return rng.normal(0.0, 1.0, size=(n, dim)) * sigma
    if scheme == "repeat":
        m = int(np.ceil(n / k))
        held = np.repeat(rng.normal(0.0, 1.0, size=(m, dim)), k, axis=0)[:n]
        return held * sigma
    if scheme == "ou":
        # dx = -x/tau dt + sqrt(2/tau) dW, stationary std 1 -> scaled by sigma.
        dt = 0.05
        a = np.exp(-dt / tau)
        s = np.sqrt(1.0 - a * a)
        x = rng.normal(0.0, 1.0, size=dim)
        out = np.empty((n, dim))
        for i in range(n):
            x = a * x + s * rng.normal(0.0, 1.0, size=dim)
            out[i] = x
        return out * sigma
    raise ValueError(f"unknown scheme {scheme!r}")


def run_episode(model, stage, seed: int, draw: int, scheme: str, sigma: np.ndarray,
                k: int, tau: float, max_steps: int) -> dict:
    """`seed` fixes the terrain and the spawn; `draw` fixes only the noise. The
    two are separate arguments so the caller can hold one and vary the other,
    which is the whole design of this measurement.
    """
    tile = generate_tile(terrain_config(stage, seed))
    env = AnalyticLanderEnv(tile=tile, params=stage.params)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(1_000_003 * (draw + 1) + seed)
    eps = noise_sequence(scheme, max_steps, sigma, k, tau, rng)

    total = 0.0
    steps = 0
    info: dict = {}
    for i in range(max_steps):
        act, _ = model.predict(obs, deterministic=True)
        act = np.clip(np.asarray(act, dtype=np.float64) + eps[i], -1.0, 1.0)
        obs, rew, term, trunc, info = env.step(act.astype(np.float32))
        total += float(rew)
        steps += 1
        if term or trunc:
            break
    # `landing_margins` is only populated on a real touchdown, so its presence
    # is the touched-down flag. Its entries are clipped to [0, 1], so a 0.0
    # cannot say by HOW much a limit was exceeded -- it is reported only as the
    # fraction of touchdowns whose lateral channel was the binding one, not as
    # an overshoot magnitude.
    m = info.get("landing_margins") or {}
    return {
        "return": total,
        "steps": steps,
        "landed": bool(info.get("landed_safely", False)),
        "touched": bool(m),
        "v_xy_binding": bool(m) and m.get("v_xy", 1.0) <= 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/bc_ppo_orbit_v2.zip")
    ap.add_argument("--stage", default="orbit_descent")
    ap.add_argument("--episodes", type=int, default=16,
                    help="terrain/spawn seeds")
    ap.add_argument("--draws", type=int, default=4,
                    help="independent noise draws PER seed; the within-seed std "
                         "is pooled over these, so >=3 is needed for it to mean "
                         "anything")
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--sigma-scale", type=float, nargs="+", default=[1.0, 4.0, 10.0],
                    help="multipliers on the checkpoint's own log_std")
    ap.add_argument("--repeat-k", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--ou-tau", type=float, nargs="+", default=[0.5, 2.0])
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.checkpoint, device="cpu")
    base_sigma = np.exp(model.policy.log_std.detach().cpu().numpy()).astype(np.float64)
    stage = STAGES_BY_NAME[args.stage]

    print(f"checkpoint     {args.checkpoint}")
    print(f"stage          {args.stage}   seeds {args.seed_base}.."
          f"{args.seed_base + args.episodes - 1} x {args.draws} noise draws "
          f"= {args.episodes * args.draws} episodes per scheme")
    print(f"policy log_std -> sigma {np.round(base_sigma, 5)}  (action box [-1, 1])")
    print()

    # (label, scheme, sigma multiplier, k, tau)
    runs: list[tuple[str, str, float, int, float]] = [("deterministic", "none", 0.0, 1, 1.0)]
    for m in args.sigma_scale:
        runs.append((f"white   x{m:g}", "white", m, 1, 1.0))
    for m in args.sigma_scale:
        for k in args.repeat_k:
            runs.append((f"repeat  x{m:g} k={k}", "repeat", m, k, 1.0))
    for m in args.sigma_scale:
        for tau in args.ou_tau:
            runs.append((f"ou      x{m:g} tau={tau:g}s", "ou", m, 1, tau))

    n_ep = args.episodes * args.draws
    print(f"{'scheme':<22} {'landed':>11} {'mean ret':>10} {'within-seed':>12} "
          f"{'vs white x1':>12} {'touched':>8} {'med steps':>9}")
    print("-" * 92)

    white_ref = None
    for label, scheme, mult, k, tau in runs:
        sigma = base_sigma * mult
        per_seed = []
        rets_all, landed, touched, steps_all = [], 0, 0, []
        for i in range(args.episodes):
            seed = args.seed_base + i
            n_draws = 1 if scheme == "none" else args.draws
            rows = [run_episode(model, stage, seed, d, scheme, sigma, k, tau,
                                args.max_steps) for d in range(n_draws)]
            r = np.array([x["return"] for x in rows])
            if n_draws >= 2:
                per_seed.append(r.std(ddof=1))
            rets_all.extend(r.tolist())
            landed += sum(x["landed"] for x in rows)
            touched += sum(x["touched"] for x in rows)
            steps_all.extend(x["steps"] for x in rows)
        rets = np.array(rets_all)
        within = float(np.mean(per_seed)) if per_seed else 0.0
        if scheme == "white" and mult == args.sigma_scale[0] and white_ref is None:
            white_ref = within or 1e-12
        rel = f"{within / white_ref:>11.1f}x" if white_ref else "          --"
        print(f"{label:<22} {landed:>4}/{len(rets):<6} {rets.mean():>10.1f} "
              f"{within:>12.3f} {rel} {touched:>4}/{len(rets):<3} "
              f"{int(np.median(steps_all)):>9}")

    print()
    print("READ IT LIKE THIS: 'within-seed' is the std of the return across noise")
    print("draws at a FIXED terrain and spawn, pooled over seeds. It is the spread")
    print("the exploration itself created, and nothing else -- the numerator of the")
    print("policy gradient's signal-to-noise ratio. The deterministic row is 0 by")
    print("construction (one draw, no noise). 'vs white x1' compares each scheme to")
    print("what PPO actually sampled for 1.5M steps. A scheme near white x1 explored")
    print("as little as v86 did, and no estimator could have recovered a gradient")
    print("from it; a scheme far above it created signal that was never available.")


if __name__ == "__main__":
    main()
