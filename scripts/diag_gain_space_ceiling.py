"""How much of the stage is reachable by modulating the controller's gains?

A flat RL curve has two very different causes, and training harder only helps
for one of them:

  (a) the policy cannot find the good gains, or
  (b) there are no good gains -- the stage's failures need a manoeuvre the
      ZEM/ZEV law cannot produce at ANY setting of the exposed knobs.

This separates them without training anything. For each episode seed it draws
`--samples` gain vectors, holds each one fixed for a whole episode, and asks
whether ANY of them lands. That number is the ceiling for a STATE-INDEPENDENT
gain policy, and therefore a lower bound on what a state-dependent one could
reach. Compared against the zero-action baseline it says how much room the
action space actually contains.

Read it like this:

  oracle ~= baseline      the gain space is the wrong lever for this stage; a
                          better optimiser cannot help. Change the action
                          space (residual, attitude setpoint) or the stage.
  oracle >> baseline      the room is real and the gap is an optimisation
                          problem; keep training, and the oracle's margin is
                          roughly what to expect.

The per-episode breakdown matters as much as the total: an episode no sample
lands is evidence of infeasibility at that entry condition, which on
`land_handoff` is a live possibility. Its release draw is alt 8-25 m with
vz in [-5, -1] and up to 21.1 deg of tilt (the spawn samples +-0.26 rad
INDEPENDENTLY per axis, so the diagonal reaches 0.26*sqrt(2)), while the RCS
needs ~8.7 s to slew to the 20 deg final-approach cap at 2.3 deg/s. An entry at
8 m and -5 m/s has 1.6 s of flight. Those two numbers do not both fit, so part
of this stage is attitude-infeasible no matter what flies it -- and a landing
rate measured against it is bounded by something that is not the policy.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS, GuidedGainEnv  # noqa: E402


def _episode(stage, gain_spec, hold_steps, seed, terrain_seed, action):
    env = GuidedGainEnv(tile=generate_tile(terrain_config(stage, terrain_seed)),
                        params=stage.params, hold_steps=hold_steps,
                        gain_spec=gain_spec)
    obs, _ = env.reset(seed=seed)
    info: dict = {}
    total = 0.0
    while True:
        obs, rew, term, trunc, info = env.step(action)
        total += float(rew)
        if term or trunc:
            break
    return bool(info.get("landed_safely", False)), total


def _one_seed(payload):
    stage_name, gain_set, hold_steps, seed, terrain_seed, samples, rng_seed = payload
    stage = STAGES_BY_NAME[stage_name]
    spec = GAIN_SETS[gain_set]
    n = len(spec)
    rng = np.random.default_rng(rng_seed)
    zero_ok, zero_ret = _episode(stage, spec, hold_steps, seed, terrain_seed,
                                 np.zeros(n, dtype=np.float32))
    hits = 0
    best = zero_ret
    for _ in range(samples):
        # Uniform on the box, not Gaussian: the question is what the action
        # space CONTAINS, so every reachable gain combination is equally
        # eligible rather than concentrated near the controller's nominal.
        a = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        ok, ret = _episode(stage, spec, hold_steps, seed, terrain_seed, a)
        hits += ok
        best = max(best, ret)
    return seed, zero_ok, hits, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="land_handoff")
    ap.add_argument("--gain-set", default="attitude", choices=sorted(GAIN_SETS))
    ap.add_argument("--hold-steps", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--terrain-seed", type=int, default=7)
    ap.add_argument("--samples", type=int, default=24,
                    help="gain vectors tried per episode")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    spec = GAIN_SETS[args.gain_set]
    print(f"stage {args.stage}  gain-set {args.gain_set} ({len(spec)} knobs)  "
          f"hold {args.hold_steps}  {args.episodes} episodes x "
          f"{args.samples} samples")
    print(f"knobs: {[k.name for k in spec]}")

    payloads = [(args.stage, args.gain_set, args.hold_steps, args.seed0 + i,
                 args.terrain_seed, args.samples, 90_000 + i)
                for i in range(args.episodes)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_one_seed, payloads))
    rows.sort()

    base = sum(r[1] for r in rows)
    oracle = sum(1 for r in rows if r[2] > 0)
    never = [r[0] for r in rows if r[2] == 0 and not r[1]]
    n = len(rows)
    print()
    print(f"zero action (the controller) : {base}/{n} = {base / n:.1%}")
    print(f"ANY of {args.samples} gain vectors: {oracle}/{n} = {oracle / n:.1%}"
          f"   <- ceiling for a state-independent gain policy")
    print(f"no sample ever landed        : {len(never)}/{n} = {len(never) / n:.1%}"
          f"   <- candidate infeasible entries")
    print()
    hitrate = np.array([r[2] / args.samples for r in rows])
    print(f"per-episode hit rate across the gain box: mean {hitrate.mean():.3f}, "
          f"median {np.median(hitrate):.3f}, max {hitrate.max():.3f}")
    print(f"episodes where >25% of gain vectors land: "
          f"{int((hitrate > 0.25).sum())}/{n}")
    print()
    if oracle <= base + 1:
        print("VERDICT: the gain space contains essentially no room on this "
              "stage. Training harder is the wrong move -- change the action "
              "space or the stage.")
    else:
        print(f"VERDICT: the gain space contains room ({base / n:.1%} -> "
              f"{oracle / n:.1%}). A flat training curve is an optimisation "
              f"problem, not a reachability one.")


if __name__ == "__main__":
    main()
