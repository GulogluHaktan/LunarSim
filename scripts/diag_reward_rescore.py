"""Which reward would have ranked the policies we already have by LANDING RATE?

Measured on 192 paired validation episodes: the guided-gain run that landed
31.2% scored mean return -20.54, while the one that landed 21.9% scored -15.94
and the controller (9.9%) -17.52. PPO maximises return, so under the shipped
reward its own objective preferred the WEAKER-landing policy. A reward the
optimiser climbs in the wrong direction cannot be fixed by a better optimiser.

This screens candidate rewards on FIXED policies, without training:

  * The reward does not feed the dynamics, and none of the candidates here
    touch `target_velocity`'s fields, so the observation is unchanged too.
    Each episode is therefore flown ONCE per arm and its return is computed
    under EVERY candidate reward at the same time, on identical states and
    actions. Any difference between candidates is the reward's and nothing
    else.
  * The criterion is the one that failed: does mean return order the arms the
    same way landing rate does? Plus a per-episode version -- across the
    paired episodes, does the return difference policy-minus-controller agree
    in sign with the landed difference? That is what a policy gradient sees.

What this does NOT say: the policies were trained under the default reward,
so this screens what a candidate WOULD HAVE PAID for behaviour already on tape,
not what it would make PPO learn. It is a necessary condition, cheap, and the
one the shipped reward fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS, GuidedGainEnv  # noqa: E402
from lunarsim.rl.reward import RewardWeights, make_apollo_reward_fn  # noqa: E402

BLOCKS = (90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000)


def candidates() -> dict[str, RewardWeights]:
    """Named reward variants. Every one keeps the guidance-field weights at
    their defaults, so the observation is identical across them."""
    d = RewardWeights()
    return {
        "shipped": d,
        # The step the previous screen sized: needs ~49 units to fix the
        # ranking on its own, i.e. landed_step_k ~983. Included at sane sizes
        # to show how far short they fall.
        "step100": replace(d, landed_step_k=100.0),
        "step300": replace(d, landed_step_k=300.0),
        # Compress the FAILURE side. The ramp spends +6..-24 grading failure
        # severity -- four times more range below the boundary than above it --
        # while the metric does not care how badly an episode failed.
        "cap120": replace(d, touchdown_penalty_cap=120.0),
        "cap60": replace(d, touchdown_penalty_cap=60.0),
        "cap120_step100": replace(d, touchdown_penalty_cap=120.0, landed_step_k=100.0),
        "cap60_step100": replace(d, touchdown_penalty_cap=60.0, landed_step_k=100.0),
        # Severity ramp OFF, pure outcome: touchdown_k=0 makes the ramp exactly
        # zero, cap=60 turns every failed touchdown into a flat -3 (equal to a
        # timeout), and the step pays +5 for landing. Binary in, binary out --
        # the shaping is still there to climb.
        "binary": replace(d, touchdown_k=0.0, touchdown_penalty_cap=60.0,
                          landed_step_k=100.0),
        # Same, and the dense shaping turned down 4x, so the outcome dominates
        # what a failing episode can earn by flying more neatly.
        "binary_quietshaping": replace(d, touchdown_k=0.0, touchdown_penalty_cap=60.0,
                                       landed_step_k=100.0, reward_scale=0.0000625),
        # DECOMPOSITION. `reward_scale` multiplies only the per-step shaping
        # (alive bonus included), so these two split the shipped return into
        # its parts exactly.
        "shaping_only": replace(d, touchdown_k=0.0, touchdown_penalty_cap=0.0,
                                timeout_penalty=0.0),
        "terminal_only": replace(d, reward_scale=0.0),
        # MIDDLE GROUND: keep the dense field for credit assignment, but cap its
        # per-step charge so one bad episode cannot pile up -200. 2000 raw is
        # 0.025/step, ~10 over 400 steps instead of 200 -- below the terminal.
        "vfcap2000": replace(d, vfield_cap=2000.0),
        "vfcap2000_step100": replace(d, vfield_cap=2000.0, landed_step_k=100.0),
        "vfcap1000_step100": replace(d, vfield_cap=1000.0, landed_step_k=100.0),
        # Pure outcome: no shaping at all, +5 for landing, -3 for any failure.
        "outcome_only": replace(d, touchdown_k=0.0, touchdown_penalty_cap=60.0,
                                landed_step_k=100.0, reward_scale=0.0),
    }


def _run(payload):
    stage_name, gain_set, hold, seed, terrain_seed, ckpt = payload
    stage = STAGES_BY_NAME[stage_name]
    spec = GAIN_SETS[gain_set]
    names = list(candidates())
    fns = [make_apollo_reward_fn(w) for w in candidates().values()]
    totals = np.zeros(len(fns))

    def composite(env, info):
        # Each candidate reads the same env/info. A candidate's reward_fn writes
        # info["reward_terminal"/"reward_shaping"], so the SHIPPED one runs last
        # and its values are what the env sees -- the episode is the shipped one.
        vals = [f(env, dict(info)) for f in fns]
        for k, v in enumerate(vals):
            totals[k] += v
        return fns[0](env, info)

    composite.reward_weights = RewardWeights()

    model = None
    if ckpt is not None:
        from stable_baselines3 import PPO
        model = PPO.load(ckpt, device="cpu")
    env = GuidedGainEnv(tile=generate_tile(terrain_config(stage, terrain_seed)),
                        params=stage.params, hold_steps=hold, reward_fn=composite,
                        gain_spec=spec)
    obs, _ = env.reset(seed=seed)
    info: dict = {}
    while True:
        act = (np.zeros(len(spec), dtype=np.float32) if model is None
               else model.predict(obs, deterministic=True)[0])
        obs, _r, term, trunc, info = env.step(act)
        if term or trunc:
            break
    out = dict(zip(names, totals.tolist()))
    out["_steps"] = float(env._steps)
    out["_touched"] = float(bool(info.get("landing_margins")))
    return bool(info.get("landed_safely", False)), out


def arm(ckpt, args, seeds):
    payloads = [(args.stage, args.gain_set, args.hold_steps, s, args.terrain_seed, ckpt)
                for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_run, payloads))
    landed = np.array([r[0] for r in rows], dtype=bool)
    rets = {k: np.array([r[1][k] for r in rows]) for k in rows[0][1]}
    return landed, rets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--stage", default="land_handoff")
    ap.add_argument("--gain-set", default="attitude", choices=sorted(GAIN_SETS))
    ap.add_argument("--hold-steps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--episodes-per-block", type=int, default=48)
    ap.add_argument("--terrain-seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    seeds = [b + i for b in BLOCKS[:args.blocks] for i in range(args.episodes_per_block)]
    arms = [("controller", None)] + [(os.path.basename(os.path.dirname(c)) or c, c)
                                    for c in args.checkpoints]
    data = {}
    for name, ck in arms:
        print(f"flying {name} on {len(seeds)} episodes...", flush=True)
        data[name] = arm(ck, args, seeds)

    names = list(candidates())
    land = {a: data[a][0] for a, _ in arms}
    print("\nmean episode length (control steps) and touchdown fraction:")
    for a, _ in arms:
        st = data[a][1]["_steps"]; tc = data[a][1]["_touched"]
        lm, fm = land[a], ~land[a]
        print(f"  {a:<14} all {st.mean():6.1f}   landed {st[lm].mean() if lm.any() else float('nan'):6.1f}"
              f"   failed {st[fm].mean():6.1f}   touched down {tc.mean():.1%}")
    rate = {a: land[a].mean() for a in land}
    order_true = sorted(land, key=lambda a: rate[a], reverse=True)
    print("\nlanding rate: " + "   ".join(f"{a} {rate[a]:.1%}" for a in order_true))
    print(f"\n{'reward':<22} " + " ".join(f"{a[:12]:>13}" for a, _ in arms)
          + f"  {'ranks by landing':>17}  {'per-episode sign agreement':>27}")
    out = {}
    base_land = land["controller"]
    for c in names:
        means = {a: data[a][1][c].mean() for a, _ in arms}
        order = sorted(means, key=lambda a: means[a], reverse=True)
        ok = order == order_true
        # Per-episode: across every (policy, episode) pair where the landed
        # outcome differs from the controller's, does the return difference
        # point the same way?
        agree = total = 0
        for a, ck in arms:
            if ck is None:
                continue
            dl = land[a].astype(int) - base_land.astype(int)
            dr = data[a][1][c] - data["controller"][1][c]
            m = dl != 0
            agree += int(np.sum(np.sign(dr[m]) == np.sign(dl[m])))
            total += int(m.sum())
        frac = agree / total if total else float("nan")
        print(f"{c:<22} " + " ".join(f"{means[a]:>13.2f}" for a, _ in arms)
              + f"  {'YES' if ok else 'no':>17}  {agree:>4}/{total:<4} = {frac:6.1%}")
        out[c] = {"means": means, "ranks_by_landing": ok,
                  "sign_agreement": frac}
    print("\n'per-episode sign agreement' counts the paired episodes whose landed "
          "outcome differs from\nthe controller's, and asks whether the return "
          "moved the same way. 100% means every\nconversion paid and every "
          "regression cost -- what a policy gradient needs.")
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"seeds": seeds, "landing_rate": rate, "candidates": out}, fh,
                      indent=1)


if __name__ == "__main__":
    main()
