"""Score a guided-gain checkpoint on FRESH seed blocks, paired against the controller.

`scripts/run_highpower_validation.sh` declared this project's protocol before
its numbers existed, and this is that protocol for the guided-gain env:

  SELECTION. The checkpoint was picked on seed block 7000 alone (the trainer's
  `--eval-seed0`). That block therefore cannot score it.
  POWER. Score on K fresh blocks, disjoint from 7000 and from 41000 (the block
  every earlier matrix used).

TWO THINGS THIS DOES THAT THE SHELL VERSION DOES NOT.

1. IT PAIRS. Both arms fly the SAME episode seeds on the same terrain, and the
   zero-action arm is the controller by construction -- so the comparison is
   paired and McNemar is the correct test, not the unpooled two-proportion z
   that `run_highpower_validation.sh` computes. That matters: an unpaired z at
   n=96 reports a +-10.0 point interval, while a paired test at a 20%
   discordance rate reports +-6.3. The project has been discarding roughly
   1.5-2x of its own power and then citing the loss as grounds for "the
   measurement cannot say". Both tests are printed here so the two are
   comparable, and the paired one is the one to read.

2. IT KEEPS THE PER-EPISODE OUTCOME VECTORS. `diag_stage_landing_rate.py`
   persists only its printed summary, so no 192-episode matrix in this project
   can be re-analysed after the fact -- "10/96 at all five checkpoints" cannot
   be checked for whether it was the same ten episodes. The 0/1 vectors go to
   `--out-json` so any later question can be asked of the data.

The baseline is RE-MEASURED here rather than quoted. Two of handover.md's
load-bearing baselines turned out to be artifacts of the harness that produced
them, so an arm that is not measured through the same code on the same seeds is
not a comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS, GuidedGainEnv  # noqa: E402
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides  # noqa: E402

# Disjoint from 7000 (selection) and 41000 (every prior matrix), same list the
# shell protocol draws from, so a block here is comparable to a block there.
BLOCKS = (90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000)


def _run(payload):
    (stage_name, gain_set, hold, resid, seed, terrain_seed, ckpt, rw) = payload
    stage = STAGES_BY_NAME[stage_name]
    spec = GAIN_SETS[gain_set]
    n_act = len(spec) + (4 if resid > 0.0 else 0)

    model = None
    if ckpt is not None:
        from stable_baselines3 import PPO
        model = PPO.load(ckpt, device="cpu")

    # A reward override here re-SCORES fixed policies under a candidate reward,
    # which is how to screen a reward change without spending a training run:
    # the episodes and the actions are identical, so any difference is the
    # reward's. Note the policies were TRAINED under the defaults, so this says
    # what the candidate reward would have rewarded, not what it would learn.
    rfn = make_apollo_reward_fn(weights_from_overrides(rw)) if rw else None
    env = GuidedGainEnv(tile=generate_tile(terrain_config(stage, terrain_seed)),
                        params=stage.params, hold_steps=hold,
                        reward_fn=rfn, gain_spec=spec, residual_scale=resid)
    obs, _ = env.reset(seed=seed)
    info: dict = {}
    total = 0.0
    while True:
        if model is None:
            act = np.zeros(n_act, dtype=np.float32)
        else:
            act, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        total += float(rew)
        if term or trunc:
            break
    return bool(info.get("landed_safely", False)), total


def arm(ckpt, stage_name, gain_set, hold, resid, seeds, terrain_seed, workers,
        rw=None):
    payloads = [(stage_name, gain_set, hold, resid, s, terrain_seed, ckpt,
                 tuple(rw) if rw else None)
                for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_run, payloads))
    return (np.array([r[0] for r in rows], dtype=bool),
            np.array([r[1] for r in rows], dtype=float))


def mcnemar(a: np.ndarray, b: np.ndarray):
    """Paired test on the discordant pairs. `b` is the candidate, `a` the base.

    Exact binomial two-sided on the discordant count, which is correct at the
    small discordance totals this design produces; the normal approximation
    the usual chi-square form uses is not trustworthy there.
    """
    b_only = int(np.sum(b & ~a))    # candidate landed, baseline did not
    a_only = int(np.sum(a & ~b))
    n = b_only + a_only
    if n == 0:
        return b_only, a_only, 1.0
    k = min(b_only, a_only)
    # two-sided exact: 2 * P(X <= k), X ~ Binom(n, 0.5), capped at 1
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return b_only, a_only, min(1.0, 2.0 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--stage", default="land_handoff")
    ap.add_argument("--gain-set", default="attitude", choices=sorted(GAIN_SETS))
    ap.add_argument("--hold-steps", type=int, default=10)
    ap.add_argument("--residual-scale", type=float, default=0.0)
    ap.add_argument("--blocks", type=int, default=4,
                    help="fresh seed blocks; 1 block = +-14.1 pts, 2 = +-10.0, "
                         "4 = +-7.1, 8 = +-5.0")
    ap.add_argument("--episodes-per-block", type=int, default=48)
    ap.add_argument("--terrain-seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reward-weight", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="re-score under a candidate reward. The policies were "
                         "trained under the defaults, so this screens what a "
                         "reward WOULD have paid for behaviour already on tape "
                         "-- cheap, and it does not say what it would learn.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.blocks > len(BLOCKS):
        raise SystemExit(f"only {len(BLOCKS)} pre-registered blocks exist")
    blocks = BLOCKS[:args.blocks]
    seeds = [b + i for b in blocks for i in range(args.episodes_per_block)]

    print(f"checkpoint  {args.checkpoint}")
    print(f"stage       {args.stage}  gain-set {args.gain_set}  "
          f"hold {args.hold_steps}")
    print(f"blocks      {list(blocks)} x {args.episodes_per_block} "
          f"= {len(seeds)} paired episodes  (terrain seed {args.terrain_seed})")
    print("NOT block 7000 -- that one selected the checkpoint.\n")

    if args.reward_weight:
        print(f"reward override: {args.reward_weight}\n")
    base_ok, base_ret = arm(None, args.stage, args.gain_set, args.hold_steps,
                            args.residual_scale, seeds, args.terrain_seed,
                            args.workers, args.reward_weight)
    cand_ok, cand_ret = arm(args.checkpoint, args.stage, args.gain_set,
                            args.hold_steps, args.residual_scale, seeds,
                            args.terrain_seed, args.workers, args.reward_weight)

    n = len(seeds)
    nb, nc = int(base_ok.sum()), int(cand_ok.sum())
    pb, pc = nb / n, nc / n
    print(f"controller (zero action) {nb:>4}/{n}  {pb:>6.1%}   "
          f"mean return {base_ret.mean():>8.2f}")
    print(f"guided-gain policy       {nc:>4}/{n}  {pc:>6.1%}   "
          f"mean return {cand_ret.mean():>8.2f}")
    print(f"difference               {pc - pb:>+6.1%}\n")

    # Per block, so a single lucky block cannot carry the total.
    print("per block:")
    for j, b in enumerate(blocks):
        sl = slice(j * args.episodes_per_block, (j + 1) * args.episodes_per_block)
        print(f"  {b}: controller {int(base_ok[sl].sum()):>3}/"
              f"{args.episodes_per_block}   policy "
              f"{int(cand_ok[sl].sum()):>3}/{args.episodes_per_block}")

    cand_only, base_only, p_exact = mcnemar(base_ok, cand_ok)
    print(f"\nPAIRED (McNemar, exact) -- the right test for this design")
    print(f"  policy landed where controller did not : {cand_only}")
    print(f"  controller landed where policy did not : {base_only}")
    print(f"  discordant pairs {cand_only + base_only} "
          f"({(cand_only + base_only) / n:.1%} of episodes)")
    print(f"  two-sided p = {p_exact:.4g}   "
          f"{'SIGNIFICANT' if p_exact < 0.05 else 'not significant'} at 0.05")

    pooled = (nb + nc) / (2 * n)
    se = math.sqrt(2.0 * pooled * (1.0 - pooled) / n) if 0 < pooled < 1 else 0.0
    z = (pc - pb) / se if se > 0 else 0.0
    print(f"\nUNPAIRED two-proportion z -- what run_highpower_validation.sh "
          f"computes, shown for comparison")
    print(f"  z = {z:.2f}, 95% CI on the difference "
          f"+-{1.96 * math.sqrt(2.0 * 0.25 / n) * 100:.1f} points")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"checkpoint": args.checkpoint, "stage": args.stage,
                       "gain_set": args.gain_set, "hold_steps": args.hold_steps,
                       "blocks": list(blocks),
                       "episodes_per_block": args.episodes_per_block,
                       "seeds": seeds,
                       "baseline_landed": base_ok.tolist(),
                       "candidate_landed": cand_ok.tolist(),
                       "baseline_returns": base_ret.tolist(),
                       "candidate_returns": cand_ret.tolist()}, fh)
        print(f"\nper-episode outcome vectors -> {args.out_json}")


if __name__ == "__main__":
    main()
