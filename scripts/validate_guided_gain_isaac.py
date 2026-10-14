"""Score a guided-gain checkpoint on Isaac, PAIRED against the controller.

The in-training eval in `scripts/train_guided_gain_isaac.py` is good enough to
SELECT a checkpoint and not good enough to score one, for a reason worth
stating: `IsaacLanderVecEnv` draws every spawn quantity from one internal
`self._rng` inside `_reset_one`, and it auto-resets each env the moment that
env's episode ends. A policy that changes episode lengths therefore changes the
ORDER of `_reset_one` calls, so the two arms fly different release states and
the comparison is unpaired -- a +-10 point interval at 48 episodes, which is
most of the effect being looked for.

THE FIX IS THE WAVE DESIGN. Re-seed `_rng` to a declared value, `reset()` all
`num_envs` envs at once, and collect exactly ONE episode per env, stopping each
env's accounting at its first termination. Those `num_envs` spawns come from
`num_envs` deterministic `_reset_one(i)` calls in index order, so re-running the
same wave seed with a different policy reproduces the same release states
exactly. Episodes that keep flying after their env auto-resets are ignored, the
way `guided_gain_vec_env` already discards sub-steps after a mid-hold
termination.

One wave is `num_envs` paired episodes. `--waves` of them, each with its own
declared seed, gives the block structure `run_highpower_validation.sh` asks
for. Both arms are measured here, in the same process, through the same env --
never quoted -- because two of handover.md's baselines turned out to be
artifacts of the harness that produced them.

The paired data then supports McNemar rather than an unpooled two-proportion z.
At a 20% discordance rate that is +-6.3 points against the z's +-10.0, which is
power this project already paid for and was discarding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", nargs="?", default=None,
                    help="omit to measure the controller arm alone")
parser.add_argument("--lunarsim-root", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--stage", type=str, default="orbit_descent")
parser.add_argument("--gain-set", type=str, default="attitude")
parser.add_argument("--hold-steps", type=int, default=20)
parser.add_argument("--residual-scale", type=float, default=0.0)
parser.add_argument("--n-envs", type=int, default=12)
parser.add_argument("--waves", type=int, default=16,
                    help="each wave is --n-envs paired episodes")
parser.add_argument("--wave-seed0", type=int, default=90000,
                    help="declared in advance; must NOT be the selection seed")
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80)
parser.add_argument("--max-rocks-per-env", type=int, default=0)
parser.add_argument("--out-json", type=str, default=None)
parser.add_argument("--torch-device", type=str, default="cuda")
parser.add_argument("--constant-action", type=str, default=None,
                    help="comma-separated fixed action held for every decision instead of a policy: "
                         "the 'is a single fixed gain set enough?' control arm")
parser.add_argument("--reward-weight", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="override a RewardWeights field for the whole run")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})
sys.path.insert(0, args.lunarsim_root)

from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS  # noqa: E402
from lunarsim.rl.guided_gain_vec_env import GuidedGainVecEnv  # noqa: E402
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides  # noqa: E402


def wave(gvenv, model, wave_seed: int, max_decisions: int):
    """One wave: `num_envs` paired episodes from a declared rng seed.

    Re-seeding before `reset()` is what makes this reproducible across arms --
    the first `num_envs` `_reset_one` calls then draw the same spawns in the
    same index order whatever policy is about to fly them.
    """
    n = gvenv.num_envs
    n_act = int(gvenv.action_space.shape[0])
    gvenv.venv._rng = np.random.default_rng(wave_seed)
    obs = gvenv.reset()

    landed = np.zeros(n, dtype=bool)
    rets = np.zeros(n, dtype=float)
    live = np.zeros(n, dtype=float)
    settled = np.zeros(n, dtype=bool)

    for _ in range(max_decisions):
        if model is None:
            act = np.zeros((n, n_act), dtype=np.float32)
        else:
            act, _ = model.predict(obs, deterministic=True)
        obs, rew, dones, infos = gvenv.step(act)
        live += np.where(settled, 0.0, rew)
        for i in range(n):
            if settled[i] or not dones[i]:
                continue
            landed[i] = bool(infos[i].get("landed_safely", False))
            rets[i] = live[i]
            settled[i] = True
        if settled.all():
            break
    return landed, rets, settled


def mcnemar(a: np.ndarray, b: np.ndarray):
    b_only = int(np.sum(b & ~a))
    a_only = int(np.sum(a & ~b))
    m = b_only + a_only
    if m == 0:
        return b_only, a_only, 1.0
    k = min(b_only, a_only)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / (2.0 ** m)
    return b_only, a_only, min(1.0, 2.0 * tail)


def main() -> None:
    stage = STAGES_BY_NAME[args.stage]
    spec = GAIN_SETS[args.gain_set]
    max_dec = int(stage.params.max_episode_s / stage.params.dt_s
                  / args.hold_steps) + 4

    max_tile = max(s.tile_size_m for s in STAGES)
    venv = IsaacLanderVecEnv(
        num_envs=args.n_envs,
        tile_fn=make_tile_fn(stage, args.terrain_grid_n),
        params=stage.params, env_spacing_m=max_tile * 1.5,
        seed=args.terrain_seed, max_rocks_per_env=args.max_rocks_per_env,
        reward_fn=make_apollo_reward_fn(weights_from_overrides(args.reward_weight)))
    gvenv = GuidedGainVecEnv(venv, gain_spec=spec, hold_steps=args.hold_steps,
                             residual_scale=args.residual_scale)

    model = None
    if args.constant_action:
        _a = np.array([float(v) for v in args.constant_action.split(",")], dtype=np.float32)

        class _Const:
            def predict(self, obs, deterministic=True):
                return np.tile(_a, (obs.shape[0], 1)), None
        model = _Const()
    elif args.checkpoint and args.checkpoint.endswith(".npz"):
        z = np.load(args.checkpoint)
        th, sizes, mu, sd = z["theta"], [int(v) for v in z["sizes"]], z["mu"], z["sd"]

        class _ES:   # ES policy in the guided action space: tanh(MLP((obs - mu) / sd))
            def predict(self, obs, deterministic=True):
                out = []
                for o in np.asarray(obs, dtype=np.float64):
                    if sizes[0] == 0:
                        out.append(np.tanh(th[:sizes[1]])); continue
                    x = (o - mu) / sd; k = 0
                    for li in range(len(sizes) - 1):
                        ai, bi = sizes[li], sizes[li + 1]
                        x = x @ th[k:k + ai * bi].reshape(ai, bi) + th[k + ai * bi:k + ai * bi + bi]; k += ai * bi + bi
                        if li < len(sizes) - 2: x = np.tanh(x)
                    out.append(np.tanh(x))
                return np.array(out, dtype=np.float32), None
        model = _ES()
    elif args.checkpoint:
        from stable_baselines3 import PPO
        model = PPO.load(args.checkpoint, device=args.torch_device, env=gvenv)

    seeds = [args.wave_seed0 + 1000 * w for w in range(args.waves)]
    print(f"stage {args.stage} on ISAAC  gain-set {args.gain_set}  "
          f"hold {args.hold_steps}")
    print(f"{args.waves} waves x {args.n_envs} envs = "
          f"{args.waves * args.n_envs} paired episodes")
    print(f"wave seeds {seeds[0]}..{seeds[-1]} (declared; selection used the "
          f"trainer's own block, not these)")
    print(f"checkpoint {args.checkpoint or '(controller arm only)'}\n", flush=True)

    base_ok, cand_ok = [], []
    base_ret, cand_ret = [], []
    for w, sd in enumerate(seeds):
        bo, br, bs = wave(gvenv, None, sd, max_dec)
        if not bs.all():
            print(f"  wave {w} seed {sd}: {int((~bs).sum())} controller "
                  f"episode(s) never terminated; dropped from both arms",
                  flush=True)
        if model is not None:
            co, cr, cs = wave(gvenv, model, sd, max_dec)
            keep = bs & cs
        else:
            co = np.zeros_like(bo)
            cr = np.zeros_like(br)
            keep = bs
        base_ok.extend(bo[keep].tolist())
        base_ret.extend(br[keep].tolist())
        cand_ok.extend(co[keep].tolist())
        cand_ret.extend(cr[keep].tolist())
        nb = int(np.sum(base_ok))
        nc = int(np.sum(cand_ok))
        print(f"  wave {w:>2} seed {sd}: controller {int(bo[keep].sum())}/"
              f"{int(keep.sum())}  policy {int(co[keep].sum())}/{int(keep.sum())}"
              f"   running {nb}/{len(base_ok)} vs {nc}/{len(cand_ok)}", flush=True)

    a = np.array(base_ok, dtype=bool)
    b = np.array(cand_ok, dtype=bool)
    n = len(a)
    pa, pb = a.mean(), b.mean()
    print(f"\ncontroller (zero action) {int(a.sum()):>4}/{n}  {pa:>6.1%}   "
          f"mean return {np.mean(base_ret):>8.2f}")
    if model is not None:
        print(f"guided-gain policy       {int(b.sum()):>4}/{n}  {pb:>6.1%}   "
              f"mean return {np.mean(cand_ret):>8.2f}")
        print(f"difference               {pb - pa:>+6.1%}")
        cand_only, base_only, p = mcnemar(a, b)
        print(f"\nPAIRED (McNemar, exact)")
        print(f"  policy landed where controller did not : {cand_only}")
        print(f"  controller landed where policy did not : {base_only}")
        print(f"  two-sided p = {p:.4g}   "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'} at 0.05")
        print(f"\nUNPAIRED z, for comparison with "
              f"run_highpower_validation.sh: 95% CI "
              f"+-{1.96 * math.sqrt(2 * 0.25 / n) * 100:.1f} points")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"checkpoint": args.checkpoint, "stage": args.stage,
                       "gain_set": args.gain_set, "hold_steps": args.hold_steps,
                       "wave_seeds": seeds, "n_envs": args.n_envs,
                       "baseline_landed": a.tolist(),
                       "candidate_landed": b.tolist(),
                       "baseline_returns": base_ret,
                       "candidate_returns": cand_ret}, fh)
        print(f"\nper-episode outcome vectors -> {args.out_json}")


main()
simulation_app.close()
