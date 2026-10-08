"""Map WHERE a landing specialist succeeds, as a function of the state it is handed.

This is the measurement the two-model split needs and the reason the split is worth
running at all. Splitting the task gives an approach policy and a landing policy, and the
approach policy's reward has to be "did you deliver the vehicle into a state the landing
policy can finish from". Without this map that reward is a guess, and a handoff built on a
guess is how two policies that are each good in their own regime compose into something
worse than one.

So: sample entry conditions from the `land_handoff` distribution, fly the specialist from
each, and report the success rate resolved by entry altitude, descent rate, lateral speed
and tilt. The output is the envelope -- and its EDGES are what the approach policy must be
charged for approaching.

Reads the entry state from the env's first observation rather than re-deriving it, so the
map is of what the policy was actually handed.

Run:
    /home/haktan/isaac-env/bin/python scripts/diag_handoff_envelope.py \
        --checkpoint out/sac_v71_land_specialist/snapshots/land_handoff_410000_steps.zip \
        --episodes 240
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
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stage", type=str, default="land_handoff")
parser.add_argument("--episodes", type=int, default=240)
parser.add_argument("--seed0", type=int, default=7000)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})
sys.path.insert(0, args.lunarsim_root)

from lunarsim.adapters.isaac.isaac_lander_env import IsaacLanderEnv  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.eval_loader import load_actor_for_eval  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402
from lunarsim.rl.reward import default_reward_fn  # noqa: E402


def main():
    stage = STAGES_BY_NAME[args.stage]
    params = stage.params
    tile = generate_tile(terrain_config(stage, args.terrain_seed, args.terrain_grid_n))
    env = IsaacLanderEnv(tile=tile, params=params, reward_fn=default_reward_fn,
                         seed=args.seed0, lunarsim_root=args.lunarsim_root)
    model = load_actor_for_eval(args.checkpoint, device="cpu")
    max_steps = int(params.max_episode_s / params.dt_s) + 5

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed0 + ep)
        o0 = np.asarray(obs, dtype=np.float32)
        entry = {
            "alt": float(o0[2] * OBS_SCALE[2]),
            "vz": float(o0[5] * OBS_SCALE[5]),
            "lat": float(np.hypot(o0[3] * OBS_SCALE[3], o0[4] * OBS_SCALE[4])),
            # tilt from the quaternion slots (6..9 are w,x,y,z)
            "tilt": float(np.hypot(o0[7], o0[8]) * 2.0),
        }
        info = {}
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        entry["ok"] = bool(info.get("landed_safely"))
        rows.append(entry)
        if (ep + 1) % 40 == 0:
            print(f"  {ep + 1}/{args.episodes}  running rate "
                  f"{np.mean([r['ok'] for r in rows]):.1%}", flush=True)

    ok = np.array([r["ok"] for r in rows])
    print(f"\n[envelope] {args.checkpoint}")
    print(f"  stage={args.stage}  episodes={len(rows)}  overall {ok.mean():.1%}\n")
    for key, label, bands in (
        ("alt", "entry altitude (m)", ((8, 13), (13, 18), (18, 25.1))),
        ("vz", "entry descent rate (m/s)", ((-5.1, -3.5), (-3.5, -2.0), (-2.0, -0.9))),
        ("lat", "entry lateral speed (m/s)", ((0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.1))),
        ("tilt", "entry tilt (rad)", ((0, 0.09), (0.09, 0.18), (0.18, 0.40))),
    ):
        v = np.array([r[key] for r in rows])
        print(f"  {label}")
        for lo, hi in bands:
            m = (v >= lo) & (v < hi)
            if m.sum() < 5:
                continue
            print(f"    {f'{lo:g} .. {hi:g}':>16}  n={int(m.sum()):>4}  "
                  f"{ok[m].mean():>6.1%}")
        print()
    print("  The EDGES of this map are what an approach policy must be rewarded for")
    print("  reaching -- not the centre, and not a guessed handoff point.")


main()
simulation_app.close()
