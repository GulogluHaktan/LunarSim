"""Fly a trained SAC checkpoint through ITS OWN stage's release condition
(not necessarily orbit_descent) on the real local IsaacLanderEnv, and
report landed_safely / lost_control / timeout counts -- a quick "did this
intermediate ramp-curriculum checkpoint actually solve ITS OWN easier
stage" check, distinct from scripts/diag_policy_telemetry.py (which is
orbit_descent-only and prints full step-by-step telemetry).

Run:
    /home/haktan/isaac-env/bin/python scripts/diag_stage_landing_rate.py \
        --checkpoint out/sac_training_run_ramp/sac_lunar_lander_isaac_ramp_20m.zip \
        --stage ramp_20m --episodes 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stage", type=str, required=True,
                     help="a stage name from lunarsim/rl/curriculum.py's STAGES")
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80,
                     help="must match the --terrain-grid-n the checkpoint was trained with")
parser.add_argument("--seed0", type=int, default=7000)
parser.add_argument("--stochastic", action="store_true",
                     help="sample from the policy instead of taking its mean. Training "
                          "collects data this way while evaluation is deterministic, so a "
                          "large gap between the two means exploration noise is costing "
                          "landings that the mean policy would make -- which poisons the "
                          "replay buffer with failures the policy did not intend.")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, args.lunarsim_root)

from stable_baselines3 import SAC  # noqa: E402

from lunarsim.adapters.isaac.isaac_lander_env import IsaacLanderEnv  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.reward import default_reward_fn  # noqa: E402

# the ONE curriculum definition (see lunarsim/rl/curriculum.py's docstring
# on the four hand-copied, drifted versions this replaces -- this script's
# own copy was already a second place the ramp stages had to be edited by
# hand to stay in step with the trainer).


def main():
    if args.stage not in STAGES_BY_NAME:
        raise SystemExit(f"no stage named {args.stage!r}; choices: {list(STAGES_BY_NAME)}")
    stage = STAGES_BY_NAME[args.stage]
    params = stage.params
    tile = generate_tile(terrain_config(stage, args.terrain_seed, args.terrain_grid_n))
    env = IsaacLanderEnv(tile=tile, params=params, reward_fn=default_reward_fn,
                          seed=args.seed0, lunarsim_root=args.lunarsim_root)
    model = SAC.load(args.checkpoint, device="cpu")
    max_steps = int(params.max_episode_s / params.dt_s) + 5

    n_safe = n_lost = n_timeout = n_left = 0
    rows = []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        obs, _ = env.reset(seed=seed)
        info = {}
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        landed = bool(info.get("landed_safely"))
        lost = bool(info.get("lost_control"))
        # left_tile is a THIRD way to not land, distinct from timing out:
        # the vehicle drifted off the terrain collider entirely (see
        # isaac_lander_env.out_of_tile). Counting it separately is the
        # whole point -- lumped into "timeout" it looks like indecision
        # when it is actually an undersized tile.
        left = bool(info.get("left_tile"))
        timed_out = truncated and not terminated and not left
        n_safe += landed
        n_lost += lost
        n_timeout += timed_out
        n_left += left
        s = env.state
        v_xy = float(np.hypot(s["vx"], s["vy"]))
        tilt_deg = float(np.degrees(np.hypot(s["tilt_x"], s["tilt_y"])))
        w_mag = float(np.linalg.norm([s.get("wx", 0.0), s.get("wy", 0.0), s.get("wz", 0.0)]))
        tag = ("LANDED" if landed else "LOST_CONTROL" if lost
               else "LEFT_TILE" if left else "TIMEOUT" if timed_out else "CRASH")
        # A touchdown that misses `landed_safely` used to print only vz/vxy/
        # tilt, and a real measured episode came back CRASH with vz=-0.21,
        # vxy=0.48, tilt=7.1 -- every printed number comfortably inside its
        # limit. The deciding criterion was one of the two this line never
        # showed: |w| (limit `safe_landing_w_rad_s`) and the leg height
        # difference (limit `safe_landing_max_leg_height_diff_m`). Those are
        # also exactly the two criteria the per-step reward has no term for,
        # so a silent failure on them reads as an unexplained crash. Print
        # the clearance, |w|, and whichever margins the env actually
        # computed, and name the binding criterion outright.
        margins = info.get("landing_margins") or {}
        worst = min(margins, key=margins.get) if margins else None
        detail = (f"  margins={{" + ", ".join(f"{k}={v:.2f}" for k, v in margins.items()) + "}"
                  + (f" worst={worst}" if worst else "")) if margins else ""
        rows.append(f"[ep {ep}] {tag:12s} t_s={info.get('t_s', float('nan')):5.1f}  "
                     f"alt={info.get('altitude_m', float('nan')):6.3f}  "
                     f"vz={s['vz']:7.2f}  vxy={v_xy:6.2f}  tilt={tilt_deg:5.1f}  w={w_mag:5.2f}{detail}")
        print(rows[-1])

    print(f"\n=== {args.stage}: {n_safe}/{args.episodes} landed_safely ({100*n_safe/args.episodes:.0f}%), "
          f"{n_lost}/{args.episodes} lost_control, {n_timeout}/{args.episodes} timeout, "
          f"{n_left}/{args.episodes} left_tile "
          f"-- checkpoint={args.checkpoint} ===")


if __name__ == "__main__":
    main()
    simulation_app.close()
