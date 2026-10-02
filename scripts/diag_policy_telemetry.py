"""Fly a trained SAC checkpoint through orbit_descent on the REAL (local,
non-docker) IsaacLanderEnv and print per-step telemetry around episode
termination -- lightweight diagnostic for "WHY is it losing control /
crashing", not a video-capture run (see scripts/isaaclab_policy_eval_capture.py
for that, docker-based and much slower to spin up). Reuses the exact same
env/physics training trains against (no docker, no camera/lidar), so this
is fast to iterate with.

Run:
    /home/haktan/isaac-env/bin/python scripts/diag_policy_telemetry.py \
        --checkpoint out/sac_training_run_ramp/sac_lunar_lander_isaac_orbit_descent.zip \
        --episodes 16
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
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80,
                     help="must match the --terrain-grid-n the checkpoint was trained with")
parser.add_argument("--seed0", type=int, default=5000)
parser.add_argument("--tail-steps", type=int, default=8, help="how many steps before termination to print")
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
# on the four hand-copied, drifted versions this replaces).
_STAGE = STAGES_BY_NAME["orbit_descent"]
ORBIT_DESCENT_PARAMS = _STAGE.params


def main():
    tile = generate_tile(terrain_config(_STAGE, args.terrain_seed, args.terrain_grid_n))
    env = IsaacLanderEnv(tile=tile, params=ORBIT_DESCENT_PARAMS, reward_fn=default_reward_fn,
                          seed=args.seed0, lunarsim_root=args.lunarsim_root)
    model = SAC.load(args.checkpoint, device="cpu")
    max_steps = int(ORBIT_DESCENT_PARAMS.max_episode_s / ORBIT_DESCENT_PARAMS.dt_s) + 5

    n_safe = n_lost = n_timeout = n_left = 0
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        obs, _ = env.reset(seed=seed)
        history = []
        info = {}
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            s = env.state
            v_xy = float(np.hypot(s["vx"], s["vy"]))
            tilt_deg = float(np.degrees(np.hypot(s["tilt_x"], s["tilt_y"])))
            alt = float(info["altitude_m"])
            history.append((info["t_s"], alt, s["vz"], v_xy, tilt_deg, float(action[0]),
                             float(action[1]), float(action[2])))
            if terminated or truncated:
                break

        landed = bool(info.get("landed_safely"))
        lost = bool(info.get("lost_control"))
        # see diag_stage_landing_rate.py's matching note: "drifted off the
        # terrain collider" is not the same failure as "ran out of clock".
        left = bool(info.get("left_tile"))
        timed_out = truncated and not terminated and not left
        n_safe += landed
        n_lost += lost
        n_timeout += timed_out
        n_left += left
        tag = ("LANDED" if landed else "LOST_CONTROL" if lost
               else "LEFT_TILE" if left else "TIMEOUT" if timed_out else "CRASH")
        print(f"\n[ep {ep}] {tag}  t_s={info.get('t_s', float('nan')):.1f}  reward_last_step={reward:.1f}")
        print("  t_s    alt    vz      vxy    tilt_deg  throttle  pitch_cmd  roll_cmd")
        for row in history[-args.tail_steps:]:
            print(f"  {row[0]:5.1f}  {row[1]:5.1f}  {row[2]:6.2f}  {row[3]:5.2f}  {row[4]:7.1f}  "
                  f"{row[5]:8.2f}  {row[6]:9.2f}  {row[7]:8.2f}")

    print(f"\n=== {n_safe}/{args.episodes} landed_safely, {n_lost}/{args.episodes} lost_control, "
          f"{n_timeout}/{args.episodes} timeout, {n_left}/{args.episodes} left_tile ===")


if __name__ == "__main__":
    main()
    simulation_app.close()
