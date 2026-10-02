"""Monte-Carlo feasibility test: fly `ZemZevController` (a hand-designed,
non-RL guidance+attitude law -- see `lunarsim/control/zemzev_controller.py`)
through the SAME curriculum-stage distributions `scripts/train_sac_isaac.py`
trains against (identical `LanderParams` + identical terrain config), using
the fast CPU-only `AnalyticLanderEnv` (no Isaac Sim / GPU needed).

Purpose: answer "is a safe landing physically achievable from this release
condition with this vehicle model at all" BEFORE touching the reward
function again. If this controller also can't land safely, that's evidence
of a genuine control-authority ceiling, not a reward-shaping problem --
see `handover.md`.

Run:
    .venv/bin/python scripts/test_landing_feasibility.py --stage orbit_descent --episodes 30
    .venv/bin/python scripts/test_landing_feasibility.py --stage all --episodes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunarsim.control.zemzev_controller import ZemZevController, ZemZevGains  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, Stage, terrain_config  # noqa: E402


# REAL DRIFT FOUND AND FIXED (this session): this file carried its own
# hand-copied STAGES table with a comment claiming it "mirrors
# scripts/train_sac_isaac.py's STAGES exactly, so a pass/fail here is
# directly comparable to what the RL curriculum was asked to solve". It did
# not: it still defined hover_only_easy / hover_only / final_approach --
# three stages DELETED from training earlier in this session -- and had
# never heard of the four ramp_* stages that replaced them. The whole value
# of this script is that it certifies the SAME scenario the trainer trains,
# so it now imports the single shared definition instead of copying it.
STAGES = STAGES_BY_NAME


def _no_reward(env, info):
    return 0.0


def run_episode(stage: Stage, seed: int, gains: ZemZevGains) -> dict:
    tile = generate_tile(terrain_config(stage, seed))
    env = AnalyticLanderEnv(tile=tile, params=stage.params, reward_fn=_no_reward)
    controller = ZemZevController(stage.params, gains)

    obs, _ = env.reset(seed=seed)
    controller.reset()
    info = {}
    # peak distance from the tile centre: this is the number that exposed
    # the undersized-tile bug (see lunarsim/rl/curriculum.py's STAGES
    # comment -- 7-15 of 24 episodes per stage used to fly off the terrain
    # collider entirely), so it is reported, not just computed.
    max_radius_m = 0.0
    for _ in range(int(stage.params.max_episode_s / stage.params.dt_s) + 5):
        action = controller.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        max_radius_m = max(max_radius_m, float(np.hypot(env.state["x"], env.state["y"])))
        if terminated or truncated:
            break

    s = env.state
    v_xy = float(np.hypot(s["vx"], s["vy"]))
    tilt = float(np.hypot(s["tilt_x"], s["tilt_y"]))
    return {
        "landed_safely": info.get("landed_safely", False),
        "lost_control": info.get("lost_control", False),
        "timed_out": info.get("truncated", False) and not info.get("terminated", False),
        "touchdown": info.get("terminated", False) and not info.get("lost_control", False),
        "vz": float(s["vz"]),
        "vxy": v_xy,
        "tilt_deg": float(np.degrees(tilt)),
        "t_s": info.get("t_s", float("nan")),
        "max_radius_m": max_radius_m,
        "fuel_frac_left": float(s["fuel_kg"] / stage.params.initial_fuel_kg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="orbit_descent", choices=list(STAGES) + ["all"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()

    gains = ZemZevGains()
    stage_names = list(STAGES) if args.stage == "all" else [args.stage]

    for name in stage_names:
        stage = STAGES[name]
        results = [run_episode(stage, args.seed0 + i, gains) for i in range(args.episodes)]

        n = len(results)
        n_safe = sum(r["landed_safely"] for r in results)
        n_lost = sum(r["lost_control"] for r in results)
        n_touchdown = sum(r["touchdown"] for r in results)
        n_timeout = sum(r["timed_out"] for r in results)
        vz = np.array([r["vz"] for r in results])
        vxy = np.array([r["vxy"] for r in results])
        tilt = np.array([r["tilt_deg"] for r in results])
        radius = np.array([r["max_radius_m"] for r in results])
        half_extent = stage.tile_size_m / 2.0

        print(f"\n=== stage: {name} ({n} episodes) ===")
        print(f"landed_safely: {n_safe}/{n} ({100 * n_safe / n:.0f}%)")
        print(f"touchdown (any): {n_touchdown}/{n}   lost_control: {n_lost}/{n}   timed_out: {n_timeout}/{n}")
        print(f"touchdown |vz|  : mean={np.abs(vz).mean():.2f}  p90={np.percentile(np.abs(vz), 90):.2f}  "
              f"max={np.abs(vz).max():.2f}  (limit {stage.params.safe_landing_v_z_m_s:.2f})")
        print(f"touchdown vxy   : mean={vxy.mean():.2f}  p90={np.percentile(vxy, 90):.2f}  "
              f"max={vxy.max():.2f}  (limit {stage.params.safe_landing_v_xy_m_s:.2f})")
        print(f"touchdown tilt  : mean={tilt.mean():.1f}deg  p90={np.percentile(tilt, 90):.1f}deg  "
              f"max={tilt.max():.1f}deg  (limit {np.degrees(stage.params.safe_landing_tilt_rad):.1f})")
        print(f"ground track    : max radius mean={radius.mean():.0f}m  p90={np.percentile(radius, 90):.0f}m  "
              f"max={radius.max():.0f}m  (tile half-extent {half_extent:.0f}m, "
              f"{int((radius > half_extent).sum())}/{n} off the tile)")


if __name__ == "__main__":
    main()
