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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunarsim.control.zemzev_controller import ZemZevController, ZemZevGains  # noqa: E402
from lunarsim.core.terrain.config import TerrainConfig  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv, LanderParams  # noqa: E402


@dataclass
class Stage:
    name: str
    tile_size_m: float
    params: LanderParams = field(default_factory=LanderParams)
    terrain_roughness_scale: float = 1.0


# mirrors scripts/train_sac_isaac.py's STAGES exactly, so a pass/fail here is
# directly comparable to what the RL curriculum was asked to solve.
STAGES = {
    "hover_only_easy": Stage(
        name="hover_only_easy", tile_size_m=40.0, terrain_roughness_scale=0.0,
        params=LanderParams(
            spawn_altitude_m=20.0, spawn_xy_radius_m=0.0,
            spawn_v_z_m_s=-2.0, spawn_horizontal_speed_m_s=(0.0, 0.0),
            max_episode_s=25.0,
            safe_landing_v_z_m_s=5.0, safe_landing_v_xy_m_s=5.0,
            safe_landing_tilt_rad=np.deg2rad(30.0), safe_landing_w_rad_s=2.0,
        ),
    ),
    "hover_only": Stage(
        name="hover_only", tile_size_m=40.0, terrain_roughness_scale=0.0,
        params=LanderParams(
            spawn_altitude_m=20.0, spawn_xy_radius_m=0.0,
            spawn_v_z_m_s=-2.0, spawn_horizontal_speed_m_s=(0.0, 0.0),
            max_episode_s=25.0,
        ),
    ),
    "final_approach": Stage(
        name="final_approach", tile_size_m=60.0,
        params=LanderParams(
            spawn_altitude_m=35.0, spawn_xy_radius_m=15.0,
            spawn_v_z_m_s=-3.0, spawn_horizontal_speed_m_s=(0.0, 3.0),
            max_episode_s=45.0,
        ),
    ),
    "orbit_descent": Stage(
        name="orbit_descent", tile_size_m=600.0,
        params=LanderParams(
            spawn_altitude_m=200.0, spawn_xy_radius_m=80.0,
            spawn_v_z_m_s=0.0, spawn_horizontal_speed_m_s=(10.0, 30.0),
            max_episode_s=60.0,
        ),
    ),
}


def _terrain_config(stage: Stage, seed: int) -> TerrainConfig:
    r = stage.terrain_roughness_scale
    return TerrainConfig(
        mode="fine", size_m=stage.tile_size_m, res_m=max(0.5, stage.tile_size_m / 80.0), seed=seed,
        coarse_source="procedural",
        hills={"amplitude_m": 0.3 * r, "wavelength_m": stage.tile_size_m / 6.0, "hurst": 0.75},
        craters={"count_scale": 0.1 * r, "d_min_m": 1.0, "d_max_m": stage.tile_size_m / 10.0, "b": 2.5,
                 "depth_ratio": 0.08, "age": 0.5},
        rocks={"density_scale": r, "d_max_m": 1.0},
        roi={"sigma_m": stage.tile_size_m / 4.0, "centers": None},
        curvature=False,
    )


def _no_reward(env, info):
    return 0.0


def run_episode(stage: Stage, seed: int, gains: ZemZevGains) -> dict:
    tile = generate_tile(_terrain_config(stage, seed))
    env = AnalyticLanderEnv(tile=tile, params=stage.params, reward_fn=_no_reward)
    controller = ZemZevController(stage.params, gains)

    obs, _ = env.reset(seed=seed)
    controller.reset()
    info = {}
    for _ in range(int(stage.params.max_episode_s / stage.params.dt_s) + 5):
        action = controller.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
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

        print(f"\n=== stage: {name} ({n} episodes) ===")
        print(f"landed_safely: {n_safe}/{n} ({100 * n_safe / n:.0f}%)")
        print(f"touchdown (any): {n_touchdown}/{n}   lost_control: {n_lost}/{n}   timed_out: {n_timeout}/{n}")
        print(f"touchdown |vz|  : mean={np.abs(vz).mean():.2f}  p90={np.percentile(np.abs(vz), 90):.2f}  "
              f"max={np.abs(vz).max():.2f}  (limit {stage.params.safe_landing_v_z_m_s:.2f})")
        print(f"touchdown vxy   : mean={vxy.mean():.2f}  p90={np.percentile(vxy, 90):.2f}  "
              f"max={vxy.max():.2f}  (limit {stage.params.safe_landing_v_xy_m_s:.2f})")
        print(f"touchdown tilt  : mean={tilt.mean():.1f}deg  p90={np.percentile(tilt, 90):.1f}deg  "
              f"max={tilt.max():.1f}deg  (limit {np.degrees(stage.params.safe_landing_tilt_rad):.1f})")


if __name__ == "__main__":
    main()
