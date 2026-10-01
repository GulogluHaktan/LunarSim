"""Curriculum training for `AnalyticLanderEnv`: starts the policy on the
easiest sub-task (final approach -- low altitude, near-zero horizontal
speed, basically hover-and-land) and progressively increases spawn
altitude + horizontal speed toward a PDI-like ("braking phase") envelope,
carrying the learned policy forward stage to stage (a reverse curriculum:
start next to the goal, move the start further away once the closer one
is solid).

Scope note: this does NOT simulate real two-body/Kepler orbital dynamics
(position relative to the Moon's center, ~1.6 km/s orbital velocity) --
`AnalyticLanderEnv`'s terrain tile is a local flat patch, not an orbit,
so a literal 15 km/1.6 km/s PDI start would cross the whole tile in a
fraction of a second. What this curriculum captures instead is the real
PDI-to-touchdown *shape*, scaled to fit the tile: a policy that first
masters vertical hover/landing control, then learns to null out
increasingly large horizontal drift from increasingly high altitude
before it's allowed to just settle down -- mirroring the real braking
phase -> approach phase -> landing phase progression (see
`lunarsim.rl.analytic_lander_env`'s module docstring on
`spawn_horizontal_speed_m_s`).

Each stage gets its own (larger) terrain tile SIZE, but a fresh, randomly
seeded terrain is generated for *every episode* (via `AnalyticLanderEnv`'s
`tile_fn`) rather than training against one fixed map -- otherwise the
policy can just memorize one specific crater/slope layout instead of
learning to land generally.

`braking_phase`'s spawn speed is deliberately large (200-350 m/s) --
"gelmesi yörüngeden çıkmış gibi hızlı olmalı": this env doesn't simulate
real orbital mechanics (see the scope note above), so there's no literal
~1.6 km/s orbital speed to inherit; this is the fastest inbound speed
that still gives the policy a fighting chance to brake within the tile's
size/episode-length budget, standing in for "arrived hot, like it just
came out of orbit" rather than a literal deorbit velocity.

Uses SAC (off-policy, replay buffer) rather than PPO: for a 4D continuous
action space like this (throttle + 3-axis RCS torque), SAC is generally
far more sample-efficient than an on-policy method, which matters since
each curriculum stage only gets a modest step budget. The replay buffer
IS reset at every stage boundary by default (`model.replay_buffer.reset()`)
-- SB3's buffer stores the reward VALUE at collection time, and a real
run demonstrated this going wrong: chaining a relaxed-safety-threshold
stage into the real-threshold stage without resetting the buffer made the
real-threshold stage's landings *worse*, not better, because stale
transitions scored under the old threshold definition kept getting
sampled alongside new ones scored under the tightened one. Pass
`--keep-buffer` only when you've confirmed consecutive stages share the
same reward definition (spawn/tile conditions differ, thresholds don't)
and you actually want the carryover.

Usage: .venv/bin/python scripts/train_sac_curriculum.py --steps-per-stage 200000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.rl import AnalyticLanderEnv, LanderParams


@dataclass
class CurriculumStage:
    name: str
    tile_size_m: float
    params: LanderParams = field(default_factory=LanderParams)
    # scales the hills/craters terrain-generation amplitude (1.0 = the
    # normal curriculum roughness below). ADDED after a real eval run on
    # `hover_only` showed flight control had actually converged (vz/vxy/
    # tilt/angular-rate all comfortably inside the safe thresholds) but
    # `landed_safely` stayed flat, because the *terrain* under the pad
    # randomly exceeded the 0.16 m footpad-height-diff tolerance (leg_diff
    # measured 0.17-0.38 m on several "otherwise perfect" touchdowns) --
    # this couples two different skills (vertical flight control vs.
    # handling rough terrain) that the early bootstrap stages shouldn't
    # have to learn at once. `hover_only`/`hover_only_easy` train on
    # near-flat terrain; roughness returns at `final_approach` onward.
    terrain_roughness_scale: float = 1.0


STAGES = [
    CurriculumStage(
        # Bootstrap stage, ADDED after a real training run showed the
        # policy couldn't hold attitude at all (39/40 eval episodes hit
        # the loss-of-control cutoff) even on the "easy" stage below --
        # isolates attitude/throttle control (straight-down spawn, zero
        # horizontal drift) before compounding it with horizontal
        # correction.
        #
        # Safety thresholds are DELIBERATELY relaxed here (5.0 m/s / 30 deg
        # vs. the real 1.0 m/s / 15 deg used from `hover_only` onward).
        # ADDED after training against the real thresholds for 1.5M steps
        # produced 0/40 successful landings out of 40 eval episodes -- the
        # agent was consistently touching down around 4.3 m/s and never
        # once sampled the sparse +landing_bonus_scale reward (even under
        # exploration noise), so it had nothing to learn a good landing
        # FROM. Relaxing thresholds first lets it experience some actual
        # successes, then `hover_only` tightens back to the real target --
        # threshold curriculum, same idea as the spawn-condition curriculum
        # applied to the reward's success criterion instead.
        name="hover_only_easy",
        tile_size_m=200.0,
        terrain_roughness_scale=0.0,
        params=LanderParams(
            spawn_altitude_m=60.0, spawn_xy_radius_m=0.0,
            spawn_v_z_m_s=-2.0, spawn_horizontal_speed_m_s=(0.0, 0.0),
            max_episode_s=60.0,
            safe_landing_v_z_m_s=5.0, safe_landing_v_xy_m_s=5.0,
            safe_landing_tilt_rad=np.deg2rad(30.0), safe_landing_w_rad_s=2.0,
        ),
    ),
    CurriculumStage(
        name="hover_only",
        tile_size_m=200.0,
        terrain_roughness_scale=0.0,
        params=LanderParams(
            spawn_altitude_m=60.0, spawn_xy_radius_m=0.0,
            spawn_v_z_m_s=-2.0, spawn_horizontal_speed_m_s=(0.0, 0.0),
            max_episode_s=60.0,
        ),
    ),
    CurriculumStage(
        name="final_approach",
        tile_size_m=200.0,
        params=LanderParams(
            spawn_altitude_m=80.0, spawn_xy_radius_m=30.0,
            spawn_v_z_m_s=-3.0, spawn_horizontal_speed_m_s=(0.0, 5.0),
            max_episode_s=90.0,
        ),
    ),
    CurriculumStage(
        name="approach_phase",
        tile_size_m=3000.0,
        params=LanderParams(
            spawn_altitude_m=500.0, spawn_xy_radius_m=150.0,
            spawn_v_z_m_s=-12.0, spawn_horizontal_speed_m_s=(60.0, 120.0),
            max_episode_s=150.0,
        ),
    ),
    CurriculumStage(
        # PDI-like, scaled to fit a local flat tile -- deliberately fast,
        # "yörüngeden çıkmış gibi": see module docstring.
        name="braking_phase",
        tile_size_m=8000.0,
        params=LanderParams(
            spawn_altitude_m=1500.0, spawn_xy_radius_m=300.0,
            spawn_v_z_m_s=-25.0, spawn_horizontal_speed_m_s=(200.0, 350.0),
            max_episode_s=240.0,
        ),
    ),
    CurriculumStage(
        # Exact mirror of train_sac_isaac.py's/test_landing_feasibility.py's
        # `orbit_descent` stage (same altitude/speed/tile/episode-length),
        # added 2026-09-30 so the cheap CPU analytic env can be used to
        # cross-check the reward revision (see reward_coefficients.md's
        # 2026-09-30 revision note) against the SAME task the expensive
        # Isaac/GPU training targets, before spending GPU time -- the
        # ZEM/ZEV feasibility controller (lunarsim/control/zemzev_controller.py)
        # already gets 75% landed_safely here with hand-tuned guidance, so
        # this is the stage that matters most for "is RL solving the real
        # target task."
        name="orbit_descent",
        tile_size_m=600.0,
        params=LanderParams(
            spawn_altitude_m=200.0, spawn_xy_radius_m=80.0,
            spawn_v_z_m_s=0.0, spawn_horizontal_speed_m_s=(10.0, 30.0),
            max_episode_s=60.0,
        ),
    ),
]


def _terrain_config(stage: CurriculumStage, seed: int) -> TerrainConfig:
    r = stage.terrain_roughness_scale
    return TerrainConfig(
        mode="fine", size_m=stage.tile_size_m, res_m=max(1.0, stage.tile_size_m / 200.0), seed=seed,
        coarse_source="procedural",
        hills={"amplitude_m": 0.5 * r, "wavelength_m": stage.tile_size_m / 10.0, "hurst": 0.75},
        craters={"count_scale": 0.2 * r, "d_min_m": 1.0, "d_max_m": stage.tile_size_m / 20.0, "b": 2.5,
                 "depth_ratio": 0.08, "age": 0.5},
        rocks={"density_scale": 0.0, "d_max_m": 0.5},
        roi={"sigma_m": stage.tile_size_m / 5.0, "centers": None},
        curvature=False,
    )


def make_env(stage: CurriculumStage, seed: int = 0):
    def _init():
        def tile_fn(rng):
            terrain_seed = int(rng.integers(0, 2**31 - 1))
            return generate_tile(_terrain_config(stage, terrain_seed))
        return AnalyticLanderEnv(tile_fn=tile_fn, params=stage.params, seed=seed)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-stage", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--only-stage", type=str, default=None,
                         help="run one stage, or a comma-separated list of stages in order, for quick iteration")
    parser.add_argument("--keep-buffer", action="store_true",
                         help="don't reset the replay buffer at stage boundaries (default resets -- see comment in main())")
    parser.add_argument("--warm-start", type=str, default=None,
                         help="load this checkpoint .zip as the starting model instead of a fresh one")
    args = parser.parse_args()

    if args.only_stage is None:
        stages = list(STAGES)
    else:
        wanted = args.only_stage.split(",")
        by_name = {s.name: s for s in STAGES}
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            raise SystemExit(f"no stage(s) named {unknown!r}; choices: {list(by_name)}")
        stages = [by_name[n] for n in wanted]

    model = None
    for stage in stages:
        print(f"=== stage: {stage.name} "
              f"(tile={stage.tile_size_m:.0f} m, alt={stage.params.spawn_altitude_m:.0f} m, "
              f"h_speed={stage.params.spawn_horizontal_speed_m_s} m/s) ===")
        vec_env = make_vec_env(make_env(stage), n_envs=args.n_envs)
        if model is None:
            if args.warm_start:
                model = SAC.load(args.warm_start, env=vec_env, device="cpu")
                print(f"warm-started from {args.warm_start}")
                if not args.keep_buffer:
                    model.replay_buffer.reset()
            else:
                # MlpPolicy is small/CPU-bound; SB3 itself warns GPU underperforms here.
                model = SAC("MlpPolicy", vec_env, verbose=1, device="cpu")
        else:
            model.set_env(vec_env)
            if not args.keep_buffer:
                # REAL BUG, FOUND VIA A FAILED RUN: SB3's replay buffer
                # stores the reward VALUE at collection time, not a
                # recipe for recomputing it. `hover_only_easy` (relaxed
                # safety thresholds) -> `hover_only` (real thresholds)
                # went from 1/40 to 0/40 landed and touchdown |vz| got
                # WORSE (7.80 -> 8.23), not better -- stale transitions
                # whose terminal reward was computed under the OLD
                # (easier) threshold definition kept getting sampled
                # alongside new ones scored under the tightened
                # definition, corrupting the critic's targets. Reset by
                # default whenever a stage boundary is crossed; pass
                # --keep-buffer only when you've checked the stages
                # share the same reward definition (only spawn/tile
                # conditions differ) and actually want the carryover.
                model.replay_buffer.reset()
        model.learn(total_timesteps=args.steps_per_stage, reset_num_timesteps=False)
        out_path = f"{args.out_dir}/sac_lunar_lander_{stage.name}.zip"
        model.save(out_path)
        print(f"saved: {out_path}")

    # sanity rollout at the hardest stage trained this run
    hardest = stages[-1]
    env = make_env(hardest)()
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(4000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"[{hardest.name}] episode ended: landed_safely={info.get('landed_safely')}, "
                  f"reward={total_reward:.1f}, t_s={info.get('t_s'):.1f}")
            break


if __name__ == "__main__":
    main()
