"""SAC training against `lunarsim.adapters.isaac.isaac_lander_vec_env.
IsaacLanderVecEnv` -- the real-PhysX-backed, parallel, per-episode-terrain
lander env, not the hand-integrated `AnalyticLanderEnv`
`scripts/train_sac_curriculum.py` trains against.

Parallelism: `--n-envs` really does run N vehicles simultaneously in the
ONE live PhysX scene (see `IsaacLanderVecEnv`'s module docstring -- this is
intra-process PhysX batching, not N Isaac Sim processes), each on its own
terrain tile, each independently re-generating that tile via `tile_fn`
every time IT resets (not synchronized across envs) -- the real-physics
analogue of `train_sac_curriculum.py`'s `tile_fn`-per-episode curriculum.

Run: /home/haktan/isaac-env/bin/python scripts/train_sac_isaac.py --steps-per-stage 50000
(or the equivalent `/isaac-sim/python.sh` inside the docker image).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--steps-per-stage", type=int, default=50_000)
parser.add_argument("--n-envs", type=int, default=16,
                     help="real parallel vehicles in the one PhysX scene -- keep this within what the "
                          "local GPU's VRAM/compute can hold; there's no subprocess-per-env cost to trade off "
                          "against here (see IsaacLanderVecEnv's module docstring), so scale it up until step "
                          "throughput stops improving, not to match core count.")
parser.add_argument("--out-dir", type=str, default="out")
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--max-rocks-per-env", type=int, default=10,
                     help="real collidable rocks spawned per env per reset (from the terrain's own "
                          "tile.rocks field) -- capped since a full reference-density rock field can be "
                          "dozens-hundreds of prims, and this authors them fresh every episode reset.")
parser.add_argument("--only-stage", type=str, default=None,
                     help="run one stage, or a comma-separated list of stages in order")
parser.add_argument("--warm-start", type=str, default=None,
                     help="load this checkpoint .zip as the starting model instead of a fresh one")
parser.add_argument("--keep-buffer", action="store_true",
                     help="don't reset the replay buffer at stage boundaries -- see "
                          "train_sac_curriculum.py's comment on why this defaults off")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, args.lunarsim_root)

from stable_baselines3 import SAC  # noqa: E402

from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
from lunarsim.core.terrain.config import TerrainConfig  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl import LanderParams  # noqa: E402


@dataclass
class Stage:
    name: str
    tile_size_m: float
    params: LanderParams = field(default_factory=LanderParams)
    terrain_roughness_scale: float = 1.0


#  REMOVED THIS SESSION, PER USER DIRECTION: the hover_only_easy ->
#  hover_only -> final_approach curriculum stages that used to precede
#  orbit_descent. Rationale from handover.md's "CURRICULUM-SHAPE
#  hypothesis": warm-starting orbit_descent (200m/10-30 m/s) from
#  final_approach (35m/0-3 m/s) is a cliff, not a smooth progression --
#  ~13M cumulative orbit_descent steps on top of that warm start never
#  reached landed_safely>0/16, with ent_coef decayed flat (converged to a
#  local optimum, not still searching). Combined with this session's
#  terminal-reward-scale fix (see reward.py's module docstring -- the
#  terminal signal was previously ~100x too small to matter against a
#  whole episode's accumulated shaping), training now runs orbit_descent
#  directly, warm-started from the existing orbit_descent checkpoint
#  itself (not from final_approach), so the policy gets a correctly-scaled
#  terminal signal without re-inheriting the cliff-adjacent weights/decayed
#  entropy a final_approach warm start would carry in. If this alone still
#  doesn't converge, the next thing to try is a speed sub-curriculum
#  WITHIN orbit_descent (see handover.md item 2), not reintroducing these
#  removed stages.
STAGES = [
    # "yorunge" stage: an uncontrolled-release-scale altitude/horizontal-
    # speed regime, matching what scripts/isaaclab_static_telemetry_capture.py's
    # actual demo descents used (120-350m release, real craters/hills/rocks
    # at full roughness) -- NOT literal orbital mechanics, same scope
    # caveat as everywhere else in this codebase (see analytic_lander_env's
    # module docstring). This is the hardest, most "final descent"-like
    # stage: real obstacles (rocks, via IsaacLanderVecEnv's tile.rocks
    # spawning), a real crater/hill field, and a tile sized to actually fit
    # the resulting ballistic ground track (fall time from 200m is ~15.7s;
    # at up to 30 m/s horizontal that's ~470m of drift, so needs real room).
    Stage(
        name="orbit_descent",
        tile_size_m=600.0,
        params=LanderParams(
            spawn_altitude_m=200.0, spawn_xy_radius_m=80.0,
            spawn_v_z_m_s=0.0, spawn_horizontal_speed_m_s=(10.0, 30.0),
            max_episode_s=60.0,
        ),
    ),
]


def _terrain_config(stage: Stage, seed: int) -> TerrainConfig:
    r = stage.terrain_roughness_scale
    return TerrainConfig(
        mode="fine", size_m=stage.tile_size_m, res_m=max(0.5, stage.tile_size_m / 80.0), seed=seed,
        coarse_source="procedural",
        hills={"amplitude_m": 0.3 * r, "wavelength_m": stage.tile_size_m / 6.0, "hurst": 0.75},
        craters={"count_scale": 0.1 * r, "d_min_m": 1.0, "d_max_m": stage.tile_size_m / 10.0, "b": 2.5,
                 "depth_ratio": 0.08, "age": 0.5},
        # real, PHYSICAL (collidable) rocks now -- see IsaacLanderVecEnv's
        # `_rebuild_terrain` docstring on the real `tile.rocks` field this
        # spawns from (capped per-env regardless of density, for reset cost).
        rocks={"density_scale": r, "d_max_m": 1.0},
        roi={"sigma_m": stage.tile_size_m / 4.0, "centers": None},
        curvature=False,
    )


def _make_tile_fn(stage: Stage):
    def tile_fn(rng: np.random.Generator):
        terrain_seed = int(rng.integers(0, 2**31 - 1))
        return generate_tile(_terrain_config(stage, terrain_seed))
    return tile_fn


def main():
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
    venv = None
    for stage in stages:
        print(f"=== stage: {stage.name} (tile={stage.tile_size_m:.0f} m, alt={stage.params.spawn_altitude_m:.0f} m, "
              f"h_speed={stage.params.spawn_horizontal_speed_m_s} m/s, n_envs={args.n_envs}) ===")
        tile_fn = _make_tile_fn(stage)
        if venv is None:
            # the PhysX scene (N vehicles + N terrain slots) is built once,
            # here, by the first IsaacLanderVecEnv -- later stages reuse the
            # SAME instance (swap `.params`/`.tile_fn`; env spacing has to
            # fit the LARGEST stage's tile, so it's sized off STAGES, not
            # just this stage, or a later bigger-tile stage would overlap
            # neighboring env slots).
            max_tile = max(s.tile_size_m for s in STAGES)
            venv = IsaacLanderVecEnv(
                num_envs=args.n_envs, tile_fn=tile_fn, params=stage.params,
                env_spacing_m=max_tile * 1.5, seed=args.terrain_seed,
                max_rocks_per_env=args.max_rocks_per_env,
            )
        else:
            venv.params = stage.params
            venv.tile_fn = tile_fn
            venv.reset()

        if model is None:
            if args.warm_start:
                model = SAC.load(args.warm_start, env=venv, device="cpu")
                print(f"warm-started from {args.warm_start}")
                if not args.keep_buffer:
                    model.replay_buffer.reset()
            else:
                model = SAC("MlpPolicy", venv, verbose=1, device="cpu")
        else:
            model.set_env(venv)
            if not args.keep_buffer:
                model.replay_buffer.reset()

        model.learn(total_timesteps=args.steps_per_stage, reset_num_timesteps=False)
        out_path = f"{args.out_dir}/sac_lunar_lander_isaac_{stage.name}.zip"
        model.save(out_path)
        print(f"saved: {out_path}")

    hardest = stages[-1]
    venv.params = hardest.params
    venv.tile_fn = _make_tile_fn(hardest)
    obs = venv.reset()
    total_reward = np.zeros(args.n_envs)
    done_once = np.zeros(args.n_envs, dtype=bool)
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        venv.step_async(action)
        obs, rewards, dones, infos = venv.step_wait()
        total_reward += rewards * (~done_once)
        for i in range(args.n_envs):
            if dones[i] and not done_once[i]:
                done_once[i] = True
                print(f"[{hardest.name}] env {i} episode ended: landed_safely={infos[i].get('landed_safely')}, "
                      f"lost_control={infos[i].get('lost_control')}, reward={total_reward[i]:.1f}, "
                      f"t_s={infos[i].get('t_s'):.1f}")
        if done_once.all():
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
