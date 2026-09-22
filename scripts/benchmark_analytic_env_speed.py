"""Steps/s for AnalyticLanderEnv at several parallel env counts (SB3 VecEnv,
CPU-only, no Isaac). Answers "do we need Isaac-style env cloning for this
backend" -- no: it's plain numpy, so scaling is just process/thread
parallelism, not GPU physics batching.

Usage: .venv/bin/python scripts/benchmark_analytic_env_speed.py
"""
import time

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.rl import AnalyticLanderEnv, LanderParams


def make_env():
    cfg = TerrainConfig(
        mode="fine", size_m=200.0, res_m=1.0, seed=7,
        coarse_source="procedural",
        hills={"amplitude_m": 0.5, "wavelength_m": 60.0, "hurst": 0.75},
        craters={"count_scale": 0.3, "d_min_m": 1.0, "d_max_m": 15.0, "b": 2.5,
                 "depth_ratio": 0.08, "age": 0.5},
        rocks={"density_scale": 0.0, "d_max_m": 0.5},
        roi={"sigma_m": 40.0, "centers": None},
        curvature=False,
    )
    tile = generate_tile(cfg)
    return AnalyticLanderEnv(tile, params=LanderParams(spawn_xy_radius_m=15.0))


def benchmark(n_envs: int, n_steps: int = 2000, use_subproc: bool = False):
    vec_env_cls = SubprocVecEnv if use_subproc else None
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=vec_env_cls)
    obs = env.reset()

    actions = [env.action_space.sample() for _ in range(n_envs)]

    # warmup
    for _ in range(50):
        env.step(actions)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(actions)
    dt = time.perf_counter() - t0
    env.close()

    step_calls_per_s = n_steps / dt
    env_steps_per_s = step_calls_per_s * n_envs
    print(f"n_envs={n_envs:4d} ({'subproc' if use_subproc else 'in-process'}): "
          f"{n_steps} vec-steps in {dt:.2f}s -> {step_calls_per_s:.1f} vec-steps/s, "
          f"{env_steps_per_s:.0f} env-steps/s")
    return env_steps_per_s


if __name__ == "__main__":
    print("=== in-process (DummyVecEnv) ===")
    for n in (4, 16, 64):
        benchmark(n, n_steps=1000, use_subproc=False)

    print("\n=== multi-process (SubprocVecEnv) ===")
    for n in (16, 64, 256):
        benchmark(n, n_steps=500, use_subproc=True)
