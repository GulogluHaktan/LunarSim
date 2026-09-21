"""Proves the RL interface end-to-end: trains a real Stable-Baselines3 PPO
policy on `AnalyticLanderEnv` (no Isaac needed -- this is the fast CPU
backend, see lunarsim/rl/). Any other Gymnasium-compatible algorithm/library
(SB3 SAC/TD3, CleanRL, RLlib, a custom loop) plugs in the same way, since
AnalyticLanderEnv is a standard gymnasium.Env.

Usage: .venv/bin/python scripts/train_ppo_analytic.py --steps 20000
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--out", type=str, default="out/ppo_lunar_lander.zip")
    args = parser.parse_args()

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    print(f"saved policy to {args.out}")

    # quick sanity rollout
    env = make_env()
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"episode ended: landed_safely={info.get('landed_safely')}, reward={total_reward:.1f}")
            break


if __name__ == "__main__":
    main()
