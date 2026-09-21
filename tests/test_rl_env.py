import numpy as np
import pytest

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.rl import AnalyticLanderEnv, LanderParams


def _flat_tile():
    cfg = TerrainConfig(
        mode="fine", size_m=200.0, res_m=1.0, seed=1,
        coarse_source="procedural",
        hills={"amplitude_m": 0.0, "wavelength_m": 50.0, "hurst": 0.75},
        craters={"count_scale": 0.0, "d_min_m": 1.0, "d_max_m": 10.0, "b": 2.5,
                 "depth_ratio": 0.1, "age": 0.5},
        rocks={"density_scale": 0.0, "d_max_m": 0.5},
        roi={"sigma_m": 40.0, "centers": None},
        curvature=False,
    )
    return generate_tile(cfg)


def test_env_conforms_to_gym_api():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_zero_throttle_falls_under_lunar_gravity():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    vz_start = env.state["vz"]
    action = np.array([-1.0, 0.0, 0.0])  # throttle=0
    for _ in range(10):
        env.step(action)
    vz_end = env.state["vz"]
    assert vz_end < vz_start  # accelerating downward


def test_full_throttle_upright_decelerates_descent():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    action = np.array([1.0, 0.0, 0.0])  # full throttle, no gimbal
    vz_history = []
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(action)
        vz_history.append(env.state["vz"])
        if terminated or truncated:
            break
    # with max_thrust_n=3000 on mass=1500kg -> accel = 2.0 m/s^2 upward net of gravity
    # (thrust/mass - g = 3000/1500 - 1.62 = 0.38 m/s^2), so vz should trend upward
    assert vz_history[-1] > vz_history[0]


def test_episode_terminates_on_ground_contact():
    env = AnalyticLanderEnv(_flat_tile(), params=LanderParams(spawn_altitude_m=2.0, dt_s=0.1), seed=0)
    env.reset(seed=0)
    action = np.array([-1.0, 0.0, 0.0])  # free-fall
    terminated = False
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert terminated
    assert info["altitude_m"] == pytest.approx(0.0, abs=1e-6)


def test_gentle_descent_lands_safely():
    params = LanderParams(spawn_altitude_m=10.0, spawn_v_z_m_s=0.0, spawn_xy_radius_m=0.0, max_episode_s=30.0)
    env = AnalyticLanderEnv(_flat_tile(), params=params, seed=0)
    env.reset(seed=0)
    # zero out reset()'s small random initial tilt: with zero gimbal input (this
    # test drives an open-loop constant action) any nonzero tilt causes a real,
    # physically-correct lateral drift under thrust that a real controller
    # would need the gimbal to correct -- not what this test is checking.
    env.state["tilt_x"] = 0.0
    env.state["tilt_y"] = 0.0
    # throttle set so thrust ~= weight (hover), slight negative bias to descend slowly
    hover_throttle = params.mass_kg * params.gravity_m_s2 / params.max_thrust_n
    action = np.array([2 * (hover_throttle * 0.97) - 1.0, 0.0, 0.0])
    landed_safely = False
    for _ in range(600):
        _, _, terminated, truncated, info = env.step(action)
        if terminated:
            landed_safely = info["landed_safely"]
            break
        if truncated:
            break
    assert landed_safely


def test_custom_reward_fn_is_used():
    calls = []

    def custom_reward(env, info):
        calls.append(1)
        return 42.0

    env = AnalyticLanderEnv(_flat_tile(), reward_fn=custom_reward, seed=0)
    env.reset(seed=0)
    _, reward, *_ = env.step(env.action_space.sample())
    assert reward == 42.0
    assert len(calls) == 1


def test_lidar_obs_changes_dimensionality():
    env_no_lidar = AnalyticLanderEnv(_flat_tile(), use_lidar_obs=False, seed=0)
    env_lidar = AnalyticLanderEnv(_flat_tile(), use_lidar_obs=True, lidar_n_rays=6, seed=0)
    obs1, _ = env_no_lidar.reset(seed=0)
    obs2, _ = env_lidar.reset(seed=0)
    assert obs1.shape[0] == 11
    assert obs2.shape[0] == 17
