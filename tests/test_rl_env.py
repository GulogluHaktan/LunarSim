import numpy as np
import pytest

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.rl import AnalyticLanderEnv, LanderParams
from lunarsim.rl.action_map import throttle_to_action


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
    action = np.array([-1.0, 0.0, 0.0, 0.0])  # throttle=0, no RCS
    for _ in range(10):
        env.step(action)
    vz_end = env.state["vz"]
    assert vz_end < vz_start  # accelerating downward


def test_full_throttle_upright_decelerates_descent():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    env.state["tilt_x"] = 0.0
    env.state["tilt_y"] = 0.0
    action = np.array([1.0, 0.0, 0.0, 0.0])  # full DPS throttle, no RCS
    vz_history = []
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(action)
        vz_history.append(env.state["vz"])
        if terminated or truncated:
            break
    # at full mass (dry + full descent propellant), DPS max thrust gives a
    # thrust-to-weight ratio > 1 (real Apollo LM DPS: ~45 kN vs ~24.5 kN
    # weight at 15.1 t), so vz should trend upward
    assert vz_history[-1] > vz_history[0]


def test_mass_decreases_as_dps_propellant_burns():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    fuel_start = env.state["fuel_kg"]
    action = np.array([1.0, 0.0, 0.0, 0.0])  # full DPS throttle
    _, _, _, _, info = env.step(action)
    assert env.state["fuel_kg"] < fuel_start
    # mass used for that step's dynamics is mass *before* this step's burn
    assert info["mass_kg"] == pytest.approx(env.params.dry_mass_kg + fuel_start)


def test_rcs_command_tilts_vehicle_and_burns_rcs_propellant():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    env.state["tilt_x"] = 0.0
    env.state["tilt_y"] = 0.0
    env.state["wx"] = 0.0
    rcs_fuel_start = env.state["rcs_fuel_kg"]
    action = np.array([-1.0, 1.0, 0.0, 0.0])  # engine off, full pitch RCS command
    for _ in range(5):
        env.step(action)
    assert env.state["wx"] != 0.0
    assert env.state["rcs_fuel_kg"] < rcs_fuel_start


def test_episode_terminates_on_ground_contact():
    env = AnalyticLanderEnv(_flat_tile(), params=LanderParams(spawn_altitude_m=2.0, dt_s=0.1), seed=0)
    env.reset(seed=0)
    action = np.array([-1.0, 0.0, 0.0, 0.0])  # free-fall
    terminated = False
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert terminated
    assert info["altitude_m"] == pytest.approx(0.0, abs=1e-6)


def test_gentle_descent_lands_safely():
    params = LanderParams(spawn_altitude_m=10.0, spawn_v_z_m_s=0.0, spawn_xy_radius_m=0.0, max_episode_s=60.0)
    env = AnalyticLanderEnv(_flat_tile(), params=params, seed=0)
    env.reset(seed=0)
    # zero out reset()'s small random initial tilt: with zero RCS input (this
    # test drives an open-loop constant action) any nonzero tilt causes a real,
    # physically-correct lateral drift under thrust that a real controller
    # would need RCS to correct -- not what this test is checking.
    env.state["tilt_x"] = 0.0
    env.state["tilt_y"] = 0.0
    # throttle set so DPS thrust ~= weight (hover), slight negative bias to descend slowly
    mass0 = params.dry_mass_kg + params.initial_fuel_kg
    weight = mass0 * params.gravity_m_s2
    hover_throttle = (weight - params.dps_thrust_min_n) / (params.dps_thrust_max_n - params.dps_thrust_min_n)
    # `throttle_to_action`, not a hand-rolled `2*t - 1`: the env's action[0]
    # -> throttle curve is not linear (see lunarsim/rl/action_map.py), and an
    # inverse written out here is just one more copy to drift. This caught it
    # for real -- under the new curve the old expression asked for throttle
    # 0.4951 instead of 0.4757, close enough to hover that the vehicle no
    # longer reached the ground inside the episode.
    action = np.array([throttle_to_action(hover_throttle * 0.97), 0.0, 0.0, 0.0])
    landed_safely = False
    for _ in range(1200):
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


def test_loss_of_control_ends_episode_immediately():
    env = AnalyticLanderEnv(_flat_tile(), params=LanderParams(spawn_altitude_m=500.0), seed=0)
    env.reset(seed=0)
    env.state["tilt_x"] = np.deg2rad(59.0)  # just under the cutoff
    env.state["wx"] = 10.0  # large angular rate pushes it over the cutoff next step
    _, reward, terminated, truncated, info = env.step(np.array([-1.0, 0.0, 0.0, 0.0]))
    assert terminated
    assert not truncated
    assert info["lost_control"] is True
    assert info["landed_safely"] is False
    assert info["altitude_m"] > 0.0  # crashed mid-air, not at the ground


def test_tile_fn_regenerates_terrain_each_reset():
    seeds_used = []

    def tile_fn(rng):
        seed = int(rng.integers(0, 1_000_000))
        seeds_used.append(seed)
        cfg = TerrainConfig(
            mode="fine", size_m=200.0, res_m=1.0, seed=seed,
            coarse_source="procedural",
            hills={"amplitude_m": 1.0, "wavelength_m": 50.0, "hurst": 0.75},
            craters={"count_scale": 0.3, "d_min_m": 1.0, "d_max_m": 10.0, "b": 2.5,
                     "depth_ratio": 0.1, "age": 0.5},
            rocks={"density_scale": 0.0, "d_max_m": 0.5},
            roi={"sigma_m": 40.0, "centers": None},
            curvature=False,
        )
        return generate_tile(cfg)

    env = AnalyticLanderEnv(tile_fn=tile_fn, seed=0)
    tile_a = env.reset(seed=0)
    height_a = env.tile.height.copy()
    tile_b = env.reset(seed=1)
    height_b = env.tile.height.copy()

    assert len(seeds_used) == 2
    assert seeds_used[0] != seeds_used[1]
    assert not np.array_equal(height_a, height_b)


def test_env_requires_exactly_one_of_tile_or_tile_fn():
    with pytest.raises(ValueError):
        AnalyticLanderEnv()
    with pytest.raises(ValueError):
        AnalyticLanderEnv(tile=_flat_tile(), tile_fn=lambda rng: _flat_tile())


def test_spawn_horizontal_speed_aims_toward_target():
    params = LanderParams(spawn_xy_radius_m=50.0, spawn_horizontal_speed_m_s=(20.0, 20.0), target_x=0.0, target_y=0.0)
    env = AnalyticLanderEnv(_flat_tile(), params=params, seed=0)
    env.reset(seed=0)
    s = env.state
    speed = float(np.hypot(s["vx"], s["vy"]))
    assert speed == pytest.approx(20.0, rel=1e-3)
    # velocity must point from spawn toward the target, i.e. opposite the
    # spawn radius vector
    dot = s["x"] * s["vx"] + s["y"] * s["vy"]
    assert dot < 0.0


def test_footpad_height_diff_blocks_landing_on_uneven_ground():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    diff = env._footpad_height_diff_m(0.0, 0.0)
    assert diff == pytest.approx(0.0, abs=1e-6)  # flat test tile


def test_leg_force_reported_only_at_touchdown():
    env = AnalyticLanderEnv(_flat_tile(), params=LanderParams(spawn_altitude_m=2.0, dt_s=0.1), seed=0)
    env.reset(seed=0)
    action = np.array([-1.0, 0.0, 0.0, 0.0])
    saw_touchdown = False
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(action)
        if not terminated:
            assert info["leg_force_n"] == 0.0
        else:
            saw_touchdown = True
            assert info["leg_force_n"] > 0.0
            break
        if truncated:
            break
    assert saw_touchdown


def test_lidar_obs_changes_dimensionality():
    env_no_lidar = AnalyticLanderEnv(_flat_tile(), use_lidar_obs=False, seed=0)
    env_lidar = AnalyticLanderEnv(_flat_tile(), use_lidar_obs=True, lidar_n_rays=6, seed=0)
    obs1, _ = env_no_lidar.reset(seed=0)
    obs2, _ = env_lidar.reset(seed=0)
    assert obs1.shape[0] == 16
    assert obs2.shape[0] == 22
