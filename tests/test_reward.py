import numpy as np
import pytest

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.rl import AnalyticLanderEnv, LanderParams, RewardWeights, make_apollo_reward_fn


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


def test_default_reward_is_finite_over_a_rollout():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    for _ in range(200):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_large_tilt_is_penalized_even_with_zero_rcs_effort():
    """The original handed-over spec only penalized RCS *effort*, not
    attitude *error* -- a policy that stops firing RCS paid zero penalty
    while tumbling. This is the regression test for the fix."""
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    env.state["tilt_x"] = env.state["tilt_y"] = 0.0
    _, reward_upright, *_ = env.step(np.array([-1.0, 0.0, 0.0, 0.0]))

    env.reset(seed=0)
    env.state["tilt_x"] = np.deg2rad(45.0)
    env.state["tilt_y"] = 0.0
    _, reward_tilted, *_ = env.step(np.array([-1.0, 0.0, 0.0, 0.0]))  # still zero RCS effort

    assert reward_tilted < reward_upright


def test_more_rcs_effort_is_penalized_more():
    env = AnalyticLanderEnv(_flat_tile(), seed=0)
    env.reset(seed=0)
    env.state["tilt_x"] = env.state["tilt_y"] = 0.0
    _, reward_no_rcs, *_ = env.step(np.array([-1.0, 0.0, 0.0, 0.0]))

    env.reset(seed=0)
    env.state["tilt_x"] = env.state["tilt_y"] = 0.0
    _, reward_full_rcs, *_ = env.step(np.array([-1.0, 1.0, 1.0, 1.0]))

    assert reward_full_rcs < reward_no_rcs


class _StubEnv:
    """Minimal (params, state) stand-in so the reward function's terminal
    logic can be unit-tested directly, without depending on exactly how
    many integrator steps it takes AnalyticLanderEnv to cross the ground
    at a given impact velocity."""

    def __init__(self, params, state):
        self.params = params
        self.state = state


def _base_state(**overrides):
    state = {
        "x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0, "yaw": 0.0, "wx": 0.0, "wy": 0.0, "wz": 0.0,
        "fuel_kg": 100.0, "rcs_fuel_kg": 10.0, "throttle": 0.0,
        "rcs_pitch": 0.0, "rcs_roll": 0.0, "rcs_yaw": 0.0,
    }
    state.update(overrides)
    return state


def test_crash_penalty_worse_than_safe_landing_reward():
    reward_fn = make_apollo_reward_fn(RewardWeights())
    params = LanderParams()
    env = _StubEnv(params, _base_state())

    reward_safe = reward_fn(env, {
        "terminated": True, "landed_safely": True,
        "landing_margins": {"v_z": 0.9, "v_xy": 0.9, "tilt": 0.9, "w": 0.9},
        "altitude_m": 0.0, "leg_force_n": 1000.0, "leg_force_max_n": 8000.0,
    })
    reward_crash = reward_fn(env, {
        "terminated": True, "landed_safely": False,
        "landing_margins": {},
        "altitude_m": 0.0, "leg_force_n": 20000.0, "leg_force_max_n": 8000.0,
    })
    assert reward_crash < reward_safe


def test_timeout_without_landing_is_penalized_not_free():
    """Regression test for the 'hover forever to dodge crash risk' exploit
    found via a real training run (39/40 episodes timed out airborne once
    timeout carried zero penalty)."""
    reward_fn = make_apollo_reward_fn(RewardWeights())
    env = _StubEnv(LanderParams(), _base_state())

    reward_still_flying = reward_fn(env, {
        "terminated": False, "truncated": False, "landed_safely": False,
        "landing_margins": {}, "altitude_m": 20.0, "leg_force_n": 0.0, "leg_force_max_n": 8000.0,
    })
    reward_timeout = reward_fn(env, {
        "terminated": False, "truncated": True, "landed_safely": False,
        "landing_margins": {}, "altitude_m": 20.0, "leg_force_n": 0.0, "leg_force_max_n": 8000.0,
    })
    assert reward_timeout < reward_still_flying


def test_softer_landing_scores_higher_than_marginal_landing():
    reward_fn = make_apollo_reward_fn(RewardWeights())
    params = LanderParams()
    env = _StubEnv(params, _base_state())

    def terminal(margin):
        return reward_fn(env, {
            "terminated": True, "landed_safely": True,
            "landing_margins": {"v_z": margin, "v_xy": margin, "tilt": margin, "w": margin},
            "altitude_m": 0.0, "leg_force_n": 1000.0, "leg_force_max_n": 8000.0,
        })

    assert terminal(0.95) > terminal(0.05)
