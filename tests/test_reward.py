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


def _step_reward(reward_fn, params, alt_m, **state_overrides):
    env = _StubEnv(params, _base_state(**state_overrides))
    return reward_fn(env, {
        "terminated": False, "truncated": False, "landed_safely": False,
        "landing_margins": {}, "altitude_m": alt_m,
        "leg_force_n": 0.0, "leg_force_max_n": 8000.0,
    })


def test_climbing_costs_more_than_hovering_which_costs_more_than_descending():
    """Regression test for the sign-blind altitude gate (see
    `altitude_vz_gate_k`'s field comment). `_braking_ratio` squares vz, so
    the gate used to open identically for a fast CLIMB and a fast descent,
    and -- because every other vz-dependent term is also even in vz -- the
    whole per-step shaping sum was an even function of vz. Measured before
    the fix at alt=100 m: cost(vz=-10) == cost(vz=+10) == 249.63, both less
    than half of cost(vz=0) == 507.75, i.e. rocketing away was strictly
    cheaper than holding station. The ordering asserted here is the whole
    point of the term: going up must never be cheaper than standing still.
    """
    reward_fn = make_apollo_reward_fn(RewardWeights())
    params = LanderParams()

    climb = _step_reward(reward_fn, params, 100.0, vz=+10.0)
    hover = _step_reward(reward_fn, params, 100.0, vz=0.0)
    descend = _step_reward(reward_fn, params, 100.0, vz=-10.0)

    assert climb < hover < descend


def test_altitude_gate_descent_side_is_unchanged_by_the_climb_fix():
    """The gate exists to stop the 'just dive straight in' failure, so the
    fix above must not weaken it on the descent side -- only on the climb
    side. A fast descent must still relax the descend incentive relative to
    hovering (that is what makes the braking term the dominant signal once
    vz is eating into the stopping budget)."""
    from lunarsim.rl.reward import _altitude_penalty, _braking_ratio

    w = RewardWeights()
    p = LanderParams()
    ratio = _braking_ratio(w, 100.0, -10.0, 15103.0, p.dps_thrust_max_n, p.gravity_m_s2)
    gated = _altitude_penalty(w, 100.0, ratio)
    ungated = _altitude_penalty(w, 100.0, 0.0)
    assert gated < ungated


def test_time_to_ground_is_continuous_through_zero_vertical_speed():
    """Regression test for the two-branch `_time_to_ground_s` (see its
    docstring). The old descending branch was `alt/|vz|`, unbounded as
    |vz| -> 0, so at alt=200 m tgo jumped 15.71 s -> 200000 s between
    vz=-1e-7 and vz=-1e-3, and `_xy_braking_envelope_penalty` collapsed
    400.00 -> 0.00 across that same hair's breadth: a stalled vehicle shed
    the entire horizontal-braking penalty by sinking one cm/s."""
    from lunarsim.rl.reward import _time_to_ground_s

    g = LanderParams().gravity_m_s2
    at_zero = _time_to_ground_s(200.0, 0.0, g, 2.0)

    # exactly the free-fall-from-rest value the old vz>=0 branch returned
    assert at_zero == pytest.approx(float(np.sqrt(2.0 * 200.0 / g)), rel=1e-9)

    for vz in (-1e-3, -1e-2, -0.1, -1.0):
        assert _time_to_ground_s(200.0, vz, g, 2.0) < at_zero
    for vz in (+1e-3, +1e-2, +0.1, +1.0):
        assert _time_to_ground_s(200.0, vz, g, 2.0) > at_zero

    # continuity: no step change across vz=0
    left = _time_to_ground_s(200.0, -1e-6, g, 2.0)
    right = _time_to_ground_s(200.0, +1e-6, g, 2.0)
    assert abs(left - right) < 1e-3


def test_horizontal_braking_penalty_does_not_collapse_when_barely_sinking():
    """The behavioural half of the test above: a vehicle carrying 25 m/s of
    horizontal speed at 200 m must not be able to zero out the horizontal
    braking term by starting a 1 cm/s sink."""
    from lunarsim.rl.reward import _xy_braking_envelope_penalty, _xy_braking_ratio

    w = RewardWeights()
    p = LanderParams()

    def penalty(vz):
        ratio = _xy_braking_ratio(w, 200.0, vz, 25.0, 15103.0,
                                   p.dps_thrust_max_n, p.gravity_m_s2)
        return _xy_braking_envelope_penalty(w, ratio)

    hanging = penalty(0.0)
    barely_sinking = penalty(-0.01)
    assert hanging > 0.0
    assert barely_sinking == pytest.approx(hanging, rel=0.05)


def test_climbing_away_is_not_cheaper_than_staying_put_on_the_xy_braking_term():
    """The old `_time_to_ground_s` used free-fall-from-rest for vz>=0, so
    climbing lengthened tgo and made the horizontal-braking term CHEAPER
    the further the vehicle ran away (400 at 200 m down to 63.6 at 1372 m).
    The ballistic solve keeps climbing expensive in the only sense this
    term can express: it must not pay to leave."""
    from lunarsim.rl.reward import _xy_braking_envelope_penalty, _xy_braking_ratio

    w = RewardWeights()
    p = LanderParams()

    def penalty(alt, vz):
        ratio = _xy_braking_ratio(w, alt, vz, 25.0, 15103.0,
                                   p.dps_thrust_max_n, p.gravity_m_s2)
        return _xy_braking_envelope_penalty(w, ratio)

    # the full per-step sum is what actually has to order correctly, and
    # `_altitude_penalty` is what carries the climb deterrent -- assert the
    # combined behaviour rather than this one term in isolation.
    reward_fn = make_apollo_reward_fn(w)
    low = _step_reward(reward_fn, p, 200.0, vz=+20.0, vx=25.0)
    high = _step_reward(reward_fn, p, 1372.0, vz=+20.0, vx=25.0)
    assert high < low


def test_spinning_is_penalized_and_yaw_rate_is_not_free():
    """Regression test for the missing angular-rate term (see `omega_k`'s
    field comment). `landed_safely` rejects on |w| > safe_landing_w_rad_s,
    but no per-step term read wx/wy/wz at all, and `_attitude_hold_penalty`
    reads tilt ANGLES only -- never yaw. A real measured episode came back
    CRASH with vz=-0.21, vxy=0.48, tilt=7.1 and w=0.65: every other
    criterion inside its limit, rejected purely on rate.
    """
    reward_fn = make_apollo_reward_fn(RewardWeights())
    params = LanderParams()

    still = _step_reward(reward_fn, params, 2.0)
    spinning = _step_reward(reward_fn, params, 2.0, wx=0.65)
    assert spinning < still

    # yaw rate alone used to cost exactly nothing -- no term read wz
    yawing = _step_reward(reward_fn, params, 2.0, wz=0.65)
    assert yawing < still


def test_angular_rate_penalty_is_negligible_below_the_limit_and_steep_at_it():
    """The term must not tax the ~0.3 rad/s rotation a real braking tilt
    maneuver needs (measured: ordinary flight ran 0.01-0.07 rad/s, the
    rejection line is 0.5). A flat quadratic strong enough to matter at
    0.5 would have made normal maneuvering expensive, so this is shaped
    like the tilt cutoff: gentle everywhere, a wall at the limit."""
    from lunarsim.rl.reward import _angular_rate_penalty

    w = RewardWeights()
    limit = LanderParams().safe_landing_w_rad_s

    gentle = _angular_rate_penalty(w, 0.1, 0.0, 0.0, limit)
    maneuver = _angular_rate_penalty(w, 0.3, 0.0, 0.0, limit)
    at_limit = _angular_rate_penalty(w, limit, 0.0, 0.0, limit)

    assert gentle < 1.0
    assert maneuver < 0.1 * at_limit
    assert at_limit > 100.0
    # capped, so a pathological tumble cannot swamp the whole shaping sum
    assert _angular_rate_penalty(w, 10.0, 10.0, 10.0, limit) == pytest.approx(w.omega_penalty_cap)


def test_angular_rate_penalty_counts_all_three_axes_equally():
    """|w| is a magnitude -- the landing test uses the norm, so the shaping
    must too, or the policy can hide rate in whichever axis is cheapest."""
    from lunarsim.rl.reward import _angular_rate_penalty

    w = RewardWeights()
    limit = LanderParams().safe_landing_w_rad_s
    per_axis = [_angular_rate_penalty(w, *axis, limit) for axis in
                ((0.4, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 0.4))]
    assert per_axis[0] == pytest.approx(per_axis[1]) == pytest.approx(per_axis[2])
