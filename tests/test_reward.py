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


def test_climbing_costs_more_than_hovering_and_overspeed_costs_more_than_in_envelope():
    """The two degenerate modes this project measured, in one test.

    Both were real: policies that climbed away under full throttle to
    680-1372 m, and policies that free-fell into the ground at 7.6x the
    touchdown limit. The rewrite answers both with one envelope -- vertical
    speed must be a descent, and no faster than `profile_c*sqrt(alt)` --
    so the costs have to order climb > hover == in-envelope < overspeed.
    Hovering and a legal descent being EQUAL per step is deliberate: what
    makes the vehicle go down is the flat time cost, not an altitude term.
    An altitude term is what made diving optimal before (corr(episode
    length, shaping) = -0.927, i.e. it scored duration, not quality).
    """
    from lunarsim.rl.reward import _descent_envelope_penalty, descent_envelope_m_s

    w = RewardWeights()
    alt = 20.0
    env_v = descent_envelope_m_s(w, alt)

    climbing = _descent_envelope_penalty(w, alt, +2.0)
    hovering = _descent_envelope_penalty(w, alt, 0.0)
    in_envelope = _descent_envelope_penalty(w, alt, -0.5 * env_v)
    overspeed = _descent_envelope_penalty(w, alt, -2.0 * env_v)

    assert hovering == in_envelope == 0.0
    assert climbing > hovering
    assert overspeed > hovering
    # and the faster you overshoot the envelope, the worse it gets -- the
    # old braking term saturated instead, leaving no gradient to brake on
    assert _descent_envelope_penalty(w, alt, -3.0 * env_v) > overspeed


def test_descent_envelope_follows_the_measured_controller_profile():
    """The envelope's sqrt shape is not an assumption. The ZemZev
    controller, which lands on real Isaac Sim, flies vz = -0.276*sqrt(alt)
    with a spread of 0.275-0.278 from 198 m down to 10 m over 5360
    telemetry points. The envelope has to be that shape and has to leave
    the proven profile INSIDE it at every altitude -- a reward that
    penalises the one trajectory known to work cannot teach it.
    """
    from lunarsim.rl.reward import _descent_envelope_penalty, descent_envelope_m_s

    w = RewardWeights()
    for alt in (200.0, 100.0, 20.0, 5.0, 1.0):
        controller_vz = -0.276 * alt ** 0.5
        assert abs(controller_vz) < descent_envelope_m_s(w, alt)
        assert _descent_envelope_penalty(w, alt, controller_vz) == 0.0

    # and it must bring the vehicle in under the touchdown limit on its own
    assert descent_envelope_m_s(w, 1.0) < LanderParams().safe_landing_v_z_m_s


def test_descent_envelope_is_continuous_through_zero_vertical_speed():
    """The term it replaces had a real discontinuity here: going from
    vz=-1e-7 to -1e-3 moved its time-to-ground estimate 15.71 s -> 200000 s
    and the penalty 400 -> 0, so a stalled vehicle shed the whole term by
    sinking 1 cm/s. Nothing may be gained by creeping across vz=0.
    """
    from lunarsim.rl.reward import _descent_envelope_penalty

    w = RewardWeights()
    vals = [_descent_envelope_penalty(w, 200.0, vz)
            for vz in (-1e-3, -1e-7, 0.0, 1e-7, 1e-3)]
    assert max(vals) - min(vals) < 1e-3


def test_horizontal_speed_penalty_depends_on_nothing_but_horizontal_speed():
    """The term this replaces coupled lateral speed to the vertical
    channel through a time-to-ground estimate, which let the penalty
    collapse to zero for a vehicle that was barely sinking. Lateral speed
    is bad near the ground whatever the vertical channel is doing.
    """
    from lunarsim.rl.reward import _horizontal_speed_penalty

    w = RewardWeights()
    assert _horizontal_speed_penalty(w, 10.0) > _horizontal_speed_penalty(w, 1.0)
    assert _horizontal_speed_penalty(w, 0.0) == 0.0


def test_climbing_away_is_never_cheaper_than_staying_put():
    """The measured exploit, end to end: a policy climbed from 198 m to
    350 m+ under throttle 0.92-0.97 and sat there, because at the time
    climbing was cheaper than holding station. Checked on the whole
    per-step shaping sum, not one term, since that is what the policy
    actually optimises.
    """
    params = LanderParams()
    reward_fn = make_apollo_reward_fn(RewardWeights())

    holding = _step_reward(reward_fn, params, 100.0, vz=0.0)
    climbing = _step_reward(reward_fn, params, 100.0, vz=5.0)
    assert climbing < holding


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
