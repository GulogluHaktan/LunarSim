"""Does the reward actually rank real behaviour the way we need it to?

This file exists because of a repeated failure in this project: four separate
"tried it, did not work" conclusions turned out to have tested an implementation
rather than the hypothesis -- an actor freeze that SB3 silently undid, a
two-timescale test that Adam normalised away, an action-saturation penalty whose
input was never written, and an entropy test run with a broken success label. A
reward redesign taken from the literature deserves the same scepticism: if the
published method works there and not here, the first thing to rule out is that we
built it wrong.

So these tests do not ask whether the reward is good. They ask whether it PREFERS
the behaviour that is measured to succeed, using the controller's own recorded
trajectories as the reference, because that is the one behaviour in this project
known to land 92% of the time.
"""
from __future__ import annotations

import numpy as np
import pytest

from lunarsim.rl.obs_norm import OBS_SCALE
from lunarsim.rl.reward import (RewardWeights, _angular_rate_penalty,
                                 _tilt_cutoff_penalty, _time_penalty,
                                 _velocity_field_penalty, target_velocity)

DEMOS = "out/zemzev_ramp35_demos.npz"


def _load():
    try:
        d = np.load(DEMOS, allow_pickle=True)
    except FileNotFoundError:
        pytest.skip(f"{DEMOS} not present")
    o = d["obs"].astype(np.float64) * OBS_SCALE[:16]
    return o, d["episode_ends"], d["landed"].astype(bool)


def _shaping(w, alt, dx, dy, vx, vy, vz, tilt=0.0, wmag=0.0):
    """Per-step shaping SUM (positive = penalty), exactly as reward_fn builds it."""
    return (_velocity_field_penalty(w, alt, dx, dy, vx, vy, vz)
            + _tilt_cutoff_penalty(w, tilt, 0.0, np.radians(40.0))
            + _angular_rate_penalty(w, wmag, 0.0, 0.0, 0.5)
            + _time_penalty(w))


def test_the_controllers_own_velocity_beats_hovering_diving_and_drifting():
    """At the controller's own states, its own velocity must score best.

    This is the test that caught the first version of the velocity field. That field
    took its vertical target from the descent ENVELOPE, which is a limit rather than a
    schedule: it asked for 4.61 m/s at 35 m where the controller actually descends at
    2.59. Scored against it the controller showed a mean tracking error of 4.64 m/s and
    a dense sum of -249 -- the reward preferred something other than the 92% policy, and
    training on it would have pushed the agent away from what works. Nothing about the
    landing rate would have revealed that for days.
    """
    o, ends, landed = _load()
    w = RewardWeights()
    alt, dx, dy = o[:, 2], o[:, 0], o[:, 1]
    vx, vy, vz = o[:, 3], o[:, 4], o[:, 5]

    # Each alternative must be a GENUINE degradation of the same state, and two of these
    # were not on the first attempt. `vx + 3.0` REDUCES the lateral error whenever vx is
    # negative, so it was an improvement half the time; it is now signed. And zeroing all
    # three velocity components also zeroed the lateral error, comparing the controller
    # against a counterfactual it cannot reach in one step -- "stop descending" has to
    # perturb the vertical channel alone to isolate it.
    idx = np.linspace(0, len(o) - 1, 400).astype(int)
    wins = {"stop descending": 0, "dive": 0, "drift": 0, "climb": 0}
    for i in idx:
        a = alt[i]
        actual = _shaping(w, a, dx[i], dy[i], vx[i], vy[i], vz[i])
        sgn = 1.0 if vx[i] >= 0 else -1.0
        cand = {
            "stop descending": _shaping(w, a, dx[i], dy[i], vx[i], vy[i], 0.0),
            "dive": _shaping(w, a, dx[i], dy[i], vx[i], vy[i], vz[i] - 4.0),
            "drift": _shaping(w, a, dx[i], dy[i], vx[i] + 3.0 * sgn, vy[i], vz[i]),
            "climb": _shaping(w, a, dx[i], dy[i], vx[i], vy[i], vz[i] + 3.0),
        }
        for k, v in cand.items():
            wins[k] += actual < v

    n = len(idx)
    for name, k in wins.items():
        assert k > 0.9 * n, (
            f"the controller's own velocity loses to {name} at {n - k}/{n} sampled "
            f"states -- the reward does not prefer the behaviour that lands 92%")


def test_the_controllers_tracking_error_is_small_against_the_field():
    """The reference policy must sit near the bottom of the dense term.

    A large error here means the field is misspecified, not that the controller is bad.
    4.64 m/s was the mis-set version; 1.90 is the calibrated one, and most of what
    remains is the spawn lateral velocity the controller has to bleed off.
    """
    o, ends, landed = _load()
    w = RewardWeights()
    alt = o[:, 2]
    vx, vy, vz = o[:, 3], o[:, 4], o[:, 5]
    err = np.empty(len(o))
    for i in range(len(o)):
        vxt, vyt, vzt, _ = target_velocity(w, alt[i])
        err[i] = np.sqrt((vx[i] - vxt) ** 2 + (vy[i] - vyt) ** 2 + (vz[i] - vzt) ** 2)
    assert err.mean() < 2.5, f"mean tracking error {err.mean():.2f} m/s -- field misspecified"
    assert np.median(err) < 2.0, f"median tracking error {np.median(err):.2f} m/s"


def test_landing_outranks_stalling_and_stalling_outranks_slamming():
    """The whole-episode ordering, end to end, including the terminals.

    The old reward had this inverted: a crash cost -60 against a timeout's -105, so
    turning a hover into a crash was a +45 improvement, and the RL policy duly produced
    23 crashes where the clone had 10 while landing no more often. The ordering must come
    out of the weights rather than being asserted by hand, so this computes it from the
    reward's own terms.
    """
    from lunarsim.rl.reward import _terminal_reward
    o, ends, landed = _load()
    w = RewardWeights()
    S = w.reward_scale
    starts = np.concatenate([[0], ends[:-1]])
    alt, dx, dy = o[:, 2], o[:, 0], o[:, 1]
    vx, vy, vz = o[:, 3], o[:, 4], o[:, 5]

    # the controller's measured dense sum over its successful episodes
    per_step = np.array([_shaping(w, alt[i], dx[i], dy[i], vx[i], vy[i], vz[i])
                         for i in range(len(o))]) * S
    dense = np.array([-per_step[s:e].sum() for s, e in zip(starts, ends)])
    n_steps = int(np.mean([e - s for s, e in zip(starts, ends)]))
    good_dense = float(dense[landed].mean())

    margins = {"v_z": 0.5, "v_xy": 0.5, "tilt": 0.5, "w": 0.5, "leg_diff": 0.5}
    land = good_dense + _terminal_reward(w, {"landed_safely": True,
                                             "landing_margins": margins}, 0.5)
    marginal_crash = good_dense + _terminal_reward(
        w, {"landed_safely": False, "landing_margins": margins}, 1.05)
    hard_crash = good_dense + _terminal_reward(
        w, {"landed_safely": False, "landing_margins": margins}, 2.5)
    # stalling: hold altitude for the episode, then time out
    hover_step = -_shaping(w, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0) * S
    stall = hover_step * n_steps - w.timeout_penalty

    assert land > marginal_crash, (land, marginal_crash)
    assert marginal_crash > stall, (
        f"stalling ({stall:.1f}) beats a near-landing crash ({marginal_crash:.1f}) -- "
        f"committing must pay")
    assert stall > hard_crash, (
        f"slamming ({hard_crash:.1f}) beats stalling ({stall:.1f}) -- "
        f"this is the inversion the old reward had")


def test_the_per_step_bonus_cannot_pay_for_doing_nothing():
    """Guard against the documented survival-bonus failure mode.

    Mania, Guy and Recht (arXiv:1803.07055, section 5.2) report that Gym's +5/step
    Humanoid survival bonus produces policies that stand still for a thousand steps, and
    that the bonus "discourage[s] the exploration of policies that cause falling early
    on". Our alive bonus exists for the opposite published reason -- a reward that is
    negative everywhere pays the agent to self-terminate -- so the two have to be
    balanced, and the test is simply that holding station must stay net negative at every
    altitude the vehicle flies through.
    """
    w = RewardWeights()
    S = w.reward_scale
    for a in (35.0, 20.0, 10.0, 5.0, 3.0, 1.0):
        r = -_shaping(w, a, 0.0, 0.0, 0.0, 0.0, 0.0) * S
        assert r < 0.0, f"hovering at {a} m pays {r:+.4f}/step -- the bonus outweighs the field"
    # and the whole-episode version: never-ending hover must lose to landing
    assert (w.alive_bonus * S) * 448 < w.touchdown_k, (
        "the alive bonus over one episode exceeds the landing bonus")


def test_the_tile_edge_is_charged_but_not_where_the_controller_flies():
    """The exploit PPO found, and the calibration that must not punish the reference.

    PPO, warm-started and run 2M steps with a value function calibrated to within 0.5 of the
    realised return, landed 0.0% and ended by LEAVING THE TILE, monotonically: left_tile went
    10 -> 24 -> 23 -> 21 -> 33 -> 31 -> 33 -> 38 of ~56 episodes. That is the reward's own
    optimum, not a flaw in PPO. The lateral target is a direction-agnostic SPEED schedule --
    deliberately, since `landed_safely` has no position term -- and `left_tile` truncation
    pays no timeout penalty, so leaving the map is a free exit from a crash worth -6 to -24.

    The calibration has to clear the controller, which reaches 0.83 of the half-extent on
    orbit_descent (696.6 m of 840) and 0.36 on ramp_35m. A free radius of 0.75 would have
    charged the reference -- the same mistake the zero lateral target made, caught this time
    before a run instead of after one.
    """
    from lunarsim.rl.reward import _edge_penalty
    from lunarsim.rl.curriculum import STAGES_BY_NAME

    w = RewardWeights()
    for stage_name, size in (("orbit_descent", 1680.0), ("ramp_35m", 320.0)):
        half = size / 2.0
        # free where the controller flies
        assert _edge_penalty(w, half * 0.83, 0.0, size) == 0.0, stage_name
        # charged before the edge, so the gradient points inward while there is still room
        assert _edge_penalty(w, half * 0.95, 0.0, size) > 0.0, stage_name
        # and growing past it
        prev = 0.0
        for frac in (0.92, 1.0, 1.1, 1.3):
            c = _edge_penalty(w, half * frac, 0.0, size)
            assert c > prev, (stage_name, frac)
            prev = c
        # bounded, so one runaway episode cannot dominate a batch
        assert _edge_penalty(w, half * 50.0, 0.0, size) == w.edge_cap

    # and the controller's own recorded trajectories must pay exactly nothing
    o, ends, landed = _load()
    for stage_name in ("orbit_descent",):
        size = STAGES_BY_NAME[stage_name].tile_size_m
        pen = [_edge_penalty(w, o[i, 0], o[i, 1], size) for i in range(0, len(o), 7)]
        assert max(pen) == 0.0, (
            f"the controller pays the edge penalty on {stage_name}; the free radius is too "
            f"tight and the reward would be punishing the reference again")
