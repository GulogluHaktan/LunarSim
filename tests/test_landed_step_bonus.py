"""The success boundary must carry weight, and only when it is crossed.

The metric this project is judged on is binary (`landed_safely`), while the
terminal reward is a ramp continuous THROUGH ZERO at that boundary -- so
crossing it by 1% of a limit was worth 0.120 reward units against a ~5-unit
shaping range on a failing episode. Measured consequence, on 192 paired
validation episodes: the run that landed 31.2% scored mean return -20.54 and
the run that landed 21.9% scored -15.94, i.e. PPO's own objective preferred the
weaker-landing policy.

`landed_step_k` adds a step on top of the ramp. These tests pin four things:

  1. it is OFF by default, so every number recorded before it reproduces;
  2. it fires exactly on the `landed_safely` condition and not one ratio over;
  3. it does not disturb the ramp's monotonicity or its floor;
  4. the gate is the same `sev < 1.0` expression the ramp uses, so the reward's
     notion of success cannot drift from the env's label -- which is how a
     label fix and a grading fix once got applied to two different states.
"""
from __future__ import annotations

from dataclasses import replace

from lunarsim.rl.reward import RewardWeights, _terminal_reward

MARGINS = {"v_z": 0.5, "v_xy": 0.5, "tilt": 0.5, "w": 0.5, "leg_diff": 0.5}


def _term(w, sev):
    return _terminal_reward(w, {"landing_margins": MARGINS}, sev)


def test_off_by_default():
    assert RewardWeights().landed_step_k == 0.0
    w = RewardWeights()
    for sev in (0.0, 0.5, 0.99, 1.0, 1.5, 9.0):
        assert _term(w, sev) == max(w.touchdown_k * (1.0 - sev),
                                    -w.touchdown_penalty_cap)


def test_step_fires_only_on_the_success_side():
    w = replace(RewardWeights(), landed_step_k=100.0)
    # A hair inside the limit pays the step; a hair outside pays none.
    assert _term(w, 0.999) - _term(w, 1.001) > 100.0
    # Exactly AT the boundary is a failure: `landed_safely` is a strict
    # inequality on every criterion, so severity 1.0 must not pay.
    assert _term(w, 1.0) == _term(RewardWeights(), 1.0)


def test_step_is_additive_not_a_replacement():
    """The ramp was introduced because a pure step gave gradient methods
    nothing to climb. Adding the step must not flatten it."""
    base = RewardWeights()
    w = replace(base, landed_step_k=100.0)
    for sev in (0.0, 0.25, 0.5, 0.9):
        assert _term(w, sev) == _term(base, sev) + 100.0
    # still strictly decreasing in severity on the success side
    vals = [_term(w, s) for s in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_step_outweighs_a_failing_episode_s_shaping_range():
    """Sizing is the whole point: the step has to beat what the policy can earn
    by improving an episode it still loses. The measured range is ~5 units
    after `reward_total_scale`, and the conversion gain it has to dominate was
    +2.52."""
    w = replace(RewardWeights(), landed_step_k=100.0)
    step_units = w.landed_step_k * w.reward_total_scale
    assert step_units >= 5.0


def test_a_tumble_still_gets_the_floor_and_no_step():
    """No margins means the vehicle never reached the surface. It must not be
    able to collect the step through the `severity=None` path."""
    w = replace(RewardWeights(), landed_step_k=100.0)
    assert _terminal_reward(w, {}, None) == -w.touchdown_penalty_cap
    assert _terminal_reward(w, {"landing_margins": {}}, 0.1) == -w.touchdown_penalty_cap


def test_leg_diff_still_blocks_the_step():
    """`leg_diff` is the fifth `landed_safely` criterion and reaches severity
    through a bump rather than a ratio. A touchdown on ground too uneven to
    stand on must not pay the step."""
    w = replace(RewardWeights(), landed_step_k=100.0)
    flat = _terminal_reward(w, {"landing_margins": MARGINS}, 0.3)
    # severity is computed upstream; here the caller passes the bumped value,
    # which is what `_touchdown_severity` produces when leg_diff saturates.
    uneven = _terminal_reward(w, {"landing_margins": dict(MARGINS, leg_diff=0.0)}, 1.0)
    assert flat > 100.0
    assert uneven < 0.0 or uneven == 0.0
