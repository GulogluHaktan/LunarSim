"""The ONE definition of how `action[0]` becomes DPS throttle, and its exact
inverse.

Why this module exists at all: the forward mapping was written out by hand in
five places (`analytic_lander_env`, `isaac_lander_env`, `isaac_lander_vec_env`,
`scripts/isaaclab_policy_eval_capture`, `scripts/isaaclab_controller_eval_capture`)
and the inverse in a sixth (`control/zemzev_controller`). That is the same
copy-paste shape `lunarsim/rl/curriculum.py` was created to end, and here it is
worse: if the env and the controller disagree about this mapping, every
demonstration is flown with one throttle curve and replayed under another.

WHY IT IS NOT LINEAR (measured 2026-10-04):

The old mapping was `throttle = (a0 + 1) / 2`, thrust linear in throttle. On
this vehicle that puts the useful control authority in a sliver of the action
box:

    action[0] = 0 (the box centre, and an untrained SAC policy's mean)
        -> throttle 0.5 -> 24856 N against 24467 N of weight -> T/W = 1.0159
    exact hover: action[0] = -0.019
    |net accel| <= 0.05 m/s^2 (what a steady slow descent holds):
        action[0] in [-0.057, +0.018]  =  3.7% OF THE BOX

So the centre of the action box is a near-perfect hover, and the band the
vehicle must stay inside to descend under control is 3.7% wide, while an
exploring policy's noise spans the whole box. The ZemZev controller, which
does land, flies throttle 0.46-0.52 -- a0 in [-0.080, +0.040]. "The policy
always hovers" has been blamed on the reward for most of this project's
history; this is the actual mechanism, and it is in the action parameterisation,
not the reward.

THE CURVE: `throttle = 0.5 + 0.5 * (e*a0**3 + (1-e)*a0)`, the linear/cubic
blend RC pilots call expo, with `e = THROTTLE_EXPO = 0.8`. Measured:

    expo e   fine band    d(throttle)/d(a0) at centre
      0.0       3.7%        0.500   (the old linear mapping)
      0.8      16.0%        0.100
      1.0      32.3%        0.000   <- pure cubic: widest, but the centre
                                       derivative vanishes, so small actions
                                       stop moving the vehicle at all

0.8 is the knee: 4.3x the control resolution where control actually happens,
with the derivative still finite. Endpoints are untouched -- a0=0 is still
throttle 0.5 and a0=+-1 is still throttle 1/0 -- so the REACHABLE set of
thrusts is exactly what it was (+1.36 / -1.31 m/s^2 of net accel at PDI mass).
Only the resolution is redistributed.

`throttle` itself keeps its old meaning everywhere downstream (the [0,1]
physical fraction that sets thrust, fuel mdot, and `_throttle_effort_penalty`);
this module only changes how `action[0]` reaches it.
"""
from __future__ import annotations

import numpy as np

# Blend between linear (0.0) and pure cubic (1.0). See the module docstring
# for the measured band/derivative trade this was picked from.
THROTTLE_EXPO = 0.8


def action_to_throttle(a0):
    """`action[0]` in [-1, 1] -> DPS throttle fraction in [0, 1].

    Scalar or array. Monotone, so it never makes two actions mean the same
    thrust.
    """
    a = np.clip(a0, -1.0, 1.0)
    shaped = THROTTLE_EXPO * a ** 3 + (1.0 - THROTTLE_EXPO) * a
    return np.clip(0.5 + 0.5 * shaped, 0.0, 1.0)


def throttle_to_action(throttle):
    """Exact inverse of `action_to_throttle` -- throttle in [0, 1] -> the
    `action[0]` that produces it.

    For a controller that reasons in newtons (see `ZemZevController.act`) this
    is the only correct way to emit an action: inverting the OLD linear curve
    while the env applies this one would hand the env a different thrust than
    the controller solved for, silently.

    Solved in closed form rather than numerically. With u = 2*throttle - 1 the
    curve is `e*a^3 + (1-e)*a = u`, i.e. the depressed cubic
    `a^3 + p*a + q = 0` with `p = (1-e)/e > 0` and `q = -u/e`. A positive `p`
    makes the discriminant positive, so there is exactly one real root and
    Cardano's formula gives it directly -- which is just the algebraic
    statement that the curve is monotone.
    """
    u = 2.0 * np.clip(throttle, 0.0, 1.0) - 1.0
    e = THROTTLE_EXPO
    if e <= 0.0:
        return u
    p = (1.0 - e) / e
    q = -u / e
    disc = np.sqrt(q * q / 4.0 + p ** 3 / 27.0)
    return np.clip(np.cbrt(-q / 2.0 + disc) + np.cbrt(-q / 2.0 - disc), -1.0, 1.0)
