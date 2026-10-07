"""Reward for the Apollo LM landing task.

REWRITTEN FROM SCRATCH 2026-10-05, replacing a nine-term reward whose
measured behaviour is archived in `reward_legacy.py`. Four facts drove the
rewrite, all measured on real Isaac Sim data, not argued:

1. **The old core term penalised ALTITUDE**, which is a time x altitude
   integral: corr(episode length, total shaping) = **-0.927**. It scored
   duration, not quality, and could not separate a controlled descent from
   a dive -- the dive scores BETTER, because it reduces altitude faster.
2. **It was almost entirely one term.** Per-term decomposition of a real
   winning flight: altitude 86.10, vel_track 4.89, braking_xy 0.71, every
   other term <= 0.02, against a 450 landing bonus. Seven of nine terms
   were numerically dead.
3. **Crashing cost the same as doing nothing.** A measured free-fall impact
   (vz=-7.60, 7.6x its limit) scored -104.5 against -105.0 for hovering to
   timeout, because the penalty averaged five margins each clipped to
   [0,1], and the two that are trivially satisfied while falling straight
   down masked the overshoot.
4. **The vertical profile of a good landing is an empirical law.** The
   ZemZev controller, which lands on real Isaac Sim, flies
   `vz = -0.276*sqrt(alt)` with a spread of 0.275-0.278 from 198 m down to
   10 m (5360 telemetry points). Not a guess -- a measurement.

THE SHAPE OF THIS REWARD

Four shaping terms, each with exactly one job, plus the terminal:

  `_descent_envelope_penalty`  vz may not exceed `v_limit(h) = c*sqrt(h)`.
      A LIMIT, not a target -- deviating BELOW it is free. That matters:
      the proven controller flies at c=0.276 while the episode budgets
      (see `profile_c`) only allow c~0.78, so a symmetric target would
      penalise the one trajectory we know is correct.
  `_time_penalty`             a flat per-step cost. This, not an altitude
      term, is what makes the vehicle want to get DOWN. A constant cost
      cannot be reduced by diving faster (the crash ends the episode at a
      severity-graded penalty), only by landing sooner.
  `_horizontal_speed_penalty` kill lateral velocity.
  `_tilt_cutoff_penalty` / `_angular_rate_penalty`  stay controlled; these
      two are carried over unchanged, with their original measured
      justifications (see their field comments).

The optimum this defines is "descend as fast as the envelope allows, which
goes to zero at the ground, and land" -- which is a controlled descent, by
construction rather than by tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs


@dataclass
class RewardWeights:
    # ---------------------------------------------------------------- #
    # Descent envelope: |vz| may not exceed profile_c * sqrt(altitude).
    # ---------------------------------------------------------------- #
    # MEASURED (hero_fixed telemetry, 5360 points, 198 m -> 10 m): the
    # controller that actually lands flies vz = -0.276*sqrt(alt), and the
    # ratio holds to +-0.002 over a 20x altitude range. The sqrt shape is
    # therefore not an assumption; it is what a working solution does, and
    # it is also what the physics says (sqrt is the arrestable-speed curve
    # for constant deceleration).
    #
    # `profile_c` is NOT 0.276 though, because that profile does not fit
    # any stage's episode budget: descending h0 at -c*sqrt(h) takes
    # 2*sqrt(h0)/c seconds, i.e. 31 s for ramp_20m (budget 20 s) and 99 s
    # for orbit_descent (budget 60 s). The coefficient that fits ~60% of
    # each budget is remarkably uniform across stages -- 0.745, 0.786,
    # 0.833, 0.816, 0.786 -- so one constant serves all five. 0.78 leaves
    # the proven profile comfortably INSIDE the envelope (it is 2.8x
    # slower) while still landing within the vz limit on its own: at 1 m
    # altitude the envelope is 0.78 m/s against a 1.0 m/s limit.
    #
    # If the budgets are ever raised to fit the measured 0.276 profile
    # (31 s / 99 s), lower this with them -- the two are one choice.
    profile_c: float = 0.78
    # quadratic on the EXCESS over the envelope, so being inside costs
    # nothing and overshoot costs ever more steeply.
    # Sized against the REAL overspeed range, not a round number: a
    # free-fall arrival near the ground is ~8 m/s against a <1 m/s
    # envelope, so the quadratic has to stay unsaturated out to ~7 m/s of
    # excess. At k=35 that is 1814 raw = 0.45/step, against 0.10/step for
    # the time cost -- felt, but not swamping. The cap exists only to stop
    # a pathological state blowing up the sum; if it binds in normal
    # flight the term has lost its gradient, which is exactly the defect
    # the old braking envelope had (it saturated within a tiny range and
    # then said nothing about whether braking harder helped).
    #
    # RAISED 35 -> 400 (2026-10-06). Once the lateral channel was fixed the
    # binding criterion moved to v_z -- 3 of the 4 remaining crashes on the
    # best ramp_35m checkpoint, at vz -1.30/-1.74/-1.83 against a 1.0 m/s
    # limit. Measured per step at alt=0.2 m, the charge separating a
    # landing from a crash was:
    #   vertical, vz -0.8 -> -1.8 :   63.7
    #   lateral,  vxy 0.9 -> 1.7  :  792.4   (12.4x more authority)
    # Worse, arriving fast was PROFITABLE: at alt=2 m, going from vz=-0.8
    # to -1.3 earned +207.8 of descent progress against +1.4 of envelope
    # penalty, a net +206 for diving. 400 puts the two channels on the same
    # order (416/step vs 792/step at the same states), which is what it has
    # to be for them to trade off instead of one being ignored -- the same
    # reasoning, and the same number, as kxy 12 -> 400.
    # RETIRED, kept at 0 so the term can be restored for an ablation. The velocity field
    # subsumes it: the field names a descent rate at every altitude, so descending too
    # slowly, too fast, or climbing are all simply tracking errors with a gradient, where
    # this term was flat inside the envelope and a wall outside it. This was the envelope's own penalty.
    profile_k: float = 0.0
    # Climbing is outside the envelope too, and needs its own gradient: with
    # a flat time cost, a climb and a hover cost the SAME per step, so
    # nothing locally opposes the climb-away exploit this project measured
    # before (sustained full-throttle climbs to 680-1372 m). Penalising
    # positive vz directly restores that gradient without reintroducing an
    # altitude integral -- the thing the rewrite exists to remove.
    # Kept at 2x profile_k, the ratio that was measured to stop the
    # climb-away exploit. v23 still timed out 6 of 24 episodes at 15-22 m
    # with vz at or above zero (+0.63, +1.06, +1.12), so this relationship
    # is still load-bearing and scales with profile_k rather than staying
    # put and becoming relatively free.
    # RETIRED, kept at 0 so the term can be restored for an ablation. The velocity field
    # subsumes it: the field names a descent rate at every altitude, so descending too
    # slowly, too fast, or climbing are all simply tracking errors with a gradient, where
    # this term was flat inside the envelope and a wall outside it. Climbing is now a large tracking error.
    profile_climb_k: float = 0.0
    # Scaled with profile_k to preserve the property the comment above
    # demands: unsaturated out to ~7 m/s of excess, which a near-ground
    # free-fall arrival actually reaches. At k=400 that needs 400*49 =
    # 19600, so 3000 would have saturated at just 2.74 m/s of excess and
    # recreated, in the vertical channel, the exact defect just removed
    # from the lateral one.
    # Raised from 25000, which saturated at -15 m/s descent and +5.59 m/s climb --
    # so a 15 m/s impact and a 20 m/s impact were priced identically, and climbing
    # at 15 m/s cost the same as at 5.6. Both are differences the policy must be
    # able to feel.
    profile_cap: float = 120000.0
    # Floors the altitude inside the sqrt. Described here as "a
    # divide-by-zero guard, not a tunable", which was wrong: it sets the
    # envelope's value AT CONTACT, i.e. the touchdown speed the reward
    # demands, so it is the most load-bearing number in this block.
    #
    # 0.25 -> 1.0 (2026-10-06). At 0.25 the floor was 0.78*sqrt(0.25) =
    # 0.39 m/s against a `safe_landing_v_z_m_s` limit of 1.0 -- the reward
    # was demanding a touchdown 2.5x gentler than the vehicle actually
    # requires, and (before profile_k was raised) had no authority to
    # enforce even that. At 1.0 the floor is 0.78 m/s, a 22% margin under
    # the real limit.
    #
    # This also attacks the hover-just-above-the-ground failure directly,
    # because the descent-progress reward is capped AT the envelope: at
    # alt=0.2 m the rate it will pay for goes 0.39 -> 0.78 m/s, which
    # doubles the reward for descending the last metre (267 -> 535 per
    # step) instead of hanging there.
    profile_alt_floor_m: float = 1.0

    # ---------------------------------------------------------------- #
    # Progress toward the ground, CAPPED AT THE ENVELOPE SPEED.
    # ---------------------------------------------------------------- #
    # MEASURED defect this fixes (run v6, 14/14 episodes): with only an
    # envelope (zero cost inside it) and a flat time cost, hovering at 35 m
    # and descending at -1.4 m/s cost EXACTLY the same per step. There was
    # no local gradient toward the ground at all -- the time cost punishes
    # duration only in total, and the agent can only shorten the episode by
    # landing, which it does not yet know how to do. Every episode survived
    # the full 20 s without reaching the ground.
    #
    # `+progress_k * min(-vz, envelope)` supplies that gradient and cannot
    # be farmed, because of the cap:
    #   - stay at or under the envelope and the total earned over a descent
    #     is `progress_k * h0 / dt` REGARDLESS of profile -- path
    #     independent, so there is nothing to game.
    #   - exceed the envelope and the excess earns NOTHING while the
    #     episode gets shorter, so a dive collects strictly LESS total
    #     progress than a legal descent, on top of paying the envelope
    #     penalty.
    # So the per-step optimum is "descend exactly at the envelope", which
    # is the controlled descent, and it is optimal locally rather than only
    # in hindsight at the terminal.
    # Paid per FRACTION of the release altitude closed, not per metre.
    # MEASURED why: at a flat per-metre rate the total earnable over a
    # descent is proportional to h0, which varies 10x across the stages
    # (20 m to 200 m). At orbit_descent that let a free-fall dive collect
    # ~565 units on the way down and score +94.6 end-to-end -- BETTER than
    # hovering -- because the progress it banked outweighed the crash.
    # Dividing by `spawn_altitude_m` makes the total earnable over a full
    # descent identical at every stage (`progress_k * reward_scale / dt`
    # ~= 120 units, against a 450 landing bonus), so the term stays a
    # guide and can never pay for its own crash.
    # RETIRED, kept at 0 so the term can be restored for an ablation. The velocity field
    # subsumes it: the field names a descent rate at every altitude, so descending too
    # slowly, too fast, or climbing are all simply tracking errors with a gradient, where
    # this term was flat inside the envelope and a wall outside it. Paying for altitude closed is also redundant with the field, which already demands a descent rate; the published design has no separate progress term.
    progress_k: float = 0.0

    # ---------------------------------------------------------------- #
    # Time: pressure to finish, on top of the progress gradient.
    # ---------------------------------------------------------------- #
    # The old reward used an altitude penalty for this and that is what
    # made diving optimal (see the module docstring). A flat per-step cost
    # has the property an altitude term does not: it cannot be reduced by
    # arriving faster, because the arrival itself is scored separately and
    # a fast arrival is scored badly. So it buys urgency without buying
    # recklessness. Sized so a full ramp_20m episode (400 steps) costs
    # ~40 units against a 450 bonus -- enough that dawdling is clearly
    # worse than landing, far from enough to outrun the terminal.
    # RETIRED, kept at 0 so the term can be restored for an ablation. The velocity field
    # subsumes it: the field names a descent rate at every altitude, so descending too
    # slowly, too fast, or climbing are all simply tracking errors with a gradient, where
    # this term was flat inside the envelope and a wall outside it. Replaced by `alive_bonus` with the opposite sign; see `_time_penalty`.
    time_k: float = 0.0

    # ---------------------------------------------------------------- #
    # Horizontal speed.
    # ---------------------------------------------------------------- #
    # In the old reward the terms that should have said "kill your lateral
    # speed" summed to 5.60 over an entire winning flight = 1.2% of the
    # landing bonus, and the policy duly flew sideways off the tile in
    # 13/16 episodes. Sized here to be the second-largest shaping term.
    # Weighted by PROXIMITY TO THE GROUND, not flat. Lateral speed at
    # 200 m is how the vehicle covers ground and is released with 10-30
    # m/s of it by definition; lateral speed at 2 m is what breaks the
    # legs. A flat quadratic taxes the unavoidable release speed: measured
    # on the winning controller flight it came to 250 of a 439 total
    # against a 450 bonus, leaving the proven trajectory a net +11. With
    # the ground weight `1/(1 + alt/kxy_alt_ref)` the same flight pays a
    # fraction of that, while a vehicle still carrying 20 m/s at touchdown
    # pays nearly the full quadratic.
    # RAISED 12 -> 400 (2026-10-05, measured on the first checkpoint that
    # ever landed). 24-episode diagnostic of v8: 12 touchdowns, and TEN of
    # them were rejected on v_xy and nothing else --
    #   vz=-0.30 ok, vxy=1.71 FAIL, tilt=4.0 ok, w=0.01 ok
    #   vz=-0.11 ok (margin 0.89!), vxy=1.50 FAIL
    # against the two that landed at vxy 1.14 and 1.16. The vertical
    # channel is solved; the whole remaining gap is 0.3-0.5 m/s of lateral
    # speed at contact.
    # At kxy=12 the difference between arriving at 1.7 and at 1.1 m/s was
    # 0.005 PER STEP, against ~1.0/step for the descent progress reward --
    # "go down" outweighed "bleed off lateral" by 200x, so the policy
    # correctly ignored the latter. At 400 that difference is 0.17/step,
    # the same order as the progress term, which is what it has to be for
    # the two to trade off instead of one dominating.
    # RETIRED, kept at 0 so the term can be restored for an ablation. The velocity field
    # subsumes it: the field names a descent rate at every altitude, so descending too
    # slowly, too fast, or climbing are all simply tracking errors with a gradient, where
    # this term was flat inside the envelope and a wall outside it. Below the switch the lateral target is 0, so lateral speed IS the tracking error -- linear rather than quadratic, which is the gentler form and avoids the saturation this project removed from three other terms.
    kxy: float = 0.0
    # 20 -> 4 m (2026-10-05, measured when ramp_50m regressed to 0/16).
    # The ground weight sets how far UP lateral speed is charged, and 20 m
    # reached much too high once the release speed grew. Measured ratio of
    # the lateral penalty to the descent progress reward, per step:
    #   ramp_20m (release 0-2 m/s):  0.2x at 20 m, 0.6x at 1 m   -- balanced
    #   ramp_50m (release 2-8 m/s):  2.8x at 50 m, 16.3x at 1 m  -- swamped
    # The two terms scale in OPPOSITE directions across the curriculum:
    # lateral is quadratic in a release speed that grows stage to stage,
    # while progress is normalised by h0 and so shrinks. The policy did
    # exactly what that says -- it bled lateral speed and then refused to
    # land, 11/12 episodes timing out, one of them hovering 0.20 m off the
    # ground with 3.78 m/s still on. At 4 m the charge is concentrated
    # where lateral speed actually breaks legs (0.7x of progress at 50 m,
    # still 5x near the ground, which is the correct priority there) and
    # ramp_20m's balance is unchanged.
    kxy_alt_ref_m: float = 4.0
    # Above this lateral speed the charge grows LINEARLY instead of
    # quadratically: `knee**2 + 2*knee*(v_xy - knee)`, which is value- and
    # slope-continuous at the knee, so everything below it is bit-identical
    # to the plain quadratic and only the fast tail changes.
    #
    # This replaces a hard cap that saturated exactly where it mattered
    # most. The old `kxy_cap=20000` was sized against the RELEASE speeds,
    # where the ground weight is tiny (at orbit_descent's 200 m the weight
    # is 0.0196, so the cap only bound above 50 m/s -- correct). Near the
    # ground the weight is 1.0 and the same cap bound at
    # sqrt(20000/400) = 7.07 m/s, i.e. inside the operating range, leaving
    # the reward FLAT in v_xy above it. Measured on the best ramp_35m
    # checkpoint (21%), both low-altitude timeouts lived in that flat zone:
    #   ep13  alt=2.75  v_xy=12.68   (saturation there: 9.18)
    #   ep15  alt=0.45  v_xy=13.62   (saturation there: 7.46)
    # Both were skimming the surface at ~13 m/s and refusing to touch down,
    # with no gradient anywhere telling them to bleed it off.
    #
    # The knee keeps the MAGNITUDE and restores the SLOPE: at ep15's state
    # the charge goes 20000 (capped, flat) -> 18152 (with a gradient).
    # Picked at 2.0 m/s, just above the 1.2 m/s `landed_safely` limit, so
    # the whole proven regime stays quadratic.
    kxy_knee_m_s: float = 2.0
    # Now a true safety net rather than part of the shaping: at the fastest
    # release any stage uses (30 m/s) the knee form gives 46400 at ground
    # weight 1, so this only binds above ~38 m/s at zero altitude, which no
    # stage reaches.
    kxy_cap: float = 60000.0

    # ---------------------------------------------------------------- #
    # Carried over unchanged, with their original measured justifications.
    # ---------------------------------------------------------------- #
    # Tilt barrier. Added after lost_control climbed 0/16 -> 16/16 across
    # successive extensions of one training run: the attitude terms of the
    # day made tilt nearly free at speed, and nothing independently
    # discouraged tilting all the way to the 60 deg cutoff. Negligible
    # through the controller-proven 0-35 deg range, steep past 45-50.
    # CONTINUOUS tilt term, referenced to the SAFE landing criterion rather than
    # to loss-of-control. Measured defect it replaces: the ratio**8 barrier below
    # is referenced to loss_of_control_tilt_rad (40 deg), so at the 15 deg
    # criterion that actually decides `landed_safely` it charged 0.0002 per step
    # against the lateral term's 0.144 at ITS limit -- 720x underpriced. Tilt was
    # effectively free through the entire operating range and only became visible
    # past 30 deg, by which point the episode is already lost. 576 is chosen so
    # that tilt AT its criterion costs the same per step as lateral speed at its
    # criterion (0.144), which is the "no criterion is unpriced" property the
    # barrier form silently broke.
    # MEASURED AND SET TO ZERO. 576 (tilt at its criterion costing the same per
    # step as lateral speed at its criterion) was tried and is wrong, and the demo
    # data says why -- the controller tilts to brake, including near the ground:
    #
    #   altitude      tilt p50   tilt p90   tilt max
    #     0-0.5 m       5.77      11.18      16.14
    #     1-2 m         8.12      13.15      17.63
    #     2-5 m         9.86      13.12      17.61   <- hardest braking
    #
    # At its own p90 tilt the controller would have paid ~0.08 per step, about 36
    # per episode against a total dense shaping sum of 32.4. That taxes the one
    # maneuver that kills lateral velocity, which is the failure mode every one of
    # its 23 rejections actually comes from.
    #
    # It is also unnecessary. `severity` is the MAX over all four touchdown
    # criteria, so the continuous terminal already prices tilt at the moment the
    # criterion applies: touchdown at 20 deg gives severity 1.33 -> -40, at 7 deg
    # -> +64. The criterion gets a gradient without flight tilt being charged for it.
    #
    # The division of labour this settles: the terminal prices the four touchdown
    # criteria, and the dense shaping prices what the terminal cannot see -- time,
    # descent progress, envelope violations, and lateral speed during flight
    # (which must be killed BEFORE touchdown, so it needs a gradient en route).
    tilt_safe_k: float = 0.0
    tilt_cutoff_k: float = 2000.0
    tilt_cutoff_power: float = 8.0
    tilt_cutoff_cap: float = 2000.0

    # Angular rate. |w| <= safe_landing_w_rad_s is one of the five landing
    # criteria and was the BINDING margin on two of the three closest
    # touchdowns ever measured. Deliberately not velocity-gated: speed can
    # justify a tilt ANGLE, never a spin. Yaw counts, which no other term
    # reads.
    # Raised from 60. With the cap lifted, the barrier still saturates just past the
    # 0.5 rad/s criterion, so between 0.5 and 1.5 rad/s the penalty only moved
    # 0.154 -> 0.184 per step: tripling the tumble rate cost 20%. The quadratic term
    # has to be strong enough to take over where the barrier flattens, which is what
    # makes the super-criterion region proportional instead of merely bounded.
    omega_k: float = 600.0
    omega_cutoff_k: float = 600.0
    omega_cutoff_power: float = 6.0
    omega_cutoff_cap: float = 600.0
    # Raised from 800. At 800 the angular-rate penalty went flat exactly where it
    # was supposed to bite: 0.154 per step at the 0.5 rad/s criterion and 0.200 at
    # 3.0 rad/s, so tumbling six times faster than the limit cost 30% more. The cap
    # now sits far enough out that the super-limit region keeps paying.
    omega_penalty_cap: float = 6000.0

    # ---------------------------------------------------------------- #
    # Terminal.
    # ---------------------------------------------------------------- #
    landing_bonus_scale: float = 450.0

    # Crash graded by OVERSHOOT (max over criteria of value/limit), not by
    # the mean of the clipped margins -- see the module docstring for the
    # measured case where a 7.6x velocity overshoot scored the same as
    # hovering. Max, not mean: one catastrophic axis is a catastrophe
    # however good the others look.
    #   severity 1.0 -> 0.50x -> -60   (beats hovering: attempting pays)
    #   severity 2.5 -> 1.18x -> -141
    #   severity 7.6 -> 3.00x -> -360  (capped)
    # ---- CONTINUOUS TOUCHDOWN TERMINAL (replaces the two branches below) ----
    # Measured reason for the change, on the 48 recorded ramp_35m demo episodes:
    #
    #   episode length                448 steps
    #   dense shaping sum, whole ep   +32.4
    #   terminal for a landing        +351 .. +450
    #     terminal / shaping          10.8x
    #     STEP at the success boundary  411
    #
    # So the term that should only SETTLE the final behaviour was 10.8x the term
    # that is supposed to carry the vehicle onto the path, and it arrived as a
    # discontinuity: +351 at v_xy = 1.19 m/s, -60 at 1.21. A gradient method
    # cannot climb a step. Improving terminal lateral speed from 1.5 to 1.3 m/s
    # earned NOTHING until the boundary was crossed, which is why the actor could
    # only ever find the cheap gradient -- stop hovering, commit, crash -- and
    # that is exactly what it did: against the clone's 33 timeouts / 10 crashes,
    # the RL policy produced 25 timeouts / 23 crashes with the landing count
    # unchanged.
    #
    # The replacement is one continuous, monotone function of `severity` (how many
    # times over its worst limit the touchdown was, 1.0 being exactly at a limit):
    #
    #   terminal = touchdown_k * (1 - severity),  floored at -touchdown_penalty_cap
    #
    #   severity 0.0  feather-soft landing      +120
    #   severity 0.5  comfortable landing        +60
    #   severity 1.0  exactly at the limit         0   <- continuous HERE
    #   severity 1.05 marginal crash              -6
    #   severity 1.5  crash                      -60   = a timeout
    #   severity 2.0  hard crash                -120
    #   severity 7.6  slam                      -480   (capped)
    #
    # severity < 1 is exactly the `landed_safely` condition, so the label boundary
    # and the reward boundary coincide by construction instead of disagreeing by
    # 411 points. Lateral speed now pays continuously all the way down, which is
    # the gradient the precision failures need: every one of the controller's own
    # 23 failures across seven stages is a touchdown rejected on lateral speed,
    # with terminal v_xy sitting at 0.5-1.3 m/s against the 1.2 limit.
    #
    # touchdown_k > timeout_penalty is REQUIRED, not cosmetic: it is what makes a
    # hard crash worse than waiting (severity 2 -> -120 against a -60 timeout)
    # while a near-landing crash is still better than waiting (severity 1.05 ->
    # -6). Both orderings matter -- the first stops "commit and slam" from being
    # free, the second keeps committing better than stalling.
    # ---- VELOCITY-FIELD DENSE CORE (arXiv:1810.08719) ----
    # `vfield_k` is calibrated so the DENSE shaping outweighs the terminal, which is
    # the balance the published design uses and the opposite of what this project had.
    # Measured: their landing bonus is kappa=10 against a tracking term worth about 20
    # over an episode, so dense/terminal ~ 2. Ours was terminal/dense = 10.8, and still
    # 3.7 after the continuous-terminal change -- the terminal was lowered when the
    # dense terms should have been raised. At vfield_k=500 a steady 2 m/s tracking
    # error costs 0.125/step, ~56 over a 448-step episode, and a 4 m/s error ~112,
    # against a terminal of 120. So a policy that tracks badly for a whole episode pays
    # about what a landing is worth, and a policy that tracks well pays almost nothing
    # and collects the terminal.
    vfield_k: float = 500.0
    vfield_cap: float = 40000.0
    # The controller's measured descent schedule over its successful episodes:
    # -vz = 0.547*alt^0.431, floored at 0.80 m/s. See `target_velocity` for why the
    # envelope's own 0.78*sqrt(alt) was the wrong target (too fast by ~2x) and why the
    # floor is needed (the controller holds a constant 0.80 m/s below 5 m).
    vfield_vz_c: float = 0.547
    vfield_vz_p: float = 0.431
    vfield_vz_floor_m_s: float = 0.80
    # LATERAL schedule. A target of zero everywhere was wrong, and badly so on the stage
    # that matters. Measured on orbit_descent, where the controller lands 50%: its mean
    # tracking error against a zero lateral target is 10.00 m/s (p90 19.29), because it
    # arrives with 10-30 m/s of lateral velocity and the measured lateral deceleration it can
    # sustain is 0.50 m/s^2 -- killing 30 m/s takes a minute. The dense penalty came to -43.9
    # per episode against a +6.0 landing bonus, so the reward was punishing the reference
    # policy 7.3 to 1 for doing the only thing physics allows. On ramp_35m, which spawns at
    # 2-5 m/s, the same target was fine (error 1.74 m/s, ratio 0.8x) -- which is exactly how
    # a stage-specific miscalibration hides.
    #
    # Gaudet's magnitude is v_o*(1 - exp(-t_go/tau)): anchored to the arrival speed and
    # DECAYING, never zero at altitude. Rejecting their aim point was right (this task has no
    # position criterion) but replacing the magnitude with 0 was not.
    #
    # Fitted over the landed demos of BOTH stages, 1 m and up:
    #     v_xy = 0.170 * alt^0.870      (200 m: 17.1 vs 15.0 measured, 50 m: 5.12 vs 4.95,
    #                                    10 m: 1.26 vs 1.35, 5 m: 0.69 vs 0.53)
    # The floor is 0.30, deliberately BELOW the 0.64 the controller achieves at 0-2 m: near
    # the ground is where precision decides the outcome, so a small standing pressure there is
    # wanted, and 0.34 m/s of it costs 0.002 per step.
    vfield_vxy_c: float = 0.170
    vfield_vxy_p: float = 0.870
    vfield_vxy_floor_m_s: float = 0.30
    # ---- TILE BOUNDARY ----
    # Added because the first learner able to actually optimise this reward found that it
    # does not protect against leaving the map. PPO, warm-started and run 2M steps on
    # orbit_descent with a value function calibrated to within 0.5 of the realised return,
    # landed 0.0% and ended its episodes by LEAVING THE TILE, monotonically:
    #
    #     left_tile   10 -> 24 -> 23 -> 21 -> 33 -> 31 -> 33 -> 38   (of ~56 episodes)
    #
    # That is not a bug in PPO, it is the reward's own optimum. The lateral target is a
    # direction-agnostic SPEED schedule -- chosen deliberately, because `landed_safely`
    # contains no position term -- so satisfying it while drifting off the map scores well.
    # But `left_tile` truncation IS a position constraint, and nothing priced it.
    # curriculum.py already recorded the geometry (orbit_descent's 1680 m tile against ~1800 m
    # of episode drift) and left it because "a competent policy brakes early and the ZemZev
    # demos never leave any tile (0/48)". A learning policy is not that policy.
    #
    # A BARRIER is the right shape here, unlike for the precision criteria. The distinction is
    # whether the thing is something to optimise or a genuine constraint: leaving the tile
    # truncates the episode and the env's own altitude reads become fictional outside it
    # (sample_height_at clamps to the grid edge while the PhysX mesh simply ends), so there is
    # nothing to be gained out there and no gradient worth providing. It starts charging
    # BEFORE the edge, at `edge_safe_frac` of the half-extent, so the gradient points inward
    # while the vehicle can still act on it.
    # MEASURED, not chosen. The controller reaches 0.83 of the half-extent on orbit_descent
    # (696.6 m of 840 m) and 0.36 on ramp_35m, so a free radius of 0.75 would have charged
    # the reference policy -- the same mistake the zero lateral target made, caught this time
    # before a run rather than after one. 0.88 leaves it free and ramps over the remaining
    # 0.12 of the half-extent.
    #
    # edge_k is sized against what the exploit AVOIDS, which is the whole terminal penalty:
    # `left_tile` truncation deliberately pays no timeout penalty (see `_terminal_reward`'s
    # caller), so leaving the map is a FREE EXIT from a crash worth -6 to -24. At 6400 the
    # charge is 0.08 per step at the boundary, so drifting out there for ~100 steps costs
    # about what the crash it dodges would have.
    edge_k: float = 6400.0
    edge_safe_frac: float = 0.88
    edge_cap: float = 120000.0

    # MUST stay below `vfield_k * vfield_vz_floor_m_s` = 500*0.80 = 400, and this was
    # caught by a test rather than by reasoning. At 500 the bonus EXCEEDED the field's
    # cost of a hover near the ground: holding station at 3 m paid +0.0152/step, so the
    # agent had no local pressure to close the last few metres -- which is the exact
    # recipe for the low-altitude stalling this project already measured as timeouts.
    #
    # The published relation (their eta = -alpha, balancing at a 1 m/s error) assumes the
    # smallest target rate is at least 1 m/s. Our floor is 0.80 m/s, so the balance point
    # has to sit below that: 250 balances at 0.5 m/s, which keeps a hover firmly negative
    # at every altitude the vehicle flies through.
    #
    # It also moves eta*T/kappa from 47% to 23%, further from the survival-bonus failure
    # mode Mania/Guy/Recht document (arXiv:1803.07055 S5.2: Gym's +5/step Humanoid bonus
    # produces policies that stand still for a thousand steps and "discourage[s] the
    # exploration of policies that cause falling early on"). Gaudet's ratio is ~40%.
    alive_bonus: float = 250.0

    touchdown_k: float = 120.0
    touchdown_penalty_cap: float = 480.0

    # ---- DEPRECATED: the old two-branch terminal. Kept so that older configs
    # and recorded `reward_weights` blobs still load; no longer read by
    # `_terminal_reward`.
    landing_bonus_scale_deprecated: float = 450.0
    crash_penalty: float = 120.0
    crash_severity_base: float = 0.5
    crash_severity_k: float = 0.45
    crash_severity_cap: float = 3.0

    # Truncation without landing (ran out of clock, or drifted off the
    # terrain tile). Must stay worse than a near-miss touchdown -- trying
    # and nearly making it has to beat never trying -- and better than a
    # real crash.
    # 60 rather than 105 so that, with touchdown_k=120, a crash at severity 1.5
    # costs exactly what a timeout costs: the crossover between "committing was
    # worth it" and "you should have kept trying" sits halfway over the limit.
    timeout_penalty: float = 60.0

    # Defensive clip on the summed per-step shaping, excluding the
    # terminal. A safety net, not a constraint -- and it had stopped being
    # one. The estimate below it ("kxy 1200") was computed before kxy was
    # raised 12 -> 400; at 400 the lateral term alone reaches 8100 at
    # v_xy=4.5 near the ground, so it CLIPPED THE WHOLE SUM by itself and
    # took the descent-envelope, tilt, omega and progress gradients down
    # with it. Measured on the same 24 episodes, the crashes at v_xy 4.70,
    # 6.50 and 14.47 were all in that state: every term flat at once.
    # Worst plausible now: profile 25000 + tilt 2000 + omega 800 + time 400
    # + kxy 46400 (30 m/s at ground weight 1) = 74600.
    # This is a no-op for the proven ramp_20m regime, where touchdown v_xy
    # is 0.13-1.17 and the sum never came near either value.
    # Raised with profile_cap/omega_penalty_cap: at 80000 the clip would have
    # re-introduced the very saturation those two changes remove, binding before
    # either term reached its own cap.
    shaping_clip_abs: float = 200000.0

    # Applied to the shaping sum ONLY. The terminal is added after, at its
    # own scale, because it has to outweigh the SUM of hundreds of shaping
    # steps rather than one of them -- conflating the two is a mistake
    # this project made once already (see reward_legacy.py).
    # Charge for sitting ON the action bounds, so the optimum is INTERIOR.
    #
    # Measured 2026-10-06: every policy this project has produced is ~95%
    # saturated, emitting literally [-1, 1, -1, 1], and the v30 checkpoint went
    # from 22.9% to 0% while its actor weight norm moved 190.7 -> 192.2. A
    # saturated policy makes the parameter-to-behaviour map a step function:
    # tiny gradient steps flip discrete action choices instead of refining
    # them, so training is a random walk in behaviour space.
    #
    # The literature says do NOT attack this through the tanh Jacobian --
    # Shamass (2026) tried restoring the missing gradient and collapsed return
    # from -31.6 to -195.5, concluding that "saturating a bound is not the same
    # as solving a problem whose optimum lives on that bound". Charging for
    # |a| near 1 is the other direction: it moves the optimum off the bound
    # instead of trying to make the bound differentiable.
    #
    # Quartic so it is nearly free through the usable band and bites only in
    # the last ~20% of the range: at |a|=0.5 it costs 0.06x the weight, at 0.9
    # 0.66x, at 1.0 the full weight. 0.0 disables it, which is the default --
    # it must be switched on deliberately and measured, like everything else
    # in this file.
    action_saturation_k: float = 0.0

    # UNIFORM scale on the finished reward, shaping AND terminal together.
    #
    # `reward_scale` above applies to the shaping only, deliberately, so that
    # the terminal can outweigh the sum of hundreds of shaping steps. That
    # argument is about the RATIO between them and a uniform factor preserves it
    # exactly -- what it changes is the numeric range the critic has to fit.
    #
    # Measured why that matters: per-step reward here is ~0.5 while the terminal
    # is -450..+450, so episode returns are O(+-400). At gamma=0.998 over ~500
    # steps the critic must carry a +-450 jump back through the whole episode,
    # and with an MSE loss over that range it diverged outright -- Q ran to 1972
    # while actual returns sat at -429. Standard continuous-control benchmarks
    # keep returns O(1-10); this was two orders of magnitude outside that.
    #
    # 1.0 reproduces every number measured before this existed.
    # (1-gamma)-style value normalisation, the heuristic the published design uses to
    # keep value targets near unity ("multiplies the rewards accumulated over an episode
    # by a factor of 1-gamma", and "it is important to ensure that the magnitude of the
    # neural network outputs are reasonably close to unity"). Measured here rather than
    # copied: at scale 1.0 a successful episode's discounted V(s0) is about -22, so 0.05
    # puts it near -1. This project previously ran 0.1 with Q measured at -7 to -41.
    reward_total_scale: float = 0.05

    reward_scale: float = 0.00025


def descent_envelope_m_s(w: RewardWeights, alt_m: float) -> float:
    """The fastest descent the vehicle is allowed at this altitude."""
    return w.profile_c * float(np.sqrt(max(alt_m, w.profile_alt_floor_m)))


def _descent_envelope_penalty(w: RewardWeights, alt_m: float, v_z: float) -> float:
    """Vertical speed must be a DESCENT, and no faster than the envelope.
    Zero anywhere inside that band, quadratic on either way out of it."""
    excess_down = max(-v_z - descent_envelope_m_s(w, alt_m), 0.0)
    excess_up = max(v_z, 0.0)
    return min(w.profile_k * excess_down * excess_down
               + w.profile_climb_k * excess_up * excess_up, w.profile_cap)


def target_velocity(w: RewardWeights, alt_m: float, dx: float = 0.0, dy: float = 0.0):
    """The velocity the vehicle should be holding at this altitude.

    A FIELD, not an envelope. That distinction is the point: an envelope is a limit, so
    inside it the reward is flat and the agent learns nothing about how to descend.
    Measured on this project's own weights, at 10 m the envelope is 2.47 m/s and the
    penalty at vz = -1.00, -2.00 and -2.47 is 0.0000, 0.0000, 0.0000 -- no gradient
    anywhere in the working band, only a wall past it. The published design for this
    problem class (Gaudet/Linares/Furfaro, arXiv:1810.08719) tracks a velocity field
    instead, and that is what this returns.

    TWO MEASUREMENTS set the shape, and both corrected a first attempt:

    1. The vertical target is the CONTROLLER's measured descent schedule, not the
       envelope. Using the envelope as a target was wrong by a factor of ~2 -- it asks
       for 4.61 m/s at 35 m where the controller actually descends at 2.59, and scoring
       the controller against it produced a mean tracking error of 4.64 m/s and a dense
       sum of -249 for the policy that lands 92% of the time. A limit's value is not a
       good target. Fitted over the successful demo episodes:

           -vz = 0.547 * alt^0.431, floored at 0.80 m/s

       The floor matters: below 5 m the controller holds a constant 0.80 m/s (p50 = 0.80
       in every band from 0 to 5 m) while the power law decays to 0.33, so without it the
       field would ask for a slower touchdown than the controller uses, against a
       `safe_landing_v_z_m_s` of 1.0.

    2. The lateral target is ZERO EVERYWHERE, and there is no aim point over the pad.
       This task has no pinpoint requirement: `landed_safely` tests vz, v_xy, tilt, |w|
       and leg height difference, and contains no position term at all. The measured
       controller confirms it -- its offset from the target is 4.4 m (p50) at 25-36 m
       altitude and 25.9 m at touchdown, i.e. it bleeds off its spawn lateral velocity
       and lands wherever that leaves it. Gaudet's field aims at a point above the pad
       because their task demands a landing ellipse under 5 m radius; importing that
       would have been solving a problem this task does not have, and would have fought
       the 92% policy. What we actually want from the lateral axis is "kill the lateral
       velocity", which a target of zero states directly and with a gradient everywhere.

    `dx`/`dy` are accepted and ignored, so the signature survives if a pinpoint variant
    is ever wanted. Returns `(vx_t, vy_t, vz_t, t_go)`; `t_go` is returned because the
    observation wants it.
    """
    rate = w.vfield_vz_c * (max(alt_m, 0.0) ** w.vfield_vz_p)
    vz_t = -max(rate, w.vfield_vz_floor_m_s)
    t_go = max(alt_m, 0.0) / max(-vz_t, 1e-6)
    # The lateral target is a SPEED SCHEDULE, pointed along the vehicle's current lateral
    # heading rather than at any particular place. So the term asks "be down to this speed by
    # this altitude" and says nothing about direction -- which is right for a task whose
    # success criteria contain no position term, and which leaves the error a pure magnitude
    # difference instead of charging for a heading the criteria do not care about.
    speed = max(w.vfield_vxy_c * (max(alt_m, 0.0) ** w.vfield_vxy_p),
                w.vfield_vxy_floor_m_s)
    return float(speed), 0.0, float(vz_t), float(t_go)


def _velocity_field_penalty(w: RewardWeights, alt_m: float, dx: float, dy: float,
                             vx: float, vy: float, vz: float) -> float:
    """`vfield_k * ||v - v_targ||`, the dense core of the reward.

    Linear in the error, as in the published form (`alpha*||v - v_targ||`), not
    quadratic: a quadratic would make a large early error dominate the whole episode
    and go nearly flat once the error is small, which is the saturation pattern this
    project has already had to remove from three other terms.
    """
    if w.vfield_k <= 0.0:
        return 0.0
    speed_t, _unused, vz_t, _ = target_velocity(w, alt_m, dx, dy)
    # `target_velocity` returns the lateral target as a MAGNITUDE in its first slot; the
    # direction is the vehicle's own, so the lateral error is a scalar speed difference.
    lat = float(np.hypot(vx, vy))
    err = float(np.sqrt((lat - speed_t) ** 2 + (vz - vz_t) ** 2))
    return min(w.vfield_k * err, w.vfield_cap)


def _time_penalty(w: RewardWeights) -> float:
    """Per-step term. NEGATIVE here historically, which is backwards.

    The published design uses a small POSITIVE per-step reward and states the reason
    plainly: with every other term negative, the agent is "incentivized to violate the
    attitude constraint and prematurely terminate the episode to maximize the total
    discounted rewards received starting from the initial state."

    This project measured exactly that failure and treated the symptom instead of the
    cause. A crash cost -60 against a timeout's -105, so converting a hover into a
    crash was a +45 improvement, and the RL policy duly turned 33 of the clone's
    timeouts into 23 crashes with the landing count unchanged. Reordering crash and
    timeout helped; the sign of THIS term is why the pressure existed at all. Ours was
    0.1 per step against their +0.01 -- opposite sign and ten times the magnitude.

    `alive_bonus` is the positive term; `time_k` stays so the old behaviour can be
    restored for an ablation, and defaults to 0.
    """
    return w.time_k - w.alive_bonus


def _descent_progress_reward(w: RewardWeights, alt_m: float, v_z: float,
                              release_alt_m: float) -> float:
    """Pay for closing altitude, never for closing it faster than the
    envelope allows, normalised by the release altitude. See `progress_k`'s
    field comment for why the cap makes this unfarmable and why the
    normalisation is needed."""
    rate = min(max(-v_z, 0.0), descent_envelope_m_s(w, alt_m))
    return w.progress_k * rate / max(release_alt_m, 1.0)


def _action_saturation_penalty(w: RewardWeights, action) -> float:
    """Quartic charge on how close the action sits to the box bounds."""
    if w.action_saturation_k <= 0.0 or action is None:
        return 0.0
    a = np.clip(np.abs(np.asarray(action, dtype=float)), 0.0, 1.0)
    return float(w.action_saturation_k * np.mean(a ** 4))


def _horizontal_speed_penalty(w: RewardWeights, v_xy: float, alt_m: float = 0.0) -> float:
    ground_weight = 1.0 / (1.0 + max(alt_m, 0.0) / max(w.kxy_alt_ref_m, 1e-6))
    knee = max(w.kxy_knee_m_s, 1e-6)
    if v_xy <= knee:
        # multiplication order kept exactly as the plain quadratic had it,
        # so the proven sub-knee regime is bit-identical and not merely
        # equal to within rounding
        return min(w.kxy * v_xy * v_xy * ground_weight, w.kxy_cap)
    # value- and slope-continuous with the quadratic at `knee`
    charge = knee * knee + 2.0 * knee * (v_xy - knee)
    return min(w.kxy * charge * ground_weight, w.kxy_cap)


def _edge_penalty(w: RewardWeights, dx: float, dy: float, tile_size_m: float) -> float:
    """Quadratic charge once outside `edge_safe_frac` of the tile's half-extent.

    Zero in the usable interior, so a policy that stays where the ground is never pays it.
    See the `edge_k` block for the measurement that made this necessary.
    """
    if w.edge_k <= 0.0 or not tile_size_m:
        return 0.0
    safe = 0.5 * float(tile_size_m) * w.edge_safe_frac
    r = float(np.hypot(dx, dy))
    over = max(r - safe, 0.0)
    if over <= 0.0:
        return 0.0
    # normalised by the remaining margin, so the charge means the same thing on a 120 m tile
    # and a 1680 m one rather than scaling with the stage's size
    margin = max(0.5 * float(tile_size_m) - safe, 1e-6)
    frac = over / margin
    return min(w.edge_k * frac * frac, w.edge_cap)


def _tilt_cutoff_penalty(w: RewardWeights, tilt_x: float, tilt_y: float,
                          loss_of_control_tilt_rad: float,
                          safe_tilt_rad: float | None = None) -> float:
    """A continuous quadratic charge plus the steep loss-of-control barrier.

    The barrier alone was the defect: referenced to loss-of-control (40 deg) and
    raised to the 8th power, it charged 0.0002 per step at the 15 deg criterion
    that decides `landed_safely`, against 0.144 for lateral speed at ITS limit. So
    the reward never asked the policy to stay upright until it was already tumbling.
    The quadratic term is referenced to the SAFE criterion instead, which is the
    one the outcome is graded on; the barrier is kept because loss-of-control is a
    genuinely different regime that should be expensive to approach.

    `safe_tilt_rad=None` keeps the old barrier-only behaviour, so callers that have
    not been updated are unchanged rather than silently re-weighted.
    """
    tilt = float(np.hypot(tilt_x, tilt_y))
    ratio = tilt / max(loss_of_control_tilt_rad, 1e-6)
    barrier = min(w.tilt_cutoff_k * float(ratio ** w.tilt_cutoff_power), w.tilt_cutoff_cap)
    if safe_tilt_rad is None or w.tilt_safe_k <= 0.0:
        return barrier
    safe_ratio = tilt / max(safe_tilt_rad, 1e-6)
    return w.tilt_safe_k * safe_ratio * safe_ratio + barrier


def _angular_rate_penalty(w: RewardWeights, wx: float, wy: float, wz: float,
                           safe_w_rad_s: float) -> float:
    w_mag = float(np.sqrt(wx * wx + wy * wy + wz * wz))
    quad = w.omega_k * w_mag * w_mag
    ratio = w_mag / max(safe_w_rad_s, 1e-6)
    barrier = min(w.omega_cutoff_k * float(ratio ** w.omega_cutoff_power), w.omega_cutoff_cap)
    return min(quad + barrier, w.omega_penalty_cap)


def _touchdown_severity(p, s, margins: dict) -> float:
    """How many times over its worst limit a touchdown was; 1.0 is exactly
    at a limit. Read from the STATE, because `landing_margins` are clipped
    to [0, 1] and so carry no information about the SIZE of a violation --
    which is the whole thing being graded. leg_diff is the one criterion
    not in the state; its margin saturating at 0 is taken as "just over",
    since terrain flatness is not the axis a dive blows out.
    """
    v_xy = float(np.hypot(s["vx"], s["vy"]))
    tilt = float(np.hypot(s["tilt_x"], s["tilt_y"]))
    w_mag = float(np.linalg.norm([s.get("wx", 0.0), s.get("wy", 0.0), s.get("wz", 0.0)]))
    sev = max(
        abs(s["vz"]) / max(p.safe_landing_v_z_m_s, 1e-6),
        v_xy / max(p.safe_landing_v_xy_m_s, 1e-6),
        tilt / max(p.safe_landing_tilt_rad, 1e-6),
        w_mag / max(p.safe_landing_w_rad_s, 1e-6),
    )
    if margins and margins.get("leg_diff", 1.0) <= 0.0:
        sev = max(sev, 1.0)
    return float(sev)


def _terminal_reward(w: RewardWeights, info: dict, severity: float | None = None) -> float:
    """One continuous, monotone ramp in touchdown severity.

    See the touchdown_k block in RewardWeights for the measurement that forced
    this shape. The short version: the previous form jumped 411 points across the
    success boundary while the entire dense shaping sum over a 448-step episode was
    32, so the only gradient the actor could perceive was "commit and crash" and it
    took it. This version is continuous at severity 1.0, so approaching the limit
    pays continuously instead of paying nothing until it is crossed.
    """
    margins = info.get("landing_margins") or {}
    if not margins:
        # No margins means no touchdown happened at all: a lost_control tumble,
        # which gets the floor. There is nothing to grade continuously here --
        # the vehicle never reached the surface.
        return -w.touchdown_penalty_cap
    # severity < 1 is the landed_safely condition, so one expression covers both
    # outcomes and the two boundaries cannot drift apart the way they did when a
    # label fix and a grading fix were applied to different states.
    sev = float(severity) if severity is not None else 1.0
    return float(max(w.touchdown_k * (1.0 - sev), -w.touchdown_penalty_cap))


def make_apollo_reward_fn(weights: RewardWeights | None = None, specs: ApolloLMSpecs | None = None):
    """Build a `reward_fn(env, info)` for `AnalyticLanderEnv`,
    `IsaacLanderEnv` and `IsaacLanderVecEnv` -- all three expose the same
    `.params` / `.state` pair, which is all this touches."""
    w = weights or RewardWeights()

    def reward_fn(env, info: dict) -> float:
        p = env.params
        s = env.state
        alt = float(info["altitude_m"])
        v_xy = float(np.hypot(s["vx"], s["vy"]))

        dx = float(s["x"]) - float(getattr(p, "target_x", 0.0))
        dy = float(s["y"]) - float(getattr(p, "target_y", 0.0))
        shaping = (
            -_descent_progress_reward(w, alt, float(s["vz"]), p.spawn_altitude_m)
            + _descent_envelope_penalty(w, alt, float(s["vz"]))
            + _velocity_field_penalty(w, alt, dx, dy,
                                       float(s["vx"]), float(s["vy"]), float(s["vz"]))
            + _edge_penalty(w, dx, dy, getattr(getattr(env, "tile", None), "size_m", 0.0))
            + _time_penalty(w)
            + _horizontal_speed_penalty(w, v_xy, alt)
            + _tilt_cutoff_penalty(w, s["tilt_x"], s["tilt_y"], p.loss_of_control_tilt_rad,
                                    p.safe_landing_tilt_rad)
            + _angular_rate_penalty(w, s.get("wx", 0.0), s.get("wy", 0.0), s.get("wz", 0.0),
                                     p.safe_landing_w_rad_s)
            + _action_saturation_penalty(w, info.get("action"))
        )
        r = -min(shaping, w.shaping_clip_abs) * w.reward_scale
        terminal_part = 0.0

        if info.get("terminated"):
            # Grade the severity from the IMPACT state when the env publishes
            # one. Both Isaac envs overwrite `env.state` with POST-substep
            # velocities before calling this function, and after contact the
            # regolith (restitution 0) has already stopped the vehicle -- so
            # reading `s` here scored every slam at vz ~ 0.
            #
            # That was measured: a 7.6 m/s impact delivered -60 instead of -360,
            # which put crashing (-60) AHEAD of timing out (-105) and made a
            # tumble (-360) six times worse than a hard slam. Fixing the LABEL
            # (`landed_safely`, `landing_margins`) without fixing this left the
            # penalty flat, which is worse than either state alone.
            #
            # The analytic env has no contact solver, so its `state` already
            # holds the true impact velocity and the fallback is correct there.
            grade_state = info.get("impact_state") or s
            term = _terminal_reward(w, info,
                                    _touchdown_severity(p, grade_state,
                                                        info.get("landing_margins") or {}))
            r += term
            terminal_part += term
        elif info.get("truncated"):
            # Charge the timeout penalty only for an actual TIMEOUT. The other
            # way an episode truncates is `left_tile`, which is an artificial
            # cut at the edge of the terrain collider rather than a failure the
            # policy should be taught to avoid by this term -- and it is also
            # the one that still bootstraps, so paying the penalty there would
            # charge for the state AND carry its value forward.
            #
            # `timed_out` is published by the envs; older callers that only set
            # `truncated` keep the previous behaviour.
            if info.get("timed_out", True):
                r -= w.timeout_penalty
                terminal_part -= w.timeout_penalty
        # Publish the SPLIT so a dual-discount critic can learn the two streams with
        # separate gamma. arXiv:1810.08719 lists multiple discount rates as a primary
        # contribution and is explicit about the cost of not having them: "Without the
        # use of multiple discount rates, the performance was actually worsened by
        # including the terminal reward term." That matches this project's own gamma
        # sweep exactly -- gamma 0.99 bounded the critic but discounted the terminal to
        # 0.011 of its value over 450 steps, gamma 0.998 kept the terminal visible and
        # the critic diverged, and gamma 0.995 split the difference and still failed.
        # The resolution is not a value between them; it is two values.
        #
        # Written into `info` rather than returned, so every existing caller of
        # `reward_fn` keeps the same scalar interface and only a buffer that wants the
        # split has to look for it.
        info["reward_terminal"] = float(terminal_part * w.reward_total_scale)
        info["reward_shaping"] = float((r - terminal_part) * w.reward_total_scale)
        return float(r * w.reward_total_scale)

    # Published on the closure so the ENVS can read the very weights this
    # reward uses when they build the guidance-field slots of the observation
    # (obs 16-19). Those slots are `target_velocity(w, ...)` evaluated at the
    # current state, so if the env guessed a default `RewardWeights()` while
    # the reward ran on tuned ones, the policy would be told to track one
    # field and paid for tracking another -- a silent train-time skew of
    # exactly the kind this project has already paid for once.
    reward_fn.reward_weights = w
    return reward_fn


default_reward_fn = make_apollo_reward_fn()
