"""Reward shaping for `AnalyticLanderEnv`, built from the term-by-term
reward specification handed over for this vehicle: attitude/RCS-based
(no TVC/gimbal term, since the real DPS has none for attitude control),
with every normalization grounded in `lunarsim.core.vehicle.apollo_lm`'s
real numbers (mass, DPS thrust range, RCS jet count/thrust, leg load
estimate) rather than left as unexplained magic constants.

The handed-over formula left its coefficients symbolic on purpose ("genel
haliyle bırakıyorum... modele göre hesaplanacak"); this module fills them
in, and documents each interpretation choice, since a few of the symbols
(I_MV, T0, zeta, theta) weren't otherwise defined:

  -alpha*(1/pos)^beta          -> `_proximity_penalty`: a genuine
                                   1/distance term (clipped near zero),
                                   which discourages loitering exactly
                                   over the pad at altitude without ever
                                   committing to touchdown, rather than a
                                   "get closer" incentive (that's the
                                   separate quadratic tracking term below).
  -exp(I_MV^theta)*T0*Height^zeta
                                -> `_braking_envelope_penalty`: read as a
                                   guided-descent "stay under the braking
                                   parabola" barrier -- I_MV is the ratio
                                   of current descent rate to a
                                   sqrt(altitude)-shaped speed limit
                                   (zeta=0.5, matching real PDI guidance
                                   profiles), theta shapes how sharply the
                                   barrier turns on, T0 is a fixed scale.
  -h*(FootPressure/FootPressureMax)^c
                                -> `_leg_load_penalty`: estimated per-leg
                                   touchdown force (from vertical KE at
                                   contact spread over the real leg count
                                   and an engineering-estimate strut
                                   stroke) over `leg_force_bounds_n`'s
                                   estimated max -- zero while airborne.
  -S*throttle^2                 -> `_throttle_effort_penalty`
  -lambda*(|pitch|+|roll|+|yaw|)
                                -> `_rcs_l1_penalty`: fuel-linear RCS
                                   effort, replacing the TVC-gimbal term
                                   this vehicle doesn't have.
  -gamma*(pitch^2+roll^2+yaw^2)
                                -> `_rcs_l2_penalty`: discourages
                                   saturating all three RCS axes at once.
  -x1*relx^x2 - x4*rely^x3 - x5*Vxy^x6
                                -> `_velocity_tracking_penalty`. The
                                   handed-over spec's relx/rely position
                                   terms were REMOVED this session (see
                                   that function's docstring): a hand-
                                   designed feasibility controller proved
                                   they actively fight the correct
                                   high-speed braking maneuver, and
                                   `landed_safely` never checks touchdown
                                   position anyway. Only -x5*Vxy^x6 remains.
  +Relative(Landing-reward) - Crash
                                -> `_terminal_reward`: graded by how far
                                   *under* each safety threshold the
                                   touchdown was (a feather-soft landing
                                   scores higher than a marginal one), or
                                   a large fixed penalty on a crash.

Only the *normalizations* claim physical grounding (throttle as a fraction
of the real DPS range, RCS commands already normalized to jet duty cycle,
leg force as a fraction of the real leg-count-based estimate). The scalar
weights multiplying each normalized term are ordinary reward-shaping
constants -- sensible defaults, meant to be retuned by watching training,
same as any RL reward.

One term was ADDED beyond the handed-over spec, found necessary by
actually training against it: `_attitude_hold_penalty`, a small
continuous `-mu*(tilt_x^2+tilt_y^2)`. The original spec only penalizes
RCS *effort* (L1/L2 above), never attitude *error* itself -- a policy
that stops firing RCS altogether pays zero attitude penalty even while
tumbling. A real training run confirmed this: the trained policy reached
200+ degrees of tilt and free-fell the rest of a 25s episode with no
corrective signal at all. `lunarsim.rl.analytic_lander_env`'s
`loss_of_control_tilt_rad` cutoff (ends the episode immediately past 60
deg) bounds the damage; this term is what should stop it from drifting
that far in the first place. It's since been made velocity-gated (see its
field comments) rather than a flat rate, once a working hand-designed
controller showed a flat version fights the tilt a correct high-speed
braking maneuver actually needs.

THIS SESSION'S REVISION (see `lunarsim/control/zemzev_controller.py` for
the hand-designed guidance+attitude controller whose trajectories were
used to find these): built and tuned a non-RL controller to check whether
`orbit_descent` (200m release, 10-30 m/s horizontal, real terrain) was
control-theoretically achievable at all with this vehicle model, before
touching this reward again -- it reached 75% `landed_safely` (zero
crashes) after fixing several of its own bugs, proving the mission IS
achievable and that the earlier training failures were not purely a hard
physical ceiling. Replaying that controller's own (successful) episodes
through this reward function surfaced three real problems with the reward
itself, all fixed here:
  1. `_altitude_penalty`/`braking_penalty_cap` (new): `_altitude_penalty`
     and `_braking_envelope_penalty` were each individually capable of
     blowing past the entire `shaping_clip_abs` budget by themselves at
     orbit_descent's 200m release altitude -- see `_altitude_penalty`'s
     field comments for the exact numbers (and a SECOND bug found later,
     via a real trained checkpoint's Isaac Sim eval: an early hard-cap fix
     for this let a policy climb away and sit at high altitude forever to
     dodge the landing risk entirely, since altitude stopped costing
     anything past the cap -- now a sqrt shape, always growing, never
     free). This silently flattened the per-step shaping reward to a
     constant during the highest-altitude portion of every orbit_descent
     episode, exactly where the policy most needed a gradient toward
     "start braking now."
  2. `_attitude_hold_penalty` was a flat quadratic on tilt, but the
     controller needed 25-35 deg of SUSTAINED tilt to solve orbit_descent
     -- a flat penalty fights that directly. Made velocity-gated (cheap
     while v_xy justifies it, full cost once v_xy is low).
  3. `_velocity_tracking_penalty` (renamed from `_position_tracking_penalty`):
     dropped the relx/rely position-chasing terms. `landed_safely` never
     checks touchdown position, and the controller demonstrated this term
     actively fighting the correct maneuver (kept tilt alive to close a
     lateral offset that didn't matter, which then couldn't unwind in time).

ANOTHER REVISION (this session, reviewed against external SAC reward-design
guidance: dense per-step shaping should guide behavior, but the sparse
terminal/goal-achievement signal must stay large enough, relative to the
SUM of per-step shaping over an episode, to actually determine outcomes --
scale/relative-magnitude matters far more than sign). Checked this
session's own orbit_descent training logs: total episode returns ran
-900 to -3900 (0/16 `landed_safely` in every sanity check), while the
terminal bonus/penalty was being shrunk by the SAME `reward_scale=0.01`
used for a single per-step term -- leaving it at +-5..6, under 1% of a
typical episode's accumulated shaping, i.e. numerically inert regardless
of how carefully its own breakeven math (crash vs. landing vs. timeout)
was tuned. Fixed: `reward_scale` now applies only to the per-step shaping
sum; the terminal bonus/penalty is added afterward at its own raw scale,
and `landing_bonus_scale`/`crash_penalty`/`timeout_penalty` were raised 3x
(ratios preserved) so the terminal signal lands within the observed
per-episode shaping range instead of merely surviving. See the
`reward_scale` and `landing_bonus_scale` field comments for the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, leg_force_bounds_n


@dataclass
class RewardWeights:
    # proximity singularity term: -alpha * (1/max(dist, dist_floor))^beta
    # REAL BUG, FOUND VIA A FAILED TRAINING RUN: at alpha=2.0 this is a
    # constant ~1/step tax merely for being near the pad, for the ENTIRE
    # episode, independent of behavior quality -- it never turns off
    # except at termination. A careful, correct landing takes longer than
    # a reckless tumble-crash, so the extra steps of this constant tax
    # exceeded the terminal reward gap between landing safely and
    # crashing: crashing fast was mathematically the better policy, and
    # 1.5M steps of SAC training correctly found it (0/40 landed, 40/40
    # hit the loss-of-control cutoff -- got WORSE with more training,
    # which is what a real optimum being found looks like, not what an
    # undertrained policy looks like). Cut 10x; see also the much larger
    # terminal-reward rebalancing below.
    alpha: float = 0.2
    beta: float = 1.0
    dist_floor_m: float = 2.0

    # altitude closure (ADDED beyond the handed-over spec, see module
    # docstring): -kappa_alt * altitude_m. FOUND NECESSARY VIA A FAILED
    # TRAINING RUN: `_proximity_penalty` above only measures HORIZONTAL
    # distance to the pad (x, y) -- nothing in the original spec's terms
    # costs remaining ALTITUDE specifically. A trained policy exploited
    # this exactly: it found a genuinely excellent, stable hover directly
    # over the pad (tilt 1.5-8 deg, vxy 0.2-1.4 m/s, vz~0 -- all already
    # within the safe-landing thresholds) at ~20m and simply never
    # descended for the entire episode, since nothing was pushing it to.
    # This restores the "encourage descending" term the pre-rewrite
    # `default_reward_fn` had (as a flat `-0.01*alt`) that got dropped
    # when the handed-over spec's term list was implemented verbatim.
    #
    # 0.05, then 0.3, both only partially worked: each retrain converged
    # to the SAME hover-forever pattern at a lower equilibrium altitude
    # (0.05 -> ~13-20m, 0.3 -> ~2.6-3.0m) -- a clean, repeated, linear
    # confirmation the term works, just not steeply enough yet to beat
    # whatever tiny residual risk-aversion stops it short. At 0.3 the
    # remaining gap is almost comically small: vz/vxy/tilt are ALREADY
    # within every safe-landing threshold while hovering at ~2.8m, it
    # just won't cross the last few meters. Raised another ~7x.
    # REAL BUG FOUND (via replaying a known-good hand-designed controller's
    # orbit_descent trajectory through this reward function and inspecting
    # each term's magnitude -- see zemzev_controller.py): the original flat
    # `kappa_alt * alt_m` was never rechecked against orbit_descent's 200m
    # release altitude. At kappa_alt=2.0, alt=200m alone gave a penalty of
    # 400 -- bigger than the ENTIRE shaping_clip_abs budget (300 at the
    # time), at step one, before any other term is even read. Every
    # per-step shaping reward for a large fraction of every orbit_descent
    # episode (the whole high-altitude portion, exactly where the policy
    # most needs to decide whether to commit to braking) was a flat -300
    # regardless of behavior -- all gradient destroyed exactly where it
    # mattered most.
    #
    # REAL BUG FOUND #2 (via an actual trained checkpoint's real Isaac Sim
    # eval, this session): the FIRST fix for the above was a hard cap
    # (`min(kappa_alt*alt, 80)`). That solved the swamping problem but
    # created a new one -- once alt exceeds ~40m the penalty is completely
    # FLAT, so climbing to 330m costs exactly the same as sitting at 41m.
    # A curriculum-trained policy found and exploited exactly this: rather
    # than attempt the risky orbit_descent landing (crash_penalty=700 on
    # failure), it just climbed away from 198m up to a stable ~330m and
    # sat there, since the ongoing shaping cost of staying high was
    # provably no worse than descending partway and no better than
    # descending fully -- a variant of the same "avoid the risk entirely"
    # exploit `timeout_penalty` was added to stop, just at a higher
    # altitude where that fix doesn't reach (an episode that never times
    # out airborne near the ground, because it's still 300m up when
    # `max_episode_s` truncates, was never covered by that fix).
    #
    # Fixed by replacing the hard cap with a sqrt shape: unbounded and
    # ALWAYS strictly increasing with altitude (no altitude is ever "free"
    # to sit at) while still growing sublinearly, so it doesn't reintroduce
    # bug #1's swamping at 200m+. Bonus: sqrt's derivative (1/(2*sqrt(alt)))
    # is LARGER near alt=0 than the old constant slope, which if anything
    # strengthens the original "cross the last few meters" incentive
    # kappa_alt was added for in the first place.
    kappa_alt_sqrt: float = 12.66  # calibrated so sqrt(40)*kappa ~= old kappa_alt(2.0)*40 = 80
    # REAL BUG FOUND (this session, direct user observation -- "it just
    # prefers to crash straight down"): see `_altitude_penalty`'s comment.
    # Gates the descend-incentive by `1/(1+altitude_vz_gate_k*braking_ratio)`.
    # k=1.0: at ratio=1 (right at the edge of the real stopping-distance
    # budget) the descend incentive is already halved; by ratio=4 it's down
    # to 20%; by ratio=9 it's under 10%. Picked as a clean, no-magic-number
    # starting point (gate=0.5 exactly at the point the braking barrier
    # itself starts mattering, ratio=1) rather than a separately-fit
    # constant -- retune together with `t0`/`braking_authority_margin` if a
    # real run shows the gate engaging too early/late.
    altitude_vz_gate_k: float = 1.0

    # braking-envelope barrier: -exp(min(ratio^theta, cap)) * t0, where
    # `ratio` is REAL required-vs-available deceleration (see
    # `_braking_envelope_penalty`'s docstring for the full redesign
    # rationale -- this replaced an arbitrary `v_limit(alt) = safe_v_z +
    # k*alt**zeta` altitude-power-law schedule this session, after that
    # version was caught reproducing the exact documented "coast then
    # panic" failure on a real trained checkpoint).
    # 0.5 -> 1.5 fixed vz (0.87, safe) but vxy got WORSE (0.65 -> 1.74) --
    # a real physical coupling, not independent knobs: tilting to correct
    # one axis induces drift on the other. Settled at the midpoint (1.0)
    # with a longer training budget so the network has room to find a
    # joint solution instead of the two terms fighting each other on
    # successive short continuations.
    # REAL BUG FOUND (this session, caught live watching a fresh
    # orbit_descent training run in the Isaac Sim viewport -- the vehicle
    # was free-falling instead of braking): t0=1.0 was tuned against the
    # OLD v_limit(alt) formula above, never re-checked against the NEW
    # stopping-distance formula's actual magnitude relative to
    # `_altitude_penalty` (kappa_alt_sqrt=12.66, ~125-180 through most of
    # a 200m release). Worked the numbers for a real orbit_descent free-
    # fall trajectory (dry+fuel mass ~9000kg, thrust-to-weight margin
    # a_avail~3.38 m/s^2): at alt=100m, vz~18 m/s (natural free-fall
    # value from a 200m release), `ratio~1.92` gave a braking penalty of
    # only ~3.7 against an altitude penalty of ~127 -- 34x weaker, i.e.
    # no real counter-pressure against "just keep descending" until the
    # vehicle is already inside ~30-50m of the ground. Raised 20x so the
    # two terms are actually comparable well before the danger zone
    # (same alt=100m point: ~74 vs ~127, same order of magnitude instead
    # of 34x apart; by alt=70m braking already exceeds altitude penalty).
    t0: float = 20.0
    theta: float = 2.0
    # floors `alt_m` in the `vz**2/(2*alt)` required-deceleration kinematic
    # formula -- purely a divide-by-zero guard near touchdown, not a
    # tunable shaping knob. Set near the final-approach-style "last few
    # meters" scale used elsewhere (e.g. the hand-designed controller's own
    # `final_approach_alt_m` default).
    braking_alt_floor_m: float = 3.0
    # REAL BUG FOUND validating the redesign above against the actual
    # failing trajectory's own numbers: comparing `a_required` to the FULL
    # `a_avail` means, by construction, `ratio` can't signal danger until
    # the vehicle is already AT the edge of its real stopping capability --
    # correct physics, but useless as an early-warning shaping signal,
    # since by the time ratio=1 there's no altitude left to actually react
    # with (the same "corner you can't unpaint yourself out of" problem the
    # hand-designed controller solved with its own
    # `thrust_authority_margin=0.85` -- see zemzev_controller.py). Scaling
    # the comparison down to a FRACTION of real authority makes the barrier
    # engage while genuine reaction room still remains, not exactly at the
    # physical limit.
    braking_authority_margin: float = 0.25
    # REAL BUG FOUND (real-vehicle telemetry from a trained orbit_descent
    # checkpoint, spotted by inspecting the eval dashboard): released at
    # 200m/~20 m/s, the policy held throttle~1.0 (net near-zero -- just
    # cancelling gravity, not actually braking) through the whole high-
    # altitude/high-speed part of the descent, then genuinely EASED OFF
    # throttle (0.77-0.9) through the middle of the flight -- exactly
    # backwards from a real braking profile -- before slamming back to
    # ~1.0 only in the last ~20m, too late to shed the remaining speed.
    # Root cause (of the OLD v_limit(alt) version): at cap=4, exp(4)~=54.6
    # is reached (and clipped flat) the instant the old ratio exceeded 2 --
    # true for nearly the WHOLE descent from a 200m/20+ m/s release, since
    # v_limit(alt) was only a few m/s through most of that range. Once
    # saturated, throttling harder earned ZERO extra reward from this term,
    # so the policy had no local gradient telling it braking now beats
    # braking later. Raising the cap alone (this comment's original fix)
    # only pushed the flat region further out -- a LATER real checkpoint
    # eval (this session, after several other fixes) reproduced the
    # identical coast-then-panic shape regardless, which is what motivated
    # the full redesign (real stopping-distance physics, see
    # `_braking_envelope_penalty`) rather than another threshold retune.
    # `exp_arg_cap` (the old exponent safety clip) is gone -- superseded by
    # the polynomial-shape redesign in `_braking_envelope_penalty`, which
    # can't overflow the way an uncapped `exp()` argument could, so no
    # equivalent guard is needed.
    #
    # REAL BUG FOUND #2 (a real trained checkpoint's Isaac Sim eval, this
    # session, after the stopping-distance redesign above): `ratio` (real
    # required-vs-available deceleration) hit 3.5 as early as alt=47m in a
    # failed landing, then kept climbing to 10, 21, and 75 all the way to
    # touchdown -- but since `exp(ratio**2)` blows past 100 once ratio
    # exceeds ~2.14, EVERYTHING from "3.5x over budget" to "75x over
    # budget" collapsed to the identical penalty (100). The policy had a
    # correct, strong "you are in trouble" signal the whole way down, but
    # zero gradient telling it whether continuing to brake harder was
    # helping AT ALL once already in that zone -- exactly the region a
    # policy that's still learning to commit to braking needs the most
    # differentiation in. Raised 2.5x (with a matching raise to
    # `shaping_clip_abs` below, so this doesn't reopen the original
    # swamping bug at more moderate violation levels).
    # Raised again alongside `t0`'s 1.0->20.0 fix (see its comment): at the
    # new t0, the old cap=250 would saturate at ratio=sqrt(250/20)~=3.5,
    # much sooner than the ~16 the previous cap/t0 pair differentiated up
    # to. Raised to 400 to keep differentiation out to ratio~4.5
    # (sqrt(400/20)) -- less extreme-tail range than before, but the new
    # t0 means the policy gets real pressure to brake much earlier, so the
    # extreme tail (ratio 5-75) matters less: a policy responding to the
    # earlier signal shouldn't be reaching those ratios in the first place.
    braking_penalty_cap: float = 400.0

    # leg touchdown load: -h * (leg_force / leg_force_max)^c
    h: float = 1.0
    c: float = 2.0

    # DPS effort: -s * throttle^2
    s: float = 0.05

    # RCS effort (replaces the TVC/gimbal term -- this vehicle has none)
    lam: float = 0.02   # L1, fuel-linear
    gamma: float = 0.01  # L2, discourages saturating all axes at once

    # attitude hold (ADDED beyond the handed-over spec -- see module docstring).
    # mu=0.5 was tried first and was too weak: a real training run still hit
    # the 60 deg loss-of-control cutoff in 39/40 eval episodes despite it.
    # Raised 10x to 5.0 -- then found (via the feasibility-controller
    # instrumentation pass) to be fighting a maneuver that's actually
    # NECESSARY: a hand-designed guidance+attitude controller could only
    # solve orbit_descent (200m release, 10-30 m/s horizontal) by
    # committing to 25-35 deg of SUSTAINED tilt for ~15-20s to brake
    # horizontally, then unwinding it before touchdown -- a flat
    # mu*(tilt_x^2+tilt_y^2) penalizes that entire commitment phase just as
    # hard as genuine loss-of-control tumbling, with no distinction between
    # "tilted because it's earning its keep" and "tilted for no reason."
    # Split into a small always-on floor (mu_base, catches real tumbling
    # regardless of speed) plus a velocity-gated component (mu_scaled) that
    # only turns on as v_xy drops toward zero -- see `_attitude_hold_penalty`.
    # mu_base + mu_scaled sums to the old mu=5.0 for the "tilted with no
    # excuse" case (v_xy~0), preserving that regression test's behavior,
    # while costing far less during a legitimate high-speed braking tilt.
    mu_base: float = 0.5
    mu_scaled: float = 4.5
    # v_xy at/above which tilt is "fully justified" (no attitude penalty
    # beyond mu_base) -- reuses the same reference speed the feasibility
    # controller's own tilt-taper needed to avoid the "can't unwind in
    # time" overshoot failure (see zemzev_controller.py's
    # `tilt_taper_v_xy_m_s`), so the reward and a known-working controller
    # agree on when tilt stops being free.
    attitude_hold_v_xy_ref_m_s: float = 12.0

    # REAL BUG FOUND (this session, real orbit_descent training telemetry:
    # lost_control rate climbing 0/16 -> 7/16 -> 11/16 -> 16/16 across
    # successive extensions of the SAME training run, despite critic_loss/
    # ent_coef staying healthy throughout -- a genuine behavioral drift,
    # not a training-instability artifact). Root cause: the velocity-gated
    # `mu_base`/`mu_scaled` term above makes tilt nearly free (down to
    # `mu_base=0.5`) once v_xy is high -- correct for ALLOWING the ~25-35
    # deg braking tilt the feasibility controller needs, but nothing
    # independently discourages tilting further, all the way toward the
    # 60 deg `loss_of_control_tilt_rad` cutoff itself, as long as v_xy is
    # still high. As training made the policy commit harder to braking,
    # it drifted past that cutoff more and more often instead of
    # saturating around the ~25-35 deg that's actually sufficient. Adds a
    # steep, velocity-INDEPENDENT barrier on proximity to the cutoff
    # (`_attitude_hold_penalty`'s second term): negligible through the
    # controller-proven useful range (~35 deg -> ratio 0.58, barrier~27),
    # meaningful by 50 deg (ratio 0.83, barrier~460), sharply large
    # approaching 59 deg (ratio 0.98, barrier~1740) -- a real ceiling the
    # velocity gate alone doesn't provide.
    tilt_cutoff_k: float = 2000.0
    tilt_cutoff_power: float = 8.0
    tilt_cutoff_cap: float = 2000.0

    # velocity tracking: -x5*vxy^x6 (renamed from "position/velocity
    # tracking" -- REAL BUG FOUND via the feasibility-controller
    # instrumentation pass: the old x1/x4 terms pulled toward the EXACT
    # (target_x, target_y), but `landed_safely` never checks touchdown
    # position, only velocity/tilt/rate. The hand-designed controller hit
    # this directly: including a position-return term kept it committing
    # tilt to close out a lateral offset that didn't matter, and because
    # attitude response is slow, that leftover tilt didn't unwind before
    # touchdown and re-accelerated the vehicle sideways in the final
    # seconds -- undoing an already-nulled horizontal velocity. Dropping
    # the position pull entirely (spawn already aims initial velocity at
    # the target, so braking naturally keeps the vehicle in the general
    # area) removes that self-inflicted conflict; only the velocity term,
    # which does matter for `landed_safely`, remains.
    # 0.05 -> 0.2 was tried alongside t0=1.5 above and overshot the other
    # direction once t0 also moved. Settled at the midpoint (0.125) --
    # see the t0 comment for why these two need to move together with
    # more training time, not be independently maxed out.
    x5: float = 0.125
    x6: float = 2.0
    position_penalty_cap: float = 200.0

    # REAL BUG FOUND (this session, user direction after a 3-seed eval
    # showed three distinct failure modes across the release envelope --
    # one almost-pure-horizontal miss, one near-free-fall hard impact, one
    # never-landed timeout): re-adding a gated pull toward (target_x,
    # target_y), after extensive training on velocity-only shaping still
    # hadn't converged to one robust strategy. This is DELIBERATELY NOT
    # the same term removed above (see x5/x6's history) -- that version
    # was flat/always-on and fought the final unwind-and-brake maneuver
    # right at touchdown. This one is gated by altitude (`position_alt_
    # gate_m`): full weight far from the ground (where heading toward the
    # pad is pure upside, no conflict with anything else), fading to zero
    # by the time the vehicle is low enough that the velocity/tilt-cutoff
    # terms need to dominate uncontested -- same "fade out before it can
    # conflict with the final maneuver" shape `_attitude_hold_penalty`'s
    # velocity gate already uses, applied to the axis (altitude) that
    # actually matters for state here.
    x1: float = 0.05
    x2: float = 2.0
    position_alt_gate_m: float = 40.0
    xy_position_penalty_cap: float = 150.0

    # terminal. The old 100/100 was too small (episode duration dominated
    # economics, not the terminal outcome -- see the alpha comment above).
    # 2000/3000 overcorrected: a real training run showed 39/40 episodes
    # timing out with the vehicle still airborne, often at *already-safe*
    # vz (0.2-1.8 m/s) but never committing to touchdown -- crash_penalty
    # was so large relative to a landing attempt's failure odds that
    # never trying became the rational policy, and timeout carried NO
    # penalty at all, making "hover forever" a completely free way to
    # dodge the risk. Fixed both: crash_penalty brought down closer to
    # landing_bonus_scale (still bigger, but not overwhelming), and
    # timeout_penalty added so passively refusing to land isn't free.
    #
    # REAL BUG FOUND #2 (this session, multiple real orbit_descent training
    # runs, surviving BOTH the altitude-penalty and braking-envelope
    # redesigns above): 500/700/400 still leaves "never attempt landing"
    # the rational choice for as long as the policy's ESTIMATED success
    # probability stays low -- exactly the situation at the start of
    # orbit_descent training (a much harder distribution than the
    # curriculum stages before it; SAC's replay buffer resets at the stage
    # boundary, so the value function has to re-learn from scratch here).
    # Expected value of attempting at success rate p: p*landing_bonus_scale
    # - (1-p)*crash_penalty. At p=700/(700+500)=0.583 this breaks even
    # against a guaranteed -crash_penalty; ANYTHING below that success rate
    # makes attempting strictly worse than accepting -timeout_penalty
    # outright. A policy climbed away from a 200m/10-30 m/s release
    # (198m -> 350m+, throttle ~0.92-0.97, i.e. actively fleeing upward,
    # not passively drifting) and stayed there for the whole episode,
    # in BOTH a run with the old braking envelope and a run with the new
    # one -- this is a risk-calculus problem, not a shaping-gradient
    # problem, and no amount of altitude/braking shaping can fix it: those
    # terms only affect the PER-STEP cost of being at a given
    # altitude/speed, never the terminal calculus that makes attempting the
    # landing look like the worse bet in expectation. Fixed by moving
    # crash_penalty BELOW landing_bonus_scale (breakeven now at
    # 500/(500+600)=0.455 -- under 50% estimated success is enough to make
    # attempting worthwhile) while keeping crash strictly worse than
    # timeout, preserving the original ordering this field's history
    # established.
    # REAL BUG FOUND (this session, reviewed against external SAC reward-
    # design guidance the user supplied: "scale matters more than sign; the
    # goal-achievement/terminal signal must not be swamped by accumulated
    # dense penalty"). Checked real orbit_descent sanity-check episode
    # returns from this session's own training logs (e.g.
    # isaac_train_tiltcutoff_v1..v5.log, isaac_train_xypos_v1.log): total
    # EPISODE reward (summed over ~600-1200 steps at dt=0.05s, i.e. a 30-60s
    # episode) ranged from about -900 to -3900 -- almost entirely dense
    # per-step shaping, since most of those episodes never land at all
    # (0/16 `landed_safely` in every one of this session's sanity checks).
    # Meanwhile the terminal bonus/penalty below was being multiplied by the
    # SAME `reward_scale=0.01` as every per-step term (see that field's
    # comment) -- correct for keeping an individual step's shaping in a
    # critic-friendly range, but `reward_scale` was calibrated against a
    # SINGLE step's magnitude and never re-checked against the SUM across an
    # entire episode. Net effect: landing_bonus_scale=600 and
    # crash_penalty=500 became +-5..6 after scaling, a mere 0.1-0.6% of a
    # typical -900..-3900 episode return. The entire carefully-tuned
    # breakeven-probability rebalancing documented below (moving
    # crash_penalty below landing_bonus_scale so attempting a landing is
    # worthwhile even at <50% estimated success) was numerically inert --
    # at that scale it could never outweigh the dense per-step cost of the
    # steps spent attempting it, which is the same "hover/avoid-risk" shape
    # this fix was originally meant to kill (see tiltcutoff_v1-v4 sanity
    # checks: most episodes ran the full 60s clock with lost_control=False,
    # i.e. never committed to touchdown at all). Fixed in `reward_fn`: the
    # terminal bonus/penalty is now added AFTER `reward_scale` is applied to
    # the per-step shaping sum, so it keeps its full raw magnitude instead
    # of being shrunk 100x. Raised the three raw constants 3x on top of that
    # (preserving their existing ratios/breakeven math exactly) so the
    # terminal signal lands solidly within the observed per-episode shaping
    # range instead of merely not-vanishing -- a real landing or crash
    # should move the total return by roughly as much as a whole episode's
    # accumulated behavior does, not by a rounding error under it.
    landing_bonus_scale: float = 1800.0
    crash_penalty: float = 1500.0
    # applied on truncation (timeout) without ever having landed -- must
    # be strictly less than crash_penalty (timing out shouldn't be worse
    # than an actual crash) but strictly more than zero (it can't be free).
    timeout_penalty: float = 1050.0

    # defensive clip on the summed per-step SHAPING terms (excludes the
    # terminal reward, added after -- see reward_fn). This must stay a
    # numerical safety net, not a real constraint: if it binds regularly
    # during training it silently flattens the gradient right when it's
    # needed most (this happened for real -- see the comment on
    # `_velocity_tracking_penalty`, and the much bigger one on
    # `altitude_penalty_cap` above -- a single uncapped term blowing past
    # this WHOLE budget by itself, every step, for the entire high-altitude
    # portion of orbit_descent, is what that bug was). With every term now
    # individually capped, raised again alongside `braking_penalty_cap`'s
    # 100->250 raise (max plausible sum: ~230 altitude at the highest
    # release altitudes + 250 braking + ~112 velocity-tracking + a few
    # single-digit terms elsewhere, roughly ~600) so that raise doesn't
    # reopen the original swamping bug at merely-moderate violation levels.
    # Raised again alongside `braking_penalty_cap`'s 250->400 raise (see
    # its comment, itself downstream of `t0`'s 1.0->20.0 fix): new max
    # plausible sum ~230 altitude + 400 braking + ~112 velocity-tracking +
    # a few single-digit terms, roughly ~750 -- headroom kept above that.
    # Raised again alongside the new `tilt_cutoff_k`/`tilt_cutoff_cap`
    # barrier in `_attitude_hold_penalty` (see its comment): that term
    # alone can reach `tilt_cutoff_cap`=2000 right as the vehicle
    # approaches the loss-of-control cutoff -- deliberately large, since
    # it needs to dominate the sum right when that's the correct signal
    # (better to have one term's cap bind hard here than let the sum
    # silently flatten right at the most dangerous moment). New max
    # plausible sum ~230 altitude + 400 braking + ~112 velocity-tracking +
    # 2000 tilt-cutoff + a few single-digit terms, roughly ~2750.
    # Raised again alongside the reintroduced `_xy_position_penalty`
    # (gated, see `x1`'s field comment): adds up to `xy_position_penalty_
    # cap`=150 more. New max plausible sum ~2900.
    shaping_clip_abs: float = 3200.0

    # REAL BUG FOUND (this session, watching orbit_descent training after
    # the t0/cap fixes above): real episode returns were landing in the
    # -70,000 to -220,000 range (shaping alone allows +-900/step, up to
    # ~1200 steps in a 60s episode), while SB3's SAC defaults (3e-4 Adam
    # learning rate, standard-size MLP critic) are built around the
    # roughly-unit-scale returns most Gym benchmarks use. Symptom seen for
    # real: critic_loss bouncing noisily between ~2k and ~44k across a
    # whole run with no clear downward trend, and `ent_coef` staying
    # pinned around ~1.2 instead of annealing down as the policy should
    # gain confidence -- both consistent with the critic struggling to fit
    # high-magnitude, high-variance targets rather than a shaping-logic
    # problem. A uniform multiplicative rescale is policy-invariant in the
    # limit (SAC's auto entropy tuning targets a fixed entropy value
    # regardless of reward scale) but matters in practice for a FIXED
    # learning rate/architecture actually converging in a reasonable
    # number of updates.
    # REAL BUG FOUND (this session, see `landing_bonus_scale`'s field
    # comment for the full story): this used to be applied as the very
    # last step in `reward_fn`, AFTER the terminal bonus/penalty was added
    # -- shrinking the one-time terminal signal by the same factor as a
    # single per-step shaping term, even though it has to outweigh the SUM
    # of hundreds of those per-step terms over a whole episode. Now applied
    # ONLY to the per-step shaping sum; the terminal bonus/penalty is added
    # afterward at its own raw scale (see `landing_bonus_scale`/
    # `crash_penalty`/`timeout_penalty`, raised accordingly). Every
    # existing per-step term/cap/threshold comment above (about RELATIVE
    # weighting within the shaping sum) stays accurate.
    reward_scale: float = 0.01


def _proximity_penalty(w: RewardWeights, dist_m: float) -> float:
    return w.alpha * (1.0 / max(dist_m, w.dist_floor_m)) ** w.beta


def _altitude_penalty(w: RewardWeights, alt_m: float, braking_ratio: float = 0.0) -> float:
    # REAL BUG FOUND (this session, direct user observation of a trained
    # checkpoint "just crashing straight down" + independent hand-derived
    # numbers): this term and `_braking_envelope_penalty` used to be two
    # purely ADDITIVE, non-interacting terms -- nothing ever suppressed
    # the "get closer to the ground" incentive even when vz was already
    # dangerously high for the current altitude, so a policy could always
    # find it locally cheaper to keep closing altitude fast (this term
    # shrinking) than to spend effort braking (that term growing), right
    # up until the braking term's cap made it not worth it -- i.e. exactly
    # the "prefers to just crash" behavior observed. Gated by
    # `braking_ratio` (the SAME real required-vs-available-deceleration
    # number `_braking_envelope_penalty` computes -- passed in by
    # `reward_fn` so it's only computed once): at ratio=0 (plenty of
    # stopping margin) the full "descend" incentive applies unchanged; as
    # ratio grows toward and past 1 (current vz is eating into/exceeding
    # the safe stopping budget for this altitude) the descend incentive
    # fades out smoothly (1/(1+k*ratio), continuous and differentiable
    # everywhere, no new kink), so the only thing left pulling on the
    # policy in that regime is "brake," not a competing "get lower" signal
    # actively fighting it.
    raw = w.kappa_alt_sqrt * np.sqrt(max(alt_m, 0.0))
    gate = 1.0 / (1.0 + w.altitude_vz_gate_k * max(braking_ratio, 0.0))
    return float(raw * gate)


def _braking_ratio(w: RewardWeights, alt_m: float, v_z_m_s: float, mass_kg: float,
                    dps_thrust_max_n: float, gravity_m_s2: float) -> float:
    """Real required-vs-available deceleration ratio -- see
    `_braking_envelope_penalty`'s docstring for the physical derivation.
    Factored out so `_altitude_penalty`'s vz-gate (see its comment) and
    `_braking_envelope_penalty`'s barrier read the exact same number
    instead of two independently-computed copies.
    """
    a_avail = max(dps_thrust_max_n / max(mass_kg, 1.0) - gravity_m_s2, 0.05)
    alt_floored = max(alt_m, w.braking_alt_floor_m)
    a_required = (v_z_m_s ** 2) / (2.0 * alt_floored)
    ratio = a_required / (w.braking_authority_margin * a_avail)
    return min(ratio, 1000.0)  # guard pow() for pathological states, not a shaping knob


def _braking_envelope_penalty(w: RewardWeights, ratio_clamped: float) -> float:
    """REDESIGNED this session (see the field comment on `t0` for the full
    history of the OLD `v_limit(alt) = safe_v_z + k*alt**zeta` version this
    replaces). That version tied "how urgent is braking" to an arbitrary
    altitude power-law schedule, disconnected from the vehicle's REAL
    deceleration authority -- confirmed via BOTH a real trained checkpoint's
    Isaac Sim eval THIS session (throttle dropped to ~0.01 for 9+ seconds,
    128m -> 15m, essentially free-fall, before panicking at 1.8m altitude --
    the exact "coast then panic" pattern the old formula's bug history
    already documented) and by direct comparison against
    `ZemZevController` (the hand-designed, non-RL guidance law that DOES
    solve this scenario, 75%+ landed_safely -- see `zemzev_controller.py`):
    that controller never uses a v_limit(alt) schedule at all -- it reasons
    directly about REQUIRED deceleration (kinematic stopping distance,
    `vz**2 / (2*alt)`) against the vehicle's ACTUAL available deceleration
    (`dps_thrust_max_n/mass - gravity`), exactly matching the ZEM/ZEV
    guidance family's own "time/distance-to-go" logic.

    This term now measures the SAME physical quantity: `ratio` is how much
    of the vehicle's real stopping-distance budget is already used up RIGHT
    NOW, given current vz and altitude. Unlike the old altitude-power-law
    schedule (arbitrary, same shape regardless of vehicle mass or thrust),
    this is grounded in the same real numbers `ApolloLMSpecs`/`LanderParams`
    already use everywhere else, and it naturally accounts for mass burning
    off over the flight (a lighter vehicle late in descent has more
    deceleration authority, exactly as the real vehicle would). `ratio`
    (real required-vs-available deceleration, signed-vz-squared so fast
    ASCENT counts too -- see `_braking_ratio`) is computed once by
    `reward_fn` and shared with `_altitude_penalty`'s vz-gate, instead of
    two independently-computed copies.

    REAL BUG FOUND (a real trained checkpoint's Isaac Sim eval, this
    session, after the redesign above): `exp(ratio**theta)` blows past ANY
    reasonable cap within a tiny range of ratio (~2.0-2.5) -- a failed
    landing was observed with `ratio` climbing from 3.5 (at 47m) to 10,
    21, and 75 (at touchdown), but the exp shape made every one of those
    points read as the identical maxed-out penalty. The policy had a
    correctly strong "you are in trouble" signal the whole way down, but
    literally zero gradient telling it whether braking harder was helping
    AT ALL once already past ratio~2.5 -- exactly the region a still-
    learning policy needs the most differentiation in. Switched to a plain
    polynomial (`ratio**theta`, no exp): still zero at ratio=0, still 1 at
    ratio=1 (barrier "starts mattering" at the same point), but grows far
    more gently, so `braking_penalty_cap` isn't reached until a much
    higher ratio -- stretching the differentiating range from ~0-2.5 to
    ~0-16 at theta=2, cap=250 (`sqrt(250)~=15.8`). Numerically safer too: a
    polynomial can't overflow to inf/nan for extreme ratios the way an
    uncapped `exp()` argument could.
    """
    return min(w.t0 * float(ratio_clamped ** w.theta), w.braking_penalty_cap)


def _leg_load_penalty(w: RewardWeights, leg_force_n: float, leg_force_max_n: float) -> float:
    if leg_force_n <= 0.0:
        return 0.0
    return w.h * (leg_force_n / max(leg_force_max_n, 1e-6)) ** w.c


def _rcs_l1_penalty(w: RewardWeights, pitch_cmd: float, roll_cmd: float, yaw_cmd: float) -> float:
    return w.lam * (abs(pitch_cmd) + abs(roll_cmd) + abs(yaw_cmd))


def _rcs_l2_penalty(w: RewardWeights, pitch_cmd: float, roll_cmd: float, yaw_cmd: float) -> float:
    return w.gamma * (pitch_cmd ** 2 + roll_cmd ** 2 + yaw_cmd ** 2)


def _attitude_hold_penalty(w: RewardWeights, tilt_x: float, tilt_y: float, v_xy: float,
                            loss_of_control_tilt_rad: float) -> float:
    # velocity-gated: tilt is cheap while v_xy still justifies it (braking
    # needs tilt), and ramps up to full cost as v_xy -> 0 (no excuse left
    # to still be tilted) -- see the mu_base/mu_scaled field comments.
    urgency = float(np.clip(1.0 - v_xy / max(w.attitude_hold_v_xy_ref_m_s, 1e-6), 0.0, 1.0))
    effective_mu = w.mu_base + w.mu_scaled * urgency
    quad = effective_mu * (tilt_x ** 2 + tilt_y ** 2)

    # REAL BUG FOUND: see `tilt_cutoff_k`'s field comment. Velocity-
    # INDEPENDENT (unlike `quad` above) -- this must stay active even at
    # high v_xy, since that's exactly when the velocity gate makes `quad`
    # weak and the policy has the least other reason to avoid the cutoff.
    tilt = float(np.hypot(tilt_x, tilt_y))
    ratio = tilt / max(loss_of_control_tilt_rad, 1e-6)
    barrier = min(w.tilt_cutoff_k * float(ratio ** w.tilt_cutoff_power), w.tilt_cutoff_cap)
    return quad + barrier


def _velocity_tracking_penalty(w: RewardWeights, v_xy: float) -> float:
    # REAL BUG, FOUND VIA A FAILED TRAINING RUN (position terms since
    # removed, see field comment): this term used to include a plain,
    # uncapped-by-itself quadratic pull toward the exact target position.
    # Once a policy drifts even a few hundred meters off target (which an
    # undertrained policy will, regularly), that swamped every other term
    # AND used to swamp the shared reward_clip_abs -- so most of a bad
    # trajectory read back as a flat, saturated penalty with zero gradient
    # telling the agent that further drift is even worse. The position
    # terms are gone now (they actively fought the correct horizontal
    # braking maneuver -- see field comment), but the cap on what remains
    # (the velocity term) stays, for the same "don't let one term swamp the
    # budget" reason.
    raw = w.x5 * abs(v_xy) ** w.x6
    return min(raw, w.position_penalty_cap)


def _xy_position_penalty(w: RewardWeights, dist_xy_m: float, alt_m: float) -> float:
    """Gated pull toward (target_x, target_y) -- see `x1`'s field comment
    for why this is back and how it differs from the flat version removed
    earlier. Fades linearly to zero as altitude drops below
    `position_alt_gate_m`, so it can't still be pulling/inducing tilt
    during the final touchdown maneuver the way the old always-on version
    did.
    """
    gate = float(np.clip(alt_m / max(w.position_alt_gate_m, 1e-6), 0.0, 1.0))
    raw = w.x1 * abs(dist_xy_m) ** w.x2
    return min(gate * raw, w.xy_position_penalty_cap)


def _terminal_reward(w: RewardWeights, info: dict) -> float:
    margins = info.get("landing_margins", {})
    if info.get("landed_safely"):
        # graded by margin under each threshold -- a feather-soft, dead-centered
        # landing scores near landing_bonus_scale; a marginal one scores less.
        quality = float(np.clip(np.mean(list(margins.values())), 0.0, 1.0)) if margins else 1.0
        return w.landing_bonus_scale * (0.5 + 0.5 * quality)

    # REAL BUG, FOUND VIA A FULL TRAINING RUN: a genuine airborne tumble
    # crash (lost_control, tilt past the cutoff) and a touchdown that missed
    # exactly ONE threshold (e.g. a real orbit_descent eval: vz=0.40 m/s and
    # tilt=12.4 deg both comfortably inside their limits, only vxy=3.73 m/s
    # over its 0.5 m/s bar) used to score IDENTICALLY: -crash_penalty either
    # way. `landing_margins` is only ever populated when the vehicle
    # actually touched down without tripping `lost_control` (see
    # IsaacLanderEnv.step's `if touched_down and not lost_control:` guard),
    # so its presence/absence here is exactly the "near miss vs. real crash"
    # distinction needed. Grading the near-miss case by how far under the
    # OTHER thresholds it still was restores a gradient toward "brake more"
    # instead of every miss reading as an equally-bad catastrophe -- while
    # keeping it strictly worse than any actual safe landing (worst safe
    # landing: landing_bonus_scale*0.5; best near-miss here: -crash_penalty*0.5).
    if margins:
        quality = float(np.clip(np.mean(list(margins.values())), 0.0, 1.0))
        return -w.crash_penalty * (1.0 - 0.5 * quality)
    return -w.crash_penalty


def make_apollo_reward_fn(weights: RewardWeights | None = None, specs: ApolloLMSpecs | None = None):
    """Build a reward_fn for `AnalyticLanderEnv(reward_fn=...)`. `weights`
    defaults to `RewardWeights()`; `specs` (real vehicle numbers) defaults
    to `ApolloLMSpecs()` and is only used for the leg-force normalization.
    """
    w = weights or RewardWeights()
    specs = specs or ApolloLMSpecs()

    def reward_fn(env, info: dict) -> float:
        p = env.params
        s = env.state

        dist_m = float(np.hypot(s["x"] - p.target_x, s["y"] - p.target_y))
        v_xy = float(np.hypot(s["vx"], s["vy"]))

        mass_kg = info.get("mass_kg", p.dry_mass_kg)
        braking_ratio = _braking_ratio(
            w, info["altitude_m"], s["vz"], mass_kg, p.dps_thrust_max_n, p.gravity_m_s2)

        r = 0.0
        r -= _proximity_penalty(w, dist_m)
        r -= _altitude_penalty(w, info["altitude_m"], braking_ratio)
        r -= _braking_envelope_penalty(w, braking_ratio)
        r -= _leg_load_penalty(w, info.get("leg_force_n", 0.0), info.get("leg_force_max_n", 1.0))
        r -= _throttle_effort_penalty(w, s["throttle"])
        r -= _rcs_l1_penalty(w, s["rcs_pitch"], s["rcs_roll"], s["rcs_yaw"])
        r -= _rcs_l2_penalty(w, s["rcs_pitch"], s["rcs_roll"], s["rcs_yaw"])
        r -= _attitude_hold_penalty(w, s["tilt_x"], s["tilt_y"], v_xy, p.loss_of_control_tilt_rad)
        r -= _velocity_tracking_penalty(w, v_xy)
        r -= _xy_position_penalty(w, dist_m, info["altitude_m"])

        # `reward_scale` applies ONLY to the per-step shaping sum (see its
        # field comment) -- the terminal bonus/penalty below is added at its
        # own raw scale so it isn't shrunk 100x relative to a whole
        # episode's accumulated shaping. Never clipped away by the shaping
        # cap either way.
        r = float(np.clip(r, -w.shaping_clip_abs, w.shaping_clip_abs)) * w.reward_scale
        if info.get("terminated"):
            r += _terminal_reward(w, info)
        elif info.get("truncated"):
            # ADDED after a real training run found "hover forever" was a
            # free way to dodge crash risk entirely (39/40 episodes timed
            # out airborne, several already at safe vz, never committing
            # to touchdown) once timeout carried zero penalty -- see the
            # crash_penalty/landing_bonus_scale comment above.
            r -= w.timeout_penalty
        return r

    return reward_fn


def _throttle_effort_penalty(w: RewardWeights, throttle: float) -> float:
    return w.s * throttle ** 2


default_reward_fn = make_apollo_reward_fn()
