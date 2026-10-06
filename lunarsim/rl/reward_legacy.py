"""ARCHIVE of the pre-2026-10-05 reward. NOT IMPORTED BY ANYTHING.

Kept only for the measurements and bug histories written into its field
comments -- a dozen real defects were found and documented in place here,
and handover.md cites them. The reward itself was deleted and rewritten
(see lunarsim/rl/reward.py) because its core term penalised ALTITUDE,
which is a time x altitude integral: measured corr(episode length, total
shaping) = -0.927, i.e. it scored duration, not quality, and so could not
tell a controlled descent from a dive -- the dive scores BETTER because it
reduces altitude faster. Per-term decomposition of a real winning flight
at the end of its life: altitude 86.10, vel_track 4.89, braking_xy 0.71,
every other term <= 0.02, against a 450 landing bonus.
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
    # ZEROED 2026-10-05 as measured dead weight. Per-term decomposition of
    # the winning controller flight (scaled, undiscounted, total over the
    # whole 94 s): altitude 86.10, vel_track 4.89, braking_xy 0.71, and
    # EVERY other term <= 0.02 against a 450 landing bonus. A term worth
    # 0.02 over an entire flight cannot teach anything; it only adds
    # variance for the critic to explain and another knob to mis-tune.
    # The criteria these covered are all still enforced where they
    # actually bite: tilt at touchdown through the terminal severity
    # grading (see `crash_severity_base`), and tilt in flight through
    # the `tilt_cutoff` barrier, which is 1000x larger than the mu
    # quadratic at any tilt where it matters. `_angular_rate_penalty`
    # was zeroed too and then RESTORED: the audit measured it as
    # non-discriminating, but it measured that on 48 demo episodes
    # flown by a controller whose |w| never leaves 0.003 -- a
    # population where nothing spins cannot show whether a spin
    # penalty matters, and this file already records |w| as the
    # BINDING margin on two of the three closest touchdowns ever
    # measured. Kept as zeroed weights
    # rather than deleted code so the measurements and bug history above
    # them stay readable.
    alpha: float = 0.0
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
    # REAL BUG FOUND (this session, direct telemetry diagnostic on a
    # trained checkpoint after `reward_scale` was cut 4x below: 6/16
    # episodes reached t_s=60 TIMEOUT at altitudes of 680-1372m with
    # vz=+28..+50 m/s -- i.e. the policy was actively ROCKETING AWAY under
    # full throttle for the entire episode, not passively hovering).
    # `reward_scale` scales every per-step term uniformly, including this
    # one -- cutting it 4x (see that field's comment) silently cut this
    # term's absolute deterrent strength 4x too, without anyone touching
    # `kappa_alt_sqrt` itself to compensate. At the old scale, climbing to
    # ~330m already cost real money (see this field's own bug history);
    # at the new scale that same climb became 4x cheaper, reopening almost
    # exactly the "climb away to dodge risk entirely" exploit this field
    # was created to close in the first place. Raised 4x (12.66 -> 50.64)
    # to restore the ORIGINAL absolute per-step cost this term had before
    # `reward_scale` moved, while everything else keeps the benefit of the
    # smaller overall scale.
    # HALVED AGAIN (this session, 50.64 -> 25.32, after the 4x raise above
    # was measured against the behavior it was meant to stop). The 4x raise
    # was justified as "restore the pre-reward_scale absolute deterrent", but
    # a direct sweep showed the thing actually neutralizing this term in the
    # climb-away regime was NOT `reward_scale` -- it was `altitude_vz_gate_k`
    # opening on ASCENT as readily as on descent (see that field's new bug
    # comment and `_altitude_penalty`'s). Measured: at alt=1000 m, vz=+50 m/s
    # the gate was handing back a 4.67x discount, i.e. cancelling the entire
    # 4x raise exactly where it was supposed to bite (net effect of the raise
    # on the targeted exploit: ~1.15x). With the gate fixed to read descent
    # speed only, the deterrent comes back on its own and the 4x raise sits
    # on top of it as pure over-weighting: at 4x this term was 42.7% of the
    # braking-phase shaping sum (12.66 was 15.7%) and 41-57% of the raw sum
    # across four real Isaac ZemZev landing episodes -- quietly undoing the
    # `t0` 1.0->20.0 fix, whose whole point was making the braking envelope
    # COMPARABLE to this term rather than 34x weaker. 2x (25.32) puts the
    # braking-phase share at 27.0%, back alongside `brake_z`/`brake_xy`
    # instead of above them, while keeping a strong climb deterrent
    # (gate-fixed, alt=1000/vz=+50 costs 800.7 vs. the old gated 85.7).
    kappa_alt_sqrt: float = 25.32
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
    # REAL BUG FOUND (this session, direct arithmetic on the shipped code,
    # then confirmed by a 4-strategy discounted-return sweep): `_braking_
    # ratio` squares vz, so this gate is BLIND TO THE SIGN of vertical speed
    # -- a fast CLIMB opened it exactly as wide as a fast descent, switching
    # off the "get lower" incentive precisely in the regime where the
    # vehicle is running away from the ground. Measured at alt=100 m:
    #     vz=-10 -> ratio 1.4682, gate 0.405, altitude penalty 205.17
    #     vz=  0 -> ratio 0.0000, gate 1.000, altitude penalty 506.40
    #     vz=+10 -> ratio 1.4682, gate 0.405, altitude penalty 205.17
    # i.e. climbing at 10 m/s cost EXACTLY what descending at 10 m/s cost,
    # and both were less than half the cost of simply holding altitude --
    # every other vz-dependent term is a function of vz**2 too, so the whole
    # instantaneous shaping sum was an even function of vz. That makes
    # "rocket away under full throttle" strictly cheaper than hovering, which
    # is the literal behavior the telemetry diagnostic recorded (6/16 episodes
    # timing out at 680-1372 m with vz=+28..+50 m/s). The gate is now fed
    # min(vz, 0.0) in `reward_fn`, so only genuine DESCENT speed relaxes the
    # descend incentive; `_braking_envelope_penalty` keeps the sign-blind
    # ratio on purpose (a fast climb SHOULD still pay the envelope term).
    # Descent-side behavior is bit-for-bit unchanged (alt=100/vz=-10 still
    # 205.17), so the original "prefers to crash straight down" fix this
    # gate exists for is untouched.
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
    # MEASURED 2026-10-04 on all 48 real orbit_descent demo episodes (20
    # landed / 28 failed): `_braking_envelope_penalty` is ANTI-correlated
    # with success -- successful landings pay 1.40x what failures pay
    # (23121 vs 16527 raw). Per altitude band, cost/step landed vs failed:
    # 60-120 m 36.3/27.0, 30-60 m 69.9/42.8, 10-30 m 97.6/56.4. Cause: a
    # successful descent is FASTER through the mid-band (mean vz -5.24 vs
    # -4.52 at 10-30 m) and `ratio = vz**2/(2*alt)` charges exactly that.
    # It cannot be retuned away -- sweeping this margin 0.25->0.85 and `t0`
    # 20->8 leaves the landed/failed ratio at 1.40 throughout, because any
    # monotone transform hits both groups equally. It can only be silenced
    # in the regime we know is correct. At 0.25 the proven winning flight
    # peaks at ratio 2.35, i.e. the barrier is FULLY ON during a textbook
    # landing; at 0.70 it peaks at 0.96, just under the knee, cutting the
    # term 87% (23121 -> 2949) while leaving it free to fire on a genuinely
    # unrecoverable descent. The term keeps its job (catch a vehicle that
    # has spent its real stopping budget) and stops taxing the profile we
    # are trying to teach.
    braking_authority_margin: float = 0.70
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
    s: float = 0.0

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
    mu_base: float = 0.0
    mu_scaled: float = 0.0
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
    # TRIED THIS SESSION, THEN REVERTED: user direction after comparing
    # real sanity-check episode totals (`lost_control` crash episodes
    # scoring WORSE than full-60s timeout episodes, e.g. -5841..-9297 vs
    # -4198..-6586) led to cutting this term 5x (2000 -> 400), on top of
    # the SAME session's separate `reward_scale` cut (0.01 -> 0.0025, see
    # that field). Those two cuts COMPOUNDED to a 20x reduction in this
    # term's actual felt magnitude (59 deg: ~17.4/step at the old scale ->
    # ~0.87/step at both cuts stacked) -- confirmed by a direct telemetry
    # diagnostic on the resulting checkpoint: 9/16 episodes lost control,
    # most showing pitch/roll RCS commands pinned near +-1.0 for dozens of
    # consecutive steps as tilt climbed straight through 55-60 deg with
    # essentially no pushback, instead of the barrier intervening before
    # the cutoff the way it did when this term was first added (see the
    # bug history above this one). The 5x cut over-corrected: `reward_
    # scale`'s OWN 4x cut already brings this term's per-step contribution
    # down into the same ballpark as the other per-step caps during a
    # tumble (a losing-control trajectory take many tens of steps to climb
    # from ~45 to 60 deg based on real telemetry, not the few steps
    # originally assumed, so the cumulative extra cost over a full tumble
    # stays modest even without an additional cut) -- stacking a second,
    # separate 5x cut on top gutted the one term whose entire job is to be
    # strong enough to actually stop the tumble before it reaches the
    # cutoff. Reverted to the original k/cap; `reward_scale` alone handles
    # bringing this term's absolute size down along with everything else.
    tilt_cutoff_k: float = 2000.0
    tilt_cutoff_power: float = 8.0
    tilt_cutoff_cap: float = 2000.0

    # REAL BUG FOUND (this session, a measured 16-episode Isaac Sim eval of
    # the first ramp_20m checkpoint trained under the fixed reward): the
    # `landed_safely` test requires |w| <= `safe_landing_w_rad_s` = 0.5
    # rad/s, but NOTHING in the per-step shaping sum was a function of
    # angular RATE at all -- `_attitude_hold_penalty` reads tilt_x/tilt_y
    # (angles, and not yaw), never wx/wy/wz. The only rate-adjacent terms
    # were the RCS effort penalties, and those are numerically invisible:
    # `lam`=0.02 and `gamma`=0.01 mean all three axes saturated at once
    # costs 0.02*3 + 0.01*3 = 0.09 raw = 0.000225 after `reward_scale`,
    # about 1/5600 of the altitude term. So the agent paid essentially
    # nothing for spinning, right up until a hard terminal rejection.
    #
    # The measured consequence, from that eval's per-episode margins:
    #     [ep 15] CRASH  vz=-0.21  vxy=0.48  tilt=7.1  w=0.65
    #             margins={v_z=0.79, v_xy=0.60, tilt=0.53, w=0.00} worst=w
    # -- a touchdown with every OTHER criterion comfortably inside its
    # limit, rejected solely on angular rate. Of the three touchdowns that
    # came closest, two had w as their binding margin (ep 15 at 0.00, the
    # successful ep 8 at 0.38). It is the one landing criterion the reward
    # never mentioned.
    #
    # Shaped the same way the tilt cutoff is (that pattern took lost_control
    # from 16/16 to 1-6/16 in an earlier session, see `tilt_cutoff_k`): a
    # gentle always-on quadratic for a global gradient, plus a steep,
    # velocity-independent barrier that only wakes up approaching the real
    # cutoff. Sized against the measured rate distribution -- ordinary
    # flight in that eval ran w=0.01-0.07, a braking tilt maneuver needs
    # ~0.3 rad/s, and 0.5 is the rejection line:
    #     |w|=0.1 -> 0.6 quad +   0.04 barrier =   0.6 raw  (negligible)
    #     |w|=0.3 -> 5.4 quad +  28.0 barrier =  33.4 raw  (modest)
    #     |w|=0.5 -> 15.0 quad + 600.0 barrier = 615.0 raw  (a real wall)
    # so a normal braking rotation is close to free and sitting on the
    # rejection line is not. Smaller than the tilt barrier's 2000 on
    # purpose: tilt past 60 deg ends the episode, |w| past 0.5 only costs
    # the landing, and only if the vehicle touches down in that state.
    omega_k: float = 60.0
    omega_cutoff_k: float = 600.0
    omega_cutoff_power: float = 6.0
    omega_cutoff_cap: float = 600.0
    # caps the WHOLE term (quadratic included), the same way
    # `_velocity_tracking_penalty`/`_xy_position_penalty` cap theirs. Caught
    # by a regression test: with only the barrier capped, a full tumble
    # (|w| ~ 17 rad/s) drove the uncapped quadratic to 18600 raw on its own
    # -- four times the entire `shaping_clip_abs` budget, from one term, in
    # a state the tilt cutoff is already handling. 800 = the 600 barrier
    # plus up to 200 of quadratic, which saturates at |w| = 1.83 rad/s;
    # past that the vehicle is tumbling, not landing.
    omega_penalty_cap: float = 800.0

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

    # ADDED THIS SESSION, horizontal analog of `_braking_envelope_penalty`
    # below (see that function's docstring for the real-stopping-distance
    # design this mirrors). REAL GAP FOUND via a direct telemetry
    # diagnostic on a trained checkpoint: 9/16 episodes reached the ground
    # for real (a genuine first -- earlier checkpoints almost never got
    # that far), but EVERY one of them touched down at 15-32 m/s
    # horizontal speed (safe limit ~1.2 m/s) while vertical speed was
    # often reasonable (one case: vz=-3.15 m/s, well-controlled) and tilt
    # moderate (13-53 deg, mostly under the loss-of-control range) -- i.e.
    # the vehicle had learned real vertical/attitude control but never
    # learned to kill horizontal speed before contact. Root cause: `x5`
    # above is a FLAT quadratic on vxy, the same cost whether the vehicle
    # is at 200m (plenty of time left) or 2m (none) -- unlike the vertical
    # channel, which has `_braking_envelope_penalty`'s real time-to-ground
    # urgency built in via `_braking_ratio`. This term gives horizontal
    # speed the same "how much of your real stopping budget is already
    # spent" urgency signal, using the same required-vs-available-
    # deceleration physics, just substituting an estimated time-to-ground
    # (`_time_to_ground_s`) for the vertical channel's direct kinematic
    # alt/vz relationship (horizontal motion has no vertical "stopping
    # distance" of its own -- time-to-go is the shared resource both
    # channels compete for). Defaults mirror `t0`/`theta`/
    # `braking_penalty_cap` exactly (same vehicle, same authority margin,
    # no reason to assume a different shape a priori) -- UNTUNED, a first
    # reasonable attempt to be refined the same way the vertical one was:
    # by watching real training and checking real telemetry, not guessed
    # in isolation.
    # CALIBRATION, see the note above `landing_bonus_scale`: 20.0 was an
    # explicitly UNTUNED guess (mirrored off the vertical `t0` for want
    # of a better prior). Measured against the winning controller run it
    # charged 559 units, 26.9% of that flight's whole shaping bill -- for
    # executing the reference-correct horizontal braking profile (18.6 ->
    # 0 m/s bled smoothly over 90 s). A term meant to penalise LEAVING
    # horizontal speed unbraked must not be the third-largest cost of
    # braking it correctly. Cut 10x, which puts the demonstrated profile
    # at ~56 units while keeping the shape (a late, still-fast approach
    # still saturates `xy_braking_penalty_cap`).
    t0_xy: float = 2.0
    xy_braking_theta: float = 2.0
    xy_braking_penalty_cap: float = 400.0
    # floors the time-to-ground estimate `_time_to_ground_s` divides by --
    # a divide-by-zero/blowup guard near touchdown (same role as
    # `braking_alt_floor_m` plays for the vertical channel), not a
    # tunable shaping knob.
    xy_braking_tgo_floor_s: float = 2.0

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
    # CALIBRATION, see the note above `landing_bonus_scale`: this term
    # charged 660 of the 2076 shaping units the winning controller run
    # paid (31.8%), because that flight touched down 613 m from
    # `target_x/target_y` -- and `landed_safely` has five criteria,
    # none of which is distance to the target (handover: "a controller
    # that lands beautifully in the wrong place scores identically").
    # So this was the single largest penalty on a flight the success
    # test calls perfect, levied for something that test never reads.
    # Set to 0. If target accuracy is supposed to matter, the fix is a
    # sixth `landed_safely` term, not a shaping penalty that punishes
    # what the criterion rewards.
    x1: float = 0.0
    x2: float = 2.0
    position_alt_gate_m: float = 40.0
    xy_position_penalty_cap: float = 150.0

    # REAL BUG FOUND (this session, user direction after BOTH a warm-started
    # continuation AND a from-scratch run of orbit_descent, under the
    # now-correctly-scaled terminal reward below, still showed 0/16
    # `landed_safely` with 11-14/16 episodes running the FULL 60s clock
    # without ever attempting touchdown): every other per-step term here
    # (`_altitude_penalty`, `_braking_envelope_penalty`, ...) costs
    # something about the CURRENT state (altitude, speed, tilt), but
    # nothing costs TIME itself -- a policy that finds some locally cheap
    # holding pattern (this vehicle's thrust-to-weight is close to 1, so
    # near-hover is a natural, low-effort attractor under a not-yet-
    # confident policy) pays only that pattern's own small per-step cost,
    # repeated 1200 times, with no additional pressure to actually commit
    # to descending. `timeout_penalty` below already charges for this, but
    # only ONCE, at the very end -- it does nothing to make step 200 of
    # stalling look any worse than step 1. This term is the opposite: a
    # flat, unconditional per-step tax charged every step the episode is
    # still running and hasn't landed (not gated by altitude or speed, so
    # it can't be dodged by finding a cheap place to loiter) -- it makes
    # the "just survive" strategy's total cost grow LINEARLY with how long
    # it keeps stalling, instead of staying flat. Sized so that going the
    # full 60s (1200 steps) alone adds about as much as a real crash's
    # terminal penalty (1200 * 1.25 = 1500 raw ~= crash_penalty), i.e.
    # "wait out the whole clock" should cost at least as much as "attempt
    # and fail outright" -- removing the free-riding timeout option
    # `timeout_penalty` alone wasn't enough to kill.
    # DISABLED (set to 0.0) THIS SESSION, after the arithmetic above was
    # checked against the shipped code -- it was never delivering anything
    # close to what this comment claims, and once it DID, it pulled against
    # a stated user requirement. Two separate findings:
    #
    # (1) SCALE ERROR, 320x. The sizing argument ("1200 * 1.25 = 1500 raw
    #     ~= crash_penalty") forgets that this term is summed into `r`,
    #     which `reward_fn` then multiplies by `reward_scale`=0.0025, while
    #     `crash_penalty` is added AFTER that multiply and is never scaled.
    #     Actually delivered over a full 60 s episode: 1200 * 1.25 * 0.0025
    #     = 3.75 return units against a 1200-unit crash penalty, i.e. 0.3%
    #     of the intended pressure. Measured share of the real per-episode
    #     shaping sum on four real Isaac ZemZev episodes: 0.1-0.3%. It has
    #     been a no-op since the day it was added.
    # (2) EVEN CORRECTLY SIZED, IT IS THE WRONG SIGN FOR THE STATED GOAL.
    #     Re-applied OUTSIDE `reward_scale` (the intended semantics) and
    #     swept at gamma=0.999 over four strategy archetypes, a flat
    #     per-step tax buys a little land-vs-stall margin by destroying
    #     crash-vs-stall margin -- it charges the LONG episode most, and a
    #     crash is the SHORTEST episode there is:
    #         tax=0.00/step: land-stall=1046, stall-crash= 605
    #         tax=0.25/step: land-stall=1095, stall-crash= 498
    #         tax=0.50/step: land-stall=1144, stall-crash= 390
    #         tax=1.00/step: land-stall=1241, stall-crash= 176
    #     The user's explicit standing requirement is "crashing must stay
    #     worse than hovering"; this term erodes exactly that ordering.
    #
    # The premise behind it ("nothing costs TIME itself") was also wrong:
    # `_altitude_penalty` already charges 1.266 scaled units/step at 100 m,
    # i.e. 1519 over a full clock -- 405x this term, and unlike a flat tax
    # it is at least informative about WHERE the vehicle is loitering. The
    # real reason "stall forever" was winning is the discount horizon, not
    # a missing time cost (see `reward_scale`'s new note and the gamma fix
    # in scripts/train_sac_isaac.py). Kept as a live, zero-valued field
    # rather than deleted so the call site and this history stay visible.
    alive_tax_per_step: float = 0.0

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

    # CALIBRATED AGAINST THE WINNING CONTROLLER RUN (2026-10-02). Method:
    # replay `out/eval_snapshots/hero_fixed/telemetry.csv` -- a real Isaac
    # Sim ZemZev landing with `landed_safely=True` (vz=-0.80, vxy=0.08,
    # tilt=3.3, w=0.014, leg_diff=0.006) -- through this reward term by
    # term, discount at the shipped gamma=0.999, and compare against
    # stall/crash archetypes. Before this change:
    #
    #   controller landing (95 s)  -895.6     <- the PROVEN-correct flight
    #   stall @100 m, timeout      -759.0
    #   stall @ 20 m, timeout      -514.5     <- optimal
    #   free-fall crash           -1328.8
    #
    # Stalling at 20 m beat a perfect landing by 381 units, which is
    # exactly the behaviour every measured run produced (the ramp_20m
    # probe hung 10/16 episodes at 20-25 m). The policy was not failing to
    # learn; it was learning what this file told it. Three terms carried
    # the misalignment -- see `x1`, `t0_xy` and `landing_bonus_scale`.
    # After: landing +145.8, best stall -514.5, crash -1189.5, i.e. the
    # proven flight is the best outcome by 660 units and crash stays worst.
    # Re-run that comparison before touching any of the three.
    # Raised 1800 -> 4500. At gamma=0.999 a 1891-step episode discounts
    # the terminal by 0.999**1891 = 0.151, so 1800 arrives as 272 return
    # units at the release point -- less than the shaping the flight
    # accumulates on the way. The two term fixes above are corrections
    # of outright misalignments; this one buys the margin that makes
    # landing win outright rather than merely tie. Breakeven attempt
    # probability against accepting a timeout is now 2.6%
    # (p*4500 - (1-p)*1200 = -1050), which is the right side to err on
    # for a policy that has never once landed.

    # DIVERGENCE FIX (2026-10-02, measured on run `out/train_cal_v1.log`):
    # the calibration above fixed WHICH strategy wins but left the absolute
    # value scale too large for SAC's entropy temperature to hold against.
    # Measured over 268k steps of that run:
    #
    #     step    critic_loss   ent_coef
    #    11696       7.15e+03      0.635
    #   132496       7.69e+04       2.08
    #   266592       3.03e+06       6.61    <- 400x, still climbing
    #
    # Mechanism: SAC's alpha loss reads only the policy's entropy against
    # `target_entropy` = -dim(A) = -4, so reward scale does not drive alpha
    # directly -- but a large reward scale makes the Q gradient dominate the
    # entropy term, the policy collapses toward deterministic, entropy falls
    # under target, and alpha climbs to push back. Alpha then re-enters the
    # critic target (`Q = r + gamma*(Q' - alpha*log_pi)`), and at gamma=0.999
    # that inflation compounds over a ~1000-step horizon, which inflates Q,
    # which inflates the gradient: a feedback loop, not a fitting problem.
    #
    # Fix: divide the ENTIRE reward -- shaping (`reward_scale`) and all three
    # terminals -- by the same 10. A uniform positive scale leaves the
    # optimal policy mathematically unchanged, so the whole calibration table
    # above (landing > stall > crash, verified again after this change) holds
    # bit for bit; only the magnitudes the critic has to fit move. Episode
    # returns now land in the tens, which is the range SB3's defaults (3e-4
    # Adam, standard MLP critic) are built around.
    # KEEP THE FOUR IN STEP: `reward_scale`, `landing_bonus_scale`,
    # `crash_penalty` and `timeout_penalty` are one scale. Changing one alone
    # changes the strategy ordering; changing all four by a common factor
    # does not.
    # Crash severity grading (2026-10-05). MEASURED defect it replaces: the
    # penalty was `-crash_penalty * (1 - 0.5*mean(margins))`, and `margins`
    # are each clipped to [0,1], so they cannot express HOW FAR over a limit
    # a touchdown was. A measured free-fall crash (vz=-7.60, i.e. 7.6x its
    # limit) came out at margins {v_z=0.00, v_xy=0.00, tilt=0.00, w=0.92,
    # leg_diff=0.40}: the two criteria that are trivially satisfied while
    # falling straight down (you are not rotating, and the ground under you
    # is as flat as it happens to be) pulled the MEAN up to 0.26 and the
    # penalty to -104.5 -- against -105.0 for hovering until timeout. So
    # flying into the ground at 7.6x the limit cost the same as doing
    # nothing, and the whole spread from "20% too fast" to "free fall" was
    # 19.5 units. Nothing said "slow down".
    #
    # Now graded by OVERSHOOT: `severity` = max over the criteria of
    # value/limit (>= 1 for any failure), and the multiplier is
    # `crash_severity_base + crash_severity_k*(severity-1)`, capped. Taking
    # the MAX, not the mean, is the point -- one catastrophic axis is a
    # catastrophe no matter how good the others look.
    #   severity 1.0 (exactly at a limit) -> 0.50x -> -60   (beats hovering:
    #                                                        trying pays)
    #   severity 1.2 (near miss)          -> 0.59x -> -71
    #   severity 2.5 (hard arrival)       -> 1.18x -> -141  (worse than hover)
    #   severity 7.6 (measured free fall) -> 3.00x -> -360  (far worse)
    # A tumble (`lost_control`, no `landing_margins`) takes the cap outright.
    crash_severity_base: float = 0.5
    crash_severity_k: float = 0.45
    crash_severity_cap: float = 3.0

    landing_bonus_scale: float = 450.0
    # REAL BUG FOUND (this session, see `tilt_cutoff_k`'s field comment for
    # the real episode numbers): narrowed the gap to `timeout_penalty`
    # (900 -> 150) alongside the tilt_cutoff fix. The ORIGINAL ordering
    # invariant (crash strictly worse than timeout) stays true -- this only
    # makes the margin small instead of large, so one unlucky crash during
    # an honest attempt isn't a catastrophically worse outcome than never
    # attempting at all. Combined with the tilt_cutoff fix, a real crash's
    # TOTAL episode return should now land close to (not far below) a
    # timeout's, instead of reliably worse by thousands.
    crash_penalty: float = 120.0
    # applied on truncation (timeout) without ever having landed -- must
    # be strictly less than crash_penalty (timing out shouldn't be worse
    # than an actual crash) but strictly more than zero (it can't be free).
    timeout_penalty: float = 105.0

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
    # RAISED again (this session, alongside `kappa_alt_sqrt`'s 12.66->50.64
    # restore and `tilt_cutoff_k`/`tilt_cutoff_cap`'s revert back to 2000 --
    # see both field comments): at a transient climb-away altitude (~330m,
    # the kind the climb-away exploit this session's `kappa_alt_sqrt` fix
    # targets actually reached), altitude alone is now 50.64*sqrt(330)~=920.
    # New max plausible sum ~920 altitude + 400 braking + ~112
    # velocity-tracking + 2000 tilt-cutoff + 150 xy-position + a few
    # single-digit terms, roughly ~3600 -- headroom kept above that. Still
    # a defensive clip, not a routine constraint: ordinary behavior (not a
    # degenerate climb-away or a mid-tumble) shouldn't come near this.
    # Raised again alongside the new `_xy_braking_envelope_penalty` (see
    # `t0_xy`'s field comment): adds up to `xy_braking_penalty_cap`=400
    # more. New max plausible sum ~4000.
    # RAISED 4200 -> 5000 alongside the new `_angular_rate_penalty` (see
    # `omega_k`), and re-derived from the caps rather than re-estimated.
    # Worst plausible simultaneous sum, every term at its ceiling in a
    # degenerate climb-away-plus-tumble state (altitude at the 1372 m such
    # a state actually reached in real telemetry):
    #     altitude 25.32*sqrt(1372) = 938   tilt_cutoff        2000
    #     brake_z                     400   omega               800
    #     brake_xy                    400   vtrack              200
    #     xy_position                 150   misc (prox/thr/rcs)   2
    #     -------------------------------------------------- total 4890
    # 4200 would have CLIPPED that, which breaks the property this field is
    # supposed to have: a defensive bound that ordinary flight never
    # approaches, not a constraint that quietly flattens the gradient in
    # the exact failure state the shaping is trying to push out of. (Note
    # this total is lower than it would have been before `kappa_alt_sqrt`
    # came back down 50.64 -> 25.32, which freed ~470 of the headroom the
    # new omega term now uses.) Measured on four real Isaac ZemZev landing
    # episodes (2489 steps total) the clip did not bind even once, and the
    # raw sum at the orbit_descent release point is ~1300, about a quarter
    # of this -- so it stays defensive.
    shaping_clip_abs: float = 5000.0

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
    # LOWERED again (this session, user direction: "normalize the rewards,
    # the numbers are extreme right now, shrink them but keep crashing
    # worse than hovering"). Real per-episode shaping sums were still
    # landing in the thousands (e.g. -4700 to -9300 total episode return,
    # mostly shaping, not the terminal term) even after the terminal-scale
    # fix above -- comparable to or bigger than the terminal bonus/penalty
    # themselves, which fights the whole point of giving the terminal
    # outcome real weight. Cut 4x (0.01 -> 0.0025): combined with the
    # tilt_cutoff_k/crash_penalty narrowing above, a typical episode's
    # shaping sum should now land in the low hundreds, not thousands,
    # while the terminal bonus/penalty (not rescaled here, see above)
    # keeps its full ~1050-1800 magnitude -- so the terminal outcome
    # dominates the final return far more reliably than before, and a
    # crash's extra shaping cost during the failing attempt (vs. a calm
    # hover) shrinks by the same 4x on top of the tilt_cutoff fix itself.
    #
    # READ THIS BEFORE RE-BALANCING ANYTHING IN THIS FILE (added this
    # session, and it invalidates the reasoning style of several comments
    # above, including this field's own paragraph): every "shaping sum vs.
    # terminal outcome" argument in this module compares UNDISCOUNTED
    # episode totals. SAC does not optimize that. It optimizes the
    # DISCOUNTED return, and `scripts/train_sac_isaac.py` was running SB3's
    # default gamma=0.99 with dt=0.05 s -- an effective horizon of
    # 1/(1-gamma) = 100 steps = 5.0 SECONDS, against episodes of 20-60 s.
    # A landing 700 steps (35 s) away was discounted by 0.99**700 = 8.8e-4,
    # so `landing_bonus_scale`=1800 was worth 1.58 return units at the
    # release point, i.e. ~1% of the decision -- the terminal signal this
    # field's paragraph below claims now "dominates the final return" was
    # in fact invisible, and cutting `reward_scale` 4x shrank the only
    # signal the agent could actually see while leaving the invisible one
    # untouched.
    #
    # Scored four strategy archetypes end to end (CPU, no Isaac) to check
    # which one the agent was actually being asked to pick:
    #     gamma=0.99 : land -198.2  STALL -146.9  crash -276.2  climb -194.1
    #     gamma=0.999: land +159.1  stall -1343.3 crash -1527.9 climb -1597.6
    # At the shipped gamma, STALLING AT 100 m WAS THE OPTIMAL POLICY --
    # better than a successful landing -- and climbing away was a tie with
    # landing. That is not a curriculum failure and not a tuning failure;
    # it is the objective being wrong, and it explains every "0/16
    # landed_safely, 11-14/16 ran the full clock" result on record. A
    # kappa/gamma sweep confirmed gamma is the binding constraint: at
    # gamma=0.99 stall wins for EVERY coefficient variant tried; at
    # gamma=0.999 landing wins for every one of them. `train_sac_isaac.py`
    # now passes gamma=0.999 (horizon 1000 steps = 50 s, matched to the
    # episode length). gamma=0.9995 was also tried and rejected: it breaks
    # the required crash-worse-than-timeout ordering.
    #
    # So: the undiscounted totals this file reasons about are only a valid
    # proxy while the discount horizon covers the episode. If episode
    # length or dt changes, gamma must move with it, or all of the balance
    # work documented above silently stops meaning anything.
    reward_scale: float = 0.00025


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


def _time_to_ground_s(alt_m: float, v_z_m_s: float, gravity_m_s2: float, floor_s: float) -> float:
    """Rough time-to-ground estimate for `_xy_braking_ratio` -- NOT a
    guidance-grade tgo solve (see `zemzev_controller.py`'s own iterative
    `_solve_feasible_tgo` for that); this only needs to be good enough to
    shape "how urgent is killing horizontal speed right now", the same way
    `_braking_ratio` doesn't need a perfect vz either.

    REAL BUG FOUND (this session, direct sweep of the shipped code): the
    original version had TWO branches -- `alt/|vz|` while descending, a
    free-fall-from-rest estimate otherwise -- and they were DISCONTINUOUS
    at vz=0, with the descending branch unbounded above as |vz| -> 0.
    Measured at alt=200 m, v_xy=25 m/s:
        vz=-1e-7 -> tgo     15.71 s, ratio 4.672  -> penalty 400.00 (cap)
        vz=-1e-3 -> tgo 200000.00 s, ratio 0.0004 -> penalty   0.00
        vz=-0.1  -> tgo   2000.00 s, ratio 0.037  -> penalty   0.03
    i.e. a vehicle hanging at 200 m with 25 m/s of horizontal speed shed
    the ENTIRE horizontal-braking penalty the instant it began sinking at
    one centimetre per second. The term added to punish stalling was in
    fact paying for it -- the single cheapest way to zero it out was to
    stop descending. (The vz>=0 branch had the mirror pathology: it used
    free-fall-from-rest, so climbing HIGHER lengthened tgo and made the
    term cheaper still -- 400 at 200 m down to 63.6 at 1372 m.)

    Replaced with the one exact ballistic solution of
    `alt + vz*t - 0.5*g*t**2 = 0`, i.e. `t = (vz + sqrt(vz**2 + 2*g*alt))/g`,
    which is continuous everywhere, bounded as vz -> 0, reproduces the old
    free-fall branch EXACTLY at vz=0 (both give sqrt(2*alt/g) = 15.7135 s
    at alt=200), and is physically right for vz>0 too (coast up, then fall
    back: vz=+30 at 200 m gives 42.81 s, not the old 15.71 s). Still not a
    guidance-grade tgo solve (see `zemzev_controller.py`'s iterative
    `_solve_feasible_tgo` for that) -- it ignores thrust, as it should,
    since this is an urgency signal about the UNPOWERED budget, not a plan.
    """
    disc = max(v_z_m_s * v_z_m_s + 2.0 * max(gravity_m_s2, 0.1) * max(alt_m, 0.0), 0.0)
    return max(float((v_z_m_s + np.sqrt(disc)) / max(gravity_m_s2, 0.1)), floor_s)


def _xy_braking_ratio(w: RewardWeights, alt_m: float, v_z_m_s: float, v_xy: float, mass_kg: float,
                       dps_thrust_max_n: float, gravity_m_s2: float) -> float:
    """Horizontal analog of `_braking_ratio` -- see `t0_xy`'s field comment
    for why this was added and `_xy_braking_envelope_penalty`'s docstring
    for the shape. Required horizontal decel is `v_xy / time_to_ground`
    (kill all horizontal speed by the time the vehicle reaches the ground),
    compared against the SAME real deceleration authority and the SAME
    `braking_authority_margin` the vertical channel uses (one vehicle, one
    thrust budget shared between both channels via tilt -- no reason to
    assume a different authority fraction for each).
    """
    a_avail = max(dps_thrust_max_n / max(mass_kg, 1.0) - gravity_m_s2, 0.05)
    tgo = _time_to_ground_s(alt_m, v_z_m_s, gravity_m_s2, w.xy_braking_tgo_floor_s)
    a_required = v_xy / tgo
    ratio = a_required / (w.braking_authority_margin * a_avail)
    return min(ratio, 1000.0)


def _xy_braking_envelope_penalty(w: RewardWeights, ratio_clamped: float) -> float:
    """Horizontal analog of `_braking_envelope_penalty` -- same polynomial
    shape (zero at ratio=0, 1 at ratio=1, grows gently past that so it
    stays differentiable instead of saturating immediately), applied to
    `_xy_braking_ratio` instead of the vertical `_braking_ratio`. See
    `t0_xy`'s field comment for why this exists: the vertical channel had
    this real time-to-ground urgency signal from early in the project,
    the horizontal channel never did, and a direct telemetry diagnostic
    this session showed exactly that gap in the resulting behavior
    (vertical speed sometimes well-controlled at touchdown, horizontal
    speed always 15-32 m/s over a ~1.2 m/s limit).
    """
    return min(w.t0_xy * float(ratio_clamped ** w.xy_braking_theta), w.xy_braking_penalty_cap)


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
    # NOTE (2026-10-05): a direction-aware version of this gate was tried and
    # REVERTED, because it is numerically irrelevant. `quad` is mu*tilt^2 with
    # tilt in RADIANS, so at 52 deg (0.908 rad) and mu_base=0.5 it is 0.412 raw
    # = 0.0001/step after `reward_scale`. The whole mu_base/mu_scaled mechanism
    # is worth 1e-4 to 1e-3 per step; at a tilt where it might matter, the
    # `tilt_cutoff` barrier below is already 1000x larger. Making the discount
    # require the tilt to be DECELERATING moved a measured exploit state
    # (tilt 52 deg, v_xy 42 m/s, accelerating) from 0.159 to 0.160 per step.
    # If this term is ever meant to bite, mu has to move by ~3 orders of
    # magnitude first -- and then the gate's blindness to tilt DIRECTION does
    # become a real loophole worth closing.
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


def _angular_rate_penalty(w: RewardWeights, wx: float, wy: float, wz: float,
                           safe_w_rad_s: float) -> float:
    """Penalize body angular RATE -- see `omega_k`'s field comment for the
    measured episode this was added for. Deliberately NOT velocity-gated
    (unlike `_attitude_hold_penalty`'s quadratic): a high horizontal speed
    justifies holding a large tilt ANGLE, it never justifies spinning, and
    the landing test rejects on rate regardless of what the vehicle is
    doing. All three axes count, including yaw, which no other term reads.
    """
    w_mag = float(np.sqrt(wx * wx + wy * wy + wz * wz))
    quad = w.omega_k * w_mag * w_mag
    ratio = w_mag / max(safe_w_rad_s, 1e-6)
    barrier = min(w.omega_cutoff_k * float(ratio ** w.omega_cutoff_power), w.omega_cutoff_cap)
    return min(quad + barrier, w.omega_penalty_cap)


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


def _touchdown_severity(p, s, margins: dict) -> float:
    """How many times over its worst limit this touchdown was. 1.0 means
    exactly at a limit; below 1.0 means it was inside every one of them.

    Computed from the state rather than from `landing_margins`, because
    those are clipped to [0, 1] and so carry no information about the size
    of a violation -- which is exactly what has to be graded here. leg_diff
    is the one criterion not in the state; its margin saturating at 0 is
    taken as "just over" (severity 1.0), since terrain flatness is not the
    axis a dive blows out.
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
    # see `crash_severity_base`'s field comment for the measured defect this
    # replaces. No `margins` means the episode ended without a touchdown
    # (a `lost_control` tumble), which takes the cap outright.
    if margins and severity is not None:
        sev = max(severity, 1.0)
        mult = min(w.crash_severity_base + w.crash_severity_k * (sev - 1.0), w.crash_severity_cap)
        return -w.crash_penalty * mult
    return -w.crash_penalty * w.crash_severity_cap


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
        # `_braking_ratio` squares vz and is therefore sign-blind. That is
        # CORRECT for `_braking_envelope_penalty` (a fast climb should pay
        # the envelope term too) and WRONG for `_altitude_penalty`'s gate,
        # which was switching the descend incentive off during climb-aways
        # -- see `altitude_vz_gate_k`'s field comment for the measured
        # numbers. Feed the gate descent speed only.
        descent_ratio = _braking_ratio(
            w, info["altitude_m"], min(s["vz"], 0.0), mass_kg, p.dps_thrust_max_n, p.gravity_m_s2)
        xy_braking_ratio = _xy_braking_ratio(
            w, info["altitude_m"], s["vz"], v_xy, mass_kg, p.dps_thrust_max_n, p.gravity_m_s2)

        r = 0.0
        r -= _proximity_penalty(w, dist_m)
        r -= _altitude_penalty(w, info["altitude_m"], descent_ratio)
        r -= _braking_envelope_penalty(w, braking_ratio)
        r -= _xy_braking_envelope_penalty(w, xy_braking_ratio)
        r -= _leg_load_penalty(w, info.get("leg_force_n", 0.0), info.get("leg_force_max_n", 1.0))
        r -= _throttle_effort_penalty(w, s["throttle"])
        r -= _rcs_l1_penalty(w, s["rcs_pitch"], s["rcs_roll"], s["rcs_yaw"])
        r -= _rcs_l2_penalty(w, s["rcs_pitch"], s["rcs_roll"], s["rcs_yaw"])
        r -= _attitude_hold_penalty(w, s["tilt_x"], s["tilt_y"], v_xy, p.loss_of_control_tilt_rad)
        r -= _angular_rate_penalty(w, s["wx"], s["wy"], s["wz"], p.safe_landing_w_rad_s)
        r -= _velocity_tracking_penalty(w, v_xy)
        r -= _xy_position_penalty(w, dist_m, info["altitude_m"])
        r -= w.alive_tax_per_step  # 0.0 by default -- see field comment for why it is disabled

        # `reward_scale` applies ONLY to the per-step shaping sum (see its
        # field comment) -- the terminal bonus/penalty below is added at its
        # own raw scale so it isn't shrunk 100x relative to a whole
        # episode's accumulated shaping. Never clipped away by the shaping
        # cap either way.
        r = float(np.clip(r, -w.shaping_clip_abs, w.shaping_clip_abs)) * w.reward_scale
        if info.get("terminated"):
            r += _terminal_reward(w, info, _touchdown_severity(p, s, info.get("landing_margins") or {}))
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
