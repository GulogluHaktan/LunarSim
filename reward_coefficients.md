# Reward Function Coefficients — `lunarsim/rl/reward.py`

This document lists every coefficient in `RewardWeights` (the SAC training
reward for the Apollo LM lander), what it controls, its current value, and
**why** it has that value — including the real training-run failures that
drove each change. Read this before touching reward.py again; most of these
values are not guesses, they're scar tissue from a specific failure mode.

Reward is computed once per physics control step (`dt_s = 0.05s`) in
`make_apollo_reward_fn`'s `reward_fn`:

```
r = 0
r -= proximity_penalty
r -= altitude_penalty
r -= braking_envelope_penalty
r -= leg_load_penalty
r -= throttle_effort_penalty
r -= rcs_l1_penalty
r -= rcs_l2_penalty
r -= attitude_hold_penalty
r -= position_tracking_penalty
r = clip(r, -shaping_clip_abs, +shaping_clip_abs)   # only the 9 terms above
if terminated: r += terminal_reward(...)            # added AFTER the clip
elif truncated: r -= timeout_penalty
```

The terminal/timeout adjustment is **not** subject to `shaping_clip_abs` —
only the 9 per-step shaping terms are clipped as a group, before the
terminal bonus/penalty is added.

---

## 2026-09-30 revision — informed by a hand-designed feasibility controller

Before touching this reward again, a non-RL guidance+attitude controller
(`lunarsim/control/zemzev_controller.py`, ZEM/ZEV guidance + PD attitude
loop, built on the real vehicle specs, no reward function involved at all)
was built and tuned to answer: **is a safe landing from `orbit_descent`
(200m release, 10-30 m/s horizontal, real terrain) physically achievable
with this vehicle model at all?** After fixing several of its own bugs it
reached **75% `landed_safely`, zero crashes**, with >95% fuel and >30%
flight-time margin remaining at touchdown — proving the mission IS
control-theoretically achievable, and that the earlier RL training
failures were not purely a hard physical ceiling.

That controller's own successful episodes were then replayed through the
*existing* reward function (before this revision) to see how each term
scored a trajectory now known to be correct. This surfaced three real
problems, all fixed in this revision:

1. **`_altitude_penalty` and `_braking_envelope_penalty` could each, alone,
   exceed the entire `shaping_clip_abs` budget** at orbit_descent's 200m
   release altitude — see the updated §2 and §3 below. This silently
   flattened the per-step shaping reward to a constant for the whole
   high-altitude portion of every orbit_descent episode, exactly where the
   policy most needed a gradient toward "start braking now." `hover_only`/
   `final_approach` never hit this (their max altitudes, 20m/35m, keep
   these terms under the old clip) — which line up exactly with which
   stages trained fine and which didn't.
2. **`_attitude_hold_penalty` was a flat quadratic on tilt**, but the
   controller needed 25-35° of *sustained* tilt for 15-20s to brake
   horizontally — a flat penalty fights that directly, treating a correct
   braking commitment the same as genuine tumbling. Now velocity-gated —
   see the updated §8.
3. **`_position_tracking_penalty` pulled toward the exact touchdown
   position**, which `landed_safely` never actually checks. The controller
   demonstrated this term actively fighting the correct maneuver: it kept
   tilt alive to close out a lateral offset that didn't matter, and because
   attitude response is slow, that leftover tilt didn't unwind before
   touchdown and re-accelerated the vehicle sideways. Position terms
   removed; renamed to `_velocity_tracking_penalty` — see the updated §9.

Verification: replaying the same controller episodes through the *revised*
reward shows the shared `shaping_clip_abs` clip no longer binds on a single
step of a successful orbit_descent landing (it bound on essentially every
step before), and a random policy still scores many orders of magnitude
worse than a safe landing (sanity check, not a regression test).

---

## 1. Proximity penalty — `_proximity_penalty`

```
penalty = alpha * (1 / max(dist_m, dist_floor_m)) ** beta
```

| param | value | meaning |
|---|---|---|
| `alpha` | 0.2 | overall weight |
| `beta` | 1.0 | exponent (plain inverse-distance, not inverse-square) |
| `dist_floor_m` | 2.0 | clip distance so the term doesn't blow up near dist=0 |

**Why it exists:** discourages loitering far from the pad at altitude,
without ever committing to a real touchdown attempt — a "get closer
horizontally" pressure, separate from the quadratic tracking term (#8).

**Real bug history:** originally `alpha=2.0`. At that weight this term is a
near-constant per-step tax for merely being near the pad, active for the
*entire* episode regardless of behavior quality. Since a careful safe
landing takes more steps than a reckless crash, the extra steps' tax
exceeded the terminal-reward gap between landing safely and crashing —
crashing fast became the mathematically better policy. Confirmed for real:
1.5M steps of training got WORSE with more training (0/40 landed, 40/40 hit
loss-of-control) — the signature of a real optimum being found, not an
undertrained policy. Cut 10x to `alpha=0.2`.

---

## 2. Altitude closure — `_altitude_penalty` (added beyond the original spec)

```
penalty = min(kappa_alt * max(alt_m, 0), altitude_penalty_cap)
```

| param | value |
|---|---|
| `kappa_alt` | 2.0 |
| `altitude_penalty_cap` | 80.0 (new) |

**Real bug history #3 (the cap, found via the feasibility-controller
replay, see the revision note above):** this term was never rechecked
against `orbit_descent`'s 200m release altitude. At `kappa_alt=2.0`,
`alt=200m` alone gives 400 — bigger than the *entire* `shaping_clip_abs`
budget (300 at the time), at step one, before any other term is even read.
`hover_only`/`final_approach` (max altitude 20m/35m, naturally topping out
at 40/70) never hit this. Capped individually at 80 (just above
`final_approach`'s natural max, so those stages are unaffected) using the
same pattern `_position_tracking_penalty` already used for its own
swamping bug (#9 below).

**Why it exists:** the handed-over reward spec's proximity term only
measures *horizontal* distance — nothing costs remaining *altitude*
specifically. Without this term a trained policy found a genuinely
excellent, stable hover at ~20m (tilt 1.5–8°, vxy 0.2–1.4 m/s, vz≈0 — all
already inside every safe-landing threshold) and simply never descended,
since nothing was pushing it to. This restores the "encourage descending"
term an earlier version of the codebase had.

**Real bug history:** tried `0.05` then `0.3` first — both only partially
worked. Each retrain converged to the *same* hover-forever pattern, just at
a progressively lower equilibrium altitude (0.05 → ~13–20m stable hover,
0.3 → ~2.6–3.0m stable hover) — a clean, repeated, linear confirmation the
term works, just not steeply enough to beat the residual risk-aversion. At
0.3 the remaining gap was almost comically small (every OTHER safety metric
already inside its threshold at 2.8m, it just wouldn't cross the last few
meters). Raised another ~7x to 2.0.

---

## 3. Braking-envelope barrier — `_braking_envelope_penalty`

```
v_limit(alt) = safe_v_z_m_s + braking_profile_k * alt_m ** zeta
ratio = |vz| / v_limit(alt)
penalty = min(t0 * exp(min(ratio ** theta, exp_arg_cap)), braking_penalty_cap)
```

| param | value | meaning |
|---|---|---|
| `t0` | 1.0 | overall scale |
| `theta` | 2.0 | how sharply the barrier turns on past the limit |
| `zeta` | 0.5 | sqrt(altitude)-shaped speed-limit profile (matches real PDI guidance profiles) |
| `braking_profile_k` | 0.3 | how fast the allowed speed grows with altitude |
| `exp_arg_cap` | **9.0** (was 4.0) | hard clip on the exponent, for numerical safety only |
| `braking_penalty_cap` | 100.0 (new) | hard clip on the *result*, so this term can't alone swamp `shaping_clip_abs` |

**Real bug history #3 (`braking_penalty_cap`, found via the
feasibility-controller replay, see the revision note above):** at
`exp_arg_cap=9.0` this term's own ceiling is `t0*exp(9)≈8103` — 27x the
entire `shaping_clip_abs` budget on its own. Capped the result at 100
(same reasoning as `altitude_penalty_cap` above): still climbs steeply
well before saturating at this lower ceiling, so the gradient over the
range that matters is unaffected, but one term can no longer flatten every
other term's contribution to a constant clip value.

**Why it exists:** read as a guided-descent "stay under the braking
parabola" barrier — `ratio` is current descent rate over an
altitude-dependent speed budget; exceeding it gets punished increasingly
sharply (the exp shape).

**Real bug history #1 (why `exp_arg_cap` exists at all):** an uncapped or
loosely-capped exponential saturates every step of nearly every episode
during early/random-policy exploration (the vehicle starts *far* outside
the safe envelope by design) and dwarfs every other reward term by 6+
orders of magnitude. This is genuinely a *numerical* safety concern (an
uncontrolled `exp()` argument can overflow to `inf`/`nan`, which then stays
`nan` through `np.clip` and corrupts training), not just a magnitude one.

**Real bug history #2 (why the cap was raised from 4.0 → 9.0, found in this
session):** watching a trained `orbit_descent` checkpoint's actual eval
dashboard (released at 200m/~20 m/s) showed the policy holding
throttle≈1.0 — net near-zero acceleration, i.e. just cancelling gravity,
not actually braking — through the whole high-speed part of the descent,
then genuinely **easing off** throttle (0.77→0.9) through the *middle* of
the flight (backwards from a real braking profile), before slamming to
~1.0 only in the last ~20m — too late to shed the remaining speed before
impact. Root cause: at `cap=4`, `exp(4)≈54.6` is reached (and clipped flat)
the instant `ratio > 2`, which is true for nearly the *entire* descent from
a 200m/20+ m/s release, since `v_limit(alt)` is only a few m/s through most
of that altitude range. Once saturated, braking *harder* earns **zero**
extra reward from this term — no local gradient says "brake now" beats
"brake later," so the throttle-effort penalty (#5) makes easing off the
locally cheaper choice. Raising the cap keeps the term differentiating
"how far over budget" across the range a real high-altitude release
actually visits, instead of going flat almost immediately.

This raise is numerically safe: `shaping_clip_abs` (below) clips the
*summed* per-step shaping reward, applied every step before the terminal
bonus/penalty is added — so even at the new max (`exp(9)*t0 ≈ 8103`) a
single step's total shaping reward still can't exceed `shaping_clip_abs`
in magnitude; it just no longer goes flat for the deep-in-violation region.

**⚠️ UNRESOLVED as of this handover:** raising the cap to 9.0 and
retraining `orbit_descent` from scratch did **not** fix the timing problem
— it flipped it. The retrained policy now eases throttle down to
*near-zero* for a long stretch mid-descent (alt 130m→9.5m, throttle
~0.001–0.2, essentially free-falling) before panicking and firing late,
still crashing (tumbling at touchdown, `lost_control`). Both the tight cap
(4.0, "coast at ~gravity-cancelling thrust the whole time") and the loose
cap (9.0, "coast at near-zero thrust the whole time") produce a "don't
commit to braking until forced" strategy — the underlying issue may not be
this term alone. See `handover.md`'s "what's actually still broken" section
for the current best hypothesis and untried ideas.

---

## 4. Leg touchdown load — `_leg_load_penalty`

```
penalty = h * (leg_force_n / leg_force_max_n) ** c        [0 while airborne]
```

| param | value |
|---|---|
| `h` | 1.0 |
| `c` | 2.0 |

**Why it exists:** estimated per-leg touchdown force (from vertical KE at
contact, spread over the real leg count and an engineering-estimate strut
stroke — see `leg_force_bounds_n`) over the estimated max the real landing
gear could take. Zero while airborne, only active at the instant of
touchdown. Not independently re-tuned; inherited from the original spec
handoff.

---

## 5. DPS throttle effort — `_throttle_effort_penalty`

```
penalty = s * throttle ** 2
```

| param | value |
|---|---|
| `s` | 0.05 |

**Why it exists:** small continuous cost for firing the descent engine, so
the policy doesn't burn fuel needlessly once genuinely safe. **Caution:**
this is also the term that made "ease off throttle" locally cheaper once
the braking-envelope term saturated (see #3, bug #2) — if the braking term
gets re-tuned again, re-check whether this penalty is now fighting it.

---

## 6/7. RCS effort — `_rcs_l1_penalty` / `_rcs_l2_penalty`

```
l1 = lam * (|pitch_cmd| + |roll_cmd| + |yaw_cmd|)
l2 = gamma * (pitch_cmd**2 + roll_cmd**2 + yaw_cmd**2)
```

| param | value | meaning |
|---|---|---|
| `lam` | 0.02 | L1, fuel-linear cost (replaces the TVC/gimbal term the real DPS doesn't have) |
| `gamma` | 0.01 | L2, discourages saturating all three RCS axes simultaneously |

Not independently re-tuned this session.

---

## 8. Attitude hold — `_attitude_hold_penalty` (added beyond the original spec)

```
urgency = clip(1 - v_xy / attitude_hold_v_xy_ref_m_s, 0, 1)
effective_mu = mu_base + mu_scaled * urgency
penalty = effective_mu * (tilt_x ** 2 + tilt_y ** 2)
```

| param | value |
|---|---|
| `mu_base` | 0.5 (new) |
| `mu_scaled` | 4.5 (new) |
| `attitude_hold_v_xy_ref_m_s` | 12.0 (new) |

(`mu_base + mu_scaled = 5.0`, the old flat `mu`, for the "tilted with no
excuse" case where `v_xy≈0`.)

**Why it exists:** the original handed-over spec only penalizes RCS
*effort* (#6/#7), never attitude *error* itself — a policy that simply
stops firing RCS pays zero penalty even while tumbling. Confirmed for real:
a trained policy reached 200+ degrees of tilt and free-fell the rest of a
25s episode with no corrective signal at all (the `loss_of_control_tilt_rad`
cutoff bounds the *damage*, but doesn't stop the drift that causes it).

**Real bug history #1:** `mu=0.5` was tried first and was too weak — a real
training run still hit the 60° loss-of-control cutoff in 39/40 eval
episodes despite it. Raised 10x to 5.0.

**Real bug history #2 (velocity-gating, found via the
feasibility-controller replay, see the revision note above):** a flat
`mu=5.0*(tilt_x²+tilt_y²)` was later found fighting a maneuver that's
actually *necessary*: the hand-designed controller could only solve
`orbit_descent` by committing to 25-35° of sustained tilt for 15-20s to
brake horizontally, then unwinding it before touchdown. A flat penalty
punishes that entire commitment phase exactly as hard as genuine
loss-of-control tumbling, with no distinction between "tilted because it's
earning its keep" and "tilted for no reason." Split into an always-on
floor (`mu_base`, still catches real tumbling regardless of speed) plus a
velocity-gated component (`mu_scaled`) that only turns on as `v_xy` drops
toward the reference speed — `attitude_hold_v_xy_ref_m_s=12.0` reuses the
exact reference speed the controller's own tilt-taper needed to avoid its
"can't unwind in time" overshoot failure, so the reward and a known-working
controller agree on when tilt stops being free.

---

## 9. Velocity tracking — `_velocity_tracking_penalty` (renamed from `_position_tracking_penalty`)

```
raw = x5*|vxy|**x6
penalty = min(raw, position_penalty_cap)
```

| param | value | meaning |
|---|---|---|
| ~~`x1`~~ | removed | was the x-position weight |
| ~~`x2`~~ | removed | was the x-position exponent |
| ~~`x3`~~ | removed | was the y-position exponent |
| ~~`x4`~~ | removed | was the y-position weight |
| `x5` | 0.125 | horizontal-velocity weight |
| `x6` | 2.0 | horizontal-velocity exponent |
| `position_penalty_cap` | 200.0 | hard cap on this term alone (name kept for continuity with the field/history below) |

**Why it exists:** the core "don't carry excess horizontal speed into
touchdown" shaping term. It used to also pull toward the exact
`(target_x, target_y)` touchdown position; that part is gone (see bug
history #2 below).

**Real bug history #1 (the cap):** originally uncapped. Once a policy
drifts even a few hundred meters off target (routine for an undertrained
policy), this single term swamped every other term *and* used to swamp the
shared `shaping_clip_abs` — so a bad trajectory read back as one flat,
saturated penalty with zero gradient telling the agent further drift is
even worse. Capping the term itself (not just the summed total) keeps the
informative quadratic gradient over the moderate-distance range while
still bounding the worst case.

**Real bug history #2 (`x5`):** `0.05 → 0.2` was tried alongside raising
`t0` (braking term, #3) from 0.5→1.5 in the same experiment, and overshot
in the other direction once `t0` also moved — these two are coupled (a
real physical coupling: tilting to correct vz induces vxy drift, and vice
versa), not independent knobs. Settled at the midpoint, `x5=0.125`, with
the understanding that more training time (not a bigger x5) is what lets
the network find a joint solution instead of the two terms fighting each
other across short, separately-tuned continuations.

**Real bug history #3 (removing `x1`/`x2`/`x3`/`x4`, found via the
feasibility-controller replay, see the revision note above):**
`landed_safely` (`analytic_lander_env.py`) never checks touchdown
position, only velocity/tilt/rate — so pulling toward the exact
`(target_x, target_y)` was pure unrewarded-by-the-actual-objective shaping.
Worse, the hand-designed controller demonstrated it actively fighting the
correct maneuver: including a position-return term kept it committing
tilt to close out a lateral offset that didn't matter, and because
attitude response is slow (~14s to unwind a 30° tilt at this vehicle's
real RCS-torque-to-inertia ratio), that leftover tilt didn't unwind before
touchdown and re-accelerated the vehicle sideways in the final seconds —
undoing an already-nulled horizontal velocity. Spawn already aims initial
velocity at the target (see `AnalyticLanderEnv.reset`), so removing the
position pull doesn't stop the vehicle from generally heading toward the
pad; it just stops fighting the timing-critical part of the maneuver near
the end. The original bug #1 above (about capping the term) predates this
fix and remains relevant for the `x5*vxy^x6` term that's left.

---

## 10. Terminal reward — `_terminal_reward`

```
if landed_safely:
    quality = mean(landing_margins.values())   # each margin in [0, 1]
    bonus = landing_bonus_scale * (0.5 + 0.5 * quality)
elif margins present (touched down, but missed >=1 safety threshold):
    quality = mean(landing_margins.values())
    penalty = -crash_penalty * (1.0 - 0.5 * quality)
else:  # genuine airborne tumble crash (lost_control), or never touched down
    penalty = -crash_penalty
```

Plus, on truncation (timeout without ever landing): `r -= timeout_penalty`.

| param | value | meaning |
|---|---|---|
| `landing_bonus_scale` | 500.0 | max bonus for a perfect landing |
| `crash_penalty` | 700.0 | penalty for a genuine crash |
| `timeout_penalty` | 400.0 | penalty for timing out still airborne |

**Why the near-miss grading exists (found THIS session, via a full training
run's 8/8 evaluated checkpoints):** a genuine airborne tumble crash and a
touchdown that missed exactly one threshold (e.g. a real `orbit_descent`
eval: vz=0.40 m/s and tilt=12.4° both comfortably inside their limits, only
vxy=3.73 m/s over its bar) used to score **identically**: `-crash_penalty`
either way. `landing_margins` is only ever populated when the vehicle
touched down without tripping `lost_control` — so its presence/absence is
exactly the "near miss vs. real crash" distinction needed. Grading the
near-miss case by how far under the *other* thresholds it still was
restores a gradient toward "brake more," instead of every miss reading as
an equally bad catastrophe — while staying strictly worse than any actual
safe landing (worst safe landing floor: `landing_bonus_scale*0.5=250`; best
near-miss ceiling: `-crash_penalty*0.5=-350`).

**Real bug history (the 500/700/400 balance):** the original 100/100 was
too small — episode *duration* dominated the economics more than the
terminal outcome did (see term #1's history). 2000/3000 overcorrected: a
real training run showed 39/40 episodes timing out with the vehicle still
airborne, often at *already-safe* vz (0.2–1.8 m/s) but never committing to
touchdown — `crash_penalty` was so large relative to a landing attempt's
failure odds that never trying became the rational policy, and timeout
carried zero penalty, making "hover forever" a completely free way to
dodge risk. Fixed both: brought `crash_penalty` down closer to
`landing_bonus_scale` (still bigger, but not overwhelming), and added
`timeout_penalty` (strictly less than `crash_penalty`, strictly more than
zero) so passively refusing to land isn't free either.

---

## Safety net — `shaping_clip_abs`

| param | value |
|---|---|
| `shaping_clip_abs` | **400.0** (was 300.0) |

Clips the **summed** per-step shaping reward (terms #1–#9 combined) to
±400, applied every step *before* the terminal/truncation adjustment is
added. Meant to stay a purely defensive numerical backstop, not a real
constraint — if it binds regularly during training it silently flattens
the gradient exactly when it's most needed (this happened for real, twice:
once for term #9's old uncapped position terms, and far more severely for
`_altitude_penalty`/`_braking_envelope_penalty` at `orbit_descent`'s
altitude — see the 2026-09-30 revision note near the top of this doc).

**Real bug history (raised 300→400):** with every term now individually
capped (§2's `altitude_penalty_cap=80`, §3's `braking_penalty_cap=100`,
§9's `position_penalty_cap=200`, plus a handful of single-digit terms), the
worst-case simultaneous sum is close to 300 already — 300 would still bind
constantly during the highest-altitude/highest-speed portion of a
descent, the same failure mode as before just less extreme. Raised to give
the now-individually-capped terms room to coexist without the group clip
re-flattening their sum. Treat any training run where this still binds
regularly as a signal one of the individual caps above needs revisiting.

---

## Landing safety thresholds (NOT in `RewardWeights` — `LanderParams`,
`lunarsim/rl/analytic_lander_env.py`)

These define `landed_safely` / `landing_margins`, which `_terminal_reward`
above reads:

| param | value | grounding |
|---|---|---|
| `safe_landing_v_z_m_s` | 1.0 | stricter than the real LM's ~3.05 m/s (10 ft/s) qualified touchdown limit — kept conservative since it was never the actual bottleneck |
| `safe_landing_v_xy_m_s` | **1.2** (was 0.5) | raised this session to match the real LM landing gear's documented ~1.22 m/s (4 ft/s) qualified horizontal touchdown tolerance (Grumman/NASA LM-10 landing gear qualification figures) — the old 0.5 m/s default was never tied to any real number |
| `safe_landing_tilt_rad` | 15° | not re-derived this session |
| `safe_landing_w_rad_s` | 0.5 | not re-derived this session |

**Real bug history:** every single checkpoint evaluated across two full
independent training curricula (8/8) failed landing safety specifically on
`vxy` (ranged 1.78–3.99 m/s, never once under 2x the *old* 0.5 m/s limit),
while `vz` and `tilt` each passed cleanly in at least one run — a single
term failing 8/8 times while the others each succeed sometimes is a
threshold-calibration signal, not purely an undertrained-policy signal.
Raised to the real vehicle's own qualified tolerance.

**Caveat found immediately after, still relevant:** raising this threshold
alone did not produce a safe landing in the one attempt tested against it —
`vxy` at touchdown is still routinely 3–12 m/s in later evals, i.e. still
2.5–10x even the *relaxed* threshold. This threshold fix is very likely
still correct/necessary, but it was not sufficient on its own — see
`handover.md`.

---

## Also relevant (not a reward term, but interacts with braking behavior)

`angular_damping_per_s = 0.5` in `LanderParams` — a stand-in for the real
Apollo LM's Digital Autopilot rate-damping inner loop. Without it the
attitude dynamics are a pure undamped double integrator (torque→rate→angle,
no decay), which is unconditionally unstable under naive position-only
control. Found necessary via a real failed training run (every trained
policy rang up to the loss-of-control cutoff without it). **NOTE**: this is
applied inside the training env (`IsaacLanderEnv`/`IsaacLanderVecEnv`) as a
manual per-substep velocity reset; `scripts/isaaclab_policy_eval_capture.py`
instead applies it via PhysX's native `PhysxRigidBodyAPI.angularDamping`
attribute on the vehicle prim (mathematically equivalent effect, different
mechanism — see that script's own REAL BUG comment for why the two differ).
