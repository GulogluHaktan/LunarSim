"""Fixed-scale observation normalization for the shared 20-element lander
observation layout (position offset x/y/z, vx/vy/vz, quaternion wxyz,
angular rate wx/wy/wz, fuel_frac, rcs_fuel_frac, time_remaining_frac,
velocity error vs. the guidance field x/y/z, t_go) used identically by
`AnalyticLanderEnv`, `IsaacLanderEnv`, `IsaacLanderVecEnv`, and
`scripts/isaaclab_policy_eval_capture.py`'s hand-built observation.

WIDENED 16 -> 20 (2026-10-07), deliberately as a WIDENING rather than as a
repurposing of existing slots. The previous change to this layout -- slot 15
going from a dead `leg_force_frac` to time-remaining while the vector stayed
16 wide -- let every pre-change checkpoint load SILENTLY into an env whose
slot 15 now ramps 1.0 -> 0.0, and the resulting 22% -> 5% drop was
misattributed to an unrelated change for days. Four appended slots change
`observation_space.shape`, so stable-baselines3 refuses an old checkpoint
with a shape error instead of running it on an input it never saw. That
loud failure is the point, not a side effect.

REAL BUG FOUND (this session, "go full depth in RL, find the reason"):
every one of those four call sites fed the RL policy completely raw,
unnormalized values -- position offset O(0-80m+), altitude O(0-200m),
velocity O(0-30+ m/s), angular rate O(rad/s), all mixed with quaternion
components and fuel fractions already O(0-1). No `VecNormalize` or any
other scaling was used anywhere in the training pipeline
(`scripts/train_sac_isaac.py`). That's 2-3 orders of magnitude of scale
mismatch across input dimensions feeding a standard MLP with default
initialization -- a well-known cause of slow/unstable learning
independent of any reward-shaping issue, and a plausible real contributor
to the noisy critic_loss / stuck ent_coef seen in real orbit_descent
training runs this session.

A static, fixed-constant scale (rather than SB3's `VecNormalize`, which
uses running statistics) was chosen deliberately: `VecNormalize`'s running
mean/std needs its own save/load discipline kept in sync between training
and every eval/render script that constructs this observation by hand
(4 call sites here) -- a real train/eval skew risk if any one of them
drifts out of sync. Fixed constants, derived from the actual physical
release envelope (`LanderParams`/`test_landing_feasibility.py`'s
`orbit_descent` stage, the hardest/widest one), are simpler to keep
identical everywhere: one import, no state, no save/load.
"""
import numpy as np

# One scale per observation dimension, in the SAME order every call site
# builds its observation vector. Chosen so a dimension's typical operating
# range maps to roughly [-1, 1] (not a hard clip -- values can exceed this
# during transients, e.g. a bad drift or a high-speed release -- just the
# scale the network sees "typical" values at).
OBS_SCALE = np.array([
    100.0, 100.0,   # x, y offset from target (m) -- orbit_descent's 80m spawn radius + margin
    200.0,          # altitude (m) -- orbit_descent's 200m release altitude
    30.0, 30.0, 30.0,  # vx, vy, vz (m/s) -- orbit_descent's up-to-30 m/s release + free-fall margin
    1.0, 1.0, 1.0, 1.0,  # quaternion wxyz -- already unit-scale
    2.0, 2.0, 2.0,  # wx, wy, wz (rad/s) -- angular_damping_per_s=0.5 keeps these small in practice
    1.0, 1.0,       # fuel_frac, rcs_fuel_frac -- already [0, 1]
    1.0,            # time_remaining_frac -- already [0, 1]; was the dead leg_force_frac slot
    # ---- appended 2026-10-07, following arXiv:1810.08719 -------------- #
    # 16-18: velocity ERROR against `reward.target_velocity`, i.e.
    # (vx, vy, vz) - (vx_t, vy_t, vz_t). Same 30.0 as the raw velocities at
    # 3-5 on purpose: this is a velocity in the same units, over the same
    # physical range (the error is bounded by the release speed plus the
    # field's own demand, both already inside that envelope), and giving the
    # error a different scale than the velocity it is computed from would
    # make the two slots disagree about what "fast" means.
    30.0, 30.0, 30.0,
    # 19: t_go, SECONDS of flight left if the vehicle flies the field.
    #
    # MEASURED to pick this, from `reward.target_velocity`'s own output
    # rather than from a round number. `t_go = alt / |vz_t(alt)|` with the
    # fitted schedule `|vz_t| = max(0.547*alt^0.431, 0.80)`, evaluated at the
    # curriculum stages' RELEASE altitudes:
    #     ramp_20m    20 m -> 10.1 s     ramp_100m 100 m -> 25.1 s
    #     ramp_35m    35 m -> 13.8 s     ramp_150m 150 m -> 31.6 s
    #     ramp_50m    50 m -> 16.9 s     orbit_descent 200 m -> 37.3 s
    # and small on its own near the ground: 2.5 s at 2 m, 1.3 s at 1 m,
    # 0.3 s at 0.25 m.
    #
    # The RELEASE value is not the typical value, though, and the schedule
    # makes the difference exact rather than a guess: with |vz| = c*h^p,
    # dh/dt = -c*h^p integrates to t_go(t) = t_go(0) - (1-p)*t, i.e. t_go
    # decays LINEARLY in time at 0.569 s per second. So along the field t_go
    # is uniform on [0, t_go(0)] and its median over an episode is exactly
    # half the release value -- 5.0, 6.9, 8.5 and 12.6 s on the four stages
    # training actually reaches (ramp_20m through ramp_100m).
    #
    # 10.0 brackets that set, so 1.0 is the middle of what the policy spends
    # its time looking at: the per-episode medians land at 0.50-1.26 and the
    # start-of-episode values at 1.0-2.5. Scaling by the largest value
    # instead (37.3 s at orbit_descent's release, the convention the altitude
    # and velocity scales above use) would read 0.13-0.34 through most of
    # every episode that is actually trained -- an order of magnitude too
    # small, which is the exact defect this module exists to remove. The
    # start-of-episode overshoot past 1.0 is the same transient the 30 m/s
    # velocity scale already tolerates at a fast release.
    10.0,
], dtype=np.float32)


def resolve_reward_weights(reward_fn, override=None):
    """The `RewardWeights` an env should build obs 16-19 from.

    Observation slots 16-19 are `reward.target_velocity` evaluated at the
    current state, so they are only meaningful if the env uses the SAME
    weights the reward function uses. `make_apollo_reward_fn` publishes them
    on the closure it returns, so the normal path needs no wiring at all; an
    explicit `override` wins, and a hand-written `reward_fn` that carries no
    weights falls back to the defaults.
    """
    from lunarsim.rl.reward import RewardWeights

    if override is not None:
        return override
    attached = getattr(reward_fn, "reward_weights", None)
    return attached if attached is not None else RewardWeights()


def guidance_obs(w, alt_m, dx, dy, vx, vy, vz) -> list[float]:
    """Observation slots 16-19: the velocity ERROR against the guidance field,
    plus `t_go`. ONE implementation, called by all three envs (and by the
    hand-built observation in `scripts/isaaclab_policy_eval_capture.py`), so
    the four slots cannot acquire different meanings in different envs.

    This follows Gaudet/Linares/Furfaro (arXiv:1810.08719), whose observation
    for 6-DOF powered descent carries the velocity error relative to the
    guidance field and the time-to-go rather than the raw state: the policy
    then closes a loop on an error that has already been computed for it,
    instead of having to rediscover the guidance law from position and
    velocity before it can act on it. The same quantity is what
    `reward._velocity_field_penalty` charges for, so the observation now shows
    the policy exactly the thing the reward grades.

    `alt_m` must be the env's OWN altitude definition -- the one behind
    `info["altitude_m"]` and the termination test -- so "altitude" keeps
    meaning one thing across the pipeline. `dx`/`dy` are the pad offsets
    (x - target_x, y - target_y); the current field ignores them (its lateral
    target is zero everywhere, see `target_velocity`), but they are passed
    through so that a pinpoint variant of the field would reach the
    observation without a second edit in three places.
    """
    from lunarsim.rl.reward import target_velocity

    vx_t, vy_t, vz_t, t_go = target_velocity(w, float(alt_m), float(dx), float(dy))
    return [float(vx) - vx_t, float(vy) - vy_t, float(vz) - vz_t, t_go]


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Divides the first `len(OBS_SCALE)` entries of `obs` by `OBS_SCALE`,
    elementwise. `obs` may have extra trailing entries (e.g. LiDAR ranges,
    see `AnalyticLanderEnv.use_lidar_obs`) which are passed through
    unscaled -- those are already roughly unit/small-integer scale.
    A SHORTER `obs` is scaled by the matching leading prefix of `OBS_SCALE`,
    which is what the explicit 16-wide legacy vintages in `IsaacLanderEnv`
    need; it is safe only because the layout is append-only -- the first 16
    scales still belong to the first 16 slots.
    Works on both a single observation (1D) and a batch (2D, one row per
    env, as `IsaacLanderVecEnv` builds).
    """
    obs = np.asarray(obs, dtype=np.float32)
    n = min(OBS_SCALE.shape[0], obs.shape[-1])
    scale = OBS_SCALE[:n]
    out = obs.copy()
    if obs.ndim == 1:
        out[:n] = out[:n] / scale
        return out
    out[:, :n] = out[:, :n] / scale
    return out
