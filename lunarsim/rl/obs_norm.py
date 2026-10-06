"""Fixed-scale observation normalization for the shared 16-element lander
observation layout (position offset x/y/z, vx/vy/vz, quaternion wxyz,
angular rate wx/wy/wz, fuel_frac, rcs_fuel_frac, time_remaining_frac) used
identically by `AnalyticLanderEnv`, `IsaacLanderEnv`, `IsaacLanderVecEnv`,
and `scripts/isaaclab_policy_eval_capture.py`'s hand-built observation.

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
], dtype=np.float32)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Divides the first `len(OBS_SCALE)` entries of `obs` by `OBS_SCALE`,
    elementwise. `obs` may have extra trailing entries (e.g. LiDAR ranges,
    see `AnalyticLanderEnv.use_lidar_obs`) which are passed through
    unscaled -- those are already roughly unit/small-integer scale.
    Works on both a single observation (1D) and a batch (2D, one row per
    env, as `IsaacLanderVecEnv` builds).
    """
    obs = np.asarray(obs, dtype=np.float32)
    n = OBS_SCALE.shape[0]
    if obs.ndim == 1:
        out = obs.copy()
        out[:n] = out[:n] / OBS_SCALE
        return out
    out = obs.copy()
    out[:, :n] = out[:, :n] / OBS_SCALE
    return out
