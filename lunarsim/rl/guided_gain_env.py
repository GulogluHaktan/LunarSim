"""RL modulates the ZEM/ZEV controller's gains; the controller still flies.

WHY THIS EXISTS, AND WHY IT IS NOT ANOTHER ALGORITHM SWAP

Every RL attempt in this project so far has asked a network to emit the env's
raw 4-vector `[throttle, pitch, roll, yaw]` at 20 Hz, warm-started from a
behaviour clone of `ZemZevController`. Measured facts about that setup, from
the checkpoints themselves rather than from theory:

  * `bc_ppo_orbit_v2.zip` carries `log_std = [-3.81, -3.01, -3.02, -5.59]`,
    i.e. sigma `[0.0221, 0.0493, 0.0490, 0.0037]` on a `[-1, 1]` box -- 1.1%,
    2.5%, 2.5% and 0.19% of the half-range. It is the BC fit's residual, not a
    chosen exploration scale, and the checkpoint was saved with `ent_coef=0.0`.
  * `ppo_v86_g999/.../1500000_steps.zip` carries sigma
    `[0.0221, 0.0494, 0.0491, 0.0037]` -- the SAME value to four decimals after
    1.5M steps at `ent_coef=0.003`, `lr=1e-5`. The exploration scale never
    moved, so every one of those 1.5M steps sampled a 2.5%-of-range
    perturbation.
  * The vehicle's attitude channel needs 6.3 s at FULL torque to reach 10 deg
    of tilt (RCS couple 1584 N*m, `I_tilt` 79,028 kg*m2 at PDI mass,
    `angular_damping_per_s = 0.5`), against a 0.05 s control step. One braking
    manoeuvre is therefore 250-380 consecutive commands, and independent
    per-step noise over that span cancels.

The last point is a theorem, not an observation. Tallec, Blier & Ollivier
(ICML 2019, arXiv:1901.09732) Thm 2: `Q_dt(s,a) = V_dt(s) + dt*A_dt(s,a)`, so
the action-ranking signal is O(dt) and any approximation error larger than that
destroys the ranking; their Thm 4: per-step i.i.d. exploration noise provably
collapses to a deterministic policy as dt -> 0. Park, Kim & Kim
(arXiv:2111.03941) extend the divergence to the policy-gradient variance, which
is why moving from SAC to PPO did not help. Wagenmaker et al.
(arXiv:2512.16911) prove standard BC can fail the action-COVERAGE condition
that is minimally necessary for RL fine-tuning -- 1.2% of the box is not
coverage.

So the fix is not a better estimator on the same action space. It is a
different action space, and the literature is unusually specific about which:

  * Furfaro, Scorsoglio, Linares & Massari (Acta Astronautica 2020,
    arXiv:2003.02182) keep the ZEM/ZEV law and learn its GAINS -- "we call
    policy specifically K_R, K_V and T_f as function of the state" -- held
    constant over each interval. Same guidance law this project already has.
  * Kaufmann, Bauersfeld & Scaramuzza (arXiv:2202.10796) run the decisive
    action-space ablation on a quadrotor at matched compute: collective thrust
    + body rates (an inner loop underneath) 0.6-12.3 cm tracking error, direct
    actuator commands undeployable on hardware, linear velocity 7.0-197.9 cm.
    "Policies that operate in concert with an underlying low-level controller
    outperform end-to-end policies."
  * Jiang et al. (arXiv:2407.15083) take ROCKET LANDING from 8% to 97% by
    using the analytic controller as a guide policy rather than as a BC target.
  * Metelli et al. (ICML 2020, arXiv:2002.06836) give the persistence/
    zero-order-hold formulation and a way to pick the hold length.

WHAT THIS BUYS, CONCRETELY

  1. `action = 0` reproduces the controller EXACTLY (the mapping is
     `gain * ratio**action`, so a zero action is a multiplier of 1.0). The
     floor is the controller's own landing rate by construction -- not a
     52.6% imitation of it with no action coverage.
  2. The controller's inner attitude loop (`kp_att`, `kd_att` on tilt error)
     and its carried `tgo` integrator stay inside the loop. The policy never
     has to learn the four-integration torque -> lateral-velocity chain, and
     the hidden-state problem that makes the controller unclonable
     (`zemzev_controller.py`'s `stateless_tgo` note) does not arise, because
     the controller runs rather than being imitated.
  3. `hold_steps` decisions instead of 1200. At the default 20 (1.0 s) an
     orbit_descent episode is ~36 decisions, so one decision spans a real
     fraction of the 6.3 s attitude timescale and the credit assignment
     distance collapses by the same factor.
  4. Exploration is automatically temporally coherent: a sampled gain applies
     for the whole hold. This is the PPO-compatible form of the correlated
     noise that Thm 4 above says is required.

WHAT IT DOES NOT DO. It cannot exceed what the ZEM/ZEV structure can express:
if the controller's failures need a manoeuvre its law cannot produce at any
gain, this ceilings out. `GAIN_SPEC` is therefore chosen from the measured
failure mode -- 94% of touchdown failures bind on lateral velocity, median
1.39x over the 1.2 m/s limit -- so the knobs exposed are the ones the
controller's own source comments identify as governing exactly that: the tilt
taper that gives the slow attitude loop its lead time, the final-approach
trigger and its horizontal cleanup gains. An optional small residual on the
controller's own output (`residual_scale`) is available for the case where
gains alone plateau; it defaults to 0.0, i.e. off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lunarsim.control.zemzev_controller import ZemZevController, ZemZevGains
from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv


@dataclass(frozen=True)
class GainKnob:
    """One modulated gain. `action = 0` must map to `nominal` exactly.

    The map is geometric -- `nominal * ratio**action` -- for two reasons: it
    cannot produce a non-positive gain however the policy saturates, and it
    makes the policy's output symmetric in the thing that actually matters
    (halving vs doubling a gain), which a linear map does not.
    """
    name: str
    ratio: float          # action=+1 -> nominal*ratio, action=-1 -> nominal/ratio

    def value(self, nominal: float, action: float) -> float:
        return float(nominal * (self.ratio ** float(np.clip(action, -1.0, 1.0))))


# Chosen from the measured failure mode, not swept blind. 94% of touchdown
# failures bind on `v_xy` against the 1.2 m/s limit (median 1.39x), and
# `zemzev_controller.py`'s own REAL-BUG comments name these as the knobs that
# govern it:
#
#   tilt_taper_v_xy_m_s      "forces the guidance to start leveling out while
#                             there's still speed left to use it on, giving the
#                             slow attitude loop the lead time it needs"
#   max_tilt_cmd_rad          braking authority vs. how long the tilt takes to
#                             unwind; the source records 45 deg -> 10% landed,
#                             30 deg -> same rate but lower worst-case v_xy
#   final_approach_alt_m,     when ZEM/ZEV is dropped for the vertical phase;
#   final_approach_trigger_   a fixed trigger was measured to engage at ~8 m
#     margin                  while still descending at 9-10 m/s
#   kd_xy_final,              the final horizontal cleanup, which the source
#   max_tilt_cmd_final_rad    says is what stops carrying drift to touchdown
#                             (measured up to 9.6 m/s against a 1.2 m/s limit)
#   kp_vz_final               vertical authority in that same phase
#
# Ratios are deliberately modest. A wide box would let the policy walk the
# controller out of the region its own feasibility sweep certified, and the
# whole point of this env is that the zero action is already competent.
GAIN_SPEC: tuple[GainKnob, ...] = (
    GainKnob("tilt_taper_v_xy_m_s", 2.0),
    GainKnob("max_tilt_cmd_rad", 1.5),
    GainKnob("final_approach_alt_m", 2.0),
    GainKnob("final_approach_trigger_margin", 1.6),
    GainKnob("kd_xy_final", 2.5),
    GainKnob("max_tilt_cmd_final_rad", 1.6),
    GainKnob("kp_vz_final", 2.0),
    GainKnob("tgo_floor_s", 1.6),
)

# `GAIN_SPEC` deliberately leaves the INNER ATTITUDE LOOP alone, because on
# orbit_descent the lateral time constant is set by the taper and the loop is
# saturated for most of the braking phase anyway (the controller's own comment:
# "the command saturates at +-1 for nearly the whole transient regardless of
# kp"). On `land_handoff` that reasoning does not hold and the measured plateau
# says so: a run with the spec above moved 6.2% -> 14.6% and then sat there for
# 100k decisions.
#
# The reason is the entry distribution. `land_handoff` releases with
# `spawn_tilt_rad = (-0.26, 0.26)` sampled INDEPENDENTLY per axis -- so total
# tilt reaches 0.26*sqrt(2) = 21.1 deg -- and `spawn_w_rad_s = (-0.15, 0.15)`,
# i.e. already rotating, from 8-25 m with up to -5 m/s of sink. That is an
# attitude-recovery problem before it is a guidance problem, and the gains that
# govern recovery from a small tilt WITH a rate are exactly the two the spec
# above does not expose. `kd_att` damps the entry rate and `kp_att` sets how
# fast the tilt error is driven out; neither is saturated at these magnitudes,
# which is precisely the regime the "kd barely matters" note excludes.
GAIN_SPEC_ATTITUDE: tuple[GainKnob, ...] = GAIN_SPEC + (
    GainKnob("kp_att", 2.0),
    GainKnob("kd_att", 4.0),
    GainKnob("thrust_authority_margin", 1.15),
)

GAIN_SETS: dict[str, tuple[GainKnob, ...]] = {
    "lateral": GAIN_SPEC,
    "attitude": GAIN_SPEC_ATTITUDE,
}


# Extra observation slots this env appends to the base env's 20, so the policy
# can see the guidance state it is modulating. Scales follow `obs_norm`'s
# convention of keeping every input near unit magnitude.
#   0  tgo / 30 s            the controller's CARRIED integrator (not the
#                            stateless `target_velocity` t_go in base slot 18)
#   1  in_final_approach     1.0 once the ZEM/ZEV phase has been dropped
#   2  v_xy / 1.2            lateral speed in units of the limit that decides
#                            94% of failures -- so "am I over" is one input,
#                            not something to be inferred from two
_EXTRA_OBS = 3


def _nominal(gains: ZemZevGains, name: str) -> float:
    return float(getattr(gains, name))


class GuidedGainEnv(gym.Env):
    """Wraps `AnalyticLanderEnv`. The policy sets ZEM/ZEV gains; the
    controller produces the env's actions.

    One `step` here advances the underlying env by up to `hold_steps` control
    steps, re-running the controller (and so the attitude loop and the `tgo`
    integrator) at the FULL 20 Hz inside the hold, while the gains stay fixed.
    That is the zero-order-hold / action-persistence construction, not an
    action repeat: the low-level command still changes every 0.05 s, only the
    DECISION is held.

    The hold breaks on termination, so the reward and the terminal state belong
    to the control step they actually happened on. That is the failure mode the
    handover warns about for a `VecEnvWrapper` action repeat -- a wrapper keeps
    stepping a terminated env and discards the first k steps of the next
    episode, which are the release states. Here there is no wrapper and no
    auto-reset inside the hold.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tile=None,
        params=None,
        reward_fn=None,
        tile_fn: Optional[Callable] = None,
        reward_weights=None,
        gain_spec: Sequence[GainKnob] = GAIN_SPEC,
        base_gains: Optional[ZemZevGains] = None,
        hold_steps: int = 20,
        residual_scale: float = 0.0,
        max_steps: Optional[int] = None,
        seed: int | None = None,
        aim_scale: float = 0.0,
        hazard_obs: bool = False,
    ):
        super().__init__()
        self.aim_scale = float(aim_scale)      # m/s per unit of the two extra (aim) action dims
        self.hazard_obs = bool(hazard_obs)
        self.env = AnalyticLanderEnv(
            tile=tile, params=params, reward_fn=reward_fn, tile_fn=tile_fn,
            reward_weights=reward_weights, seed=seed)
        self.gain_spec = tuple(gain_spec)
        self.base_gains = base_gains or ZemZevGains()
        self.hold_steps = int(hold_steps)
        self.residual_scale = float(residual_scale)
        self.params = self.env.params
        self.max_steps = int(max_steps) if max_steps is not None \
            else int(round(self.params.max_episode_s / self.params.dt_s)) + 1

        valid = {f.name for f in fields(ZemZevGains)}
        bad = [k.name for k in self.gain_spec if k.name not in valid]
        if bad:
            raise ValueError(f"not ZemZevGains fields: {bad}")
        self._nominals = np.array(
            [_nominal(self.base_gains, k.name) for k in self.gain_spec], dtype=np.float64)

        self.controller = ZemZevController(self.params, self.base_gains)

        n_gain = len(self.gain_spec)
        n_act = n_gain + (4 if self.residual_scale > 0.0 else 0) + (2 if self.aim_scale > 0.0 else 0)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), dtype=np.float32)
        n_obs = int(self.env.observation_space.shape[0]) + _EXTRA_OBS + (9 if self.hazard_obs else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), dtype=np.float32)

        self._base_obs = None
        self._steps = 0
        self._in_final = False

    # -- observation ------------------------------------------------------

    def _augment(self, base_obs: np.ndarray) -> np.ndarray:
        s = self.env.state
        tgo = self.controller.tgo
        v_xy = math.hypot(float(s["vx"]), float(s["vy"]))
        extra = np.array([
            0.0 if tgo is None else float(tgo) / 30.0,
            1.0 if self._in_final else 0.0,
            v_xy / max(self.params.safe_landing_v_xy_m_s, 1e-6),
        ], dtype=np.float32)
        parts = [np.asarray(base_obs, dtype=np.float32), extra]
        if self.hazard_obs:
            from lunarsim.core.terrain.hazard import attach_hazard
            from lunarsim.evo.hazfeat import haz_features, safe_dir
            tl = attach_hazard(self.env.tile)
            parts.append(haz_features(s["x"], s["y"], s["vx"], s["vy"], tl.haz_dist, tl.res_m).astype(np.float32))
            parts.append(safe_dir(s["x"], s["y"], tl.haz_dist, tl.res_m).astype(np.float32))
        return np.concatenate(parts)

    # -- gym API ----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._base_obs, info = self.env.reset(seed=seed, options=options)
        self.controller.reset()
        self.controller.g = self.base_gains
        self._steps = 0
        self._in_final = False
        return self._augment(self._base_obs), info

    def _gains_from_action(self, action: np.ndarray) -> ZemZevGains:
        upd = {}
        for i, knob in enumerate(self.gain_spec):
            upd[knob.name] = knob.value(float(self._nominals[i]), float(action[i]))
        return replace(self.base_gains, **upd)

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        n_gain = len(self.gain_spec)
        self.controller.g = self._gains_from_action(a[:n_gain])
        resid = (a[n_gain:n_gain + 4] * self.residual_scale) \
            if self.residual_scale > 0.0 else None
        if self.aim_scale > 0.0:
            from lunarsim.core.terrain.hazard import attach_hazard
            from lunarsim.evo.hazfeat import aim_point
            ai = np.clip(a[-2:], -1.0, 1.0) * self.aim_scale
            tl = attach_hazard(self.env.tile); s_ = self.env.state
            self.controller.aim_pos = aim_point(s_["x"], s_["y"], tl.haz_dist, tl.res_m, (float(ai[0]), float(ai[1])))
            self.controller.pos_fade_alt_m = 30.0
            self.controller.final_pos_kp = 0.05

        total = 0.0
        term = trunc = False
        info: dict = {}
        for _ in range(self.hold_steps):
            # The controller reads `env.state` directly and carries `tgo`, so it
            # must be called every control step -- that is what keeps the inner
            # attitude loop closed at 20 Hz while the gains are held.
            cmd = self.controller.act(self.env)
            if resid is not None:
                cmd = np.clip(np.asarray(cmd, dtype=np.float64) + resid, -1.0, 1.0)
            self._base_obs, rew, term, trunc, info = self.env.step(
                np.asarray(cmd, dtype=np.float32))
            total += float(rew)
            self._steps += 1
            # `act()` sets tgo=None on entering the final-approach branch and
            # re-arms if the vehicle bounces back up, so this tracks the phase
            # rather than latching it.
            self._in_final = self.controller.tgo is None
            if term or trunc or self._steps >= self.max_steps:
                trunc = trunc or (not term and self._steps >= self.max_steps)
                break

        return self._augment(self._base_obs), total, bool(term), bool(trunc), info

    def close(self):
        self.env.close()
