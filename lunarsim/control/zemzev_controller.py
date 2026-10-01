"""A hand-designed (non-RL) guidance + attitude control law for
`AnalyticLanderEnv`, purpose-built to answer one question: **is a safe
landing physically achievable from a given release condition (altitude,
horizontal speed) with THIS vehicle model (real Apollo LM thrust/RCS specs)
and THIS environment (gravity, angular damping, safe-landing thresholds)** --
independent of whether the RL reward function/curriculum can currently teach
a policy to find it.

If this controller lands safely, the mission is control-theoretically
possible and any remaining failure in the RL pipeline is a reward-shaping /
training problem. If this controller ALSO can't land safely (even after
gain tuning), that's evidence of a genuine control-authority ceiling
(thrust-to-weight, RCS torque, or flight-time budget) that no amount of
reward-shaping can fix -- see `handover.md` point 4 under "ideas not yet
tried".

RESULT (see `scripts/test_landing_feasibility.py`, n=60/stage): hover_only*
100%, final_approach 48% (limited almost entirely by uneven-terrain leg
height diff, not velocity -- a site-selection problem, out of scope for a
pure translational/attitude controller), orbit_descent (200m release,
10-30 m/s horizontal, real terrain) **75% landed_safely**, zero crashes
(`lost_control`) in either stage, touchdown fuel remaining >95% throughout
-- i.e. the mission IS control-theoretically achievable with this vehicle
model; the remaining ~25% orbit_descent misses are marginal (touchdown
vxy/tilt a factor of ~1.5-4x over threshold, not order-of-magnitude), not
catastrophic. Getting there took fixing FOUR real, distinct bugs (three in
this guidance law, one pure gain-tuning finding) -- all documented in
place below where they were found:
1. Double-counting gravity in the ZEM/ZEV-to-thrust conversion.
2. Unbounded multiplicative tgo drift (`_solve_feasible_tgo`).
3. Sizing throttle off the desired accel vector's norm instead of the
   vehicle's ACTUAL current attitude (`_throttle_from_vertical_need`).
4. The one that mattered most: the feasibility search only checked thrust
   MAGNITUDE against engine limits, never that the required vertical
   accel was non-negative -- an achievable-norm command can still demand
   physically-impossible "thrust downward" if `tgo` is shorter than the
   natural free-fall arrival time, and the old check happily accepted it
   (see `_solve_feasible_tgo`'s `min_a_cmd_z` condition). Fixing this alone
   took orbit_descent from 5% to 73% landed_safely.
Plus one pure tuning finding: the attitude PD's `kp_att` was low enough
that a 30 deg tilt step settled in ~36s when the vehicle's own physical
limit (RCS torque vs. inertia vs. passive angular damping) is ~14s --
raising it to saturate the command through most of the transient (a
near-bang-bang system at this torque-to-inertia ratio) alone took
orbit_descent from 10% to 43%.

Guidance law: ZEM/ZEV (Zero-Effort-Miss / Zero-Effort-Velocity) feedback
guidance (D'Souza 1997; the same family used in real fuel-optimal planetary
lander G&C, e.g. ALHAT/Morpheus). Given current relative position/velocity
to the landing point, gravity, and an estimated time-to-go `tgo`, it computes
the (unique, minimum-acceleration) constant+linear acceleration command that
drives the vehicle to zero relative position AND zero velocity exactly at
`tgo` seconds from now:

    ZEM = -(r + v*tgo + 0.5*g_vec*tgo**2)      # miss if no more control applied
    ZEV = -(v + g_vec*tgo)                     # velocity miss if no more control applied
    a_cmd = 6*ZEM/tgo**2 - 2*ZEV/tgo

`tgo` is not derivable from state alone (it's a guidance design choice, not
a physical quantity) -- it starts from a kinematic estimate and is closed-loop
counted down every step, with a hard floor and an "if commanded thrust
saturates, back off tgo" adaptation to avoid demanding accelerations beyond
what the DPS can actually deliver (that saturation-driven back-off number is
itself useful feasibility evidence: a run where tgo keeps ballooning while
altitude keeps shrinking is a policy racing the ground and losing).

Attitude is a simple PD (no gimbal on the DPS -- direction is set purely by
body tilt, tilt is purely RCS-torque-controlled, exactly as this vehicle
model requires, see `analytic_lander_env`'s module docstring).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv, LanderParams


@dataclass
class ZemZevGains:
    # REAL BUG FOUND via this feasibility test: kp_att=2.5/kd_att=4.0 settled
    # a 30 deg step in ~36s -- far more than the true physical minimum. RCS
    # torque authority here is so low relative to inertia (max angular
    # accel ~0.02 rad/s^2 at full mass) that the command saturates at +-1
    # for nearly the whole transient regardless of kp once kp is large
    # enough to reach saturation quickly -- a step-response sweep found the
    # rise time floors at ~14s for a 30 deg step once kp>=6, i.e. this is a
    # near-bang-bang system and the old kp was simply too low to reach that
    # floor. kd barely matters (cmd stays saturated most of the time); a
    # small value just damps the final few degrees of settling.
    kp_att: float = 8.0
    kd_att: float = 1.0      # angular-rate damping gain (on top of the env's own passive damping)
    kp_yaw: float = 1.5
    kd_yaw: float = 2.0
    # 30 deg (not the 60 deg loss-of-control cutoff) is the empirically
    # better choice found via this feasibility sweep: the limiting factor
    # for orbit_descent isn't fuel or thrust (touchdown fuel remaining was
    # consistently >95%) -- it's how long the RCS takes to UNWIND a large
    # tilt again once velocity is nulled (see the tilt_taper comment below).
    # A smaller max tilt takes longer to kill 10-30 m/s of horizontal speed,
    # but there's plenty of spare flight time (touchdown at ~35-40s of a
    # 60s episode budget), and the smaller tilt is cheap to unwind, which
    # empirically wins: max_tilt=45deg -> 10% landed_safely / vxy up to
    # 7.5 m/s at touchdown; max_tilt=30deg -> same order success rate but
    # noticeably lower worst-case touchdown vxy (~5.7 vs ~7.5 m/s).
    max_tilt_cmd_rad: float = math.radians(30.0)
    tilt_taper_v_xy_m_s: float = 12.0              # full tilt only above this v_xy; tapers linearly below it
    thrust_authority_margin: float = 0.85          # back off tgo once demand exceeds this fraction of a_max
    min_a_cmd_z: float = 0.3                       # require some real margin above "just barely non-negative"
    tgo_floor_s: float = 3.0
    tgo_relax_factor: float = 1.08
    final_approach_alt_m: float = 8.0              # below this: drop ZEM/ZEV, hold a gentle vertical descent
    # margin multiplier on the kinematic stopping distance used to widen
    # the final-approach trigger altitude when vz is still high at the
    # fixed floor above -- see the REAL BUG note at its use site in act().
    final_approach_trigger_margin: float = 1.5
    final_approach_vz_m_s: float = -0.8            # under every stage's safe_landing_v_z_m_s (>=1.0) with margin
    kp_vz_final: float = 1.5
    # final-approach horizontal cleanup: independent small-angle PD on
    # relative position/velocity, NOT just "hold level" -- a "hold level"
    # final phase was found (via this very feasibility test) to silently
    # carry over whatever horizontal drift the ZEM/ZEV phase hadn't fully
    # cancelled yet, landing at up to 9.6 m/s horizontal (vs. a 1.2 m/s
    # limit) despite a perfectly good vertical touchdown -- the vertical and
    # horizontal channels must both stay closed-loop all the way to contact.
    kd_xy_final: float = 0.7
    max_tilt_cmd_final_rad: float = math.radians(20.0)


class ZemZevController:
    """Stateful (holds `tgo` across steps) guidance+attitude controller for
    one `AnalyticLanderEnv` episode. Call `reset()` at episode start, then
    `act(env)` once per step to get the 4-vector action `env.step` expects.
    """

    def __init__(self, params: LanderParams, gains: ZemZevGains | None = None):
        self.p = params
        self.g = gains or ZemZevGains()
        self.tgo: float | None = None

    def reset(self) -> None:
        self.tgo = None

    def _mass_kg(self, s: dict) -> float:
        return self.p.dry_mass_kg + s["fuel_kg"]

    @staticmethod
    def _throttle_from_vertical_need(a_z_thrust_desired: float, tilt_x: float, tilt_y: float,
                                      mass_kg: float, dps_min_n: float, dps_max_n: float) -> float:
        """Thrust magnitude (N) needed to hit a desired vertical thrust-accel
        component GIVEN THE VEHICLE'S ACTUAL CURRENT ATTITUDE (not the
        attitude it's steering toward). REAL BUG FOUND via this feasibility
        test: sizing thrust off the desired 3D acceleration vector's norm
        (which implicitly assumes the vehicle is ALREADY at the target
        tilt) made the controller apply near-full thrust while still
        pointed mostly straight up during the ~10-15s the slow RCS attitude
        loop took to rotate toward a 35-45 deg braking tilt -- thrust_z
        during that whole transient stayed close to the full thrust
        magnitude (since actual tilt was still near zero), producing
        sustained net-positive vertical acceleration that flew the vehicle
        away from the ground instead of descending, even though the
        guidance law's OWN target vertical acceleration was negative
        (wanted to fall faster, not climb). Decoupling throttle from the
        attitude command like this -- solve for whatever thrust makes the
        vertical channel correct given wherever the vehicle actually is
        pointed right now -- keeps vertical velocity under control
        independent of how fast attitude happens to be converging.

        The DPS can only push along body -Z, i.e. it can never produce a
        negative thrust_z contribution (thrust always exits the bell in the
        vehicle's own "up" direction, whichever way that's currently
        tilted) -- so a negative desired vertical accel is floored to 0
        (closest achievable: engine off, free-fall at -g).
        """
        cos_prod = max(math.cos(tilt_x) * math.cos(tilt_y), 0.15)
        a_z_floored = max(a_z_thrust_desired, 0.0)
        thrust_mag_n = mass_kg * a_z_floored / cos_prod
        return float(np.clip(thrust_mag_n, dps_min_n, dps_max_n))

    def _a_max(self, mass_kg: float) -> float:
        """Net deceleration authority available: full DPS thrust minus what
        gravity always takes back."""
        return self.p.dps_thrust_max_n / mass_kg - self.p.gravity_m_s2

    @staticmethod
    def _a_cmd(relx: float, rely: float, alt: float, vx: float, vy: float, vz: float,
               grav: float, tgo: float) -> np.ndarray:
        """Vertical channel: full D'Souza ZEM/ZEV feedback guidance to the
        ground plane (position AND velocity both matter there -- touchdown
        must happen at z=ground with vz within the safe limit). Gravity is
        already folded into the ZEM/ZEV terms via `g_vec`, so the caller
        must NOT add gravity again (verified via the tgo -> infinity
        asymptote, which must reduce to a pure hover command a_cmd_z ->
        +grav -- adding +grav on top of that was a real bug caught during
        feasibility testing: it made the controller demand 2x the thrust
        needed at every altitude, saturating the DPS and flying the vehicle
        away from the ground instead of down to it).

        Horizontal channel: velocity-only damping (`-2*v/tgo`), deliberately
        NOT the full ZEM/ZEV (which would also chase `relx,rely` back to
        the exact `target_x,target_y`). `landed_safely` in
        `analytic_lander_env` never checks touchdown position, only
        velocity/tilt/rate -- REAL BUG FOUND via this feasibility test: with
        the position term included, the guidance kept commanding several
        degrees of tilt in the last few meters to close out a lateral
        position error that didn't matter, and because RCS attitude
        response is slow (~0.02 rad/s^2 per unit torque-duty-cycle at full
        mass -- see `analytic_lander_env`'s angular_damping comment), that
        leftover tilt didn't unwind before touchdown and re-accelerated
        the vehicle sideways in the final seconds, undoing an
        already-nulled horizontal velocity.
        """
        zem_z = -(alt + vz * tgo - 0.5 * grav * tgo ** 2)
        zev_z = -(vz - grav * tgo)
        return np.array([
            -2.0 * vx / tgo,
            -2.0 * vy / tgo,
            6.0 * zem_z / tgo ** 2 - 2.0 * zev_z / tgo,
        ])

    def _solve_feasible_tgo(self, relx: float, rely: float, alt: float, vx: float, vy: float, vz: float,
                             grav: float, engine_a_max: float, tgo0: float) -> tuple[float, np.ndarray]:
        """Geometric search (not unbounded multiplicative drift -- an earlier
        version multiplied `self.tgo` by the relax factor every single
        control step whenever saturated, which compounds exponentially over
        hundreds of steps and blew tgo up to ~1e17; this instead re-searches
        from the current best estimate each call, so it can only grow as
        far as this call's own loop, and it also shrinks back down as soon
        as the state makes a shorter tgo feasible again) for the smallest
        tgo (starting from `tgo0`) whose required thrust accel stays within
        `thrust_authority_margin` of what the engine can actually deliver.
        """
        tgo = max(tgo0, self.g.tgo_floor_s)
        a_cmd = self._a_cmd(relx, rely, alt, vx, vy, vz, grav, tgo)
        for _ in range(60):
            # REAL BUG FOUND via this feasibility test: this only checked
            # the vector's MAGNITUDE against the engine's max thrust -- it
            # never checked that a_cmd_z was even the right SIGN. If `tgo`
            # is shorter than the natural free-fall arrival time for the
            # current altitude/vz, the ZEM/ZEV closed form calls for a
            # NEGATIVE thrust accel (falling faster than gravity alone --
            # impossible for a single fixed-direction engine that can only
            # push "up"). That negative-z solution can still have a small
            # enough NORM to pass the old magnitude-only check (its x/y
            # components are small), so the search would happily accept a
            # physically-impossible vertical command and declare victory --
            # producing a ~zero-throttle "coast" that let vz reach -16 m/s
            # over 11s before the guidance ever noticed. Requiring a_cmd_z
            # to clear a small positive floor forces `tgo` to grow past the
            # free-fall-consistent threshold before it's accepted.
            feasible = (
                float(np.linalg.norm(a_cmd)) <= self.g.thrust_authority_margin * engine_a_max
                and a_cmd[2] >= self.g.min_a_cmd_z
            )
            if feasible:
                return tgo, a_cmd
            tgo *= self.g.tgo_relax_factor
            a_cmd = self._a_cmd(relx, rely, alt, vx, vy, vz, grav, tgo)
        return tgo, a_cmd  # best effort -- genuinely infeasible within the search cap

    def act(self, env: AnalyticLanderEnv) -> np.ndarray:
        p, g = self.p, self.g
        s = env.state
        dt = p.dt_s

        ground_z = env._ground_z(s["x"], s["y"])
        alt = s["z"] - ground_z
        relx, rely = s["x"] - p.target_x, s["y"] - p.target_y
        vx, vy, vz = s["vx"], s["vy"], s["vz"]
        v_xy = math.hypot(vx, vy)
        grav = p.gravity_m_s2

        mass_kg = self._mass_kg(s)
        engine_a_max = self.p.dps_thrust_max_n / mass_kg  # absolute max |thrust accel| the DPS can give

        # -- final approach override: once low enough, stop chasing a ZEM/ZEV
        # solution (which is singular as tgo -> 0) and just hold level
        # attitude + a slow constant sink rate to touchdown. --
        # REAL BUG FOUND via a real Isaac Sim capture (not just the analytic
        # simulator): a FIXED trigger altitude implicitly assumes vz is
        # already small by the time alt crosses it. It usually is (that's
        # why this passed 75-80% of analytic Monte Carlo runs) -- but a real
        # run was observed entering this branch at ~8m altitude while still
        # descending at ~9-10 m/s (the main-phase ZEM/ZEV hadn't fully
        # reined in vz yet), which this branch's kp_vz_final=1.5 gain simply
        # cannot arrest within 8m of remaining altitude -- it hit hard,
        # bounced, and picked up a violent tumble on the rebound. Widening
        # the trigger altitude to the actual kinematic stopping distance
        # needed (with a safety margin), not a fixed constant, means this
        # branch engages EARLIER whenever vz is unexpectedly high, while
        # leaving the common (already-slow-by-8m) case untouched (stopping
        # distance is tiny there, so the max() falls back to the fixed
        # floor).
        a_max_net_for_trigger = max(self._a_max(mass_kg), 0.3)
        stopping_alt = (vz ** 2) / (2.0 * a_max_net_for_trigger) * g.final_approach_trigger_margin if vz < 0 else 0.0
        final_approach_trigger_alt = max(g.final_approach_alt_m, stopping_alt)
        if alt <= final_approach_trigger_alt:
            self.tgo = None  # re-arm guidance if we ever bounce back up
            ax_des = -g.kd_xy_final * vx  # velocity-only, see _a_cmd's docstring on why position doesn't matter
            ay_des = -g.kd_xy_final * vy
            tilt_y_des = float(np.clip(math.atan2(ax_des, grav), -g.max_tilt_cmd_final_rad, g.max_tilt_cmd_final_rad))
            tilt_x_des = float(np.clip(math.atan2(-ay_des, grav), -g.max_tilt_cmd_final_rad, g.max_tilt_cmd_final_rad))
            a_needed_z = grav + g.kp_vz_final * (g.final_approach_vz_m_s - vz)  # a_total + grav = a_thrust
            thrust_mag_n = self._throttle_from_vertical_need(
                a_needed_z, s["tilt_x"], s["tilt_y"], mass_kg, p.dps_thrust_min_n, p.dps_thrust_max_n)
        else:
            if self.tgo is None:
                a_max_net = max(self._a_max(mass_kg), 0.05)
                # REAL BUG FOUND via this feasibility test: orbit_descent
                # spawns with spawn_v_z_m_s=0.0 (released, not yet falling),
                # which hit the old `else` branch (`alt / (0.3*a_max_net)`)
                # and produced tgo=489s for a 200m/~19 m/s release -- the
                # guidance then spent the ENTIRE 60s episode coasting at
                # near-hover throttle (a_cmd scales with 1/tgo, so a huge
                # tgo means a near-zero command), never meaningfully
                # descending OR braking, timing out at 193m altitude having
                # bled off only ~3.5 m/s of horizontal speed. Free-fall time
                # from rest (`sqrt(2*alt/g)`) is the right estimate for a
                # non-descending release; a horizontal-kill-time estimate is
                # needed unconditionally too (the old code only considered
                # it when already falling).
                tgo_vert = (2.0 * alt / max(-vz, 1.0)) if vz < 0 else math.sqrt(2.0 * alt / max(grav, 0.1))
                tgo_horiz = v_xy / max(0.6 * a_max_net, 0.1)
                self.tgo = max(tgo_vert, tgo_horiz, dt * 5)
            tgo0 = max(self.tgo - dt, g.tgo_floor_s)

            tgo, a_thrust = self._solve_feasible_tgo(relx, rely, alt, vx, vy, vz, grav, engine_a_max, tgo0)
            self.tgo = tgo

            # attitude target: direction of the desired accel vector. The DPS
            # can't produce a negative thrust_z, so floor the z component
            # (at a small positive epsilon, not 0, to keep asin well-defined)
            # purely for picking a direction to steer toward -- the actual
            # thrust MAGNITUDE is decided below from the real current tilt,
            # not from this vector's norm (see `_throttle_from_vertical_need`).
            dir_input = np.array([a_thrust[0], a_thrust[1], max(a_thrust[2], 1e-3)])
            dir_norm = float(np.linalg.norm(dir_input))
            dirvec = dir_input / dir_norm if dir_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
            tilt_y_des = math.asin(float(np.clip(dirvec[0], -0.95, 0.95)))
            tilt_x_des = math.asin(float(np.clip(-dirvec[1], -0.95, 0.95)))
            # REAL BUG FOUND via this feasibility test: a fixed tilt cap let
            # the guidance hold ~35 deg of tilt (needed to kill 25-30 m/s of
            # horizontal speed) for so long that by the time v_xy actually
            # reached ~0, the RCS (very low torque authority -- unwinding
            # 35 deg takes several seconds even at max commanded rate) was
            # still many seconds from leveling out -- the still-tilted
            # thrust vector then kept accelerating the vehicle in the SAME
            # direction well past the zero-velocity point, re-accelerating
            # it back up to 5-8 m/s by touchdown despite a clean null a few
            # seconds earlier. Tapering the allowed tilt down with v_xy
            # itself (not with time or altitude) forces the guidance to
            # start leveling out while there's still speed left to using it
            # on, giving the slow attitude loop the lead time it needs.
            dynamic_max_tilt = g.max_tilt_cmd_rad * float(np.clip(v_xy / g.tilt_taper_v_xy_m_s, 0.1, 1.0))
            tilt_x_des = float(np.clip(tilt_x_des, -dynamic_max_tilt, dynamic_max_tilt))
            tilt_y_des = float(np.clip(tilt_y_des, -dynamic_max_tilt, dynamic_max_tilt))

            thrust_mag_n = self._throttle_from_vertical_need(
                float(a_thrust[2]), s["tilt_x"], s["tilt_y"], mass_kg, p.dps_thrust_min_n, p.dps_thrust_max_n)

        pitch_cmd = g.kp_att * (tilt_x_des - s["tilt_x"]) - g.kd_att * s["wx"]
        roll_cmd = g.kp_att * (tilt_y_des - s["tilt_y"]) - g.kd_att * s["wy"]
        yaw_cmd = -g.kp_yaw * s["yaw"] - g.kd_yaw * s["wz"]

        pitch_cmd = float(np.clip(pitch_cmd, -1.0, 1.0))
        roll_cmd = float(np.clip(roll_cmd, -1.0, 1.0))
        yaw_cmd = float(np.clip(yaw_cmd, -1.0, 1.0))

        throttle_norm = 2.0 * (thrust_mag_n - p.dps_thrust_min_n) / (p.dps_thrust_max_n - p.dps_thrust_min_n) - 1.0
        throttle_norm = float(np.clip(throttle_norm, -1.0, 1.0))

        return np.array([throttle_norm, pitch_cmd, roll_cmd, yaw_cmd], dtype=np.float32)
