"""A Gymnasium-compatible lunar lander environment driven entirely by
`lunarsim.core` (terrain + a real-Apollo-LM-spec rigid-body rocket model) --
no Isaac Sim required. This is the fast, CPU-only backend for RL algorithm
development/iteration; `lunarsim.adapters.isaac` has the (Isaac-Sim-backed,
physically simulated) heavy-fidelity path once real contact dynamics,
plume/dust, and camera/LiDAR sensing on the real render/physics engine are
wanted instead of this analytic approximation.

Dynamics match the real Apollo LM control split, not a generic gimballed
rocket:

- The Descent Propulsion System (DPS) is a single engine, fixed along the
  body's -Z axis, that only throttles thrust magnitude (`dps_thrust_min_n`
  .. `dps_thrust_max_n`). Its real +-6 deg gimbal exists purely for
  CG-offset trim and is not an RL-controlled actuator here.
- Attitude (roll/pitch/yaw) is controlled the way the real vehicle does
  it: 16 Reaction Control System (RCS) thrusters in 4 quads, each
  producing a fixed 445 N (100 lbf). The action commands a torque duty
  cycle per axis (as if firing a couple of jets); consumed RCS propellant
  is tracked in a separate tank from DPS propellant, exactly as on the
  real vehicle.
- Vehicle mass decreases as DPS propellant burns (`dry_mass_kg + fuel_kg`),
  which is what actually happens during a real descent (~15.1 t at PDI
  down to ~6.9 t dry) and materially changes thrust-to-weight and angular
  response over the course of an episode -- a fixed-mass approximation
  would get both wrong by more than 2x.

All real-vehicle numbers live in `lunarsim.core.vehicle.apollo_lm` so the
Isaac-side USD asset (`lunarsim/adapters/isaac/lander.py`) and this env can
never drift apart on what "the vehicle" weighs or how hard it can push.

Body tilt still redirects thrust the way it would on the real vehicle
(small-angle): the DPS points along the body's own -Z axis, and RCS-driven
attitude change is what tilts that axis, not the other way around. Ground
contact and crash/success classification use the SAME real terrain
heightmap `core.terrain` would export to Isaac, so an episode's landing
site is the same tile geometry either backend would see.

Observation matches the sensor suite this vehicle actually has: IMU
(attitude quaternion + angular velocity), a radar/LiDAR-style altitude,
DPS/RCS propellant remaining, and per-leg touchdown load -- see
`_observation`. `use_lidar_obs` appends a ring of downward-ish LiDAR
ranges on top of that for terrain-relative sensing.

`spawn_horizontal_speed_m_s` (a (low, high) range, default (0, 0)) sets an
initial horizontal speed aimed from the spawn point toward
`(target_x, target_y)`, which is what turns this into a curriculum-capable
env: a high-altitude, high-horizontal-speed spawn forces the policy to
brake and null out drift before it can descend, the way a real approach
phase does, without needing actual two-body orbital dynamics (this env's
terrain tile is a local flat patch, not a Moon-centered orbit) -- see
`scripts/train_ppo_curriculum.py` for the staged schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lunarsim.core.terrain.generate import Tile
from lunarsim.core.terrain.rocks import sample_height_at
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, G0, leg_force_bounds_n, moment_of_inertia
from lunarsim.rl.obs_norm import normalize_obs

_SPECS = ApolloLMSpecs()


@dataclass
class LanderParams:
    # mass (real Apollo LM breakdown at Powered Descent Initiation)
    dry_mass_kg: float = _SPECS.dry_mass_kg
    initial_fuel_kg: float = _SPECS.descent_propellant_kg

    # DPS (main descent engine): throttle-only, fixed along body -Z
    dps_thrust_min_n: float = _SPECS.dps_thrust_min_n
    dps_thrust_max_n: float = _SPECS.dps_thrust_max_n
    dps_isp_s: float = _SPECS.dps_isp_s

    # RCS (attitude control): torque duty-cycle per axis, separate propellant tank
    rcs_thrust_n: float = _SPECS.rcs_thruster_thrust_n
    rcs_jets_per_couple: int = _SPECS.rcs_jets_per_couple
    rcs_quad_radius_m: float = _SPECS.rcs_quad_radius_m
    rcs_isp_s: float = _SPECS.rcs_isp_s
    initial_rcs_fuel_kg: float = _SPECS.rcs_propellant_kg

    # geometry feeding the (engineering-estimate) inertia tensor and leg loads
    body_radius_m: float = _SPECS.body_radius_m
    body_height_m: float = _SPECS.height_m
    leg_count: int = _SPECS.leg_count
    leg_stroke_m: float = _SPECS.leg_stroke_m

    gravity_m_s2: float = 1.62
    dt_s: float = 0.05
    max_episode_s: float = 90.0

    # angular-rate damping: real spacecraft don't fly raw open-loop torque
    # commands straight to the RCS jets -- there's a stability-augmentation
    # inner loop (the real Apollo LM's Digital Autopilot did this too)
    # damping rotation rate around whatever the outer command asks for.
    # FOUND NECESSARY VIA A FAILED TRAINING RUN: without this, the attitude
    # dynamics are a pure, undamped double integrator (torque -> rate ->
    # angle, no decay term at all), which is unconditionally unstable under
    # naive position-only control -- exactly the failure mode observed
    # (every trained policy rang up to the loss-of-control cutoff, and a
    # direct probe of the policy showed it fighting its own restoring
    # rotation rather than damping it). This is a physical fidelity fix,
    # not an RL crutch: it reflects a control layer the real vehicle had
    # and this model was missing.
    angular_damping_per_s: float = 0.5

    # landing safety constraints.
    # REAL BUG FOUND (after two full training runs, 8/8 evaluated checkpoints
    # across both): `safe_landing_v_xy_m_s` was left at an ungrounded default
    # of 0.5 m/s -- not tied to any real Apollo LM number, unlike almost
    # every other constant in this module. Every single evaluated checkpoint
    # across two independent full curricula failed landing safety on THIS
    # term specifically (vxy ranged 1.78-3.99 m/s, never once under 2x the
    # limit), while vz and tilt each passed cleanly at least once -- a
    # single term failing 8/8 times while the others each succeed sometimes
    # is a threshold-calibration signal, not (only) an undertrained-policy
    # signal. The real LM landing gear's qualified touchdown envelope was
    # documented around 3.05 m/s (10 ft/s) vertical and 1.22 m/s (4 ft/s)
    # horizontal (Grumman/NASA LM-10 landing gear qualification figures) --
    # i.e. the REAL vehicle tolerated more horizontal drift than this sim
    # was demanding. Raised to match; vz left at 1.0 (already stricter than
    # the real 3.05 m/s bound, kept conservative since it was never the
    # actual bottleneck).
    safe_landing_v_z_m_s: float = 1.0
    safe_landing_v_xy_m_s: float = 1.2
    safe_landing_tilt_rad: float = np.deg2rad(15.0)
    safe_landing_w_rad_s: float = 0.5
    safe_landing_max_leg_height_diff_m: float = 0.16

    # loss-of-control cutoff: found necessary empirically (a training run
    # let an undertrained policy tumble to 200+ deg tilt and free-fall for
    # the rest of a 25s episode with no way back -- pure wasted steps that
    # also drowned the learning signal in shaping penalty from a state no
    # real vehicle could recover from anyway). Past this tilt, the episode
    # ends immediately as a crash rather than continuing to simulate an
    # already-lost vehicle.
    loss_of_control_tilt_rad: float = np.deg2rad(60.0)

    # landing target (relative-position reward/observation is measured against this)
    target_x: float = 0.0
    target_y: float = 0.0

    # spawn / curriculum
    spawn_altitude_m: float = 80.0
    spawn_xy_radius_m: float = 30.0
    spawn_v_z_m_s: float = -3.0
    spawn_horizontal_speed_m_s: tuple[float, float] = (0.0, 0.0)


RewardFn = Callable[["AnalyticLanderEnv", dict], float]


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Aerospace roll(x)-pitch(y)-yaw(z) Euler angles -> quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return float(qw), float(qx), float(qy), float(qz)


class AnalyticLanderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tile: Optional[Tile] = None,
        params: Optional[LanderParams] = None,
        reward_fn: Optional[RewardFn] = None,
        use_lidar_obs: bool = False,
        lidar_n_rays: int = 8,
        seed: int | None = None,
        tile_fn: Optional[Callable[[np.random.Generator], Tile]] = None,
    ):
        """`tile` is a fixed terrain tile reused for every episode (the
        original behavior, still the default for a single fixed map).
        Pass `tile_fn` instead (a callable taking this env's own RNG and
        returning a fresh `Tile`) to regenerate terrain -- a different
        seed, so a different patch of the Moon -- on every `reset()`;
        this is what curriculum/generalization training should use so the
        policy doesn't overfit to one map's craters/slopes. Exactly one of
        `tile`/`tile_fn` must be given.
        """
        super().__init__()
        if (tile is None) == (tile_fn is None):
            raise ValueError("AnalyticLanderEnv needs exactly one of `tile` or `tile_fn`")
        self.tile = tile
        self.tile_fn = tile_fn
        self.params = params or LanderParams()
        if reward_fn is None:
            from lunarsim.rl.reward import default_reward_fn
            reward_fn = default_reward_fn
        self.reward_fn = reward_fn
        self.use_lidar_obs = use_lidar_obs
        self.lidar_n_rays = lidar_n_rays
        self._rng = np.random.default_rng(seed)

        # [relx, rely, alt, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz,
        #  fuel_frac, rcs_fuel_frac, leg_force_frac]
        obs_dim = 16 + (lidar_n_rays if use_lidar_obs else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # [throttle, pitch_cmd, roll_cmd, yaw_cmd] all in [-1, 1]
        # throttle scales the DPS between dps_thrust_min_n/dps_thrust_max_n
        # (0 = engine off); the other three are RCS torque duty cycles.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.state: dict = {}
        self._t = 0.0

    def _ground_z(self, x: float, y: float) -> float:
        return float(sample_height_at(self.tile.height, self.tile.res_m, np.array([x]), np.array([y]))[0])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        p = self.params
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.tile_fn is not None:
            self.tile = self.tile_fn(self._rng)

        r = self._rng.uniform(0, p.spawn_xy_radius_m)
        theta = self._rng.uniform(0, 2 * np.pi)
        x0, y0 = r * np.cos(theta), r * np.sin(theta)
        ground_z0 = self._ground_z(x0, y0)

        speed = self._rng.uniform(*p.spawn_horizontal_speed_m_s)
        dx, dy = p.target_x - x0, p.target_y - y0
        dist = max(1e-6, float(np.hypot(dx, dy)))
        vx0, vy0 = speed * dx / dist, speed * dy / dist

        self.state = {
            "x": x0, "y": y0, "z": ground_z0 + p.spawn_altitude_m,
            "vx": vx0, "vy": vy0, "vz": p.spawn_v_z_m_s,
            "tilt_x": self._rng.uniform(-0.05, 0.05),
            "tilt_y": self._rng.uniform(-0.05, 0.05),
            "yaw": 0.0,
            "wx": 0.0, "wy": 0.0, "wz": 0.0,
            "fuel_kg": p.initial_fuel_kg,
            "rcs_fuel_kg": p.initial_rcs_fuel_kg,
            "throttle": 0.0,
            "rcs_pitch": 0.0, "rcs_roll": 0.0, "rcs_yaw": 0.0,
        }
        self._t = 0.0
        return self._observation(), {}

    def _footpad_height_diff_m(self, x: float, y: float) -> float:
        """Terrain-height spread under the 4 footpads at the current (x, y)
        -- an uneven landing spot is a landing-safety failure independent of
        vehicle velocity/tilt at contact."""
        radius = ApolloLMSpecs().footpad_span_m / 2.0
        heights = [
            self._ground_z(x + radius * np.cos(a), y + radius * np.sin(a))
            for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])
        ]
        return float(max(heights) - min(heights))

    def step(self, action: np.ndarray):
        p = self.params
        s = self.state

        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        pitch_cmd = float(np.clip(action[1], -1.0, 1.0))
        roll_cmd = float(np.clip(action[2], -1.0, 1.0))
        yaw_cmd = float(np.clip(action[3], -1.0, 1.0))

        if s["fuel_kg"] <= 0.0:
            throttle = 0.0
        if s["rcs_fuel_kg"] <= 0.0:
            pitch_cmd = roll_cmd = yaw_cmd = 0.0

        # DPS: throttle-only, fixed along body -Z (real gimbal is CG trim, not RL-controlled)
        thrust_mag = 0.0 if throttle <= 0.0 else p.dps_thrust_min_n + throttle * (p.dps_thrust_max_n - p.dps_thrust_min_n)

        mass_kg = p.dry_mass_kg + s["fuel_kg"]
        i_tilt, i_yaw = moment_of_inertia(mass_kg, p.body_radius_m, p.body_height_m)
        max_torque = p.rcs_thrust_n * p.rcs_quad_radius_m * p.rcs_jets_per_couple

        # RCS: 16-jet attitude control, torque duty-cycle per axis, own propellant tank
        s["wx"] += (pitch_cmd * max_torque / i_tilt) * p.dt_s
        s["wy"] += (roll_cmd * max_torque / i_tilt) * p.dt_s
        s["wz"] += (yaw_cmd * max_torque / i_yaw) * p.dt_s
        damping = max(0.0, 1.0 - p.angular_damping_per_s * p.dt_s)
        s["wx"] *= damping
        s["wy"] *= damping
        s["wz"] *= damping
        s["tilt_x"] += s["wx"] * p.dt_s
        s["tilt_y"] += s["wy"] * p.dt_s
        s["yaw"] += s["wz"] * p.dt_s

        lost_control = bool(np.hypot(s["tilt_x"], s["tilt_y"]) > p.loss_of_control_tilt_rad)

        # body tilt (from RCS), not gimbal, redirects the fixed-direction DPS thrust
        thrust_x = thrust_mag * np.sin(s["tilt_y"])
        thrust_y = -thrust_mag * np.sin(s["tilt_x"])
        thrust_z = thrust_mag * np.cos(s["tilt_x"]) * np.cos(s["tilt_y"])

        ax = thrust_x / mass_kg
        ay = thrust_y / mass_kg
        az = thrust_z / mass_kg - p.gravity_m_s2

        s["vx"] += ax * p.dt_s
        s["vy"] += ay * p.dt_s
        s["vz"] += az * p.dt_s
        s["x"] += s["vx"] * p.dt_s
        s["y"] += s["vy"] * p.dt_s
        s["z"] += s["vz"] * p.dt_s

        dps_mdot = thrust_mag / (p.dps_isp_s * G0)
        rcs_mdot = (abs(pitch_cmd) + abs(roll_cmd) + abs(yaw_cmd)) * (p.rcs_thrust_n * p.rcs_jets_per_couple) / (p.rcs_isp_s * G0)
        s["fuel_kg"] = max(0.0, s["fuel_kg"] - dps_mdot * p.dt_s)
        s["rcs_fuel_kg"] = max(0.0, s["rcs_fuel_kg"] - rcs_mdot * p.dt_s)
        s["throttle"] = throttle
        s["rcs_pitch"], s["rcs_roll"], s["rcs_yaw"] = pitch_cmd, roll_cmd, yaw_cmd
        self._t += p.dt_s

        ground_z = self._ground_z(s["x"], s["y"])
        touched_down = bool(s["z"] <= ground_z)
        timed_out = self._t >= p.max_episode_s

        terminated = touched_down or lost_control
        truncated = bool(timed_out and not terminated)

        landed_safely = False
        landing_margins: dict = {}
        leg_force_n = 0.0
        leg_force_max_n = 1.0
        if touched_down and not lost_control:
            v_xy = float(np.hypot(s["vx"], s["vy"]))
            tilt = float(np.hypot(s["tilt_x"], s["tilt_y"]))
            w = float(np.sqrt(s["wx"] ** 2 + s["wy"] ** 2 + s["wz"] ** 2))
            leg_diff = self._footpad_height_diff_m(s["x"], s["y"])

            landed_safely = bool(
                abs(s["vz"]) <= p.safe_landing_v_z_m_s
                and v_xy <= p.safe_landing_v_xy_m_s
                and tilt <= p.safe_landing_tilt_rad
                and w <= p.safe_landing_w_rad_s
                and leg_diff <= p.safe_landing_max_leg_height_diff_m
            )
            # margin = how far *under* each threshold we were, in [0, 1]; used
            # to grade a soft landing higher than a marginal one (reward.py).
            landing_margins = {
                "v_z": float(np.clip(1.0 - abs(s["vz"]) / max(p.safe_landing_v_z_m_s, 1e-6), 0.0, 1.0)),
                "v_xy": float(np.clip(1.0 - v_xy / max(p.safe_landing_v_xy_m_s, 1e-6), 0.0, 1.0)),
                "tilt": float(np.clip(1.0 - tilt / max(p.safe_landing_tilt_rad, 1e-6), 0.0, 1.0)),
                "w": float(np.clip(1.0 - w / max(p.safe_landing_w_rad_s, 1e-6), 0.0, 1.0)),
            }

            _, leg_force_max_n = leg_force_bounds_n(ApolloLMSpecs(), mass_kg, p.safe_landing_v_z_m_s, p.gravity_m_s2)
            ke_j = 0.5 * mass_kg * s["vz"] ** 2
            leg_force_n = mass_kg * p.gravity_m_s2 / p.leg_count + ke_j / (p.leg_count * p.leg_stroke_m)

            s["z"] = ground_z  # clamp to surface

        info = {
            "terminated": terminated,
            "truncated": truncated,
            "landed_safely": landed_safely,
            "lost_control": lost_control,
            "landing_margins": landing_margins,
            "altitude_m": s["z"] - ground_z,
            "fuel_kg": s["fuel_kg"],
            "rcs_fuel_kg": s["rcs_fuel_kg"],
            "mass_kg": mass_kg,
            "leg_force_n": leg_force_n,
            "leg_force_max_n": leg_force_max_n,
            "t_s": self._t,
        }
        reward = self.reward_fn(self, info)

        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        p = self.params
        s = self.state
        ground_z = self._ground_z(s["x"], s["y"])
        qw, qx, qy, qz = _euler_to_quat(s["tilt_x"], s["tilt_y"], s["yaw"])
        obs = [
            s["x"] - p.target_x, s["y"] - p.target_y, s["z"] - ground_z,
            s["vx"], s["vy"], s["vz"],
            qw, qx, qy, qz,
            s["wx"], s["wy"], s["wz"],
            s["fuel_kg"] / p.initial_fuel_kg,
            s["rcs_fuel_kg"] / p.initial_rcs_fuel_kg,
            0.0,  # leg_force_frac: legs aren't loaded until the terminal touchdown step
        ]
        if self.use_lidar_obs:
            obs.extend(self._lidar_ranges())
        return normalize_obs(np.array(obs, dtype=np.float32))

    def _lidar_ranges(self) -> list[float]:
        from lunarsim.core.metadata.lidar import LidarScanPattern, raycast_lidar

        s = self.state
        pattern = LidarScanPattern.spinning(
            n_channels=1, vertical_fov_deg=(-90, -90), horizontal_res_deg=360.0 / self.lidar_n_rays
        )
        dirs = pattern.ray_directions()[: self.lidar_n_rays]
        pc = raycast_lidar(
            self.tile.height, self.tile.res_m,
            np.array([s["x"], s["y"], s["z"]]), dirs, max_range_m=self.params.spawn_altitude_m * 2,
        )
        ranges = pc.range_m.copy()
        ranges[np.isnan(ranges)] = self.params.spawn_altitude_m * 2
        return ranges.tolist()
