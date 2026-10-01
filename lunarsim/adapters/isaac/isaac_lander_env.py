"""Isaac Sim/PhysX-backed lunar lander env: the SAME Gymnasium observation
layout, `LanderParams`, and `reward_fn` contract as
`lunarsim.rl.analytic_lander_env.AnalyticLanderEnv` (a reward_fn written
against one works unchanged against the other -- both only ever touch
`env.params` and `env.state`), but position/velocity/attitude come from a
REAL rigid-body PhysX simulation against the REAL terrain collision mesh,
not hand-integrated Newton equations. Gravity, mass, inertia, and ground
contact are simulated by PhysX itself.

Actuator modeling intentionally still matches the analytic env's own
simplification, not a full per-jet force model: DPS thrust is applied as a
single body-local force along -Z through the body origin (no lever arm --
matches `lander.py`'s `DPS_Engine` locator, itself authored at the body
origin, and matches the analytic env's own no-lever-arm DPS model), and RCS
is applied as a single commanded torque per axis from the SAME
`max_torque = rcs_thrust_n * rcs_quad_radius_m * rcs_jets_per_couple` model
the analytic env uses, not 16 individual jet forces at the real jet
locators. So the actuator envelope a trained policy sees is identical
between the two backends -- only the *rigid-body dynamics + ground contact*
differ, which is the entire point of this backend. The analytic env's
angular-rate damping (`LanderParams.angular_damping_per_s`, standing in for
the real Apollo DAP inner loop -- see that module's docstring on why it's
physically necessary, not an RL crutch) is carried over the same way, as a
direct post-step angular-velocity decay: PhysX gives no free damping in
vacuum, so without this the same "undamped double integrator" instability
would show up here too.

Mass and inertia are written back to the PhysX rigid body every step as DPS
propellant burns (`RigidPrim.set_masses`/`set_inertias`) -- unlike the
analytic env, where `mass_kg` only exists as a term in a hand-written `a =
F/m`, here it has to actually reach the physics engine to affect real
dynamics.

Requires an already-running Isaac Sim process (`isaacsim.core.api.World`
gets created/reused here) -- see `scripts/train_sac_isaac.py`. No
camera/render is stepped during training (`world.step(render=False)`) --
physics-only, headless-fast; rendering is what
`scripts/isaaclab_static_telemetry_capture.py` / `isaaclab_landing_render.py`
are for.

CURRENT LIMITATION: `tile` is a single fixed terrain (no `tile_fn`
regen-per-episode like the analytic env's curriculum trainer uses) --
rebuilding the PhysX heightfield collider every reset is possible but
expensive, and out of scope for this first physics-backed pass. Curriculum
training against this backend has to vary spawn altitude/speed only, not
terrain, until that's added.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lunarsim.core.terrain.generate import Tile
from lunarsim.core.terrain.rocks import sample_height_at
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, G0, leg_force_bounds_n, moment_of_inertia
from lunarsim.rl.analytic_lander_env import LanderParams, _euler_to_quat
from lunarsim.rl.obs_norm import normalize_obs

_REPO_ROOT = Path(__file__).resolve().parents[3]

RewardFn = Callable[["IsaacLanderEnv", dict], float]


def _quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Inverse of `analytic_lander_env._euler_to_quat` (same roll(x)-pitch(y)
    -yaw(z) aerospace convention) -- verified by round-trip against it."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


class IsaacLanderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tile: Tile,
        params: Optional[LanderParams] = None,
        reward_fn: Optional[RewardFn] = None,
        seed: int | None = None,
        lunarsim_root: str | Path | None = None,
    ):
        super().__init__()
        self.tile = tile
        self.params = params or LanderParams()
        if reward_fn is None:
            from lunarsim.rl.reward import default_reward_fn
            reward_fn = default_reward_fn
        self.reward_fn = reward_fn
        self._rng = np.random.default_rng(seed)
        self._specs = ApolloLMSpecs()
        self._half_height_m = self._specs.height_m / 2.0
        self._lunarsim_root = Path(lunarsim_root) if lunarsim_root is not None else _REPO_ROOT

        obs_dim = 16
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # [throttle, pitch_cmd, roll_cmd, yaw_cmd] all in [-1, 1] -- same as AnalyticLanderEnv
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._build_scene()

        self.state: dict = {}
        self._t = 0.0

    # ------------------------------------------------------------------
    # Scene construction (once per process)
    # ------------------------------------------------------------------
    def _build_scene(self):
        from pxr import Gf, UsdPhysics, UsdShade
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim

        from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, apply_regolith_physics_material
        from lunarsim.adapters.isaac.lander import spawn_apollo_lm

        # REAL BUG FOUND VIA A SMOKE TEST: World() defaults to a 60Hz physics
        # step regardless of `LanderParams.dt_s` -- an earlier version of
        # this method left that default in place, so every dt_s-scaled
        # quantity in step() (fuel burn, angular damping, episode-time
        # accounting) silently ran against the wrong step size while real
        # position/velocity integrated at the true (60Hz) rate, desyncing
        # "simulated seconds" from real seconds. `LanderParams.dt_s` (0.05s,
        # a 20Hz control rate) is too large a single PhysX step for stable
        # rigid-body integration, so it's kept as the RL control rate and
        # substepped internally at a stable physics rate instead (see
        # `_n_substeps` in `step()`), the same zero-order-hold pattern any
        # real flight controller running slower than its physics has.
        self._physics_dt = 1.0 / 120.0
        self._n_substeps = max(1, round(self.params.dt_s / self._physics_dt))
        self.world = World(stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene",
                            physics_dt=self._physics_dt, rendering_dt=self._physics_dt)
        stage = self.world.stage

        physics_scene = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr(self.params.gravity_m_s2)

        add_heightfield_collision(stage, "/World/Terrain/Collision", self.tile)
        regolith_physics_mat = apply_regolith_physics_material(stage, "/World/PhysicsMaterials/Regolith")
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Terrain/Collision")).Bind(
            regolith_physics_mat, materialPurpose="physics")

        lm_asset_path = str(self._lunarsim_root / "assets/models/apollo_lm/Apollo_Lunar_Module.usdz")
        spawn_apollo_lm(stage, "/World/LM", lm_asset_path, specs=self._specs,
                         fuel_kg=self.params.initial_fuel_kg, visual_only=False)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/LM/PhysicsProxy")).Bind(
            regolith_physics_mat, materialPurpose="physics")

        self.world.reset()
        self.body = RigidPrim("/World/LM")
        self.body.initialize()

    def _ground_z(self, x: float, y: float) -> float:
        return float(sample_height_at(self.tile.height, self.tile.res_m, np.array([x]), np.array([y]))[0])

    def _footpad_height_diff_m(self, x: float, y: float) -> float:
        radius = self._specs.footpad_span_m / 2.0
        heights = [
            self._ground_z(x + radius * np.cos(a), y + radius * np.sin(a))
            for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])
        ]
        return float(max(heights) - min(heights))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        p = self.params
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        r = self._rng.uniform(0, p.spawn_xy_radius_m)
        theta = self._rng.uniform(0, 2 * np.pi)
        x0, y0 = r * np.cos(theta), r * np.sin(theta)
        ground_z0 = self._ground_z(x0, y0)

        speed = self._rng.uniform(*p.spawn_horizontal_speed_m_s)
        dx, dy = p.target_x - x0, p.target_y - y0
        dist = max(1e-6, float(np.hypot(dx, dy)))
        vx0, vy0 = speed * dx / dist, speed * dy / dist

        tilt_x0 = float(self._rng.uniform(-0.05, 0.05))
        tilt_y0 = float(self._rng.uniform(-0.05, 0.05))
        # `z` is the real body-center root position (unlike AnalyticLanderEnv,
        # where the vehicle is a zero-height point and z0 == altitude above
        # ground directly) -- offset by the real half-height so
        # `spawn_altitude_m` still means "clearance above the ground for the
        # vehicle's belly", the same real-world quantity in both backends.
        z0 = ground_z0 + p.spawn_altitude_m + self._half_height_m

        qw, qx, qy, qz = _euler_to_quat(tilt_x0, tilt_y0, 0.0)
        self.body.set_world_poses(
            positions=np.array([[x0, y0, z0]], dtype=np.float32),
            orientations=np.array([[qw, qx, qy, qz]], dtype=np.float32),
        )
        self.body.set_velocities(
            np.array([[vx0, vy0, p.spawn_v_z_m_s, 0.0, 0.0, 0.0]], dtype=np.float32)
        )

        self.state = {
            "x": x0, "y": y0, "z": z0,
            "vx": vx0, "vy": vy0, "vz": p.spawn_v_z_m_s,
            "tilt_x": tilt_x0, "tilt_y": tilt_y0, "yaw": 0.0,
            "wx": 0.0, "wy": 0.0, "wz": 0.0,
            "fuel_kg": p.initial_fuel_kg,
            "rcs_fuel_kg": p.initial_rcs_fuel_kg,
            "throttle": 0.0,
            "rcs_pitch": 0.0, "rcs_roll": 0.0, "rcs_yaw": 0.0,
        }
        self._t = 0.0
        self._write_mass_inertia()
        return self._observation(), {}

    def _write_mass_inertia(self):
        p = self.params
        s = self.state
        mass_kg = p.dry_mass_kg + s["fuel_kg"]
        i_tilt, i_yaw = moment_of_inertia(mass_kg, p.body_radius_m, p.body_height_m)
        self.body.set_masses(np.array([mass_kg], dtype=np.float32))
        inertia = np.array([[i_tilt, 0.0, 0.0, 0.0, i_tilt, 0.0, 0.0, 0.0, i_yaw]], dtype=np.float32)
        self.body.set_inertias(inertia)
        return mass_kg

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

        thrust_mag = 0.0 if throttle <= 0.0 else (
            p.dps_thrust_min_n + throttle * (p.dps_thrust_max_n - p.dps_thrust_min_n))
        max_torque = p.rcs_thrust_n * p.rcs_quad_radius_m * p.rcs_jets_per_couple

        # DPS: REAL BUG CAUGHT VIA THE HOVER SMOKE TEST (thrust-at-hover
        # sank far faster than gravity alone predicts): the engine NOZZLE
        # points along body -Z (exhaust fires down), but the thrust FORCE
        # ON THE VEHICLE is the opposite reaction, +Z -- exactly what the
        # analytic env's own `thrust_z = +thrust_mag*cos(tilt)...` already
        # encodes; a literal "-Z" force here was applying thrust backwards.
        # Body-local so real physics rotates it into world space for us --
        # no small-angle approximation needed here, unlike the analytic env.
        force_local = np.array([[0.0, 0.0, thrust_mag]], dtype=np.float32)
        # RCS: commanded torque per body axis, same scalar model as the analytic env.
        torque_local = np.array([[pitch_cmd * max_torque, roll_cmd * max_torque, yaw_cmd * max_torque]],
                                 dtype=np.float32)
        # zero-order hold: the action is applied unchanged across every
        # physics substep making up one `dt_s` control step (see
        # `_n_substeps`) -- external PhysX forces don't persist between
        # steps on their own, so both the force/torque and the angular-rate
        # damping (the DAP-inner-loop stand-in, same rationale as the
        # analytic env) have to be reapplied every substep, at the real
        # physics rate, not once per control step.
        damping_per_substep = max(0.0, 1.0 - p.angular_damping_per_s * self._physics_dt)
        for _ in range(self._n_substeps):
            self.body.apply_forces_and_torques_at_pos(forces=force_local, torques=torque_local, is_global=False)
            self.world.step(render=False)
            ang_vel = self.body.get_angular_velocities()[0]
            damped_ang_vel = ang_vel * damping_per_substep
            self.body.set_angular_velocities(np.array([damped_ang_vel], dtype=np.float32))

        pos, quat = self.body.get_world_poses()
        lin_vel = self.body.get_linear_velocities()[0]
        pos, quat = pos[0], quat[0]
        roll, pitch, yaw = _quat_to_euler(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

        s["x"], s["y"], s["z"] = float(pos[0]), float(pos[1]), float(pos[2])
        s["vx"], s["vy"], s["vz"] = float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])
        s["tilt_x"], s["tilt_y"], s["yaw"] = roll, pitch, yaw
        s["wx"], s["wy"], s["wz"] = float(damped_ang_vel[0]), float(damped_ang_vel[1]), float(damped_ang_vel[2])

        dps_mdot = thrust_mag / (p.dps_isp_s * G0)
        rcs_mdot = (abs(pitch_cmd) + abs(roll_cmd) + abs(yaw_cmd)) * (p.rcs_thrust_n * p.rcs_jets_per_couple) / (
            p.rcs_isp_s * G0)
        s["fuel_kg"] = max(0.0, s["fuel_kg"] - dps_mdot * p.dt_s)
        s["rcs_fuel_kg"] = max(0.0, s["rcs_fuel_kg"] - rcs_mdot * p.dt_s)
        s["throttle"] = throttle
        s["rcs_pitch"], s["rcs_roll"], s["rcs_yaw"] = pitch_cmd, roll_cmd, yaw_cmd
        self._t += p.dt_s
        mass_kg = self._write_mass_inertia()

        ground_z = self._ground_z(s["x"], s["y"])
        # REAL BUG CAUGHT VIA A TOUCHDOWN SMOKE TEST: PhysX collision (real,
        # against the real terrain mesh) correctly rests the vehicle with
        # its root (body CENTER, not a zero-height point like the analytic
        # env's `z`) at `ground_z + half_height_m` and stops it there -- but
        # comparing that real resting height directly against `ground_z`
        # (the analytic env's touchdown test) never fires, since the root
        # never reaches the ground itself. Has to account for the real
        # vehicle's physical extent.
        belly_z = s["z"] - self._half_height_m
        touched_down = bool(belly_z <= ground_z + 1e-3)
        lost_control = bool(np.hypot(s["tilt_x"], s["tilt_y"]) > p.loss_of_control_tilt_rad)
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
            landing_margins = {
                "v_z": float(np.clip(1.0 - abs(s["vz"]) / max(p.safe_landing_v_z_m_s, 1e-6), 0.0, 1.0)),
                "v_xy": float(np.clip(1.0 - v_xy / max(p.safe_landing_v_xy_m_s, 1e-6), 0.0, 1.0)),
                "tilt": float(np.clip(1.0 - tilt / max(p.safe_landing_tilt_rad, 1e-6), 0.0, 1.0)),
                "w": float(np.clip(1.0 - w / max(p.safe_landing_w_rad_s, 1e-6), 0.0, 1.0)),
            }
            _, leg_force_max_n = leg_force_bounds_n(self._specs, mass_kg, p.safe_landing_v_z_m_s, p.gravity_m_s2)
            ke_j = 0.5 * mass_kg * s["vz"] ** 2
            leg_force_n = mass_kg * p.gravity_m_s2 / p.leg_count + ke_j / (p.leg_count * p.leg_stroke_m)

            # no `s["z"] = ground_z` clamp here (unlike the analytic env):
            # real PhysX collision already rests the vehicle at the
            # physically correct height on its own -- overwriting it would
            # fight the real physics instead of reading it.

        info = {
            "terminated": terminated,
            "truncated": truncated,
            "landed_safely": landed_safely,
            "lost_control": lost_control,
            "landing_margins": landing_margins,
            "altitude_m": belly_z - ground_z,
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
            s["x"] - p.target_x, s["y"] - p.target_y, (s["z"] - self._half_height_m) - ground_z,
            s["vx"], s["vy"], s["vz"],
            qw, qx, qy, qz,
            s["wx"], s["wy"], s["wz"],
            s["fuel_kg"] / p.initial_fuel_kg,
            s["rcs_fuel_kg"] / p.initial_rcs_fuel_kg,
            0.0,
        ]
        return normalize_obs(np.array(obs, dtype=np.float32))

    def close(self):
        pass
