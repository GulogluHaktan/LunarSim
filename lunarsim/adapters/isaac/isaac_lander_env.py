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
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, G0, leg_force_bounds_n, moment_of_inertia
from lunarsim.rl.analytic_lander_env import LanderParams, _euler_to_quat
from lunarsim.rl.action_map import action_to_throttle
from lunarsim.rl.obs_norm import normalize_obs

_REPO_ROOT = Path(__file__).resolve().parents[3]

RewardFn = Callable[["IsaacLanderEnv", dict], float]

# Azimuths of the contact-footprint stencil (see `contact_clearance_m`): the
# vehicle rests on a ring of footpads, so the ground it can come to rest on
# is the highest surface anywhere under that ring, not the single point
# under the body origin.
#
# 16 samples rather than the 8 this used while the contact ring was the
# 2.1 m body radius: the ring is now the 4.7 m footpad stance, so 8 samples
# would sit 3.7 m apart along it and miss far more relief between them than
# the measured stencil error `TOUCHDOWN_CONTACT_EPS_M` is sized for. At 16
# the arc spacing is 1.84 m, slightly tighter than the 1.65 m the old
# calibration was taken at.
_FOOTPRINT_AZIMUTHS_RAD = np.deg2rad(np.arange(0.0, 360.0, 22.5))

# Tolerance on the "lowest point of the collider has reached the ground"
# test below. Two real, measured error sources have to fit inside it:
#   1. the 17-point footprint stencil under-estimates the true maximum of
#      the collision surface over the contact ring whenever the peak falls
#      between samples. RE-MEASURED after the contact ring moved from the
#      2.1 m body radius to the 4.7 m footpad stance (and the stencil from
#      8 azimuths to 16), against a 256-point dense sampling of the same
#      exact mesh, over 1200 random sites x 3 tiles per stage: worst case
#      0.0215 m (ramp_20m, res=0.75 m -- the finest/roughest grid in the
#      curriculum), <=0.0053 m at every other stage. That is BETTER than
#      the 0.039 m the old 8-point/2.1 m stencil measured: the bigger ring
#      sampled 16 times has tighter arc spacing (1.84 m vs 1.65 m, so
#      comparable) while covering ground whose relief the small disc used
#      to miss entirely.
#   2. PhysX's own contact offset (default 0.02 m) -- contacts are generated
#      before the surfaces literally coincide.
# 0.0215 + 0.02 = 0.0415 -> 0.05 m still covers both with margin, and is
# ~0.05x the tightest safety threshold it interacts with
# (safe_landing_v_z_m_s=1.0 m/s crosses 0.05 m in 0.05 s, exactly one dt_s
# control step), i.e. it cannot meaningfully flatter a touchdown's measured
# velocity/attitude.
TOUCHDOWN_CONTACT_EPS_M = 0.05


def collision_mesh_height_at(height: np.ndarray, res_m: float, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Height of the ACTUAL PhysX collision surface at (x, y) -- i.e. of the
    exact triangle mesh `lunarsim.adapters.isaac.heightfield.
    _mesh_from_heightfield` authors from the same `height` grid, including
    its triangulation (each grid cell split along the p00-p11 diagonal:
    faces (p00, p10, p11) and (p00, p11, p01), with grid index i <-> x and
    j <-> y).

    REAL BUG THIS FIXES: everything on the Isaac side previously asked
    `lunarsim.core.terrain.rocks.sample_height_at` where the ground was.
    That function is a NEAREST-SAMPLE lookup, which is the right answer for
    `AnalyticLanderEnv` (there the heightfield IS the world) but not here,
    where PhysX collides against the interpolated mesh between those
    samples. Measured discrepancy between the two, over 2500 random sites x
    5 tiles at each curriculum stage: p99 ~0.026 m, max 0.054 m, and it does
    NOT shrink with grid resolution (the terrain amplitude is fixed, so a
    finer grid just has proportionally steeper cells). 0.054 m is 54x the
    1 mm touchdown tolerance this module used to apply -- see
    `contact_clearance_m`.

    Verified exactly (max abs error 4.9e-15 m over 500 random query points)
    against a brute-force barycentric lookup over the literal face list
    `_mesh_from_heightfield` emits.
    """
    n = height.shape[0]
    center = (n - 1) / 2.0
    fi = np.clip(np.asarray(x_m, dtype=np.float64) / res_m + center, 0.0, n - 1.0)
    fj = np.clip(np.asarray(y_m, dtype=np.float64) / res_m + center, 0.0, n - 1.0)
    i0 = np.minimum(fi.astype(np.int64), n - 2)
    j0 = np.minimum(fj.astype(np.int64), n - 2)
    a, b = fi - i0, fj - j0
    h00, h10 = height[i0, j0], height[i0 + 1, j0]
    h01, h11 = height[i0, j0 + 1], height[i0 + 1, j0 + 1]
    # a >= b is the (p00, p10, p11) triangle; b > a is (p00, p11, p01)
    return np.where(a >= b,
                    h00 + (h10 - h00) * a + (h11 - h10) * b,
                    h00 + (h01 - h00) * b + (h11 - h01) * a)


def contact_clearance_m(height: np.ndarray, res_m: float, x: float, y: float, z: float,
                        tilt_x: float, tilt_y: float, half_height_m: float,
                        stance_radius_m: float) -> tuple[float, float]:
    """`(clearance_m, ground_z)`: how far the LOWEST point of the real
    collision proxy is above the HIGHEST collision surface under its
    footprint, and that surface height.

    REAL BUG FOUND (this session, from a real Isaac Sim capture of the
    hand-designed ZemZev controller --
    `out/eval_snapshots/orbit_descent_multiseed_44/`): that episode flew a
    textbook landing, came to rest at t=47s with the vehicle motionless
    (pos (-296.60, 49.11, 3.34) unchanged for the last 16 s, tilt 0.6 deg,
    throttle 0.04), and the run was nevertheless classified
    `end_reason=max_episode_s`, `landed_safely=None` -- the sim never
    noticed it had landed. Cause: touchdown was tested as
    `(z - half_height_m) <= ground_z + 1e-3`, i.e. a POINT model (the centre
    of the belly) against the ground height at the body origin, with a 1 mm
    tolerance. The real collider is a `body_radius_m`-radius,
    `height_m`-tall cylinder, which at rest sits with its lowest RIM point
    on the highest ground under that disc -- so the belly centre rests
    ABOVE the sampled ground by (terrain relief under the footprint) +
    `body_radius_m * sin(tilt)`. In that episode the reported resting
    altitude was 0.067 m: 67x the tolerance, so the test could never fire.

    Measured over 2400 random sites x 6 tiles per stage, this resting offset
    is median 0.006 m (orbit_descent) to 0.059 m (ramp_20m) from terrain
    relief alone, and only 1.7-28% of sites ever let the belly centre reach
    within 1 mm of the sampled ground. Residual tilt adds a further
    `body_radius_m * sin(tilt)` = 0.037 m per degree. So on real terrain the
    old test fired essentially ONLY when the vehicle was moving fast enough
    to punch the belly centre through the surface within one step -- i.e.
    the only touchdowns the trainer ever saw were crashes, and every soft
    arrival was silently relabelled "timed out". That asymmetry is enough on
    its own to prevent any policy from ever being paid the terminal landing
    bonus (see `reward.py`), independent of reward tuning.

    Geometry: `stance_radius_m` is the radius of the ring the vehicle rests
    on, at `half_height_m` below the body origin. Tilted by `tilt` from
    vertical, the ring's lowest point sits
    `half_height_m*cos(tilt) + stance_radius_m*sin(tilt)` below the origin
    (the downhill side of the ring swings down). The ground it can rest on
    is the max of the collision surface over that ring, sampled at the
    centre plus `_FOOTPRINT_AZIMUTHS_RAD`.

    `stance_radius_m` is the FOOTPAD stance (`footpad_span_m / 2`), not the
    body radius: the collider is four footpads at that stance plus a raised
    descent-stage cylinder (see `adapters.isaac.lander.spawn_apollo_lm`).
    A continuous ring rather than the four discrete pads, deliberately --
    yaw is not an argument here, and the ring is the yaw-marginalised
    version of four pads at unknown heading. It is also the conservative
    direction: the max over a whole ring is never below the max over four
    points on it, so contact is reported no later than it really happens.
    """
    tilt = float(np.hypot(tilt_x, tilt_y))
    lowest_z = z - (half_height_m * np.cos(tilt) + stance_radius_m * np.sin(tilt))
    xs = np.concatenate(([x], x + stance_radius_m * np.cos(_FOOTPRINT_AZIMUTHS_RAD)))
    ys = np.concatenate(([y], y + stance_radius_m * np.sin(_FOOTPRINT_AZIMUTHS_RAD)))
    ground_z = float(collision_mesh_height_at(height, res_m, xs, ys).max())
    return float(lowest_z - ground_z), ground_z


def out_of_tile(tile: Tile, x: float, y: float, margin_m: float) -> bool:
    """Is (x, y) outside the part of `tile` that actually has ground under it?

    REAL BUG FOUND (this session, from every single real orbit_descent
    capture on disk): `sample_height_at` CLAMPS out-of-range indices to the
    grid edge, so it keeps returning a plausible-looking height forever
    outside the tile -- while the PhysX collision mesh simply ENDS at the
    tile boundary. A vehicle that drifts off the tile therefore flies over
    a void while the env reports a confident, entirely fictional altitude.
    Real evidence, 600 m tile (half-extent 300 m):
      orbit_descent_margins_check_2: crosses r=300 m at t~13 s of a 60 s
        episode and ends at r=524 m with a reported altitude of -32.8 m,
        i.e. 33 m "underground" with nothing to collide with;
      orbit_descent_margins_check_1: ends at r=650 m, reported alt -3.2 m,
        logged as `end_reason=touchdown` with vxy=17.45 m/s -- a "crash"
        against ground that does not exist;
      orbit_descent_multiseed_33 (ZemZev): ends at r=487 m, alt -2.4 m,
        also logged as a touchdown.
    Terminating these as `truncated` keeps the fiction out of the replay
    buffer and makes an undersized tile fail loudly instead of silently.
    """
    half = tile.size_m / 2.0 - margin_m
    return bool(abs(x) > half or abs(y) > half)


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
        legacy_obs15: bool = False,
    ):
        super().__init__()
        self.tile = tile
        # OBSERVATION VINTAGE. Slot 15 was `leg_force_frac`, hardcoded 0.0 in
        # every env -- a dead input. It now carries time-remaining. That keeps
        # the vector 16-wide so old checkpoints LOAD, which is convenient and
        # also a trap: a policy trained when the slot was always 0 has arbitrary
        # weights on it, and feeding it a 1.0 -> 0.0 ramp changes its behaviour
        # with no warning and no space-check failure. Set this to evaluate a
        # pre-change checkpoint on the input it was actually trained with, so a
        # measured difference can be attributed to the thing under test rather
        # than to the slot.
        self.legacy_obs15 = bool(legacy_obs15)
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
        return float(collision_mesh_height_at(
            self.tile.height, self.tile.res_m, np.array([x]), np.array([y]))[0])

    def _contact(self) -> tuple[float, float]:
        """`(clearance_m, ground_z)` for the current state -- see
        `contact_clearance_m`. One call costs a single 9-point vectorized
        heightfield lookup (~20 us), negligible against the 6 PhysX substeps
        in the same control step."""
        s = self.state
        return contact_clearance_m(
            self.tile.height, self.tile.res_m, s["x"], s["y"], s["z"],
            s["tilt_x"], s["tilt_y"], self._half_height_m, self._specs.footpad_span_m / 2.0)

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

        throttle = float(action_to_throttle(action[0]))
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
        # Impact state, snapshotted BEFORE the physics substeps. See the long
        # note above the vec env's substep loop: grading from the POST-substep
        # state reads ~0 velocity on any impact faster than 1 m/s, because the
        # 0.05 m contact band is narrower than one control step's travel and the
        # regolith has restitution 0, so PhysX has already stopped the vehicle.
        pre_vz = float(s["vz"])
        pre_vxy = float(np.hypot(s["vx"], s["vy"]))
        pre_tilt = float(np.hypot(s["tilt_x"], s["tilt_y"]))
        pre_w = float(np.sqrt(s["wx"] ** 2 + s["wy"] ** 2 + s["wz"] ** 2))

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

        # REAL BUG CAUGHT VIA A TOUCHDOWN SMOKE TEST: PhysX collision (real,
        # against the real terrain mesh) correctly rests the vehicle with
        # its root (body CENTER, not a zero-height point like the analytic
        # env's `z`) above the ground and stops it there -- but comparing
        # that real resting height directly against `ground_z` (the analytic
        # env's touchdown test) never fires, since the root never reaches
        # the ground itself. Has to account for the real vehicle's physical
        # extent -- and, as a LATER REAL BUG showed (see
        # `contact_clearance_m`'s docstring: a real ZemZev capture that
        # landed, parked for 16 s and was still logged as a timeout), the
        # extent that matters is the whole tilted cylinder against the
        # ground under its whole footprint, not the belly centre against the
        # ground under the body origin.
        clearance_m, ground_z = self._contact()
        touched_down = bool(clearance_m <= TOUCHDOWN_CONTACT_EPS_M)
        lost_control = bool(np.hypot(s["tilt_x"], s["tilt_y"]) > p.loss_of_control_tilt_rad)
        timed_out = self._t >= p.max_episode_s
        # off the edge of the terrain tile there is no collision mesh at all
        # -- see `out_of_tile`. Margin = the footpad sampling radius, so the
        # leg-height-diff stencil can still land on real grid cells.
        left_tile = out_of_tile(self.tile, s["x"], s["y"], self._specs.footpad_span_m / 2.0)

        terminated = touched_down or lost_control
        truncated = bool((timed_out or left_tile) and not terminated)

        landed_safely = False
        landing_margins: dict = {}
        leg_force_n = 0.0
        leg_force_max_n = 1.0
        if touched_down and not lost_control:
            # pre-substep state, for the reason given above the substep loop
            v_z, v_xy, tilt, w = pre_vz, pre_vxy, pre_tilt, pre_w
            leg_diff = self._footpad_height_diff_m(s["x"], s["y"])

            landed_safely = bool(
                abs(v_z) <= p.safe_landing_v_z_m_s
                and v_xy <= p.safe_landing_v_xy_m_s
                and tilt <= p.safe_landing_tilt_rad
                and w <= p.safe_landing_w_rad_s
                and leg_diff <= p.safe_landing_max_leg_height_diff_m
            )
            landing_margins = {
                "v_z": float(np.clip(1.0 - abs(v_z) / max(p.safe_landing_v_z_m_s, 1e-6), 0.0, 1.0)),
                "v_xy": float(np.clip(1.0 - v_xy / max(p.safe_landing_v_xy_m_s, 1e-6), 0.0, 1.0)),
                "tilt": float(np.clip(1.0 - tilt / max(p.safe_landing_tilt_rad, 1e-6), 0.0, 1.0)),
                "w": float(np.clip(1.0 - w / max(p.safe_landing_w_rad_s, 1e-6), 0.0, 1.0)),
                # leg_diff is the FIFTH `landed_safely` criterion and was the
                # only one missing here, so `_terminal_reward`'s quality average
                # graded 4 of 5 and a touchdown rejected purely on footpad
                # flatness reported margins that all looked comfortable -- the
                # "CRASH with every printed number inside its limit" case
                # diag_stage_landing_rate.py's own comment describes.
                "leg_diff": float(np.clip(
                    1.0 - leg_diff / max(p.safe_landing_max_leg_height_diff_m, 1e-6), 0.0, 1.0)),
            }
            _, leg_force_max_n = leg_force_bounds_n(self._specs, mass_kg, p.safe_landing_v_z_m_s, p.gravity_m_s2)
            ke_j = 0.5 * mass_kg * v_z ** 2  # impact speed, not the post-contact ~0
            leg_force_n = mass_kg * p.gravity_m_s2 / p.leg_count + ke_j / (p.leg_count * p.leg_stroke_m)

            # no `s["z"] = ground_z` clamp here (unlike the analytic env):
            # real PhysX collision already rests the vehicle at the
            # physically correct height on its own -- overwriting it would
            # fight the real physics instead of reading it.

        info = {
            "terminated": terminated,
            "truncated": truncated,
            # which KIND of truncation, so the reward can charge a timeout
            # without charging an artificial tile-edge cut (see reward.py).
            "timed_out": bool(truncated and not left_tile),
            # the action that produced this step. See the matching note in the
            # vec env: reward.py's `_action_saturation_penalty` reads
            # `info["action"]` and nothing used to write it, so that term was
            # identically zero and the run meant to test it measured nothing.
            "action": np.asarray(action, dtype=float).copy(),
            # the impact state, so `_touchdown_severity` grades the speed the
            # vehicle actually hit at rather than the ~0 PhysX leaves after
            # the contact substep. Keys match `_STATE_FIELDS` names so the
            # severity function needs no special case.
            "impact_state": {"vz": pre_vz, "vx": pre_vxy, "vy": 0.0,
                             "tilt_x": pre_tilt, "tilt_y": 0.0,
                             "wx": pre_w, "wy": 0.0, "wz": 0.0},
            "landed_safely": landed_safely,
            "lost_control": lost_control,
            "left_tile": left_tile,
            "landing_margins": landing_margins,
            # "altitude" is now the real clearance of the collider's lowest
            # point above the ground it can rest on, so it reaches 0 exactly
            # at touchdown -- it used to be the belly CENTRE above the
            # ground at the body origin, which bottomed out at a stage-
            # dependent +0.006..+0.059 m and never reached 0 (see
            # `contact_clearance_m`). Same quantity the observation and
            # `reward.py`'s shaping terms read.
            "altitude_m": clearance_m,
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
        # same clearance definition the termination test and info["altitude_m"]
        # use, so "altitude" means one thing everywhere in the pipeline.
        clearance_m, _ = self._contact()
        qw, qx, qy, qz = _euler_to_quat(s["tilt_x"], s["tilt_y"], s["yaw"])
        obs = [
            s["x"] - p.target_x, s["y"] - p.target_y, clearance_m,
            s["vx"], s["vy"], s["vz"],
            qw, qx, qy, qz,
            s["wx"], s["wy"], s["wz"],
            s["fuel_kg"] / p.initial_fuel_kg,
            s["rcs_fuel_kg"] / p.initial_rcs_fuel_kg,
            # TIME REMAINING, as a fraction of the episode budget.
            #
            # This slot used to be `leg_force_frac`, hardcoded 0.0 in all three
            # envs -- a permanently dead input. It now carries the one quantity
            # whose absence made this a non-Markovian problem: the env truncates
            # at `max_episode_s`, so the value of a state genuinely depends on
            # how much clock is left, and without it in the observation the
            # critic is fitting mutually inconsistent targets for states that
            # look identical. That defect is invisible to every hyperparameter,
            # which matches the measured fact that gamma, learning rate, replay
            # ratio, entropy coefficient and a log-std cap all failed to stop
            # the mid-run collapse.
            #
            # It also matters because the margin here is thin: on ramp_35m the
            # expert touches down at a median 22.4 s against a 25 s budget, an
            # ~11% slack, so "how long have I got" is decision-relevant rather
            # than academic.
            #
            # Reusing the dead slot keeps the observation 16-wide, so existing
            # checkpoints still load.
            0.0 if self.legacy_obs15 else max(0.0, 1.0 - self._t / p.max_episode_s),
        ]
        return normalize_obs(np.array(obs, dtype=np.float32))

    def close(self):
        pass
