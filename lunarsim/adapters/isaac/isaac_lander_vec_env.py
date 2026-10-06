"""Vectorized, real-PhysX-backed lander training env: N vehicles, each on its
OWN independently-generated (and independently re-generated on every reset)
terrain tile, all simulated inside a single Isaac Sim process and stepped
together -- the actual real-physics analogue of `AnalyticLanderEnv`'s
`tile_fn`-per-episode curriculum training, not the fixed-single-tile
`IsaacLanderEnv` (see that module's docstring on why it was fixed-tile-only).

Parallelism model: NOT N processes (`IsaacLanderEnv`'s docstring's original
"N subprocesses would each pay Isaac's startup cost" reasoning) -- N COPIES
of the vehicle rigid body inside the ONE running PhysX scene, arranged in a
row of env slots (`/World/envs/env_000/...`, `env_001/...`, ...) offset far
enough apart (`env_spacing_m`) that neither their terrain tiles nor their
vehicles can ever interact across the env boundary. `isaacsim.core.prims.
RigidPrim`'s regex path (`/World/envs/env_*/LM`) already gives a single
BATCHED view over all N vehicles -- one `apply_forces_and_torques_at_pos`/
`get_world_poses`/... call per physics substep drives/reads all of them at
once, so stepping N envs costs about the same as stepping 1 (PhysX itself
is what parallelizes), unlike N independent Python-level env objects.

Per-episode terrain regen: verified directly (not assumed) that deleting a
terrain collider prim and authoring a brand new one at the same path mid-
simulation is picked up by PhysX on the very next step with no
`world.reset()` needed -- see the probe this was built from. Each env's
`_reset_one(i)` does exactly that: remove `env_i/Terrain`, generate a fresh
`Tile` via `tile_fn(self._rng)`, rebuild the collider. Only the PhysX
collision mesh is rebuilt (no render mesh/regolith material) -- training is
physics-only, headless, no camera ever renders a frame, so a visual mesh
would be pure wasted build time.

This is a custom `stable_baselines3.common.vec_env.VecEnv`, not N
`gym.Env`s wrapped in `DummyVecEnv`/`SubprocVecEnv` -- SB3's own vectorized-
env contract (batched `step`/`reset`, per-index auto-reset with
`infos[i]["terminal_observation"]` / `infos[i]["TimeLimit.truncated"]`) is
implemented directly against the batched PhysX reads/writes.

Reward: reuses `reward_fn` (e.g. `lunarsim.rl.reward.default_reward_fn`)
COMPLETELY UNCHANGED, one env at a time, through a tiny per-env scalar
`_EnvView(params, state)` proxy -- re-deriving that reward's math in
vectorized form would risk silently drifting from the already-tuned
analytic-env reward. The per-env Python loop this costs (`num_envs` reward
calls per step) is negligible next to the PhysX substep cost.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from lunarsim.core.terrain.generate import Tile
from lunarsim.core.terrain.rocks import sample_height_at
from lunarsim.rl.action_map import action_to_throttle
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, G0, leg_force_bounds_n, moment_of_inertia
from lunarsim.rl.analytic_lander_env import LanderParams, _euler_to_quat
from lunarsim.rl.obs_norm import normalize_obs
from lunarsim.adapters.isaac.isaac_lander_env import (
    TOUCHDOWN_CONTACT_EPS_M, _quat_to_euler, _REPO_ROOT, collision_mesh_height_at,
    contact_clearance_m, out_of_tile,
)

TileFn = Callable[[np.random.Generator], Tile]
RewardFn = Callable[[object, dict], float]

_STATE_FIELDS = (
    "x", "y", "z", "vx", "vy", "vz", "tilt_x", "tilt_y", "yaw", "wx", "wy", "wz",
    "fuel_kg", "rcs_fuel_kg", "throttle", "rcs_pitch", "rcs_roll", "rcs_yaw",
)


class _EnvView:
    """Scalar-state stand-in for one env, so an unmodified single-env
    `reward_fn` (only ever touches `.params`/`.state`) can be called per-env
    against this vectorized backend."""
    __slots__ = ("params", "state")

    def __init__(self, params, state):
        self.params = params
        self.state = state


class IsaacLanderVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        tile_fn: TileFn,
        params: Optional[LanderParams] = None,
        reward_fn: Optional[RewardFn] = None,
        env_spacing_m: float = 200.0,
        seed: int | None = None,
        lunarsim_root=None,
        max_rocks_per_env: int = 10,
    ):
        self.tile_fn = tile_fn
        self.params = params or LanderParams()
        self.max_rocks_per_env = max_rocks_per_env
        if reward_fn is None:
            from lunarsim.rl.reward import default_reward_fn
            reward_fn = default_reward_fn
        self.reward_fn = reward_fn
        self.env_spacing_m = env_spacing_m
        self._rng = np.random.default_rng(seed)
        self._specs = ApolloLMSpecs()
        self._half_height_m = self._specs.height_m / 2.0
        self._lunarsim_root = lunarsim_root or _REPO_ROOT

        obs_dim = 16
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self.env_origins = np.stack(
            [np.arange(num_envs) * env_spacing_m, np.zeros(num_envs), np.zeros(num_envs)], axis=-1
        ).astype(np.float64)
        self._tiles: list[Tile | None] = [None] * num_envs
        self.state = {k: np.zeros(num_envs, dtype=np.float64) for k in _STATE_FIELDS}
        self._t = np.zeros(num_envs, dtype=np.float64)
        self._actions = np.zeros((num_envs, 4), dtype=np.float32)

        self._build_scene()
        for i in range(num_envs):
            self._reset_one(i)

    # ------------------------------------------------------------------
    def _env_root(self, i: int) -> str:
        return f"/World/envs/env_{i:03d}"

    def _build_scene(self):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim

        from lunarsim.adapters.isaac.heightfield import apply_regolith_physics_material
        from lunarsim.adapters.isaac.lander import spawn_apollo_lm

        self._physics_dt = 1.0 / 120.0
        self._n_substeps = max(1, round(self.params.dt_s / self._physics_dt))
        self.world = World(stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene",
                            physics_dt=self._physics_dt, rendering_dt=self._physics_dt)
        stage = self.world.stage

        physics_scene = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr(self.params.gravity_m_s2)

        self._regolith_physics_mat = apply_regolith_physics_material(stage, "/World/PhysicsMaterials/Regolith")

        lm_asset_path = str(self._lunarsim_root / "assets/models/apollo_lm/Apollo_Lunar_Module.usdz") \
            if hasattr(self._lunarsim_root, "__truediv__") else \
            f"{self._lunarsim_root}/assets/models/apollo_lm/Apollo_Lunar_Module.usdz"

        for i in range(self.num_envs):
            root = self._env_root(i)
            env_xform = UsdGeom.Xform.Define(stage, root)
            env_xform.AddTranslateOp().Set(Gf.Vec3d(*self.env_origins[i].tolist()))
            spawn_apollo_lm(stage, f"{root}/LM", lm_asset_path, specs=self._specs,
                             fuel_kg=self.params.initial_fuel_kg, visual_only=False)
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{root}/LM/PhysicsProxy")).Bind(
                self._regolith_physics_mat, materialPurpose="physics")

        self.world.reset()
        self.body = RigidPrim("/World/envs/env_*/LM")
        self.body.initialize()
        assert self.body.count == self.num_envs, (
            f"RigidPrim batched view found {self.body.count} vehicles, expected {self.num_envs} "
            f"-- prim-path glob/ordering assumption broke.")

    def _rebuild_terrain(self, i: int, tile: Tile):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        stage = self.world.stage
        from lunarsim.adapters.isaac.heightfield import add_heightfield_collision

        root = self._env_root(i)
        terrain_path = f"{root}/Terrain"
        if stage.GetPrimAtPath(terrain_path):
            stage.RemovePrim(terrain_path)
        add_heightfield_collision(stage, terrain_path, tile)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(terrain_path)).Bind(
            self._regolith_physics_mat, materialPurpose="physics")

        # -- REAL, PHYSICAL (collidable) rocks, from the terrain's own
        # `tile.rocks` RockField -- this is a genuine field the terrain
        # generator already produces (real x/y/diameter, exponential size
        # distribution) that nothing in this codebase actually spawned as
        # a physics object before now (the capture script's "decorative
        # rocks" were a separate, hand-rolled random scatter with no
        # collision at all). Capped at `max_rocks_per_env` regardless of
        # how many `tile.rocks` generated -- a real reset happens far more
        # often during training than during one video capture, and
        # authoring/CollisionAPI-applying a full rock field's worth of
        # prims (can be dozens-hundreds for a reference density over a
        # training-scale tile) every single episode reset, across every
        # parallel env, is a real per-step-throughput cost training can't
        # afford the way one capture run could.
        rocks_path = f"{root}/Rocks"
        if stage.GetPrimAtPath(rocks_path):
            stage.RemovePrim(rocks_path)
        n_rocks = min(len(tile.rocks.x_m), self.max_rocks_per_env)
        if n_rocks > 0:
            idx = self._rng.choice(len(tile.rocks.x_m), size=n_rocks, replace=False) \
                if len(tile.rocks.x_m) > n_rocks else np.arange(len(tile.rocks.x_m))
            for k, ri in enumerate(idx):
                rx, ry = float(tile.rocks.x_m[ri]), float(tile.rocks.y_m[ri])
                diam = max(0.1, float(tile.rocks.diameter_m[ri]))
                rz = float(sample_height_at(tile.height, tile.res_m, np.array([rx]), np.array([ry]))[0])
                rock = UsdGeom.Sphere.Define(stage, f"{rocks_path}/rock_{k}")
                rock.CreateRadiusAttr(diam / 2.0)
                xf = UsdGeom.Xformable(rock.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, rz + diam * 0.15))
                UsdPhysics.CollisionAPI.Apply(rock.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(rock.GetPrim()).Bind(
                    self._regolith_physics_mat, materialPurpose="physics")

    def _ground_z(self, i: int, x_world: float, y_world: float) -> float:
        # the REAL collision surface, not the nearest heightfield sample --
        # see `isaac_lander_env.collision_mesh_height_at` for the measured
        # (up to 0.054 m) discrepancy that cost this project every soft
        # touchdown it ever flew.
        ox, oy, _ = self.env_origins[i]
        tile = self._tiles[i]
        return float(collision_mesh_height_at(
            tile.height, tile.res_m, np.array([x_world - ox]), np.array([y_world - oy]))[0])

    def _contact(self, i: int) -> tuple[float, float]:
        """`(clearance_m, ground_z)` for env `i` -- see
        `isaac_lander_env.contact_clearance_m`."""
        ox, oy, _ = self.env_origins[i]
        s, tile = self.state, self._tiles[i]
        return contact_clearance_m(
            tile.height, tile.res_m, s["x"][i] - ox, s["y"][i] - oy, s["z"][i],
            s["tilt_x"][i], s["tilt_y"][i], self._half_height_m, self._specs.footpad_span_m / 2.0)

    def _footpad_height_diff_m(self, i: int, x_world: float, y_world: float) -> float:
        radius = self._specs.footpad_span_m / 2.0
        heights = [
            self._ground_z(i, x_world + radius * np.cos(a), y_world + radius * np.sin(a))
            for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])
        ]
        return float(max(heights) - min(heights))

    def _reset_one(self, i: int):
        p = self.params
        ox, oy, _ = self.env_origins[i]

        tile = self.tile_fn(self._rng)
        self._tiles[i] = tile
        self._rebuild_terrain(i, tile)

        r = self._rng.uniform(0, p.spawn_xy_radius_m)
        theta = self._rng.uniform(0, 2 * np.pi)
        x0_local, y0_local = r * np.cos(theta), r * np.sin(theta)
        ground_z0 = self._ground_z(i, ox + x0_local, oy + y0_local)

        speed = self._rng.uniform(*p.spawn_horizontal_speed_m_s)
        dx, dy = p.target_x - x0_local, p.target_y - y0_local
        dist = max(1e-6, float(np.hypot(dx, dy)))
        vx0, vy0 = speed * dx / dist, speed * dy / dist

        tilt_x0 = float(self._rng.uniform(-0.05, 0.05))
        tilt_y0 = float(self._rng.uniform(-0.05, 0.05))
        x0, y0 = ox + x0_local, oy + y0_local
        z0 = ground_z0 + p.spawn_altitude_m + self._half_height_m

        qw, qx, qy, qz = _euler_to_quat(tilt_x0, tilt_y0, 0.0)
        self.body.set_world_poses(
            positions=np.array([[x0, y0, z0]], dtype=np.float32),
            orientations=np.array([[qw, qx, qy, qz]], dtype=np.float32),
            indices=np.array([i]),
        )
        self.body.set_velocities(
            np.array([[vx0, vy0, p.spawn_v_z_m_s, 0.0, 0.0, 0.0]], dtype=np.float32),
            indices=np.array([i]),
        )

        s = self.state
        s["x"][i], s["y"][i], s["z"][i] = x0, y0, z0
        s["vx"][i], s["vy"][i], s["vz"][i] = vx0, vy0, p.spawn_v_z_m_s
        s["tilt_x"][i], s["tilt_y"][i], s["yaw"][i] = tilt_x0, tilt_y0, 0.0
        s["wx"][i] = s["wy"][i] = s["wz"][i] = 0.0
        s["fuel_kg"][i] = p.initial_fuel_kg
        s["rcs_fuel_kg"][i] = p.initial_rcs_fuel_kg
        s["throttle"][i] = 0.0
        s["rcs_pitch"][i] = s["rcs_roll"][i] = s["rcs_yaw"][i] = 0.0
        self._t[i] = 0.0

        mass_kg = p.dry_mass_kg + s["fuel_kg"][i]
        i_tilt, i_yaw = moment_of_inertia(mass_kg, p.body_radius_m, p.body_height_m)
        self.body.set_masses(np.array([mass_kg], dtype=np.float32), indices=np.array([i]))
        self.body.set_inertias(
            np.array([[i_tilt, 0.0, 0.0, 0.0, i_tilt, 0.0, 0.0, 0.0, i_yaw]], dtype=np.float32),
            indices=np.array([i]),
        )

    def _obs_one(self, i: int) -> np.ndarray:
        p = self.params
        s = self.state
        # same clearance definition the termination test and
        # info["altitude_m"] use, so "altitude" means one thing everywhere.
        clearance_m, _ = self._contact(i)
        qw, qx, qy, qz = _euler_to_quat(s["tilt_x"][i], s["tilt_y"][i], s["yaw"][i])
        obs = [
            s["x"][i] - (self.env_origins[i][0] + p.target_x), s["y"][i] - (self.env_origins[i][1] + p.target_y),
            clearance_m,
            s["vx"][i], s["vy"][i], s["vz"][i],
            qw, qx, qy, qz,
            s["wx"][i], s["wy"][i], s["wz"][i],
            s["fuel_kg"][i] / p.initial_fuel_kg,
            s["rcs_fuel_kg"][i] / p.initial_rcs_fuel_kg,
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
            max(0.0, 1.0 - float(self._t[i]) / p.max_episode_s),
        ]
        return normalize_obs(np.array(obs, dtype=np.float32))

    def _write_mass_inertia_all(self):
        p = self.params
        mass_kg = p.dry_mass_kg + self.state["fuel_kg"]
        i_tilt, i_yaw = moment_of_inertia(mass_kg, p.body_radius_m, p.body_height_m)
        self.body.set_masses(mass_kg.astype(np.float32))
        n = self.num_envs
        inertia = np.zeros((n, 9), dtype=np.float32)
        inertia[:, 0] = i_tilt
        inertia[:, 4] = i_tilt
        inertia[:, 8] = i_yaw
        self.body.set_inertias(inertia)
        return mass_kg

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------
    def reset(self):
        for i in range(self.num_envs):
            self._reset_one(i)
        return np.stack([self._obs_one(i) for i in range(self.num_envs)])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)

    def step_wait(self):
        p = self.params
        s = self.state
        n = self.num_envs
        a = self._actions

        throttle = action_to_throttle(a[:, 0])
        pitch_cmd, roll_cmd, yaw_cmd = a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy()
        throttle = np.where(s["fuel_kg"] <= 0.0, 0.0, throttle)
        no_rcs = s["rcs_fuel_kg"] <= 0.0
        pitch_cmd = np.where(no_rcs, 0.0, pitch_cmd)
        roll_cmd = np.where(no_rcs, 0.0, roll_cmd)
        yaw_cmd = np.where(no_rcs, 0.0, yaw_cmd)

        thrust_mag = np.where(throttle <= 0.0, 0.0, p.dps_thrust_min_n + throttle * (p.dps_thrust_max_n - p.dps_thrust_min_n))
        max_torque = p.rcs_thrust_n * p.rcs_quad_radius_m * p.rcs_jets_per_couple

        force_local = np.zeros((n, 3), dtype=np.float32)
        force_local[:, 2] = thrust_mag
        torque_local = np.stack([pitch_cmd * max_torque, roll_cmd * max_torque, yaw_cmd * max_torque], axis=-1).astype(np.float32)

        # IMPACT STATE, snapshotted BEFORE the physics substeps run.
        #
        # REAL BUG (found 2026-10-06 by audit): termination was graded from the
        # state read AFTER all `_n_substeps` substeps. One control step is
        # dt_s=0.05 s, so the vehicle travels 0.05*|vz| m, while the contact
        # band `TOUCHDOWN_CONTACT_EPS_M` is a fixed 0.05 m. For |vz| > 1.0 m/s
        # the band is NARROWER than one step of travel, so contact typically
        # happens mid-loop -- and the regolith material is authored with
        # restitution=0.0 (heightfield.py), so PhysX has already killed the
        # velocity by the time it is read. The grader therefore saw vz ~ 0 for
        # an arbitrarily hard slam: `landed_safely` passed on impacts it should
        # have failed, and `_touchdown_severity`, whose whole job is "how many
        # times over its limit was this touchdown", reported ~0 severity above
        # 1 m/s. Roughly 1/|vz| of fast impacts were caught honestly.
        #
        # That made the terminal LABEL a lottery whose odds depend on descent
        # rate -- a defect no hyperparameter can touch, which matches gamma,
        # learning rate, replay ratio, entropy coefficient and a log-std cap all
        # failing to stop the mid-run collapse. The ZemZev controller was
        # unaffected because it flies its terminal descent at 0.80 m/s, inside
        # the honest regime, while an RL policy exploring faster descents was
        # graded by coin flip.
        #
        # Grading from the pre-step state costs nothing (no extra reads) and is
        # off by at most one step of acceleration, ~0.08 m/s here, against a
        # 1.0 m/s limit -- versus being off by the entire impact velocity.
        pre_vz = np.array(s["vz"], dtype=float)
        pre_vxy = np.hypot(np.asarray(s["vx"], dtype=float), np.asarray(s["vy"], dtype=float))
        pre_tilt = np.hypot(np.asarray(s["tilt_x"], dtype=float), np.asarray(s["tilt_y"], dtype=float))
        pre_w = np.sqrt(np.asarray(s["wx"], dtype=float) ** 2
                        + np.asarray(s["wy"], dtype=float) ** 2
                        + np.asarray(s["wz"], dtype=float) ** 2)

        damping_per_substep = max(0.0, 1.0 - p.angular_damping_per_s * self._physics_dt)
        for _ in range(self._n_substeps):
            self.body.apply_forces_and_torques_at_pos(forces=force_local, torques=torque_local, is_global=False)
            self.world.step(render=False)
            ang_vel = self.body.get_angular_velocities()
            self.body.set_angular_velocities((ang_vel * damping_per_substep).astype(np.float32))

        pos, quat = self.body.get_world_poses()
        lin_vel = self.body.get_linear_velocities()
        pos, quat, lin_vel = np.asarray(pos), np.asarray(quat), np.asarray(lin_vel)
        ang_vel = np.asarray(self.body.get_angular_velocities())

        s["x"], s["y"], s["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
        s["vx"], s["vy"], s["vz"] = lin_vel[:, 0], lin_vel[:, 1], lin_vel[:, 2]
        for i in range(n):
            roll, pitch, yaw = _quat_to_euler(float(quat[i, 0]), float(quat[i, 1]), float(quat[i, 2]), float(quat[i, 3]))
            s["tilt_x"][i], s["tilt_y"][i], s["yaw"][i] = roll, pitch, yaw
        s["wx"], s["wy"], s["wz"] = ang_vel[:, 0], ang_vel[:, 1], ang_vel[:, 2]

        dps_mdot = thrust_mag / (p.dps_isp_s * G0)
        rcs_mdot = (np.abs(pitch_cmd) + np.abs(roll_cmd) + np.abs(yaw_cmd)) * (p.rcs_thrust_n * p.rcs_jets_per_couple) / (p.rcs_isp_s * G0)
        s["fuel_kg"] = np.maximum(0.0, s["fuel_kg"] - dps_mdot * p.dt_s)
        s["rcs_fuel_kg"] = np.maximum(0.0, s["rcs_fuel_kg"] - rcs_mdot * p.dt_s)
        s["throttle"] = throttle
        s["rcs_pitch"], s["rcs_roll"], s["rcs_yaw"] = pitch_cmd, roll_cmd, yaw_cmd
        self._t += p.dt_s
        mass_kg_all = self._write_mass_inertia_all()

        obs_batch = np.zeros((n, self.observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        infos: list[dict] = [dict() for _ in range(n)]

        for i in range(n):
            # whole tilted collider against the ground under its whole
            # footprint, not the belly centre against the ground under the
            # body origin -- see `isaac_lander_env.contact_clearance_m` for
            # the real landed-but-logged-as-a-timeout capture this fixes.
            clearance_m, ground_z = self._contact(i)
            touched_down = bool(clearance_m <= TOUCHDOWN_CONTACT_EPS_M)
            lost_control = bool(np.hypot(s["tilt_x"][i], s["tilt_y"][i]) > p.loss_of_control_tilt_rad)
            timed_out = bool(self._t[i] >= p.max_episode_s)
            ox, oy, _ = self.env_origins[i]
            left_tile = out_of_tile(self._tiles[i], s["x"][i] - ox, s["y"][i] - oy,
                                     self._specs.footpad_span_m / 2.0)
            terminated = touched_down or lost_control
            truncated = bool((timed_out or left_tile) and not terminated)

            landed_safely = False
            landing_margins: dict = {}
            leg_force_n = 0.0
            leg_force_max_n = 1.0
            if touched_down and not lost_control:
                # graded from the PRE-SUBSTEP state -- see the note above the
                # substep loop for why the post-substep state reads ~0 velocity
                # on a hard impact.
                v_z = float(pre_vz[i])
                v_xy = float(pre_vxy[i])
                tilt = float(pre_tilt[i])
                w = float(pre_w[i])
                leg_diff = self._footpad_height_diff_m(i, s["x"][i], s["y"][i])
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
                _, leg_force_max_n = leg_force_bounds_n(self._specs, mass_kg_all[i], p.safe_landing_v_z_m_s, p.gravity_m_s2)
                ke_j = 0.5 * mass_kg_all[i] * v_z ** 2  # impact speed, not the post-contact ~0
                leg_force_n = mass_kg_all[i] * p.gravity_m_s2 / p.leg_count + ke_j / (p.leg_count * p.leg_stroke_m)

            info = {
                "terminated": terminated, "truncated": truncated,
                "landed_safely": landed_safely, "lost_control": lost_control,
                "left_tile": left_tile,
                # the action that produced this step, so a reward term can charge
                # for it. `_action_saturation_penalty` in reward.py read
                # `info["action"]` and NOTHING ever wrote it, which made that
                # term identically zero and silently voided the experiment that
                # was meant to test it (out/train_v34_actsat.log measured a
                # disabled penalty).
                "action": np.asarray(a[i], dtype=float).copy(),
    # the impact state, so `_touchdown_severity` grades the speed the
                # vehicle actually hit at rather than the ~0 PhysX leaves after
                # the contact substep. Keys match `_STATE_FIELDS` names so the
                # severity function needs no special case.
                "impact_state": {"vz": float(pre_vz[i]), "vx": float(pre_vxy[i]), "vy": 0.0,
                                 "tilt_x": float(pre_tilt[i]), "tilt_y": 0.0,
                                 "wx": float(pre_w[i]), "wy": 0.0, "wz": 0.0},
                "landing_margins": landing_margins,
                # real clearance of the collider's lowest point above the
                # ground it can rest on; reaches 0 exactly at touchdown.
                "altitude_m": clearance_m,
                "fuel_kg": s["fuel_kg"][i], "rcs_fuel_kg": s["rcs_fuel_kg"][i],
                "mass_kg": mass_kg_all[i], "leg_force_n": leg_force_n, "leg_force_max_n": leg_force_max_n,
                "t_s": self._t[i],
            }
            scalar_state = {k: s[k][i] for k in _STATE_FIELDS}
            view = _EnvView(p, scalar_state)
            rewards[i] = self.reward_fn(view, info)
            done = terminated or truncated
            dones[i] = done

            if done:
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                info["terminal_observation"] = self._obs_one(i)
                self._reset_one(i)
                infos[i] = info
                obs_batch[i] = self._obs_one(i)
            else:
                infos[i] = info
                obs_batch[i] = self._obs_one(i)

        return obs_batch, rewards, dones, infos

    # ------------------------------------------------------------------
    def close(self) -> None:
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * (self.num_envs if indices is None else len(list(indices)))

    def set_attr(self, attr_name, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else len(list(indices))
        return [False] * n
