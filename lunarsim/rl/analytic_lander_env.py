"""A Gymnasium-compatible lunar lander environment driven entirely by
`lunarsim.core` (terrain + a small-angle rigid-body rocket model) -- no Isaac
Sim required. This is the fast, CPU-only backend for RL algorithm
development/iteration; `lunarsim.adapters.isaac` has the (Isaac-Sim-backed,
physically simulated) heavy-fidelity path once real contact dynamics,
plume/dust, and camera/LiDAR sensing on the real render/physics engine are
wanted instead of this analytic approximation.

Dynamics: a 2-axis gimballed engine (matches the "any model/RL on top"
brief) creates torque that tilts the vehicle body; thrust direction follows
the body's own tilt (small-angle), exactly as with a real thrust-vector-
controlled rocket -- gimbal deflection does not instantaneously redirect
thrust, it redirects the vehicle, which then redirects thrust. Ground
contact and crash/success classification use the SAME real terrain
heightmap `core.terrain` would export to Isaac, so an episode's landing
site is the same tile geometry either backend would see.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lunarsim.core.terrain.generate import Tile
from lunarsim.core.terrain.rocks import sample_height_at


@dataclass
class LanderParams:
    mass_kg: float = 1500.0
    max_thrust_n: float = 3000.0  # ~2g of thrust at lunar gravity, typical margin
    gimbal_max_rad: float = np.deg2rad(8.0)
    torque_gain: float = 4e-4  # angular accel per (N * rad of gimbal), tuned for a ~2s response time
    fuel_flow_kg_s_at_full_throttle: float = 3.0
    initial_fuel_kg: float = 400.0
    moment_of_inertia_gain: float = 1.0  # placeholder for future per-axis inertia tuning
    gravity_m_s2: float = 1.62
    dt_s: float = 0.05
    max_episode_s: float = 60.0
    safe_landing_v_z_m_s: float = 2.0
    safe_landing_v_xy_m_s: float = 1.0
    safe_landing_tilt_rad: float = np.deg2rad(10.0)
    safe_landing_w_rad_s: float = np.deg2rad(15.0)
    spawn_altitude_m: float = 80.0
    spawn_xy_radius_m: float = 30.0
    spawn_v_z_m_s: float = -3.0


RewardFn = Callable[["AnalyticLanderEnv", dict], float]


def default_reward_fn(env: "AnalyticLanderEnv", info: dict) -> float:
    """Shaped reward: closing on altitude/lateral distance is rewarded, high
    tilt/velocity/fuel use is penalized, terminal bonus/penalty on landing.
    Replace this with any custom reward for a specific RL task -- pass your
    own callable as `reward_fn` to the constructor.
    """
    s = env.state
    ground_z = env._ground_z(s["x"], s["y"])
    alt = s["z"] - ground_z

    r = 0.0
    r -= 0.01 * alt  # encourage descending
    r -= 0.01 * np.hypot(s["x"], s["y"])  # encourage staying near the pad
    r -= 0.05 * np.hypot(s["vx"], s["vy"])
    r -= 0.02 * abs(s["tilt_x"]) + 0.02 * abs(s["tilt_y"])
    r -= 0.001 * s["throttle"]  # tiny fuel-use penalty

    if info.get("terminated"):
        r += 100.0 if info.get("landed_safely") else -100.0
    return float(r)


class AnalyticLanderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tile: Tile,
        params: Optional[LanderParams] = None,
        reward_fn: RewardFn = default_reward_fn,
        use_lidar_obs: bool = False,
        lidar_n_rays: int = 8,
        seed: int | None = None,
    ):
        super().__init__()
        self.tile = tile
        self.params = params or LanderParams()
        self.reward_fn = reward_fn
        self.use_lidar_obs = use_lidar_obs
        self.lidar_n_rays = lidar_n_rays
        self._rng = np.random.default_rng(seed)

        obs_dim = 11 + (lidar_n_rays if use_lidar_obs else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # [throttle, gimbal_x, gimbal_y] all in [-1, 1], scaled internally
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.state: dict = {}
        self._t = 0.0

    def _ground_z(self, x: float, y: float) -> float:
        return float(sample_height_at(self.tile.height, self.tile.res_m, np.array([x]), np.array([y]))[0])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        r = self._rng.uniform(0, self.params.spawn_xy_radius_m)
        theta = self._rng.uniform(0, 2 * np.pi)
        x0, y0 = r * np.cos(theta), r * np.sin(theta)
        ground_z0 = self._ground_z(x0, y0)

        self.state = {
            "x": x0, "y": y0, "z": ground_z0 + self.params.spawn_altitude_m,
            "vx": 0.0, "vy": 0.0, "vz": self.params.spawn_v_z_m_s,
            "tilt_x": self._rng.uniform(-0.05, 0.05),
            "tilt_y": self._rng.uniform(-0.05, 0.05),
            "wx": 0.0, "wy": 0.0,
            "fuel_kg": self.params.initial_fuel_kg,
            "throttle": 0.0,
        }
        self._t = 0.0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        p = self.params
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        gimbal_x = float(np.clip(action[1], -1.0, 1.0)) * p.gimbal_max_rad
        gimbal_y = float(np.clip(action[2], -1.0, 1.0)) * p.gimbal_max_rad

        s = self.state
        if s["fuel_kg"] <= 0.0:
            throttle = 0.0

        thrust_mag = throttle * p.max_thrust_n

        # gimbal torques the body; body tilt then redirects thrust (real TVC behavior)
        s["wx"] += gimbal_y * thrust_mag * p.torque_gain * p.dt_s
        s["wy"] += -gimbal_x * thrust_mag * p.torque_gain * p.dt_s
        s["tilt_x"] += s["wx"] * p.dt_s
        s["tilt_y"] += s["wy"] * p.dt_s

        thrust_x = thrust_mag * np.sin(s["tilt_y"])
        thrust_y = -thrust_mag * np.sin(s["tilt_x"])
        thrust_z = thrust_mag * np.cos(s["tilt_x"]) * np.cos(s["tilt_y"])

        ax = thrust_x / p.mass_kg
        ay = thrust_y / p.mass_kg
        az = thrust_z / p.mass_kg - p.gravity_m_s2

        s["vx"] += ax * p.dt_s
        s["vy"] += ay * p.dt_s
        s["vz"] += az * p.dt_s
        s["x"] += s["vx"] * p.dt_s
        s["y"] += s["vy"] * p.dt_s
        s["z"] += s["vz"] * p.dt_s

        s["fuel_kg"] = max(0.0, s["fuel_kg"] - throttle * p.fuel_flow_kg_s_at_full_throttle * p.dt_s)
        s["throttle"] = throttle
        self._t += p.dt_s

        ground_z = self._ground_z(s["x"], s["y"])
        touched_down = bool(s["z"] <= ground_z)
        timed_out = self._t >= p.max_episode_s

        terminated = touched_down
        truncated = bool(timed_out and not touched_down)

        landed_safely = False
        if touched_down:
            v_xy = np.hypot(s["vx"], s["vy"])
            tilt = np.hypot(s["tilt_x"], s["tilt_y"])
            w = np.hypot(s["wx"], s["wy"])
            landed_safely = bool(
                abs(s["vz"]) <= p.safe_landing_v_z_m_s
                and v_xy <= p.safe_landing_v_xy_m_s
                and tilt <= p.safe_landing_tilt_rad
                and w <= p.safe_landing_w_rad_s
            )
            s["z"] = ground_z  # clamp to surface

        info = {
            "terminated": terminated,
            "landed_safely": landed_safely,
            "altitude_m": s["z"] - ground_z,
            "fuel_kg": s["fuel_kg"],
            "t_s": self._t,
        }
        reward = self.reward_fn(self, info)

        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        s = self.state
        ground_z = self._ground_z(s["x"], s["y"])
        obs = [
            s["x"], s["y"], s["z"] - ground_z,
            s["vx"], s["vy"], s["vz"],
            s["tilt_x"], s["tilt_y"], s["wx"], s["wy"],
            s["fuel_kg"] / self.params.initial_fuel_kg,
        ]
        if self.use_lidar_obs:
            obs.extend(self._lidar_ranges())
        return np.array(obs, dtype=np.float32)

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
