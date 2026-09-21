"""Lunar dust kinematics: no atmosphere -> no drag, so a lofted particle's
trajectory is a pure ballistic parabola under lunar gravity alone. This is
the real, well-documented reason lunar dust famously travels in long, fast,
low-angle arcs when disturbed (rocket plume, wheel, footstep) -- there is no
air resistance to slow horizontal motion or scatter the trajectory, unlike
terrestrial dust. This module is exact kinematics, not a fitted/looks-right
particle-system approximation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOON_GRAVITY_M_S2 = 1.62


@dataclass
class DustEvent:
    """One disturbance (plume impingement, wheel slip, footstep) that lofts
    a burst of regolith particles from a point.
    """

    origin_m: np.ndarray  # (3,), world position of the disturbance
    n_particles: int
    speed_range_m_s: tuple[float, float] = (0.5, 15.0)
    # low-angle-biased launch is the real lunar-plume signature (Apollo LM
    # descent-stage dust observations, Metzger et al. ejecta-angle studies):
    # most ejecta leaves at a shallow angle from the local horizontal.
    elevation_angle_range_deg: tuple[float, float] = (2.0, 35.0)
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0)
    seed: int = 0


def sample_launch_velocities(event: DustEvent) -> np.ndarray:
    """(n_particles, 3) initial velocity vectors, m/s."""
    rng = np.random.default_rng(event.seed)
    n = event.n_particles

    # Speed distribution: most ejecta is slow, a long tail reaches higher
    # speed (Rayleigh-like falloff is a reasonable, simple stand-in for
    # measured lunar ejecta speed distributions without over-claiming a
    # specific published model).
    lo, hi = event.speed_range_m_s
    u = rng.random(n)
    speed = lo + (hi - lo) * np.sqrt(u)  # biased toward the low end

    el = np.deg2rad(rng.uniform(*event.elevation_angle_range_deg, n))
    az = np.deg2rad(rng.uniform(*event.azimuth_range_deg, n))

    vx = speed * np.cos(el) * np.cos(az)
    vy = speed * np.cos(el) * np.sin(az)
    vz = speed * np.sin(el)
    return np.stack([vx, vy, vz], axis=-1)


def position_at(event: DustEvent, velocities_m_s: np.ndarray, t_s: np.ndarray | float, g: float = MOON_GRAVITY_M_S2) -> np.ndarray:
    """Particle positions at time(s) `t_s` (no drag): x = x0 + v*t, z = z0 + vz*t - 0.5*g*t^2.

    `t_s` broadcasts against `velocities_m_s`'s leading (n_particles,) axis:
    pass a scalar for one shared time, or an (n_particles,) array for
    per-particle time (e.g. each particle's own landing time).
    """
    t = np.asarray(t_s, dtype=float)
    if t.ndim == 0:
        t = np.full(velocities_m_s.shape[0], float(t))

    pos = event.origin_m[None, :] + velocities_m_s * t[:, None]
    pos[:, 2] -= 0.5 * g * t**2
    return pos


def flat_ground_landing_time(event: DustEvent, velocities_m_s: np.ndarray, ground_z_m: float = 0.0, g: float = MOON_GRAVITY_M_S2) -> np.ndarray:
    """Closed-form time each particle returns to a flat ground plane at `ground_z_m`.

    z0 + vz*t - 0.5*g*t^2 = ground_z  ->  0.5*g*t^2 - vz*t - (z0 - ground_z) = 0
    """
    z0 = event.origin_m[2]
    vz = velocities_m_s[:, 2]
    a, b, c = 0.5 * g, -vz, -(z0 - ground_z_m)
    disc = np.clip(b**2 - 4 * a * c, 0.0, None)
    t = (-b + np.sqrt(disc)) / (2 * a)
    return np.clip(t, 0.0, None)


def landing_positions_on_heightmap(
    event: DustEvent,
    velocities_m_s: np.ndarray,
    height: np.ndarray,
    res_m: float,
    g: float = MOON_GRAVITY_M_S2,
    n_time_steps: int = 200,
    max_flight_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Land each particle on a real heightmap (not just a flat plane): march
    each particle's parabola forward in time and take the first sample where
    it's at or below local terrain height. Coarser than an exact root-find
    but simple, vectorized, and accurate enough for the visual/dust-coverage
    use case (sub-time-step error << particle visual size).

    Returns (landing_positions (n,3), landing_time_s (n,)).
    """
    from lunarsim.core.terrain.rocks import sample_height_at

    n = velocities_m_s.shape[0]
    if max_flight_time_s is None:
        vz = velocities_m_s[:, 2]
        max_flight_time_s = float((vz.max() + np.sqrt(max(vz.max(), 0) ** 2 + 2 * g * 50)) / g) + 1.0

    # start strictly after t=0: particles are launched FROM ground level, so
    # checking z <= ground_z at t=0 itself would mark every particle "landed"
    # before it ever leaves the surface.
    t_grid = np.linspace(0, max_flight_time_s, n_time_steps)[1:]
    landed = np.zeros(n, dtype=bool)
    land_pos = np.zeros((n, 3))
    land_t = np.full(n, max_flight_time_s)

    for t in t_grid:
        pos = position_at(event, velocities_m_s, t, g)
        ground_z = sample_height_at(height, res_m, pos[:, 0], pos[:, 1])
        newly_landed = (~landed) & (pos[:, 2] <= ground_z)
        land_pos[newly_landed] = pos[newly_landed]
        land_t[newly_landed] = t
        landed |= newly_landed
        if landed.all():
            break

    # anything that never landed within the time budget: last computed position
    if not landed.all():
        pos = position_at(event, velocities_m_s, t_grid[-1], g)
        land_pos[~landed] = pos[~landed]

    return land_pos, land_t
