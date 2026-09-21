"""Dust particle authoring: bakes `core.dust.plume` ballistic trajectories
(exact vacuum kinematics, no drag) as USD time-sampled `PointInstancer`
positions, so lofted regolith visibly moves along its real parabolic arc
without needing a live physics step per particle.

This mirrors the plan's `dust.mode` quality-profile knob:
- `off`: don't call any of this.
- `visual`: `spawn_dust_burst` with a modest particle count, position-only
  animation (this module).
- `visual_physics` / `particles`: same authoring, but with more particles and
  (future work) an actual PhysX particle system for local ROI dust that
  should interact with the lander/wheels, not just look right from a
  distance -- not implemented here, flagged as a TODO since it needs a live
  Isaac particle-system test to get right (see adapters/isaac/README.md).
"""
from __future__ import annotations

import numpy as np

from lunarsim.core.dust.plume import DustEvent, landing_positions_on_heightmap, position_at, sample_launch_velocities


def spawn_dust_burst(
    stage,
    prim_path: str,
    event: DustEvent,
    prototype_path: str,
    height: np.ndarray,
    res_m: float,
    fps: float = 30.0,
    start_time_code: float = 0.0,
    particle_radius_m: float = 0.02,
):
    """Author a `UsdGeom.PointInstancer` animating `event`'s particles along
    their real ballistic trajectories, from launch to landing on `height`.

    Positions are written as USD time samples (one per particle per frame,
    vectorized per frame) so any USD-compliant viewer/renderer plays the
    motion back identically, with no per-frame Python driving needed at
    render time.
    """
    from pxr import Gf, UsdGeom

    velocities = sample_launch_velocities(event)
    _, land_t = landing_positions_on_heightmap(event, velocities, height, res_m)

    max_t = float(land_t.max())
    n_frames = max(2, int(np.ceil(max_t * fps)) + 1)

    instancer = UsdGeom.PointInstancer.Define(stage, prim_path)
    instancer.CreatePrototypesRel().AddTarget(prototype_path)
    instancer.CreateProtoIndicesAttr([0] * event.n_particles)

    for frame in range(n_frames):
        t = frame / fps
        # clamping t to each particle's own landing time freezes it in place
        # (at its landing position) for all frames after it lands
        t_clamped = np.minimum(t, land_t)
        pos = position_at(event, velocities, t_clamped)

        time_code = start_time_code + frame
        instancer.GetPositionsAttr().Set(
            [Gf.Vec3f(*p) for p in pos.tolist()], time=time_code
        )

        # simple fade proxy: shrink per-instance scale as the particle settles
        # (visual dust settling cue, not a physical opacity model). Scale is
        # PointInstancer's standard per-instance size control -- a plain
        # custom "widths" attribute (the UsdGeomPoints convention) isn't
        # meaningful here and trips Fabric/Hydra warnings about an
        # unrecognized array attribute on a PointInstancer.
        settling = np.clip((t - land_t) / 0.5, 0.0, 1.0)  # fade over 0.5s after landing
        scale = particle_radius_m * (1.0 - 0.6 * settling)
        instancer.GetScalesAttr().Set(
            [Gf.Vec3f(s, s, s) for s in scale.tolist()], time=time_code
        )

    return instancer


def dust_event_from_disturbance(origin_m: np.ndarray, intensity: float, seed: int = 0) -> DustEvent:
    """Convenience constructor: `intensity` in [0, 1] scales particle count and
    max ejecta speed (e.g. throttle fraction for a plume, or wheel slip ratio).
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    return DustEvent(
        origin_m=origin_m,
        n_particles=max(1, int(50 + 450 * intensity)),
        speed_range_m_s=(0.3, 3.0 + 12.0 * intensity),
        elevation_angle_range_deg=(2.0, 35.0),
        seed=seed,
    )
